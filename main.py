import cv2 as cv
import cv2
from cv2.typing import MatLike
import numpy as np
from threading import Thread
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.websockets import WebSocket
from starlette.routing import Route, WebSocketRoute
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_until_first_complete, run_in_threadpool
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK
import orjson as json
import anyio
from anyio import TASK_STATUS_IGNORED
from anyio import to_thread
from anyio import create_memory_object_stream
from anyio.streams.memory import MemoryObjectReceiveStream
from pydantic import BaseModel
import uvicorn
import logging
import contextlib
import click
import struct
from typing import Optional, TypeVar, Generic, cast, Any
from enum import Enum, auto
from time import time

ws_conn = set[WebSocket]()


async def poll_stream(recv: MemoryObjectReceiveStream[bytes]):
    from loguru import logger
    logger.info("poll stream")
    async for message in recv:
        for ws in ws_conn.copy():
            try:
                await ws.send_bytes(message)
            except WebSocketDisconnect:
                ws_conn.remove(ws)
            except ConnectionClosedOK:
                ws_conn.remove(ws)
            except Exception as e:
                logger.error(e)


class BitDepth(Enum):
    UINT8 = cv.CV_8U
    INT8 = cv.CV_8S
    UINT16 = cv.CV_16U
    INT16 = cv.CV_16S
    INT32 = cv.CV_32S
    FLOAT32 = cv.CV_32F
    FLOAT64 = cv.CV_64F
    FLOAT16 = cv.CV_16F


def np_dtype_to_cv(dtype: Any) -> BitDepth:
    dtype_to_bit_depth = {
        np.uint8: BitDepth.UINT8,
        np.int8: BitDepth.INT8,
        np.uint16: BitDepth.UINT16,
        np.int16: BitDepth.INT16,
        np.int32: BitDepth.INT32,
        np.float32: BitDepth.FLOAT32,
        np.float64: BitDepth.FLOAT64,
        np.float16: BitDepth.FLOAT16,
    }
    return dtype_to_bit_depth[dtype.type]


class ImageDimension(BaseModel):
    width: int
    height: int
    stride: int
    channels: int
    bit_depth: BitDepth

    @staticmethod
    def struct_format() -> str:
        return ">HHHBB"

    def marshal(self) -> bytes:
        return struct.pack(self.struct_format(), self.width, self.height,
                           self.stride, self.channels, self.bit_depth.value)

    @staticmethod
    def bytes_size() -> int:
        return struct.calcsize(ImageDimension.struct_format())

    def __bytes__(self) -> bytes:
        return self.marshal()

    @staticmethod
    def unmarshal(data: bytes) -> "ImageDimension":
        width, height, stride, channel, bit_depth = struct.unpack(
            ImageDimension.struct_format(), data)
        return ImageDimension(width=width,
                              height=height,
                              stride=stride,
                              channels=channel,
                              bit_depth=BitDepth(bit_depth))

    @staticmethod
    def from_cv_mat(mat: MatLike) -> "ImageDimension":
        bit_depth = np_dtype_to_cv(mat.dtype)
        return ImageDimension(width=mat.shape[1],
                              height=mat.shape[0],
                              stride=mat.strides[0],
                              channels=mat.shape[2],
                              bit_depth=bit_depth)


# can't use cv.imshow in another thread
CHANNEL_NAME = "video"
T = TypeVar("T")

logger = logging.getLogger('uvicorn')


# https://anyio.readthedocs.io/en/stable/streams.html
async def ws_handler(ws: WebSocket):
    try:
        await ws.accept()
        ws_conn.add(ws)
        async for message in ws.iter_bytes():
            pass
    except WebSocketDisconnect:
        logger.warning("websocket {} disconnected".format(ws.client))
    finally:
        ws_conn.remove(ws)


@contextlib.asynccontextmanager
async def lifespan(_app: Starlette):
    logger.info("lifespan starts")
    async with anyio.create_task_group() as tg:
        # https://anyio.readthedocs.io/en/latest/threads.html#spawning-tasks-from-worker-threads
        def with_status(task_status=TASK_STATUS_IGNORED):
            return run_video_cap()

        tg.start_soon(run_video_cap)
        yield
        tg.cancel_scope.cancel()
    logger.info("lifespan end")


EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480


class SimpleMovingAverage:
    _sum: float = 0.0
    _count: int = 0
    _max_count: int = 10
    _value: Optional[float] = None

    def __init__(self, max_count: int = 10):
        self._max_count = max_count
        self._sum = 0.0
        self._count = 0
        self._value = None

    @property
    def value(self) -> Optional[float]:
        return self._value

    def next(self, val: float) -> float:
        if self._count < self._max_count:
            self._sum += val
            self._count += 1
            self._value = self._sum / self._count
        else:
            self._sum -= self._sum / self._max_count
            self._sum += val
            self._value = self._sum / self._max_count
        return self._value

    def reset(self):
        self._sum = 0.0
        self._count = 0
        self._value = None


# https://anyio.readthedocs.io/en/latest/threads.html#spawning-tasks-from-worker-threads
async def run_video_cap():
    # use GStreamer to get a video stream from the test video
    # https://github.com/opencv/opencv/blob/625eebad54a34a7bdad6812f3e9ec050a1b3adc5/modules/videoio/src/cap_gstreamer.cpp#L1342-L1344
    # https://stackoverflow.com/questions/51213730/how-to-get-gstreamer-live-stream-using-opencv-and-python
    from loguru import logger
    send_stream, receive_stream = create_memory_object_stream[bytes](0)

    async def run_cap():
        pipeline = "videotestsrc is-live=true ! timeoverlay ! videoconvert ! appsink name=opencvsink"
        cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
        # cap = cv.VideoCapture(1, cv.CAP_AVFOUNDATION)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, EXPECTED_WIDTH)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, EXPECTED_HEIGHT)
        cap.set(cv.CAP_PROP_FPS, 30)
        process_time_sma = SimpleMovingAverage(30)
        frame_count = 0
        if not cap.isOpened():
            logger.error("capture is not opened. Exiting ...")
            return
        try:
            while True:
                ret, frame = await to_thread.run_sync(cap.read)
                if not ret:
                    logger.error(
                        "Can't receive frame (stream end?). Exiting ...")
                    break
                start = time()
                if frame.shape[1] > EXPECTED_WIDTH or frame.shape[
                        0] > EXPECTED_HEIGHT:
                    frame = cv.resize(frame, (EXPECTED_WIDTH, EXPECTED_HEIGHT))
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                info = ImageDimension.from_cv_mat(rgb)
                with anyio.move_on_after(0.1):
                    await send_stream.send(info.marshal() + rgb.tobytes())
                    diff = time() - start
                    process_time_sma.next(diff)
                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info(f"process time: {process_time_sma.value}")

        except KeyboardInterrupt:
            cap.release()
        except Exception as e:
            cap.release()
            logger.error(e)

    # make sure GStreamer is Yes in Video I/O
    # logger.info(cv2.getBuildInformation())
    async with anyio.create_task_group() as tg:
        tg.start_soon(poll_stream, receive_stream)
        await run_cap()


@click.command()
@click.option("--port", default=8000, help="Port number")
@click.option("--host", default="0.0.0.0", help="Host")
def main(port: int, host: str):
    from importlib.metadata import version
    anyio_version_str = version("anyio").split(".")
    anyio_version = tuple([int(x) for x in anyio_version_str])
    assert anyio_version[0] >= 4 and anyio_version[
        1] >= 3, "anyio version must be >= 4.3"
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ]
    app = Starlette(debug=True,
                    routes=[
                        Route('/',
                              lambda _request: JSONResponse({'test': 'test'})),
                        WebSocketRoute('/', ws_handler, name='ws'),
                    ],
                    middleware=middleware,
                    lifespan=lifespan)  # type: ignore
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
