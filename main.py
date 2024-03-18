import cv2 as cv
import cv2
from cv2.typing import MatLike
import numpy as np
from threading import Thread
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.websockets import WebSocket
from starlette.routing import Route, WebSocketRoute
from starlette.concurrency import run_until_first_complete, run_in_threadpool
from broadcaster import Broadcast
import orjson as json
import anyio
from anyio import create_memory_object_stream
from anyio.streams.memory import MemoryObjectReceiveStream
from pydantic import BaseModel
import starlette
import uvicorn
import logging
import contextlib
import click
import struct
from typing import Optional, TypeVar, Generic, cast, Any
from enum import Enum, auto


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
    channel: int
    bit_depth: BitDepth

    @staticmethod
    def struct_format() -> str:
        return ">HHHBB"

    def marshal(self) -> bytes:
        return struct.pack(self.struct_format(), self.width, self.height,
                           self.stride, self.channel, self.bit_depth.value)

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
                              channel=channel,
                              bit_depth=BitDepth(bit_depth))

    @staticmethod
    def from_cv_mat(mat: MatLike) -> "ImageDimension":
        bit_depth = np_dtype_to_cv(mat.dtype)
        return ImageDimension(width=mat.shape[1],
                              height=mat.shape[0],
                              stride=mat.strides[0],
                              channel=mat.shape[2],
                              bit_depth=bit_depth)


SHOW_WINDOW = False
CHANNEL_NAME = "video"
T = TypeVar("T")


class Event(Generic[T]):
    """
    Just for type annotation; see also broadcaster.Event
    """
    channel: str
    message: T


broadcast: Optional[Broadcast] = None
logger = logging.getLogger('uvicorn')


async def ws_receiver(websocket: WebSocket):
    async for _message in websocket.iter_text():
        pass


async def ws_sender(websocket: WebSocket):
    assert broadcast is not None, "broadcast is not initialized"
    async with broadcast.subscribe(channel=CHANNEL_NAME) as subscriber:
        async for event in subscriber:
            event = cast(Event[bytes], event)
            await websocket.send_bytes(event.message)


# https://anyio.readthedocs.io/en/stable/streams.html
async def ws_handler(ws: WebSocket):
    await ws.accept()
    await run_until_first_complete((ws_receiver, {
        "websocket": ws
    }), (ws_sender, {
        "websocket": ws
    }))


@contextlib.asynccontextmanager
async def lifespan(_app: Starlette):
    global broadcast
    logger.info("lifespan starts")
    async with anyio.create_task_group() as tg:
        broadcast = Broadcast("memory://")
        assert broadcast is not None
        assert broadcast is not None
        await broadcast.connect()
        yield
        await broadcast.disconnect()
        await tg.cancel_scope.cancel()
    logger.info("lifespan end")


def run_video_cap():
    # use GStreamer to get a video stream from the test video
    # https://github.com/opencv/opencv/blob/625eebad54a34a7bdad6812f3e9ec050a1b3adc5/modules/videoio/src/cap_gstreamer.cpp#L1342-L1344
    # https://stackoverflow.com/questions/51213730/how-to-get-gstreamer-live-stream-using-opencv-and-python
    from loguru import logger
    # make sure GStreamer is Yes in Video I/O
    logger.info(cv2.getBuildInformation())
    pipeline = "videotestsrc is-live=true ! timeoverlay ! videoconvert ! appsink name=opencvsink"
    cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
    if not cap.isOpened():
        logger.error("capture is not opened. Exiting ...")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Can't receive frame (stream end?). Exiting ...")
                break
            # cv.imshow("frame", frame)
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            global broadcast
            broadcast = cast(Broadcast, broadcast)
            # TODO: a struct/header for stride, width, height, frame info, etc.
            info = ImageDimension.from_cv_mat(rgb)
            anyio.run(broadcast.publish, CHANNEL_NAME, info.marshal() + rgb.tobytes())
            if SHOW_WINDOW:
                cv.namedWindow("frame", cv.WINDOW_NORMAL)
                cv.imshow("frame", frame)
                k = cv.waitKey(1)
                if k == ord('q'):
                    break

    except KeyboardInterrupt:
        cap.release()
        cv.destroyAllWindows()
    except Exception as e:
        cap.release()
        cv.destroyAllWindows()
        print(e)


@click.command()
@click.option("--port", default=8000, help="Port number")
@click.option("--host", default="0.0.0.0", help="Host")
def main(port: int, host: str):
    video_thread = Thread(target=run_video_cap)
    app = Starlette(
        debug=True,
        routes=[
            Route('/', lambda _request: JSONResponse({'test': 'test'})),
            # I'm not sure what the name is used for the websocket endpoint
            WebSocketRoute('/', ws_handler, name='ws'),
        ],
        lifespan=lifespan)  # type: ignore
    video_thread.start()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
