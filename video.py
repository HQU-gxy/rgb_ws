import cv2 as cv
import cv2
from cv2.typing import MatLike

def run_video_cap():
    # use GStreamer to get a video stream from the test video
    # https://github.com/opencv/opencv/blob/625eebad54a34a7bdad6812f3e9ec050a1b3adc5/modules/videoio/src/cap_gstreamer.cpp#L1342-L1344
    # https://stackoverflow.com/questions/51213730/how-to-get-gstreamer-live-stream-using-opencv-and-python
    from loguru import logger
    # make sure GStreamer is Yes in Video I/O
    logger.info(cv2.getBuildInformation())
    # pipeline = "videotestsrc is-live=true ! timeoverlay ! videoconvert ! appsink name=opencvsink"
    # cap = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
    cap = cv.VideoCapture(1, cv.CAP_AVFOUNDATION)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        logger.error("capture is not opened. Exiting ...")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Can't receive frame (stream end?). Exiting ...")
                break
            cv.imshow("frame", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        cap.release()
        cv.destroyAllWindows()
    except Exception as e:
        cap.release()
        cv.destroyAllWindows()
        logger.error(e)

        
if __name__ == "__main__":
    run_video_cap()