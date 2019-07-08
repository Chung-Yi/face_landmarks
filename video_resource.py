import numpy as np
import cv2
import os
import time
import multiprocessing
import threading


def init_capture_device(source, fps, height, width):
    max_sleep_sec = 33.0
    cur_sleep_sec = 0.5
    ttl = 5
    while True:
        if ttl < 0:
            raise RuntimeError
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            break

        print("source not opened, sleeping {}s and try again!".format(
            cur_sleep_sec))

        cap.release()
        time.sleep(cur_sleep_sec)
        if cur_sleep_sec < max_sleep_sec:
            cur_sleep_sec *= 2
            cur_sleep_sec = min(cur_sleep_sec, max_sleep_sec)
            ttl -= 1
            continue

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS)

    return cap


class FrameThreadHandle:
    def __init__(self, source, fps, height, width):
        self.thread = None
        self.source = source
        self.fps = fps
        self.width = width
        self.height = height

    def start(self, shared_frame):
        self.thread = threading.Thread(
            target=self.detect_frame, args=(shared_frame, ), name='v_source')
        self.thread.setDaemon(True)
        self.thread.start()
        print("capture thread start")

    def detect_frame(self, shared_frame):
        try:
            capture = init_capture_device(self.source, self.fps, self.height,
                                          self.width)
        except RuntimeError as e:
            print("init VideoCapture ERROR:{}, exit program".format(e))
            os._exit(1)

        while True:
            ret, frame = capture.read()
            if not ret:
                capture.release()
                print(
                    "FrameThreadHandler ERROR: capture.read() ret is not True, exit program"
                )
                os._exit(1)
            np.copyto(shared_frame, frame)


class ThreadingVideoResource:
    def __init__(self, source, width, height, fps=None):
        self.source = source
        self.shared_frame = np.zeros((height, width, 3), dtype='uint8')
        self.handler = FrameThreadHandle(source, fps, height, width)
        self.handler.start(self.shared_frame)

    def get_frame_date(self):
        return cv2.cvtColor(self.shared_frame, cv2.COLOR_BGR2RGB), time.time()

    def release(self):
        cv2.destroyAllWindows()
        cv2.waitKey(4)