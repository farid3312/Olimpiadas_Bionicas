"""Hilo de captura de cámara nativo a la resolución del sensor."""
import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class CameraThread:
    def __init__(self, src: int = 0, width: int = 128, height: int = 128, fps: int = 30):
        self.w, self.h = width, height
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_DSHOW
        self.cap = cv2.VideoCapture(src, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara en índice {src}.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame: Optional[np.ndarray] = None
        self.ret = False
        self.frame_id = -1
        self.real_fps = 0.0
        self.running = True
        self.lock = threading.Lock()

        for _ in range(40):
            ok, f = self.cap.read()
            if ok and f is not None:
                if f.shape[1] != width or f.shape[0] != height:
                    f = cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA)
                with self.lock:
                    self.frame = f
                    self.ret = True
                    self.frame_id = 0
                break
            time.sleep(0.05)
        if self.frame is None:
            raise RuntimeError("La cámara no entregó fotogramas válidos.")

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        n, timer = 0, time.perf_counter()
        while self.running:
            ok, f = self.cap.read()
            if not ok or f is None:
                time.sleep(0.002)
                continue
            if f.shape[1] != self.w or f.shape[0] != self.h:
                f = cv2.resize(f, (self.w, self.h), interpolation=cv2.INTER_AREA)
            with self.lock:
                self.frame = f
                self.ret = ok
                self.frame_id += 1
            n += 1
            el = time.perf_counter() - timer
            if el >= 1.0:
                self.real_fps = n / el
                n, timer = 0, time.perf_counter()

    def read(self) -> Tuple[bool, Optional[np.ndarray], float, int]:
        with self.lock:
            if self.frame is None:
                return False, None, 0.0, -1
            return self.ret, self.frame.copy(), self.real_fps, self.frame_id

    def stop(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()