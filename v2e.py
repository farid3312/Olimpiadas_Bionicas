"""Buffer de eventos y emulador neuromórfico V2E (intactos)."""
import threading
from collections import deque
from typing import Optional

import cv2
import numpy as np
from v2ecore.emulator import EventEmulator

from config import Config


class EventBuffer:
    def __init__(self, duration_ms: float):
        self.duration_us = duration_ms * 1000.0
        self._buf: deque = deque()
        self._lock = threading.Lock()
        self._count = 0

    def append(self, block: np.ndarray):
        with self._lock:
            self._buf.append(block)
            self._count += len(block)

    def trim(self, now_us: float):
        cutoff = now_us - self.duration_us
        with self._lock:
            while self._buf and self._buf[0][-1, 0] < cutoff:
                self._count -= len(self._buf.popleft())

    def snapshot(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._buf:
                return None
            return np.concatenate(list(self._buf), axis=0)

    @property
    def total_events(self) -> int:
        with self._lock:
            return self._count


class V2EEngine:
    def __init__(self, cfg: Config, device: str):
        self.cfg = cfg
        self.emulator = EventEmulator(
            pos_thres=0.15, neg_thres=0.15, sigma_thres=0.03,
            cutoff_hz=15.0, leak_rate_hz=0.01, shot_noise_rate_hz=0.5,
            output_width=cfg.sensor_size[1],
            output_height=cfg.sensor_size[0],
            device=device,
        )
        self.sim_time_s = 0.0
        self.canvas = np.full((cfg.sensor_size[0], cfg.sensor_size[1], 3),
                              127, dtype=np.uint8)

    def step(self, bgr_frame: np.ndarray, dt_sec: float) -> Optional[np.ndarray]:
        self.sim_time_s += dt_sec
        gray = (bgr_frame if bgr_frame.ndim == 2
                else cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY))

        evs = self.emulator.generate_events(gray, self.sim_time_s)
        if evs is None or len(evs) == 0:
            self._fade()
            return None

        t_us = (evs[:, 0] * 1e6).astype(np.int64)
        x = evs[:, 1].astype(np.int64)
        y = evs[:, 2].astype(np.int64)
        p = np.where(evs[:, 3] > 0, 1, 0).astype(np.int64)

        H, W = self.canvas.shape[:2]
        valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if valid.any():
            xs, ys, ps = x[valid], y[valid], p[valid]
            # Eventos positivos (ON): Cyan Eléctrico (255, 230, 0 en BGR)
            # Eventos negativos (OFF): Magenta Neón (180, 0, 255 en BGR)
            self.canvas[ys, xs] = np.where(ps[:, None] > 0,
                                           (255, 240, 20),
                                           (40, 30, 250)).astype(np.uint8)

        self._fade()
        return np.column_stack([t_us, x, y, p])

    def _fade(self):
        self.canvas = np.add(
            self.canvas * 0.85,
            np.full_like(self.canvas, 20) * 0.15
        ).astype(np.uint8)

    def cleanup(self):
        try:
            self.emulator.cleanup()
        except Exception:
            pass