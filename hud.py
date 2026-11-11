"""Utilidades de dibujo para el HUD Cyberpunk."""
from typing import Tuple

import cv2
import numpy as np


def draw_cyber_corners(img: np.ndarray, x: int, y: int, w: int, h: int,
                       color: Tuple[int, int, int], arm: int = 14, thickness: int = 2):
    """Dibuja esquinas tácticas estilo interfaz cyberpunk."""
    # Top-Left
    cv2.line(img, (x, y), (x + arm, y), color, thickness)
    cv2.line(img, (x, y), (x, y + arm), color, thickness)
    # Top-Right
    cv2.line(img, (x + w, y), (x + w - arm, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + arm), color, thickness)
    # Bottom-Left
    cv2.line(img, (x, y + h), (x + arm, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - arm), color, thickness)
    # Bottom-Right
    cv2.line(img, (x + w, y + h), (x + w - arm, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - arm), color, thickness)


def draw_cyber_reticle(img: np.ndarray, cx: int, cy: int,
                       size: int = 12, color: Tuple[int, int, int] = (0, 255, 255)):
    """Retícula de mira central sutil."""
    cv2.line(img, (cx - size, cy), (cx - 4, cy), color, 1)
    cv2.line(img, (cx + 4, cy), (cx + size, cy), color, 1)
    cv2.line(img, (cx, cy - size), (cx, cy - 4), color, 1)
    cv2.line(img, (cx, cy + 4), (cx, cy + size), color, 1)
    cv2.circle(img, (cx, cy), 2, color, -1)