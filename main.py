"""Punto de entrada de la aplicación Spike-HAR++ Cyberpunk HUD."""
import argparse
import sys
from pathlib import Path

import cv2

cv2.setNumThreads(1)

from config import Config
from runtime import LiveRuntime


def main():
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Spike-HAR++ Live · Cyberpunk HUD (Viewports 320x240, Entrada 128x128)"
    )
    parser.add_argument("--ckpt", type=str,
                        default=str(cfg.system.model_path),
                        help="Ruta al checkpoint .pth")
    parser.add_argument("--camera", type=int, default=cfg.system.camera_index,
                        help="Índice de la cámara")
    parser.add_argument("--infer_every_ms", type=float,
                        default=cfg.system.infer_every_ms,
                        help="Cadencia de inferencia (ms)")
    parser.add_argument("--no-scanlines", action="store_true",
                        help="Desactiva el efecto CRT scanlines en el HUD")
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"[ERROR] No existe el checkpoint: {ckpt.resolve()}")
        sys.exit(1)

    LiveRuntime(cfg, args.ckpt, args.camera, args.infer_every_ms,
                enable_scanlines=not args.no_scanlines).run()


if __name__ == "__main__":
    main()