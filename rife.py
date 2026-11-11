"""Interpolador RIFE (con carga protegida del módulo)."""
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.append("tu ruta aqui")
try:
    # Tu archivo se llama RIFE_HDv3.py, no RIFE.py
    from model.RIFE_HDv3 import Model as RIFEModel
    RIFE_AVAILABLE = True
    print("[+] Módulo RIFE importado correctamente.")
except ImportError as e:
    RIFE_AVAILABLE = False
    print(f"[!] Error importando RIFE: {e}")


class RIFEInterpolator:
    def __init__(self, device: torch.device, model_dir: str):
        self.device = device
        self.model: nn.Module | None = None
        if device.type == "cuda" and RIFE_AVAILABLE:
            try:
                m = RIFEModel()
                m.load_model(model_dir, -1)  # Carga flownet.pkl desde la ruta indicada
                m.eval()
                m.device()
                self.model = m
                print(f"[+] RIFE cargado en GPU desde: {model_dir}")
            except Exception as e:
                print(f"[!] RIFE no disponible ({e}). Se usará blend lineal.")

    def interpolate(self, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
        if self.model is None:
            return cv2.addWeighted(img0, 0.5, img1, 0.5, 0)

        I0 = (torch.from_numpy(img0).to(self.device, dtype=torch.float16)
              .permute(2, 0, 1).unsqueeze(0) / 255.0)
        I1 = (torch.from_numpy(img1).to(self.device, dtype=torch.float16)
              .permute(2, 0, 1).unsqueeze(0) / 255.0)

        with torch.inference_mode():
            mid = self.model.inference(I0, I1)

        return (mid[0].clamp(0, 1).cpu().float().numpy()
                .transpose(1, 2, 0) * 255.0).astype(np.uint8)
