"""Configuración del sistema, hiperparámetros y constantes globales."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List


CLASS_NAMES_7: List[str] = [
    "Clapping",
    "Meet and Split",
    "Sitting",
    "Standing Still",
    "Walking",
    "Walking While Reading Book",
    "Walking While Using Phone",
]

# Mapeo de las 7 clases SNN a los 4 gestos del juego
SNN_TO_GAME_GESTURE = {
    "Clapping": "Clapping",
    "Meet and Split": "Meet and Split",
    "Sitting": "Standing Still",
    "Standing Still": "Standing Still",
    "Walking": "Walking",
    "Walking While Reading Book": "Walking",
    "Walking While Using Phone": "Walking",
}


def to_game_gesture(snn_label: str) -> str:
    return SNN_TO_GAME_GESTURE.get(snn_label, snn_label)


# Paleta de colores Cyberpunk BGR para OpenCV
CYBER_COLORS = {
    "Clapping": (0, 230, 255),                    # Amarillo Neon
    "Meet and Split": (255, 120, 0),              # Azul Eléctrico
    "Sitting": (0, 165, 255),                     # Naranja Cyber
    "Standing Still": (180, 180, 180),            # Platino
    "Walking": (255, 255, 0),                     # Cyan Láser
    "Walking While Reading Book": (200, 0, 255), # Magenta / Neón Púrpura
    "Walking While Using Phone": (255, 0, 180),   # Rosa Caliente
    "Default": (0, 255, 255),                     # Amarillo
}


@dataclass
class SystemParams:
    model_path: Path = Path("/home/immersivetech/proyectos/v2e/models/SpikeHarDseantV2E.pth")
    sensor_size: Tuple[int, int, int] = (128, 128, 2)  # ENTRADA DE DATOS: 128x128 (INTACTO)
    num_classes: int = 7
    sample_duration_ms: float = 1500.0
    target_v2e_fps: float = 60.0
    infer_every_ms: float = 250.0
    camera_index: int = 0
    camera_fps: int = 30


@dataclass
class HyperParams:
    # Voxelización
    n_time_bins: int = 16
    voxel_max_count: int = 16
    max_events: int = 80000

    # DTEASN (Atención espacio-temporal)
    use_attention: bool = True
    K_neighbors: int = 60
    tau_time_ms: float = 16.6
    tau_space_px: float = 2.0
    lambda_pol: float = 0.4
    alpha: float = 0.26
    beta: float = 0.46
    gamma: float = 0.28
    attn_clip_min: float = 0.1
    attn_clip_max: float = 5.0
    sw_clip_min: float = 0.5
    sw_clip_max: float = 2.0

    # SNN
    embed_dim: int = 48
    expansion_mlp: int = 2
    max_groups_gn: int = 4
    fc_hidden: int = 128
    pool_size_head: Tuple[int, int] = (4, 4)
    surrogate_type: str = "atan"
    beta_lif: float = 0.70
    threshold_lif: float = 0.90
    dropout_pe: float = 0.30
    dropout_head: float = 0.30


@dataclass
class Config:
    system: SystemParams = field(default_factory=SystemParams)
    hp: HyperParams = field(default_factory=HyperParams)

    def __getattr__(self, name):
        for section in (self.system, self.hp):
            if hasattr(section, name):
                return getattr(section, name)
        raise AttributeError(f"'Config' no posee el atributo '{name}'")