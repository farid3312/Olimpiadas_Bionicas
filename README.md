# Ciencia_al_parque

# Spike-HAR++ Cyberpunk HUD

Pipeline en vivo de **Reconocimiento de Actividades Humanas (HAR)** basado en
**Redes Neuronales Pulsantes (SNN)**, con emulación neuromórfica V2E,
interpolación de frames RIFE y una interfaz HUD con estética **Cyberpunk Retro**.

Incluye un **juego del ahorcado** que se controla mediante gestos corporales
detectados por la red neuronal en tiempo real.

```
┌──────────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌──────┐   ┌─────────┐
│  Cámara      │──▶│  RIFE  │──▶│  V2E   │──▶│ DTEASN │──▶│ SNN  │──▶│   HUD   │
│  128x128     │   │ 60 FPS │   │ eventos│   │ voxel. │   │ 7 cls│   │ Cyber   │
└──────────────┘   └────────┘   └────────┘   └────────┘   └──────┘   └─────────┘
```

---

## 📑 Tabla de contenidos

- [Características](#-características)
- [Arquitectura del pipeline](#-arquitectura-del-pipeline)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Requisitos del sistema](#-requisitos-del-sistema)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Controles](#-controles)
- [El juego del ahorcado](#-el-juego-del-ahorcado)
- [Métricas del HUD](#-métricas-del-hud)
- [Solución de problemas](#-solución-de-problemas)
- [Notas técnicas](#-notas-técnicas)
- [Licencia](#-licencia)

---

## ✨ Características

- **Pipeline neuromórfico completo**: cámara RGB → eventos sintéticos
  (V2E) → voxelización temporal → SNN → clasificación.
- **Interpolación RIFE** en GPU para alcanzar **60 FPS estables** desde una
  cámara de 30 FPS.
- **SNN `Spike-HAR++ Hybrid`** de 7 clases (arquitectura intacta respecto al
  modelo original):
  `Clapping`, `Meet and Split`, `Sitting`, `Standing Still`, `Walking`,
  `Walking While Reading Book`, `Walking While Using Phone`.
- **Atención espacio-temporal DTEASN** para ponderar eventos por relevancia.
- **HUD Cyberpunk** con:
  - Viewports duales (cámara óptica + canvas DVS) con retículas tácticas.
  - Banner de clase predicha con código cromático por acción.
  - Barra segmentada de carga de spikes SNN (16 bins).
  - Consola de telemetría en tiempo real (FPS, latencias, overruns, buffer).
  - Scanlines CRT opcionales.
- **Juego del ahorcado** controlado por gestos corporales con preguntas y
  palabras adaptadas a niños.
- **Arquitectura multihilo** desacoplada por colas (captura/RIFE, V2E,
  inferencia, render) para máxima estabilidad.

---

## 🧠 Arquitectura del pipeline

### Etapas

| Etapa | Componente | Descripción |
|---|---|---|
| 1. Captura | `CameraThread` | Cámara nativa a **128×128** a 30 FPS (V4L2/DSHOW). |
| 2. Interpolación | `RIFEInterpolator` | Genera frames intermedios por árbol binario para llegar a **60 FPS**. Fallback a blend lineal si no hay GPU/RIFE. |
| 3. Emulación V2E | `V2EEngine` | Convierte frames RGB en eventos neuromórficos `(t, x, y, p)` con thresholds configurables. |
| 4. Buffer | `EventBuffer` | Ventana deslizante de eventos de **1500 ms** con recorte temporal. |
| 5. Atención | `compute_event_attention` | DTEASN: pondera cada evento por proximidad temporal, espacial y de polaridad (K=60 vecinos). |
| 6. Voxelización | `create_frame_sequence` | Eventos → tensor `[T=16, C=2, H=128, W=128]`. |
| 7. Inferencia | `SpikeHAPlusPlusHybrid` | SNN con patch embedding, atención spiking (CB-S3A), rama MLP y cabeza LIF. |
| 8. Render | `LiveRuntime._render` | HUD Cyberpunk a **360×240** por viewport + panel de juego. |

### Hilos

```
┌─────────────────────────┐   queue.Queue   ┌───────────────────┐
│ Hilo 1: cámara + RIFE   │ ───────────────▶│ Hilo 2: V2E 60Hz  │
│ (produce frames)        │                 │ (produce eventos) │
└─────────────────────────┘                 └─────────┬─────────┘
                                                      │ EventBuffer
                                                      ▼
                                            ┌───────────────────┐
                                            │ Hilo 3: Inferencia│
                                            │ (cada 250 ms)     │
                                            └─────────┬─────────┘
                                                      │ result (lock)
                                                      ▼
                                            ┌───────────────────┐
                                            │ Hilo principal    │
                                            │ (render @ vsync)  │
                                            └───────────────────┘
```

---

## 📁 Estructura del proyecto

```
spike_har_cyberpunk/
├── config.py           # Parámetros, hiperparámetros, paleta, mapeos
├── snn_model.py        # Arquitectura SNN Spike-HAR++
├── preprocessing.py    # DTEASN + voxelización de eventos
├── camera.py           # Captura de cámara (hilo)
├── rife.py             # Interpolador RIFE (con carga protegida)
├── v2e.py              # Emulador de eventos + buffer
├── hud.py              # Utilidades de dibujo Cyberpunk
├── quiz_game.py        # Juego del ahorcado con gestos
├── runtime.py          # Orquestador LiveRuntime (3 hilos)
├── main.py             # Punto de entrada CLI
├── requirements.txt
└── README.md
```

| Archivo | Responsabilidad |
|---|---|
| `config.py` | `Config`, `SystemParams`, `HyperParams`, `CLASS_NAMES_7`, `CYBER_COLORS`, `to_game_gesture()`. |
| `snn_model.py` | Clases `SpikeHAPlusPlusHybrid`, `ConvSimplifiedSpikingAttention`, `SpikingMLPBranch`. |
| `preprocessing.py` | `compute_event_attention()` y `create_frame_sequence()`. |
| `camera.py` | `CameraThread` con captura nativa y FPS real. |
| `rife.py` | `RIFEInterpolator` con fallback a blend lineal. |
| `v2e.py` | `EventBuffer` (deque temporal) y `V2EEngine` (emulador + canvas DVS). |
| `hud.py` | `draw_cyber_corners()`, `draw_cyber_reticle()`. |
| `quiz_game.py` | `QuizHangmanGame` (estado, lógica y dibujo del juego). |
| `runtime.py` | `LiveRuntime` (hilos, inferencia, render, telemetría). |
| `main.py` | Parser de argumentos y arranque de `LiveRuntime`. |

---

## 💻 Requisitos del sistema

- **SO**: Linux (probado en Ubuntu 22.04) o Windows.
- **Python**: 3.10+
- **GPU**: NVIDIA con CUDA 12.6 (probado en RTX 3050). CPU también funciona
  pero sin interpolación RIFE.
- **Cámara**: webcam USB estándar (`/dev/video0` en Linux).
- **RAM**: ≥ 8 GB recomendado.
- **Disco**: ~ 3 GB para PyTorch + CUDA.

### Dependencias externas (no instalables con pip)

| Recurso | Ubicación esperada | Notas |
|---|---|---|
| **Practical-RIFE** | `/home/immersivetech/proyectos/v2e/Practical-RIFE` | Clonado desde [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE) con `flownet.pkl` en `train_log/`. Ajustar ruta en `rife.py` y `runtime.py` si difiere. |
| **`diccionario_ninos.py`** | `/home/immersivetech/proyectos/v2e/pruebainterpolacion/` | Debe exponer `obtener_palabras()` y `obtener_preguntas()`. |
| **Checkpoint SNN** | Ruta configurable | `SpikeHarDseantV2E.pth` (por defecto `/home/immersivetech/proyectos/v2e/models/`). |

---

## 🚀 Instalación

### 1. Crear y activar entorno virtual

```bash
python3 -m venv ~/venvs/v2e_env
source ~/venvs/v2e_env/bin/activate
python -m pip install --upgrade pip
```

### 2. Instalar PyTorch con CUDA 12.6

Los wheels `+cu126` **no** están en PyPI estándar; requieren índice especial:

```bash
pip install torch==2.14.0+cu126 \
            torchvision==0.29.0+cu126 \
            torchaudio==2.11.0+cu126 \
            --index-url https://download.pytorch.org/whl/cu126
```

### 3. Instalar el resto de dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar V2E en modo editable (desde tu repo local)

```bash
cd /home/immersivetech/proyectos/v2e
pip install -e .
```

### 5. Verificar CUDA

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → CUDA: True NVIDIA GeForce RTX 3050
```

### 6. Preparar recursos externos

```bash
# RIFE (si aún no lo tienes)
git clone https://github.com/hzwer/Practical-RIFE \
    /home/immersivetech/proyectos/v2e/Practical-RIFE
# Colocar flownet.pkl en Practical-RIFE/train_log/

# Verificar que exista el checkpoint
ls -lh /home/immersivetech/proyectos/v2e/models/SpikeHarDseantV2E.pth

# Verificar diccionario_ninos.py
python -c "from diccionario_ninos import obtener_palabras, obtener_preguntas; print('OK')"
```

---

## ▶️ Uso

### Ejecución básica

```bash
cd spike_har_cyberpunk
python main.py --ckpt /home/immersivetech/proyectos/v2e/models/SpikeHarDseantV2E.pth
```

### Con todas las opciones

```bash
python main.py \
    --ckpt /home/immersivetech/proyectos/v2e/models/SpikeHarDseantV2E.pth \
    --camera 0 \
    --infer_every_ms 250 \
    --no-scanlines
```

### Argumentos CLI

| Argumento | Tipo | Default | Descripción |
|---|---|---|---|
| `--ckpt` | `str` | `models/SpikeHarDseantV2E.pth` | Ruta al checkpoint `.pth` del SNN. |
| `--camera` | `int` | `0` | Índice de la cámara (`/dev/video0`, `/dev/video1`, ...). |
| `--infer_every_ms` | `float` | `250.0` | Cadencia de inferencia SNN en milisegundos. |
| `--no-scanlines` | flag | `False` | Desactiva el efecto CRT scanlines del HUD. |

---

## 🎮 Controles

| Tecla | Acción |
|---|---|
| `q` | Salir de la aplicación. |
| `r` / `R` | Reiniciar el juego del ahorcado. |

---

## 🕹 El juego del ahorcado

El HUD incluye un minijuego de **preguntas y respuestas** controlado por gestos
corporales. La lógica está en `quiz_game.py`.

### Mecánica

1. Aparece una **pregunta** con 4 opciones (`A`, `B`, `C`, `D`).
2. Cada opción está ligada a un **gesto fijo**:

   | Opción | Gesto |
   |---|---|
   | `A` | `Meet and Split` |
   | `B` | `Walking` |
   | `C` | `Standing Still` |
   | `D` | `Clapping` |

3. Realizas el gesto de la opción que crees correcta.
4. **✅ Acierto**: revela 1 letra de la palabra del ahorcado.
5. **❌ Fallo**: pierde 1 vida (3 vidas totales).

### Parámetros de detección

Definidos como constantes de clase en `QuizHangmanGame`:

| Constante | Valor | Descripción |
|---|---|---|
| `MIN_CONFIDENCE` | `0.30` | Actividad mínima (spikes/16) para aceptar una predicción. |
| `DEBOUNCE_COUNT` | `2` | Inferencias consecutivas iguales requeridas. |
| `COOLDOWN` | `5.0` s | Bloqueo tras responder. |
| `PREP_TIME` | `10.0` s | Cuenta regresiva antes de poder responder. |

### HUD del juego

- **Vidas** en la esquina superior derecha del panel.
- **Pregunta resaltada** en cian durante el periodo de preparación.
- **4 tarjetas** de opciones con el color del gesto asociado.
- **Badge de cuenta atrás** en la esquina inferior derecha durante el *prep*.
- **Palabra del ahorcado** con letras reveladas (`_` para ocultas, `X` para fallos).
- **Feedback** `CORRECTO` / `INCORRECTO` durante 1.5 s.

---

## 📊 Métricas del HUD

La consola de telemetría (parte inferior) muestra tres columnas:

### `// STREAM RATES`
| Métrica | Descripción |
|---|---|
| `CAM-IN` | FPS reales de la cámara de entrada. |
| `V2E-OUT` | FPS efectivos del emulador V2E (objetivo: 60). |
| `RIFE-GEN` | Total acumulado de frames interpolados. |

### `// LATENCY PROFILE`
| Métrica | Descripción |
|---|---|
| `RIFE` | Latencia media de interpolación (ms) + estado `OK`/`FALLBACK`. |
| `V2E` | Latencia media del emulador (ms). |
| `DTEASN` | Tiempo de atención + voxelización (ms). |
| `SNN` | Tiempo de forward de la red (ms). |

### `// EVENT BUFFER & SYNC`
| Métrica | Descripción |
|---|---|
| `DVS EVENTS` | Eventos actualmente en el buffer. |
| `OVERRUNS` | Resincronizaciones por sobrepasar el deadline de 60 FPS. |
| `BUF. SIZE` | Ventana temporal del buffer (ms). |

---

## 🔧 Solución de problemas

### `[!] RIFE no disponible (…). Se usará blend lineal.`

- **Causa**: falta el repo `Practical-RIFE`, el archivo `flownet.pkl`, o no
  estás en GPU.
- **Solución**:
  ```bash
  git clone https://github.com/hzwer/Practical-RIFE /home/immersivetech/proyectos/v2e/Practical-RIFE
  # Descargar flownet.pkl y colocarlo en Practical-RIFE/train_log/
  ```
  Verifica la ruta en `rife.py` (`sys.path.append`) y en `runtime.py`
  (`rife_path`).

### `No existe el checkpoint: …`

- Comprueba la ruta pasada a `--ckpt`.
- Si el modelo está en otro sitio:
  ```bash
  python main.py --ckpt /otra/ruta/SpikeHarDseantV2E.pth
  ```

### `Checkpoint incompatible con num_classes.`

- El `.pth` no corresponde a las **7 clases** configuradas en
  `SystemParams.num_classes`. Verifica que estés usando el checkpoint correcto.

### `No se pudo abrir la cámara en índice N.`

- **Linux**: verifica permisos y que el dispositivo exista:
  ```bash
  ls -l /dev/video*
  sudo usermod -aG video $USER   # requiere cerrar sesión
  ```
- **Windows**: usa `--camera 0`, `1`, … y comprueba que no esté en uso por otra app.

### `ModuleNotFoundError: No module named 'diccionario_ninos'`

- El archivo debe estar en `/home/immersivetech/proyectos/v2e/pruebainterpolacion/`.
- O ajusta el `sys.path.append` en `quiz_game.py` a su ubicación real.

### La inferencia SNN tarda demasiado

- Baja `--infer_every_ms` (por ejemplo a `400` o `500`).
- Reduce `max_events` en `HyperParams` (default `80000`).
- Baja `K_neighbors` en DTEASN (default `60`).

### `Overruns` altos en el HUD

- Significa que el hilo V2E no llega a los 60 FPS.
- Comprueba que `torch.cuda.is_available()` sea `True` (sin GPU, RIFE cae a
  blend lineal y V2E podría ir más lento).
- Aumenta `--infer_every_ms` para liberar CPU/GPU.
- Reduce `target_v2e_fps` en `SystemParams` a `45.0` si es necesario.

### El juego no acepta mis gestos

- Comprueba que `MIN_CONFIDENCE` (0.30) sea adecuado para tu escena.
- Asegúrate de mantener el gesto **2 inferencias consecutivas** (~500 ms).
- Espera a que termine el **PREP_TIME** (10 s) y el **COOLDOWN** (5 s).
- Revisa la clase predicha en el banner inferior: debe coincidir con uno de los
  4 gestos (`Meet and Split`, `Walking`, `Standing Still`, `Clapping`).

### `CUDA out of memory`

- Reduce `--infer_every_ms` (menos inferencias simultáneas).
- Cierra otras aplicaciones que usen GPU.
- Como último recurso, ejecuta en CPU:
  ```bash
  CUDA_VISIBLE_DEVICES="" python main.py --ckpt ...
  ```

---

## 📝 Notas técnicas

### Arquitectura del SNN (intacta respecto al original)

- **Patch Embedding**: 2 bloques Conv → GN → LIF con MaxPool.
- **Ramas paralelas**:
  - `ConvSimplifiedSpikingAttention` (CB-S3A): atención spiking
    `Q ⊙ K ⊙ V + x`.
  - `SpikingMLPBranch`: FFN modulado por máscara continua.
  - Rama de atención: `Conv1x1 → GN → ReLU → Conv1x1 → Sigmoid`.
- **Cabeza**: `AdaptiveAvgPool(4x4) → FC → LIF → FC → LIF(output)`.
- **Surrogate**: `atan` (configurable: `atan`, `fast_sigmoid`, `sigmoid`).
- **LIF**: `beta=0.70`, `threshold=0.90`, `init_hidden=True`.

### Preprocesamiento DTEASN

- **Atención espacio-temporal** por K-vecinos (`K=60`):
  - Peso temporal: `exp(-Δt / τ_t)`, con `τ_t = 16.6 ms`.
  - Peso espacial: `exp(-d / τ_s)`, con `τ_s = 2.0 px`.
  - Peso de polaridad: `γ` si igual, `γ·λ_pol` si distinta.
  - Coeficientes: `α=0.26`, `β=0.46`, `γ=0.28`, `λ_pol=0.4`.
- **Voxelización**: `T=16` bins temporales, `max_count=16` por voxel,
  polaridades ON/OFF separadas en 2 canales.

### Parámetros clave (`config.py`)

| Parámetro | Valor | Ubicación |
|---|---|---|
| Sensor de entrada | `128×128×2` | `SystemParams.sensor_size` |
| Duración de muestra | `1500 ms` | `SystemParams.sample_duration_ms` |
| FPS objetivo V2E | `60` | `SystemParams.target_v2e_fps` |
| Inferencia | cada `250 ms` | `SystemParams.infer_every_ms` |
| Embed dim | `48` | `HyperParams.embed_dim` |
| FC hidden | `128` | `HyperParams.fc_hidden` |
| Pool head | `(4, 4)` | `HyperParams.pool_size_head` |
| Bins temporales | `16` | `HyperParams.n_time_bins` |

### Ventana de visualización

- **Entrada SNN**: `128×128` (sin cambios — sensor neuromórfico).
- **Viewports HUD**: `360×240` por cámara (cámara óptica + DVS).
- **Panel de juego**: `726×480`.
- **Ventana total**: ~`750×944` píxeles.

### Ajustes rápidos

| Quiero… | Modificar |
|---|---|
| Cambiar la cámara | `--camera 1` |
| Desactivar scanlines | `--no-scanlines` |
| Inferir más rápido | `--infer_every_ms 150` |
| Cambiar modelo | `--ckpt /otra/ruta.pth` |
| Cambiar resolución del sensor | `SystemParams.sensor_size` en `config.py` |
| Cambiar paleta Cyberpunk | `CYBER_COLORS` en `config.py` |
| Ajustar sensibilidad de gestos | `MIN_CONFIDENCE` en `quiz_game.py` |
| Cambiar palabras/preguntas | Editar `diccionario_ninos.py` |

---

## 📄 Licencia

Este proyecto es de uso interno/académico. Los componentes de terceros
mantienen sus licencias originales:

- **PyTorch** — BSD-3-Clause
- **snnTorch** — MIT
- **Practical-RIFE** — MIT
- **V2E** — ver [SensorsINI/v2e](https://github.com/SensorsINI/v2e)
- **OpenCV** — Apache 2.0

---

## 🙏 Créditos

- **Spike-HAR++**: arquitectura base del SNN.
- **DTEASN**: atención espacio-temporal para eventos neuromórficos.
- **Practical-RIFE**: interpolación de frames por GPU.
- **V2E**: emulación de sensores de eventos neuromórficos.
- **snnTorch**: primitivas de SNN sobre PyTorch.