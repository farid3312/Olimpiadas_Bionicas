"""Orquestador principal en vivo: Pipeline + HUD Cyberpunk."""
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from camera import CameraThread
from config import CYBER_COLORS, CLASS_NAMES_7, Config, to_game_gesture
from hud import draw_cyber_corners, draw_cyber_reticle
from preprocessing import create_frame_sequence
from quiz_game import QuizHangmanGame
from rife import RIFEInterpolator
from snn_model import SpikeHAPlusPlusHybrid
from v2e import EventBuffer, V2EEngine


class LiveRuntime:
    def __init__(self, cfg: Config, ckpt: str, camera_idx: int,
                 infer_ms: float, enable_scanlines: bool = True):
        self.cfg = cfg
        self.infer_ms = infer_ms
        self.enable_scanlines = enable_scanlines
        self.last_render_time = time.perf_counter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        print("=" * 70)
        print(f"[+] DISPOSITIVO SNN: {self.device}")
        if self.device.type == "cuda":
            print(f"    GPU: {torch.cuda.get_device_name(0)}")

        # Modelo SNN
        self.model = SpikeHAPlusPlusHybrid(cfg).to(self.device)
        raw = torch.load(Path(ckpt), map_location=self.device)
        sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        if sd["fc2.weight"].shape[0] != cfg.num_classes:
            raise ValueError("Checkpoint incompatible con num_classes.")
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        print("[+] Checkpoint Spike-HAR++ cargado con éxito.")
        print("=" * 70)

        # Entrada nativa
        self.camera = CameraThread(camera_idx,
                                   cfg.sensor_size[1],
                                   cfg.sensor_size[0],
                                   cfg.camera_fps)

        # RUTA ABSOLUTA DE RIFE (Asegúrate de que esta ruta sea la correcta)
        rife_path = "/home/immersivetech/proyectos/v2e/Practical-RIFE/train_log"
        self.interp = RIFEInterpolator(self.device, rife_path)

        self.v2e = V2EEngine(cfg, self.device.type)
        self.buffer = EventBuffer(cfg.sample_duration_ms)

        # Cola para desacoplar RIFE de V2E
        self.frame_queue = queue.Queue(maxsize=120)
        self.last_v2e_frame = None

        # Estado compartido
        self.running = True
        self.lock = threading.Lock()
        self.result = {
            "label": "INICIALIZANDO...",
            "readout": "Esperando ráfaga de spikes...",
            "spikes": 0,
            "max_spikes": 16,
            "percent": 0.0,
            "t_dteasn": 0.0,
            "t_snn": 0.0,
            "ts": 0.0,
        }

        self.metrics = {
            "v2e_frames": 0,
            "v2e_overruns": 0,
            "rife_ms": deque(maxlen=30),
            "v2e_ms": deque(maxlen=60),
            "v2e_timestamps": deque(maxlen=180),
            "rife_frames": 0,
        }
        self.last_mid_frame = None
        self.hangman = QuizHangmanGame()

    def _generate_intermediate_frames(self, A, B, total_frames):
        """Genera N frames intermedios entre A y B usando RIFE."""
        if total_frames <= 1:
            return [B]
        if total_frames == 2:
            mid = self.interp.interpolate(A, B)
            self.metrics["rife_frames"] += 1
            return [mid, B]

        depth = 1
        while (2 ** depth) < total_frames:
            depth += 1

        def build_tree(node_A, node_B, d):
            if d == 0:
                return [node_B]
            mid = self.interp.interpolate(node_A, node_B)
            self.metrics["rife_frames"] += 1
            return build_tree(node_A, mid, d - 1) + build_tree(mid, node_B, d - 1)

        full_tree = build_tree(A, B, depth)

        if len(full_tree) > total_frames:
            indices = np.linspace(0, len(full_tree) - 1, total_frames).astype(int)
            return [full_tree[i] for i in indices]

        return full_tree

    # ---------- HILO 1: PRODUCTOR (Cámara -> RIFE -> Cola) ----------
    def _capture_rife_loop(self):
        prev_frame = None
        last_fid = -1
        last_time = time.perf_counter()

        while self.running:
            ok, curr, _, fid = self.camera.read()
            if not ok or fid == last_fid:
                time.sleep(0.005)
                continue
            last_fid = fid

            if prev_frame is None:
                prev_frame = curr
                last_time = time.perf_counter()
                continue

            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            # ¿Cuántos frames necesitamos para llenar este hueco de tiempo a 60fps?
            num_frames_to_gen = max(1, int(round(dt * self.cfg.target_v2e_fps)))

            # Generar frames
            t_r = time.perf_counter()
            frames = self._generate_intermediate_frames(prev_frame, curr, num_frames_to_gen)
            self.metrics["rife_ms"].append((time.perf_counter() - t_r) * 1000)

            # Empujar a la cola (si se llena, descartamos los más viejos)
            for f in frames:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(f)

            prev_frame = curr

    # ---------- HILO 2: CONSUMIDOR (Cola -> V2E a exactos 60 FPS) ----------
    def _v2e_loop(self):
        interval = 1.0 / self.cfg.target_v2e_fps
        next_tick = time.perf_counter()

        while self.running:
            next_tick += interval
            now = time.perf_counter()
            sleep_time = next_tick - now

            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -0.1:  # Overrun grave
                self.metrics["v2e_overruns"] += 1
                next_tick = time.perf_counter() + interval

            # Obtener frame de la cola
            try:
                frame = self.frame_queue.get_nowait()
                self.last_v2e_frame = frame
            except queue.Empty:
                # Si RIFE no ha terminado, repetimos el último frame.
                # V2E no generará eventos (no hay movimiento).
                frame = self.last_v2e_frame

            if frame is not None:
                t_v = time.perf_counter()
                block = self.v2e.step(frame, interval)
                self.metrics["v2e_ms"].append((time.perf_counter() - t_v) * 1000)
                self.metrics["v2e_frames"] += 1
                self.metrics["v2e_timestamps"].append(time.perf_counter())

                if block is not None:
                    self.buffer.append(block)

            self.buffer.trim(self.v2e.sim_time_s * 1e6)

    # ---------- HILO 3: INFERENCIA SNN ----------
    def _inference_loop(self):
        last = 0.0
        while self.running:
            wait = max(0.0, (self.infer_ms / 1000.0) - (time.perf_counter() - last))
            time.sleep(wait)
            last = time.perf_counter()
            try:
                self._infer_once()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[!] _infer_once crash: {e}")
                time.sleep(0.1)

    def _infer_once(self):
        if self.buffer.total_events < 100:
            with self.lock:
                self.result["label"] = "EN ESPERA (MOVIMIENTO)"
                self.result["readout"] = f"Eventos acumulados: {self.buffer.total_events}"
                self.result["spikes"] = 0
                self.result["percent"] = 0.0
            return

        ev_all = self.buffer.snapshot()
        if ev_all is None or len(ev_all) < 100:
            return

        if len(ev_all) > self.cfg.max_events:
            step = len(ev_all) // self.cfg.max_events
            ev_all = ev_all[::max(step, 1)]

        evs = np.empty(len(ev_all),
                       dtype=[("t", np.int64), ("x", np.int64),
                              ("y", np.int64), ("p", np.int64)])
        evs["t"], evs["x"], evs["y"], evs["p"] = ev_all.T

        t0 = time.perf_counter()
        voxels = create_frame_sequence(evs, self.cfg).unsqueeze(0).to(self.device)
        t_dteasn = (time.perf_counter() - t0) * 1000.0

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.inference_mode():
            spk = self.model(voxels)
            counts = spk.sum(dim=0)[0]
            k = int(counts.argmax().item())
            sk = int(counts[k].item())
            tot = max(int(counts.sum().item()), 1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_snn = (time.perf_counter() - t1) * 1000.0

        percent = (sk / 16.0) * 100.0
        with self.lock:
            self.result.update(
                label=CLASS_NAMES_7[k],
                readout=f"Actividad SNN: {sk}/16 ({percent:.1f}%)",
                spikes=sk,
                max_spikes=16,
                percent=percent,
                t_dteasn=t_dteasn,
                t_snn=t_snn,
                ts=time.perf_counter(),
            )

            if not self.hangman.game_over:
                print(f"[INF] calling update: label={to_game_gesture(CLASS_NAMES_7[k])} sk={sk}/16")
                self.hangman.update(
                    to_game_gesture(CLASS_NAMES_7[k]),
                    sk, 16, self.infer_ms / 1000.0,
                )

    # ---------- RENDER (HUD Cyberpunk) ----------
    def _render(self):
        VIEW_W, VIEW_H = 360, 240
        HANGMAN_W = VIEW_W * 2 + 6  # Ocupa el ancho de las dos cámaras
        HANGMAN_H = 480             # Altura del panel del juego
        HEADER_H = 46
        FOOTER_H = 174

        # Ancho total ahora es solo el de las 2 cámaras + márgenes
        TOTAL_W = VIEW_W * 2 + 30
        # Altura total suma el juego abajo
        TOTAL_H = HEADER_H + VIEW_H + HANGMAN_H + FOOTER_H + 40

        hud = np.full((TOTAL_H, TOTAL_W, 3), (12, 14, 18), dtype=np.uint8)

        # 1. Preparar los 3 feeds visuales
        _, curr, cam_fps, _ = self.camera.read()
        if curr is not None:
            cam_view = cv2.resize(curr, (VIEW_W, VIEW_H), interpolation=cv2.INTER_LINEAR)
        else:
            cam_view = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)

        dvs_view = cv2.resize(self.v2e.canvas, (VIEW_W, VIEW_H),
                              interpolation=cv2.INTER_NEAREST)

        hm_view = np.zeros((HANGMAN_H, HANGMAN_W, 3), dtype=np.uint8)
        self.hangman.draw(hm_view, 0, 0, HANGMAN_W, HANGMAN_H)

        # Scanlines opcionales
        if self.enable_scanlines:
            cam_view[::3] = (cam_view[::3] * 0.72).astype(np.uint8)
            dvs_view[::3] = (dvs_view[::3] * 0.72).astype(np.uint8)
            hm_view[::3] = (hm_view[::3] * 0.72).astype(np.uint8)

        # 2. Posicionamiento (Cámaras arriba, Juego abajo)
        cam_x, cam_y = 12, HEADER_H + 6
        dvs_x, dvs_y = cam_x + VIEW_W + 6, HEADER_H + 6
        hm_x, hm_y = 12, cam_y + VIEW_H + 6  # Justo debajo de la cámara

        hud[cam_y:cam_y + VIEW_H, cam_x:cam_x + VIEW_W] = cam_view
        hud[dvs_y:dvs_y + VIEW_H, dvs_x:dvs_x + VIEW_W] = dvs_view
        hud[hm_y:hm_y + HANGMAN_H, hm_x:hm_x + HANGMAN_W] = hm_view

        with self.lock:
            r = dict(self.result)

        class_color = CYBER_COLORS.get(r["label"], CYBER_COLORS["Default"])

        # 3. Retículas tácticas y esquinas
        draw_cyber_corners(hud, cam_x, cam_y, VIEW_W, VIEW_H,
                           (0, 230, 255), arm=18, thickness=2)
        draw_cyber_corners(hud, dvs_x, dvs_y, VIEW_W, VIEW_H,
                           (255, 230, 0), arm=18, thickness=2)
        draw_cyber_corners(hud, hm_x, hm_y, HANGMAN_W, HANGMAN_H,
                           (255, 0, 255), arm=18, thickness=2)

        draw_cyber_reticle(hud, cam_x + VIEW_W // 2, cam_y + VIEW_H // 2,
                           size=10, color=(0, 200, 255))
        draw_cyber_reticle(hud, dvs_x + VIEW_W // 2, dvs_y + VIEW_H // 2,
                           size=10, color=(255, 200, 0))

        # 4. Etiquetas sobre las ventanas
        cv2.rectangle(hud, (cam_x, cam_y), (cam_x + 185, cam_y + 20), (20, 20, 24), -1)
        cv2.putText(hud, "CAM OPTICA // 128x128", (cam_x + 6, cam_y + 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 240, 255), 1, cv2.LINE_AA)

        cv2.rectangle(hud, (dvs_x, dvs_y), (dvs_x + 195, dvs_y + 20), (20, 20, 24), -1)
        cv2.putText(hud, "DVS NEUROMORFICO // V2E", (dvs_x + 6, dvs_y + 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, (255, 240, 0), 1, cv2.LINE_AA)

        cv2.rectangle(hud, (hm_x, hm_y), (hm_x + 180, hm_y + 20), (20, 20, 24), -1)
        cv2.putText(hud, "TERMINAL DE JUEGO", (hm_x + 6, hm_y + 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, (255, 0, 255), 1, cv2.LINE_AA)

        # 5. Barra Superior Táctica (Header)
        cv2.line(hud, (12, 8), (TOTAL_W - 12, 8), (40, 50, 60), 1)
        cv2.line(hud, (12, HEADER_H), (TOTAL_W - 12, HEADER_H), (0, 200, 255), 1)
        cv2.putText(hud, "SPIKE-HAR++ CYBERNETIC HUD", (14, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, "PROTOCOL: SNN-DTEASN // 60 FPS", (360, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 180, 200), 1, cv2.LINE_AA)

        cv2.circle(hud, (TOTAL_W - 30, 24), 5, (0, 255, 0), -1)
        cv2.putText(hud, "ONLINE", (TOTAL_W - 75, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

        # 6. Banner de Acción Predicha (Debajo del juego)
        p_top = hm_y + HANGMAN_H + 10
        cv2.rectangle(hud, (12, p_top), (TOTAL_W - 12, p_top + 60), (18, 22, 28), -1)
        cv2.rectangle(hud, (12, p_top), (TOTAL_W - 12, p_top + 60), class_color, 1)
        draw_cyber_corners(hud, 12, p_top, TOTAL_W - 24, 60, class_color, arm=12, thickness=2)

        cv2.putText(hud, "[ CLASE PREDICHA // TARGET ACTION ]", (24, p_top + 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (140, 160, 180), 1, cv2.LINE_AA)

        action_name = r["label"].upper()
        cv2.putText(hud, action_name, (24, p_top + 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(hud, action_name, (24, p_top + 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, class_color, 2, cv2.LINE_AA)

        bar_x = 550
        bar_y = p_top + 34
        sk_val = int(r.get("spikes", 0))
        cv2.putText(hud, f"SPIKE CHARGE: {sk_val}/16", (bar_x, p_top + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 240), 1, cv2.LINE_AA)

        for b in range(16):
            bx = bar_x + b * 11
            color_block = class_color if b < sk_val else (40, 45, 55)
            cv2.rectangle(hud, (bx, bar_y), (bx + 8, bar_y + 14), color_block, -1)

        # 7. Consola de Telemetría (Al final de todo)
        t_top = p_top + 70

        ts = list(self.metrics["v2e_timestamps"])
        if len(ts) >= 2:
            now = time.perf_counter()
            ts_win = [t for t in ts if (now - t) <= 1.0]
            if len(ts_win) >= 2:
                span = ts_win[-1] - ts_win[0]
                v2e_fps = (len(ts_win) - 1) / span if span > 1e-6 else 0.0
            else:
                v2e_fps = 0.0
        else:
            v2e_fps = 0.0

        v2e_ms_arr = list(self.metrics["v2e_ms"])
        v2e_avg = float(np.mean(v2e_ms_arr[-20:])) if v2e_ms_arr else 16.67
        rife_ms = float(np.mean(self.metrics["rife_ms"])) if self.metrics["rife_ms"] else 0.0
        r_frames = self.metrics.get("rife_frames", 0)

        col_w = (TOTAL_W - 24 - 24) // 3

        # Columna 1
        c1_x = 12
        cv2.rectangle(hud, (c1_x, t_top), (c1_x + col_w, t_top + 84), (18, 20, 26), -1)
        cv2.rectangle(hud, (c1_x, t_top), (c1_x + col_w, t_top + 84), (35, 42, 52), 1)
        cv2.putText(hud, "// STREAM RATES", (c1_x + 8, t_top + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 240, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, f"CAM-IN  : {cam_fps:5.1f} FPS", (c1_x + 8, t_top + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

        color_fps = (0, 255, 180) if v2e_fps >= 0.95 * self.cfg.target_v2e_fps else (0, 180, 255)
        cv2.putText(hud, f"V2E-OUT : {v2e_fps:5.1f}/{self.cfg.target_v2e_fps:.0f} FPS",
                    (c1_x + 8, t_top + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color_fps, 1, cv2.LINE_AA)
        cv2.putText(hud, f"RIFE-GEN: {r_frames} frames", (c1_x + 8, t_top + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1, cv2.LINE_AA)

        # Columna 2
        c2_x = c1_x + col_w + 6
        cv2.rectangle(hud, (c2_x, t_top), (c2_x + col_w, t_top + 84), (18, 20, 26), -1)
        cv2.rectangle(hud, (c2_x, t_top), (c2_x + col_w, t_top + 84), (35, 42, 52), 1)
        cv2.putText(hud, "// LATENCY PROFILE", (c2_x + 8, t_top + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 200, 0), 1, cv2.LINE_AA)

        rife_status = "OK" if rife_ms > 0.1 else "FALLBACK"
        rife_color = (200, 200, 200) if rife_ms > 0.1 else (0, 100, 255)
        cv2.putText(hud, f"RIFE  : {rife_ms:4.1f} ms [{rife_status}]", (c2_x + 8, t_top + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, rife_color, 1, cv2.LINE_AA)
        cv2.putText(hud, f"V2E   : {v2e_avg:4.1f} ms", (c2_x + 8, t_top + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(hud, f"DTEASN: {r['t_dteasn']:4.1f} ms | SNN: {r['t_snn']:4.1f} ms",
                    (c2_x + 8, t_top + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        # Columna 3
        c3_x = c2_x + col_w + 6
        cv2.rectangle(hud, (c3_x, t_top), (c3_x + col_w, t_top + 84), (18, 20, 26), -1)
        cv2.rectangle(hud, (c3_x, t_top), (c3_x + col_w, t_top + 84), (35, 42, 52), 1)
        cv2.putText(hud, "// EVENT BUFFER & SYNC", (c3_x + 8, t_top + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 100, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, f"DVS EVENTS: {self.buffer.total_events:,}", (c3_x + 8, t_top + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(hud, f"OVERRUNS  : {self.metrics['v2e_overruns']} resyncs",
                    (c3_x + 8, t_top + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 180, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, f"BUF. SIZE : {self.cfg.sample_duration_ms:.0f} ms",
                    (c3_x + 8, t_top + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Spike-HAR++ CYBERPUNK HUD [Presiona 'q' para salir]", hud)

    # ---------- BUCLE PÚBLICO ----------
    def run(self):
        # Lanzamos TRES hilos separados
        th_capture = threading.Thread(target=self._capture_rife_loop, daemon=True)
        th_v2e = threading.Thread(target=self._v2e_loop, daemon=True)
        th_inf = threading.Thread(target=self._inference_loop, daemon=True)

        th_capture.start()
        th_v2e.start()
        th_inf.start()

        print("[INFO] Sistema Cyberpunk HUD activo. Presiona 'q' para salir.\n")
        try:
            while self.running:
                self._render()
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("r") or key == ord("R"):
                    self.hangman.reset()
                time.sleep(0.001)
        finally:
            self.running = False
            th_capture.join(timeout=1.0)
            th_v2e.join(timeout=1.0)
            th_inf.join(timeout=1.0)
            self.camera.stop()
            self.v2e.cleanup()
            cv2.destroyAllWindows()
            print(f"\n[*] V2E frames totales generados: {self.metrics['v2e_frames']}")
            print(f"[*] Frames interpolados por RIFE: {self.metrics['rife_frames']}")
            print(f"[*] Overruns recuperados: {self.metrics['v2e_overruns']}")
            print("[*] Aplicación finalizada correctamente.")