"""Juego del ahorcado con preguntas y gestos SNN."""
import random
import sys
import time

import cv2
import numpy as np

# Mantiene exactamente la misma ruta original (aunque apunta a un fichero)
sys.path.append("/home/immersivetech/proyectos/v2e/pruebainterpolacion/diccionario_ninos.py")

from diccionario_ninos import obtener_palabras, obtener_preguntas
from hud import draw_cyber_corners


class QuizHangmanGame:
    """
    - Lanza una pregunta del diccionario con 4 opciones (A/B/C/D).
    - Cada letra tiene un gesto FIJO:
        A -> Meet and Split
        B -> Walking
        C -> Standing Still
        D -> Clapping
    - El usuario HACE el gesto de la opción que cree correcta.
    - ✅ Acierto -> revela 1 letra de la palabra del ahorcado.
    - ❌ Fallo   -> pierde 1 vida (3 vidas totales).
    """

    # ── Asignación FIJA letra→gesto (NO tocar el orden) ──
    OPCION_MOVS_FIJA = ["Meet and Split", "Walking", "Standing Still", "Clapping"]

    # ── Parámetros de detección ──
    ACCIONES = ["Clapping", "Walking", "Standing Still", "Meet and Split"]
    MIN_CONFIDENCE = 0.30    # antes 0.50 → "Meet and Split" rara vez superaba 8/16
    DEBOUNCE_COUNT = 2       # 2 inferencias consecutivas iguales
    COOLDOWN = 5.0           # s tras responder (antes 2.0)
    PREP_TIME = 10.0         # s de cuenta atrás (antes 8.0)

    def __init__(self):
        self.palabras_ninos = obtener_palabras()
        self.preguntas = obtener_preguntas()
        self.reset()

    # ---------------- estado ----------------
    def reset(self):
        self.target_word = random.choice(self.palabras_ninos)
        self.display_chars = [' ' if c == ' ' else '_' for c in self.target_word]
        self.current_idx = 0
        self.lives = 3
        self.max_lives = 3
        self.game_over = False
        self.win = False
        self.cooldown = 0.0
        self._last_pred = None
        self._consecutive = 0
        self.last_result = None
        self.last_result_time = 0.0
        self._must_release = False
        self._released_pred = None
        self._resolve_time = 0.0
        self.prep_time_left = self.PREP_TIME
        self._next_question()
        self._skip_spaces()

    def _skip_spaces(self):
        while self.current_idx < len(self.target_word) and self.target_word[self.current_idx] == ' ':
            self.current_idx += 1
        if self.current_idx >= len(self.target_word):
            self.game_over = True
            self.win = (self.lives > 0 and 'X' not in self.display_chars)

    def _next_question(self):
        q = random.choice(self.preguntas)
        self.pregunta = q["pregunta"]
        self.opciones = list(q["opciones"])
        # Empareja 1 gesto por opción. Como ACCIONES tiene exactamente 4 gestos
        # y opciones 4 textos, esto asigna SIEMPRE los 4 gestos barajados:
        # no se puede repetir y no se puede quedar ninguno fuera.
        self.opcion_movs = list(self.OPCION_MOVS_FIJA)      # ← FIJO, no sample()
        self.correcta_idx = q["correcta"]
        self._last_pred = None
        self._consecutive = 0
        self.prep_time_left = self.PREP_TIME

    def _reveal_letter(self):
        if self.current_idx < len(self.target_word):
            self.display_chars[self.current_idx] = self.target_word[self.current_idx]
            self.current_idx += 1
            self._skip_spaces()
            if not self.game_over:
                self._next_question()

    def _fail_letter(self):
        if self.current_idx < len(self.target_word):
            self.display_chars[self.current_idx] = 'X'
            self.current_idx += 1
            self.lives -= 1
            self._skip_spaces()
            if self.lives <= 0:
                self.game_over = True
                self.win = False
            if not self.game_over:
                self._next_question()

    def _resolver(self, idx_elegido):
        if idx_elegido == self.correcta_idx:
            self.last_result = "CORRECTO"
            self._reveal_letter()
        else:
            self.last_result = "INCORRECTO"
            self._fail_letter()
        self.last_result_time = time.perf_counter()
        self.cooldown = self.COOLDOWN
        self._must_release = True
        self._released_pred = self._last_pred
        self._resolve_time = time.perf_counter()   # ← NUEVO

    # ---------------- API llamada desde el hilo de inferencia ----------------
    def update(self, prediction_label, spikes, max_spikes, dt):
        if self.game_over:
            return
        if self.cooldown > 0:
            self.cooldown -= dt
            return
        if self.prep_time_left > 0:
            self.prep_time_left -= dt
            return

        confidence = spikes / max(float(max_spikes), 1.0)

        if self._must_release:
            elapsed = time.perf_counter() - self._resolve_time
            if (prediction_label != self._released_pred
                    or confidence < self.MIN_CONFIDENCE
                    or elapsed > 5.0):
                self._must_release = False
            else:
                print(f"[QUIZ] must_release: pred={prediction_label} "
                      f"released={self._released_pred} elapsed={elapsed:.1f}s")
                return

        if confidence < self.MIN_CONFIDENCE or prediction_label not in self.opcion_movs:
            print(f"[QUIZ] reject: pred={prediction_label} conf={confidence:.2f} "
                  f"min={self.MIN_CONFIDENCE} options={self.opcion_movs}")
            self._last_pred = None
            self._consecutive = 0
            return

        if prediction_label == self._last_pred:
            self._consecutive += 1
        else:
            self._last_pred = prediction_label
            self._consecutive = 1
        print(f"[QUIZ] pred={prediction_label} consec={self._consecutive}/{self.DEBOUNCE_COUNT}")

        if self._consecutive >= self.DEBOUNCE_COUNT:
            idx = self.opcion_movs.index(prediction_label)
            self._resolver(idx)

    # ---------------- dibujo ----------------
    def draw(self, hud, x, y, w, h):
        cv2.rectangle(hud, (x, y), (x + w, y + h), (15, 18, 24), -1)
        cv2.putText(hud, "SISTEMA DE DECODIFICACION // QUIZ", (x + 12, y + 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 0, 255), 1, cv2.LINE_AA)

        # Vidas
        info_x = x + w - 340
        cv2.putText(hud, f"VIDAS: {self.lives}/{self.max_lives}", (info_x, y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 100) if self.lives > 1 else (0, 0, 255), 1, cv2.LINE_AA)
        for i in range(self.max_lives):
            cx_h = info_x + 210 + i * 30
            color_h = (0, 0, 255) if i < self.lives else (60, 60, 60)
            cv2.circle(hud, (cx_h, y + 22), 8, color_h, -1)

        # ── Pregunta (resaltada y agrandada durante el prep) ──
        q_y = y + 78
        in_prep = self.prep_time_left > 0
        cv2.rectangle(hud, (x + 12, q_y - 34), (x + w - 12, q_y + 24), (20, 24, 32), -1)
        if in_prep:
            # Borde cian brillante
            cv2.rectangle(hud, (x + 12, q_y - 34), (x + w - 12, q_y + 24), (0, 255, 255), 2)
            # Texto más grande con glow negro
            base_scale = 0.95
            (tw0, _), _ = cv2.getTextSize(self.pregunta, cv2.FONT_HERSHEY_SIMPLEX, base_scale, 3)
            max_w = w - 60
            scale = base_scale if tw0 <= max_w else base_scale * (max_w / max(tw0, 1))
            (tw, th), _ = cv2.getTextSize(self.pregunta, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
            qx = x + (w - tw) // 2
            qy = q_y + th // 2
            cv2.putText(hud, self.pregunta, (qx, qy),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(hud, self.pregunta, (qx, qy),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(hud, (x + 12, q_y - 34), (x + w - 12, q_y + 24), (0, 200, 255), 1)
            cv2.putText(hud, self.pregunta, (x + 24, q_y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # 4 opciones (asignación fija A/B/C/D)
        mov_colors = {"Clapping":        (0, 230, 255),
                      "Walking":         (255, 255, 0),
                      "Standing Still":  (180, 180, 180),
                      "Meet and Split":  (255, 120, 0)}
        op_y0 = y + 130
        box_w = (w - 48) // 4
        for i, op in enumerate(self.opciones):
            bx = x + 12 + i * (box_w + 8)
            mov = self.opcion_movs[i]
            col = mov_colors.get(mov, (200, 200, 200))
            cv2.rectangle(hud, (bx, op_y0), (bx + box_w, op_y0 + 115), (24, 28, 36), -1)
            cv2.rectangle(hud, (bx, op_y0), (bx + box_w, op_y0 + 115), col, 2)
            cv2.putText(hud, f"{'ABCD'[i]}", (bx + 10, op_y0 + 32),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, col, 2, cv2.LINE_AA)
            cv2.putText(hud, op, (bx + 40, op_y0 + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(hud, f"({mov.upper()})", (bx + 10, op_y0 + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            cv2.putText(hud, "[ HAZ EL GESTO ]", (bx + 10, op_y0 + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 150, 150), 1, cv2.LINE_AA)

        # Palabra del ahorcado
        display_str = " ".join(self.display_chars)
        cv2.putText(hud, f"PALABRA: {display_str}", (x + 24, y + h - 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        # Feedback reciente (oculto durante prep para no chocar con el badge)
        if (self.last_result and not in_prep
                and (time.perf_counter() - self.last_result_time) < 1.5):
            col_res = (0, 255, 0) if self.last_result == "CORRECTO" else (0, 0, 255)
            cv2.putText(hud, self.last_result, (x + w - 220, y + h - 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, col_res, 2, cv2.LINE_AA)

        # Fin de juego
        if self.game_over:
            msg = "¡LO LOGRASTE!" if self.win else "PERDISTE"
            color = (0, 255, 0) if self.win else (0, 0, 255)
            cv2.putText(hud, msg, (x + w // 2 - 110, y + h // 2 + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3, cv2.LINE_AA)
            cv2.putText(hud, "Presiona 'R' para reiniciar",
                        (x + w // 2 - 140, y + h // 2 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # ─── Badge countdown en esquina inferior derecha (sin oscurecer el panel) ───
        if in_prep:
            secs = int(np.ceil(self.prep_time_left))
            color_cd = (0, 255, 100) if secs > 2 else (0, 100, 255)

            badge_w, badge_h = 170, 92
            bx = x + w - badge_w - 14
            by = y + h - badge_h - 14

            # Fondo translúcido SOLO dentro del badge
            roi_b = hud[by:by + badge_h, bx:bx + badge_w]
            dark_b = np.zeros_like(roi_b)
            dark_b[:] = (15, 18, 28)
            hud[by:by + badge_h, bx:bx + badge_w] = cv2.addWeighted(dark_b, 0.85, roi_b, 0.15, 0)

            # Borde neón + esquinas tácticas
            cv2.rectangle(hud, (bx, by), (bx + badge_w, by + badge_h), color_cd, 2)
            draw_cyber_corners(hud, bx, by, badge_w, badge_h, color_cd, arm=10, thickness=2)

            # Etiqueta "PREPARATE"
            txt = "PREPARATE"
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.putText(hud, txt, (bx + (badge_w - tw) // 2, by + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1, cv2.LINE_AA)

            # Número grande
            num = str(secs)
            (nw, nh), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_DUPLEX, 1.7, 3)
            cv2.putText(hud, num, (bx + (badge_w - nw) // 2, by + badge_h - 16),
                        cv2.FONT_HERSHEY_DUPLEX, 1.7, color_cd, 3, cv2.LINE_AA)