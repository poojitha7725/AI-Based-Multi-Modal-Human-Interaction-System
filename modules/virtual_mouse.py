"""
Virtual Mouse Module – MediaPipe Tasks API (v1.x compatible)
Provides desktop mouse control via hand landmarks using mp.tasks.vision.HandLandmarker.
- Index fingertip moves the cursor; thumb-index pinch triggers left click / drag.
- Supports right-click, scroll, and pinch-hold drag.
"""
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import os
import time

# ── MediaPipe Tasks API ───────────────────────────────────────────────────────
_tasks   = mp.tasks
_vision  = mp.tasks.vision
_BaseOpt = _tasks.BaseOptions
_HL      = _vision.HandLandmarker
_HLOpts  = _vision.HandLandmarkerOptions
_RunMode = _vision.RunningMode

# Connections
_CONNS = _vision.HandLandmarksConnections.HAND_CONNECTIONS

def _conn_pairs():
    return [(c.start, c.end) for c in _CONNS]

HAND_CONNECTIONS = _conn_pairs()
FINGERTIP_IDS    = {4, 8, 12, 16, 20}

_MODEL_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
_MODEL_PATH = os.path.join(_MODEL_DIR, 'hand_landmarker.task')

# Colors (BGR)
BLUE_LINE   = (220, 100, 20)
BLUE_DOT    = (255,  50,  5)
BLUE_TIP    = (255, 200,  0)
BLUE_WRIST  = (220, 180, 50)
PINCH_COLOR = (0, 220, 255)
MOVE_COLOR  = (255, 180,  0)
CLICK_COLOR = (0, 255, 120)

# Thresholds
PINCH_THRESH    = 0.08
CLICK_THRESH_PX = 35
DRAG_HOLD_S     = 0.08
CLICK_COOLDOWN_S = 0.30
RIGHT_COOLDOWN_S = 0.50


class VirtualMouseController:
    """Control the OS cursor using hand gestures.
    Uses MediaPipe HandLandmarker (Tasks API) and pyautogui for mouse actions.
    """

    def __init__(self, sensitivity: float = 1.5):
        self.sensitivity = float(sensitivity)
        self.is_active   = False
        self.screen_w, self.screen_h = pyautogui.size()
        # EMA smoothing
        self._prev_mx     = self.screen_w  // 2
        self._prev_my     = self.screen_h // 2
        self._smooth_alpha = 0.35
        # Click / drag state
        self._last_click  = 0.0
        self._last_right  = 0.0
        self._pinch_start = None
        self._dragging    = False
        # FPS
        self._fps       = 0.0
        self._fps_prev  = time.time()
        self._fps_alpha = 0.1
        self.status = {
            'mode': 'Idle', 'action': 'None',
            'position': (0, 0), 'pinch': False,
            'hand_count': 0, 'fps': 0.0,
        }
        self._landmarker = None
        self._model_ok   = False

    # ── Model init ────────────────────────────────────────────────────────
    def _init_model(self):
        if not os.path.exists(_MODEL_PATH):
            print(f"⚠️  hand_landmarker.task not found at {_MODEL_PATH}. Virtual mouse disabled.")
            self._model_ok = False
            return
        try:
            options = _HLOpts(
                base_options=_BaseOpt(model_asset_path=_MODEL_PATH),
                running_mode=_RunMode.IMAGE,
                num_hands=4,
                min_hand_detection_confidence=0.60,
                min_hand_presence_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self._landmarker = _HL.create_from_options(options)
            self._model_ok   = True
            print("✅ Virtual mouse module ready (mediapipe.tasks HandLandmarker).")
        except Exception as e:
            print(f"❌ Virtual mouse init error: {e}")
            self._model_ok = False

    # ── Smoothing ─────────────────────────────────────────────────────────
    def _smooth(self, raw_x: float, raw_y: float) -> tuple:
        self._prev_mx = int(self._smooth_alpha * raw_x + (1 - self._smooth_alpha) * self._prev_mx)
        self._prev_my = int(self._smooth_alpha * raw_y + (1 - self._smooth_alpha) * self._prev_my)
        return (max(0, min(self._prev_mx, self.screen_w - 1)),
                max(0, min(self._prev_my, self.screen_h - 1)))

    # ── Distance helpers ──────────────────────────────────────────────────
    def _px_dist(self, a, b, w, h) -> float:
        return float(np.sqrt((a.x*w - b.x*w)**2 + (a.y*h - b.y*h)**2))

    def _norm_dist(self, a, b) -> float:
        return float(np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2))

    # ── Drawing ───────────────────────────────────────────────────────────
    def _draw_skeleton(self, frame, lm_px, hand_idx):
        line_col = BLUE_LINE if hand_idx == 0 else (160, 70, 15)
        dot_col  = BLUE_DOT  if hand_idx == 0 else (200, 40,  5)
        for a, b in HAND_CONNECTIONS:
            if a < len(lm_px) and b < len(lm_px):
                cv2.line(frame, lm_px[a], lm_px[b], line_col, 2, cv2.LINE_AA)
        for i, pt in enumerate(lm_px):
            if i in FINGERTIP_IDS:
                cv2.circle(frame, pt, 8, BLUE_TIP, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 10, line_col, 1, cv2.LINE_AA)
            elif i == 0:
                cv2.circle(frame, pt, 7, BLUE_WRIST, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 9, line_col, 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 5, dot_col, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 6, line_col, 1, cv2.LINE_AA)

    def _draw_status(self, frame, mode, action, pinch, hand_count, sx, sy):
        lines = [
            (f"Mode: {mode}", (255, 180, 60)),
            (f"Action: {action}", (100, 255, 180)),
            (f"Pinch: {'YES' if pinch else 'no'}", PINCH_COLOR if pinch else (120, 120, 120)),
            (f"Hands: {hand_count}", (180, 255, 220)),
            (f"Cursor: ({sx},{sy})", (200, 200, 200)),
            (f"FPS: {self._fps:.1f}", (160, 200, 255)),
        ]
        pad, lh = 8, 22
        ph = len(lines) * lh + pad * 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (285, ph), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        for i, (txt, col) in enumerate(lines):
            cv2.putText(frame, txt, (pad, pad + (i+1)*lh - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    # ── Main processing ──────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray):
        if not self.is_active:
            return frame, self.status

        now_t = time.time()
        inst  = 1.0 / max(now_t - self._fps_prev, 1e-6)
        self._fps = self._fps_alpha * inst + (1 - self._fps_alpha) * self._fps
        self._fps_prev = now_t

        h, w = frame.shape[:2]
        if self._landmarker is None:
            self._init_model()

        if not self._model_ok:
            cv2.putText(frame, "Model file missing - see system check", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2, cv2.LINE_AA)
            return frame, self.status

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            results = self._landmarker.detect(mp_image)
        except Exception:
            return frame, self.status

        mode       = 'Idle'
        action     = 'None'
        pinch      = False
        pos        = (self._prev_mx, self._prev_my)
        hand_count = 0

        if results and results.hand_landmarks:
            hand_count = len(results.hand_landmarks)
            for idx, hand_lm in enumerate(results.hand_landmarks):
                lm_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
                self._draw_skeleton(frame, lm_px, idx)

            # Primary hand
            primary = results.hand_landmarks[0]
            lm_px   = [(int(lm.x * w), int(lm.y * h)) for lm in primary]
            ix, iy  = lm_px[8]
            raw_x   = np.interp(primary[8].x, [0.07, 0.93], [0, self.screen_w])
            raw_y   = np.interp(primary[8].y, [0.07, 0.93], [0, self.screen_h])
            sx, sy  = self._smooth(raw_x, raw_y)
            pos     = (sx, sy)

            pinch_dist = self._norm_dist(primary[4], primary[8])
            if pinch_dist < PINCH_THRESH:
                pinch = True
                pyautogui.moveTo(sx, sy, _pause=False)
                if self._pinch_start is None:
                    self._pinch_start = time.time()
                held = time.time() - self._pinch_start
                if held >= DRAG_HOLD_S and not self._dragging:
                    self._dragging = True
                    pyautogui.mouseDown(_pause=False)
                    action = 'Drag Start'
                    mode   = 'Drag'
                elif self._dragging:
                    action = f'Dragging ({sx},{sy})'
                    mode   = 'Drag'
                else:
                    action = 'Pinch Hold'
                cv2.line(frame, lm_px[4], lm_px[8], PINCH_COLOR, 3, cv2.LINE_AA)
                mid_pt = ((lm_px[4][0]+lm_px[8][0])//2, (lm_px[4][1]+lm_px[8][1])//2)
                cv2.circle(frame, mid_pt, 12, PINCH_COLOR, -1, cv2.LINE_AA)
                cv2.putText(frame, "PINCH", (mid_pt[0]+14, mid_pt[1]+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, PINCH_COLOR, 2, cv2.LINE_AA)
            else:
                if self._dragging:
                    pyautogui.mouseUp(_pause=False)
                    self._dragging = False
                    action = 'Drop'
                elif self._pinch_start is not None and (time.time() - self._pinch_start) < DRAG_HOLD_S:
                    if time.time() - self._last_click > CLICK_COOLDOWN_S:
                        pyautogui.click(sx, sy, _pause=False)
                        action = 'Left Click (Pinch)'
                        self._last_click = time.time()
                self._pinch_start = None

                fingers_up = self._fingers_up(primary)
                thumb, index, middle, ring, pinky = fingers_up
                if index and not middle and not ring:
                    mode   = 'Move'
                    pyautogui.moveTo(sx, sy, _pause=False)
                    action = f'Move ({sx},{sy})'
                    cv2.circle(frame, (ix, iy), 12, MOVE_COLOR, 2, cv2.LINE_AA)
                elif index and middle and not ring and not pinky:
                    mode = 'Click'
                    dist_px = self._px_dist(primary[8], primary[12], w, h)
                    pyautogui.moveTo(sx, sy, _pause=False)
                    if dist_px < CLICK_THRESH_PX and time.time() - self._last_click > CLICK_COOLDOWN_S:
                        pyautogui.click(_pause=False)
                        action = 'Left Click'
                        self._last_click = time.time()
                    cv2.circle(frame, (ix, iy), 14, CLICK_COLOR, cv2.FILLED, cv2.LINE_AA)
                elif index and ring and not middle:
                    mode = 'Right Click'
                    if time.time() - self._last_right > RIGHT_COOLDOWN_S:
                        pyautogui.rightClick(_pause=False)
                        action = 'Right Click'
                        self._last_right = time.time()
                elif index and pinky and not middle and not ring:
                    mode  = 'Scroll'
                    delta = int((0.5 - primary[8].y) * 20)
                    if delta != 0:
                        pyautogui.scroll(delta, _pause=False)
                    action = f'Scroll {delta:+d}'

            self._draw_status(frame, mode, action, pinch, hand_count, sx, sy)
        else:
            if self._dragging:
                try:
                    pyautogui.mouseUp(_pause=False)
                except Exception:
                    pass
                self._dragging = False
            self._pinch_start = None
            cv2.putText(frame, "No Hand Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 180), 1, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {self._fps:.1f}", (10, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1, cv2.LINE_AA)

        self.status.update({
            'mode': mode, 'action': action, 'position': pos,
            'pinch': pinch, 'hand_count': hand_count,
            'fps': round(self._fps, 1),
        })
        return frame, self.status

    # ── Helper: finger state ─────────────────────────────────────────────
    def _fingers_up(self, lm):
        wrist_x   = lm[0].x
        mid_mcp_x = lm[9].x
        fingers   = []
        if wrist_x < mid_mcp_x:
            fingers.append(lm[4].x < lm[3].x)
        else:
            fingers.append(lm[4].x > lm[3].x)
        for tip in (8, 12, 16, 20):
            fingers.append(lm[tip].y < lm[tip - 2].y)
        return fingers

    # ── Public API ────────────────────────────────────────────────────────
    def start(self):
        self.is_active = True
        if self._landmarker is None:
            self._init_model()

    def stop(self):
        if self._dragging:
            try:
                pyautogui.mouseUp(_pause=False)
            except Exception:
                pass
        self.is_active    = False
        self._dragging    = False
        self._pinch_start = None
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
        self.status = {
            'mode': 'Idle', 'action': 'None',
            'position': (0, 0), 'pinch': False,
            'hand_count': 0, 'fps': 0.0,
        }

    def update_sensitivity(self, s: float):
        self.sensitivity   = float(s)
        self._smooth_alpha = min(0.9, max(0.1, s / 3.0))
        if self._landmarker:
            self._landmarker.close()
        self._landmarker = None
        self._model_ok   = False
        if self.is_active:
            self._init_model()

    def __del__(self):
        try:
            if self._landmarker:
                self._landmarker.close()
                self._landmarker = None
        except Exception:
            pass
