"""
Hand Gesture Recognition Module – MediaPipe Tasks API (v1.x / Python 3.14 compatible)

Supports 12 gestures with temporal smoothing:
  Open Palm, Closed Fist, Thumbs Up, Thumbs Down, Victory Sign,
  One Finger, Two Fingers, Three Fingers, Four Fingers,
  Pinch, Pointing, OK Sign

Each gesture returns a confidence value 0–100 based on how cleanly
the finger/thumb conditions are met.
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import time
from collections import deque

# ── MediaPipe Tasks API ───────────────────────────────────────────────────────
_tasks   = mp.tasks
_vision  = mp.tasks.vision
_BaseOpt = _tasks.BaseOptions
_HL      = _vision.HandLandmarker
_HLOpts  = _vision.HandLandmarkerOptions
_RunMode = _vision.RunningMode
_CONNS   = _vision.HandLandmarksConnections.HAND_CONNECTIONS

HAND_CONNECTIONS = [(c.start, c.end) for c in _CONNS]
FINGERTIP_IDS    = {4, 8, 12, 16, 20}

# Model path
_MODEL_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
_MODEL_PATH = os.path.join(_MODEL_DIR, 'hand_landmarker.task')

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
BLUE_LINE   = (220, 100, 20)
BLUE_DOT    = (255,  50,  5)
BLUE_TIP    = (255, 200,  0)
BLUE_WRIST  = (220, 180, 50)
PINCH_COLOR = (0, 220, 255)

# ── Gesture → action map ──────────────────────────────────────────────────────
ACTION_MAP = {
    'Open Palm':    'pause_virtual_mouse',
    'Closed Fist':  'stop_cursor_movement',
    'Thumbs Up':    'confirm_action',
    'Thumbs Down':  'cancel_action',
    'Victory Sign': 'scroll_mode',
    'One Finger':   'move_cursor',
    'Two Fingers':  'scroll_mode',
    'Three Fingers':'three_fingers',
    'Four Fingers': 'four_fingers',
    'Pinch':        'left_click',
    'Pointing':     'move_cursor',
    'OK Sign':      'confirm_action',
}

# ── Thresholds ────────────────────────────────────────────────────────────────
PINCH_THRESH      = 0.07   # normalised thumb-index distance for pinch
OK_THRESH         = 0.09   # normalised distance for OK sign
PINCH_COOLDOWN_S  = 0.25
SMOOTHING_FRAMES  = 7      # majority-vote window


class HandGestureRecognizer:
    """Real-time hand gesture recognizer with 12 gesture classes,
    temporal smoothing, and per-gesture confidence scores.
    """

    def __init__(self, sensitivity: float = 0.6):
        self.sensitivity      = float(sensitivity)
        self.is_active        = False
        self.current_gesture  = 'None'
        self.confidence       = 0.0
        self.gesture_history  = []
        self.last_log_time    = 0.0
        self._last_pinch_time = 0.0

        # Per-frame live data
        self.hand_count     = 0
        self.hand_labels    = []
        self.all_landmarks  = []
        self.pinch_active   = False
        self.pinch_distance = 0.0
        self.fingers_state  = []

        # Temporal smoothing buffer
        self._smooth_buffers = {}   # hand_idx -> deque of gesture strings

        # FPS
        self._fps       = 0.0
        self._fps_prev  = time.time()
        self._fps_alpha = 0.1

        # MediaPipe landmarker (lazy init)
        self._landmarker = None
        self._model_ok   = False

    # ─────────────────────────────────────────────────────────────────────────
    # Model initialisation
    # ─────────────────────────────────────────────────────────────────────────
    def _init_model(self):
        if not os.path.exists(_MODEL_PATH):
            print(f"⚠️  hand_landmarker.task not found at {_MODEL_PATH}.")
            self._model_ok = False
            return
        try:
            det   = max(0.20, self.sensitivity - 0.15)
            track = max(0.20, self.sensitivity - 0.15)
            opts  = _HLOpts(
                base_options=_BaseOpt(model_asset_path=_MODEL_PATH),
                running_mode=_RunMode.IMAGE,
                num_hands=4,
                min_hand_detection_confidence=det,
                min_hand_presence_confidence=track,
                min_tracking_confidence=track,
            )
            self._landmarker = _HL.create_from_options(opts)
            self._model_ok   = True
            print("✅ HandGestureRecognizer ready (MediaPipe Tasks HandLandmarker).")
        except Exception as e:
            print(f"❌ HandLandmarker init error: {e}")
            self._model_ok = False

    # ─────────────────────────────────────────────────────────────────────────
    # Landmark helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _fingers_state(self, lm):
        """Return [thumb, index, middle, ring, pinky] True=extended."""
        fingers   = []
        wrist_x   = lm[0].x
        mid_mcp_x = lm[9].x
        # Thumb – horizontal check, direction depends on hand orientation
        if wrist_x < mid_mcp_x:   # right hand (mirror = left on screen)
            fingers.append(lm[4].x < lm[3].x)
        else:                       # left hand
            fingers.append(lm[4].x > lm[3].x)
        # Other four fingers: tip y < pip y → extended
        for tip in (8, 12, 16, 20):
            fingers.append(lm[tip].y < lm[tip - 2].y)
        return fingers

    def _norm_dist(self, a, b) -> float:
        return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2))

    def _thumb_direction(self, lm):
        """Return 'up', 'down', or 'side' based on thumb tip vs wrist."""
        tip   = lm[4]
        wrist = lm[0]
        dy    = tip.y - wrist.y   # positive = tip below wrist (down)
        dx    = abs(tip.x - wrist.x)
        if dy < -0.12 and dy < -dx:    # tip significantly above wrist
            return 'up'
        if dy > 0.12 and dy > dx:      # tip significantly below wrist
            return 'down'
        return 'side'

    # ─────────────────────────────────────────────────────────────────────────
    # Gesture classification  →  (name, confidence 0-100)
    # ─────────────────────────────────────────────────────────────────────────
    def _classify(self, lm, fingers):
        thumb, idx, mid, ring, pinky = fingers
        extended = sum(fingers)
        conf = 0.0

        # ── Pinch – thumb & index close together (overrides other gestures) ──
        pinch_d = self._norm_dist(lm[4], lm[8])
        if pinch_d < PINCH_THRESH:
            conf = round((1 - pinch_d / PINCH_THRESH) * 100, 1)
            return 'Pinch', conf

        # ── OK Sign – thumb & index form circle, other fingers extended ──────
        ok_d = self._norm_dist(lm[4], lm[8])
        if ok_d < OK_THRESH and mid and ring and pinky:
            conf = round((1 - ok_d / OK_THRESH) * 100, 1)
            return 'OK Sign', conf

        # ── Open Palm – all five extended ─────────────────────────────────────
        if extended == 5:
            return 'Open Palm', 95.0

        # ── Closed Fist – none extended ───────────────────────────────────────
        if extended == 0:
            return 'Closed Fist', 95.0

        # ── Thumbs Up / Thumbs Down – only thumb extended ────────────────────
        if thumb and not idx and not mid and not ring and not pinky:
            direction = self._thumb_direction(lm)
            if direction == 'up':
                return 'Thumbs Up', 90.0
            if direction == 'down':
                return 'Thumbs Down', 90.0
            return 'Thumbs Up', 70.0   # side – default to up

        # ── Pointing / One Finger – only index extended, thumb folded ─────────
        if not thumb and idx and not mid and not ring and not pinky:
            return 'One Finger', 90.0

        # ── Victory Sign / Two Fingers – index + middle, no ring/pinky ───────
        if not thumb and idx and mid and not ring and not pinky:
            return 'Victory Sign', 92.0

        # ── Three Fingers – index + middle + ring ─────────────────────────────
        if not thumb and idx and mid and ring and not pinky:
            return 'Three Fingers', 90.0

        # ── Four Fingers – index + middle + ring + pinky (no thumb) ──────────
        if not thumb and idx and mid and ring and pinky:
            return 'Four Fingers', 90.0

        # ── Pointing – thumb + index only ─────────────────────────────────────
        if thumb and idx and not mid and not ring and not pinky:
            return 'Pointing', 80.0

        # ── Two Fingers (thumb + index + middle) ──────────────────────────────
        if thumb and idx and mid and not ring and not pinky:
            return 'Two Fingers', 75.0

        return 'Unknown', 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal smoothing
    # ─────────────────────────────────────────────────────────────────────────
    def _smooth_gesture(self, hand_idx: int, raw_gesture: str) -> str:
        """Majority vote over last SMOOTHING_FRAMES to reduce flicker."""
        if hand_idx not in self._smooth_buffers:
            self._smooth_buffers[hand_idx] = deque(maxlen=SMOOTHING_FRAMES)
        buf = self._smooth_buffers[hand_idx]
        buf.append(raw_gesture)
        if not buf:
            return raw_gesture
        # Find most common
        counts = {}
        for g in buf:
            counts[g] = counts.get(g, 0) + 1
        return max(counts, key=counts.get)

    # ─────────────────────────────────────────────────────────────────────────
    # Drawing helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_skeleton(self, frame, lm_px, hand_idx, label, gesture, conf):
        line_col = BLUE_LINE if hand_idx == 0 else (160, 70, 15)
        dot_col  = BLUE_DOT  if hand_idx == 0 else (200, 40,  5)

        # Connections
        for a, b in HAND_CONNECTIONS:
            if a < len(lm_px) and b < len(lm_px):
                cv2.line(frame, lm_px[a], lm_px[b], line_col, 2, cv2.LINE_AA)

        # Landmarks
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

        # Label badge above wrist
        wrist = lm_px[0]
        txt   = f"{label} | {gesture} {conf:.0f}%"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        lx = max(0, wrist[0] - tw // 2)
        ly = max(22, wrist[1] - 18)
        cv2.rectangle(frame, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), (10, 10, 10), -1)
        cv2.putText(frame, txt, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 220, 255), 1, cv2.LINE_AA)

    def _draw_pinch(self, frame, lm_px, pinch_active, pinch_dist):
        p4, p8 = lm_px[4], lm_px[8]
        if pinch_active:
            cv2.line(frame, p4, p8, PINCH_COLOR, 3, cv2.LINE_AA)
            mid = ((p4[0]+p8[0])//2, (p4[1]+p8[1])//2)
            cv2.circle(frame, mid, 10, PINCH_COLOR, -1, cv2.LINE_AA)
            cv2.putText(frame, "PINCH!", (mid[0]+12, mid[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, PINCH_COLOR, 2, cv2.LINE_AA)
        else:
            cv2.line(frame, p4, p8, (60, 60, 180), 1, cv2.LINE_AA)

    def _draw_info_overlay(self, frame, hand_count, gesture, conf, pinch_active, pinch_dist, hand_labels):
        lines = [
            (f"Hands: {hand_count}", (100, 255, 180)),
            (f"Labels: {', '.join(hand_labels) or 'N/A'}", (180, 220, 255)),
            (f"Gesture: {gesture}", (255, 255, 180)),
            (f"Confidence: {conf:.1f}%", (200, 200, 255)),
            (f"Pinch: {'ACTIVE' if pinch_active else 'off'} d={pinch_dist:.3f}",
             PINCH_COLOR if pinch_active else (120, 120, 120)),
            (f"FPS: {self._fps:.1f}", (160, 200, 255)),
        ]
        pad, lh, pw = 8, 22, 320
        ph = len(lines) * lh + pad * 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (pw, ph), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        for i, (txt, col) in enumerate(lines):
            cv2.putText(frame, txt, (pad, pad + (i+1)*lh - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)

    def _draw_no_hand_overlay(self, frame):
        cv2.putText(frame, "No Hand Detected — show hand to camera",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (80, 80, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {self._fps:.1f}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 100, 100), 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    # Main frame processor
    # ─────────────────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray):
        if not self.is_active:
            return frame, {
                'gesture': 'Inactive', 'confidence': 0.0,
                'fingers': [], 'hand_count': 0, 'hand_labels': [],
                'pinch_active': False, 'pinch_distance': 0.0,
                'fps': 0.0, 'all_landmarks_px': [],
            }

        # FPS
        now  = time.time()
        inst = 1.0 / max(now - self._fps_prev, 1e-6)
        self._fps      = self._fps_alpha * inst + (1 - self._fps_alpha) * self._fps
        self._fps_prev = now

        h, w = frame.shape[:2]

        if self._landmarker is None:
            self._init_model()

        if not self._model_ok:
            cv2.putText(frame, "hand_landmarker.task missing — see System Check",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 80, 255), 2, cv2.LINE_AA)
            return frame, {
                'gesture': 'Model Missing', 'confidence': 0.0,
                'fingers': [], 'hand_count': 0, 'hand_labels': [],
                'pinch_active': False, 'pinch_distance': 0.0,
                'fps': round(self._fps, 1), 'all_landmarks_px': [],
            }

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            results = self._landmarker.detect(mp_image)
        except Exception as exc:
            print(f"Hand processing error: {exc}")
            results = None

        # ── Defaults ──────────────────────────────────────────────────────────
        gesture      = 'No Hand Detected'
        confidence   = 0.0
        fingers      = []
        hand_count   = 0
        hand_labels  = []
        all_lm_px    = []
        pinch_active = False
        pinch_dist   = 0.0

        if results and results.hand_landmarks:
            hand_count  = len(results.hand_landmarks)
            handedness_ = results.handedness or []

            for idx, hand_lm in enumerate(results.hand_landmarks):
                label  = handedness_[idx][0].display_name if idx < len(handedness_) else 'Right'
                hand_labels.append(label)
                lm_px  = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
                all_lm_px.append(lm_px)

                fingers_i    = self._fingers_state(hand_lm)
                raw_g, raw_c = self._classify(hand_lm, fingers_i)
                smoothed_g   = self._smooth_gesture(idx, raw_g)
                # Confidence stays from current frame for responsiveness
                conf_i = raw_c if smoothed_g == raw_g else round(raw_c * 0.85, 1)

                self._draw_skeleton(frame, lm_px, idx, label, smoothed_g, conf_i)

                if idx == 0:
                    fingers    = fingers_i
                    gesture    = smoothed_g
                    confidence = conf_i

                    # Pinch tracking
                    pinch_dist = self._norm_dist(hand_lm[4], hand_lm[8])
                    now_ts     = time.time()
                    if pinch_dist < PINCH_THRESH and (now_ts - self._last_pinch_time) > PINCH_COOLDOWN_S:
                        pinch_active = True
                        self._last_pinch_time = now_ts
                    else:
                        pinch_active = pinch_dist < PINCH_THRESH  # sticky while held
                    self._draw_pinch(frame, lm_px, pinch_active, pinch_dist)

            self._draw_info_overlay(frame, hand_count, gesture, confidence,
                                     pinch_active, pinch_dist, hand_labels)
        else:
            # Clear smoothing buffers when hands disappear
            self._smooth_buffers.clear()
            self._draw_no_hand_overlay(frame)

        # ── Update state ──────────────────────────────────────────────────────
        self.hand_count      = hand_count
        self.hand_labels     = hand_labels
        self.all_landmarks   = all_lm_px
        self.pinch_active    = pinch_active
        self.pinch_distance  = pinch_dist
        self.fingers_state   = fingers
        self.current_gesture = gesture
        self.confidence      = confidence
        self._log_gesture(gesture, confidence)

        return frame, {
            'gesture':          gesture,
            'confidence':       round(confidence, 1),
            'fingers':          fingers,
            'hand_count':       hand_count,
            'hand_labels':      hand_labels,
            'pinch_active':     pinch_active,
            'pinch_distance':   round(pinch_dist, 4),
            'fps':              round(self._fps, 1),
            'all_landmarks_px': all_lm_px,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # History logging
    # ─────────────────────────────────────────────────────────────────────────
    def _log_gesture(self, gesture: str, confidence: float):
        skip = {'No Hand Detected', 'None', 'Unknown', 'Inactive',
                'No Hand', 'Model Missing'}
        now  = time.time()
        if gesture not in skip and now - self.last_log_time > 1.0:
            self.gesture_history.append({
                'gesture':    gesture,
                'confidence': round(confidence, 1),
                'action':     ACTION_MAP.get(gesture, ''),
                'timestamp':  time.strftime('%H:%M:%S'),
            })
            if len(self.gesture_history) > 50:
                self.gesture_history.pop(0)
            self.last_log_time = now

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    def start(self):
        self.is_active = True
        if self._landmarker is None:
            self._init_model()

    def stop(self):
        self.is_active       = False
        self.current_gesture = 'None'
        self.confidence      = 0.0
        self.hand_count      = 0
        self.pinch_active    = False
        self._smooth_buffers.clear()
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

    def get_gesture_action(self, gesture: str) -> str:
        return ACTION_MAP.get(gesture, 'no_action')

    def get_history(self, n: int = 10) -> list:
        return self.gesture_history[-n:]

    def update_sensitivity(self, sensitivity: float):
        self.sensitivity = float(sensitivity)
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
