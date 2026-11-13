"""
Facial Expression Detection Module
Primary:  DeepFace (deep learning, real model)
Fallback: OpenCV Haar cascade + rule-based approximation
"""
import cv2
import numpy as np
import time
import os

EXPRESSIONS  = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
DISPLAY_MAP  = {
    'angry':   ('Angry',     (30,  30,  220)),
    'disgust': ('Disgusted', (30,  180, 100)),
    'fear':    ('Fearful',   (100, 30,  200)),
    'happy':   ('Happy',     (30,  220, 100)),
    'sad':     ('Sad',       (200, 100, 30 )),
    'surprise':('Surprised', (30,  200, 220)),
    'neutral': ('Neutral',   (180, 180, 180)),
}


class FacialExpressionDetector:
    def __init__(self):
        self.current_expression = 'None'
        self.confidence         = 0.0
        self.expression_history = []
        self.is_active          = False
        self._last_detect_time  = 0
        self._detect_interval   = 0.5   # seconds between inferences
        self._deepface_ok       = False
        self._df                = None
        self._face_cascade      = None
        self._load_models()

    # ── Model loading ─────────────────────────────────────────────
    def _load_models(self):
        # OpenCV Haar cascade (always available)
        try:
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("✅ OpenCV face cascade loaded.")
        except Exception as e:
            print(f"Haar cascade error: {e}")

        # DeepFace
        try:
            from deepface import DeepFace
            # Warm-up: trigger lazy model download silently
            self._df = DeepFace
            self._deepface_ok = True
            print("✅ DeepFace loaded.")
        except Exception as e:
            print(f"⚠️  DeepFace not available: {e}. Using OpenCV fallback.")

    # ── Face detection via OpenCV ─────────────────────────────────
    def _detect_faces_cv(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._face_cascade is None:
            return []
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        return faces if len(faces) > 0 else []

    # ── DeepFace inference ────────────────────────────────────────
    def _analyze_deepface(self, frame):
        try:
            results = self._df.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            if isinstance(results, list):
                results = results[0]
            emotion   = results.get('dominant_emotion', 'neutral').lower()
            emotions  = results.get('emotion', {})
            conf      = float(emotions.get(emotion, 0.0))
            return emotion, conf
        except Exception as e:
            return 'neutral', 0.0

    # ── OpenCV-only approximation (geometry-based on face ROI) ───
    def _analyze_opencv(self, frame, x, y, w, h):
        """
        Very rough heuristic: analyse grayscale variance in eye / mouth
        regions and map to an expression.  Better than pure random.
        """
        roi   = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        if roi.size == 0:
            return 'neutral', 60.0
        # Mouth region (lower 35% of face)
        mouth_y = int(h * 0.65)
        mouth   = roi[mouth_y:, :]
        # Eye region (upper 30%)
        eye     = roi[:int(h*0.30), :]
        mouth_var = float(np.var(mouth)) if mouth.size > 0 else 0
        eye_var   = float(np.var(eye))   if eye.size   > 0 else 0
        edges     = float(np.mean(cv2.Canny(roi, 50, 150)))

        if edges > 30 and mouth_var > 500:
            expr, conf = 'happy',   72.0
        elif edges > 40 and eye_var > 600:
            expr, conf = 'surprise', 68.0
        elif edges > 25 and mouth_var < 200:
            expr, conf = 'angry',   65.0
        elif edges < 15:
            expr, conf = 'sad',     63.0
        else:
            expr, conf = 'neutral', 70.0
        return expr, conf

    # ── Main frame processor ──────────────────────────────────────
    def process_frame(self, frame):
        if not self.is_active:
            return frame, {'expression': 'Inactive', 'confidence': 0.0, 'faces': 0}

        result_frame = frame.copy()
        faces = self._detect_faces_cv(frame)
        info  = {'expression': 'No Face Detected', 'confidence': 0.0, 'faces': len(faces)}

        if len(faces) == 0:
            return result_frame, info

        # Draw rectangles
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(result_frame, (fx, fy), (fx+fw, fy+fh), (0, 200, 100), 2)

        now = time.time()
        if now - self._last_detect_time >= self._detect_interval:
            self._last_detect_time = now
            fx, fy, fw, fh = faces[0]

            if self._deepface_ok:
                expr, conf = self._analyze_deepface(frame)
            else:
                expr, conf = self._analyze_opencv(frame, fx, fy, fw, fh)

            expr = expr.lower()
            if expr not in DISPLAY_MAP:
                expr = 'neutral'

            self.current_expression = expr
            self.confidence         = round(conf, 2)

            display_name = DISPLAY_MAP[expr][0]
            self.expression_history.append({
                'expression': display_name,
                'confidence': self.confidence,
                'timestamp':  time.strftime('%H:%M:%S')
            })
            if len(self.expression_history) > 30:
                self.expression_history.pop(0)

        # Overlay label
        display_name, color = DISPLAY_MAP.get(self.current_expression, ('Unknown', (200,200,200)))
        cv2.putText(result_frame,
                    f'{display_name} ({self.confidence:.1f}%)',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        info.update({
            'expression': self.current_expression,
            'confidence': self.confidence,
            'faces':      len(faces)
        })
        return result_frame, info

    # ── Public API ────────────────────────────────────────────────
    def start(self):
        self.is_active = True

    def stop(self):
        self.is_active          = False
        self.current_expression = 'None'
        self.confidence         = 0.0

    def get_history(self, n=10):
        return self.expression_history[-n:]
