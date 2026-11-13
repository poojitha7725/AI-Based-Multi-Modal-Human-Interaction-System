# AI-Based Multi-Modal Human Interaction System

> **B.Tech Data Science Major Project**  
> Combines Hand Gesture Recognition, Voice Commands, and Facial Expression Detection into one intelligent dashboard.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.14, Flask 2.3, Flask-SocketIO |
| Database | SQLite via Flask-SQLAlchemy |
| AI / CV | MediaPipe 1.2.2 Tasks API, OpenCV 4.8, DeepFace |
| Voice | SpeechRecognition, pyttsx3 |
| System | PyAutoGUI, Pillow |
| Frontend | HTML5, CSS3, JavaScript, Chart.js, Socket.io |

---

## Project Structure

```
project/
├── app.py                    ← Main Flask app (all routes + SocketIO)
├── config.py                 ← Configuration
├── requirements.txt          ← Dependencies
├── run.bat                   ← Windows one-click launcher
├── database.db               ← SQLite (auto-created on first run)
├── models/
│   ├── gesture_recognizer.task   ← MediaPipe GestureRecognizer model
│   └── hand_landmarker.task      ← MediaPipe HandLandmarker model
├── database/
│   └── database.py           ← SQLAlchemy models: User, UserSettings, ActivityLog
├── modules/
│   ├── hand_gesture.py       ← MediaPipe Tasks gestures
│   ├── voice_recognition.py  ← SpeechRecognition + pyttsx3
│   ├── facial_expression.py  ← DeepFace + OpenCV fallback
│   ├── virtual_mouse.py      ← MediaPipe Tasks hand landmarks
│   ├── application_control.py
│   ├── presentation_control.py
│   └── interaction_manager.py
├── templates/                ← 12 Jinja2 pages
└── static/
    └── css/style.css         ← Premium dark theme
```

---

## Installation (Windows 11 + VS Code)

### Step 1 — Prerequisites
- Python 3.9–3.14 from https://python.org  
- VS Code + Python extension  
- A working webcam and microphone

### Step 2 — Clone / Open project
Open the project folder in VS Code.

### Step 3 — Create virtual environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 4 — Install dependencies
```powershell
pip install -r requirements.txt
```

### Step 5 — Install PyAudio (voice recognition)
PyAudio requires a C compiler on Windows. Easiest way:
```powershell
pip install pipwin
pipwin install pyaudio
```
If that fails, download the matching `.whl` from  
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio  
then: `pip install PyAudio‑0.2.14‑cpXX‑cpXX‑win_amd64.whl`

### Step 6 — MediaPipe model files
The two model files are pre-downloaded into the `models/` folder:
- `gesture_recognizer.task` — classifies Open Palm, Closed Fist, Thumbs Up, Victory, etc.
- `hand_landmarker.task` — provides 21 hand landmarks (used by Virtual Mouse)

If absent, re-download:
```powershell
Invoke-WebRequest "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task" -OutFile models\gesture_recognizer.task
Invoke-WebRequest "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile models\hand_landmarker.task
```

### Step 7 — Run the application
```powershell
python app.py
```
Or simply double-click **`run.bat`**.

### Step 8 — Open in browser
```
http://localhost:5000
```

---

## First Use

1. **Register** a new account on the Register page.
2. **Log in** with your credentials.
3. You land on the **Dashboard** — all 4 module status cards show Inactive.
4. Click **▶ Start** on any module card to activate it.
5. The live camera feed streams via MJPEG.

---

## Feature Summary

| Module | Description |
|--------|-------------|
| Hand Gesture | 6 gestures → system actions (Open Palm, Closed Fist, Thumbs Up, Victory, One Finger, Two Fingers) |
| Voice Assistant | 12 commands via Google Speech API (open YouTube, screenshot, volume, scroll …) |
| Facial Expression | DeepFace: Happy, Sad, Angry, Surprised, Neutral, Fearful, Disgusted |
| Virtual Mouse | Index finger moves cursor; pinch → click; index+ring → right-click; index+pinky → scroll |
| App Control | Safe whitelist launch of calc, notepad, browser URLs, media keys |
| Presentation | PowerPoint keyboard shortcuts via gesture mapping |
| Dashboard | Live MJPEG feed, status pills, Chart.js activity charts, Socket.io live updates |
| Activity History | Paginated log, 7-day trend chart, top commands bar charts, CSV export |
| Settings | Per-user sensitivity, language, theme, module toggles — saved to SQLite |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `mediapipe has no attribute solutions` | You have MediaPipe ≥ 1.0. This project uses the Tasks API — no fix needed. |
| Camera not found | Ensure no other app is using the webcam. Check Device Manager. |
| Voice not working | Install PyAudio (see Step 5). Ensure microphone is set as default device. |
| DeepFace slow first run | It downloads VGG-Face model (~500MB) on first call. Give it time. |
| Port 5000 in use | Change port in `app.py` last line: `socketio.run(app, port=5001, ...)` |

---

## License
Academic use only — B.Tech Data Science Major Project.
