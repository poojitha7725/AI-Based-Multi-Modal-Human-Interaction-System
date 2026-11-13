"""
AI-Based Multi-Modal Human Interaction System
Main Flask Application
"""
import os
import cv2
import base64
import json
import threading
import time
from datetime import datetime
from flask import (Flask, render_template, request, redirect, url_for,
                   flash, jsonify, Response, session)
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, current_user)
from flask_socketio import SocketIO, emit
from config import config

# Database & Models
from database.database import db, User, UserSettings, ActivityLog, init_db

# AI Modules
from modules.hand_gesture import HandGestureRecognizer
from modules.voice_recognition import VoiceRecognitionSystem
from modules.facial_expression import FacialExpressionDetector
from modules.virtual_mouse import VirtualMouseController
from modules.application_control import ApplicationController
from modules.presentation_control import PresentationController
from modules.interaction_manager import InteractionManager

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(config['development'])

db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Global AI module instances (per-process singletons) ───────────────────────
gesture_recognizer  = HandGestureRecognizer()
facial_detector     = FacialExpressionDetector()
virtual_mouse       = VirtualMouseController()
app_controller      = ApplicationController()
presentation_ctrl   = PresentationController()
interaction_mgr     = InteractionManager()
voice_system        = None          # initialised lazily on first use
camera_lock         = threading.Lock()
active_camera       = None          # shared OpenCV capture

# ── Login Manager ─────────────────────────────────────────────────────────────
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            user.last_login = datetime.utcnow()
            db.session.commit()
            login_user(user, remember=bool(remember))
            _log_activity(user.id, 'system', 'Login', f'User {username} logged in')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username  = request.form.get('username', '').strip()
        email     = request.form.get('email', '').strip()
        full_name = request.form.get('full_name', '').strip()
        password  = request.form.get('password', '')
        confirm   = request.form.get('confirm_password', '')

        if User.query.filter_by(username=username).first():
            flash('Username already taken.', 'danger')
            return render_template('register.html')
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return render_template('register.html')
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('register.html')

        user = User(username=username, email=email, full_name=full_name)
        user.set_password(password)
        db.session.add(user)
        db.session.flush()
        # Default settings
        settings = UserSettings(user_id=user.id)
        db.session.add(settings)
        db.session.commit()
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    _log_activity(current_user.id, 'system', 'Logout', f'User {current_user.username} logged out')
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGES
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/dashboard')
@login_required
def dashboard():
    settings   = _get_or_create_settings()
    stats      = _get_dashboard_stats()
    recent_act = ActivityLog.query.filter_by(user_id=current_user.id)\
                    .order_by(ActivityLog.timestamp.desc()).limit(10).all()
    return render_template('dashboard.html', user=current_user,
                           settings=settings, stats=stats,
                           recent_activities=[a.to_dict() for a in recent_act])

@app.route('/gestures')
@login_required
def gestures():
    settings = _get_or_create_settings()
    return render_template('gestures.html', user=current_user, settings=settings)

@app.route('/gesture-test')
@login_required
def gesture_test():
    settings = _get_or_create_settings()
    return render_template('gesture_test.html', user=current_user, settings=settings)

@app.route('/voice')
@login_required
def voice():
    settings = _get_or_create_settings()
    return render_template('voice.html', user=current_user, settings=settings)

@app.route('/facial')
@login_required
def facial():
    settings = _get_or_create_settings()
    return render_template('facial.html', user=current_user, settings=settings)

@app.route('/virtual-mouse')
@login_required
def virtual_mouse_page():
    settings = _get_or_create_settings()
    return render_template('virtual_mouse.html', user=current_user, settings=settings)

@app.route('/applications')
@login_required
def applications():
    safe_apps = app_controller.get_safe_apps()
    return render_template('applications.html', user=current_user,
                           safe_apps=safe_apps)

@app.route('/history')
@login_required
def history():
    page       = request.args.get('page', 1, type=int)
    act_type   = request.args.get('type', 'all')
    query      = ActivityLog.query.filter_by(user_id=current_user.id)
    if act_type != 'all':
        query = query.filter_by(activity_type=act_type)
    logs = query.order_by(ActivityLog.timestamp.desc()).paginate(
        page=page, per_page=20, error_out=False)
    # Analytics
    analytics  = _get_analytics()
    return render_template('history.html', user=current_user,
                           logs=logs, analytics=analytics,
                           current_type=act_type)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings_page():
    settings = _get_or_create_settings()
    if request.method == 'POST':
        try:
            settings.gesture_sensitivity  = float(request.form.get('gesture_sensitivity', 0.7))
            settings.voice_language        = request.form.get('voice_language', 'en-US')
            settings.voice_feedback        = 'voice_feedback' in request.form
            settings.camera_enabled        = 'camera_enabled' in request.form
            settings.microphone_enabled    = 'microphone_enabled' in request.form
            settings.theme                 = request.form.get('theme', 'dark')
            settings.default_mode          = request.form.get('default_mode', 'multimodal')
            settings.gesture_module        = 'gesture_module' in request.form
            settings.voice_module          = 'voice_module' in request.form
            settings.facial_module         = 'facial_module' in request.form
            settings.virtual_mouse_module  = 'virtual_mouse_module' in request.form
            db.session.commit()
            # Apply settings to modules
            gesture_recognizer.update_sensitivity(settings.gesture_sensitivity)
            interaction_mgr.set_mode(settings.default_mode)
            flash('Settings saved successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error saving settings: {e}', 'danger')
    return render_template('settings.html', user=current_user, settings=settings)

@app.route('/about')
@login_required
def about():
    return render_template('about.html', user=current_user)

@app.route('/how-to-use')
@login_required
def how_to_use():
    return render_template('how_to_use.html', user=current_user)

@app.route('/system-check')
@login_required
def system_check():
    return render_template('system_check.html', user=current_user)

# ══════════════════════════════════════════════════════════════════════════════
# CAMERA STREAM (MJPEG)
# ══════════════════════════════════════════════════════════════════════════════
def _open_camera():
    global active_camera
    if active_camera is None or not active_camera.isOpened():
        active_camera = cv2.VideoCapture(0)
        active_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        active_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return active_camera

def _generate_frames(mode='gesture'):
    cap = _open_camera()
    if not cap.isOpened():
        return
    while True:
        with camera_lock:
            ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        try:
            if mode == 'gesture':
                frame, info = gesture_recognizer.process_frame(frame)
            elif mode == 'facial':
                frame, info = facial_detector.process_frame(frame)
            elif mode == 'mouse':
                frame, info = virtual_mouse.process_frame(frame)
            else:
                pass
        except Exception:
            pass
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.route('/video_feed/<mode>')
@login_required
def video_feed(mode):
    return Response(_generate_frames(mode),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ══════════════════════════════════════════════════════════════════════════════
# GESTURE API
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/gesture/start', methods=['POST'])
@login_required
def gesture_start():
    gesture_recognizer.start()
    return jsonify({'status': 'started'})

@app.route('/api/gesture/stop', methods=['POST'])
@login_required
def gesture_stop():
    gesture_recognizer.stop()
    return jsonify({'status': 'stopped'})

@app.route('/api/gesture/status')
@login_required
def gesture_status():
    return jsonify({
        'is_active':       gesture_recognizer.is_active,
        'current_gesture': gesture_recognizer.current_gesture,
        'confidence':      round(gesture_recognizer.confidence, 1),
        'hand_count':      gesture_recognizer.hand_count,
        'hand_labels':     gesture_recognizer.hand_labels,
        'pinch_active':    gesture_recognizer.pinch_active,
        'pinch_distance':  round(gesture_recognizer.pinch_distance, 4),
        'fingers_state':   gesture_recognizer.fingers_state,
        'fps':             round(gesture_recognizer._fps, 1),
        'history':         gesture_recognizer.get_history(),
    })

@app.route('/api/gesture/log', methods=['POST'])
@login_required
def gesture_log():
    data    = request.json or {}
    gesture = data.get('gesture', '')
    action  = gesture_recognizer.get_gesture_action(gesture)
    if gesture and gesture not in ('None', 'No Hand Detected', 'Unknown', 'Inactive'):
        _log_activity(current_user.id, 'gesture', gesture,
                      f'Action: {action}', action_taken=action)
    return jsonify({'status': 'ok', 'action': action})

# ══════════════════════════════════════════════════════════════════════════════
# VOICE API
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/voice/start', methods=['POST'])
@login_required
def voice_start():
    global voice_system
    settings = _get_or_create_settings()
    voice_system = VoiceRecognitionSystem(
        language=settings.voice_language,
        voice_feedback=settings.voice_feedback
    )
    def on_command(result):
        if result.get('status') == 'executed':
            with app.app_context():
                _log_activity(current_user.id, 'voice',
                              result.get('matched', 'unknown'),
                              result.get('command'),
                              action_taken=result.get('action'))
        socketio.emit('voice_command', result)
    voice_system.start(callback=on_command)
    return jsonify({'status': 'started'})

@app.route('/api/voice/stop', methods=['POST'])
@login_required
def voice_stop():
    global voice_system
    if voice_system:
        voice_system.stop()
    return jsonify({'status': 'stopped'})

@app.route('/api/voice/command', methods=['POST'])
@login_required
def voice_command():
    """Process a manually typed command (for testing)."""
    global voice_system
    settings = _get_or_create_settings()
    if not voice_system:
        voice_system = VoiceRecognitionSystem(
            language=settings.voice_language,
            voice_feedback=settings.voice_feedback)
    data    = request.json or {}
    text    = data.get('command', '')
    result  = voice_system.process_command(text)
    if result.get('status') == 'executed':
        _log_activity(current_user.id, 'voice',
                      result.get('matched', 'unknown'),
                      result.get('command'),
                      action_taken=str(result.get('action')))
    return jsonify(result)

@app.route('/api/voice/status')
@login_required
def voice_status():
    global voice_system
    if not voice_system:
        return jsonify({'is_running': False, 'last_command': 'None', 'history': [],
                        'mic_available': False, 'mic_type': 'none'})
    return jsonify({
        'is_running': voice_system.is_running,
        'is_listening': voice_system.is_listening,
        'last_command': voice_system.last_command,
        'history': voice_system.get_history(),
        'mic_available': voice_system.mic_available,
        'mic_type': voice_system.mic_type
    })

@app.route('/api/voice/mic-status')
@login_required
def voice_mic_status():
    """Check microphone availability."""
    from modules.voice_recognition import PYAUDIO_OK, SOUNDDEVICE_OK
    try:
        if PYAUDIO_OK:
            import speech_recognition as sr
            mics = sr.Microphone.list_microphone_names()
            return jsonify({'available': True, 'type': 'pyaudio',
                            'devices': mics[:5], 'count': len(mics)})
        elif SOUNDDEVICE_OK:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devs = [d['name'] for d in devices if d['max_input_channels'] > 0]
            return jsonify({'available': len(input_devs) > 0, 'type': 'sounddevice',
                            'devices': input_devs[:5], 'count': len(input_devs)})
        else:
            return jsonify({'available': False, 'type': 'none',
                            'error': 'Neither PyAudio nor sounddevice is available.',
                            'fix': 'Install PyAudio: pip install pyaudio or sounddevice: pip install sounddevice'})
    except Exception as e:
        return jsonify({'available': False, 'type': 'unknown', 'error': str(e)})

# ══════════════════════════════════════════════════════════════════════════════
# FACIAL API
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/facial/start', methods=['POST'])
@login_required
def facial_start():
    facial_detector.start()
    return jsonify({'status': 'started'})

@app.route('/api/facial/stop', methods=['POST'])
@login_required
def facial_stop():
    facial_detector.stop()
    return jsonify({'status': 'stopped'})

@app.route('/api/facial/status')
@login_required
def facial_status():
    return jsonify({
        'is_active': facial_detector.is_active,
        'expression': facial_detector.current_expression,
        'confidence': facial_detector.confidence,
        'history': facial_detector.get_history()
    })

@app.route('/api/facial/log', methods=['POST'])
@login_required
def facial_log():
    data = request.json or {}
    expr = data.get('expression', '')
    conf = data.get('confidence', 0)
    if expr and expr not in ('None', 'No Face Detected', 'Inactive'):
        _log_activity(current_user.id, 'facial', expr,
                      f'Confidence: {conf}%', confidence=conf)
    return jsonify({'status': 'ok'})

# ══════════════════════════════════════════════════════════════════════════════
# VIRTUAL MOUSE API
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/mouse/start', methods=['POST'])
@login_required
def mouse_start():
    virtual_mouse.start()
    return jsonify({'status': 'started'})

@app.route('/api/mouse/stop', methods=['POST'])
@login_required
def mouse_stop():
    virtual_mouse.stop()
    return jsonify({'status': 'stopped'})

@app.route('/api/mouse/sensitivity', methods=['POST'])
@login_required
def mouse_sensitivity():
    data = request.json or {}
    virtual_mouse.update_sensitivity(data.get('sensitivity', 1.5))
    return jsonify({'status': 'ok'})

# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION CONTROL API
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/app/open', methods=['POST'])
@login_required
def app_open():
    data   = request.json or {}
    result = app_controller.open_application(data.get('app', ''))
    if result['status'] == 'success':
        _log_activity(current_user.id, 'system', 'Open App',
                      result['message'], action_taken=result['message'])
    return jsonify(result)

@app.route('/api/app/browser', methods=['POST'])
@login_required
def app_browser():
    data   = request.json or {}
    result = app_controller.browser_control(data.get('action', ''))
    return jsonify(result)

@app.route('/api/app/media', methods=['POST'])
@login_required
def app_media():
    data   = request.json or {}
    result = app_controller.media_control(data.get('action', ''))
    return jsonify(result)

@app.route('/api/app/screenshot', methods=['POST'])
@login_required
def app_screenshot():
    result = app_controller.take_screenshot()
    if result['status'] == 'success':
        _log_activity(current_user.id, 'system', 'Screenshot',
                      result['message'], action_taken=result.get('path'))
    return jsonify(result)

# ══════════════════════════════════════════════════════════════════════════════
# PRESENTATION API
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/presentation/<action>', methods=['POST'])
@login_required
def presentation_action(action):
    actions = {
        'start': presentation_ctrl.start_presentation,
        'next':  presentation_ctrl.next_slide,
        'prev':  presentation_ctrl.prev_slide,
        'end':   presentation_ctrl.end_presentation,
        'blank': presentation_ctrl.blank_screen,
    }
    fn = actions.get(action)
    if fn:
        result = fn()
        _log_activity(current_user.id, 'system', f'Presentation: {action}',
                      str(result))
        return jsonify(result)
    return jsonify({'status': 'error', 'message': 'Unknown action'})

# ══════════════════════════════════════════════════════════════════════════════
# INTERACTION MANAGER API
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/interaction/mode', methods=['POST'])
@login_required
def set_interaction_mode():
    data = request.json or {}
    ok   = interaction_mgr.set_mode(data.get('mode', 'multimodal'))
    return jsonify({'status': 'ok' if ok else 'error',
                    'mode': interaction_mgr.active_mode})

@app.route('/api/interaction/status')
@login_required
def interaction_status():
    return jsonify(interaction_mgr.get_status())

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS / HISTORY API
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/analytics')
@login_required
def api_analytics():
    return jsonify(_get_analytics())

@app.route('/api/history/clear', methods=['POST'])
@login_required
def clear_history():
    ActivityLog.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({'status': 'cleared'})

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS API
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/settings', methods=['GET'])
@login_required
def api_get_settings():
    settings = _get_or_create_settings()
    return jsonify(settings.to_dict())

@app.route('/api/settings', methods=['POST'])
@login_required
def api_save_settings():
    data     = request.json or {}
    settings = _get_or_create_settings()
    for key, val in data.items():
        if hasattr(settings, key):
            setattr(settings, key, val)
    db.session.commit()
    return jsonify({'status': 'saved'})

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM DIAGNOSTICS API
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/system/check')
@login_required
def system_check_api():
    """Run real-time system diagnostics and return results."""
    import sys
    results = []

    # 1. Python
    results.append({'name': 'Python', 'status': 'ok',
                    'detail': f'Python {sys.version.split()[0]}'})

    # 2. Flask
    try:
        import flask
        results.append({'name': 'Flask', 'status': 'ok', 'detail': f'v{flask.__version__}'})
    except Exception as e:
        results.append({'name': 'Flask', 'status': 'fail', 'detail': str(e), 'fix': 'pip install flask'})

    # 3. Flask-SocketIO
    try:
        import flask_socketio
        results.append({'name': 'Flask-SocketIO', 'status': 'ok', 'detail': f'v{flask_socketio.__version__}'})
    except Exception as e:
        results.append({'name': 'Flask-SocketIO', 'status': 'fail', 'detail': str(e), 'fix': 'pip install flask-socketio'})

    # 4. OpenCV
    try:
        import cv2
        results.append({'name': 'OpenCV (cv2)', 'status': 'ok', 'detail': f'v{cv2.__version__}'})
    except Exception as e:
        results.append({'name': 'OpenCV (cv2)', 'status': 'fail', 'detail': str(e), 'fix': 'pip install opencv-python'})

    # 5. MediaPipe
    try:
        import mediapipe as mp
        results.append({'name': 'MediaPipe', 'status': 'ok', 'detail': f'v{mp.__version__}'})
    except Exception as e:
        results.append({'name': 'MediaPipe', 'status': 'fail', 'detail': str(e), 'fix': 'pip install mediapipe'})

    # 6. Gesture Model file
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'gesture_recognizer.task')
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024*1024)
        results.append({'name': 'Gesture Model (.task)', 'status': 'ok',
                        'detail': f'gesture_recognizer.task ({size_mb:.1f} MB)'})
    else:
        results.append({'name': 'Gesture Model (.task)', 'status': 'fail',
                        'detail': 'gesture_recognizer.task not found in models/ folder',
                        'fix': 'Download from MediaPipe Model Cards page and place in models/ folder'})

    # 7. Hand Landmarker Model file
    lm_path = os.path.join(os.path.dirname(__file__), 'models', 'hand_landmarker.task')
    if os.path.exists(lm_path):
        size_mb = os.path.getsize(lm_path) / (1024*1024)
        results.append({'name': 'Hand Landmarker Model (.task)', 'status': 'ok',
                        'detail': f'hand_landmarker.task ({size_mb:.1f} MB)'})
    else:
        results.append({'name': 'Hand Landmarker Model (.task)', 'status': 'fail',
                        'detail': 'hand_landmarker.task not found in models/ folder',
                        'fix': 'Download from MediaPipe Model Cards page and place in models/ folder'})

    # 8. Camera
    try:
        import cv2
        with camera_lock:
            cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                results.append({'name': 'Camera (Webcam)', 'status': 'ok',
                                'detail': 'Camera device 0 accessible and streaming'})
            else:
                results.append({'name': 'Camera (Webcam)', 'status': 'warn',
                                'detail': 'Camera opened but could not read a frame',
                                'fix': 'Ensure camera is not blocked or in use by another app'})
        else:
            results.append({'name': 'Camera (Webcam)', 'status': 'fail',
                            'detail': 'No camera detected at index 0',
                            'fix': 'Connect a USB webcam and ensure drivers are installed'})
    except Exception as e:
        results.append({'name': 'Camera (Webcam)', 'status': 'fail', 'detail': str(e)})

    # 9. Microphone
    try:
        from modules.voice_recognition import PYAUDIO_OK, SOUNDDEVICE_OK
        if PYAUDIO_OK:
            import speech_recognition as sr
            mics = sr.Microphone.list_microphone_names()
            results.append({'name': 'Microphone (PyAudio)', 'status': 'ok',
                            'detail': f'{len(mics)} device(s) found via PyAudio'})
        elif SOUNDDEVICE_OK:
            import sounddevice as sd
            devs = [d for d in sd.query_devices() if d['max_input_channels'] > 0]
            s = 'ok' if devs else 'warn'
            results.append({'name': 'Microphone (sounddevice)', 'status': s,
                            'detail': f'{len(devs)} input device(s) found' if devs else 'No input devices found'})
        else:
            results.append({'name': 'Microphone', 'status': 'fail',
                            'detail': 'Neither PyAudio nor sounddevice available',
                            'fix': 'Install PyAudio: pip install pyaudio  OR  pip install sounddevice'})
    except Exception as e:
        results.append({'name': 'Microphone', 'status': 'fail', 'detail': str(e)})

    # 10. SpeechRecognition
    try:
        import speech_recognition as sr
        results.append({'name': 'SpeechRecognition', 'status': 'ok', 'detail': f'v{sr.__version__}'})
    except Exception as e:
        results.append({'name': 'SpeechRecognition', 'status': 'fail', 'detail': str(e),
                        'fix': 'pip install SpeechRecognition'})

    # 11. PyAutoGUI
    try:
        import pyautogui
        w, h = pyautogui.size()
        results.append({'name': 'PyAutoGUI (system control)', 'status': 'ok',
                        'detail': f'Screen resolution: {w}x{h} px'})
    except Exception as e:
        results.append({'name': 'PyAutoGUI (system control)', 'status': 'fail', 'detail': str(e),
                        'fix': 'pip install pyautogui'})

    # 12. DeepFace
    try:
        from deepface import DeepFace
        results.append({'name': 'DeepFace (facial AI)', 'status': 'ok', 'detail': 'Available and imported'})
    except Exception as e:
        results.append({'name': 'DeepFace (facial AI)', 'status': 'warn',
                        'detail': f'Not available: {str(e)[:80]}. OpenCV fallback will be used.',
                        'fix': 'pip install deepface  (Warning: requires TensorFlow/Keras)'})

    # 13. Database
    try:
        from database.database import User
        count = User.query.count()
        db_path = os.path.join(os.path.dirname(__file__), 'database.db')
        size_kb = os.path.getsize(db_path) / 1024 if os.path.exists(db_path) else 0
        results.append({'name': 'Database (SQLite)', 'status': 'ok',
                        'detail': f'{count} user(s) registered | DB size: {size_kb:.1f} KB'})
    except Exception as e:
        results.append({'name': 'Database (SQLite)', 'status': 'fail', 'detail': str(e),
                        'fix': 'Run: python app.py (it will auto-create the database)'})

    ok_count   = sum(1 for r in results if r['status'] == 'ok')
    fail_count = sum(1 for r in results if r['status'] == 'fail')
    warn_count = sum(1 for r in results if r['status'] == 'warn')
    return jsonify({
        'results': results,
        'summary': {'ok': ok_count, 'fail': fail_count, 'warn': warn_count, 'total': len(results)}
    })

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def _get_or_create_settings():
    settings = UserSettings.query.filter_by(user_id=current_user.id).first()
    if not settings:
        settings = UserSettings(user_id=current_user.id)
        db.session.add(settings)
        db.session.commit()
    return settings

def _log_activity(user_id, activity_type, name, details=None,
                  confidence=None, action_taken=None):
    try:
        log = ActivityLog(user_id=user_id, activity_type=activity_type,
                          activity_name=name, details=details,
                          confidence=confidence, action_taken=action_taken)
        db.session.add(log)
        db.session.commit()
    except Exception:
        db.session.rollback()

def _get_dashboard_stats():
    uid = current_user.id
    total   = ActivityLog.query.filter_by(user_id=uid).count()
    gestures= ActivityLog.query.filter_by(user_id=uid, activity_type='gesture').count()
    voice   = ActivityLog.query.filter_by(user_id=uid, activity_type='voice').count()
    facial  = ActivityLog.query.filter_by(user_id=uid, activity_type='facial').count()
    system  = ActivityLog.query.filter_by(user_id=uid, activity_type='system').count()
    return {'total': total, 'gestures': gestures, 'voice': voice,
            'facial': facial, 'system': system}

def _get_analytics():
    from sqlalchemy import func
    uid  = current_user.id
    # Top gestures
    top_gestures = db.session.query(
        ActivityLog.activity_name,
        func.count(ActivityLog.id).label('count')
    ).filter_by(user_id=uid, activity_type='gesture')\
     .group_by(ActivityLog.activity_name)\
     .order_by(func.count(ActivityLog.id).desc()).limit(5).all()

    top_voice = db.session.query(
        ActivityLog.activity_name,
        func.count(ActivityLog.id).label('count')
    ).filter_by(user_id=uid, activity_type='voice')\
     .group_by(ActivityLog.activity_name)\
     .order_by(func.count(ActivityLog.id).desc()).limit(5).all()

    top_facial = db.session.query(
        ActivityLog.activity_name,
        func.count(ActivityLog.id).label('count')
    ).filter_by(user_id=uid, activity_type='facial')\
     .group_by(ActivityLog.activity_name)\
     .order_by(func.count(ActivityLog.id).desc()).limit(5).all()

    # Daily activity last 7 days
    from datetime import timedelta
    today = datetime.utcnow().date()
    daily = []
    for i in range(6, -1, -1):
        day  = today - timedelta(days=i)
        cnt  = ActivityLog.query.filter(
            ActivityLog.user_id == uid,
            func.date(ActivityLog.timestamp) == day
        ).count()
        daily.append({'date': str(day), 'count': cnt})

    return {
        'top_gestures': [{'name': r[0], 'count': r[1]} for r in top_gestures],
        'top_voice':    [{'name': r[0], 'count': r[1]} for r in top_voice],
        'top_facial':   [{'name': r[0], 'count': r[1]} for r in top_facial],
        'daily_activity': daily,
        'stats': _get_dashboard_stats()
    }

# ── SocketIO Events ───────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    emit('connected', {'data': 'Connected to AI System'})

@socketio.on('request_status')
def on_request_status():
    emit('system_status', {
        'gesture': {
            'active':         gesture_recognizer.is_active,
            'gesture':        gesture_recognizer.current_gesture,
            'hand_count':     gesture_recognizer.hand_count,
            'hand_labels':    gesture_recognizer.hand_labels,
            'pinch_active':   gesture_recognizer.pinch_active,
            'pinch_distance': round(gesture_recognizer.pinch_distance, 4),
            'fingers_state':  gesture_recognizer.fingers_state,
            'fps':            round(gesture_recognizer._fps, 1),
        },
        'facial': {
            'active':     facial_detector.is_active,
            'expression': facial_detector.current_expression,
            'confidence': facial_detector.confidence,
        },
        'voice': {
            'active':       voice_system.is_running if voice_system else False,
            'last_command': voice_system.last_command if voice_system else 'None',
        },
        'mouse': {
            'active':     virtual_mouse.is_active,
            'mode':       virtual_mouse.status.get('mode', 'Idle'),
            'action':     virtual_mouse.status.get('action', 'None'),
            'position':   virtual_mouse.status.get('position', (0, 0)),
            'pinch':      virtual_mouse.status.get('pinch', False),
            'hand_count': virtual_mouse.status.get('hand_count', 0),
            'fps':        virtual_mouse.status.get('fps', 0.0),
        },
        'interaction': interaction_mgr.get_status()
    })

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("✅ Database tables created.")
    print("🚀 Starting AI-Based Multi-Modal Human Interaction System...")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=False, allow_unsafe_werkzeug=True)


