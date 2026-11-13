from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(120))
    avatar = db.Column(db.String(256), default='default.png')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)

    settings = db.relationship('UserSettings', backref='user', uselist=False, cascade='all, delete-orphan')
    activities = db.relationship('ActivityLog', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'


class UserSettings(db.Model):
    __tablename__ = 'user_settings'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)
    gesture_sensitivity = db.Column(db.Float, default=0.7)
    voice_language = db.Column(db.String(10), default='en-US')
    voice_feedback = db.Column(db.Boolean, default=True)
    camera_enabled = db.Column(db.Boolean, default=True)
    microphone_enabled = db.Column(db.Boolean, default=True)
    theme = db.Column(db.String(20), default='dark')
    default_mode = db.Column(db.String(30), default='multimodal')
    gesture_module = db.Column(db.Boolean, default=True)
    voice_module = db.Column(db.Boolean, default=True)
    facial_module = db.Column(db.Boolean, default=True)
    virtual_mouse_module = db.Column(db.Boolean, default=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'gesture_sensitivity': self.gesture_sensitivity,
            'voice_language': self.voice_language,
            'voice_feedback': self.voice_feedback,
            'camera_enabled': self.camera_enabled,
            'microphone_enabled': self.microphone_enabled,
            'theme': self.theme,
            'default_mode': self.default_mode,
            'gesture_module': self.gesture_module,
            'voice_module': self.voice_module,
            'facial_module': self.facial_module,
            'virtual_mouse_module': self.virtual_mouse_module
        }


class ActivityLog(db.Model):
    __tablename__ = 'activity_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    activity_type = db.Column(db.String(50), nullable=False)  # gesture, voice, facial, system
    activity_name = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text)
    confidence = db.Column(db.Float)
    action_taken = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.activity_type,
            'name': self.activity_name,
            'details': self.details,
            'confidence': self.confidence,
            'action': self.action_taken,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("Database initialized successfully.")
