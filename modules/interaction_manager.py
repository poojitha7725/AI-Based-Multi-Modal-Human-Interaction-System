"""
Interaction Manager Module
Coordinates multi-modal input (gesture + voice + facial) and resolves conflicts.
"""
import time
import threading

class InteractionManager:
    MODES = ['multimodal', 'gesture_only', 'voice_only', 'facial_only']

    def __init__(self):
        self.active_mode = 'multimodal'
        self.gesture_module_enabled = True
        self.voice_module_enabled = True
        self.facial_module_enabled = True
        self.virtual_mouse_enabled = False

        # Current state from each channel
        self.state = {
            'gesture': {'active': False, 'last': 'None', 'timestamp': None},
            'voice': {'active': False, 'last': 'None', 'timestamp': None},
            'facial': {'active': False, 'last': 'None', 'timestamp': None},
            'virtual_mouse': {'active': False, 'last': 'None', 'timestamp': None},
        }

        self.conflict_log = []
        self.interaction_log = []
        self._lock = threading.Lock()

    def update_state(self, channel, data):
        """Update state for a specific input channel."""
        with self._lock:
            if channel in self.state:
                self.state[channel].update({
                    'last': data.get('value', 'None'),
                    'timestamp': time.time(),
                    'active': True
                })
                self.interaction_log.append({
                    'channel': channel,
                    'data': data,
                    'timestamp': time.strftime('%H:%M:%S')
                })
                if len(self.interaction_log) > 50:
                    self.interaction_log.pop(0)

    def resolve(self, gesture=None, voice=None):
        """
        Resolve multi-modal interaction:
        - Voice selects application, gesture confirms.
        - Gesture controls media, voice handles extra commands.
        Returns resolved action dict.
        """
        resolved = {"action": "none", "source": "none", "details": ""}

        # Priority: Voice for application selection, Gesture for confirmation
        if voice and gesture:
            # Multimodal: voice picks target, thumbs up confirms
            if gesture == "Thumbs Up" and voice:
                resolved = {
                    "action": "confirm_voice_command",
                    "source": "multimodal",
                    "details": f"Confirmed voice command '{voice}' with gesture"
                }
                self._log_conflict("multimodal_confirm", resolved)
            elif gesture == "Closed Fist":
                resolved = {
                    "action": "cancel",
                    "source": "gesture",
                    "details": "Voice command cancelled by Closed Fist gesture"
                }
                self._log_conflict("gesture_cancel", resolved)
            else:
                # Both active simultaneously - voice takes priority
                resolved = {
                    "action": "voice_priority",
                    "source": "voice",
                    "details": f"Voice command '{voice}' took priority over gesture '{gesture}'"
                }
        elif voice and not gesture:
            resolved = {"action": "voice_command", "source": "voice",
                        "details": f"Voice command: {voice}"}
        elif gesture and not voice:
            resolved = {"action": "gesture_command", "source": "gesture",
                        "details": f"Gesture command: {gesture}"}

        return resolved

    def _log_conflict(self, conflict_type, resolution):
        self.conflict_log.append({
            'type': conflict_type,
            'resolution': resolution,
            'timestamp': time.strftime('%H:%M:%S')
        })
        if len(self.conflict_log) > 20:
            self.conflict_log.pop(0)

    def set_mode(self, mode):
        if mode in self.MODES:
            self.active_mode = mode
            # Update channel states based on mode
            self.gesture_module_enabled = mode in ['multimodal', 'gesture_only']
            self.voice_module_enabled = mode in ['multimodal', 'voice_only']
            self.facial_module_enabled = mode in ['multimodal', 'facial_only']
            return True
        return False

    def toggle_module(self, module, state):
        if module == 'gesture':
            self.gesture_module_enabled = state
        elif module == 'voice':
            self.voice_module_enabled = state
        elif module == 'facial':
            self.facial_module_enabled = state
        elif module == 'virtual_mouse':
            self.virtual_mouse_enabled = state

    def get_status(self):
        return {
            'mode': self.active_mode,
            'gesture_enabled': self.gesture_module_enabled,
            'voice_enabled': self.voice_module_enabled,
            'facial_enabled': self.facial_module_enabled,
            'virtual_mouse_enabled': self.virtual_mouse_enabled,
            'state': self.state,
            'recent_conflicts': self.conflict_log[-5:]
        }

    def get_interaction_log(self, n=10):
        return self.interaction_log[-n:]
