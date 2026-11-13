"""
Voice Recognition Module
Uses SpeechRecognition library with sounddevice backend (no PyAudio needed).
Falls back to Google Speech API for recognition.
"""
import speech_recognition as sr
import threading
import time
import subprocess
import os
import webbrowser
import queue
import numpy as np

# Try sounddevice for audio capture (works without PyAudio)
try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except ImportError:
    SOUNDDEVICE_OK = False

# Try pyaudio (may not be available)
try:
    import pyaudio
    PYAUDIO_OK = True
except ImportError:
    PYAUDIO_OK = False

# Try pyttsx3 for TTS
try:
    import pyttsx3
    PYTTSX3_OK = True
except ImportError:
    PYTTSX3_OK = False


class SoundDeviceMicrophone:
    """A microphone source that uses sounddevice instead of PyAudio."""
    SAMPLE_RATE = 16000
    CHUNK_SIZE  = 1024

    def __init__(self):
        self._queue = queue.Queue()
        self._stream = None
        self.SAMPLE_WIDTH = 2  # 16-bit

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, *args):
        self._stop()

    def _callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        audio_bytes = (indata * 32767).astype(np.int16).tobytes()
        self._queue.put(audio_bytes)

    def _start(self):
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=self.CHUNK_SIZE,
            callback=self._callback
        )
        self._stream.start()

    def _stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self, size):
        """Read audio bytes - compatible with sr.AudioSource interface."""
        data = b''
        try:
            chunk = self._queue.get(timeout=0.5)
            data = chunk[:size * 2]  # 2 bytes per sample
        except queue.Empty:
            data = b'\x00' * (size * 2)
        return data


class VoiceRecognitionSystem:
    SUPPORTED_COMMANDS = {
        "open youtube":       ("browser", "https://www.youtube.com"),
        "open google":        ("browser", "https://www.google.com"),
        "open calculator":    ("app", "calc"),
        "open notepad":       ("app", "notepad"),
        "play music":         ("media", "play"),
        "pause music":        ("media", "pause"),
        "increase volume":    ("system", "volume_up"),
        "decrease volume":    ("system", "volume_down"),
        "scroll up":          ("scroll", "up"),
        "scroll down":        ("scroll", "down"),
        "take screenshot":    ("system", "screenshot"),
        "stop assistant":     ("system", "stop"),
        "open file explorer": ("app", "explorer"),
        "open browser":       ("browser", "https://www.google.com"),
        "open settings":      ("app", "ms-settings:"),
    }

    def __init__(self, language='en-US', voice_feedback=True):
        self.recognizer       = sr.Recognizer()
        self.language         = language
        self.voice_feedback   = voice_feedback
        self.is_listening     = False
        self.is_running       = False
        self.last_command     = "None"
        self.command_history  = []
        self.listen_thread    = None
        self._tts_lock        = threading.Lock()
        self.mic_available    = PYAUDIO_OK or SOUNDDEVICE_OK
        self.mic_type         = "pyaudio" if PYAUDIO_OK else ("sounddevice" if SOUNDDEVICE_OK else "none")
        self._init_tts()

    def _init_tts(self):
        self.tts_engine = None
        if not PYTTSX3_OK:
            return
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 165)
            self.tts_engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"TTS init error: {e}")
            self.tts_engine = None

    def speak(self, text):
        if not self.voice_feedback or not self.tts_engine:
            return
        def _speak():
            with self._tts_lock:
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"TTS error: {e}")
        threading.Thread(target=_speak, daemon=True).start()

    def listen_once(self):
        """Listen for a single voice command and return recognized text."""
        if not self.mic_available:
            time.sleep(1)
            return None

        try:
            if PYAUDIO_OK:
                # Use standard SpeechRecognition microphone (PyAudio)
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
            elif SOUNDDEVICE_OK:
                # Use sounddevice-based recording
                audio = self._record_sounddevice(duration=5)
                if audio is None:
                    return None
            else:
                return None

            text = self.recognizer.recognize_google(audio, language=self.language)
            return text.lower().strip()
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Google Speech API error: {e}")
            return None
        except Exception as e:
            print(f"listen_once error: {e}")
            return None

    def _record_sounddevice(self, duration=5):
        """Record audio using sounddevice and return sr.AudioData."""
        try:
            sample_rate = 16000
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='int16',
                blocking=True
            )
            audio_data = sr.AudioData(recording.tobytes(), sample_rate, 2)
            return audio_data
        except Exception as e:
            print(f"sounddevice record error: {e}")
            return None

    def process_command(self, command_text):
        """Match command text to a known command and execute it."""
        if not command_text:
            return {"status": "no_input", "command": None, "action": None}

        matched_cmd = None
        for cmd_phrase, (action_type, action_value) in self.SUPPORTED_COMMANDS.items():
            if cmd_phrase in command_text:
                matched_cmd = (cmd_phrase, action_type, action_value)
                break

        self.last_command = command_text
        entry = {
            'command': command_text,
            'matched': matched_cmd[0] if matched_cmd else None,
            'timestamp': time.strftime('%H:%M:%S'),
            'status': 'executed' if matched_cmd else 'unknown'
        }
        self.command_history.append(entry)
        if len(self.command_history) > 20:
            self.command_history.pop(0)

        if matched_cmd:
            cmd_phrase, action_type, action_value = matched_cmd
            result = self._execute_action(action_type, action_value, cmd_phrase)
            return {"status": "executed", "command": command_text, "matched": cmd_phrase,
                    "action": result, "type": action_type}
        else:
            self.speak(f"Sorry, I don't understand the command: {command_text}")
            return {"status": "unknown", "command": command_text, "action": None}

    def _execute_action(self, action_type, action_value, cmd_phrase):
        """Execute detected command safely."""
        try:
            if action_type == "browser":
                webbrowser.open(action_value)
                self.speak(f"Opening {cmd_phrase}")
                return f"Opened browser: {action_value}"
            elif action_type == "app":
                if action_value == "ms-settings:":
                    os.startfile("ms-settings:")
                else:
                    subprocess.Popen([action_value], shell=True)
                self.speak(f"Opening {cmd_phrase}")
                return f"Opened application: {action_value}"
            elif action_type == "media":
                import pyautogui
                if action_value == "play":
                    pyautogui.press('playpause')
                    self.speak("Playing media")
                elif action_value == "pause":
                    pyautogui.press('playpause')
                    self.speak("Pausing media")
                return f"Media control: {action_value}"
            elif action_type == "system":
                import pyautogui
                if action_value == "volume_up":
                    for _ in range(5):
                        pyautogui.press('volumeup')
                    self.speak("Volume increased")
                elif action_value == "volume_down":
                    for _ in range(5):
                        pyautogui.press('volumedown')
                    self.speak("Volume decreased")
                elif action_value == "screenshot":
                    import time as t
                    ss = pyautogui.screenshot()
                    fname = f"screenshot_{int(t.time())}.png"
                    fpath = os.path.join(os.path.expanduser("~"), "Desktop", fname)
                    ss.save(fpath)
                    self.speak(f"Screenshot saved as {fname}")
                    return f"Screenshot saved: {fpath}"
                elif action_value == "stop":
                    self.stop()
                    self.speak("Stopping assistant")
                return f"System action: {action_value}"
            elif action_type == "scroll":
                import pyautogui
                if action_value == "up":
                    pyautogui.scroll(5)
                    self.speak("Scrolling up")
                elif action_value == "down":
                    pyautogui.scroll(-5)
                    self.speak("Scrolling down")
                return f"Scroll: {action_value}"
        except Exception as e:
            return f"Error executing action: {e}"

    def continuous_listen(self, callback=None):
        """Continuously listen and process commands in a background thread."""
        self.is_running   = True
        self.is_listening = True
        while self.is_running and self.is_listening:
            text = self.listen_once()
            if text:
                result = self.process_command(text)
                if callback:
                    callback(result)
            time.sleep(0.1)

    def start(self, callback=None):
        if not self.is_running:
            self.listen_thread = threading.Thread(
                target=self.continuous_listen, args=(callback,), daemon=True)
            self.listen_thread.start()
            self.speak("Voice assistant started. Listening for commands.")

    def stop(self):
        self.is_running   = False
        self.is_listening = False

    def toggle_listen(self):
        self.is_listening = not self.is_listening
        return self.is_listening

    def get_history(self, n=10):
        return self.command_history[-n:]

    def update_settings(self, language=None, voice_feedback=None):
        if language:
            self.language = language
        if voice_feedback is not None:
            self.voice_feedback = voice_feedback

    def get_mic_status(self):
        """Return microphone availability info."""
        return {
            'available': self.mic_available,
            'type': self.mic_type
        }
