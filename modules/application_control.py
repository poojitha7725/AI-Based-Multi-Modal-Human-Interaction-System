"""
Application Control Module
Safely opens and controls supported applications via voice/gesture commands.
"""
import subprocess
import webbrowser
import os
import pyautogui
import time

class ApplicationController:
    SAFE_APPS = {
        "calculator": "calc",
        "notepad": "notepad",
        "file explorer": "explorer",
        "paint": "mspaint",
        "word pad": "wordpad",
        "task manager": "taskmgr",
        "settings": "ms-settings:",
        "control panel": "control",
    }

    BROWSER_URLS = {
        "youtube": "https://www.youtube.com",
        "google": "https://www.google.com",
        "gmail": "https://mail.google.com",
        "github": "https://www.github.com",
        "wikipedia": "https://www.wikipedia.org",
        "stackoverflow": "https://stackoverflow.com",
    }

    def __init__(self):
        self.action_history = []

    def open_application(self, app_name):
        app_name = app_name.lower().strip()
        result = {"app": app_name, "status": "error", "message": ""}

        if app_name in self.SAFE_APPS:
            try:
                cmd = self.SAFE_APPS[app_name]
                if cmd.startswith("ms-"):
                    os.startfile(cmd)
                else:
                    subprocess.Popen([cmd], shell=True)
                result.update({"status": "success", "message": f"Opened {app_name}"})
            except Exception as e:
                result["message"] = str(e)
        elif app_name in self.BROWSER_URLS:
            webbrowser.open(self.BROWSER_URLS[app_name])
            result.update({"status": "success", "message": f"Opened {app_name} in browser"})
        else:
            result["message"] = f"Application '{app_name}' is not in the allowed list."

        self._log(result)
        return result

    def browser_control(self, action):
        """Control browser with keyboard shortcuts."""
        controls = {
            "new_tab": ['ctrl', 't'],
            "close_tab": ['ctrl', 'w'],
            "refresh": ['f5'],
            "back": ['alt', 'left'],
            "forward": ['alt', 'right'],
            "zoom_in": ['ctrl', '+'],
            "zoom_out": ['ctrl', '-'],
            "fullscreen": ['f11'],
        }
        if action in controls:
            keys = controls[action]
            if len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)
            result = {"action": action, "status": "success", "message": f"Browser: {action}"}
        else:
            result = {"action": action, "status": "error", "message": f"Unknown browser action: {action}"}
        self._log(result)
        return result

    def media_control(self, action):
        """Control media playback."""
        controls = {
            "play_pause": 'playpause',
            "next_track": 'nexttrack',
            "prev_track": 'prevtrack',
            "volume_up": 'volumeup',
            "volume_down": 'volumedown',
            "mute": 'volumemute',
        }
        if action in controls:
            pyautogui.press(controls[action])
            result = {"action": action, "status": "success", "message": f"Media: {action}"}
        else:
            result = {"action": action, "status": "error", "message": f"Unknown media action: {action}"}
        self._log(result)
        return result

    def take_screenshot(self, save_path=None):
        import datetime
        try:
            screenshot = pyautogui.screenshot()
            if not save_path:
                desktop = os.path.join(os.path.expanduser("~"), "Desktop")
                fname = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                save_path = os.path.join(desktop, fname)
            screenshot.save(save_path)
            result = {"action": "screenshot", "status": "success",
                      "message": f"Saved: {save_path}", "path": save_path}
        except Exception as e:
            result = {"action": "screenshot", "status": "error", "message": str(e)}
        self._log(result)
        return result

    def _log(self, result):
        entry = dict(result)
        entry['timestamp'] = time.strftime('%H:%M:%S')
        self.action_history.append(entry)
        if len(self.action_history) > 30:
            self.action_history.pop(0)

    def get_history(self, n=10):
        return self.action_history[-n:]

    def get_safe_apps(self):
        return list(self.SAFE_APPS.keys()) + list(self.BROWSER_URLS.keys())
