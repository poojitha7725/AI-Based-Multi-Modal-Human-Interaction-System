"""
Presentation Control Module
Controls PowerPoint presentations via keyboard shortcuts.
"""
import pyautogui
import time

class PresentationController:
    def __init__(self):
        self.is_presenting = False
        self.slide_number = 0
        self.action_history = []

    def start_presentation(self):
        """Start slideshow (F5 in PowerPoint)."""
        pyautogui.press('f5')
        self.is_presenting = True
        self.slide_number = 1
        self._log("start_presentation", "Presentation started (F5)")
        return {"status": "success", "action": "start_presentation"}

    def next_slide(self):
        """Go to next slide."""
        pyautogui.press('right')
        self.slide_number += 1
        self._log("next_slide", f"Moved to slide {self.slide_number}")
        return {"status": "success", "action": "next_slide", "slide": self.slide_number}

    def prev_slide(self):
        """Go to previous slide."""
        pyautogui.press('left')
        self.slide_number = max(1, self.slide_number - 1)
        self._log("prev_slide", f"Moved to slide {self.slide_number}")
        return {"status": "success", "action": "prev_slide", "slide": self.slide_number}

    def end_presentation(self):
        """End presentation (Escape)."""
        pyautogui.press('escape')
        self.is_presenting = False
        self._log("end_presentation", "Presentation ended (Esc)")
        return {"status": "success", "action": "end_presentation"}

    def blank_screen(self):
        """Toggle blank screen (B key in PowerPoint)."""
        pyautogui.press('b')
        self._log("blank_screen", "Screen blanked/restored")
        return {"status": "success", "action": "blank_screen"}

    def execute_gesture_action(self, gesture):
        """Map gestures to presentation actions."""
        mapping = {
            "One Finger": self.next_slide,
            "Victory Sign": self.prev_slide,
            "Thumbs Up": self.start_presentation,
            "Closed Fist": self.end_presentation,
            "Open Palm": self.blank_screen,
        }
        action_fn = mapping.get(gesture)
        if action_fn:
            return action_fn()
        return {"status": "no_action", "gesture": gesture}

    def _log(self, action, message):
        self.action_history.append({
            "action": action,
            "message": message,
            "timestamp": time.strftime('%H:%M:%S')
        })
        if len(self.action_history) > 20:
            self.action_history.pop(0)

    def get_history(self, n=10):
        return self.action_history[-n:]
