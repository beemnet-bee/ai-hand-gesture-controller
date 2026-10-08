"""
GestureController Module
Extracts 21 hand landmarks, executes kinematic vector checks, smooths coordinates, and dispatches actions.
"""

import math
import time
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

class GestureController:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_w = screen_width
        self.screen_h = screen_height

        # Initialize MediaPipe Hands solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Smoothing & Debounce State
        self.prev_x, self.prev_y = self.screen_w // 2, self.screen_h // 2
        self.smooth_alpha = 0.28  # EMA coefficient (lower = smoother, higher = faster)
        self.deadband = 4.0        # Pixel distance threshold to eliminate jitter

        self.is_pinching = False
        self.pinch_start_time = 0.0
        self.last_scroll_y = 0.0
        self.scroll_cooldown = 0.0

    def process_frame(self, frame: np.ndarray, draw_overlay: bool = True):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        telemetry = {
            "gesture": "NONE",
            "pinch_ratio": 1.0,
            "cursor_x": self.prev_x,
            "cursor_y": self.prev_y,
            "hands_detected": 0
        }

        if not results.multi_hand_landmarks:
            return frame, telemetry

        hand_landmarks = results.multi_hand_landmarks[0]
        telemetry["hands_detected"] = 1

        # Landmark indices
        # 0: Wrist, 4: Thumb Tip, 8: Index Tip, 12: Middle Tip, 16: Ring Tip, 20: Pinky Tip
        lm = hand_landmarks.landmark
        wrist = np.array([lm[0].x * w, lm[0].y * h])
        thumb_tip = np.array([lm[4].x * w, lm[4].y * h])
        index_tip = np.array([lm[8].x * w, lm[8].y * h])
        index_pip = np.array([lm[6].x * w, lm[6].y * h])
        middle_tip = np.array([lm[12].x * w, lm[12].y * h])
        middle_pip = np.array([lm[10].x * w, lm[10].y * h])
        ring_tip = np.array([lm[16].x * w, lm[16].y * h])
        ring_pip = np.array([lm[14].x * w, lm[14].y * h])
        pinky_tip = np.array([lm[20].x * w, lm[20].y * h])
        pinky_pip = np.array([lm[18].x * w, lm[18].y * h])

        # Hand scale: Euclidean distance from wrist to middle knuckle
        hand_scale = max(20.0, np.linalg.norm(middle_pip - wrist))

        # Check finger extension state
        is_index_ext = np.linalg.norm(index_tip - wrist) > np.linalg.norm(index_pip - wrist) * 1.15
        is_middle_ext = np.linalg.norm(middle_tip - wrist) > np.linalg.norm(middle_pip - wrist) * 1.15
        is_ring_ext = np.linalg.norm(ring_tip - wrist) > np.linalg.norm(ring_pip - wrist) * 1.15
        is_pinky_ext = np.linalg.norm(pinky_tip - wrist) > np.linalg.norm(pinky_pip - wrist) * 1.15

        # Pinch detection (Thumb Tip to Index Tip)
        pinch_dist = np.linalg.norm(thumb_tip - index_tip)
        pinch_ratio = pinch_dist / hand_scale
        telemetry["pinch_ratio"] = round(float(pinch_ratio), 3)

        # Coordinate mapping with active margin
        margin_x, margin_y = int(w * 0.15), int(h * 0.15)
        clamped_x = np.clip(index_tip[0], margin_x, w - margin_x)
        clamped_y = np.clip(index_tip[1], margin_y, h - margin_y)
        norm_x = (clamped_x - margin_x) / (w - 2 * margin_x)
        norm_y = (clamped_y - margin_y) / (h - 2 * margin_y)

        target_screen_x = norm_x * self.screen_w
        target_screen_y = norm_y * self.screen_h

        # Exponential Moving Average smoothing
        smoothed_x = self.prev_x + (target_screen_x - self.prev_x) * self.smooth_alpha
        smoothed_y = self.prev_y + (target_screen_y - self.prev_y) * self.smooth_alpha

        # Deadband jitter rejection
        if np.hypot(smoothed_x - self.prev_x, smoothed_y - self.prev_y) > self.deadband:
            self.prev_x, self.prev_y = smoothed_x, smoothed_y

        now = time.time()

        # 1. Pinch Click & Drag
        if pinch_ratio < 0.28:
            if not self.is_pinching:
                self.is_pinching = True
                self.pinch_start_time = now
                pyautogui.mouseDown()
                telemetry["gesture"] = "PINCH_CLICK"
            else:
                telemetry["gesture"] = "PINCH_DRAG"
            # Move while pinching (drag)
            pyautogui.moveTo(int(self.prev_x), int(self.prev_y))
        else:
            if self.is_pinching:
                self.is_pinching = False
                pyautogui.mouseUp()

            # 2. Two-Finger Scroll
            if is_index_ext and is_middle_ext and not is_ring_ext and not is_pinky_ext:
                curr_two_finger_y = (index_tip[1] + middle_tip[1]) / 2.0
                if self.last_scroll_y != 0.0:
                    delta_y = curr_two_finger_y - self.last_scroll_y
                    if abs(delta_y) > 4.0:
                        scroll_amount = int(-delta_y * 1.5)
                        pyautogui.scroll(scroll_amount)
                        telemetry["gesture"] = "TWO_FINGER_SCROLL"
                self.last_scroll_y = curr_two_finger_y
            else:
                self.last_scroll_y = 0.0

                # 3. Pointer Move (Index finger only)
                if is_index_ext and not is_middle_ext:
                    pyautogui.moveTo(int(self.prev_x), int(self.prev_y))
                    telemetry["gesture"] = "POINTER_MOVE"

                # 4. Fist Mute
                elif not is_index_ext and not is_middle_ext and not is_ring_ext and not is_pinky_ext:
                    telemetry["gesture"] = "FIST_MUTE"

        telemetry["cursor_x"] = int(self.prev_x)
        telemetry["cursor_y"] = int(self.prev_y)

        if draw_overlay:
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(255, 180, 0), thickness=2)
            )

        return frame, telemetry
