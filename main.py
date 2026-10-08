#!/usr/bin/env python3
"""
GestureOS: AI Hand Gesture Computer Controller
Main application loop capturing video, running landmark estimation, and dispatching OS events.
"""

import sys
import time
import cv2
import pyautogui
from gesture_controller import GestureController

# Configure PyAutoGUI fail-safe settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.001

def main():
    print("=" * 60)
    print(" GestureOS: AI Webcam Hand Gesture Computer Control")
    print(" Starting camera capture...")
    print(" Press 'q' to exit | 's' to toggle skeleton | 'm' to mirror")
    print("=" * 60)

    # Initialize video capture (camera index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam. Please check camera permissions.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    screen_w, screen_h = pyautogui.size()
    controller = GestureController(screen_width=screen_w, screen_height=screen_h)

    show_skeleton = True
    mirror = True
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Blank camera frame received. Retrying...")
            time.sleep(0.05)
            continue

        if mirror:
            frame = cv2.flip(frame, 1)

        # Process frame through computer vision pipeline
        annotated_frame, telemetry = controller.process_frame(frame, draw_overlay=show_skeleton)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / max(0.001, curr_time - prev_time)
        prev_time = curr_time

        # Render on-screen HUD
        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)} | GESTURE: {telemetry['gesture']}",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("GestureOS - Vision Telemetry", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_skeleton = not show_skeleton
        elif key == ord('m'):
            mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Controller terminated safely.")

if __name__ == "__main__":
    main()
