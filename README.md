# GestureOS: AI-Powered Webcam Hand Gesture Computer Control

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-06B6D4.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, low-latency computer vision interface that transforms standard webcam video feeds into real-time operating system control gestures. Move your mouse, click, drag, scroll long documents, and modulate system audio volume with touchless hand kinematics.

---

## 🚀 Key Features

- **Touchless Virtual Cursor**: Moves operating system mouse cursor via index fingertip kinematics with Exponential Moving Average (EMA) jitter suppression.
- **Micro-Pinch Left Click**: Index-to-thumb tip distance detection (<28% hand scale) triggers instant system clicks or hold-to-drag.
- **Two-Finger Inertial Scroll**: Natural two-finger extension mimics trackpad scrolling for documents and web pages.
- **Dial & Volume Control**: Tilt your hand or show thumbs up/down to adjust system master volume.
- **Zero Specialty Hardware**: Operates on any standard 720p/1080p RGB webcam at 30–60 FPS with lightweight CPU/GPU inference.
- **Cross-Platform**: Tested on Windows 11, macOS Sequoia/Sonoma, and Ubuntu Linux.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11 installed
- Standard integrated or USB webcam

### 2. Clone the Repository
```bash
git clone https://github.com/gesture-os/ai-hand-gesture-controller.git
cd ai-hand-gesture-controller
```

### 3. Create Virtual Environment
```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Running the Controller

```bash
python main.py
```

### Keyboard Controls
- `q`: Quit application safely
- `s`: Toggle visual skeleton overlay
- `c`: Trigger center calibration
- `m`: Toggle camera mirroring

---

## 📐 AI & Computer Vision Architecture

```
Webcam Feed (RGB) 
      │
      ▼
1. Palm Detection Model (SSD Anchor Bounding Box)
      │
      ▼
2. 21 3D Skeletal Landmark Regression (MediaPipe)
      │
      ▼
3. Biomechanical Vector Normalization (Scale Invariant)
      │
      ▼
4. Temporal Exponential Moving Average (EMA) Filter
      │
      ▼
5. State Machine & Debounce Classifier
      │
      ▼
6. OS Event Dispatcher (PyAutoGUI Native Mouse/Key Injection)
```

---

## 📄 License
MIT License - open for personal, academic, and commercial research.
