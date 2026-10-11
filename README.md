# Hand Gesture Recognition Application

A professional, real-time Python application that uses **OpenCV** and **MediaPipe** to detect a single hand from your webcam and recognize specific gestures. It overlays hand landmarks, dynamic HUD bounding cards, gesture names, and frame-rate (FPS) calculations in real time.

---

## Features

- **Real-Time Landmark Detection**: Tracks 21 3D hand joint landmarks using Google MediaPipe.
- **Gesture Classification**: Uses mathematical/geometric heuristics based on finger extension states and relative landmark distances:
  - **Open Palm**: All fingers and thumb fully extended.
  - **Fist**: All fingers and thumb folded inward.
  - **Thumbs Up**: Thumb extended upward while all other fingers are folded.
  - **Peace Sign**: Index and middle fingers extended; thumb, ring, and pinky folded.
  - **OK Sign**: Thumb and index finger tips touching in a circle, other fingers extended.
- **Sleek HUD Overlay**: Shows a clean, semi-transparent card with active status information and smoothed real-time FPS.
- **Robust Error Handling**: Handles camera input errors and provides proper resource cleanups.

---

## Directory Structure

```
hand-gesture/
├── main.py                    # Application Entry Point
├── requirements.txt           # Python Project Dependencies
├── README.md                  # Project Documentation
├── src/
│   ├── __init__.py            # Src Package Init
│   ├── camera.py              # OpenCV Video Stream, Overlay, & Loop Management
│   └── gesture_recognizer.py  # MediaPipe Setup & Heuristic Gesture Logic
└── tests/
    ├── __init__.py            # Tests Package Init
    └── test_gesture_recognizer.py # Unit tests for Gesture Logic with Mock Data
```

---

## Installation

### Prerequisites
- Python 3.8 or higher.
- A functional webcam.

### Steps
1. **Clone or Navigate to the Directory**:
   ```bash
   cd hand-gesture
   ```

2. **Install Dependencies**:
   Install the required libraries (`opencv-python` and `mediapipe`) using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Start the application with default settings:
```bash
python main.py
```

### Custom Configurations
You can specify a different camera index or frame size using CLI arguments:
```bash
python main.py --camera 0 --width 1280 --height 720
```

### Controls
- Press **`q`** on your keyboard while focusing on the video window to cleanly exit the application.
- Pressing **`Ctrl + C`** in the terminal will also raise a `KeyboardInterrupt` and shut down the camera stream gracefully.

---

## Running Tests
To run the automated gesture classifier unit tests (which run offline using mock hand landmark inputs):
```bash
python -m unittest discover -s tests
```

---

## How It Works (Heuristic Logic)
1. **Scale Normalization**: To prevent errors due to the hand being closer or further from the camera, we calculate `hand_scale` as the Euclidean distance between the wrist (landmark `0`) and middle finger MCP joint (landmark `9`). All other key distances are normalized by dividing by `hand_scale`.
2. **Finger Extension Check**:
   - For index, middle, ring, and pinky fingers, we check if the y-coordinate of the TIP is smaller than the y-coordinate of the PIP joint. Since image coordinates go from `0` (top) to `1` (bottom), a smaller y value means the finger is extended upwards.
3. **Thumb Extension Check**:
   - We measure the distance between the thumb tip (landmark `4`) and index finger MCP joint (landmark `5`). If the normalized distance exceeds `0.4`, the thumb is considered extended.
4. **OK Sign Proximity**:
   - Calculated by measuring the distance between thumb tip (landmark `4`) and index finger tip (landmark `8`). If they are closer than `0.25` normalized distance while middle, ring, and pinky remain extended, the "OK" gesture is triggered.
