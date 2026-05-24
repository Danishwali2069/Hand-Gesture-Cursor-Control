# Touchless System Control (Windows)

This project contains a real touchless control foundation for:

- Windows (Python + OpenCV + MediaPipe + OS cursor APIs)

## What is implemented

### 1) Windows real system cursor control

In `windows/gesture_control.py`:

- Single ON/OFF UI button only
- ON:
  - starts camera
  - starts MediaPipe hand tracking
  - shows global always-on-top floating pointer
  - moves the real OS cursor
- OFF:
  - stops camera loop
  - removes floating pointer
- Gesture mapping:
  - index finger move -> cursor move
  - thumb + middle pinch -> action click
  - double thumb + middle pinch -> open file/app (double click)
  - thumb + index pinch + move -> drag
  - release thumb + index -> drop

## Windows setup

1. Create Python venv and install deps:
   - `pip install -r windows/requirements.txt`
2. Run:
   - `python windows/gesture_control.py`
3. Press ON in app.

That is the correct non-root system-wide approach for touchless control.
