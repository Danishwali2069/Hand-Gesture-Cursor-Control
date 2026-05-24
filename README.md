# Touchless System Control (Windows + Android)

This project contains a real touchless control foundation for:

- Windows (Python + OpenCV + MediaPipe + OS cursor APIs)
- Android (Kotlin + CameraX + AccessibilityService + global overlay pointer)

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

### 2) Android system-wide architecture

In `android/`:

- MainActivity with one ON/OFF toggle and small status text
- Accessibility service (`GestureAccessibilityService`) required for global interaction
- Overlay pointer (`OverlayPointerController`) shown above apps
- CameraX pipeline (`GestureCameraController`) prepared for MediaPipe gesture feed
- Gesture mapper (`GestureInterpreter`) for move/pinch/hold/fist mapping

## Windows setup

1. Create Python venv and install deps:
   - `pip install -r windows/requirements.txt`
2. Run:
   - `python windows/gesture_control.py`
3. Press ON in app.

## Android setup

1. Open `android/` in Android Studio.
2. Sync Gradle and install on device.
3. Grant Camera + Overlay permissions.
4. Enable Accessibility Service for this app.
5. Press ON in app.

## Important platform note

On Android, true hardware-level cursor injection is restricted for non-system/root apps.
This implementation uses the supported production path:

- floating global overlay pointer + Accessibility gesture dispatch for global interaction.

That is the correct non-root system-wide approach for touchless control.
