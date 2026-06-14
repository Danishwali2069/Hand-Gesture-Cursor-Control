import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


MP_BACKEND = "mediapipe.solutions"
TASKS_VISION = None
try:
    MP_SOLUTIONS = mp.solutions
except AttributeError:
    MP_SOLUTIONS = None
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        TASKS_VISION = (mp_python, mp_vision)
        MP_BACKEND = "mediapipe.tasks"
    except Exception:
        TASKS_VISION = None
        MP_BACKEND = "unavailable"

pyautogui.FAILSAFE = False


def print_startup_diagnostics() -> None:
    mp_version = getattr(mp, "__version__", "unknown")
    hands_ok = (MP_SOLUTIONS is not None and hasattr(MP_SOLUTIONS, "hands")) or TASKS_VISION is not None
    print("=== Touchless Startup Diagnostics ===")
    print(f"MediaPipe version : {mp_version}")
    print(f"MediaPipe backend : {MP_BACKEND}")
    print(f"Hands module OK   : {hands_ok}")
    print("=====================================")
    if not hands_ok:
        raise RuntimeError("MediaPipe Hands module is not available in this installation.")



class MpTasksHandTracker:
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )

    def __init__(self):
        mp_python, mp_vision = TASKS_VISION
        model_path = self._ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def _ensure_model(self) -> Path:
        model_dir = Path(__file__).parent / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "hand_landmarker.task"
        if not model_path.exists():
            print("Downloading MediaPipe hand model...")
            urlretrieve(self.MODEL_URL, model_path)
        return model_path

    def process(self, rgb_image: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        self.timestamp_ms += 33
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        if result.hand_landmarks:
            return type(
                "LegacyResult", (),
                {"multi_hand_landmarks": [
                    type("LandmarkObj", (), {"landmark": result.hand_landmarks[0]})()
                ]},
            )()
        return type("LegacyResult", (), {"multi_hand_landmarks": None})()



@dataclass
class GestureState:
    action_pinch_active: bool = False
    drag_pinch_active: bool = False
    close_pinch_active: bool = False        # ring + thumb
    dragging: bool = False
    drag_pinch_start: Optional[Tuple[int, int]] = None
    pending_single_click_time: Optional[float] = None
    pending_single_click_pos: Optional[Tuple[int, int]] = None
    action_pinch_ratio_ema: float = 1.0
    drag_pinch_ratio_ema: float = 1.0
    close_pinch_ratio_ema: float = 1.0     # ring + thumb EMA
    drag_cursor_anchor_offset: Optional[Tuple[float, float]] = None
    action_cursor_anchor_offset: Optional[Tuple[float, float]] = None


@dataclass
class SessionStats:
    clicks: int = 0
    double_clicks: int = 0
    drags: int = 0
    closes: int = 0
    start_time: float = field(default_factory=time.time)

    def uptime_str(self) -> str:
        secs = int(time.time() - self.start_time)
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"



class FloatingCursor:
    SIZE = 20

    def __init__(self):
        self.root = tk.Toplevel()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", "black")
        self.root.configure(bg="black")
        s = self.SIZE
        self.canvas = tk.Canvas(self.root, width=s, height=s, bg="black", highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_oval(1, 1, s - 1, s - 1, outline="#00FFB0", width=2, fill="")
        self.canvas.create_oval(7, 7, s - 7, s - 7, fill="#00FFB0", outline="")
        self.root.geometry(f"{s}x{s}+200+200")

    def move(self, x: int, y: int) -> None:
        self.root.geometry(f"{self.SIZE}x{self.SIZE}+{x - self.SIZE // 2}+{y - self.SIZE // 2}")

    def destroy(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass



class GestureController:
    ACTION_PINCH_ON_RATIO  = 0.50
    ACTION_PINCH_OFF_RATIO = 0.68
    DRAG_PINCH_ON_RATIO    = 0.38
    DRAG_PINCH_OFF_RATIO   = 0.52
    CLOSE_PINCH_ON_RATIO   = 0.42          # ring + thumb — slightly looser than drag
    CLOSE_PINCH_OFF_RATIO  = 0.60
    CLOSE_HOLD_SEC         = 0.40          # must hold pinch this long to fire close
    DRAG_START_PIXELS      = 20
    DOUBLE_PINCH_WINDOW_SEC  = 0.58
    CLICK_COMMIT_DELAY_SEC   = 0.62
    NO_HAND_TIMEOUT_SEC      = 0.45
    TARGET_FPS               = 30.0
    LANDMARK_MEDIAN_WINDOW   = 5
    RELATIVE_GAIN_X          = 2.2
    RELATIVE_GAIN_Y          = 2.0
    RELATIVE_DEADZONE        = 0.006
    MAX_STEP_PIXELS          = 42

    def __init__(self):
        self.running = False
        self.capture_thread = None
        self.cap = None
        self.state = GestureState()
        self.stats = SessionStats()
        self.screen_w, self.screen_h = pyautogui.size()
        self.smooth_x = self.screen_w // 2
        self.smooth_y = self.screen_h // 2
        self.raw_anchor_x = None
        self.raw_anchor_y = None
        self.anchor_x_hist: deque = deque(maxlen=self.LANDMARK_MEDIAN_WINDOW)
        self.anchor_y_hist: deque = deque(maxlen=self.LANDMARK_MEDIAN_WINDOW)
        self.last_hand_seen_ts = time.time()
        self.hand_visible = False
        self.live_action_ratio: float = 1.0
        self.live_drag_ratio: float = 1.0
        self.live_close_ratio: float = 1.0
        self._close_pinch_start_ts: Optional[float] = None   # when ring+thumb pinch began

        if MP_SOLUTIONS is not None and hasattr(MP_SOLUTIONS, "hands"):
            self.mp_hands = MP_SOLUTIONS.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        elif TASKS_VISION is not None:
            self.hands = MpTasksHandTracker()
        else:
            raise RuntimeError("No supported MediaPipe hand tracking backend found.")

    def adjust_gain(self, val: float) -> None:
        v = float(np.clip(val, 0.8, 4.0))
        self.RELATIVE_GAIN_X = v
        self.RELATIVE_GAIN_Y = v

    def adjust_deadzone(self, val: float) -> None:
        self.RELATIVE_DEADZONE = float(np.clip(val, 0.001, 0.03))

    def adjust_max_step(self, val: int) -> None:
        self.MAX_STEP_PIXELS = int(np.clip(val, 14, 120))

    def reset_tracking_state(self) -> None:
        self.raw_anchor_x = None
        self.raw_anchor_y = None
        self.anchor_x_hist.clear()
        self.anchor_y_hist.clear()
        s = self.state
        s.pending_single_click_time = None
        s.pending_single_click_pos = None
        s.drag_cursor_anchor_offset = None
        s.action_cursor_anchor_offset = None

    def start(self, app: "ControlApp"):
        if self.running:
            return
        self.running = True
        self.state = GestureState()
        self.stats = SessionStats()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.hand_visible = False
        app.hide_cursor_overlay(set_off_status=False)
        self.capture_thread = threading.Thread(target=self._loop, args=(app,), daemon=True)
        self.capture_thread.start()

    def stop(self, app: "ControlApp"):
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.5)
        if self.cap:
            self.cap.release()
            self.cap = None
        app.hide_cursor_overlay()

    def _loop(self, app: "ControlApp"):
        while self.running and self.cap:
            try:
                ok, frame = self.cap.read()
                if not ok:
                    app.queue_status("Camera frame lost, retrying...", "warn")
                    time.sleep(0.02)
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb)

                if result.multi_hand_landmarks:
                    if not self.hand_visible:
                        self.hand_visible = True
                        self.reset_tracking_state()
                        app.show_cursor_overlay()
                    self.last_hand_seen_ts = time.time()
                    self._handle_hand(result.multi_hand_landmarks[0], app)
                else:
                    self._flush_pending_single_click(app)
                    if (time.time() - self.last_hand_seen_ts) > self.NO_HAND_TIMEOUT_SEC:
                        if self.hand_visible:
                            self.hand_visible = False
                            self.reset_tracking_state()
                            app.hide_cursor_overlay(set_off_status=False)
                        app.queue_status("No hand detected", "warn")
                    self.live_action_ratio = 1.0
                    self.live_drag_ratio = 1.0
                    self.live_close_ratio = 1.0

                app.queue_pinch_update(self.live_action_ratio, self.live_drag_ratio, self.live_close_ratio)
                time.sleep(0.001)
            except Exception as exc:
                app.queue_status(f"Recovering: {str(exc)[:30]}", "error")
                time.sleep(0.03)

    def _lm_to_screen(self, lm_x: float, lm_y: float) -> Tuple[int, int]:
        if self.raw_anchor_x is None or self.raw_anchor_y is None:
            self.raw_anchor_x = lm_x
            self.raw_anchor_y = lm_y
            self.smooth_x = int(np.clip(lm_x * self.screen_w, 0, self.screen_w - 1))
            self.smooth_y = int(np.clip(lm_y * self.screen_h, 0, self.screen_h - 1))
            return self.smooth_x, self.smooth_y

        dx_norm = lm_x - self.raw_anchor_x
        dy_norm = lm_y - self.raw_anchor_y
        self.raw_anchor_x = lm_x
        self.raw_anchor_y = lm_y

        if abs(dx_norm) < self.RELATIVE_DEADZONE:
            dx_norm = 0.0
        if abs(dy_norm) < self.RELATIVE_DEADZONE:
            dy_norm = 0.0

        step_x = int(np.clip(dx_norm * self.screen_w * self.RELATIVE_GAIN_X, -self.MAX_STEP_PIXELS, self.MAX_STEP_PIXELS))
        step_y = int(np.clip(dy_norm * self.screen_h * self.RELATIVE_GAIN_Y, -self.MAX_STEP_PIXELS, self.MAX_STEP_PIXELS))

        self.smooth_x = int(np.clip(self.smooth_x + step_x, 0, self.screen_w - 1))
        self.smooth_y = int(np.clip(self.smooth_y + step_y, 0, self.screen_h - 1))
        return self.smooth_x, self.smooth_y

    def _flush_pending_single_click(self, app: "ControlApp") -> None:
        t0 = self.state.pending_single_click_time
        if t0 is None:
            return
        if (time.time() - t0) >= self.CLICK_COMMIT_DELAY_SEC:
            x, y = self.state.pending_single_click_pos
            pyautogui.click(x=x, y=y, _pause=False)
            self.stats.clicks += 1
            app.queue_status("Click", "ok")
            app.queue_stats(self.stats)
            self.state.pending_single_click_time = None
            self.state.pending_single_click_pos = None

    def _handle_hand(self, hand, app: "ControlApp"):
        lm = hand.landmark
        index_tip  = lm[8]
        index_pip  = lm[6]   # FIX: original used undefined `index_knuckle`
        middle_tip = lm[12]
        ring_tip   = lm[16]  # NEW: ring finger tip for close gesture
        thumb_tip  = lm[4]
        index_mcp  = lm[5]
        middle_mcp = lm[9]
        wrist      = lm[0]

        palm_x = 0.45 * index_mcp.x + 0.45 * middle_mcp.x + 0.10 * wrist.x
        palm_y = 0.45 * index_mcp.y + 0.45 * middle_mcp.y + 0.10 * wrist.y
        forward_x = (index_tip.x - index_mcp.x) * 0.55
        forward_y = (index_tip.y - index_mcp.y) * 0.55
        anchor_x = palm_x + forward_x
        anchor_y = palm_y + forward_y
        self.anchor_x_hist.append(anchor_x)
        self.anchor_y_hist.append(anchor_y)
        anchor_x = float(np.median(self.anchor_x_hist))
        anchor_y = float(np.median(self.anchor_y_hist))

        palm_ref = max(np.hypot(lm[5].x - lm[17].x, lm[5].y - lm[17].y), 1e-5)
        action_pinch_dist = np.hypot(middle_tip.x - thumb_tip.x, middle_tip.y - thumb_tip.y)
        drag_pinch_dist   = np.hypot(index_tip.x  - thumb_tip.x, index_tip.y  - thumb_tip.y)
        close_pinch_dist  = np.hypot(ring_tip.x   - thumb_tip.x, ring_tip.y   - thumb_tip.y)
        action_ratio = action_pinch_dist / palm_ref
        drag_ratio   = drag_pinch_dist   / palm_ref
        close_ratio  = close_pinch_dist  / palm_ref

        self.state.action_pinch_ratio_ema = 0.65 * self.state.action_pinch_ratio_ema + 0.35 * action_ratio
        self.state.drag_pinch_ratio_ema   = 0.65 * self.state.drag_pinch_ratio_ema   + 0.35 * drag_ratio
        self.state.close_pinch_ratio_ema  = 0.65 * self.state.close_pinch_ratio_ema  + 0.35 * close_ratio
        self.live_action_ratio = float(self.state.action_pinch_ratio_ema)
        self.live_drag_ratio   = float(self.state.drag_pinch_ratio_ema)
        self.live_close_ratio  = float(self.state.close_pinch_ratio_ema)

        if self.state.action_pinch_active:
            action_pinch = self.state.action_pinch_ratio_ema < self.ACTION_PINCH_OFF_RATIO or action_pinch_dist < 0.080
        else:
            action_pinch = self.state.action_pinch_ratio_ema < self.ACTION_PINCH_ON_RATIO  or action_pinch_dist < 0.060

        if self.state.drag_pinch_active:
            drag_pinch = self.state.drag_pinch_ratio_ema < self.DRAG_PINCH_OFF_RATIO
        else:
            drag_pinch = self.state.drag_pinch_ratio_ema < self.DRAG_PINCH_ON_RATIO

        if self.state.close_pinch_active:
            close_pinch = self.state.close_pinch_ratio_ema < self.CLOSE_PINCH_OFF_RATIO
        else:
            close_pinch = self.state.close_pinch_ratio_ema < self.CLOSE_PINCH_ON_RATIO

        cursor_x, cursor_y = self._lm_to_screen(anchor_x, anchor_y)

        if drag_pinch and self.state.drag_cursor_anchor_offset is not None:
            ox, oy = self.state.drag_cursor_anchor_offset
            lock_x = int(np.clip(index_tip.x * self.screen_w + ox, 0, self.screen_w - 1))
            lock_y = int(np.clip(index_tip.y * self.screen_h + oy, 0, self.screen_h - 1))
            cursor_x = int(cursor_x * 0.60 + lock_x * 0.40)
            cursor_y = int(cursor_y * 0.60 + lock_y * 0.40)
        elif action_pinch and self.state.action_cursor_anchor_offset is not None:
            ox, oy = self.state.action_cursor_anchor_offset
            lock_x = int(np.clip(middle_tip.x * self.screen_w + ox, 0, self.screen_w - 1))
            lock_y = int(np.clip(middle_tip.y * self.screen_h + oy, 0, self.screen_h - 1))
            cursor_x = int(cursor_x * 0.65 + lock_x * 0.35)
            cursor_y = int(cursor_y * 0.65 + lock_y * 0.35)

        pyautogui.moveTo(cursor_x, cursor_y, duration=0, _pause=False)
        app.move_cursor_overlay(cursor_x, cursor_y)
        self._flush_pending_single_click(app)

        # Drag (thumb + index)
        if drag_pinch and not self.state.drag_pinch_active:
            self.state.drag_pinch_active = True
            self.state.drag_cursor_anchor_offset = (
                cursor_x - index_tip.x * self.screen_w,
                cursor_y - index_tip.y * self.screen_h,
            )
            self.state.drag_pinch_start = (cursor_x, cursor_y)
            app.queue_status("Index+Thumb -> drag ready", "ok")

        if self.state.drag_pinch_active and not self.state.dragging and self.state.drag_pinch_start:
            sx, sy = self.state.drag_pinch_start
            if abs(cursor_x - sx) > self.DRAG_START_PIXELS or abs(cursor_y - sy) > self.DRAG_START_PIXELS:
                pyautogui.mouseDown(_pause=False)
                self.state.dragging = True
                self.stats.drags += 1
                app.queue_status("Dragging...", "ok")
                app.queue_stats(self.stats)

        if not drag_pinch and self.state.drag_pinch_active:
            self.state.drag_pinch_active = False
            if self.state.dragging:
                pyautogui.mouseUp(_pause=False)
                app.queue_status("Drop", "ok")
            self.state.dragging = False
            self.state.drag_pinch_start = None
            self.state.drag_cursor_anchor_offset = None

        # Click / open (thumb + middle)
        if action_pinch and not self.state.action_pinch_active:
            self.state.action_pinch_active = True
            self.state.action_cursor_anchor_offset = (
                cursor_x - middle_tip.x * self.screen_w,
                cursor_y - middle_tip.y * self.screen_h,
            )
            app.queue_status("Middle+Thumb pinch", "ok")

        if not action_pinch and self.state.action_pinch_active:
            self.state.action_pinch_active = False
            now = time.time()
            t0  = self.state.pending_single_click_time
            pos = self.state.pending_single_click_pos
            if t0 and pos and (now - t0) < self.DOUBLE_PINCH_WINDOW_SEC:
                x, y = pos
                pyautogui.doubleClick(x=x, y=y, _pause=False)
                self.stats.double_clicks += 1
                app.queue_status("Double-click (open)", "ok")
                app.queue_stats(self.stats)
                self.state.pending_single_click_time = None
                self.state.pending_single_click_pos  = None
            else:
                self.state.pending_single_click_time = now
                self.state.pending_single_click_pos  = (cursor_x, cursor_y)
            self.state.action_cursor_anchor_offset = None

        if (
            not action_pinch
            and not drag_pinch
            and not close_pinch
            and index_tip.y < index_pip.y  # finger pointing up (FIX)
            and self.state.pending_single_click_time is None
        ):
            app.queue_status("Moving", "ok")

        if close_pinch and not self.state.close_pinch_active:
            self.state.close_pinch_active = True
            self._close_pinch_start_ts = time.time()
            app.queue_status("Ring+Thumb: hold to close...", "warn")

        if close_pinch and self.state.close_pinch_active and self._close_pinch_start_ts is not None:
            held = time.time() - self._close_pinch_start_ts
            if held >= self.CLOSE_HOLD_SEC:
                pyautogui.hotkey("alt", "f4", _pause=False)
                self.stats.closes += 1
                app.queue_status("Closed! (Alt+F4)", "ok")
                app.queue_stats(self.stats)
                # Reset so it doesn't fire repeatedly while still pinching
                self._close_pinch_start_ts = None

        if not close_pinch and self.state.close_pinch_active:
            was_pending = self._close_pinch_start_ts is not None
            self.state.close_pinch_active = False
            self._close_pinch_start_ts = None
            if was_pending:
                app.queue_status("Close cancelled", "warn")



C = {
    "bg":      "#0D0F14",
    "surface": "#161A24",
    "border":  "#252B3B",
    "accent":  "#00FFB0",
    "accent2": "#0077FF",
    "warn":    "#FFB800",
    "error":   "#FF4466",
    "text":    "#E8EAF0",
    "muted":   "#6B7299",
}

FT = ("Courier New", 12, "bold")   # title
FL = ("Courier New",  9)           # label
FS = ("Courier New", 10, "bold")   # status
FX = ("Courier New",  8)           # small



class ControlApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Touchless Control")
        self.root.configure(bg=C["bg"])
        self.root.resizable(False, False)

        self.controller = GestureController()
        self.floating_cursor: Optional[FloatingCursor] = None
        self.enabled = False
        self._ui_queue: list = []

        self._build_ui()
        self._bind_keys()
        self._tick()


    def _build_ui(self):
        PAD = dict(padx=14, pady=5)

        # Title
        tf = tk.Frame(self.root, bg=C["surface"], pady=10)
        tf.pack(fill="x")
        tk.Label(tf, text="* TOUCHLESS CONTROL *", font=FT,
                 fg=C["accent"], bg=C["surface"]).pack()
        tk.Label(tf, text="Hand gesture cursor  |  Windows", font=FX,
                 fg=C["muted"], bg=C["surface"]).pack()

        # Status row
        sf = tk.Frame(self.root, bg=C["bg"], pady=6)
        sf.pack(fill="x", **PAD)
        self.hand_dot = tk.Label(sf, text="●", font=("Courier New", 18),
                                  fg=C["muted"], bg=C["bg"])
        self.hand_dot.pack(side="left")
        self.status_label = tk.Label(sf, text="OFFLINE", font=FS,
                                      fg=C["muted"], bg=C["bg"], anchor="w")
        self.status_label.pack(side="left", padx=8)

        # Toggle button
        bf = tk.Frame(self.root, bg=C["bg"])
        bf.pack(pady=8)
        self.toggle_btn = tk.Button(
            bf, text="[  START  ]", font=("Courier New", 14, "bold"), width=14,
            bg=C["surface"], fg=C["accent"],
            activebackground=C["accent"], activeforeground=C["bg"],
            relief="flat", bd=0, cursor="hand2",
            highlightbackground=C["accent"], highlightthickness=1,
            command=self.toggle,
        )
        self.toggle_btn.pack()

        # Separator
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x", padx=14, pady=6)

        # Gesture meters
        mf = tk.Frame(self.root, bg=C["bg"])
        mf.pack(fill="x", **PAD)
        tk.Label(mf, text="PINCH STRENGTH", font=FL, fg=C["muted"], bg=C["bg"]).pack(anchor="w")
        self._make_meter(mf, "CLICK  mid+thumb", "action", C["accent"])
        self._make_meter(mf, "DRAG   idx+thumb", "drag",   C["accent2"])
        self._make_meter(mf, "CLOSE  rng+thumb", "close",  C["error"])

        # Separator
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x", padx=14, pady=6)

        # Sliders
        sliders = tk.Frame(self.root, bg=C["bg"])
        sliders.pack(fill="x", **PAD)
        tk.Label(sliders, text="TUNING", font=FL, fg=C["muted"], bg=C["bg"]).pack(anchor="w")

        self._gain_var = tk.DoubleVar(value=self.controller.RELATIVE_GAIN_X)
        self._dz_var   = tk.DoubleVar(value=self.controller.RELATIVE_DEADZONE)
        self._step_var = tk.IntVar(value=self.controller.MAX_STEP_PIXELS)

        self._make_slider(sliders, "Sensitivity", self._gain_var, 0.8, 4.0, self._on_gain)
        self._make_slider(sliders, "Deadzone",    self._dz_var, 0.001, 0.03, self._on_dz, res=0.001)
        self._make_slider(sliders, "Speed cap",   self._step_var, 14, 120, self._on_step)

        # Separator
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x", padx=14, pady=6)

        # Session stats — counters row + uptime row
        stats_outer = tk.Frame(self.root, bg=C["bg"])
        stats_outer.pack(fill="x", padx=14, pady=(0, 12))
        tk.Label(stats_outer, text="SESSION", font=FL, fg=C["muted"], bg=C["bg"]).pack(anchor="w")
        # Row 1: four event counters
        stat_row = tk.Frame(stats_outer, bg=C["bg"])
        stat_row.pack(fill="x")
        self._s_click  = self._make_stat(stat_row, "Clicks",  "0")
        self._s_open   = self._make_stat(stat_row, "Opens",   "0")
        self._s_drag   = self._make_stat(stat_row, "Drags",   "0")
        self._s_close  = self._make_stat(stat_row, "Closes",  "0")
        # Row 2: uptime spanning full width
        uptime_row = tk.Frame(stats_outer, bg=C["bg"])
        uptime_row.pack(fill="x", pady=(2, 0))
        uptime_f = tk.Frame(uptime_row, bg=C["surface"], padx=6, pady=4)
        uptime_f.pack(fill="x", padx=2)
        tk.Label(uptime_f, text="Uptime", font=FX, fg=C["muted"], bg=C["surface"]).pack(side="left", padx=(2, 8))
        self._s_uptime = tk.Label(uptime_f, text="--:--:--", font=("Courier New", 11, "bold"),
                                   fg=C["accent"], bg=C["surface"])
        self._s_uptime.pack(side="left")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.update_idletasks()
        self.root.geometry(f"320x{self.root.winfo_reqheight()}")

    def _make_meter(self, parent, label, key, color):
        row = tk.Frame(parent, bg=C["bg"])
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, font=FX, fg=C["muted"], bg=C["bg"], width=20, anchor="w").pack(side="left")
        bg_bar = tk.Frame(row, bg=C["border"], height=10, width=130)
        bg_bar.pack(side="left", padx=4)
        bg_bar.pack_propagate(False)
        fill = tk.Frame(bg_bar, bg=C["muted"], height=10, width=0)
        fill.place(x=0, y=0, height=10)
        setattr(self, f"_{key}_bar", fill)
        setattr(self, f"_{key}_color", color)

    def _make_slider(self, parent, label, var, lo, hi, cmd, res=0.05):
        row = tk.Frame(parent, bg=C["bg"])
        row.pack(fill="x", pady=2)
        tk.Label(row, text=f"{label:<13}", font=FX, fg=C["text"], bg=C["bg"], width=13, anchor="w").pack(side="left")
        tk.Scale(
            row, variable=var, from_=lo, to=hi, orient="horizontal", length=165,
            bg=C["bg"], fg=C["text"], troughcolor=C["border"],
            activebackground=C["accent"], highlightthickness=0, bd=0,
            resolution=res, command=cmd, font=FX, showvalue=1,
        ).pack(side="left")

    def _make_stat(self, parent, label, init):
        f = tk.Frame(parent, bg=C["surface"], padx=6, pady=4)
        f.pack(side="left", expand=True, fill="x", padx=2, pady=2)
        tk.Label(f, text=label, font=FX, fg=C["muted"], bg=C["surface"]).pack()
        lbl = tk.Label(f, text=init, font=("Courier New", 11, "bold"),
                        fg=C["accent"], bg=C["surface"])
        lbl.pack()
        return lbl


    def _bind_keys(self):
        r = self.root
        r.bind("<KeyPress-plus>",  lambda _: self._on_gain(self.controller.RELATIVE_GAIN_X + 0.15))
        r.bind("<KeyPress-equal>", lambda _: self._on_gain(self.controller.RELATIVE_GAIN_X + 0.15))
        r.bind("<KeyPress-minus>", lambda _: self._on_gain(self.controller.RELATIVE_GAIN_X - 0.15))
        r.bind("<KeyPress-bracketleft>",  lambda _: self._on_dz(self.controller.RELATIVE_DEADZONE + 0.001))
        r.bind("<KeyPress-bracketright>", lambda _: self._on_dz(self.controller.RELATIVE_DEADZONE - 0.001))
        r.bind("<KeyPress-comma>",  lambda _: self._on_step(self.controller.MAX_STEP_PIXELS - 3))
        r.bind("<KeyPress-period>", lambda _: self._on_step(self.controller.MAX_STEP_PIXELS + 3))
        r.bind("<KeyPress-r>", lambda _: self.controller.reset_tracking_state())
        r.bind("<KeyPress-R>", lambda _: self.controller.reset_tracking_state())

 
    def _on_gain(self, val):
        self.controller.adjust_gain(float(val))
        self._gain_var.set(self.controller.RELATIVE_GAIN_X)

    def _on_dz(self, val):
        self.controller.adjust_deadzone(float(val))
        self._dz_var.set(self.controller.RELATIVE_DEADZONE)

    def _on_step(self, val):
        self.controller.adjust_max_step(int(float(val)))
        self._step_var.set(self.controller.MAX_STEP_PIXELS)

 
    def queue_status(self, text: str, level: str = "ok"):
        self._ui_queue.append(("status", text, level))

    def queue_pinch_update(self, action_ratio: float, drag_ratio: float, close_ratio: float):
        self._ui_queue.append(("pinch", action_ratio, drag_ratio, close_ratio))

    def queue_stats(self, stats: SessionStats):
        self._ui_queue.append(("stats", stats))

    def _tick(self):
        for item in list(self._ui_queue):
            self._ui_queue.remove(item)
            k = item[0]
            if k == "status":
                self._do_status(item[1], item[2])
            elif k == "pinch":
                self._do_pinch(item[1], item[2], item[3])
            elif k == "stats":
                self._do_stats(item[1])
            elif k == "_show_cursor":
                if self.floating_cursor is None:
                    self.floating_cursor = FloatingCursor()
                self._do_status("Hand detected", "ok")
            elif k == "_hide_cursor":
                if self.floating_cursor:
                    try:
                        self.floating_cursor.destroy()
                    except Exception:
                        pass
                    self.floating_cursor = None
                if item[1]:
                    self._do_status("OFFLINE", "warn")

        if self.enabled:
            self._s_uptime.config(text=self.controller.stats.uptime_str())

        self.root.after(33, self._tick)

    def _do_status(self, text: str, level: str):
        color = {"ok": C["accent"], "warn": C["warn"], "error": C["error"]}.get(level, C["text"])
        self.status_label.config(text=text.upper()[:28], fg=color)
        dot_color = C["accent"] if level == "ok" else C["warn"] if level == "warn" else C["error"]
        self.hand_dot.config(fg=dot_color)

    def _do_pinch(self, action_ratio: float, drag_ratio: float, close_ratio: float):
        BAR_W = 130

        def bar_w(ratio, on_t, off_t):
            norm = float(np.clip(1.0 - ratio / off_t, 0.0, 1.0))
            return int(norm * BAR_W)

        aw = bar_w(action_ratio, self.controller.ACTION_PINCH_ON_RATIO, self.controller.ACTION_PINCH_OFF_RATIO)
        dw = bar_w(drag_ratio,   self.controller.DRAG_PINCH_ON_RATIO,   self.controller.DRAG_PINCH_OFF_RATIO)
        cw = bar_w(close_ratio,  self.controller.CLOSE_PINCH_ON_RATIO,  self.controller.CLOSE_PINCH_OFF_RATIO)

        a_col = C["accent"]  if aw > BAR_W * 0.6 else C["warn"] if aw > BAR_W * 0.3 else C["muted"]
        d_col = C["accent2"] if dw > BAR_W * 0.6 else C["warn"] if dw > BAR_W * 0.3 else C["muted"]
        c_col = C["error"]   if cw > BAR_W * 0.6 else C["warn"] if cw > BAR_W * 0.3 else C["muted"]

        self._action_bar.place(x=0, y=0, width=aw, height=10)
        self._action_bar.configure(bg=a_col)
        self._drag_bar.place(x=0, y=0, width=dw, height=10)
        self._drag_bar.configure(bg=d_col)
        self._close_bar.place(x=0, y=0, width=cw, height=10)
        self._close_bar.configure(bg=c_col)

    def _do_stats(self, stats: SessionStats):
        self._s_click.config(text=str(stats.clicks))
        self._s_open.config(text=str(stats.double_clicks))
        self._s_drag.config(text=str(stats.drags))
        self._s_close.config(text=str(stats.closes))

    # ── Cursor overlay (called from bg thread) ───
    def show_cursor_overlay(self):
        self._ui_queue.append(("_show_cursor",))

    def move_cursor_overlay(self, x: int, y: int):
        if self.floating_cursor:
            self.root.after(0, lambda: self.floating_cursor and self.floating_cursor.move(x, y))

    def hide_cursor_overlay(self, set_off_status: bool = True):
        self._ui_queue.append(("_hide_cursor", set_off_status))


    def toggle(self):
        if not self.enabled:
            self.enabled = True
            self.toggle_btn.config(text="[  STOP   ]", fg=C["error"],
                                   highlightbackground=C["error"])
            self._do_status("Initialising...", "warn")
            self.root.focus_force()
            self.controller.start(self)
        else:
            self.enabled = False
            self.toggle_btn.config(text="[  START  ]", fg=C["accent"],
                                   highlightbackground=C["accent"])
            self.controller.stop(self)
            self._do_status("OFFLINE", "warn")
            self.hand_dot.config(fg=C["muted"])
            self._do_pinch(1.0, 1.0, 1.0)
            self._s_uptime.config(text="--:--")

    def on_close(self):
        if self.enabled:
            self.controller.stop(self)
        self.root.destroy()

    def run(self):
        self.root.mainloop()



if __name__ == "__main__":
    print_startup_diagnostics()
    ControlApp().run()
