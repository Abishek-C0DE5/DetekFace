# Live Webcam Detection

The `detekface.camera()` function provides an out-of-the-box real-time webcam face detection experience with customizable overlays, keyboard-driven photo capture, and performance metrics.

---

## Basic Usage

```python
import detekface

detekface.camera()
```

By default, this:
1. Connects to camera device `0` (your primary built-in or USB webcam).
2. Mirrors the feed horizontally (`mirror=True`) so movement feels natural, like looking into a mirror.
3. Draws a clean green rectangle around detected faces.
4. Cleans up all system resources when you exit.

---

## Function Parameters

```python
def camera(
    camera_index: int = 0,
    mirror: bool = True,
    confidence_threshold: float = 0.7,
    detector: Optional[FaceDetector] = None,
    shape: str = "rectangle",
    color: Union[str, Tuple[int, int, int]] = "#00FF88",
    thickness: int = 3,
    show_confidence: bool = False,
    show_fps: bool = False,
    show_count: bool = False,
    capture_dir: Optional[Union[str, Path]] = "captures",
    on_capture: Optional[Callable[[FaceCapture], None]] = None,
    window_title: str = "detekface - Live Face Detection",
    max_frames: Optional[int] = None,
) -> None:
    ...
```

### Parameter Details

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `camera_index` | `int` | `0` | Hardware device index. Use `0` for default camera, `1` or `2` for secondary or external webcams. |
| `mirror` | `bool` | `True` | Horizontally flips the live video stream. |
| `confidence_threshold` | `float` | `0.7` | Minimum detection certainty required (between `0.0` and `1.0`). |
| `detector` | `Optional[FaceDetector]` | `None` | Pre-initialized `FaceDetector` instance if you wish to reuse an existing model. |
| `shape` | `str` | `"rectangle"` | Visual overlay shape: `"rectangle"`, `"corners"`, `"circle"`, `"rounded"`, or `"hud"`. |
| `color` | `str` or `tuple` | `"#00FF88"` | RGB hex string (e.g. `"#00FF88"`), named color (`"cyan"`, `"neon_green"`), or `(R, G, B)` tuple. |
| `thickness` | `int` | `3` | Line stroke width in pixels. |
| `show_confidence` | `bool` | `False` | Renders a confidence percentage badge (e.g. `94.2%`) near each face. |
| `show_fps` | `bool` | `False` | Displays measured processing throughput in the top-left corner. |
| `show_count` | `bool` | `False` | Displays total detected faces count in the top-left status bar. |
| `capture_dir` | `str` or `Path` | `"captures"` | Destination folder for face crops captured with the `c` key. |
| `on_capture` | `Callable` | `None` | Callback function invoked with the `FaceCapture` object when `c` is pressed. |
| `window_title` | `str` | `"..."` | Title bar text of the OpenCV display window. |
| `max_frames` | `Optional[int]` | `None` | Automatically terminates after processing this many frames (useful for testing). |

---

## Interactive Keyboard Controls

During live camera preview:

| Key | Action | What It Does |
| :---: | :--- | :--- |
| **`c`** or **`C`** | **Capture Face** | Extracts the largest face currently visible in the frame, saves a timestamped JPEG image to `capture_dir`, and shows a green on-screen confirmation banner for 1 second. |
| **`q`** or **`Q`** | **Quit** | Breaks the capture loop and cleanly closes the camera feed. |
| **`Escape`** | **Exit** | Immediately closes the camera and window. |
| **Window `X`** | **OS Close Button** | Clicking the window close button cleanly terminates without hanging. |

---

## Live Information Overlays

You can turn on metrics badges to monitor processing throughput and detection details:

```python
import detekface

detekface.camera(
    shape="hud",
    color="#00FF88",
    thickness=2,
    show_confidence=True, # Shows score badge near face (e.g., "94.2%")
    show_fps=True,        # Shows smoothed FPS counter (e.g., "FPS: 51.2")
    show_count=True,      # Shows face count banner (e.g., "1 Face")
)
```

### Stable FPS Calculation
The FPS display uses a 30-frame rolling window of time deltas rather than single-frame intervals. This eliminates erratic numerical jitter while remaining immediately responsive to frame drops.

---

## Using Capture Callbacks

If you want to trigger custom actions whenever the user captures a face:

```python
import detekface

def my_callback(face: detekface.FaceCapture) -> None:
    print(f"Face captured successfully!")
    print(f"  Dimensions : {face.width}x{face.height} pixels")
    print(f"  Confidence : {face.confidence * 100:.1f}%")
    print(f"  File saved : {face.path}")

detekface.camera(on_capture=my_callback)
```

> **Note on Callback Resilience**: If your callback raises an unhandled exception, DetekFace catches the error, prints a warning message, displays a red on-screen `"Callback failed!"` banner, and **keeps the camera stream running** without crashing.

---

## Guaranteed Resource Cleanup

DetekFace guarantees that the video device is released and all OpenCV windows are destroyed, even if an unexpected exception occurs during capture:

```python
# camera.py internal structure guarantees cleanup:
try:
    ...
finally:
    cap.release()
    cv2.destroyAllWindows()
```

This prevents the webcam hardware from remaining locked in your operating system.
