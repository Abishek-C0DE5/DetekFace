# Live Webcam Detection

The `detekface.camera()` function provides a real-time webcam face detection experience with customizable overlay shapes, colors, keyboard-driven photo capture, and live performance metrics.

---

## 1. How to Start the Webcam

### The Simplest 2-Line Start

To launch the webcam with default settings:

```python
import detekface

detekface.camera()
```

### What Happens When You Run This:
1. **Camera Connection**: Automatically connects to device `0` (your primary built-in or USB webcam).
2. **Automatic Model Loading**: Loads the YuNet neural network from local cache (downloads it automatically on first run).
3. **Natural Mirroring**: The feed is flipped horizontally (`mirror=True`) so movement feels natural, like looking into a mirror.
4. **Live Overlay**: Draws a green rectangular bounding box around detected faces.
5. **Clean Exit**: Press <kbd>Q</kbd> or <kbd>Esc</kbd> to close the window and release the webcam hardware.

---

### Selecting a Specific Camera Device

If you have an external USB webcam, capture card, or multiple connected cameras, pass `camera_index`:

```python
import detekface

# Open secondary / external USB webcam (index 1)
detekface.camera(camera_index=1)
```

### Disabling Mirror Mode

If you are pointing the camera at a scene or document rather than a selfie view:

```python
import detekface

# Show the true camera orientation (no horizontal flip)
detekface.camera(mirror=False)
```

---

## 2. How to Change Overlay Shapes (With Examples)

DetekFace includes **5 built-in overlay shapes**: `"rectangle"`, `"corners"`, `"circle"`, `"rounded"`, and `"hud"`.

### Style 1: Classic Rectangle (Default)
Standard 4-sided bounding box. Clean, reliable, and classic.

```python
import detekface

detekface.camera(
    shape="rectangle",
    color="#00FF88",
    thickness=2,
)
```

---

### Style 2: Tactical Corner Brackets
Draws bracket accents only at the 4 corners of the bounding box, leaving the center unobstructed. Great for camera viewfinder UIs.

```python
import detekface

detekface.camera(
    shape="corners",
    color="cyan",
    thickness=2,
)
```

---

### Style 3: Geometric Circle
Draws a smooth circular enclosure centered on the detected face. Ideal for avatar framing and social media style previews.

```python
import detekface

detekface.camera(
    shape="circle",
    color="magenta",
    thickness=2,
)
```

> *Note: The circle overlay is a geometric enclosure centered on the face's bounding box; it is not a 3D facial contour.*

---

### Style 4: Modern Rounded Rectangle
Draws smooth curved corners on the bounding box for a sleek, modern UI look.

```python
import detekface

detekface.camera(
    shape="rounded",
    color="#FF4081",
    thickness=3,
)
```

---

### Style 5: Sci-Fi Tactical HUD
Draws corner brackets, center reticle crosshairs, and a top identity tag bar for an aerospace / cyberpunk aesthetic.

```python
import detekface

detekface.camera(
    shape="hud",
    color="#00FF88",
    thickness=3,
    show_confidence=True,
    show_fps=True,
    show_count=True,
)
```

---

## 3. How to Change Colors (With Examples)

The `color` argument accepts three convenient formats: **Hex strings**, **Predefined Color Names**, or **RGB tuples**.

### Option A: Using CSS-Style Hex Strings
Specify exact RGB colors using 6-digit or 3-digit hex codes (with or without `#`):

```python
import detekface

# Vibrant Cyberpunk Neon Green
detekface.camera(color="#00FF88")

# Electric Blue
detekface.camera(color="#00D2FF")

# Sunset Orange
detekface.camera(color="#FF6B00")

# Hot Pink
detekface.camera(color="#FF007F")
```

---

### Option B: Using Predefined Color Names
Choose from 13 built-in human-readable color names:

```python
import detekface

detekface.camera(color="cyan")
```

#### Available Named Colors:
- `"green"`
- `"neon_green"`
- `"cyan"`
- `"red"`
- `"blue"`
- `"yellow"`
- `"magenta"`
- `"orange"`
- `"white"`
- `"lime"`
- `"teal"`
- `"purple"`
- `"black"`

---

### Option C: Using RGB Tuples
Provide explicit `(Red, Green, Blue)` values from 0 to 255:

```python
import detekface

# Custom Warm Gold: Red=255, Green=215, Blue=0
detekface.camera(color=(255, 215, 0), shape="rounded")
```

> **Automatic BGR Conversion**: OpenCV uses BGR channel ordering internally. DetekFace automatically converts RGB hex codes, names, and tuples to BGR so you never get red/blue channel inversions.

---

## 4. Adjusting Line Thickness

Control the stroke width in pixels using `thickness`:

```python
import detekface

# Subtle, refined lines (1px)
detekface.camera(thickness=1)

# Standard readable stroke (2px - 3px)
detekface.camera(thickness=3)

# Bold, high-contrast stroke (5px)
detekface.camera(thickness=5)
```

---

## 5. Ready-to-Use Aesthetic Presets

Copy and run these curated combinations for different visual styles:

### Preset 1: The Cyberpunk Terminal
```python
import detekface

detekface.camera(
    shape="hud",
    color="#00FF88",
    thickness=3,
    show_confidence=True,
    show_fps=True,
    show_count=True,
    window_title="DetekFace // CYBER-HUD",
)
```

### Preset 2: Minimalist Camera Viewfinder
```python
import detekface

detekface.camera(
    shape="corners",
    color="white",
    thickness=2,
    mirror=True,
    window_title="Viewfinder",
)
```

### Preset 3: Modern Social Avatar Framer
```python
import detekface

detekface.camera(
    shape="circle",
    color="#00E5FF",
    thickness=2,
    show_confidence=True,
    window_title="Avatar Framing",
)
```

---

## 6. Interactive Keyboard Controls

While the camera preview is active, you can use these keyboard shortcuts:

| Key | Action | Description |
| :---: | :--- | :--- |
| <kbd>C</kbd> | **Capture Face** | Crops the largest visible face from the clean original frame, saves it to `./captures/`, and shows a green on-screen confirmation banner. |
| <kbd>Q</kbd> | **Quit** | Breaks the capture loop and cleanly closes the camera feed. |
| <kbd>Esc</kbd> | **Exit** | Immediately closes the camera and destroys all OpenCV windows. |
| <kbd>✕</kbd> | **OS Close Button** | Clicking the window close button cleanly terminates without hanging. |

---

## 7. Using Capture Callbacks

To receive the captured face object programmatically when the user presses <kbd>C</kbd>:

```python
import detekface

def on_face_captured(face: detekface.FaceCapture) -> None:
    print(f"\n[EVENT] Face captured!")
    print(f"  Dimensions : {face.width}x{face.height} pixels")
    print(f"  Confidence : {face.confidence:.2%}")
    print(f"  File saved : {face.path}")

detekface.camera(
    shape="hud",
    color="cyan",
    on_capture=on_face_captured,
)
```

---

## 8. Full Parameter Reference

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

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `camera_index` | `int` | `0` | Hardware device index (`0` for primary, `1` for secondary). |
| `mirror` | `bool` | `True` | Horizontally flips the live feed for natural mirror view. |
| `confidence_threshold` | `float` | `0.7` | Minimum detection certainty required (`0.0` to `1.0`). |
| `shape` | `str` | `"rectangle"` | Overlay shape: `"rectangle"`, `"corners"`, `"circle"`, `"rounded"`, `"hud"`. |
| `color` | `str` / `tuple` | `"#00FF88"` | Hex string (`"#00FF88"`), named color (`"cyan"`), or `(R, G, B)` tuple. |
| `thickness` | `int` | `3` | Line stroke width in pixels (`>= 1`). |
| `show_confidence` | `bool` | `False` | Renders a confidence percentage badge near each face. |
| `show_fps` | `bool` | `False` | Displays smoothed real-time processing FPS. |
| `show_count` | `bool` | `False` | Displays total detected faces count. |
| `capture_dir` | `str` / `Path` | `"captures"` | Destination folder for face crops captured with <kbd>C</kbd>. |
| `on_capture` | `Callable` | `None` | Callback invoked with `FaceCapture` when <kbd>C</kbd> is pressed. |
| `window_title` | `str` | `"..."` | Custom text shown on the desktop window title bar. |
| `max_frames` | `Optional[int]` | `None` | Max frames to process before closing (useful for automated testing). |
