# Configuration & Styling Guide

`detekface` is designed with sensible defaults while offering extensive styling, telemetry, and detection tuning options across all modes (`camera()`, `detect()`, `video()`, and `result.save()`).

---

## Visual Overlay Styles (`shape`)

`detekface` provides five distinct rendering styles for drawing detected faces:

| Shape Name | Appearance & Design | Best Used For |
| :--- | :--- | :--- |
| `"rectangle"` | Clean, classic 4-corner bounding box enclosing the face. Default style. | General purpose computer vision, debugging, standard pipelines. |
| `"corners"` | Sleek corner brackets drawn at the 4 vertices of the box with open edges. | Modern minimalist UI, camera viewfinders, surveillance dashboards. |
| `"circle"` | Smooth circular overlay centered on the face with a radius covering the bounding box. | Avatar creators, profile framing, social media applications. |
| `"rounded"` | Elegant rectangle with rounded corners. | Premium application interfaces, polished user experiences. |
| `"hud"` | High-tech Sci-Fi Heads-Up Display featuring corner brackets, crosshairs at the center, and a top identity bar. | Cyberpunk aesthetics, aerospace/military-style demos, robotics. |

### Visual Comparison Example

```python
import detekface

result = detekface.detect("person.jpg")

# Test different visual styles
for style in ["rectangle", "corners", "circle", "rounded", "hud"]:
    result.save(f"preview_{style}.jpg", shape=style, color="#00FF88", thickness=2)
```

> **Note on Geometric Accuracy**: Overlays such as `"circle"` represent geometric visual enclosures around the detected facial bounding box. They do not represent exact facial boundary contours.

---

## Color Customization (`color`)

The `color` parameter accepts three convenient formats:

### 1. Hex Strings
Standard CSS-style 6-character or 3-character hex color codes with or without a leading `#`:
```python
detekface.camera(color="#00FF88")  # Neon green (Hex)
detekface.camera(color="#FF4500")  # Orange-red
detekface.camera(color="00E5FF")   # Vibrant cyan
detekface.camera(color="#0F8")     # Shorthand 3-digit hex
```

### 2. Predefined Named Colors
Built-in human-readable color names:
```python
detekface.camera(color="neon_green")
```

Supported named colors include:
- `green` `(0, 255, 0)`
- `neon_green` `(0, 255, 136)`
- `red` `(255, 0, 0)`
- `blue` `(0, 0, 255)`
- `cyan` `(0, 255, 255)`
- `yellow` `(255, 255, 0)`
- `magenta` `(255, 0, 255)`
- `orange` `(255, 165, 0)`
- `white` `(255, 255, 255)`
- `black` `(0, 0, 0)`
- `lime` `(50, 205, 50)`
- `teal` `(0, 128, 128)`
- `purple` `(128, 0, 128)`

### 3. RGB Tuples
Direct 3-integer tuples in `(Red, Green, Blue)` format (0 to 255):
```python
detekface.camera(color=(0, 255, 136))
```

> **Color Conversion Guarantee**: OpenCV internally stores images in BGR order. `detekface.parse_color_to_bgr()` handles all conversions, preventing channel swapping mistakes.

---

## Line Thickness (`thickness`)

Control the stroke width in pixels of bounding boxes and shapes:
```python
detekface.camera(thickness=2)  # Refined 2px stroke
detekface.camera(thickness=4)  # Bold 4px stroke
```
- Must be a positive integer (`>= 1`).
- Default is `2` (or `3` for HUD accentuation).

---

## Telemetry & Information Overlays

Enable real-time performance and detection metrics in `camera()` or `video()`:

```python
detekface.camera(
    show_confidence=True,  # Displays confidence badge (e.g., '96.4%') near each face
    show_fps=True,         # Displays live smoothed processing FPS in top banner
    show_count=True,       # Displays detected face count in top banner
)
```

### 1. `show_confidence` (Boolean, Default: `False`)
Draws a readable semi-transparent badge with the detection percentage next to each face.
- Automatically adjusts position to remain within frame boundaries even if the face touches the top edge.

### 2. `show_fps` (Boolean, Default: `False`)
Measures real frame-to-frame processing and display throughput using a sliding-window rolling average (`FPSCounter`).
- Accurately measures system performance; never a hardcoded placeholder.

### 3. `show_count` (Boolean, Default: `False`)
Displays the exact count of faces detected in the current frame (e.g., `Faces: 3`).

---

## Detector Tuning

Adjust core model thresholds when calling `detect()`, `camera()`, or initializing `FaceDetector`:

```python
from detekface import FaceDetector

detector = FaceDetector(
    confidence_threshold=0.80,  # Stricter detection (fewer false positives)
    nms_threshold=0.25,         # Aggressive suppression of overlapping boxes
    top_k=5000,                 # Candidates evaluated prior to NMS
)
```

| Parameter | Default | Recommended Range | Purpose |
| :--- | :--- | :--- | :--- |
| `confidence_threshold` | `0.7` | `0.5` – `0.85` | **Precision vs Recall**. Lower values (e.g., `0.55`) help detect small, distant, or dimly lit faces. Higher values (e.g., `0.85`) eliminate spurious false detections. |
| `nms_threshold` | `0.3` | `0.2` – `0.4` | **Overlap suppression**. Governs how aggressively multiple overlapping boxes on the same face are merged. |
---

## Detection Backends (`backend`)

`detekface` supports multiple detection engines via an adapter architecture.

### Supported Backends

| Backend | Flag | Dependency | License | Default | Facial Keypoints |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **YuNet** | `backend="yunet"` | Built-in (`opencv-python`) | MIT / Apache 2.0 | **Yes** | 5 points: right eye, left eye, nose tip, right mouth, left mouth |
| **MediaPipe** | `backend="mediapipe"` | Optional (`pip install "detekface[mediapipe]"`) | Apache 2.0 | No | 6 points: right eye, left eye, nose tip, mouth center, right ear, left ear |

### Using YuNet (Default)

YuNet requires no extra dependencies beyond `detekface`:

```python
import detekface

# Both of these use YuNet by default:
detekface.camera()
result = detekface.detect("photo.jpg", backend="yunet")
```

### Using MediaPipe (Optional)

To use Google's MediaPipe BlazeFace short-range detector:

1. Install the optional dependency extra:
   ```bash
   pip install "detekface[mediapipe]"
   ```

2. Specify `backend="mediapipe"`:
   ```python
   import detekface

   # Real-time camera with MediaPipe backend:
   detekface.camera(backend="mediapipe")

   # Still image detection:
   result = detekface.detect("photo.jpg", backend="mediapipe")

   # Prerecorded video processing:
   detekface.video("clip.mp4", backend="mediapipe")
   ```

### Landmark Contracts: Never Invented

Each backend normalizes its outputs into `detekface`'s standard `DetectionResult` and `FaceDetection` objects:

- **YuNet** provides 5 landmarks: `right_eye`, `left_eye`, `nose_tip`, `right_mouth`, `left_mouth`. MediaPipe-specific landmarks (`mouth_center`, `right_ear`, `left_ear`) are `None`.
- **MediaPipe** provides 6 landmarks: `right_eye`, `left_eye`, `nose_tip`, `mouth_center`, `right_ear`, `left_ear`. Because MediaPipe does not provide separate mouth corners, `right_mouth` and `left_mouth` are `None`.
- **Integrity Guarantee**: `detekface` will **never** invent or hallucinate missing landmarks. Code accessing `face.landmarks.mouth_center` or `face.landmarks.mouth_left` should check for `None` or use the backend appropriate for their task.

### Strict Error Handling: No Silent Fallbacks

If `backend="mediapipe"` is requested but `mediapipe` is not installed, `detekface` raises `DependencyNotFoundError` with clear installation instructions:

```
detekface.exceptions.DependencyNotFoundError: The 'mediapipe' backend requires the optional 'mediapipe' package.
Install it with: pip install "detekface[mediapipe]"
```

`detekface` **never silently falls back** to a different backend if the requested backend is unavailable or fails, ensuring deterministic behavior in production pipelines.

---

## Model & Cache Configuration

By default, `detekface` automatically manages the official YuNet ONNX model file.

### Cache Directory Customization
You can configure where model weights are stored:

1. **Via Python argument**:
   ```python
   detekface.detect("photo.jpg", cache_dir="C:/app_cache/models")
   ```
2. **Via Environment Variable**:
   ```powershell
   $env:DETEKFACE_CACHE_DIR = "D:\CustomModels\DetekFace"
   ```

### Default Cache Paths
- **Windows**: `%LOCALAPPDATA%\detekface` (e.g., `C:\Users\<User>\AppData\Local\detekface`)
- **Linux**: `~/.cache/detekface`
- **macOS**: `~/Library/Caches/detekface`

### Using Custom / Local ONNX Models
If working offline or using a custom-trained YuNet model:
```python
# Pass an absolute or relative path to your local .onnx file
result = detekface.detect(
    "photo.jpg",
    model="models/custom_face_detection_yunet_2023mar.onnx"
)
```

---

## Advanced Camera Session (`open_camera`)

For custom pipelines, web applications, background processing, or programmatic face capture without the built-in GUI preview window, use `detekface.open_camera()` or `detekface.CameraSession`:

```python
import detekface

# Headless frame iteration with programmatic face capture:
with detekface.open_camera(camera_index=0, backend="yunet") as session:
    for result in session:
        print(f"FPS: {session.fps:.1f} | Faces: {result.count}")
        if result.has_faces:
            # Captures largest face crop programmatically from the current frame:
            face = session.capture(save_dir="captures")
            print(f"Captured face to {face.path}")
            break
```

### Callback-Based Processing

```python
with detekface.open_camera(camera_index=0) as session:
    def handle_frame(result: detekface.DetectionResult):
        for face in result.faces:
            print(f"Face at {face.bbox.as_tuple()} with confidence {face.confidence:.2f}")
        return True  # Return False to stop processing

    session.process(handle_frame, max_frames=100)
```

---

## Batch Image Detection (`detect_many`)

Process lists of image files, directories, or NumPy arrays sequentially with error isolation:

```python
import detekface

# Process multiple files with per-image error collection (no lost inputs):
results = detekface.detect_many([
    "images/one.jpg",
    "images/two.png",
    "images/missing.jpg",
])

for item in results:
    if item.is_success:
        print(f"{item.source}: Found {item.count} faces")
    else:
        print(f"{item.source}: Failed with error: {item.error}")

print("Total faces detected:", results.total_faces)
```

### Batch Configuration Options

- **Fail-Fast Mode** (`fail_fast=True`):
  Stops immediately and raises an exception upon encountering the first corrupt or missing file.
- **Directory Scanning & Recursion** (`recursive=True`):
  Scans all supported images (`.jpg`, `.png`, `.webp`, `.bmp`) within a folder and its subdirectories.
- **Memory Conservation** (`keep_images=False`):
  Frees image pixel buffers from memory immediately after inference, enabling large-scale batches (thousands of images) without RAM exhaustion.
- **Progress Callback** (`progress_callback=callback`):
  Monitor processing progress without external dependencies:
  ```python
  def progress(item, current, total):
      print(f"[{current}/{total}] {item.source}")

  detekface.detect_many("dataset/faces", progress_callback=progress)
  ```

---

## Error Handling

Invalid configuration parameters raise `ConfigurationError` with a corrective hint:

```python
import detekface
from detekface.exceptions import ConfigurationError

try:
    detekface.camera(shape="triangle")
except ConfigurationError as err:
    print(f"Configuration error: {err.message}")
    print(f"Resolution hint: {err.hint}")
```
