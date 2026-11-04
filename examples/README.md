# DetekFace Examples

This directory contains complete, runnable example scripts demonstrating key capabilities of `detekface`.

---

## Example Scripts Overview

| Script | Command | Description |
| :--- | :--- | :--- |
| **`webcam.py`** | `python examples/webcam.py` | Opens a live interactive webcam preview with real-time HUD overlays, smoothed FPS counter, face count badge, and keyboard face capture (`C`). |
| **`detect_image.py`** | `python examples/detect_image.py` | Detects faces in an image file, prints coordinates and 5 facial landmarks, and saves an annotated output image. |
| **`batch_detection.py`** | `python examples/batch_detection.py` | Runs batch face detection across folders or image lists with progress callbacks and optional crop saving. |
| **`crop_face.py`** | `python examples/crop_face.py` | Programmatically crops the most prominent face from a photo and saves the unannotated crop to disk. |
| **`video_detection.py`** | `python examples/video_detection.py` | Processes a prerecorded video file frame-by-frame with customizable overlay styles and exports an annotated video. |
| **`benchmark.py`** | `python examples/benchmark.py` | Benchmarks inference latency (Mean, Median, P95) and FPS throughput across multiple resolutions (320x240, 640x480, 1280x720) on your local hardware. |
| **`face_mesh_webcam.py`** | `python examples/face_mesh_webcam.py` | Live webcam feed with 478 dense facial landmark wireframe mesh overlay. |
| **`neon_mesh_webcam.py`** | `python examples/neon_mesh_webcam.py` | Live webcam feed with high-contrast neon mesh styling and anatomical anchors. |
| **`live_scan_webcam.py`** | `python examples/live_scan_webcam.py` | Live webcam feed with computer-vision laser scanning beam, reticle lock, and telemetry. |
| **`still_image_mesh.py`** | `python examples/still_image_mesh.py` | Still photo face detection with dense landmark extraction and mesh image export. |
| **`multi_face_mesh.py`** | `python examples/multi_face_mesh.py` | Multi-face photo with independent wireframe mesh rendering. |

---

## Running the Examples

Ensure your virtual environment is active before running any example:

```powershell
# Activate virtual environment on Windows
.\.venv\Scripts\Activate.ps1
```

### 1. Live Webcam Feed
```powershell
python examples/webcam.py
```
*Keyboard controls:* Press **`C`** to capture a face, or **`Q`** / **`Esc`** to quit.

### 2. Live Facial Mesh & Scanner
```powershell
# Basic wireframe mesh
python examples/face_mesh_webcam.py

# Glowing neon mesh
python examples/neon_mesh_webcam.py

# Futuristic Live FX laser scanner
python examples/live_scan_webcam.py
```
*Controls:* Press **`M`** to cycle mesh styles, **`F`** to cycle Live FX styles.

### 3. Still Image Detection with Mesh
```powershell
python examples/still_image_mesh.py tests/assets/sample_face.jpg
```

### 4. Multi-Face Mesh Rendering
```powershell
python examples/multi_face_mesh.py
```

### 5. Still Image Detection with Custom Style
```powershell
python examples/detect_image.py tests/assets/sample_face.jpg --save output.jpg --shape hud --color "#00FF88" --show-confidence
```

### 6. Programmatic Face Cropping
```powershell
python examples/crop_face.py tests/assets/sample_face.jpg
```

### 7. Video File Processing
```powershell
python examples/video_detection.py path/to/clip.mp4 --output annotated_clip.mp4 --shape rounded --color cyan
```

### 8. Performance Benchmark
```powershell
python examples/benchmark.py
```
