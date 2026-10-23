# DetekFace

A clean, reliable, and beginner-friendly Python face detection package powered by **YuNet** and **OpenCV**.

[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

1. [Features](#features)
2. [Supported Environments & Compatibility](#supported-environments--compatibility)
3. [Installation](#installation)
   - [Windows 11 Setup](#windows-11-setup)
   - [Development Installation](#development-installation)
4. [Quick Start & Usage](#quick-start--usage)
   - [1. Live Webcam Feed](#1-live-webcam-feed)
   - [2. Still-Image Detection & Annotated Saving](#2-still-image-detection--annotated-saving)
   - [3. Programmatic Face Cropping & Capture](#3-programmatic-face-cropping--capture)
   - [4. Prerecorded Video Processing](#4-prerecorded-video-processing)
5. [Visual Styles & Rendering Options](#visual-styles--rendering-options)
   - [Supported Overlay Shapes](#supported-overlay-shapes)
   - [Colors & Line Thickness](#colors--line-thickness)
   - [Information Overlays (Confidence, FPS, Count)](#information-overlays)
6. [Interactive Keyboard Controls](#interactive-keyboard-controls)
7. [Model Management & Cache Behavior](#model-management--cache-behavior)
8. [Troubleshooting](#troubleshooting)
9. [Known Limitations](#known-limitations)
10. [Licensing & Credits](#licensing--credits)

---

## Features

- **Sensible, Minimal Public API**: Use `detekface.camera()`, `detekface.detect()`, `detekface.capture()`, or `detekface.video()`.
- **High-Accuracy YuNet Engine**: Microsecond-fast deep learning inference through OpenCV's FaceDetectorYN.
- **Rich Visual Overlay Rendering**: 5 distinct overlay styles (`rectangle`, `corners`, `circle`, `rounded`, `hud`) with custom colors and stroke thickness.
- **Live Throughput & Count Metrics**: Real-time smoothed FPS throughput, face count badges, and confidence percentages.
- **Zero Crop Pollution**: Face captures and crops are strictly extracted from pristine original frames, guaranteeing no drawn boxes, shapes, or text in saved images.
- **Automatic Model Resolution**: Trusted remote download, SHA256 integrity validation, atomic user caching, and OpenCV 4.x / 5.x graph engine compatibility.
- **Beginner-Friendly Error Handling**: Actionable hints for camera disconnections, missing files, corrupt weights, or unsupported codecs.

---

## Supported Environments & Compatibility

- **Operating Systems**:
  - Windows 11 (fully tested and verified on AMD64)
  - Windows 10
  - Linux (Ubuntu 20.04+, Debian, Fedora, Arch)
  - macOS (Intel & Apple Silicon)
- **Python Versions**:
  - `Python >= 3.9` through `Python 3.15+` (tested on Python 3.15.1)
- **OpenCV Runtime**:
  - `opencv-python >= 4.8` or `opencv-contrib-python >= 4.8` (tested on OpenCV 5.0.0 graph engine)
  - *Note: Requires GUI-capable OpenCV builds (standard `opencv-python`) to render live preview windows (`cv2.imshow`). Do not mix with `opencv-python-headless` if preview windows are needed.*

---

## Installation

### Windows 11 Setup

1. **Verify Python & Pip Installation**:
   Open Windows Terminal (PowerShell) and ensure Python is accessible:
   ```powershell
   python --version
   pip --version
   ```

2. **Create and Activate a Virtual Environment** (Recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(If script execution is disabled on PowerShell, enable it via `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

3. **Install DetekFace**:
   Install from a wheel distribution or local project directory:
   ```powershell
   pip install detekface
   ```
   Or install directly from source:
   ```powershell
   pip install git+https://github.com/Abishek-C0DE5/DetekFace.git
   ```

### Development Installation

To install in editable mode with development tools:
```powershell
git clone https://github.com/Abishek-C0DE5/DetekFace.git
cd DetekFace
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Run the automated test suite:
```powershell
pytest
```

---

## Quick Start & Usage

### 1. Live Webcam Feed

Open your default webcam with real-time detection, HUD overlay, and performance metrics:

```python
import detekface

# Launch live preview
detekface.camera(
    camera_index=0,
    mirror=True,           # Horizontally flip feed for natural selfie view
    shape="hud",           # 'rectangle', 'corners', 'circle', 'rounded', 'hud'
    color="#00FF88",       # Hex code or color name ('cyan', 'neon_green', etc.)
    show_confidence=True,  # Display confidence scores near each face
    show_fps=True,         # Display measured FPS throughput
    show_count=True,       # Display current detected face count
)
```

### 2. Still-Image Detection & Annotated Saving

Detect faces in an image file or NumPy array, inspect detections, and save annotated outputs:

```python
import detekface

# Detect faces in an image file (str, Path, or BGR NumPy array)
result = detekface.detect("photo.jpg", confidence_threshold=0.7)

print(f"Total faces detected: {result.count}")

for i, face in enumerate(result.faces, start=1):
    print(f"Face #{i}: Confidence={face.confidence * 100:.1f}%")
    print(f"  BoundingBox: (x={face.bbox.x}, y={face.bbox.y}, w={face.bbox.width}, h={face.bbox.height})")
    if face.landmarks:
        print(f"  Right Eye  : {face.landmarks.right_eye.as_tuple()}")
        print(f"  Left Eye   : {face.landmarks.left_eye.as_tuple()}")
        print(f"  Nose Tip   : {face.landmarks.nose_tip.as_tuple()}")

# Display annotated image in a GUI window (press any key or 'X' to close)
result.show(shape="hud", color="cyan", show_confidence=True)

# Save annotated image with custom styling
result.save("annotated_photo.jpg", shape="hud", color="#00FF88", show_confidence=True)
```

### 3. Programmatic Face Cropping & Capture

Crop the largest face or a specific face index. Captured crops are always extracted strictly from the unannotated original image:

```python
import detekface

result = detekface.detect("group.jpg")

# Crop and save the largest face by visible area
face_crop = result.capture(save_dir="captures", filename="largest_face.jpg")
print(f"Saved {face_crop.width}x{face_crop.height}px crop to: {face_crop.path}")

# Or capture without saving to disk (retrieve pure NumPy array)
memory_crop = result.capture(face_index=0, save=False)
crop_array = memory_crop.image  # BGR uint8 NumPy array

# Convert to RGB for matplotlib or PIL
crop_rgb = memory_crop.rgb_image
```

### 4. Prerecorded Video Processing

Process a video file frame by frame, display playback, and export annotated video with identical resolution and frame rate:

```python
import detekface

# Process video file and save annotated MP4
saved_video = detekface.video(
    path="clip.mp4",
    output_path="annotated_clip.mp4",
    show=True,             # Set show=False for headless server processing
    shape="hud",
    color="#00FF88",
    show_confidence=True,
    show_fps=True,
    show_count=True,
)
print(f"Annotated video saved to: {saved_video}")
```

---

## Visual Styles & Rendering Options

### Supported Overlay Shapes

`detekface` provides five distinct face overlay rendering styles:

| Style | Name | Description |
| :--- | :--- | :--- |
| **Rectangle** | `"rectangle"` | Clean, classic rectangular bounding box (default). |
| **Corner Brackets** | `"corners"` | Minimalist tactical corner brackets framing the face. |
| **Circle** | `"circle"` | Geometric circle encompassing the detected face. *(Note: Geometric overlay indicator, not facial contour).* |
| **Rounded** | `"rounded"` | Modern rounded rectangle with smooth circular corners. |
| **HUD** | `"hud"` | High-tech sci-fi overlay featuring corner brackets, reticle crosshair ticks, and target badges. |

### Colors & Line Thickness

- **Hex Strings**: Any 3-digit or 6-digit RGB hex code (e.g. `"#00FF88"`, `"#00FFFF"`, `"#FF0055"`).
- **Documented Color Names**: `"green"`, `"neon_green"`, `"cyan"`, `"red"`, `"blue"`, `"yellow"`, `"magenta"`, `"orange"`, `"white"`, `"lime"`, `"teal"`, `"purple"`.
- **RGB Tuples**: Integer tuples in `(R, G, B)` format (e.g. `(0, 255, 136)`).
- **Thickness**: Any positive integer >= 1 (e.g. `thickness=2` or `thickness=3`).

### Information Overlays

- `show_confidence=True`: Renders a high-contrast percentage badge (e.g. `94.2%`) adjacent to the face bounding box across all styles.
- `show_fps=True`: Renders real-time measured processing throughput (e.g. `FPS: 51.2`) in the top-left status bar. Calculated via a 30-frame rolling window for smooth stability.
- `show_count=True`: Renders the number of faces detected in the current frame (e.g. `1 Face` or `3 Faces`).

---

## Interactive Keyboard Controls

During live camera preview or video playback (`detekface.camera()` or `detekface.video(show=True)`):

| Key | Action | Behavior |
| :--- | :--- | :--- |
| **`c`** or **`C`** | **Capture Face** | Crops the largest visible face from the clean original frame, writes it to `capture_dir`, displays on-screen confirmation, and invokes `on_capture` callback if provided. |
| **`q`** or **`Q`** | **Quit** | Closes preview window and cleanly releases camera/video resources. |
| **`Escape`** | **Exit** | Immediately closes window and releases resources. |
| **Window `X`** | **Close Button** | Clicking the operating system close button exits the loop immediately without hanging. |

---

## Model Management & Cache Behavior

`detekface` automatically manages the official YuNet ONNX model weights:
- **No Manual Downloads Required**: On first import or detection, the package automatically downloads the official YuNet model from the official OpenCV Model Zoo repository.
- **Compatibility Selection**:
  - Automatically selects `face_detection_yunet_2026may.onnx` (dynamic input shape) for modern OpenCV 5.x graph engines.
  - Automatically selects `face_detection_yunet_2023mar.onnx` for OpenCV 4.x.
- **Cache Location**:
  - Windows: `%LOCALAPPDATA%\detekface` (e.g. `C:\Users\<User>\AppData\Local\detekface`)
  - Linux / macOS: `~/.cache/detekface` (or `$XDG_CACHE_HOME/detekface`)
- **Integrity Validation**:
  - Downloaded models are verified against hardcoded official SHA256 checksums.
  - Corrupted or partial downloads are automatically purged.
- **Custom Local Models**: Pass an explicit local model path to bypass remote downloads entirely:
  ```python
  detector = detekface.FaceDetector(model="path/to/my_yunet.onnx")
  ```

---

## Troubleshooting

### 1. Camera Not Opening (`CameraNotFoundError`)
- **Cause**: Camera device index is invalid or another application (e.g. Zoom, Teams) has locked the webcam.
- **Fix**:
  1. Verify camera permissions in **Windows Settings -> Privacy & Security -> Camera**.
  2. Close any open video conferencing apps.
  3. Try `detekface.camera(camera_index=1)` if multiple video devices are connected.

### 2. GUI Window Not Opening (`DetekFaceError` / OpenCV Display Error)
- **Cause**: Running in a headless environment (e.g. Docker, SSH, Windows Server) or `opencv-python-headless` is installed instead of `opencv-python`.
- **Fix**:
  - For headless video processing or server tasks, use `show=False`:
    ```python
    detekface.video("clip.mp4", output_path="out.mp4", show=False)
    ```
  - For desktop use, ensure standard `opencv-python` is installed:
    ```powershell
    pip uninstall -y opencv-python-headless
    pip install opencv-python
    ```

### 3. Model Download Fails Behind Corporate Proxy (`ModelDownloadError`)
- **Cause**: Network firewall or proxy blocking access to GitHub releases.
- **Fix**:
  - Download `face_detection_yunet_2026may.onnx` manually from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
  - Place it in your cache directory (`%LOCALAPPDATA%\detekface`) or supply it directly:
    ```python
    detekface.detect("photo.jpg", model="C:/models/face_detection_yunet_2026may.onnx")
    ```

---

## Known Limitations

1. **Identity Tracking**: `detekface` performs independent frame-by-frame face detection and landmark localization. It does not track face identities over time or associate unique face IDs across video frames.
2. **Tiny Faces**: Extremely small faces (smaller than ~20x20 pixels in high-resolution images) may require lowering the confidence threshold (`confidence_threshold=0.35`–`0.45`) or cropping the region of interest.
3. **Extreme Profile Views**: YuNet is optimized for frontal and semi-profile face angles (up to ~60° yaw). Faces turned fully perpendicular to the camera may experience reduced detection confidence.
4. **Video Audio**: When writing annotated output video files via `detekface.video()`, only the visual video stream is processed and saved; audio tracks are not demuxed or re-encoded.

---

## Licensing & Credits

### Software License
`detekface` is released under the **MIT License**. See [LICENSE](LICENSE) for details.

### YuNet Model Credits & License
`detekface` utilizes the official **YuNet** face detection model created by Shiqi Yu and maintained in the OpenCV Model Zoo:
- **Author**: Shiqi Yu (`libfacedetection`) & Shenzhen Institute of Artificial Intelligence and Robotics for Society.
- **Model License**: MIT License (Copyright (c) 2020 Shiqi Yu).
- **Source Repository**: [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
