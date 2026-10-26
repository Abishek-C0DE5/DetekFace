<p align="center">
  <a href="https://github.com/Abishek-C0DE5/DetekFace">
    <img src="docs/assets/detekface_main_logo.png" alt="DetekFace Logo" width="560" />
  </a>
</p>

<p align="center">
  <strong>High-performance, beginner-friendly face detection for Python powered by YuNet, OpenCV & MediaPipe.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/detekface/"><img src="https://img.shields.io/pypi/v/detekface.svg?color=00FF88&logo=pypi&logoColor=white" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/detekface/"><img src="https://img.shields.io/pypi/pyversions/detekface.svg?logo=python&logoColor=white" alt="Python Versions"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/Abishek-C0DE5/DetekFace/actions/workflows/ci.yml"><img src="https://github.com/Abishek-C0DE5/DetekFace/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="https://github.com/Abishek-C0DE5/DetekFace"><img src="https://img.shields.io/badge/GitHub-DetekFace-181717?logo=github&logoColor=white" alt="GitHub"></a>
</p>

---

## ⚡ Instant Installation

DetekFace is officially published on PyPI. Install it in seconds:

```bash
pip install detekface
```

To enable the optional Google MediaPipe BlazeFace backend:

```bash
pip install "detekface[mediapipe]"
```

> **Zero configuration needed**: YuNet ONNX neural weights are automatically downloaded from official OpenCV mirrors, verified with SHA256 checksums, and cached locally on first run.

---

## 🧭 Interactive Quick Navigation

<div align="center">

| 🚀 Getting Started | 🎨 Features & Workflows | 📚 Deep-Dive Documentation |
| :--- | :--- | :--- |
| • [**Minimal Quickstart**](#6-minimal-quick-start)<br>• [**Installation Requirements**](#4-installation-requirements)<br>• [**Windows 11 & OS Setup**](#5-windows-11--operating-system-setup) | • [**Live Webcam**](#7-live-webcam-usage--keyboard-controls)<br>• [**Camera Session API**](#8-advanced-camera-integration-session-lifecycle)<br>• [**5 Overlay Styles**](#9-visual-overlay-styles--color-customization)<br>• [**Face Cropping**](#10-face-capture-cropping--callbacks)<br>• [**Still Image Detection**](#11-still-image-detection--result-objects)<br>• [**Batch Processing**](#12-batch-image-detection-detect_many)<br>• [**Video Processing**](#13-prerecorded-video-processing)<br>• [**Command-Line Tool**](#14-command-line-interface-cli)<br>• [**Backend Selection**](#15-multiple-backend-selection-yunet--mediapipe) | • [**Documentation Index**](docs/README.md)<br>• [**Installation Guide**](docs/installation.md)<br>• [**Quick Start Guide**](docs/quickstart.md)<br>• [**Webcam Guide**](docs/webcam.md)<br>• [**Image Detection**](docs/image-detection.md)<br>• [**Batch Guide**](docs/batch.md)<br>• [**Video Guide**](docs/video.md)<br>• [**Face Capture Guide**](docs/face-capture.md)<br>• [**Backend Guide**](docs/backends.md)<br>• [**Configuration Reference**](docs/configuration.md)<br>• [**API Reference**](docs/api-reference.md)<br>• [**Troubleshooting Guide**](docs/troubleshooting.md)<br>• [**Validation Guide**](docs/validation.md)<br>• [**Development Guide**](docs/development.md)<br>• [**Licensing Guide**](docs/licensing.md) |

</div>

---

## 1. Project Overview

**DetekFace** brings state-of-the-art deep learning face detection to Python with zero boilerplate. Whether you want a 2-line real-time camera viewfinder with sci-fi HUD graphics, automatic unpolluted face cropping for avatar pipelines, batch processing of still photos, or offline video analysis, DetekFace gives you a clean, pythonic interface that just works.

---

## 2. Implemented Features vs. Planned Improvements

### ✅ Fully Implemented Features
- 📹 **2-Line Live Webcam**: Start a real-time face detection camera stream with `detekface.camera()`.
- 🔌 **Multiple Backends**: Seamlessly switch between built-in YuNet (`backend="yunet"`) and optional Google MediaPipe (`backend="mediapipe"`).
- 🔄 **Advanced Camera Lifecycle**: Context manager `open_camera()` and `CameraSession` for headless frame iteration and programmatic capture.
- 🖼️ **Still-Image Face Detection**: Inspect bounding boxes, confidence scores, and landmarks with `detekface.detect()`.
- 📦 **Batch Image Processing**: Process directories or image lists with `detekface.detect_many()` with error tracking and progress callbacks.
- 🎬 **Prerecorded Video Processing**: Process video files frame-by-frame with frame-skipping and resolution scaling via `detekface.video()`.
- 💻 **Command-Line Interface**: Run camera, photo, batch, or video detection straight from terminal via `detekface`.
- ✂️ **Pristine Face Cropping**: Extract clean, unpolluted face crops programmatically or with the <kbd>C</kbd> key. Drawn overlays never contaminate saved crops.
- 🎨 **5 Distinct Overlay Styles**: Choose between tactical corner brackets, modern rounded boxes, classic rectangles, geometric circles, or a full sci-fi HUD.
- 📊 **Real-Time Telemetry**: Display smoothed rolling-average FPS, detected face count badges, and confidence percentage tags.
- 🔒 **Automatic Model Management**: Official YuNet ONNX and MediaPipe TFLite models are automatically downloaded, SHA256-verified, and cached locally.
- 💡 **Actionable Error Messages**: Exception hierarchy with friendly `.hint` attributes for disconnected webcams, missing files, or corrupted data.
- 🧪 **Comprehensive CI/CD**: Matrix testing across Linux, Windows, and macOS (Python 3.9–3.13) with PyPI Trusted Publishing.

### 🔮 Planned Improvements (Roadmap)
- 📐 **Face Alignment Normalization**: Optional 2D affine rotation aligning eye landmarks horizontally before saving crops.
- 🧵 **Multi-Threaded Batch Inference**: Optional parallel worker pool for batch image processing.
- 🎥 **Video Audio Multiplexing**: Optional FFmpeg integration for preserving original audio streams during video re-encoding.

---

## 3. How Face Detection Works: A Visual Overview

```
                      +-----------------------------+
                      |   Input Source              |
                      |   (Webcam / Image / Video)  |
                      +--------------+--------------+
                                     |
                                     v [BGR Image Frame]
                      +-----------------------------+
                      |   Deep Learning Model       |
                      |   (YuNet ONNX / MediaPipe)  |
                      +--------------+--------------+
                                     |
                                     v [Coordinates & Confidences]
                      +-----------------------------+
                      |   Detection Engine Parser   |
                      |   (Bounding Box + Landmarks)|
                      +--------------+--------------+
                                     |
             +-----------------------+-----------------------+
             |                                               |
             v                                               v
   +--------------------+                         +--------------------+
   |  Overlay Renderer  |                         |  Face Capture Crop |
   |  (Draws Overlays   |                         |  (Extracts from    |
   |   on Display Copy) |                         |   Original Frame)  |
   +---------+----------+                         +---------+----------+
             |                                               |
             v                                               v
 [Annotated Preview / Video]                       [Clean Unpolluted Crop]
```

1. **Frame Ingestion**: Input images, video frames, or webcam feeds are standardized into 3-channel BGR uint8 NumPy arrays.
2. **Inference**: Frames pass through the deep learning detector (YuNet via OpenCV `FaceDetectorYN` or MediaPipe BlazeFace).
3. **Parsing**: Network proposals are converted into structured dataclasses containing bounding box coordinates `(x, y, width, height)`, confidence scores `(0.0 to 1.0)`, and anatomical keypoints.
4. **Overlay Rendering**: Visual indicators are drawn onto a *copy* of the frame for preview display or video export.
5. **Crop Purity**: When cropping faces, DetekFace slices directly from the *pristine original frame*, ensuring saved face images remain 100% clean and free of drawn graphics.

---

## 4. Installation Requirements

- **Python**: `>= 3.9` through `3.13` (tested on 3.10, 3.11, 3.12, 3.15).
- **Core Dependencies**:
  - `numpy>=1.20`
  - `opencv-python>=4.8` *(standard GUI-capable wheel; use `opencv-python-headless` for headless cloud environments)*.
- **Optional Extras**:
  - `mediapipe>=0.10.0`: Install via `pip install "detekface[mediapipe]"` for Google's BlazeFace backend.
- **Developer Extras**:
  - `pytest>=7.0`, `ruff>=0.1.0`, `mypy>=1.0`, `build>=1.0`, `twine>=5.0`: Install via `pip install -e ".[dev]"`.

---

## 5. Windows 11 & Operating System Setup

<details>
<summary><strong>🪟 Step-by-Step Windows 11 Setup</strong></summary>

<br>

1. **Open Windows Terminal or PowerShell**:
   ```powershell
   python --version
   pip --version
   ```

2. **Create and Activate a Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(If script execution is disabled on PowerShell, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

3. **Install DetekFace from PyPI**:
   ```powershell
   pip install detekface
   ```

</details>

<details>
<summary><strong>🐧 Linux (Ubuntu / Debian) Setup</strong></summary>

<br>

1. **Install OpenCV system libraries**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y --no-install-recommends libgl1 libglib2.0-0
   ```

2. **Create virtual environment and install**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install detekface
   ```

</details>

<details>
<summary><strong>🍎 macOS Setup (Apple Silicon & Intel)</strong></summary>

<br>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install detekface
```

</details>

---

## 6. Minimal Quick-Start

The simplest working example requires only **2 lines of Python**:

```python
import detekface

detekface.camera()
```

This immediately opens your default camera, runs real-time face detection, and displays a responsive bounding box around detected faces.

---

## 7. Live Webcam Usage & Keyboard Controls

Launch the camera with custom options:

```python
import detekface

detekface.camera(
    camera_index=0,  # 0 for default webcam, 1 for external USB camera
    mirror=True,  # Horizontally flip for natural mirror view
    confidence_threshold=0.7,
    shape="hud",  # 'rectangle', 'corners', 'circle', 'rounded', 'hud'
    color="#00FF88",  # Hex code or name ('cyan', 'neon_green', 'red', etc.)
    thickness=3,  # Stroke width in pixels
    show_confidence=True,  # Show confidence score near face
    show_fps=True,  # Show smoothed real-time FPS
    show_count=True,  # Show current face count
)
```

### 🎮 Interactive Keyboard Controls

During webcam or video playback, use these keyboard shortcuts:

| Key | Action | What it does |
| :---: | :--- | :--- |
| <kbd>C</kbd> | **Capture Face** | Crops the largest visible face from the clean original frame and saves it to `./captures/`. |
| <kbd>Q</kbd> | **Quit** | Cleanly exits the preview window and releases the camera device. |
| <kbd>Esc</kbd> | **Emergency Exit** | Immediately closes all OpenCV windows and terminates the loop. |
| <kbd>✕</kbd> | **Window Close** | Clicking the OS window close button terminates safely without hanging. |

👉 For full details, see the **[Live Webcam Guide](docs/webcam.md)**.

---

## 8. Advanced Camera Integration (Session Lifecycle)

For web frameworks (FastAPI, Flask), robotics, or headless background workers, use `open_camera()` and `CameraSession`:

```python
import detekface

# Open camera with explicit lifecycle management:
with detekface.open_camera(camera_index=0, mirror=True) as session:
    print(f"Camera opened: {session.camera_index}")

    for i in range(30):
        # Read the latest frame and structured detections:
        result = session.read()

        if result.has_faces:
            print(f"Frame #{i}: Detected {result.count} face(s)")

        # Programmatically capture a face while the camera is active:
        if i == 15 and result.has_faces:
            crop = session.capture(save_dir="captures", filename="avatar.jpg")
            print(f"Saved programmatic crop to: {crop.path}")

# Camera hardware is automatically released upon exit!
```

---

## 9. Visual Overlay Styles & Color Customization

DetekFace includes **5 built-in face overlay shapes**:

| Style | Parameter | Visual Preview & Aesthetic |
| :--- | :--- | :--- |
| **Rectangle** | `shape="rectangle"` | `┌────────┐`<br>`│ [Face] │` Classic bounding box framing the face (default).<br>`└────────┘` |
| **Corners** | `shape="corners"` | `┌      ┐`<br>`  [Face]  ` Minimalist tactical corner brackets at the 4 vertices.<br>`└      ┘` |
| **Circle** | `shape="circle"` | `╭──────╮`<br>`│ [Face] │` Geometric circle indicator enclosing the face.<br>`╰──────╯` |
| **Rounded** | `shape="rounded"` | `╭──────╮`<br>`│ [Face] │` Modern rounded rectangle with smooth circular corners.<br>`╰──────╯` |
| **HUD** | `shape="hud"` | `┌─[ID]─┐`<br>`┤  ┼   ├` High-tech Sci-Fi HUD with corner brackets and crosshair.<br>`└──────┘` |

### 🎨 Color Formats
Colors accept any of the following formats:
- **RGB Hex Strings**: `"#00FF88"`, `"#00FFFF"`, `"#FF3366"`, `"#0F8"`
- **Predefined Color Names**: `"green"`, `"neon_green"`, `"cyan"`, `"red"`, `"blue"`, `"yellow"`, `"magenta"`, `"orange"`, `"white"`, `"lime"`, `"teal"`, `"purple"`
- **RGB Tuples**: `(0, 255, 136)`, `(255, 0, 0)`

👉 For visual details, see the **[Configuration & Styling Guide](docs/configuration.md)**.

---

## 10. Face Capture, Cropping, & Callbacks

### Programmatic Capture
Crop the largest face or a specific face index from any `DetectionResult`:

```python
import detekface

result = detekface.detect("group_photo.jpg")

# 1. Capture and save the largest face to disk
face = result.capture(save_dir="captures", filename="avatar.jpg")
print(f"Saved {face.width}x{face.height}px crop to: {face.path}")

# 2. Capture in-memory without saving to disk
memory_face = result.capture(face_index=0, save=False)
bgr_array = memory_face.image  # NumPy array in BGR format
rgb_array = memory_face.rgb_image  # NumPy array converted to RGB for PIL / Matplotlib
```

### Webcam Capture Callback
Receive captured faces programmatically whenever the user presses <kbd>C</kbd>:

```python
import detekface


def handle_capture(face: detekface.FaceCapture) -> None:
    print(f"Captured face: {face.width}x{face.height}px, score={face.confidence:.2%}")
    print(f"Saved file: {face.path}")


detekface.camera(on_capture=handle_capture)
```

👉 For more details, see the **[Face Capture & Cropping Guide](docs/face-capture.md)**.

---

## 11. Still-Image Detection & Result Objects

Detect faces in image files or raw NumPy arrays:

```python
import detekface

# Run detection
result = detekface.detect("photo.jpg", confidence_threshold=0.6)

# Inspect detection results
print(f"Faces found: {result.count}")
print(f"Has faces  : {result.has_faces}")

for face in result.faces:
    print(f"Confidence : {face.confidence * 100:.1f}%")
    print(f"Bounding Box: x={face.bbox.x}, y={face.bbox.y}, w={face.bbox.width}, h={face.bbox.height}")
    if face.landmarks:
        print(f"  Right Eye: {face.landmarks.right_eye.as_tuple()}")
        print(f"  Left Eye : {face.landmarks.left_eye.as_tuple()}")
        print(f"  Nose Tip : {face.landmarks.nose_tip.as_tuple()}")

# Preview annotated image in a window
result.show(shape="hud", color="cyan", show_confidence=True)

# Save annotated image with custom styling
result.save("output_annotated.jpg", shape="hud", color="#00FF88", show_confidence=True)
```

👉 For more details, see the **[Image Detection Guide](docs/image-detection.md)**.

---

## 12. Batch Image Detection (`detect_many`)

Process lists of image paths or entire directories with error reporting and progress callbacks:

```python
import detekface

# 1. Process a list of images or a folder path
results = detekface.detect_many(
    "dataset/photos/",
    progress_callback=lambda done, total: print(f"Processing: {done}/{total}"),
    keep_images=False,  # Low memory usage for large batches
)

print(f"Batch Summary: {results.total_faces} faces across {len(results.successes)} photos.")

# 2. Iterate through items
for item in results:
    if item.is_success:
        print(f"✓ {item.source}: {item.count} face(s)")
    else:
        print(f"✗ {item.source}: error - {item.error}")
```

👉 For more details, see the **[Batch Detection Guide](docs/batch.md)**.

---

## 13. Prerecorded Video Processing

Process video files frame-by-frame and export annotated clips with preserved resolution and framerate:

```python
import detekface

saved_path = detekface.video(
    path="input_clip.mp4",
    output_path="annotated_clip.mp4",
    show=True,  # Set show=False for headless processing
    shape="hud",
    color="#00FF88",
    show_confidence=True,
    show_fps=True,
    show_count=True,
    frame_skip=1,  # Skip 1 frame between detections for 2x speedup
)
print(f"Exported annotated video to: {saved_path}")
```

*(Note: Audio tracks are not re-encoded; only visual video frames are processed and exported).*

👉 For more details, see the **[Video Processing Guide](docs/video.md)**.

---

## 14. Command-Line Interface (CLI)

`detekface` provides a lightweight, beginner-friendly command-line tool with zero extra dependencies:

```bash
# View general help and available commands
detekface --help

# Launch live webcam viewfinder
detekface camera --shape hud --color "#00FF88" --show-fps

# Detect faces in a photo and save annotated output
detekface image photo.jpg --output annotated.jpg --overwrite

# Crop the largest face directly from terminal
detekface image photo.jpg --crop-face avatar.jpg --overwrite

# Batch process an entire folder of photos
detekface batch ./photos --output ./annotated_photos --recursive

# Process and export an annotated video file
detekface video clip.mp4 --output clip_detected.mp4 --no-preview
```

---

## 15. Multiple Backend Selection (YuNet & MediaPipe)

DetekFace allows you to select between the built-in YuNet backend (default) and Google's MediaPipe BlazeFace:

```python
import detekface

# 1. Use default YuNet backend (5 landmarks)
result_yunet = detekface.detect("photo.jpg", backend="yunet")

# 2. Use optional MediaPipe backend (6 keypoints)
result_mp = detekface.detect("photo.jpg", backend="mediapipe")
```

| Feature | YuNet (Default) | MediaPipe (Optional) |
| :--- | :--- | :--- |
| **Dependencies** | None (`opencv-python` included) | `pip install "detekface[mediapipe]"` |
| **Landmarks** | 5 points (eyes, nose, mouth corners) | 6 keypoints (eyes, nose, mouth center, ears) |
| **Fail Policy** | Built-in | Raises `DependencyNotFoundError` with install hint |

👉 For more details, see the **[Backend Selection Guide](docs/backends.md)**.

---

## 16. Configuration & Error Handling

DetekFace provides structured, actionable exceptions:

```python
import detekface
from detekface.exceptions import ImageNotFoundError, CameraNotFoundError

try:
    result = detekface.detect("missing_file.jpg")
except ImageNotFoundError as e:
    print(f"Error: {e.message}")
    print(f"Hint : {e.hint}")
```

<details>
<summary><strong>🌲 View Full Exception Class Hierarchy</strong></summary>

<br>

```text
DetekFaceError
 ├── ModelError
 │    ├── ModelNotFoundError
 │    ├── ModelDownloadError
 │    ├── ModelCorruptionError
 │    └── ModelCompatibilityError
 ├── ImageError
 │    ├── ImageNotFoundError
 │    ├── InvalidImageError
 │    └── ImageSaveError
 ├── ConfigurationError
 ├── DependencyNotFoundError
 ├── InvalidSettingError
 ├── CameraError
 │    ├── CameraNotFoundError
 │    └── CameraReadError
 ├── CaptureError
 │    ├── NoFaceDetectedError
 │    ├── InvalidCropError
 │    └── CaptureSaveError
 ├── BatchError
 └── VideoError
      ├── VideoNotFoundError
      ├── InvalidVideoError
      └── VideoWriteError
```

</details>

👉 For full configuration details, see the **[Configuration Reference](docs/configuration.md)** and **[Troubleshooting Guide](docs/troubleshooting.md)**.

---

## 17. Supported Platforms & Dependencies

| Component | Supported Versions | Tested Environment |
| :--- | :--- | :--- |
| **Python** | `>= 3.9`, `3.10`, `3.11`, `3.12`, `3.13`, `3.15` | `3.15.1 64-bit` |
| **NumPy** | `>= 1.20` | `2.5.4` |
| **OpenCV** | `>= 4.8` (`opencv-python` or `opencv-contrib-python`) | `5.0.0.93` |
| **Operating Systems** | Windows 11, Windows 10, Linux, macOS | Windows 11 (AMD64) |

---

## 18. Model Acquisition, Caching, & Checksums

- **Automatic Download**: On first use, the model file is automatically downloaded from the official OpenCV Model Zoo.
- **Model Files**:
  - `face_detection_yunet_2026may.onnx` (dynamic shape for OpenCV 5.x graph engine)
  - `face_detection_yunet_2023mar.onnx` (fixed shape for OpenCV 4.x)
  - `blaze_face_short_range.tflite` (for MediaPipe backend)
- **Local Cache Directory**:
  - Windows: `%LOCALAPPDATA%\detekface\models`
  - Linux/macOS: `~/.cache/detekface/models`
- **Integrity Verification**: Downloads are validated against official SHA256 checksums before caching.
- **Custom Local Model**: Pass an explicit model path to bypass remote downloads:
  ```python
  import detekface

  detector = detekface.FaceDetector(model="C:/models/my_yunet.onnx")
  ```

---

## 19. Troubleshooting Common Issues

<details>
<summary><strong>🔧 Click to expand Common Fixes & Troubleshooting Guides</strong></summary>

<br>

### Camera Not Opening (`CameraNotFoundError`)
- **Fix**: Check Windows Privacy Settings (**Settings -> Privacy & Security -> Camera**) and ensure camera access is allowed for desktop apps. Close other apps (Zoom, Teams, OBS) using the webcam. Try `camera_index=1` if multiple devices are attached.

### Display Errors in Headless / Server Environments
- **Fix**: If running on a headless server or Docker container without a graphical display, set `show=False` in `detekface.video()` or avoid calling `result.show()`.

### Corporate Proxy Blocking Model Downloads
- **Fix**: Download `face_detection_yunet_2026may.onnx` manually from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), place it in `%LOCALAPPDATA%\detekface\models`, or pass the local path via `model="path/to/model.onnx"`.

👉 For comprehensive troubleshooting steps, visit the **[Detailed Troubleshooting Guide](docs/troubleshooting.md)**.

</details>

---

## 20. Testing & Performance Benchmarks

Run the complete automated test suite using `pytest`:

```bash
pytest -v
```

Execute repeatable performance benchmarks on your CPU:

```bash
python examples/benchmark.py
```

<details>
<summary><strong>📊 Measured Hardware Benchmark Results (AMD64 12-Core, Windows 11)</strong></summary>

<br>

```text
Resolution   | Mean Latency | Median Latency | P95 Latency | Throughput (FPS)
-------------|--------------|----------------|-------------|-----------------
320x240      | 9.83 ms      | 9.14 ms        | 13.60 ms    | 101.7 FPS
640x480      | 29.15 ms     | 28.52 ms       | 37.83 ms    | 34.3 FPS
1280x720     | 87.27 ms     | 85.84 ms       | 103.99 ms   | 11.5 FPS
```

</details>

---

## 21. Quality Validation Suite

Execute the reproducible quality validation suite comparing backends across real-world challenge scenarios (no faces, distant faces, low light, harsh shadows, profile angles, occlusions):

```bash
python validation/download_dataset.py
python validation/evaluate.py --compare
```

See the **[Quality Validation Methodology Guide](docs/validation.md)** for mathematical metric definitions (Precision, Recall, F1, IoU), dataset licensing, and detailed backend tradeoffs.

---

## 22. Development Setup & Contributing

We welcome contributions!
1. Clone the repository: `git clone https://github.com/Abishek-C0DE5/DetekFace.git`
2. Install developer dependencies: `pip install -e ".[dev,mediapipe]"`
3. Run linter and formatter: `ruff check .` and `ruff format --check .`
4. Run type checker: `mypy`
5. Run unit tests: `pytest -v -m "not network and not optional_backend"`

Check out our **[Contributing Guide](CONTRIBUTING.md)** and **[Development Guide](docs/development.md)**.

---

## 23. Licensing & Credits

### Software License
DetekFace is released under the **MIT License**. See [LICENSE](LICENSE) for details.

### YuNet Model License & Credits
The YuNet face detection model was created by Shiqi Yu and is distributed under the **MIT License**:
- **Author**: Shiqi Yu (`libfacedetection`) & Shenzhen Institute of Artificial Intelligence and Robotics for Society
- **Model License**: MIT License (Copyright (c) 2020 Shiqi Yu)
- **Source**: [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)

### MediaPipe BlazeFace License & Credits
- **Author**: Google MediaPipe Team & Google LLC
- **Model License**: Apache-2.0 License

👉 For complete IP breakdown, see the **[Licensing & Intellectual Property Guide](docs/licensing.md)**.
