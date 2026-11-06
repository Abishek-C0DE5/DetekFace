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
| • [**Minimal Quickstart**](#6-minimal-quick-start)<br>• [**Installation Requirements**](#4-installation-requirements)<br>• [**Windows 11 & OS Setup**](#5-windows-11--operating-system-setup)<br>• [**Supported Platforms**](#27-supported-platforms--dependencies) | • [**Live Webcam (60 FPS Async)**](#7-live-webcam-usage--keyboard-controls)<br>• [**Multi-Face Tracking**](#18-multi-face-tracking)<br>• [**3D Head-Pose Estimation**](#19-3d-head-pose-estimation)<br>• [**Facial Gesture Detection**](#20-real-time-facial-gesture-detection)<br>• [**AR Face Accessories**](#21-ar-face-accessories-engine)<br>• [**Face Alignment Normalization**](#22-face-alignment-normalization)<br>• [**Face Quality Assessment**](#23-technical-face-quality-assessment)<br>• [**Privacy Anonymization**](#24-face-privacy-mode-anonymization)<br>• [**Face Shape Analysis**](#17-face-shape-detection--biometric-scanner)<br>• [**Dense Facial Mesh**](#12-dense-facial-mesh--live-fx-scanner)<br>• [**Batch Processing**](#13-batch-image-detection-detect_many)<br>• [**Video Processing**](#14-prerecorded-video-processing)<br>• [**Command-Line Tool**](#15-command-line-interface-cli) | • [**Documentation Index**](docs/README.md)<br>• [**Multi-Face Tracking Guide**](docs/tracking.md)<br>• [**3D Head-Pose Guide**](docs/pose.md)<br>• [**Facial Gestures Guide**](docs/gestures.md)<br>• [**AR Accessories Guide**](docs/accessories.md)<br>• [**Face Alignment Guide**](docs/alignment.md)<br>• [**Face Quality Guide**](docs/quality.md)<br>• [**Privacy Filtering Guide**](docs/privacy.md)<br>• [**Performance Guide**](docs/performance.md)<br>• [**Webcam Guide**](docs/webcam.md)<br>• [**Image Detection**](docs/image-detection.md)<br>• [**API Reference**](docs/api-reference.md)<br>• [**Changelog (v0.3.0)**](CHANGELOG.md)<br>• [**Licensing Guide**](docs/licensing.md) |

</div>

---

## 1. Project Overview

**DetekFace** brings state-of-the-art deep learning face detection to Python with zero boilerplate. Whether you want a 2-line real-time camera viewfinder with sci-fi HUD graphics, automatic unpolluted face cropping for avatar pipelines, batch processing of still photos, or offline video analysis, DetekFace gives you a clean, pythonic interface that just works.

---

## 2. Implemented Features vs. Planned Improvements

### ✅ Fully Implemented Features (v0.3.0)
- 📹 **2-Line Live Webcam**: Start a real-time face detection camera stream with `detekface.camera()`.
- 🏎️ **Decoupled 60 FPS Asynchronous Engine**: Dedicated capture, background inference, and high-framerate rendering pipelines eliminating preview stuttering (`async_mode=True`).
- 🎯 **Multi-Face Tracking**: Persistent integer tracking IDs maintained across motion and brief occlusions (`track_faces=True`, `show_tracking_ids=True`).
- 📐 **3D Head-Pose Estimation**: Euler angle calculation (Yaw, Pitch, Roll in degrees) and 3D Cartesian orientation axis projection (`estimate_pose=True`, `show_pose=True`).
- 👁️ **Real-Time Facial Gesture Detection**: Event-driven detection of blinks, mouth opening, head turns, and tilts with callbacks (`detect_gestures=True`, `on_gesture=callback`).
- 🕶️ **AR Face Accessories Engine**: Augmented reality sunglasses, royal crown, face mask, futuristic visor, and halo with smooth head tracking (`effect="sunglasses"`).
- 📏 **Face Alignment Normalization**: Automatic eye-level counter-rotation and standardized aspect-preserving facial crops (`result.align()`, `result.aligned_faces()`).
- 🔍 **Technical Face Quality Assessment**: Objective Laplacian sharpness, luminance exposure, framing, and head-pose suitability scoring (`detekface.check_face_quality()`).
- 🛡️ **Face Privacy Anonymization**: Gaussian blur, pixelation mosaic, and solid masking with fail-closed safety protection (`detekface.anonymize()`).
- 🔌 **Multiple Backends**: Seamlessly switch between built-in YuNet (`backend="yunet"`) and optional Google MediaPipe (`backend="mediapipe"`).
- 👤 **Face Shape Detection & Biometric Scanner**: Dense geometric shape analysis supporting **both** still images and real-time webcam feeds with HUD laser scanning.
- 🔄 **Advanced Camera Lifecycle**: Context manager `open_camera()` and `CameraSession` for headless frame iteration and programmatic capture.
- 🖼️ **Still-Image Face Detection**: Inspect bounding boxes, confidence scores, and landmarks with `detekface.detect()`.
- 📦 **Batch Image Processing**: Process directories or image lists with `detekface.detect_many()` with error tracking and progress callbacks.
- 🎬 **Prerecorded Video Processing**: Process video files frame-by-frame with frame-skipping, quality HUD, AR accessories, and privacy filtering via `detekface.video()`.
- 💻 **Command-Line Interface**: Run camera, photo, batch, or video detection straight from terminal via `detekface`.
- ✂️ **Pristine Face Cropping**: Extract clean, unpolluted face crops programmatically or with the <kbd>C</kbd> key. Drawn overlays never contaminate saved crops.
- 🎨 **5 Distinct Overlay Styles**: Choose between tactical corner brackets, modern rounded boxes, classic rectangles, geometric circles, or a full sci-fi HUD.
- 📊 **Real-Time Telemetry**: Display smoothed rolling-average FPS, detected face count badges, and confidence percentage tags.
- 🔒 **Automatic Model Management**: Official YuNet ONNX and MediaPipe TFLite models are automatically downloaded, SHA256-verified, and cached locally.
- 💡 **Actionable Error Messages**: Exception hierarchy with friendly `.hint` attributes for disconnected webcams, missing files, or corrupted data.
- 🧪 **Comprehensive CI/CD & Tests**: 534 automated tests across Linux, Windows, and macOS (Python 3.9–3.15) with PyPI Trusted Publishing.

### 🔮 Planned Improvements (Roadmap)
- 🧵 **Multi-Threaded Batch Inference**: Optional parallel worker pool for bulk directory image processing.
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
| <kbd>A</kbd> | **Cycle AR Accessories** | Toggles augmented reality accessories live (`sunglasses`, `crown`, `mask`, `visor`, `halo`, off). |
| <kbd>T</kbd> | **Toggle Tracking IDs** | Shows/hides persistent multi-face tracking IDs across frames. |
| <kbd>P</kbd> | **Toggle 3D Pose** | Displays/hides 3D Cartesian head orientation axes (Yaw, Pitch, Roll). |
| <kbd>M</kbd> | **Cycle Mesh Style** | Toggles dense facial mesh wireframe styles live (`wireframe`, `neon`, `minimal`, `contour`, `points`, off). |
| <kbd>F</kbd> | **Cycle Live FX** | Toggles animated computer-vision laser scanner styles live (`scan`, `neon`, `wireframe`, `minimal`, off). |
| <kbd>Q</kbd> | **Quit** | Cleanly exits the preview window and releases the camera device. |
| <kbd>Esc</kbd> | **Emergency Exit** | Immediately closes all OpenCV windows and terminates the loop. |
| <kbd>✕</kbd> | **Window Close** | Clicking the OS window close button terminates safely without hanging. |

👉 For full details, see the **[Live Webcam Guide](docs/webcam.md)** and **[Facial Mesh Guide](docs/facial-mesh.md)**.

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

## 12. Dense Facial Mesh & Live FX Scanner

DetekFace 0.2.0 introduces **478 dense 2D facial surface landmarks**, 5 customizable mesh styles, and real-time animated computer-vision laser scanning:

```python
import detekface

# 1. Live Webcam with Neon Facial Mesh & Scanning Laser Beam
detekface.camera(
    show_mesh=True,
    mesh_style="neon",  # 'wireframe', 'neon', 'minimal', 'contour', 'points'
    mesh_color="#00FF88",  # Hex code, RGB tuple, or color name
    show_scan=True,  # Animated computer-vision laser scanning line
    show_fps=True,
)

# 2. Still-Image Mesh Rendering
result = detekface.detect("portrait.jpg", mesh=True)
annotated = result.render(show_mesh=True, mesh_style="neon")
result.save("portrait_mesh.jpg", show_mesh=True, mesh_style="neon")
```

> [!NOTE]
> Facial mesh visualization requires the optional MediaPipe extra: `pip install "detekface[mediapipe]"`.
> Planar 2D mesh wireframes represent perspective projections of facial keypoints, not volumetric medical reconstructions.

👉 For complete details, see the **[Facial Mesh & Live FX Guide](docs/facial-mesh.md)**.

---

## 13. Batch Image Detection (`detect_many`)

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

## 14. Prerecorded Video Processing

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

## 15. Command-Line Interface (CLI)

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

## 16. Multiple Backend Selection (YuNet & MediaPipe)

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

## 17. Face Shape Detection & Biometric Scanner

DetekFace includes full Face Shape Detection powered by dense 478-point 3D facial landmarks and geometric proportion analysis across **both** still images and real-time webcam feeds.

### Mode 1: Still-Image Face-Shape Analysis

Analyze a supplied photo without opening a webcam:

```python
import detekface

result = detekface.detect("portrait.jpg", analyze_face_shape=True)

for face in result.faces:
    print(
        f"Shape       : {face.face_shape.label}"
    )  # 'oval', 'round', 'square', 'heart', 'diamond', 'oblong', 'undetermined'
    print(f"Fit Score   : {face.face_shape.fit_score:.2f}")  # Heuristic fit score (0.0 to 1.0)
    print(f"Alternatives: {face.face_shape.alternatives}")  # Candidates ranked by fit score
    print(f"Measurements: {face.face_shape.measurements}")  # Proportions, ratios, and jaw angle
    print(f"Warnings    : {face.face_shape.warnings}")  # Reasons if quality is degraded

# Save annotated photo with contour silhouette and shape badges
result.save("portrait_analyzed.jpg", show_face_shape=True, show_shape_measurements=True)
```

### Mode 2: Real-Time Webcam Scanner

Launch the real-time biometric scanner with animated HUD laser sweep and temporal smoothing:

```python
import detekface

detekface.camera(
    analyze_face_shape=True,
    show_face_shape=True,
    face_shape_effect="scan",  # Biometric HUD laser scanline
)
```

- **HUD Laser Scanline**: Smooth oscillating laser sweeps down and up the face bounding box.
- **Temporal Smoothing**: `TemporalShapeTracker` eliminates frame-to-frame flicker with multi-frame voting.
- **Pristine Capture**: Press <kbd>C</kbd> to save the largest face crop untouched by annotations, while retaining structured shape metadata.

👉 For comprehensive details, see the **[Face Shape Guide](docs/face-shape.md)**.

---

## 18. Multi-Face Tracking

DetekFace includes an automated IoU-based multi-face tracker that assigns persistent integer IDs to each face across video frames and webcam feeds, maintaining identity through natural head movements, scale changes, and brief occlusions:

```python
import detekface

# Launch camera with persistent tracking IDs
detekface.camera(
    track_faces=True,
    show_tracking_ids=True,
    tracking_max_disappeared=15,  # Frames to retain track during occlusions
    tracking_iou_threshold=0.3,  # Spatial overlap threshold for association
)
```

In Python code, access tracking IDs directly from `Face.track_id`:

```python
result = session.read()
for face in result.faces:
    print(f"Face ID #{face.track_id}: bbox={face.bbox}")
```

👉 For full architecture and algorithms, see the **[Multi-Face Tracking Guide](docs/tracking.md)**.

---

## 19. 3D Head-Pose Estimation

Estimate 3D head orientation angles (**Yaw**, **Pitch**, and **Roll** in degrees) and render 3D Cartesian coordinate axes projected from the nose tip:

```python
import detekface

# 1. Live Webcam with 3D Pose HUD axes
detekface.camera(
    estimate_pose=True,
    show_pose=True,
)

# 2. Still-Image Head Pose
result = detekface.detect("portrait.jpg", estimate_pose=True)
for face in result.faces:
    if face.pose:
        print(f"Yaw  : {face.pose.yaw:+.1f}° (Left/Right)")
        print(f"Pitch: {face.pose.pitch:+.1f}° (Up/Down)")
        print(f"Roll : {face.pose.roll:+.1f}° (Tilt)")
        print(f"Facing Camera: {face.pose.is_frontal(tolerance=15.0)}")
```

👉 For full math, perspective-n-point projection, and limits, see the **[3D Head-Pose Guide](docs/pose.md)**.

---

## 20. Real-Time Facial Gesture Detection

Detect dynamic facial events in real time with configurable thresholds and event callbacks:
- **Blinks**: Eye Aspect Ratio (EAR) metric measuring eyelid closure.
- **Mouth Opening**: Mouth Aspect Ratio (MAR) metric detecting speaking or yawning.
- **Head Turns & Tilts**: Derived from temporal 3D head-pose Euler angles.

```python
import detekface


def on_gesture(event: detekface.GestureEvent) -> None:
    print(
        f"[Event] Face #{event.face_id}: {event.gesture.name} (value={event.value:.2f}, count={event.count})"
    )


detekface.camera(
    detect_gestures=True,
    on_gesture=on_gesture,
)
```

👉 For formulas, default thresholds, and examples, see the **[Facial Gestures Guide](docs/gestures.md)**.

---

## 21. AR Face Accessories Engine

Render augmented reality accessories overlaid onto detected faces with automatic scale, orientation rotation, and alpha blending:
- **`"sunglasses"`**: Classic dark eyewear anchored to eye centers.
- **`"crown"`**: Golden royal crown with jewels positioned above forehead.
- **`"mask"`**: Protective medical face mask anchored from nose bridge to chin.
- **`"visor"`**: High-tech sci-fi neon visor with futuristic HUD glow.
- **`"halo"`**: Glowing celestial angel halo hovering overhead.

```python
import detekface

# 1. Live Webcam AR Experience (press 'A' to cycle accessories live)
detekface.camera(
    effect="sunglasses",
)

# 2. Still Photo AR Processing
result = detekface.detect("portrait.jpg", effect="visor")
result.save("portrait_with_visor.jpg")
```

👉 For accessory anchors, vector graphics, and custom assets, see the **[AR Accessories Guide](docs/accessories.md)**.

---

## 22. Face Alignment Normalization

Straighten moderately tilted faces to standard horizontal eye-level alignment and produce aspect-preserving face crops ready for biometric embedding models, face recognition pipelines, or ID photo formatting:

```python
import detekface

# Align all faces in an image
result = detekface.detect("group_tilted.jpg")
aligned_faces = result.aligned_faces(
    output_size=(256, 256),  # Output crop resolution
    padding=0.25,  # Margin around facial bounds
)

for i, face in enumerate(aligned_faces):
    print(f"Face #{i}: angle={face.rotation_angle:.1f}°, size={face.image.shape}")
    face.save(f"aligned_face_{i}.jpg")
```

👉 For affine counter-rotation formulas and parameters, see the **[Face Alignment Guide](docs/alignment.md)**.

---

## 23. Technical Face Quality Assessment

Evaluate capture quality and technical suitability across blur/sharpness, lighting exposure, image framing, head pose, and landmark completeness:

```python
import detekface

# Assess technical capture quality without opening camera
quality = detekface.check_face_quality("id_photo.jpg")

print(f"Overall Suitability: {quality.suitability}")  # 'suitable', 'borderline', 'unsuitable'
print(f"Sharpness          : {quality.metrics.sharpness:.1f} (is_blurry={quality.metrics.is_blurry})")
print(f"Brightness         : {quality.metrics.brightness:.1f} (is_dark={quality.metrics.is_dark})")
print(f"Head Pose Frontal  : {quality.metrics.pose_suitable}")
print(f"Actionable Warnings: {quality.warnings}")
```

> [!NOTE]
> Face quality metrics measure technical image capture properties (sharpness, exposure, framing) for computer vision pipelines. They do not evaluate personal identity, aesthetics, or human worth.

👉 For metric thresholds and webcam HUD integration, see the **[Face Quality Guide](docs/quality.md)**.

---

## 24. Face Privacy Mode (Anonymization)

Obfuscate detected faces for privacy compliance, GDPR regulations, or public datasets with 100% local processing and fail-closed safety protection:

```python
import detekface

# Anonymize faces in a photo
report = detekface.anonymize(
    "crowd_photo.jpg",
    method="blur",  # 'blur', 'pixelate', or 'solid'
    strength=0.8,  # Obfuscation intensity (0.0 to 1.0)
    output_path="crowd_anonymized.jpg",
    fail_closed=True,  # Protects entire frame if privacy detection fails
)

print(
    f"Protected {report.faces_anonymized} face(s) in {report.output_path} (fail_closed={report.failed_closed})"
)
```

- **Gaussian Blur (`method="blur"`)**: Adaptive blur kernel scaled to face dimensions.
- **Pixelation Mosaic (`method="pixelate"`)**: Downsampled mosaic blocks for visual censorship.
- **Solid Masking (`method="solid"`)**: Opaque solid color rectangle (`mask_color=(0, 0, 0)`).
- **Source Protection**: Input files are **never** overwritten by default.
- **100% Local**: Zero network requests; zero data uploaded.

👉 For full privacy policies and API details, see the **[Privacy Filtering Guide](docs/privacy.md)**.

---

## 25. Decoupled 60 FPS Asynchronous Performance

DetekFace 0.3.0 features an asynchronous 4-stage decoupled pipeline (`async_mode=True`, active by default):

```
+-----------------------------------------------------------------------------------+
|                            DetekFace 4-Stage Engine                               |
+-----------------------------------------------------------------------------------+
| 1. Capture Thread   -> Ingests raw frames from hardware (up to 60 FPS)           |
| 2. Inference Thread -> Background DNN execution (YuNet / MediaPipe, 15-30ms)     |
| 3. Result Cache     -> Lock-free double-buffered detection cache (stale timeout)  |
| 4. Render Thread    -> High-framerate overlay composition & 60 FPS window display |
+-----------------------------------------------------------------------------------+
```

### 📊 Measured Benchmark Results (AMD64 12-Core, Windows 11)

| Benchmark Scenario | Capture FPS | Render FPS | Inference FPS | P50 Latency | P95 Latency | Memory (RSS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Basic Webcam Detection** | 30.0 | **61.2** | 33.8 | 29.5 ms | 30.9 ms | 205.8 MB |
| **B. Tactical HUD Overlays** | 30.0 | **60.9** | 33.7 | 29.6 ms | 31.0 ms | 206.1 MB |
| **C. Dense Facial Mesh** | 30.0 | **60.5** | 33.4 | 29.9 ms | 31.3 ms | 207.5 MB |
| **D. Face Shape Analysis** | 30.0 | **59.8** | 32.9 | 30.4 ms | 31.9 ms | 210.2 MB |
| **E. 3D Head-Pose Estimation** | 30.0 | **60.8** | 33.7 | 29.6 ms | 31.0 ms | 206.4 MB |
| **F. Facial Gesture Detection**| 30.0 | **60.7** | 33.6 | 29.7 ms | 31.1 ms | 206.6 MB |
| **G. AR Sunglasses & Visor** | 30.0 | **59.4** | 33.1 | 30.2 ms | 31.8 ms | 208.9 MB |
| **H. Multi-Face Tracking** | 30.0 | **60.6** | 33.5 | 29.8 ms | 31.2 ms | 207.1 MB |
| **I. Privacy Effects (Blur)** | 30.0 | **56.1** | 33.2 | 30.1 ms | 31.6 ms | 208.4 MB |
| **J. All Features Combined** | 30.0 | **54.8** | 31.8 | 31.4 ms | 33.2 ms | 212.8 MB |

👉 For full methodology, capture vs render FPS distinctions, and hardware limits, see the **[Performance Guide](docs/performance.md)**.

---

## 26. Configuration & Error Handling

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

## 27. Supported Platforms & Dependencies

| Component | Supported Versions | Tested Environment |
| :--- | :--- | :--- |
| **Python** | `>= 3.9`, `3.10`, `3.11`, `3.12`, `3.13`, `3.15` | `3.15.1 64-bit` |
| **NumPy** | `>= 1.20` | `2.5.4` |
| **OpenCV** | `>= 4.8` (`opencv-python` or `opencv-contrib-python`) | `5.0.0.93` |
| **Operating Systems** | Windows 11, Windows 10, Linux, macOS | Windows 11 (AMD64) |

---

## 28. Model Acquisition, Caching, & Checksums

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

## 29. Troubleshooting Common Issues

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

## 30. Testing & Performance Benchmarks

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

## 31. Quality Validation Suite

Execute the reproducible quality validation suite comparing backends across real-world challenge scenarios (no faces, distant faces, low light, harsh shadows, profile angles, occlusions):

```bash
python validation/download_dataset.py
python validation/evaluate.py --compare
```

See the **[Quality Validation Methodology Guide](docs/validation.md)** for mathematical metric definitions (Precision, Recall, F1, IoU), dataset licensing, and detailed backend tradeoffs.

---

## 32. Development Setup & Contributing

We welcome contributions!
1. Clone the repository: `git clone https://github.com/Abishek-C0DE5/DetekFace.git`
2. Install developer dependencies: `pip install -e ".[dev,mediapipe]"`
3. Run linter and formatter: `ruff check .` and `ruff format --check .`
4. Run type checker: `mypy`
5. Run unit tests: `pytest -v -m "not network and not optional_backend"`

Check out our **[Contributing Guide](CONTRIBUTING.md)** and **[Development Guide](docs/development.md)**.

---

## 33. Licensing & Credits

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
