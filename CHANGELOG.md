# Changelog

All notable changes to the **DetekFace** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-10-24

### Added
- **Core Detection Engine**:
  - High-performance face detection using OpenCV's deep learning `FaceDetectorYN` and official YuNet ONNX models.
  - Automatic model downloading, SHA256 checksum verification, and OS-appropriate user caching (`%LOCALAPPDATA%\detekface` on Windows, `~/.cache/detekface` on Linux/macOS).
  - Automatic model selection based on OpenCV runtime version (OpenCV 5.x graph engine vs OpenCV 4.x DNN).
- **Multiple Backend Architecture**:
  - Adapter-based backend architecture supporting the built-in default `"yunet"` backend and the optional `"mediapipe"` backend.
  - Optional dependency extra `detekface[mediapipe]` enabling Google BlazeFace model inference.
  - Normalized landmark output: YuNet returns 5 canonical points (eyes, nose tip, mouth corners); MediaPipe returns 6 keypoints (eyes, nose tip, mouth center, ears).
  - Strict error handling with `DependencyNotFoundError` providing exact installation hints without silent fallbacks.
- **Still Image API**:
  - `detekface.detect()` accepting file paths (`str`, `Path`) or in-memory NumPy BGR arrays.
  - Flexible detection tuning: `confidence_threshold`, `nms_threshold`, `top_k`, and `input_size` downscaling.
  - `DetectionResult` container with `.faces`, `.count`, `.has_faces`, `.image` (pristine BGR), and `.rgb_image` (RGB for PIL/Matplotlib).
  - `.show()` and `.save()` helper methods for interactive preview and annotated file export.
- **Batch Processing API**:
  - `detekface.detect_many()` for processing image collections or directories with recursive path discovery.
  - `BatchDetectionResult` and `BatchItemResult` preserving input order and tracking individual successes and failures.
  - `fail_fast=True/False` error handling and custom `progress_callback(completed, total)`.
  - Memory-efficient `keep_images=False` option for large datasets.
- **Live Webcam Integration**:
  - Beginner-friendly 2-line camera loop: `import detekface; detekface.camera()`.
  - Natural mirror flip (`mirror=True`), device index selection (`camera_index`), and customizable window titles.
  - Interactive keyboard controls: <kbd>Q</kbd> to quit cleanly, <kbd>Esc</kbd> for emergency exit, and <kbd>C</kbd> to crop the largest detected face.
- **Advanced Camera Session API**:
  - `detekface.open_camera()` and `CameraSession` context manager for explicit lifecycle control.
  - Frame-by-frame iterator `session.read()` yielding structured `DetectionResult` objects.
  - Headless camera operation for background services without preview windows.
  - Callback-based streaming with `session.stream(callback=...)`.
  - Programmatic face capture while camera is active via `session.capture()`.
- **Face Capture & Cropping**:
  - `detekface.capture()` extracting unpolluted face crops directly from original frames (never contaminated by drawn boxes).
  - Automatic largest-face selection or explicit `face_index` selection.
  - Automatic coordinate clamping to prevent index-out-of-bounds errors on border-touching faces.
  - In-memory cropping (`save=False`) and automatic disk persistence to customizable directories.
  - `on_capture` callback support across `camera()`, `open_camera()`, and `video()`.
- **Visual Overlay Styles & Telemetry**:
  - 5 distinct visual overlay styles: `"rectangle"`, `"corners"`, `"circle"`, `"rounded"`, and sci-fi `"hud"`.
  - Flexible color definitions accepting hex codes (`"#00FF88"`), named strings (`"cyan"`, `"neon_green"`), or RGB tuples.
  - Real-time telemetry overlays: rolling-window FPS counter, face count badges, and detection confidence tags.
- **Prerecorded Video Processing**:
  - `detekface.video()` for frame-by-frame video annotation with preserved dimensions and framerate.
  - Video frame skipping (`frame_skip=N`) for high-throughput video processing.
  - Headless video processing with `show=False`.
- **Command-Line Interface (CLI)**:
  - Lightweight console command `detekface` with subcommands: `camera`, `image`, `batch`, and `video`.
  - Full argument parsing, informative `--help` screens, and nonzero exit codes on error.
- **Automated Quality & CI Infrastructure**:
  - Comprehensive test suite (270+ tests) with webcam and GUI mocks.
  - Pytest markers separating ordinary offline unit tests (`not network and not optional_backend`), optional backend tests (`optional_backend`), and live model download tests (`network`).
  - GitHub Actions CI workflow covering Linux (Ubuntu), Windows, and macOS across Python 3.9 through 3.13.
  - Distribution building (`python -m build`), strict Twine metadata validation (`twine check --strict`), and wheel installation smoke tests.
  - Release workflow with PyPI Trusted Publishing (OIDC token exchange) without hardcoded secrets.
- **Validation Suite & Benchmarking**:
  - Reproducible detector evaluation script (`validation/evaluate.py`) measuring IoU, Precision, Recall, and F1 across scenario challenges.
  - Dataset manifest covering small faces, border proximities, lighting extremes, angles, and occlusions.
  - Hardware latency and throughput benchmarking script (`examples/benchmark.py`).
