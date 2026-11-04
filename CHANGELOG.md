# Changelog

All notable changes to the **DetekFace** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-11-04

### Added
- **Dense Facial Mesh Visualization Engine**:
  - `FaceMeshRenderer` and `draw_face_meshes()` providing topological wireframe visualization across 478 dense surface facial landmarks.
  - 5 customizable mesh styles:
    - `"wireframe"`: Full connected Delaunay-style topological triangulation.
    - `"neon"`: High-contrast glowing lines, emphasized contours, and highlighted anatomical anchor keypoints (irises, nose bridge, lips).
    - `"minimal"`: Clean, subtle structural mesh lines with minimal visual clutter.
    - `"contour"`: Silhouettes only (facial oval, eye rings, nose ridge, lip contours).
    - `"points"`: Discrete vertex dots rendering landmark locations without line connections.
  - Fully customizable line thickness, vertex radii, and feature region filters (`contours`, `eyes`, `eyebrows`, `nose`, `lips`).
  - Anatomical coordinate mapping safely bound to arbitrary image dimensions.
  - Clear architectural documentation distinguishing 2D planar perspective projections from volumetric craniofacial medical reconstructions.
- **Animated Live FX Computer-Vision Scanner**:
  - `LiveFXRenderer` driving modern computer-vision HUD animations for live video feeds and snapshots.
  - Continuous sinusoidal laser beam sweep driven by `time.perf_counter()` clock timestamps (strictly frame-rate independent).
  - Converging HUD corner brackets and reticle lock animations transitioning smoothly from acquiring to locked states.
  - Real-time telemetry badges: `[FACE LOCKED #ID]`, `[SCANNING #ID]`, and biometric shape tags.
  - Multi-face tracking with exponential coordinate smoothing (`alpha = 0.65`) and 0.25-second occlusion grace period recovery.
- **Biometric Face Shape Analysis Engine**:
  - Real-time and still-image geometric face-shape classifier (`analyze_face_shape=True`).
  - Measures key facial proportions: face length-to-width ratio, forehead-to-jaw ratio, cheekbone-to-jaw ratio, and jaw angles.
  - Heuristic geometric fit scoring across 6 canonical shapes: `oval`, `round`, `square`, `heart`, `diamond`, `oblong`, and `undetermined`.
  - Temporal smoothing filter (`TemporalShapeTracker`) stabilizing classifications across webcam frames to eliminate flickering.
- **Webcam & Image API Integration**:
  - Enhanced `detekface.camera()` with `show_mesh`, `mesh_style`, `show_contour`, `show_scan`, `live_fx`, and `live_fx_style`.
  - Interactive keyboard controls in webcam window:
    - <kbd>M</kbd> to cycle live through facial mesh styles.
    - <kbd>F</kbd> to cycle live through Live FX scanner styles.
    - <kbd>C</kbd> to capture pristine unannotated face crops.
  - Still-image support in `detekface.detect(image, mesh=True)` and unified rendering parity in `DetectionResult.render()`, `.annotate()`, `.draw()`, and `.save()`.
  - Seamless multi-face support rendering independent wireframes and tracking identities without cross-interference.
- **CLI Enhancements**:
  - Added `--mesh`, `--mesh-style`, `--live-fx`, `--live-fx-style`, and `--face-shape` flags to `detekface camera`.
- **Documentation & Examples**:
  - Created comprehensive `docs/facial-mesh.md` guide and `docs/face-shape.md`.
  - Added 5 complete, runnable example scripts in `examples/`:
    - `face_mesh_webcam.py`: Basic live wireframe mesh feed.
    - `neon_mesh_webcam.py`: Glowing neon mesh with anchor points.
    - `live_scan_webcam.py`: Animated computer-vision laser scanner.
    - `still_image_mesh.py`: Still-photo mesh extraction and image saving.
    - `multi_face_mesh.py`: Independent multi-face mesh visualization.

### Changed
- **Performance & Optimization**:
  - Optimized `DetectionResult.render()` to eliminate redundant image copies and draw directly onto the primary canvas copy.
  - Cached topological connection tables and singleton MediaPipe Face Landmarker model loader.
  - Lazy loading: landmark extractor model is only initialized when mesh or face-shape analysis is requested. Standard YuNet face detection incurs zero landmark overhead.
  - Enhanced resource cleanup in `CameraSession.close()` ensuring internal tracking state, memory buffers, and temporary references are freed cleanly.

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
