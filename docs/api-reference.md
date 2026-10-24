# API Reference

Complete reference for all public functions, classes, data structures, constants, and exceptions in `detekface`.

---

## Table of Contents

- [Functions](#functions)
  - [detekface.camera](#detekfacecamera)
  - [detekface.detect](#detekfacedetect)
  - [detekface.capture](#detekfacecapture)
  - [detekface.video](#detekfacevideo)
- [Classes & Data Structures](#classes--data-structures)
  - [FaceDetector](#facedetector)
  - [DetectionResult](#detectionresult)
  - [FaceDetection](#facedetection)
  - [BoundingBox](#boundingbox)
  - [Point](#point)
  - [FaceLandmarks](#facelandmarks)
  - [FaceCapture](#facecapture)
  - [FPSCounter](#fpscounter)
- [Drawing & Utility Functions](#drawing--utility-functions)
  - [draw_faces](#draw_faces)
  - [draw_face_overlay](#draw_face_overlay)
  - [draw_metrics_overlay](#draw_metrics_overlay)
  - [parse_color_to_bgr](#parse_color_to_bgr)
  - [format_confidence](#format_confidence)
  - [format_fps](#format_fps)
  - [format_face_count](#format_face_count)
- [Model Management](#model-management)
  - [resolve_model](#resolve_model)
  - [get_cache_dir](#get_cache_dir)
  - [ModelMetadata](#modelmetadata)
- [Constants](#constants)
- [Exceptions](#exceptions)

---

## Functions

### `detekface.camera`

```python
def camera(
    camera_index: int = 0,
    mirror: bool = True,
    shape: str = "rectangle",
    color: Union[str, Sequence[int]] = "#00FF88",
    thickness: int = 2,
    show_confidence: bool = False,
    show_fps: bool = False,
    show_count: bool = False,
    on_capture: Optional[Callable[[FaceCapture], None]] = None,
    save_dir: Union[str, Path] = "captures",
    confidence_threshold: float = 0.7,
    detector: Optional[FaceDetector] = None,
    window_name: str = "DetekFace - Live Webcam",
    backend: str = "yunet",
) -> None:
    ...
```

Opens a live webcam feed, runs real-time face detection on each frame, draws customizable overlays, and handles interactive keyboard commands.

#### Parameters
- **`camera_index`** (`int`, default: `0`): OpenCV device index for the video capture hardware.
- **`mirror`** (`bool`, default: `True`): If `True`, flips the webcam image horizontally for a natural mirror experience.
- **`shape`** (`str`, default: `"rectangle"`): Overlay style. Options: `"rectangle"`, `"corners"`, `"circle"`, `"rounded"`, `"hud"`.
- **`color`** (`Union[str, Sequence[int]]`, default: `"#00FF88"`): Hex code, named color string, or `(R, G, B)` tuple.
- **`thickness`** (`int`, default: `2`): Stroke width in pixels for drawn shapes. Must be `>= 1`.
- **`show_confidence`** (`bool`, default: `False`): If `True`, displays detection confidence percentage near each face.
- **`show_fps`** (`bool`, default: `False`): If `True`, displays real smoothed FPS in the status banner.
- **`show_count`** (`bool`, default: `False`): If `True`, displays detected face count in the status banner.
- **`on_capture`** (`Optional[Callable[[FaceCapture], None]]`, default: `None`): Callback invoked when user presses `C` and a face is captured.
- **`save_dir`** (`Union[str, Path]`, default: `"captures"`): Directory to save cropped face images upon `C` keypress.
- **`confidence_threshold`** (`float`, default: `0.7`): Minimum confidence threshold for detection.
- **`detector`** (`Optional[FaceDetector]`, default: `None`): Existing detector instance to reuse.
- **`window_name`** (`str`, default: `"DetekFace - Live Webcam"`): Title of the OpenCV display window.
- **`backend`** (`str`, default: `"yunet"`): Detection backend (`"yunet"` or `"mediapipe"`).

#### Keyboard Controls
- **`Q`** or **`Escape`**: Quit and close the camera preview.
- **`C`**: Capture and crop the largest detected face in the current frame.

#### Raises
- `CameraNotFoundError`: Camera index cannot be opened or device is not found.
- `CameraReadError`: Camera opened but failed to read initial video frames.
- `ConfigurationError`: Invalid style name, color definition, or non-positive thickness.
- `DependencyNotFoundError`: Selected optional backend (e.g. MediaPipe) is not installed.

---

### `detekface.open_camera`

```python
def open_camera(
    camera_index: int = 0,
    mirror: bool = True,
    confidence_threshold: float = 0.7,
    detector: Optional[FaceDetector] = None,
    backend: str = "yunet",
    window_title: Optional[str] = None,
    auto_open: bool = True,
) -> CameraSession:
    ...
```

Opens and returns a managed `CameraSession` context manager for headless processing, frame iteration, custom pipelines, and programmatic face capture.

#### Example Usage

```python
import detekface

with detekface.open_camera(camera_index=0) as session:
    for result in session:
        print(f"Detected {result.count} faces")
        if result.has_faces:
            captured = session.capture()
            print(f"Face crop saved to {captured.path}")
            break
```

---

### `detekface.detect`

```python
def detect(
    image: Union[str, Path, np.ndarray],
    confidence_threshold: float = 0.7,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
    model: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    backend: str = "yunet",
) -> DetectionResult:
    ...
```

Detects human faces in a still image or raw NumPy BGR array.

#### Parameters
- **`image`** (`Union[str, Path, np.ndarray]`): File path to an image or an OpenCV BGR NumPy array (`uint8`).
- **`confidence_threshold`** (`float`, default: `0.7`): Minimum confidence score threshold (`0.0` to `1.0`).
- **`nms_threshold`** (`float`, default: `0.3`): Non-Maximum Suppression IoU threshold (`0.0` to `1.0`).
- **`top_k`** (`int`, default: `5000`): Maximum candidate face proposals evaluated before NMS.
- **`model`** (`Optional[Union[str, Path]]`, default: `None`): Model identifier or path to custom `.onnx` model.
- **`cache_dir`** (`Optional[Union[str, Path]]`, default: `None`): Custom directory for caching model weights.
- **`backend`** (`str`, default: `"yunet"`): Detection backend (`"yunet"` or `"mediapipe"`).

#### Returns
- `DetectionResult`: Dataclass containing detected faces and the unannotated original BGR image.

#### Raises
- `ImageNotFoundError`: Specified image path does not exist.
- `InvalidImageError`: Corrupted image or invalid NumPy array structure.
- `ConfigurationError`: Threshold values outside valid range.

---

### `detekface.capture`

```python
def capture(
    result: DetectionResult,
    face_index: Optional[int] = None,
    save_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
    save: bool = True,
) -> FaceCapture:
    ...
```

Crops a detected face from a `DetectionResult` using the original, unannotated image frame.

#### Parameters
- **`result`** (`DetectionResult`): Result instance containing faces and original image.
- **`face_index`** (`Optional[int]`, default: `None`): Index of face to crop. If `None`, selects the largest face by area.
- **`save_dir`** (`Optional[Union[str, Path]]`, default: `"captures"`): Target directory for saving cropped file.
- **`filename`** (`Optional[str]`, default: `None`): Custom filename. If `None`, uses timestamped name.
- **`save`** (`bool`, default: `True`): If `False`, skips writing to disk and returns crop in memory only.

#### Returns
- `FaceCapture`: Cropped face object with image arrays and metadata.

#### Raises
- `InvalidImageError`: `DetectionResult` does not contain a valid image array.
- `NoFaceDetectedError`: No faces were detected in `result`.
- `IndexError`: `face_index` is outside the range of detected faces.
- `InvalidCropError`: Clamped bounding box produces an empty crop.
- `CaptureSaveError`: Failed to write crop to disk.

---

### `detekface.video`

```python
def video(
    source: Union[str, Path],
    output: Optional[Union[str, Path]] = None,
    show: bool = True,
    shape: str = "rectangle",
    color: Union[str, Sequence[int]] = "#00FF88",
    thickness: int = 2,
    show_confidence: bool = False,
    show_fps: bool = False,
    show_count: bool = False,
    confidence_threshold: float = 0.7,
    detector: Optional[FaceDetector] = None,
    window_name: str = "DetekFace - Video File",
    backend: str = "yunet",
) -> int:
    ...
```

Processes a prerecorded video file frame-by-frame, rendering detection overlays and optionally exporting an annotated video.

#### Parameters
- **`source`** (`Union[str, Path]`): File path to input video file.
- **`output`** (`Optional[Union[str, Path]]`, default: `None`): Destination path for saving annotated video (`.mp4`, `.avi`).
- **`show`** (`bool`, default: `True`): Whether to display playback window during processing.
- **`shape`** (`str`, default: `"rectangle"`): Overlay style name.
- **`color`** (`Union[str, Sequence[int]]`, default: `"#00FF88"`): Color definition.
- **`thickness`** (`int`, default: `2`): Stroke width in pixels.
- **`show_confidence`** (`bool`, default: `False`): Display confidence badges.
- **`show_fps`** (`bool`, default: `False`): Display smoothed FPS overlay.
- **`show_count`** (`bool`, default: `False`): Display detected face count.
- **`confidence_threshold`** (`float`, default: `0.7`): Detection threshold.
- **`detector`** (`Optional[FaceDetector]`, default: `None`): Reusable detector instance.
- **`window_name`** (`str`, default: `"DetekFace - Video File"`): Title for display window.
- **`backend`** (`str`, default: `"yunet"`): Detection backend (`"yunet"` or `"mediapipe"`).

#### Returns
- `int`: Total number of frames successfully processed.

#### Raises
- `VideoNotFoundError`: Video file does not exist.
- `InvalidVideoError`: Unsupported video format or damaged file.
- `VideoWriteError`: Failure to initialize output video writer or write frames.
- `DependencyNotFoundError`: Selected optional backend (e.g. MediaPipe) is not installed.

---

## Classes & Data Structures

### `CameraSession`

Managed camera session providing lifecycle management, frame iteration, programmatic face capture, and optional GUI rendering.

```python
class CameraSession:
    def __init__(
        self,
        camera_index: int = 0,
        mirror: bool = True,
        confidence_threshold: float = 0.7,
        detector: Optional[FaceDetector] = None,
        backend: str = "yunet",
        window_title: Optional[str] = None,
        auto_open: bool = True,
    ) -> None: ...

    @property
    def is_open(self) -> bool: ...
    @property
    def is_active(self) -> bool: ...
    @property
    def latest_frame(self) -> Optional[np.ndarray]: ...
    @property
    def latest_result(self) -> Optional[DetectionResult]: ...
    @property
    def fps(self) -> float: ...
    @property
    def frames_read(self) -> int: ...

    def open(self) -> "CameraSession": ...
    def close(self) -> None: ...
    def read(self) -> DetectionResult: ...
    def read_raw(self) -> np.ndarray: ...
    def stream(self, max_frames: Optional[int] = None) -> Iterator[DetectionResult]: ...
    def process(self, callback: Callable[[DetectionResult], Optional[bool]], max_frames: Optional[int] = None) -> int: ...
    def capture(self, face_index: Optional[int] = None, save_dir: Optional[Union[str, Path]] = "captures", filename: Optional[str] = None, save: bool = True) -> FaceCapture: ...
    def render(self, shape: str = "rectangle", color: Union[str, Sequence[int]] = "#00FF88", thickness: int = 3, show_confidence: bool = False, show_fps: bool = False, show_count: bool = False) -> np.ndarray: ...
    def show(self, shape: str = "rectangle", color: Union[str, Sequence[int]] = "#00FF88", thickness: int = 3, show_confidence: bool = False, show_fps: bool = False, show_count: bool = False, window_title: Optional[str] = None, wait_ms: int = 1) -> int: ...
```

#### Frame Ownership & Memory Semantics
- **Read & Ingestion**: Each call to `read()` retrieves the latest frame from the video capture device and flips it horizontally if `mirror=True`.
- **Pristine Clean Frame**: A separate copy (`clean_frame = frame.copy()`) is stored in `latest_frame` and provided as `DetectionResult.image`.
- **Consumer Ownership**: The consumer owns `DetectionResult.image`. It is never mutated or drawn over by overlay routines.
- **Overlays & Rendering**: When `session.render()` or `session.show()` is called, overlays are rendered on a newly allocated display copy, guaranteeing that face captures remain unpolluted.

---

### `FaceDetector`

Detection engine supporting interchangeable backends (`"yunet"` default, `"mediapipe"` optional).

```python
class FaceDetector:
    def __init__(
        self,
        model: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        cache_dir: Optional[Union[str, Path]] = None,
        backend: str = "yunet",
    ) -> None: ...

    @property
    def backend_name(self) -> str: ...

    @property
    def backend(self) -> BaseFaceBackend: ...

    def detect(self, image: Union[str, Path, np.ndarray]) -> DetectionResult: ...
    def detect_frame(self, frame: np.ndarray) -> List[FaceDetection]: ...
```

---

### `DetectionResult`

Container for complete detection results on a single image.

```python
@dataclass
class DetectionResult:
    image: np.ndarray
    faces: List[FaceDetection] = field(default_factory=list)

    @property
    def count(self) -> int: ...
    @property
    def has_faces(self) -> bool: ...
    @property
    def rgb_image(self) -> np.ndarray: ...

    def show(self, title: str = "DetekFace Detection", shape: str = "rectangle",
             color: Union[str, Sequence[int]] = "#00FF88", thickness: int = 2,
             show_confidence: bool = False, show_count: bool = False) -> None: ...

    def save(self, path: Union[str, Path], shape: str = "rectangle",
             color: Union[str, Sequence[int]] = "#00FF88", thickness: int = 2,
             show_confidence: bool = False, show_count: bool = False) -> Path: ...

    def render(self, shape: str = "rectangle", color: Union[str, Sequence[int]] = "#00FF88",
               thickness: int = 2, show_confidence: bool = False,
               show_count: bool = False) -> np.ndarray: ...

    def capture(self, face_index: Optional[int] = None,
                save_dir: Optional[Union[str, Path]] = None,
                filename: Optional[str] = None, save: bool = True) -> FaceCapture: ...
```

---

### `FaceDetection`

Information representing an individual detected face.

```python
@dataclass
class FaceDetection:
    box: BoundingBox
    confidence: float
    landmarks: Optional[FaceLandmarks] = None
```

---

### `BoundingBox`

Pixel coordinates and dimensions of a detected region.

```python
@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int: ...
    @property
    def center(self) -> Point: ...
    @property
    def x2(self) -> int: ...  # x + width
    @property
    def y2(self) -> int: ...  # y + height

    def as_tuple(self) -> Tuple[int, int, int, int]: ...
    def as_ltrb(self) -> Tuple[int, int, int, int]: ...
    def clamp(self, max_width: int, max_height: int) -> BoundingBox: ...
```

---

### `Point`

2D coordinate structure.

```python
@dataclass
class Point:
    x: float
    y: float

    def as_tuple(self) -> Tuple[float, float]: ...
    def as_int_tuple(self) -> Tuple[int, int]: ...
```

---

### `FaceLandmarks`

Facial feature landmarks normalized across backends. Landmarks not supplied by a specific backend remain `None` (never artificially invented).

```python
@dataclass
class FaceLandmarks:
    left_eye: Point
    right_eye: Point
    nose_tip: Point
    mouth_left: Optional[Point] = None      # Provided by YuNet; None for MediaPipe
    mouth_right: Optional[Point] = None     # Provided by YuNet; None for MediaPipe
    mouth_center: Optional[Point] = None    # Provided by MediaPipe; None for YuNet
    right_ear: Optional[Point] = None       # Provided by MediaPipe; None for YuNet
    left_ear: Optional[Point] = None        # Provided by MediaPipe; None for YuNet

    def as_list(self) -> List[Point]: ...
    def as_tuples(self) -> List[Tuple[float, float]]: ...
```

#### Landmark Supply by Backend

| Landmark | YuNet (Default) | MediaPipe (Optional) | Description |
| :--- | :---: | :---: | :--- |
| `right_eye` | Yes | Yes | Center of subject's right eye |
| `left_eye` | Yes | Yes | Center of subject's left eye |
| `nose_tip` | Yes | Yes | Apex / tip of nose |
| `right_mouth` / `mouth_right` | Yes | No (`None`) | Right corner of mouth |
| `left_mouth` / `mouth_left` | Yes | No (`None`) | Left corner of mouth |
| `mouth_center` | No (`None`) | Yes | Center point between lips |
| `right_ear` | No (`None`) | Yes | Right ear tragion |
| `left_ear` | No (`None`) | Yes | Left ear tragion |

---

### `FaceCapture`

Cropped face image result and associated metadata.

```python
@dataclass
class FaceCapture:
    image: np.ndarray
    box: BoundingBox
    confidence: float
    path: Optional[Path] = None
    face: Optional[FaceDetection] = None

    @property
    def rgb_image(self) -> np.ndarray: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...

    def save(self, destination: Union[str, Path]) -> Path: ...
    def show(self, title: str = "Captured Face") -> None: ...
```

---

### `FPSCounter`

Sliding-window frames-per-second calculator.

```python
class FPSCounter:
    def __init__(self, window_size: int = 30) -> None: ...
    def tick(self, current_time: Optional[float] = None) -> float: ...
    @property
    def fps(self) -> float: ...
    def reset(self) -> None: ...
```

---

## Detection Backends & Adapters

`detekface` decouples the detection engine behind a clean adapter interface (`BaseFaceBackend`).

- **`BaseFaceBackend`**: Abstract base class for all backend adapters.
- **`YuNetBackend`**: Default OpenCV `FaceDetectorYN` adapter (zero extra dependencies).
- **`MediaPipeBackend`**: Optional Google MediaPipe BlazeFace adapter (`pip install "detekface[mediapipe]"`).
- **`get_backend(name, confidence_threshold, ...)`**: Factory function instantiating backend adapters.
- **`SUPPORTED_BACKENDS`**: `("yunet", "mediapipe")`
- **`DEFAULT_BACKEND`**: `"yunet"`

---

## Drawing & Utility Functions

- **`draw_faces(image, faces, shape, color, thickness, show_confidence)`**: Draws overlays on an image array.
- **`draw_face_overlay(image, face, shape, color, thickness, show_confidence)`**: Draws a single face overlay.
- **`draw_metrics_overlay(image, count, fps, color)`**: Draws top status banner with face count and/or FPS.
- **`parse_color_to_bgr(color)`**: Converts hex, named colors, or RGB tuples to OpenCV BGR `(B, G, R)`.
- **`format_confidence(confidence)`**: Returns formatted string (e.g. `"95.2%"`).
- **`format_fps(fps)`**: Returns formatted string (e.g. `"FPS: 54.1"`).
- **`format_face_count(count)`**: Returns formatted string (e.g. `"Faces: 2"`).

---

## Model Management

- **`resolve_model(model_name_or_path, cache_dir)`**: Resolves YuNet model name or local path, downloading and caching official weights if needed.
- **`resolve_mediapipe_model(model_name_or_path, cache_dir)`**: Resolves MediaPipe BlazeFace model weights.
- **`get_cache_dir()`**: Returns path to the local cache directory (`%LOCALAPPDATA%\detekface` on Windows).
- **`ModelMetadata`**: Dataclass with model name, filename, sha256 checksum, and download URL.

---

## Constants

- **`SUPPORTED_BACKENDS`**: `("yunet", "mediapipe")`
- **`DEFAULT_BACKEND`**: `"yunet"`
- **`SUPPORTED_SHAPES`**: `{"rectangle", "corners", "circle", "rounded", "hud"}`
- **`NAMED_COLORS`**: Dict mapping color names (`"green"`, `"neon_green"`, `"red"`, `"blue"`, `"cyan"`, etc.) to RGB tuples.
- **`SUPPORTED_IMAGE_EXTENSIONS`**: `{".jpg", ".jpeg", ".png", ".webp", ".bmp"}`
- **`SUPPORTED_VIDEO_EXTENSIONS`**: `{".mp4", ".avi", ".mov", ".mkv", ".webm"}`

---

## Exceptions

All exceptions derive from `DetekFaceError` and include actionable `.message` and `.hint` attributes:

```
DetekFaceError (Base)
├── ConfigurationError
│   └── InvalidSettingError
├── DependencyNotFoundError (raised when an optional backend extra is not installed)
├── ModelError
│   ├── ModelNotFoundError
│   ├── ModelDownloadError
│   ├── ModelCorruptionError
│   └── ModelCompatibilityError
├── ImageError
│   ├── ImageNotFoundError
│   ├── InvalidImageError
│   └── ImageSaveError
├── CameraError
│   ├── CameraNotFoundError
│   └── CameraReadError
├── CaptureError
│   ├── NoFaceDetectedError
│   ├── InvalidCropError
│   └── CaptureSaveError
└── VideoError
    ├── VideoNotFoundError
    ├── InvalidVideoError
    └── VideoWriteError
```
