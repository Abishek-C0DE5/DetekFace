# Image Detection Guide

`detekface` provides a streamlined, Pythonic interface for detecting faces in still images. Whether working with image files stored on disk or in-memory NumPy arrays from OpenCV, PIL, or scikit-image, `detekface.detect()` returns structured dataclasses with exact coordinates, facial landmarks, and confidence metrics.

---

## Quick Example

```python
import detekface

# 1. Detect faces in a still photograph
result = detekface.detect("people.jpg")

# 2. Inspect detection summary
print(f"Detected {result.count} face(s)")

# 3. Save an annotated image with visual bounding boxes
result.save("people_detected.jpg", shape="hud", color="#00FF88")

# 4. Display annotated window on desktop (press any key to dismiss)
result.show(title="Face Detection Preview")
```

---

## Core Concepts & Terminology

If you are new to computer vision, here are the fundamental terms used across `detekface`:

- **Bounding Box (`box`)**: A rectangular region defining the location and scale of a detected face. In `detekface`, coordinates are represented as `(x, y, width, height)` where:
  - `x`: Horizontal pixel offset from the top-left corner of the image.
  - `y`: Vertical pixel offset from the top-left corner of the image.
  - `width`: Horizontal span in pixels.
  - `height`: Vertical span in pixels.
- **Confidence Score (`confidence`)**: A floating-point value ranging from `0.0` (0%) to `1.0` (100%) indicating how certain the YuNet neural network is that the detected region represents a human face. By default, detections below `0.7` (70%) are discarded.
- **Facial Landmarks (`landmarks`)**: Distinct geometric feature points identified on the face. YuNet detects five canonical 2D coordinates:
  1. `left_eye`: Center of the subject's right eye (viewer's left side).
  2. `right_eye`: Center of the subject's left eye (viewer's right side).
  3. `nose_tip`: Tip of the nose.
  4. `mouth_left`: Left corner of the mouth.
  5. `mouth_right`: Right corner of the mouth.
- **Color Format (BGR vs RGB)**: OpenCV loads and manipulates images in **BGR** (Blue, Green, Red) channel order. Standard Python libraries such as Pillow and matplotlib use **RGB** (Red, Green, Blue). `detekface` internally maintains the native BGR format for zero-overhead OpenCV processing, but provides convenient `.rgb_image` properties when integrating with Pillow or matplotlib.

---

## The `detekface.detect()` Function

```python
def detect(
    image: Union[str, Path, np.ndarray],
    confidence_threshold: float = 0.7,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
    model: Optional[Union[str, Path]] = None,
    backend: str = "yunet",
    cache_dir: Optional[Union[str, Path]] = None,
    input_size: Optional[Tuple[int, int]] = None,
) -> DetectionResult: ...
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `str`, `Path`, `np.ndarray` | *Required* | File path to an image (`.jpg`, `.png`, `.webp`, `.bmp`) or an existing OpenCV BGR NumPy array (`uint8`). |
| `confidence_threshold` | `float` | `0.7` | Filtering threshold between `0.0` and `1.0`. Only candidate faces with scores equal to or higher than this value are returned. |
| `nms_threshold` | `float` | `0.3` | Non-Maximum Suppression (NMS) intersection-over-union (IoU) threshold used to suppress duplicate overlapping boxes. |
| `top_k` | `int` | `5000` | Maximum candidate proposals evaluated before NMS filtering. |
| `model` | `str`, `Path`, `None` | `None` | Custom local model path or official model alias. Defaults to the official YuNet release. |
| `backend` | `str` | `"yunet"` | Detection backend: `"yunet"` (default) or `"mediapipe"`. |
| `cache_dir` | `str`, `Path`, `None` | `None` | Custom directory for downloading and caching the ONNX model file. |
| `input_size` | `Optional[Tuple[int, int]]` | `None` | Optional `(width, height)` downscaling resolution for model inference. Coordinates are automatically rescaled to original dimensions! |

---

## Working with `DetectionResult`

The `detekface.detect()` function returns a `DetectionResult` dataclass instance containing the complete detection context:

```python
result = detekface.detect("team_photo.jpg")
```

### Properties

| Property | Type | Description |
| :--- | :--- | :--- |
| `result.count` | `int` | Number of detected faces (`len(result.faces)`). |
| `result.has_faces` | `bool` | Convenience boolean: `True` if `count > 0`, else `False`. |
| `result.faces` | `List[FaceDetection]` | List of individual face detection records ordered by detector proposal rank. |
| `result.image` | `np.ndarray` | The **unmodified original image** in OpenCV BGR format. Drawing or saving never pollutes this array. |
| `result.rgb_image` | `np.ndarray` | A copy of the original image converted to standard RGB format. |

### Methods

#### `result.show(...)`
Displays the image with rendered detection overlays in an interactive OpenCV desktop window:
```python
result.show(
    title="Face Detection",
    shape="rounded",
    color="cyan",
    thickness=2,
    show_confidence=True,
    show_count=True,
)
```
- Blocks until the user presses any key or clicks the window close button.
- Cleanly closes the OpenCV window upon exit.

#### `result.save(...)`
Renders detection overlays and writes the resulting annotated image to disk:
```python
result.save("output_annotated.jpg", shape="hud", color="#00FF88")
```

#### `result.render(...)`
Produces and returns an annotated copy of the image as a NumPy array without showing or writing to disk:
```python
annotated_bgr = result.render(shape="corners", color="yellow")
```

#### `result.capture(...)`
Convenience shortcut that passes `self` to `detekface.capture()` to crop detected faces:
```python
crop = result.capture(save_dir="avatars")
```

---

## Inspecting Individual Faces (`FaceDetection`)

Each face inside `result.faces` is an instance of `FaceDetection`:

```python
for idx, face in enumerate(result.faces):
    print(f"Face #{idx + 1}:")
    print(f"  Confidence: {face.confidence:.1%}")
    print(f"  Bounding Box: x={face.box.x}, y={face.box.y}, w={face.box.width}, h={face.box.height}")
    print(f"  Area: {face.box.area} pixels")
    print(f"  Center: ({face.box.center.x:.1f}, {face.box.center.y:.1f})")

    # Access 5 facial landmarks
    if face.landmarks is not None:
        lm = face.landmarks
        print(f"  Left Eye:  ({lm.left_eye.x}, {lm.left_eye.y})")
        print(f"  Right Eye: ({lm.right_eye.x}, {lm.right_eye.y})")
        print(f"  Nose Tip:  ({lm.nose_tip.x}, {lm.nose_tip.y})")
        print(f"  Mouth L/R: ({lm.mouth_left.x}, {lm.mouth_left.y}) / ({lm.mouth_right.x}, {lm.mouth_right.y})")
```

### BoundingBox Helper Methods

The `face.box` object (`BoundingBox`) includes built-in geometric utilities:

- `face.box.as_tuple()`: Returns `(x, y, width, height)`.
- `face.box.as_ltrb()`: Returns `(left, top, right, bottom)` for compatibility with libraries expecting LTRB formats.
- `face.box.clamp(max_width, max_height)`: Returns a new `BoundingBox` clipped to ensure coordinates do not exceed valid image borders.

---

## Input Types & Integration Examples

### 1. In-Memory OpenCV NumPy Array
```python
import cv2
import detekface

# Read frame from custom pipeline
frame = cv2.imread("sample.jpg")

# Pass directly to detect()
result = detekface.detect(frame)
```

### 2. Integration with Pillow (PIL)
```python
from PIL import Image
import numpy as np
import detekface

# Open with Pillow and convert to numpy array (RGB)
pil_img = Image.open("portrait.png")
rgb_array = np.array(pil_img)

# Convert RGB to BGR for detekface
bgr_array = rgb_array[:, :, ::-1]

result = detekface.detect(bgr_array)
print(f"Found {result.count} face(s)")
```

### 3. Integration with Matplotlib
```python
import matplotlib.pyplot as plt
import detekface

result = detekface.detect("family.jpg")

# Render with desired style
annotated_bgr = result.render(shape="hud", color="#00FF88")

# Convert to RGB for matplotlib display
annotated_rgb = annotated_bgr[:, :, ::-1]

plt.figure(figsize=(10, 6))
plt.imshow(annotated_rgb)
plt.axis("off")
plt.title(f"DetekFace - {result.count} Face(s) Detected")
plt.show()
```

---

## Error Handling

`detekface` provides explicit, descriptive exceptions to simplify debugging:

```python
from detekface import detect
from detekface.exceptions import ImageNotFoundError, InvalidImageError, ImageSaveError

try:
    result = detect("missing_file.jpg")
except ImageNotFoundError as err:
    print(f"Error: {err.message}")
    print(f"Hint: {err.hint}")
except InvalidImageError as err:
    print(f"Corrupted or unsupported image: {err.message}")
```

| Exception | Root Cause |
| :--- | :--- |
| `ImageNotFoundError` | The specified file path does not exist on the filesystem. |
| `InvalidImageError` | The file is empty, corrupted, not an image, or an invalid NumPy array. |
| `ImageSaveError` | Writing the annotated image failed due to permission denial or invalid directory path. |
| `ConfigurationError` | Invalid parameter values (e.g. `confidence_threshold < 0.0` or unknown shape). |
