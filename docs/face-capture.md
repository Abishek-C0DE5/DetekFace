# Face Capture & Cropping Guide

`detekface` includes dedicated functionality for extracting high-resolution, unpolluted face crops from still images and live webcam streams. 

Whether building avatar generators, biometric dataset pipelines, or authentication systems, `detekface.capture()` guarantees that face crops are extracted directly from the original pixel source—completely free of drawn bounding boxes or interface overlays.

---

## Key Features

1. **Pure Original Source**: Crops are sliced from the original, unannotated camera frame or image array.
2. **Safe Clamping**: Coordinates are automatically clamped to valid image boundaries to prevent out-of-bounds indexing errors and reject empty crops.
3. **Smart Default Selection**: When multiple faces are present, `detekface` automatically selects the largest face by bounding-box area (most prominent subject), with support for explicit indexing.
4. **Flexible Storage**: Save crops to disk automatically, specify custom directories/filenames, or work entirely in memory (`save=False`).
5. **Interactive & Programmatic Support**: Capture faces during live video previews with the `C` key, or programmatically via Python scripts.

---

## Programmatic Capture API

The primary entry point for cropping faces is `detekface.capture()`, which accepts a `DetectionResult`:

```python
import detekface

# 1. Detect faces in an image
result = detekface.detect("headshot.jpg")

# 2. Capture the most prominent face
capture = detekface.capture(result, save_dir="avatars")

print(f"Captured face saved to: {capture.path}")
print(f"Resolution: {capture.width}x{capture.height} px")
print(f"Detection confidence: {capture.confidence:.1%}")
```

Alternatively, use the convenience method on the `DetectionResult` instance:

```python
capture = result.capture(save_dir="avatars")
```

---

## Function Signature: `detekface.capture()`

```python
def capture(
    result: DetectionResult,
    face_index: Optional[int] = None,
    save_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
    save: bool = True,
) -> FaceCapture: ...
```

### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `result` | `DetectionResult` | *Required* | A valid detection result returned by `detekface.detect()` containing detected faces and the original BGR frame. |
| `face_index` | `int`, `None` | `None` | Zero-based index of a specific face in `result.faces`. If `None`, automatically selects the largest face by pixel area. |
| `save_dir` | `str`, `Path`, `None` | `"captures"` | Target directory where the cropped JPEG file will be saved. Directory is automatically created if it does not exist. |
| `filename` | `str`, `None` | `None` | Custom output filename (e.g., `"user_profile.jpg"`). If `None`, generates a timestamped name: `capture_YYYYMMDD_HHMMSS_ffffff.jpg`. |
| `save` | `bool` | `True` | Set to `False` to perform in-memory cropping only without writing to disk. |

---

## The `FaceCapture` Object

The return value of `capture()` is a `FaceCapture` dataclass providing access to both disk paths and in-memory arrays:

| Property / Method | Type | Description |
| :--- | :--- | :--- |
| `capture.image` | `np.ndarray` | BGR cropped image array (`uint8`). Original pixels only, no annotations. |
| `capture.rgb_image` | `np.ndarray` | RGB cropped image array for integration with Pillow or matplotlib. |
| `capture.path` | `Optional[Path]` | Filesystem path where the crop was saved, or `None` if `save=False`. |
| `capture.box` | `BoundingBox` | The image-clamped bounding box used to produce the crop. |
| `capture.confidence` | `float` | YuNet detection confidence score for the captured face. |
| `capture.face` | `FaceDetection` | Full detection metadata including landmarks. |
| `capture.width` | `int` | Width of the cropped image in pixels. |
| `capture.height` | `int` | Height of the cropped image in pixels. |
| `capture.save(dest)` | `Path` | Manually write the cropped image array to a new destination path. |
| `capture.show(title)` | `None` | Open an interactive desktop preview window of the cropped face. |

---

## Multi-Face Selection

When an image contains multiple individuals:

```python
result = detekface.detect("team_photo.jpg")

print(f"Total faces found: {result.count}")

# Capture the largest face (default)
prominent_face = detekface.capture(result)

# Capture every face individually into an organized directory
for idx in range(result.count):
    crop = detekface.capture(
        result,
        face_index=idx,
        save_dir="team_avatars",
        filename=f"member_{idx + 1}.jpg",
    )
    print(f"Saved member #{idx + 1} to {crop.path}")
```

---

## In-Memory Cropping (No Disk Writes)

If you are streaming crops to a database, message queue, or deep learning classifier, disable disk writes:

```python
import detekface

result = detekface.detect("id_card.png")

# Crop in-memory only (save=False)
face = detekface.capture(result, save=False)

# Access raw NumPy array directly
bgr_pixels = face.image
rgb_pixels = face.rgb_image

print("Shape:", bgr_pixels.shape)  # e.g., (180, 180, 3)
```

---

## Live Webcam Capture & Callbacks

In interactive camera mode (`detekface.camera()`), users can capture faces live:

### 1. Keyboard Trigger
Press **`C`** on the keyboard while the camera preview window is focused.
- A HUD notification displays: `[Captured: capture_...jpg]`.
- The largest detected face in that specific frame is saved to `./captures/`.
- If no face was visible when `C` was pressed, a warning notification displays: `[Capture failed: No face detected]`.

### 2. Custom Capture Callback (`on_capture`)
Pass a callable to `on_capture` to receive the `FaceCapture` instance programmatically as soon as the user presses `C`:

```python
import detekface
from detekface import FaceCapture


def handle_face_captured(face: FaceCapture) -> None:
    print(f"\n[EVENT] New face captured!")
    print(f"  Saved path: {face.path}")
    print(f"  Confidence: {face.confidence:.2%}")
    print(f"  Dimensions: {face.width}x{face.height} px")
    # Trigger downstream tasks: upload to cloud, database update, etc.


detekface.camera(
    shape="hud",
    color="#00FF88",
    on_capture=handle_face_captured,
)
```

---

## Error Handling

```python
import detekface
from detekface.exceptions import (
    NoFaceDetectedError,
    InvalidCropError,
    CaptureSaveError,
    InvalidImageError,
)

result = detekface.detect("empty_room.jpg")

try:
    crop = detekface.capture(result)
except NoFaceDetectedError as err:
    print(f"No face available to crop: {err.message}")
    print(f"Hint: {err.hint}")
except IndexError:
    print("Requested face_index is out of range.")
except InvalidCropError as err:
    print(f"Crop was invalid or empty: {err.message}")
except CaptureSaveError as err:
    print(f"Failed to save crop to disk: {err.message}")
```
