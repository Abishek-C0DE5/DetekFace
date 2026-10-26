# Backend Selection & Optional Dependencies

DetekFace features a modular, adapter-based detection architecture. By default, it operates with zero heavy dependencies using the built-in **YuNet** backend. Users who need Google's **MediaPipe** backend can opt in by installing the `[mediapipe]` extra.

---

## 1. Overview of Supported Backends

| Attribute | YuNet (Default) | MediaPipe (Optional) |
| :--- | :--- | :--- |
| **Backend Key** | `backend="yunet"` | `backend="mediapipe"` |
| **Model Architecture** | YuNet ONNX (libfacedetection) | BlazeFace Short-Range TFLite |
| **Inference Engine** | OpenCV `FaceDetectorYN` (C++ DNN) | MediaPipe Tasks / Solutions |
| **Extra Dependencies** | None (`opencv-python` included) | `pip install "detekface[mediapipe]"` |
| **Facial Landmarks** | **5 anatomical points**:<br>• Right Eye<br>• Left Eye<br>• Nose Tip<br>• Right Mouth Corner<br>• Left Mouth Corner | **6 keypoints**:<br>• Right Eye<br>• Left Eye<br>• Nose Tip<br>• Mouth Center<br>• Right Ear<br>• Left Ear |
| **Mouth Landmarks** | Left & right mouth corners | Single mouth center (`right/left_mouth` are `None`) |
| **Best Used For** | Fast general-purpose detection, zero extra install footprint, standard desktop viewports | Close-range selfies, mobile-oriented keypoints, specialized MediaPipe ecosystems |

> **Landmark Contract Guarantee**: DetekFace **never invents or interpolates** missing landmarks. When using MediaPipe, `landmarks.mouth_center` is populated while `right_mouth` and `left_mouth` remain `None`. When using YuNet, `right_mouth` and `left_mouth` are populated while `mouth_center`, `right_ear`, and `left_ear` remain `None`.

---

## 2. Installing the Optional MediaPipe Backend

To enable MediaPipe support, install DetekFace with the `mediapipe` optional extra:

```bash
pip install "detekface[mediapipe]"
```

If you attempt to use the MediaPipe backend without installing the extra, DetekFace will raise a helpful `DependencyNotFoundError`:

```python
import detekface
from detekface.exceptions import DependencyNotFoundError

try:
    detekface.detect("photo.jpg", backend="mediapipe")
except DependencyNotFoundError as e:
    print(f"Error: {e.message}")
    print(f"Hint : {e.hint}")
```

Output:
```text
Error: The 'mediapipe' backend requires the optional 'mediapipe' package.
Hint : Install it with: pip install "detekface[mediapipe]"
```

DetekFace **never silently falls back** to YuNet if you explicitly request MediaPipe. This guarantees predictable, testable behavior in production systems.

---

## 3. Selecting Backends Across APIs

You can select your preferred backend across all public entry points by setting `backend="yunet"` (default) or `backend="mediapipe"`:

### 1. Still Image Detection (`detekface.detect`)
```python
import detekface

# Run detection using YuNet (default)
result_yunet = detekface.detect("photo.jpg", backend="yunet")

# Run detection using MediaPipe
result_mp = detekface.detect("photo.jpg", backend="mediapipe")
```

### 2. Live Webcam Stream (`detekface.camera`)
```python
import detekface

# Launch live camera with MediaPipe backend and HUD overlay
detekface.camera(backend="mediapipe", shape="hud", color="cyan")
```

### 3. Batch Image Processing (`detekface.detect_many`)
```python
import detekface

images = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]
results = detekface.detect_many(images, backend="mediapipe")
```

### 4. Prerecorded Video Processing (`detekface.video`)
```python
import detekface

detekface.video(
    path="clip.mp4",
    output_path="annotated_clip.mp4",
    backend="mediapipe",
    show=True,
)
```

### 5. Unified Detector Class (`FaceDetector`)
```python
from detekface import FaceDetector

detector = FaceDetector(backend="mediapipe", confidence_threshold=0.6)
print(f"Active backend: {detector.backend_name}")
```

### 6. Command-Line Interface (CLI)
```bash
# Camera with MediaPipe backend
detekface camera --backend mediapipe

# Single image with MediaPipe backend
detekface image input.jpg --backend mediapipe --output output.jpg

# Batch detection with MediaPipe backend
detekface batch ./photos --backend mediapipe --output ./annotated
```

---

## 4. Comparing Output Formats

Regardless of which backend performs the detection, DetekFace normalizes all outputs into identical `DetectionResult`, `FaceDetection`, and `BoundingBox` objects:

```python
import detekface

result = detekface.detect("sample.jpg", backend="mediapipe")

for face in result.faces:
    # Bounding boxes are identical across all backends:
    x, y, w, h = face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height
    print(f"Face Box: [{x}, {y}, {w}, {h}], Confidence: {face.confidence:.1%}")

    # Inspect landmarks:
    if face.landmarks:
        print(f"  Right Eye: {face.landmarks.right_eye}")
        print(f"  Left Eye : {face.landmarks.left_eye}")
        print(f"  Nose Tip : {face.landmarks.nose_tip}")
        if face.landmarks.mouth_center:
            print(f"  Mouth Center (MediaPipe): {face.landmarks.mouth_center}")
        if face.landmarks.right_mouth:
            print(f"  Mouth Corners (YuNet)   : {face.landmarks.right_mouth}, {face.landmarks.left_mouth}")
```

---

## 5. Backend Tradeoffs & Advice

- **Use YuNet if**:
  - You want minimal installation overhead (no additional pip dependencies).
  - You require mouth corner landmarks for expressions or mouth width calculation.
  - You want predictable execution across older Python versions (3.9+) and custom architectures.
- **Use MediaPipe if**:
  - You are integrating into an existing Google MediaPipe computer vision pipeline.
  - You specifically need ear landmarks (`right_ear`, `left_ear`) or single mouth center keypoints.
  - You are deploying on Python 3.10–3.12 where MediaPipe wheels are readily available.
