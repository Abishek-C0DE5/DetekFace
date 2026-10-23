# DetekFace

A reliable, beginner-friendly Python face detection package powered by YuNet and OpenCV.

## Features (Planned & Roadmap)

- **Milestone 1**: Clean src-layout package foundation, packaging configuration, and test suite.
- **Milestone 2**: Internal model loading system, trusted downloading, caching, SHA256 integrity validation, and OpenCV compatibility.
- **Milestone 3**: Still-image face detection engine, structured dataclasses (`FaceDetection`, `BoundingBox`, `FaceLandmarks`, `DetectionResult`), and input normalization.
- **Milestone 4**: Simplest working live webcam experience (`detekface.camera()`), green bounding boxes, Q/Escape exit, and guaranteed resource cleanup.
- **Milestone 5 (Current)**: Face cropping and capture (`detekface.capture()`), largest-face selection, boundary clamping, disk saving, and interactive 'C' key webcam capture with callbacks.
- **Milestone 6**: Polish, examples, documentation, and PyPI readiness.

## Installation

### For Development

1. Clone the repository:
   ```bash
   git clone https://github.com/Abishek-C0DE5/DetekFace.git
   cd DetekFace
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

Execute pytest from the project root:

```bash
pytest
```

## Quick Verification

### 1. Still-Image Face Detection & Cropping

```python
import detekface

# Detect faces
result = detekface.detect("photo.jpg", confidence_threshold=0.7)
print(f"Found {result.count} face(s)")

# Crop and save the largest face
face = detekface.capture(result, save_dir="captures", filename="my_face.jpg")
print(f"Captured {face.width}x{face.height}px face to: {face.path}")
```

### 2. Live Webcam Face Detection & Interactive Capture

```python
import detekface

def on_face_captured(face):
    print("New face captured!", face.path)

# Open webcam: press 'c' to capture largest face, press 'q' or Escape to close
detekface.camera(on_capture=on_face_captured)
```

## Model Licensing & Credits

`detekface` utilizes the official **YuNet** face detection model created by Shiqi Yu and maintained in the [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet):
- **Model Author**: Shiqi Yu (`libfacedetection`) & Shenzhen Institute of Artificial Intelligence and Robotics for Society
- **Model License**: MIT License (Copyright (c) 2020 Shiqi Yu)
- **Primary Source**: OpenCV Model Zoo (`https://github.com/opencv/opencv_zoo`)
- **Compatibility**:
  - `face_detection_yunet_2026may.onnx`: Dynamic input shape for OpenCV 5.x graph engine.
  - `face_detection_yunet_2023mar.onnx`: Fixed input shape for OpenCV 4.x DNN engine.

## License

MIT
