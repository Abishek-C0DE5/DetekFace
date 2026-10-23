# DetekFace

A reliable, beginner-friendly Python face detection package powered by YuNet and OpenCV.

## Features (Planned & Roadmap)

- **Milestone 1**: Clean src-layout package foundation, packaging configuration, and test suite.
- **Milestone 2**: Internal model loading system, trusted downloading, caching, SHA256 integrity validation, and OpenCV compatibility.
- **Milestone 3 (Current)**: Still-image face detection engine, structured dataclasses (`FaceDetection`, `BoundingBox`, `FaceLandmarks`, `DetectionResult`), and input normalization.
- **Milestone 4**: Clean visualization & drawing utilities for annotations and bounding boxes.
- **Milestone 5**: Webcam and real-time video stream support with graceful degradation.
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

```python
import detekface

# 1. Detect faces in an image file or NumPy BGR array
result = detekface.detect("photo.jpg", confidence_threshold=0.7)

print(f"Found {result.count} face(s)")

for face in result.faces:
    print("Bounding Box:", face.bbox.as_tuple())  # (x, y, width, height)
    print("Confidence :", f"{face.confidence * 100:.1f}%")
    if face.landmarks:
        print("Right Eye  :", face.landmarks.right_eye.as_tuple())
        print("Left Eye   :", face.landmarks.left_eye.as_tuple())
        print("Nose Tip   :", face.landmarks.nose_tip.as_tuple())
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
