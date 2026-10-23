# DetekFace

A reliable, beginner-friendly Python face detection package powered by YuNet and OpenCV.

## Features (Planned & Roadmap)

- **Milestone 1**: Clean src-layout package foundation, packaging configuration, and test suite.
- **Milestone 2 (Current)**: Internal model loading system, trusted downloading, caching, SHA256 integrity validation, and OpenCV compatibility.
- **Milestone 3**: Core detector interface, face detections, bounding box & landmark dataclasses.
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
from detekface import resolve_model, verify_model_initialization

print("Version:", detekface.__version__)

# Resolve model (auto-selects compatible YuNet ONNX model, downloads to cache if needed)
model_path = resolve_model()
print("Model path:", model_path)

# Verify compatibility with OpenCV
verify_model_initialization(model_path)
print("Model verified successfully!")
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
