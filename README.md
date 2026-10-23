# DetekFace

A reliable, beginner-friendly Python face detection package powered by YuNet and OpenCV.

## Features (Planned & Roadmap)

- **Milestone 1 (Current)**: Clean src-layout package foundation, packaging configuration, and test suite.
- **Milestone 2**: Core detector with YuNet ONNX model, bounding box & landmark dataclasses.
- **Milestone 3**: Clean visualization & drawing utilities for annotations and bounding boxes.
- **Milestone 4**: Webcam and real-time video stream support with graceful degradation.
- **Milestone 5**: Polish, examples, documentation, and PyPI readiness.

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

print(detekface.__version__)
```

## License

MIT
