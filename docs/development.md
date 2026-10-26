# Development & Contribution Guide

This guide is for developers and open-source contributors looking to modify, test, benchmark, or contribute to `detekface`.

---

## 1. Development Prerequisites

- **Python**: `>= 3.9` through `3.13` (tested on 3.10, 3.11, 3.12, 3.15).
- **Git**: Installed and available in your shell.
- **C/C++ Compiler**: Not required because OpenCV and NumPy provide pre-compiled binary wheels for Windows, macOS, and Linux.

---

## 2. Environment Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/Abishek-C0DE5/DetekFace.git
cd DetekFace
```

### Step 2: Create and Activate a Virtual Environment
```bash
# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On Linux / macOS (bash/zsh):
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Package in Editable Mode with Developer Extras
```bash
python -m pip install --upgrade pip
pip install -e ".[dev,mediapipe]"
```

This installs `detekface` in editable mode (`-e`), meaning any changes made to `src/detekface/` take effect immediately without reinstallation.

---

## 3. Code Style, Formatting, & Type Checking

DetekFace enforces strict code quality and formatting standards via [Ruff](https://astral.sh/ruff) and [Mypy](https://mypy-lang.org/):

### Run Ruff Linter
```bash
ruff check .
```

### Run Ruff Formatter
```bash
# Check formatting without modifying files:
ruff format --check .

# Automatically reformat files:
ruff format .
```

### Run Mypy Static Type Checker
```bash
mypy
```

All 16 source files in `src/detekface/` must pass static typing with zero errors.

---

## 4. Running the Test Suite

DetekFace contains an automated test suite with 270+ tests covering unit tests, camera mocks, CLI commands, and model validation.

### Run Offline Unit Tests (Recommended during local development)
Runs fully offline without requiring a physical camera or downloading model files:
```bash
pytest -v -m "not network and not optional_backend"
```

### Run Optional Backend Tests (MediaPipe)
```bash
pytest -v -m "optional_backend"
```

### Run Network-Dependent Model Download Tests
```bash
pytest -v -m "network"
```

### Run Complete Test Suite
```bash
pytest -v
```

### Run Specific Test Modules
```bash
# Test webcam module with mocked hardware
pytest tests/test_camera.py

# Test advanced camera session and lifecycle
pytest tests/test_camera_session.py

# Test face cropping and clamping
pytest tests/test_capture.py

# Test batch image processing
pytest tests/test_batch.py

# Test CLI commands and arguments
pytest tests/test_cli.py

# Test video processing
pytest tests/test_video.py

# Test CI workflow syntax and security
pytest tests/test_ci_workflows.py
```

---

## 5. Running Benchmarks & Runnable Examples

All example scripts are located in `examples/` and can be run directly:

### Run Hardware Performance Benchmark
Measures actual execution latency, percentiles, and FPS throughput across multiple resolutions on your CPU:
```bash
python examples/benchmark.py
```

### Run Single Image Detection Example
```bash
python examples/detect_image.py
```

### Run Face Cropping Example
```bash
python examples/crop_face.py
```

### Run Prerecorded Video Detection Example
```bash
python examples/video_detection.py
```

### Run Interactive Live Webcam Example
```bash
python examples/webcam.py
```

---

## 6. Building Distribution Packages & Verifying Wheel

DetekFace uses PEP 517 / PEP 621 packaging via `pyproject.toml` and `setuptools`.

### Step 1: Build Source Distribution and Binary Wheel
```bash
python -m build
```
This generates `dist/detekface-0.1.0.tar.gz` and `dist/detekface-0.1.0-py3-none-any.whl`.

### Step 2: Validate Distribution Metadata with Twine
```bash
twine check --strict dist/*
```

### Step 3: Test Wheel Installation & Import Smoke Test
Verify that the package installs cleanly from wheel and can be imported outside the source directory:
```bash
# On Linux / macOS:
pip install dist/*.whl

# On Windows:
$wheel = (Get-ChildItem dist/*.whl).FullName
pip install $wheel

# Verify import:
python -c "import detekface; print('DetekFace installed:', detekface.__version__)"

# Verify console CLI script:
detekface --version
detekface --help
```

---

## 7. Continuous Integration Architecture

DetekFace uses GitHub Actions workflows defined in `.github/workflows/`:
1. **`ci.yml`**:
   - `lint-and-typecheck`: Ruff linter, Ruff formatter, and Mypy on Ubuntu.
   - `test-base`: Matrix across Linux (Ubuntu 3.9–3.13), Windows (3.10, 3.12), and macOS (3.11, 3.12) running base unit tests without optional extras.
   - `test-optional-backends`: Matrix across Linux, Windows, and macOS testing the `mediapipe` optional backend.
   - `model-integration-tests`: Tests live model downloading and real inference.
   - `build-and-verify-dist`: Tests building sdist/wheel, strict Twine validation, and wheel installation.
2. **`release.yml`**:
   - Triggered on git release tags `v*`.
   - Uses PyPI Trusted Publishing (OIDC token exchange) without hardcoded credentials.
---

## 8. Code Standards & Architecture

- **Source Code**: Located strictly inside `src/detekface/`.
- **Typing**: All public functions and methods must include Python type annotations (`mypy` compatible).
- **Docstrings**: Use Google-style docstrings with `Args:`, `Returns:`, and `Raises:` sections.
- **Exceptions**: All internal errors must raise subclassed `DetekFaceError` types with informative `.message` and `.hint` attributes.
- **Zero Inventions**: Never document hypothetical arguments or features that are not yet implemented and tested.

---

## 9. Pull Request Workflow

1. **Fork the Repository** on GitHub.
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. **Implement Changes**: Write clean code and keep commits focused.
4. **Add Automated Tests**: Every new feature or bug fix must have accompanying tests under `tests/`.
5. **Run the Full Test Suite**: Ensure all tests pass (`pytest`).
6. **Submit a Pull Request**: Provide a clear description of the problem solved, changes made, and test commands executed.
