# Development & Contribution Guide

This guide is for developers and open-source contributors looking to modify, test, or contribute to `detekface`.

---

## 1. Development Prerequisites

- **Python**: Version 3.8 or higher.
- **Git**: Installed and available in your command line.
- **C/C++ Build Tools**: Generally not required because OpenCV and NumPy provide pre-compiled wheels for all major operating systems.

---

## 2. Environment Setup

### Step 1: Clone the Repository
```powershell
git clone https://github.com/Abishek-C0DE5/DetekFace.git
cd DetekFace
```

### Step 2: Create and Activate a Virtual Environment
```powershell
# Create isolated environment in .venv
python -m venv .venv

# Activate on Windows PowerShell
.\.venv\Scripts\Activate.ps1

# (Activate on Linux/macOS)
# source .venv/bin/activate
```

### Step 3: Install Package in Editable Mode with Dev Dependencies
```powershell
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

This installs `detekface` in editable mode (`-e`), meaning any changes made to `src/detekface/` are reflected immediately without re-installing. It also installs development tools including `pytest` and `build`.

---

## 3. Running the Test Suite

The test suite is built on `pytest` and includes unit tests, integration tests, mock hardware tests, and real neural network validation.

### Run All Tests
```powershell
pytest
```

### Run with Verbose Details and Timing
```powershell
pytest -v
```

### Run Specific Test Modules
```powershell
# Test webcam module with simulated cameras
pytest tests/test_camera.py

# Test face cropping and clamping
pytest tests/test_capture.py

# Test overlay rendering and color conversions
pytest tests/test_drawing.py

# Run real YuNet model validation against test imagery
pytest tests/test_model_validation.py
```

---

## 4. Running Benchmarks & Examples

### Run Model Inference Benchmark
To measure the real execution latency and FPS throughput on your hardware across multiple resolutions:
```powershell
python examples/benchmark.py
```

### Run Image Detection Example
```powershell
python examples/detect_image.py
```

### Run Video Detection Example
```powershell
python examples/detect_video.py
```

---

## 5. Building the Distribution Packages

`detekface` uses `pyproject.toml` with `setuptools` conforming to modern PEP 517 / PEP 621 packaging standards.

To build both a source distribution (`.tar.gz`) and a binary wheel (`.whl`):

```powershell
python -m build
```

The resulting distribution files are output to the `dist/` directory:
- `dist/detekface-0.1.0.tar.gz` (Source distribution)
- `dist/detekface-0.1.0-py3-none-any.whl` (Pure Python wheel)

To test installing the generated wheel into a clean test environment:
```powershell
pip install dist/detekface-0.1.0-py3-none-any.whl
```

---

## 6. Project Architecture & Code Standards

- **Source Code**: Located strictly inside `src/detekface/`.
- **Typing**: All new public functions and methods must include Python type annotations (`mypy` compatible).
- **Docstrings**: Use Google-style docstrings with `Args:`, `Returns:`, and `Raises:` sections.
- **Exceptions**: All internal errors must raise subclassed `DetekFaceError` types with informative `.message` and `.hint` attributes.
- **Zero Inventions**: Never document hypothetical arguments or features that are not yet implemented and tested.

---

## 7. Contribution Workflow

1. **Fork the Repository** on GitHub.
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. **Implement Changes**: Write clean code and keep commits focused.
4. **Add Automated Tests**: Every new feature or bug fix must have accompanying tests under `tests/`.
5. **Run the Full Test Suite**: Ensure all tests pass (`pytest`).
6. **Submit a Pull Request**: Provide a clear description of the problem solved, changes made, and test commands executed.
