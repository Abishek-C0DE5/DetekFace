# Contributing to DetekFace

Thank you for your interest in improving DetekFace! We welcome contributions from developers of all skill levels.

---

## Code of Conduct

DetekFace is dedicated to providing a welcoming, harassment-free experience for everyone. Please be respectful, constructive, and kind in all project communications and interactions.

---

## Getting Started

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/Abishek-C0DE5/DetekFace.git
   cd DetekFace
   ```

2. **Set up a Virtual Environment**:
   ```bash
   # On Windows (PowerShell):
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # On Linux / macOS (bash/zsh):
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install in Editable Mode with Developer Extras**:
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev,mediapipe]"
   ```

---

## Development Workflow & Quality Standards

Before opening a pull request, ensure all automated quality checks pass locally:

### 1. Code Formatting and Linting
We use [Ruff](https://astral.sh/ruff) for blazing-fast linting and code formatting:
```bash
# Check code style and common bugs:
ruff check .

# Check formatting compliance:
ruff format --check .

# Auto-format your code:
ruff format .
```

### 2. Static Type Checking
We use [Mypy](https://mypy-lang.org/) for strict type verification:
```bash
mypy
```

### 3. Automated Test Suite
Run the test suite with [pytest](https://docs.pytest.org/):
```bash
# Run all offline unit tests (does not require camera or internet access):
pytest -v -m "not network and not optional_backend"

# Run optional backend tests (requires mediapipe extra):
pytest -v -m "optional_backend"

# Run all tests:
pytest -v
```

### 4. Distribution Build & Metadata Check
Ensure package distributions build cleanly and conform to PyPI standards:
```bash
python -m build
twine check --strict dist/*
```

---

## Pull Request Guidelines

1. **Create a Topic Branch**:
   ```bash
   git checkout -b feature/my-feature-name
   ```
2. **Write Focused Tests**:
   - Every bug fix or new feature must be accompanied by unit tests in `tests/`.
   - Never rely on physical webcam hardware; use mocks for `cv2.VideoCapture` and GUI windows.
   - Mark tests requiring live remote downloads with `@pytest.mark.network`.
   - Mark tests requiring optional dependencies with `@pytest.mark.optional_backend`.
3. **Keep Examples & Docs Synchronized**:
   - Update `README.md` or relevant guides under `docs/` if modifying public APIs.
4. **Submit Your Pull Request**:
   - Provide a clear description of the problem solved and the design approach taken.
   - Verify that all GitHub Actions CI checks pass.

---

## Detailed Documentation

For a comprehensive walkthrough of the project architecture, release procedures, and benchmarking methodology, refer to the **[Contributor & Development Guide](docs/development.md)**.
