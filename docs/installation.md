# Installation Guide

This guide covers installing **DetekFace** across Windows 11, Linux, and macOS environments, configuring virtual environments, managing OpenCV dependencies, and verifying your setup.

---

## Technical Prerequisites

Before installing DetekFace, ensure your environment meets the minimum version requirements:

| Component | Minimum Version | Recommended / Tested |
| :--- | :--- | :--- |
| **Python** | `>= 3.9` | `Python 3.11` to `3.15` (64-bit) |
| **pip** | `>= 21.0` | Latest |
| **OpenCV** | `>= 4.8` | `5.0.0+` or `4.10+` |
| **NumPy** | `>= 1.20` | `1.26+` or `2.0+` |

---

## Windows 11 Setup Instructions

### 1. Verify Python Installation
Open **Windows Terminal** (PowerShell or Command Prompt) and check your installed Python version:

```powershell
python --version
pip --version
```

If Python is not recognized, ensure Python is installed from [python.org](https://www.python.org/) and that the **"Add Python to PATH"** checkbox was selected during setup.

### 2. Create an Isolated Virtual Environment
Creating an isolated virtual environment prevents library conflicts between projects:

```powershell
# Create a virtual environment named .venv in your project directory
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

> **Note for PowerShell Users**: If you encounter an error stating `running scripts is disabled on this system`, enable script execution for your current session:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```

### 3. Install DetekFace

Install the package directly into your active virtual environment:

```powershell
pip install detekface
```

Or install from a local checkout:

```powershell
cd DetekFace
pip install .
```

---

## OpenCV Variant Guidance (GUI vs. Headless)

DetekFace requires OpenCV to perform neural network inference and display live GUI preview windows (`cv2.imshow`, `cv2.namedWindow`).

### Recommended Package
- **`opencv-python`** (default dependency): Includes OpenCV's HighGUI module, which provides native window creation and keyboard interaction.

### When to Use Headless
- If running on a headless server, Docker container, or cloud CI pipeline without an attached display monitor, you do not need GUI preview windows.
- To avoid GUI library dependencies in headless environments, you can use:
  ```bash
  pip install opencv-python-headless
  ```
- In headless mode, ensure you set `show=False` when calling `detekface.video()` or avoid calling `result.show()`.

> **Warning**: Do not install both `opencv-python` and `opencv-python-headless` in the same environment, as they share the same top-level `cv2` namespace and can create conflicting module bindings.

---

## Verifying the Installation

Verify that DetekFace imports and initializes correctly with a quick one-line smoke test:

```powershell
python -c "import detekface; print('DetekFace version:', detekface.__version__)"
```

Expected output:
```text
DetekFace version: 0.1.0
```

---

## Next Steps

Now that DetekFace is installed, continue to the [Quickstart Guide](quickstart.md) or explore [Webcam Usage](webcam.md).
