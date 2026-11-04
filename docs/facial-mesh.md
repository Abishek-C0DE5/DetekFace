# Facial Mesh Visualization & Live FX Scanner Guide

<p align="center">
  <img src="assets/detekface_main_logo.png" alt="DetekFace Logo" width="500" />
</p>

This guide explains how **DetekFace 0.2.0** performs dense facial landmark localization, renders customizable facial mesh wireframes, and drives real-time animated Live FX computer-vision scanning.

---

## Table of Contents

1. [What is Facial Mesh Visualization?](#1-what-is-facial-mesh-visualization)
2. [How Dense Facial Landmarks Work](#2-how-dense-facial-landmarks-work)
3. [5-Point Landmarks vs. 478 Dense Landmarks](#3-5-point-landmarks-vs-478-dense-landmarks)
4. [Anatomical Depth & 3D Reconstruction Boundary](#4-anatomical-depth--3d-reconstruction-boundary)
5. [Available Mesh Styles](#5-available-mesh-styles)
6. [Live FX Computer-Vision Scanner](#6-live-fx-computer-vision-scanner)
7. [Webcam Usage & Interactive Hotkeys](#7-webcam-usage--interactive-hotkeys)
8. [Still-Image Mesh Detection & Rendering](#8-still-image-mesh-detection--rendering)
9. [Custom Colors, Thickness & Regions](#9-custom-colors-thickness--regions)
10. [Optional Installation Requirements](#10-optional-installation-requirements)
11. [Model Caching & Directory Structure](#11-model-caching--directory-structure)
12. [Performance & Latency Characteristics](#12-performance--latency-characteristics)
13. [Troubleshooting & Common Errors](#13-troubleshooting--common-errors)

---

## 1. What is Facial Mesh Visualization?

Facial mesh visualization maps connected planar line segments between anatomical facial keypoints across the face surface. It provides a topological wireframe representing the contour, eyes, eyebrows, nose, and lips.

In DetekFace, mesh visualization is implemented as a dedicated rendering layer on top of the primary face detector. It operates both in real-time webcam streams and on still images.

---

## 2. How Dense Facial Landmarks Work

Dense facial landmarking uses a lightweight neural network graph (powered by Google MediaPipe Face Landmarker) to predict **478 surface landmark coordinates** normalized to `[0.0, 1.0]`.

```
Raw Image / Video Frame
       │
       ▼
1. Primary Face Detection (YuNet / BlazeFace)
       │  (Finds bounding box & orientation)
       ▼
2. Face Landmarker Inference (MediaPipe)
       │  (Extracts 478 3D keypoints per face)
       ▼
3. Landmark Projection & Bounding Box Association
       │  (Maps normalized coordinates to pixel space [0..W, 0..H])
       ▼
4. Mesh Renderer & Live FX Compositing
```

---

## 3. 5-Point Landmarks vs. 478 Dense Landmarks

| Feature | Basic 5-Point Landmarks | Dense 478-Point Landmarks |
|---|---|---|
| **Primary Backend** | YuNet (`FaceDetector(backend="yunet")`) | Google MediaPipe (`detekface[mediapipe]`) |
| **Keypoints Covered** | Right eye, left eye, nose tip, right mouth corner, left mouth corner | Full facial oval (36 pts), upper/lower lips, eye rings, irises, eyebrow ridges |
| **Primary Use Case** | Fast face alignment, affine rotation correction, fast tracking | Facial mesh wireframes, biometric face-shape analysis, facial animation |
| **Inference Latency** | Included in base detection (~24 ms on CPU) | Additional ~20 ms on CPU |
| **Memory Footprint** | Negligible (5 Point objects) | 478 Point objects per face |

---

## 4. Anatomical Depth & 3D Reconstruction Boundary

> [!IMPORTANT]
> **Planar Projection vs. Volumetric Medical Reconstruction**:
> The 2D facial mesh wireframe rendered by DetekFace is a **planar perspective projection** of predicted surface keypoints onto the camera image plane. While the underlying neural network estimates relative z-depth coordinates to maintain topological consistency under head rotation, **it is NOT a complete anatomical craniofacial reconstruction or volumetric soft-tissue 3D model**. It should not be used as a medical, maxillofacial, or surgical diagnostic tool.

---

## 5. Available Mesh Styles

DetekFace provides 5 distinct topological mesh styles:

```python
annotated = result.render(show_mesh=True, mesh_style="neon")
```

| Style | Visual Description |
|---|---|
| **`wireframe`** | Clean, technical triangulation connecting all dense facial landmarks. |
| **`neon`** | High-contrast glowing lines with emphasized outer silhouette and highlighted anatomical anchors (irises, nose bridge, lip corners). |
| **`minimal`** | Thin, clean structural mesh lines with minimal visual clutter. |
| **`contour`** | Silhouettes only: outer face oval, eye rings, nose ridge, and lip outlines. |
| **`points`** | Renders discrete landmark keypoints as anti-aliased vertex dots without connecting lines. |

---

## 6. Live FX Computer-Vision Scanner

Live FX adds an animated computer-vision HUD scanner layer:
- **Continuous Sinusoidal Laser Sweep**: Moves smoothly across the detected face bounds based on `time.perf_counter()` clock timestamps (frame-rate independent).
- **Lock-in HUD Brackets**: Reticle corner brackets converge inward and transition from amber (acquiring) to bright cyan/emerald (locked).
- **Telemetry Badges**: Small dark HUD tags: `[FACE LOCKED #1]`, `[SCANNING #1]`, and biometric shape tags `[SHAPE: OVAL (91%)]`.
- **Grace Period Occlusion Recovery**: Tracks persist smoothly across a `0.25s` grace period to prevent jarring flickers if detection momentarily drops.

---

## 7. Webcam Usage & Interactive Hotkeys

Launch live webcam face mesh and scanner:

```python
import detekface

detekface.camera(
    show_mesh=True,
    mesh_style="neon",
    show_scan=True,
    show_fps=True,
)
```

### Interactive Webcam Hotkeys:
- **`m`** / **`M`**: Cycle through mesh styles (`wireframe` ➔ `neon` ➔ `minimal` ➔ `contour` ➔ `points` ➔ off).
- **`f`** / **`F`**: Cycle through Live FX styles (`scan` ➔ `neon` ➔ `wireframe` ➔ `minimal` ➔ off).
- **`c`** / **`C`**: Capture a clean, pristine face crop to disk (free of overlay lines).
- **`q`** / **`ESC`**: Quit and release webcam cleanly.

---

## 8. Still-Image Mesh Detection & Rendering

Detect faces in an image, compute dense landmarks, and render or save:

```python
import detekface

# 1. Detect faces with dense landmarks enabled
result = detekface.detect("portrait.jpg", mesh=True)

# 2. Render mesh onto a copy of the image
annotated = result.render(
    show_mesh=True,
    mesh_style="neon",
    mesh_color="#00FF88",
    mesh_thickness=2,
    show_contour=True,
)

# 3. Save to disk
result.save("output_mesh.jpg", show_mesh=True, mesh_style="neon")
```

---

## 9. Custom Colors, Thickness & Regions

Custom styling can be specified via RGB hex strings (`"#00FF88"`), named color strings (`"cyan"`, `"neon_green"`, `"red"`), or integer RGB tuples (`(0, 255, 136)`):

```python
annotated = result.render(
    show_mesh=True,
    mesh_style="wireframe",
    mesh_color="#00FFFF",           # Cyan mesh
    mesh_thickness=1,               # Fine lines
    mesh_regions=["contours", "lips"], # Only render facial contour and lips
    mesh_draw_points=True,          # Draw vertex points
)
```

---

## 10. Optional Installation Requirements

Face mesh and dense landmark analysis require Google MediaPipe:

```bash
# Standard basic detection (YuNet only, ~1.5 MB download)
pip install detekface

# Full feature set with Facial Mesh & Face Shape Analysis:
pip install "detekface[mediapipe]"
```

If MediaPipe is not installed and mesh rendering is invoked, DetekFace raises a clean `DependencyNotFoundError` with installation instructions.

---

## 11. Model Caching & Directory Structure

When `mesh=True` or `analyze_face_shape=True` is first called, DetekFace automatically resolves and caches the official MediaPipe Face Landmarker model:

- **Default Cache Directory**:
  - Windows: `%USERPROFILE%\.detekface\models\`
  - Linux/macOS: `~/.detekface/models/`
- **Model Asset**: `face_landmarker.task` (~3.7 MB).
- **Custom Cache Directory**: You can pass `cache_dir="/custom/path"` to any detection function or constructor.

---

## 12. Performance & Latency Characteristics

Measured CPU performance on an Intel Core processor with a 640×480 video feed:

| Pipeline Combination | Latency (ms) | Frame Rate (FPS) |
|---|---|---|
| **Base Face Detection** (YuNet) | **23.13 ms** | **43.2 FPS** |
| **Detection + Facial Mesh** (`show_mesh=True`) | **42.41 ms** | **23.6 FPS** |
| **Detection + Mesh + Live Scan FX** (`show_scan=True`) | **39.78 ms** | **25.1 FPS** |
| **Detection with Multiple Faces** (2 faces) | **39.05 ms** | **25.6 FPS** |
| **Mesh Vector Drawing Overhead** | **1.65 ms** | **> 600 FPS equivalent** |

---

## 13. Troubleshooting & Common Errors

### Error: `DependencyNotFoundError: Face shape analysis requires the optional 'mediapipe' package`
- **Cause**: Attempted to use `show_mesh=True` or `mesh=True` without the MediaPipe extra.
- **Solution**: Run `pip install "detekface[mediapipe]"`.

### Error: `ConfigurationError: Invalid mesh style 'matrix'`
- **Cause**: An unsupported style name was passed.
- **Solution**: Choose from: `'wireframe'`, `'neon'`, `'minimal'`, `'contour'`, or `'points'`.

### Error: No facial mesh appears on face
- **Cause**: Face confidence was below threshold, or the face was severely occluded (> 45° profile view).
- **Solution**: Ensure good frontal lighting and near-frontal head pose (< 25° yaw).

