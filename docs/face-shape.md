# Face Shape Detection & Biometric Scanner Guide

DetekFace includes a complete Face Shape Analysis feature powered by dense 478-point 3D facial landmarks and a shared geometric classification engine.

It works across **both** operational modes:
1. **Mode 1: Still-Image Analysis** — Analyze photos without opening a webcam.
2. **Mode 2: Real-Time Webcam Scanner** — Live biometric scanning HUD with animated laser sweep and multi-frame temporal smoothing.

Both modes share the **exact same underlying landmark extraction and geometric analysis engine**.

---

## 📦 Requirements & Installation

Face shape analysis uses the optional Google MediaPipe Face Landmarker model:

```bash
pip install "detekface[mediapipe]"
```

> **Zero configuration needed**: On first use with `analyze_face_shape=True`, the 478-point `face_landmarker.task` model bundle is automatically downloaded from Google mirrors, verified via SHA256 checksum, and cached locally. Users who only need standard face detection are never forced to install or download this model.

---

## 🖼️ Mode 1: Still-Image Face-Shape Analysis

Use `analyze_face_shape=True` with `detekface.detect()`:

```python
import detekface

result = detekface.detect(
    "portrait.jpg",
    analyze_face_shape=True
)

for face in result.faces:
    print(f"Shape Label : {face.face_shape.label}")
    print(f"Fit Score   : {face.face_shape.fit_score:.2f}")
    print(f"Alternatives: {face.face_shape.alternatives}")
    print(f"Measurements: {face.face_shape.measurements}")
    print(f"Warnings    : {face.face_shape.warnings}")
```

### Result Object Structure (`FaceShapeResult`)

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `label` | `str` | Classified shape (`"oval"`, `"round"`, `"square"`, `"heart"`, `"diamond"`, `"oblong"`, `"undetermined"`). |
| `fit_score` | `float` | Heuristic geometric fit score (0.0 to 1.0) based on Gaussian similarity to prototype dimensions. |
| `alternatives` | `Dict[str, float]` | Relative fit scores for candidate shapes, sorted descending. |
| `measurements` | `Dict[str, float]` | Absolute distances (pixels) and proportional ratios. |
| `quality_status`| `str` | Reliability indicator (`"good"`, `"warning"`, `"rejected"`, `"uncertain"`, `"analyzing"`). |
| `warnings` | `List[str]` | Cautionary explanations when confidence is degraded or undetermined. |
| `reasons` | `List[str]` | Property alias returning `warnings`. |

### Dense Landmarks Access

Each detected face also exposes `face.dense_landmarks`, providing 478 `Point` objects in pixel coordinates:

```python
for pt in face.dense_landmarks:
    print(pt.x, pt.y)
```

### Rendering Overlays on Still Images

```python
# Render face overlay, contour silhouette, and shape badge
annotated = result.annotate(
    show_face_shape=True,
    show_shape_contour=True,
    show_shape_measurements=True,
)

# Save directly to disk
result.save(
    "annotated_shape.jpg",
    show_face_shape=True,
    show_shape_measurements=True,
)
```

---

## 📹 Mode 2: Real-Time Webcam Scanner

Launch the real-time webcam scanner using `detekface.camera()`:

```python
import detekface

detekface.camera(
    analyze_face_shape=True,
    show_face_shape=True,
    face_shape_effect="scan",  # Animated biometric HUD scanline
)
```

### Live Scanner Features
- **Biometric Scanline Effect** (`face_shape_effect="scan"`): An oscillating glowing horizontal laser sweeps down and up the face bounding box.
- **Contour Silhouette**: Draws the anatomical face perimeter in real time.
- **Temporal Smoothing** (`TemporalShapeTracker`): Tracks faces across consecutive frames using IoU tracking and sliding-window voting. Initial frames show `[ SCANNING GEOMETRY... ]` before stabilizing to the classified shape, preventing jitter.
- **Pristine Capture**: Press <kbd>C</kbd> to save the largest face. Captured face crops remain clean and unpolluted by HUD graphics while retaining structured face shape metadata.

### Advanced Camera Session Integration

```python
import detekface

with detekface.open_camera(analyze_face_shape=True) as session:
    for result in session:
        for face in result.faces:
            if face.face_shape and face.face_shape.quality_status == "good":
                print(f"Face shape: {face.face_shape.label} ({face.face_shape.fit_score:.1%})")
```

---

## 📐 Supported Face Shapes & Geometric Heuristics

The shared classification engine measures key biometric landmarks:
- **Face Length**: Trichion (top forehead center) to Menton (lowest point of chin).
- **Cheekbone Width**: Outermost lateral zygomatic points.
- **Forehead Width**: Upper temple boundary contour.
- **Jaw Width**: Left and right gonion jaw angle vertices.
- **Jaw Angle**: Angle in degrees formed at the jaw vertices.

| Shape | Proportions & Visual Characteristics |
| :--- | :--- |
| **Oval** | Length > Width (~1.35–1.50). Cheekbones widest, soft curved jawline, forehead slightly wider than jaw. |
| **Round** | Length ≈ Width (~1.00–1.18). Cheekbones widest, soft curved jaw, forehead and jaw similar width. |
| **Square** | Length ≈ Width (~1.00–1.20). Wide angular jaw (jaw width ≈ cheekbone width), sharp jaw corner angle (~110°–120°). |
| **Oblong** | Length >> Width (ratio >= 1.55). Forehead, cheeks, and jaw similar in width. |
| **Heart** | Forehead significantly wider than jaw (forehead-to-jaw ratio >= 1.25), cheekbones prominent, tapering pointed chin. |
| **Diamond** | Cheekbones significantly wider than both forehead and jaw, narrow hairline and pointed chin. |
| **Undetermined** | Returned when pose angle exceeds frontal tolerance (>25° yaw), face is partially occluded, or resolution is too low. |

---

## 💻 Command-Line Interface (CLI)

Face shape analysis is available directly from the command line with `--face-shape`:

```bash
# Still image analysis with printed shape metrics
detekface image portrait.jpg --face-shape

# Save annotated image with contour and shape badges
detekface image portrait.jpg --output annotated.jpg --face-shape

# Launch live webcam biometric scanner
detekface camera --face-shape

# Batch process an entire directory with face shape classification
detekface batch ./photos --output ./results --face-shape
```

