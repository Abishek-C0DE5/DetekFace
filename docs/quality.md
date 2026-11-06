# Face Quality Assessment in DetekFace

`detekface` 0.3.0 includes an objective technical capture-quality assessment function that evaluates whether a detected face meets technical standards for downstream computer-vision processing.

---

## ⚖️ Important Disclaimer

> [!IMPORTANT]
> **Ethical & Objective Scope**: DetekFace quality assessment measures **technical image capture parameters** (sharpness, lighting exposure, framing, and head pose).
> - It **does not** evaluate personal identity, demographic characteristics, or subjective facial attractiveness.
> - Quality thresholds are application-dependent: a frame marked "suboptimal" for high-resolution biometric enrollment may remain entirely suitable for casual social webcam previews or security presence detection.

---

## 🚀 Quick Usage

### Assessing Still Image Quality

```python
import detekface

quality = detekface.check_face_quality("portrait.jpg")

print(f"Total Faces Evaluated: {quality.face_count}")

if quality.primary_face is not None:
    face_q = quality.primary_face
    print(f"Sharpness Score : {face_q.sharpness:.1f} (Threshold >= 80.0)")
    print(f"Mean Brightness : {face_q.brightness:.1f} / 255 (Acceptable: 50 - 200)")
    print(f"Face Area Ratio : {face_q.area_ratio * 100:.2f}% of image")
    print(f"Boundary Status : {'Clipped at border' if face_q.is_clipped else 'Cleanly framed'}")
    print(f"Quality Warnings: {face_q.warnings if face_q.warnings else 'None'}")
```

### Live Webcam Quality Feedback HUD

```python
import detekface

# Display real-time capture quality feedback tags directly in the webcam HUD
detekface.camera(
    show_quality=True,
    shape="hud",
)
```

---

## 📊 Technical Quality Metrics

| Metric | Measurement Technique | Optimal Range | Purpose |
| :--- | :--- | :---: | :--- |
| **Sharpness** | Variance of the Laplacian operator ($\nabla^2 I$) over grayscale face ROI | $\ge 80.0$ | Detects motion blur or out-of-focus capture. |
| **Brightness** | Mean luminance pixel intensity in grayscale ROI | $50.0 - 200.0$ | Identifies underexposed (dark) or overexposed (washed out) faces. |
| **Face Area Ratio** | $\frac{\text{Face Area}}{\text{Image Area}}$ | $\ge 0.5\%$ | Flags faces that are too small or distant for reliable landmarking. |
| **Framing / Boundary** | Distance from bounding box edges to image perimeter | $\ge 4\text{ px}$ | Flags heads cropped by camera field of view edges. |
| **Head-Pose Suitability** | Absolute Yaw and Pitch Euler angles | $\|\text{Yaw}\| \le 25^\circ, \|\text{Pitch}\| \le 20^\circ$ | Identifies non-frontal poses that occlude key biometrics. |
| **Landmark Completeness** | Verification of key landmark presence | 5/5 or 468/468 | Ensures landmarks are not collapsed or missing. |

---

## ⚠️ Actionable Warning Strings

When a metric falls outside recommended thresholds, structured warning strings are appended to `face.warnings`:

- `"IMAGE_BLURRY"`: Laplacian sharpness score is below `blur_threshold`.
- `"LIGHTING_TOO_DARK"`: Mean luminance is below `min_brightness`.
- `"LIGHTING_TOO_BRIGHT"`: Mean luminance exceeds `max_brightness`.
- `"FACE_TOO_SMALL"`: Face bounding box dimensions or area ratio are below minimum requirements.
- `"FACE_CLIPPED"`: Face touches or overflows the image boundary.
- `"HEAD_POSE_NON_FRONTAL"`: Head rotation exceeds frontal suitability thresholds.

---

## ⚙️ Customizing Quality Thresholds

```python
import detekface

quality = detekface.check_face_quality(
    "portrait.jpg",
    blur_threshold=120.0,    # Stricter sharpness requirement
    min_brightness=60.0,     # Disallow dimly lit images
    max_brightness=190.0,    # Stricter highlight cap
    min_face_size=80,        # Minimum 80x80 pixels
    border_margin=10,        # Minimum 10px from frame edges
)
```

