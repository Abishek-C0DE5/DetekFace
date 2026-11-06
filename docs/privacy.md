# Face Privacy Mode & Anonymization in DetekFace

`detekface` 0.3.0 provides local, privacy-preserving face anonymization across still images, live webcam feeds, and prerecorded video files.

---

## 🔒 Important Privacy Disclaimer

> [!WARNING]
> **Detection Limitations**: Privacy filters can only obscure faces that are successfully detected by the underlying computer-vision model.
> - Faces with extreme occlusions, backlighting, or off-angle profiles might not be detected.
> - **Never assume complete identity protection** without human verification in high-risk privacy workflows.
> - For privacy-critical pipelines, use `fail_closed=True` to automatically obscure the entire frame whenever face detection fails or returns zero candidates.

---

## 🚀 Quick Usage

### Anonymizing an Image File

```python
import detekface

# 1. Gaussian Blur Anonymization
blurred = detekface.anonymize("group.jpg", method="blur", strength=0.6)
blurred.save("group_blurred.jpg")

# 2. Pixelation Mosaic Anonymization
pixelated = detekface.anonymize("group.jpg", method="pixelate", strength=0.7)
pixelated.save("group_pixelated.jpg")

# 3. Solid Color Masking (Elliptical)
masked = detekface.anonymize(
    "group.jpg",
    method="solid",
    color=(0, 0, 0),         # Black fill
    mask_shape="ellipse",    # Natural head contour
)
masked.save("group_masked.jpg")
```

### Live Webcam Privacy Mode

```python
import detekface

# Run live webcam with real-time blur privacy mode
detekface.camera(
    anonymize="blur",
    anonymize_strength=0.5,
    shape="none",  # Do not draw bounding boxes over privacy blurred areas
)
```

### Video File Anonymization

```python
import detekface

detekface.video(
    "cctv_footage.mp4",
    output_path="cctv_anonymized.mp4",
    anonymize="pixelate",
    anonymize_strength=0.7,
)
```

---

## 🎭 Anonymization Methods

| Method | Visual Effect | Customization |
| :--- | :--- | :--- |
| `"blur"` | Smooth Gaussian blur obscuring facial features | `strength: float` (0.05–1.0) or `kernel_size: int` |
| `"pixelate"` | Pixelated mosaic blocks removing high-frequency detail | `strength: float` (0.05–1.0) or `pixel_size: int` |
| `"solid"` | Opaque color fill completely redacting the face | `color: (B, G, R)` or hex, `mask_shape="box" \| "ellipse"` |

---

## 🛡️ Fail-Closed Safety Mode

In automated privacy compliance pipelines, if a face detector fails or detects zero faces in an image known to require redaction, standard filters would leave the frame unredacted. DetekFace provides `fail_closed=True`:

```python
import detekface

# If zero faces are detected, fail_closed=True anonymizes the ENTIRE image
result = detekface.anonymize(
    "confidential_doc.jpg",
    method="blur",
    fail_closed=True,
)

if result.fail_closed_triggered:
    print("Warning: Zero faces found! Entire image was obscured for safety.")
```

---

## 🚫 Source File Overwrite Protection

To prevent inadvertent destruction of original image archives, `result.save()` refuses to overwrite source image files by default:

```python
result = detekface.anonymize("photo.jpg", method="blur")

# Raises ImageSaveError to protect the source file!
result.save("photo.jpg")

# Explicit override requires overwrite=True:
result.save("photo.jpg", overwrite=True)
```

---

## 🔒 100% Local Processing

All face detection and image redaction routines execute strictly in memory on your local CPU/GPU using OpenCV and NumPy. No image bytes or biometric data are transmitted across networks.

