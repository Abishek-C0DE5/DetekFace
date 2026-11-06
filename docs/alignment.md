# Face Alignment & Normalization in DetekFace

`detekface` 0.3.0 provides robust face alignment that straightens moderately tilted faces and extracts standardized, aspect-ratio-preserving facial crops for downstream biometric, avatar, or recognition pipelines.

---

## 🚀 Quick Usage

### Aligning the Primary Detected Face

```python
import detekface

result = detekface.detect("tilted_portrait.jpg")

# Straighten and crop face to 112x112 pixels
aligned = result.align(output_size=(112, 112), padding=0.25, center_on="eyes")

# Save normalized crop
aligned.save("aligned_face.jpg")
```

### Aligning All Faces in a Multi-Person Image

```python
import detekface

result = detekface.detect("group_photo.jpg")

# Returns a list of AlignedFace objects
aligned_faces = result.aligned_faces(output_size=(224, 224))
for idx, face_crop in enumerate(aligned_faces):
    face_crop.save(f"captures/person_{idx}.jpg")
```

---

## 📐 Alignment Algorithm & Geometry

1. **Rotation Estimation**: Calculates eye tilt angle $\theta$ from the left and right eye landmark positions:
   $$\theta = \arctan2(\Delta y, \Delta x) \cdot \frac{180}{\pi}$$
2. **Affine Transformation Matrix**: Computes an affine similarity matrix centering the rotation around the midpoint between the eyes (or bbox center when `center_on="face"`).
3. **Aspect Ratio Preservation**:
   - The face is scaled isotropically along both X and Y axes.
   - Proportions are strictly preserved—facial features are never artificially stretched or squished.
4. **Boundary Handling**: If alignment rotation or padding causes the bounding crop to touch or cross image boundaries, OpenCV border reflection (`cv2.BORDER_REFLECT_101`) or constant padding fills exterior regions cleanly without cropping artifacts.

---

## 📦 `AlignedFace` Dataclass Properties

| Attribute | Type | Description |
| :--- | :---: | :--- |
| `image` | `np.ndarray` | BGR uint8 NumPy array of the aligned face crop. |
| `width` | `int` | Width of the crop in pixels. |
| `height` | `int` | Height of the crop in pixels. |
| `rotation_deg` | `float` | Applied counter-rotation angle in degrees. |
| `scale_factor` | `float` | Isotropic scaling factor applied to the crop. |
| `center` | `Tuple[float, float]` | Original image coordinates of the transformation center point. |
| `transform_matrix` | `np.ndarray` | 2×3 affine transformation matrix used for warping. |
| `source_landmarks` | `Optional[FaceLandmarks]` | Original landmarks before rotation. |
| `aligned_landmarks` | `Optional[FaceLandmarks]` | Transformed landmarks mapped to crop coordinate space. |

---

## ⚙️ Configuration Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `output_size` | `Optional[Tuple[int, int]]` | `(112, 112)` | Output width and height in pixels. Set to `None` for natural unscaled dimensions. |
| `padding` | `float` | `0.25` | Fractional margin added around the face bounding box (e.g. `0.25` adds 25% padding). |
| `center_on` | `str` | `"eyes"` | Point of alignment center: `"eyes"` for eye midpoint or `"face"` for bbox center. |
| `border_mode` | `int` | `cv2.BORDER_REFLECT_101` | OpenCV border interpolation mode for boundary handling. |
| `border_value` | `Tuple[int, int, int]` | `(0, 0, 0)` | Fill color if using constant border mode. |

---

## 🛡️ Limitations

- **Extreme Incline (> 60°)**: Alignment handles moderate to severe head tilts ($\pm 45^\circ$). Upside-down faces (> 90° roll) should be pre-rotated prior to landmarking.
- **Occluded Eyes**: If eye landmarks cannot be localized due to heavy occlusion, alignment falls back safely to upright bbox orientation.

