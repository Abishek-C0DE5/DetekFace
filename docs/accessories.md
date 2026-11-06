# AR Face Accessories in DetekFace

`detekface` 0.3.0 includes a modular augmented reality (AR) accessory engine that renders transparent virtual accessories on detected faces with smooth landmark tracking and head orientation alignment.

---

## 🚀 Quick Usage

### Live Webcam AR Accessories

```python
import detekface

# Launch camera with AR sunglasses
detekface.camera(
    effect="sunglasses",
    shape="corners",
)
```

### Still Image AR Overlay

```python
import detekface

result = detekface.detect("portrait.jpg")

# Render with futuristic visor
annotated = result.annotate(
    effect="futuristic_visor",
    accessory_opacity=0.9,
    shape="none",  # Suppress bounding boxes for pure AR mode
)
result.save("futuristic_portrait.jpg", effect="futuristic_visor", shape="none")
```

---

## 🎭 Built-In Accessories Catalog

| Accessory Identifier | Visual Style | Anchor Anatomical Keypoints | Best Suited For |
| :--- | :--- | :--- | :--- |
| `"sunglasses"` | Classic dark-tinted sunglasses | Left & right eye pupils, inter-pupillary axis | Avatars, social photo filters |
| `"crown"` | Golden regal crown | Upper forehead hairline, facial vertical axis | Gamification, celebration filters |
| `"face_mask"` | Medical protective mask | Nose bridge, mouth corners, chin lower edge | Compliance checks, avatars |
| `"futuristic_visor"` | Sci-fi glowing HUD visor | Full eye span, temporal temples | Tech demos, cyber aesthetic |
| `"halo"` | Floating golden angelic ring | Top of head, floating offset | Fun photo overlays |

---

## 📐 Dynamic Transformation & Alignment

Each accessory automatically adjusts in real-time according to:

1. **Center of Anchor**: Computed from landmark coordinates (e.g. eye midpoint for glasses, forehead edge for crown).
2. **Dynamic Scale**: Proportional to inter-pupillary distance or bounding box width:
   $$\text{Scale} = \frac{\text{Inter-Pupillary Distance}}{\text{Asset Pupil Distance}} \cdot \text{scale\_adjust}$$
3. **2D & 3D Angular Tilt**: Rotates dynamically to match facial roll and yaw angles.
4. **Temporal Motion Smoothing**: Exponential moving average prevents jitter on minor webcam landmark shifts.
5. **Alpha Blending**: True per-pixel alpha transparency overlaying original frame pixels without bounding artifact boxes.

---

## ⚙️ Fine-Tuning Scale & Offsets

```python
import detekface

detekface.camera(
    effect="crown",
    accessory_opacity=0.95,  # Transparency in [0.0, 1.0]
    accessory_scale=1.10,    # 10% larger than default
    accessory_y_offset=-0.05, # Shift slightly upward
)
```

---

## 🛡️ Limitations

- **Asset Licensing**: All built-in accessories are original vector designs licensed under the same MIT license as DetekFace.
- **Occlusion Handling**: If eye landmarks are severely occluded (e.g. hand covering face), the accessory falls back smoothly to estimated bounding box proportions.

