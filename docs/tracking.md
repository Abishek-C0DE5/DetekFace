# Multi-Face Tracking in DetekFace

`detekface` 0.3.0 provides real-time multi-face tracking that assigns persistent integer IDs to detected faces across video and webcam frames.

---

## 🚀 Quick Usage

### Live Webcam Tracking

```python
import detekface

# Launch camera with multi-face tracking IDs displayed in HUD
detekface.camera(
    track_faces=True,
    show_tracking_ids=True,
    shape="hud",
    color="#00FF88",
)
```

### Video File Tracking

```python
import detekface

detekface.video(
    "crowd.mp4",
    output_path="tracked_crowd.mp4",
    track_faces=True,
    show_tracking_ids=True,
)
```

---

## 🧠 Tracking Architecture

The `FaceTracker` uses an Intersection-over-Union (IoU) spatial association algorithm optimized for real-time performance:

1. **Detection Input**: Receives bounding boxes from the detector once per frame.
2. **IoU Cost Matrix**: Computes spatial overlap between predicted previous-frame track positions and incoming detections.
3. **Hungarian / Greedy Matching**: Matches detections to existing tracks when `IoU >= min_iou` (default `0.30`).
4. **Track Lifecycle**:
   - **Tentative**: New tracks are initialized upon detection.
   - **Confirmed**: Promoted to confirmed after `min_hits` consecutive matches.
   - **Lost / Grace Period**: If momentarily occluded, tracks are preserved for `max_age` frames (default `15` frames) before termination.
5. **Output**: Each `FaceDetection` object carries a stable `track_id: Optional[int]` property.

---

## 💻 Programmatic Usage via `CameraSession`

```python
import detekface

with detekface.open_camera(async_mode=True, track_faces=True) as session:
    while session.is_open:
        result = session.read()
        for face in result.faces:
            print(f"Face #{face.track_id}: BoundingBox={face.bbox}, Conf={face.confidence:.2f}")

        # Render with tracking badges
        frame = session.render(shape="hud")
```

---

## ⚙️ Configuration Parameters

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `track_faces` | `bool` | `False` | Enables persistent multi-face tracking engine. |
| `show_tracking_ids` | `bool` | `False` | Renders `[ID #X]` badge over bounding boxes. Automatically sets `track_faces=True`. |
| `max_age` | `int` | `15` | Frames to keep a lost face alive across occlusions before expiring. |
| `min_iou` | `float` | `0.30` | Minimum bounding-box IoU overlap required to maintain identity continuity. |

---

## 🛡️ Limitations

- **Pure Spatial/Kinematic Tracking**: Uses bounding box geometry and motion continuity without deep facial recognition embeddings. If two faces completely cross or swap positions during heavy occlusion, ID swapping can theoretically occur.
- **Extreme Camera Movement**: Rapid camera pan/shake exceeding bounding box overlap requires a lower `min_iou` or higher capture framerate.

