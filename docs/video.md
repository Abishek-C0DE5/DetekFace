# Video Processing Guide

The `detekface.video()` function enables frame-by-frame face detection, real-time GUI preview, interactive photo cropping, and annotated video export for prerecorded video files (`.mp4`, `.avi`, `.mov`, `.mkv`).

---

## 1. Quick Start: Interactive Video Player

To open and watch a video file with real-time face detection overlays:

```python
import detekface

# Open video with real-time detection overlay
detekface.video("sample_video.mp4")
```

### Keyboard Controls During Playback:
| Key | Action | Description |
| :---: | :--- | :--- |
| <kbd>C</kbd> | **Capture Face** | Extracts the largest visible face from the clean original video frame and saves it to `./captures/`. |
| <kbd>Q</kbd> | **Quit** | Gracefully closes the window and terminates playback. |
| <kbd>Esc</kbd> | **Emergency Exit** | Immediately closes all OpenCV windows and stops the loop. |

---

## 2. Exporting Annotated Video Files

To write an annotated copy of the video with drawn bounding boxes, styling, and status banners:

```python
import detekface

output_path = detekface.video(
    path="conference.mp4",
    output_path="conference_annotated.mp4",
    shape="hud",
    color="#00FF88",
    thickness=2,
    show_confidence=True,
    show_fps=True,
    show_count=True,
)

print(f"Saved annotated video to: {output_path}")
```

> **Audio Note**: `detekface.video()` operates on visual frames only; audio streams are not re-encoded into the exported file.

---

## 3. Headless Processing for Servers & Cloud Pipelines

When running in a Docker container, headless Linux server, or automated processing pipeline without an attached monitor, set `show=False` to disable desktop window rendering:

```python
import detekface

# Process video in background without creating a GUI desktop window
detekface.video(
    path="security_footage.mp4",
    output_path="security_annotated.mp4",
    show=False,  # Headless mode: no cv2.imshow calls
    shape="corners",
    color="cyan",
)
```

---

## 4. Performance Optimization: Frame Skipping (`frame_skip`)

Face detection on high-resolution 60 FPS video can be computationally demanding. By setting `frame_skip`, DetekFace runs neural network inference every `(N + 1)` frames while reusing detection boxes on the skipped frames.

This delivers a massive throughput improvement with almost zero visual difference:

```python
import detekface

# Detect on every 2nd frame (2x throughput boost):
detekface.video(
    "traffic.mp4",
    output_path="traffic_fast.mp4",
    frame_skip=1,  # Skips 1 frame between detections
    show=False,
)

# Detect on every 3rd frame (3x throughput boost):
detekface.video(
    "traffic.mp4",
    output_path="traffic_fastest.mp4",
    frame_skip=2,  # Skips 2 frames between detections
    show=False,
)
```

---

## 5. Input Resolution Downscaling (`input_size`)

Combine frame skipping with model resolution controls for maximum processing speed:

```python
import detekface

detekface.video(
    "4k_video.mp4",
    output_path="4k_annotated.mp4",
    input_size=(640, 360),  # Scales 4K frame down to 640x360 for lightning-fast inference
    frame_skip=1,
    show=False,
)
```

Bounding boxes and landmarks are automatically rescaled to match the video's original resolution upon export!

---

## 6. Programmatic Video Capture Callbacks (`on_capture`)

Receive captured face objects programmatically whenever the user presses <kbd>C</kbd> during video playback:

```python
import detekface


def handle_video_capture(capture: detekface.FaceCapture) -> None:
    print(f"Captured face at {capture.width}x{capture.height}px")
    print(f"Detection confidence: {capture.confidence:.1%}")
    print(f"Saved file path: {capture.path}")


detekface.video("interview.mp4", on_capture=handle_video_capture)
```

---

## 7. Command-Line Interface (CLI)

Process videos directly from terminal or bash scripts:

```bash
# Watch video with HUD styling:
detekface video interview.mp4 --shape hud --color "#00FF88"

# Export annotated video without desktop preview:
detekface video input.mp4 --output output.mp4 --no-preview

# Fast processing with frame skipping:
detekface video input.mp4 --output output.mp4 --frame-skip 2 --no-preview

# Process only the first 300 frames:
detekface video input.mp4 --output output.mp4 --max-frames 300
```
