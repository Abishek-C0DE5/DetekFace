# Quickstart Guide

This guide introduces fundamental face detection workflows using DetekFace, explains key concepts in plain language, and walks through your first practical programs.

---

## Key Terms Explained for Beginners

Before writing code, it helps to understand a few common computer vision terms:

- **Frame**: A single digital image within a video or camera stream. A live webcam feed is simply a continuous sequence of individual frames displayed rapidly.
- **Bounding Box**: An imaginary or drawn rectangular box that encloses a detected face. In DetekFace, it is defined by pixel coordinates: `(x, y)` for the top-left corner, and its `width` and `height`.
- **Confidence Score**: A number between `0.0` (0%) and `1.0` (100%) indicating how certain the neural network is that a region is indeed a human face. A threshold of `0.7` means only detections with 70% or greater certainty are returned.
- **Facial Landmarks**: Distinct anatomical points located on the face. DetekFace predicts 5 keypoints: the center of the right eye, center of the left eye, tip of the nose, right corner of the mouth, and left corner of the mouth.
- **FPS (Frames Per Second)**: The number of frames processed and rendered per second. Standard webcams usually operate at 30 FPS.

---

## 1. Live Camera Detection

The simplest DetekFace program opens your default camera and displays face detections:

```python
import detekface

# Open default webcam with real-time face detection
detekface.camera()
```

### What Happens Behind the Scenes:
1. DetekFace checks whether the YuNet model is cached locally; if not, it automatically downloads and verifies it.
2. It opens camera device `0`.
3. It captures video frames, passes each to the deep learning model, and draws green bounding boxes around faces.
4. Pressing **`Q`** or **`Escape`** releases the camera and closes the window.

### Customizing Shapes and Colors:
You can easily change the overlay style and color:
```python
import detekface

# Launch camera with a neon cyan HUD overlay
detekface.camera(shape="hud", color="cyan", thickness=2)
```

*(See [Webcam Documentation](webcam.md) for all 5 shapes and color options).*

---

## 2. Detecting Faces in a Still Photo

To analyze an existing photograph on disk:

```python
import detekface

# Run detection on an image file
result = detekface.detect("photo.jpg", confidence_threshold=0.6)

print(f"Total faces detected: {result.count}")

# Iterate through detected faces
for i, face in enumerate(result.faces, start=1):
    print(f"Face #{i}:")
    print(f"  Confidence : {face.confidence * 100:.1f}%")
    print(f"  Top-Left   : x={face.bbox.x}, y={face.bbox.y}")
    print(f"  Dimensions : width={face.bbox.width}px, height={face.bbox.height}px")
```

---

## 3. Displaying and Saving Annotated Results

DetekFace allows you to display and save images with visual indicators without manual OpenCV drawing code:

```python
import detekface

result = detekface.detect("group.jpg")

# 1. Open an interactive preview window (press any key to close)
result.show(shape="hud", color="#00FF88", show_confidence=True)

# 2. Save the annotated image to a file
result.save("group_annotated.jpg", shape="hud", color="#00FF88", show_confidence=True)
```

---

## 4. Cropping Faces for Profile Avatars

To extract just the face portion of an image:

```python
import detekface

result = detekface.detect("headshot.jpg")

# Automatically crop and save the largest detected face
face_crop = result.capture(save_dir="avatars", filename="my_avatar.jpg")

print(f"Saved cropped face to: {face_crop.path}")
print(f"Cropped dimensions   : {face_crop.width}x{face_crop.height} pixels")
```

> **Important**: The cropped face is always taken from the clean original image. Even if you draw HUD or rectangular overlays on the image, the captured face crop remains completely free of lines or text.

---

## Next Steps

- For advanced camera settings and keyboard controls, see [Webcam Documentation](webcam.md).
- To learn more about bounding box geometry and facial landmarks, see [Still-Image Detection](image-detection.md).
- To customize overlays, see [Configuration](configuration.md).
