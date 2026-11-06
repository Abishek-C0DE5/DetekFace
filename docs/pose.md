# 3D Head-Pose Estimation in DetekFace

`detekface` 0.3.0 includes real-time 3D head-pose estimation, computing head rotation angles (Yaw, Pitch, Roll in degrees) and projecting 3D Cartesian orientation axes directly from facial landmarks.

---

## 🚀 Quick Usage

### Live Webcam 3D Pose HUD

```python
import detekface

# Launch camera with 3D pose axes and angle HUD
detekface.camera(
    estimate_pose=True,
    show_pose=True,
    shape="corners",
    color="#00D4FF",
)
```

### Still Image Pose Analysis

```python
import detekface

result = detekface.detect("portrait.jpg", estimate_pose=True)

for face in result.faces:
    if face.head_pose is not None:
        pose = face.head_pose
        print(f"Yaw  : {pose.yaw:+.1f}°  (negative=left, positive=right)")
        print(f"Pitch: {pose.pitch:+.1f}° (negative=up, positive=down)")
        print(f"Roll : {pose.roll:+.1f}°  (negative=tilt left, positive=tilt right)")
```

---

## 📐 Mathematical Model & Coordinate System

Head pose estimation operates via Perspective-n-Point (PnP) pose solving using a 3D canonical facial model and the camera intrinsic matrix:

- **Yaw ($\psi$)**: Horizontal turning angle (turning head left or right). Range $[-90^\circ, +90^\circ]$.
- **Pitch ($\theta$)**: Vertical inclination angle (looking up or looking down). Range $[-90^\circ, +90^\circ]$.
- **Roll ($\phi$)**: Lateral tilt angle (tilting head toward left or right shoulder). Range $[-180^\circ, +180^\circ]$.

### 3D Orientation Axes

When `show_pose=True` is enabled, Cartesian orientation axes are projected outward from the nose bridge keypoint:
- 🔴 **Red Axis (X)**: Points along the transverse axis (lateral left/right).
- 🟢 **Green Axis (Y)**: Points downward along the vertical facial axis.
- 🔵 **Blue Axis (Z)**: Points directly outward along the facial normal vector.

---

## 🌊 Temporal Pose Smoothing

In real-time webcam feeds, raw frame-by-frame landmark jitter can cause noisy rotational spikes. DetekFace incorporates a temporal exponential moving average (EMA) smoother (`TemporalPoseSmoother`):

$$\text{Pose}_t = \alpha \cdot \text{Pose}_{\text{raw}} + (1 - \alpha) \cdot \text{Pose}_{t-1}$$

This produces smooth visual orientation axes while retaining responsiveness during swift head turns.

---

## 💻 Video Processing with Pose Overlays

```python
import detekface

detekface.video(
    "interview.mp4",
    output_path="interview_pose.mp4",
    estimate_pose=True,
    show_pose=True,
)
```

---

## 🛡️ Limitations

- **Extreme Profile Views**: Landmark detection degradation beyond $\pm 75^\circ$ yaw can lead to inaccurate pitch estimation.
- **Intrinsic Camera Matrix Approximation**: When focal length is not explicitly supplied by hardware calibration, DetekFace uses standard pinhole focal length approximation ($f_x = f_y \approx \text{image width}$).

