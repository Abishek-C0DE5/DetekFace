# Real-Time Facial Gesture Detection in DetekFace

`detekface` 0.3.0 features an event-driven facial gesture engine supporting eye blinks, mouth movements, and head orientation gestures with asynchronous event callbacks.

---

## 🚀 Quick Usage

### Live Webcam Gesture Detection

```python
import detekface


def handle_gesture(event: detekface.GestureEvent) -> None:
    print(f"Triggered Gesture: {event.name.upper()} (Face #{event.track_id}, Duration: {event.duration_sec:.2f}s)")


# Run camera with gesture HUD and event callbacks
detekface.camera(
    detect_gestures=True,
    on_gesture=handle_gesture,
    show_gestures=True,
    shape="corners",
)
```

---

## 📋 Supported Gestures

| Gesture Constant | Name | Trigger Condition | Biometric Metric |
| :--- | :--- | :--- | :--- |
| `GESTURE_LEFT_EYE_BLINK` | `"left_eye_blink"` | Left eye closes and reopens within 0.05s–0.45s | EAR < 0.20 |
| `GESTURE_RIGHT_EYE_BLINK` | `"right_eye_blink"` | Right eye closes and reopens within 0.05s–0.45s | EAR < 0.20 |
| `GESTURE_BOTH_EYES_CLOSED` | `"both_eyes_closed"` | Both eyes closed simultaneously for > 0.40s | Left & Right EAR < 0.20 |
| `GESTURE_MOUTH_OPENING` | `"mouth_opening"` | Mouth opens wide for > 0.10s | MAR > 0.50 |
| `GESTURE_HEAD_TURN_LEFT` | `"head_turn_left"` | Head yaw rotates left beyond threshold | Yaw < -25.0° |
| `GESTURE_HEAD_TURN_RIGHT` | `"head_turn_right"` | Head yaw rotates right beyond threshold | Yaw > +25.0° |
| `GESTURE_HEAD_TILT` | `"head_tilt"` | Head roll tilts toward either shoulder | \|Roll\| > 20.0° |

---

## 🧮 Aspect Ratio Calculation

### Eye Aspect Ratio (EAR)
Calculates vertical opening relative to horizontal length:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

### Mouth Aspect Ratio (MAR)
Calculates vertical lip separation relative to mouth corner width:

$$\text{MAR} = \frac{\|p_{\text{upper}} - p_{\text{lower}}\|}{\|p_{\text{left}} - p_{\text{right}}\|}$$

---

## ⚙️ Customizing Thresholds via `GestureConfig`

```python
import detekface
from detekface.gestures import GestureConfig

config = GestureConfig(
    ear_closed_threshold=0.18,  # Stricter eye closure threshold
    mar_open_threshold=0.55,    # Wider mouth opening threshold
    min_blink_duration=0.04,    # Minimum duration in seconds
    max_blink_duration=0.40,    # Maximum duration for blink
    debounce_seconds=0.25,      # Anti-flapping cooldown between triggers
)

session = detekface.CameraSession(
    detect_gestures=True,
    gesture_config=config,
    on_gesture=lambda ev: print(f"Gesture: {ev.name}"),
)
```

---

## 🛡️ Limitations

- **Lighting & Occlusion**: Thick-rimmed glasses or heavy dark eye makeup can occasionally shift EAR baselines. Use `ear_closed_threshold` tuning if needed.
- **Extreme Profile Angles**: Eye aspect ratios cannot be evaluated when one eye is occluded in profile views.

