"""Real-time live face detection, tracking, 3D head pose, and gesture detection preview.

Controls:
- Look at camera and try:
  * Wink left or right eye
  * Open your mouth
  * Turn your head left or right
  * Tilt your head sideways
- Press 'Q' or 'Escape' in the preview window to exit.
- Press 'C' to capture and save a face snapshot.
"""

import sys
import detekface


def on_gesture_detected(event: detekface.GestureEvent) -> None:
    """Print detected facial gesture events to terminal console."""
    track_str = f"Face ID: {event.track_id}" if event.track_id is not None else f"Face #{event.face_index}"
    print(f"\n[GESTURE DETECTED] >> {event.name.upper()} << | {track_str} | Duration: {event.duration:.2f}s")


def main() -> None:
    print("=" * 65)
    print(" detekface 0.3.0 - Live Webcam Face & Gesture Detection")
    print("=" * 65)
    print("Starting webcam preview...")
    print("Features enabled:")
    print("  * Temporary multi-face tracking (Face ID: 1, Face ID: 2)")
    print("  * 3D Head-Pose Estimation (Yaw, Pitch, Roll & 3D axes)")
    print("  * Real-Time Gesture Detection (Blinks, Winks, Mouth, Tilt, Turn)")
    print()
    print("Look at the camera window on your screen and test:")
    print("  1. Blink or wink with one eye (Left / Right wink)")
    print("  2. Open your mouth (Mouth opening)")
    print("  3. Turn your head left or right (Head turn)")
    print("  4. Tilt your head sideways toward your shoulder (Head tilt)")
    print()
    print("Press 'Q' or 'Escape' on your keyboard to exit.\n")

    try:
        detekface.camera(
            camera_index=0,
            mirror=True,
            shape="corners",
            color="#00FF88",
            track_faces=True,
            show_tracking_ids=True,
            show_pose=True,
            detect_gestures=True,
            show_gestures=True,
            show_fps=True,
            show_count=True,
            on_gesture=on_gesture_detected,
            window_title="detekface - Live Face Gesture Detection",
        )
    except detekface.CameraNotFoundError as e:
        print(f"\n[ERROR] Camera device not accessible: {e.message}")
        print(f"Hint: {e.hint}")
        sys.exit(1)
    except detekface.DetekFaceError as e:
        print(f"\n[ERROR] {e.message}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nWebcam preview terminated by user.")


if __name__ == "__main__":
    main()

