"""Head Pose Estimation Webcam Example (detekface 0.3.0).

Demonstrates real-time 3D head pose estimation:
- Calculates Yaw (turning left/right), Pitch (tilting up/down), and Roll (tilting sideways)
- Projects 3D Cartesian orientation axes (Red=X, Green=Y, Blue=Z)
- Displays pose angle telemetry in degrees
"""

import detekface


def main() -> None:
    print("Starting detekface 3D head pose estimation webcam stream...")
    print("Yaw (turning left/right), Pitch (looking up/down), Roll (tilting)")
    print("Press 'Q' or Esc to exit.")

    detekface.camera(
        estimate_pose=True,
        show_pose=True,
        show_fps=True,
        show_count=True,
        color="#00D4FF",
        shape="corners",
        window_title="detekface - 3D Head Pose Estimation",
    )


if __name__ == "__main__":
    main()

