"""Multi-Face Tracking & 60 FPS Async Camera Example (detekface 0.3.0).

Demonstrates real-time multi-face tracking with persistent IDs:
- IoU-based bounding box association
- Tracking identity continuity across brief occlusions
- High-framerate asynchronous preview decoupled from model latency
- Real-time telemetry: render FPS, capture FPS, inference latency
"""

import detekface


def main() -> None:
    print("Starting detekface multi-face tracking webcam stream...")
    print("Targeting 60 FPS async decoupled preview.")
    print("Press 'Q' or Esc to exit.")

    detekface.camera(
        async_mode=True,
        target_fps=60.0,
        track_faces=True,
        show_tracking_ids=True,
        show_fps=True,
        show_count=True,
        shape="hud",
        color="#00FF88",
        window_title="detekface - Multi-Face Tracking (60 FPS)",
    )


if __name__ == "__main__":
    main()

