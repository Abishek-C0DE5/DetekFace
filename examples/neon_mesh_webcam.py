"""Example 2: Neon glowing facial mesh webcam preview.

Demonstrates real-time dense facial landmark wireframe rendering with the 'neon' style,
highlighting key anatomical anchors (irises, nose bridge, lip corners) and emphasized contours.
Press 'Q' or 'Escape' to exit.
"""

import detekface


def main() -> None:
    print("[detekface] Starting Neon live facial mesh webcam...")
    print("[detekface] Controls: Press 'M' to cycle styles, 'C' to capture face, 'Q' or 'ESC' to exit.")

    try:
        detekface.camera(
            camera_index=0,
            show_mesh=True,
            mesh_style="neon",
            mesh_color="#00FF88",
            shape="corners",
            show_fps=True,
            show_confidence=True,
        )
    except detekface.CameraNotFoundError as e:
        print(f"[Error] {e.message}\nHint: {e.hint}")
    except detekface.DependencyNotFoundError as e:
        print(f"[Error] {e.message}\nHint: {e.hint}")


if __name__ == "__main__":
    main()
