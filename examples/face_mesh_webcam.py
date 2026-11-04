"""Example 1: Basic face mesh live webcam preview.

Demonstrates opening an interactive webcam feed with 478 dense facial landmark wireframe rendering.
Press 'Q' or 'Escape' in the preview window to exit.
Press 'M' to cycle between mesh visualization styles live.
"""

import detekface


def main() -> None:
    print("[detekface] Starting basic live facial mesh webcam...")
    print("[detekface] Controls: Press 'M' to cycle mesh styles, 'Q' or 'ESC' to exit.")

    try:
        detekface.camera(
            camera_index=0,
            show_mesh=True,
            mesh_style="wireframe",
            color="#00FF88",
            show_fps=True,
            show_count=True,
        )
    except detekface.CameraNotFoundError as e:
        print(f"[Error] {e.message}\nHint: {e.hint}")
    except detekface.DependencyNotFoundError as e:
        print(f"[Error] {e.message}\nHint: {e.hint}")


if __name__ == "__main__":
    main()
