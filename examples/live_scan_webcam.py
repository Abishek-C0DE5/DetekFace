"""Example 3: Modern computer-vision scanning animation webcam preview.

Demonstrates real-time animated horizontal laser scanning beam across detected faces,
subtle HUD corner locking reticles, and live telemetry tags ([FACE LOCKED], [SCANNING]).
Press 'F' to cycle Live FX styles, 'Q' or 'Escape' to exit.
"""

import detekface


def main() -> None:
    print("[detekface] Starting Live FX animated scanner webcam...")
    print("[detekface] Controls: Press 'F' to cycle Live FX styles, 'Q' or 'ESC' to exit.")

    try:
        detekface.camera(
            camera_index=0,
            show_scan=True,
            live_fx=True,
            live_fx_style="scan",
            show_fps=True,
            show_count=True,
        )
    except detekface.CameraNotFoundError as e:
        print(f"[Error] {e.message}\nHint: {e.hint}")
    except detekface.DependencyNotFoundError as e:
        print(f"[Error] {e.message}\nHint: {e.hint}")


if __name__ == "__main__":
    main()
