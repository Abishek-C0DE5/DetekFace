"""AR Face Accessories Webcam Example (detekface 0.3.0).

Demonstrates real-time augmented reality face accessories:
- Sunglasses
- Royal Crown
- Protective Face Mask
- Futuristic Visor
- Angelic Halo

Controls:
- Press 1-5 to switch accessories live:
    [1] Sunglasses
    [2] Crown
    [3] Face Mask
    [4] Futuristic Visor
    [5] Halo
- Press 'O' to toggle off accessories
- Press 'Q' or Esc to quit
"""

import cv2
import detekface

ACCESSORIES = [
    "sunglasses",
    "crown",
    "face_mask",
    "futuristic_visor",
    "halo",
]


def main() -> None:
    print("Starting detekface AR accessories webcam feed...")
    print("Keyboard shortcuts:")
    print("  [1] Sunglasses")
    print("  [2] Crown")
    print("  [3] Face Mask")
    print("  [4] Visor")
    print("  [5] Halo")
    print("  [O] Turn off accessories")
    print("  [Q] Quit")

    current_accessory = "sunglasses"

    with detekface.open_camera(async_mode=True, target_fps=60.0) as session:
        while session.is_open:
            result = session.read()
            if not result.has_faces:
                # Render clean frame if no faces
                frame = session.latest_frame.copy() if session.latest_frame is not None else None
            else:
                frame = session.render(
                    shape="corners",
                    effect=current_accessory,
                    show_fps=True,
                    show_count=True,
                )

            if frame is not None:
                cv2.imshow("detekface - AR Accessories", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == ord("1"):
                current_accessory = "sunglasses"
                print("Switched accessory to: sunglasses")
            elif key == ord("2"):
                current_accessory = "crown"
                print("Switched accessory to: crown")
            elif key == ord("3"):
                current_accessory = "face_mask"
                print("Switched accessory to: face_mask")
            elif key == ord("4"):
                current_accessory = "futuristic_visor"
                print("Switched accessory to: futuristic_visor")
            elif key == ord("5"):
                current_accessory = "halo"
                print("Switched accessory to: halo")
            elif key in (ord("o"), ord("O")):
                current_accessory = None
                print("Turned off accessories")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

