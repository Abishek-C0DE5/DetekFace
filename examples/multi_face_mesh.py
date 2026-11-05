"""Example 5: Multiple-face mesh visualization and rendering.

Demonstrates independent dense facial mesh rendering across multiple faces in a single photo or frame.
"""

from pathlib import Path
import sys

import cv2
import numpy as np

import detekface

MULTI_FACE_SAMPLE = Path(__file__).resolve().parent.parent / "tests" / "assets" / "sample_face.jpg"
OUTPUT_MULTI_MESH = Path(__file__).resolve().parent.parent / "output_multi_mesh.jpg"


def main() -> None:
    # If a specific image path is provided via command line, use it
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        if img_path.exists():
            result = detekface.detect(img_path, mesh=True)
        else:
            print(f"[Error] Image not found: {img_path}")
            return
    else:
        # Create a synthetic dual-face canvas for reproducible demonstration
        base = (
            cv2.imread(str(MULTI_FACE_SAMPLE))
            if MULTI_FACE_SAMPLE.exists()
            else np.zeros((480, 800, 3), dtype=np.uint8)
        )
        h, w = base.shape[:2]
        canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
        canvas[:, :w] = base
        canvas[:, w:] = cv2.flip(base, 1)

        result = detekface.detect(canvas, mesh=True)

    print(f"[detekface] Detected {result.count} face(s) in multi-face composition.")
    for idx, face in enumerate(result.faces, start=1):
        lms_count = len(face.dense_landmarks) if face.dense_landmarks else 0
        print(f"  Face #{idx}: bbox={face.bbox}, confidence={face.confidence:.2%}, landmarks={lms_count}")

    # Save multi-face image with distinct wireframe mesh overlays
    saved = result.save(
        OUTPUT_MULTI_MESH,
        shape="hud",
        color="cyan",
        show_confidence=True,
        show_mesh=True,
        mesh_style="wireframe",
    )
    print(f"[detekface] Saved multi-face annotated mesh visualization to: {saved}")


if __name__ == "__main__":
    main()
