"""Example 4: Still-image facial mesh visualization and saving.

Demonstrates detecting faces in a photo, extracting 478 dense facial landmarks,
rendering the topological wireframe mesh, and exporting the annotated image to disk.
"""

from pathlib import Path
import sys

import detekface

DEFAULT_IMAGE = Path(__file__).resolve().parent.parent / "tests" / "assets" / "sample_face.jpg"
OUTPUT_IMAGE = Path(__file__).resolve().parent.parent / "output_mesh.jpg"


def main() -> None:
    input_path = DEFAULT_IMAGE
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"[Error] Image not found at {input_path}")
        return

    print(f"[detekface] Detecting faces and dense mesh on {input_path.name}...")
    result = detekface.detect(input_path, mesh=True)
    print(f"[detekface] Detected {result.count} face(s).")

    for i, face in enumerate(result.faces, start=1):
        n_lms = len(face.dense_landmarks) if face.dense_landmarks else 0
        print(f"  Face #{i}: Box={face.bbox}, Confidence={face.confidence:.2f}, Dense Landmarks={n_lms}")

    # Save annotated image with neon mesh style to disk
    saved_path = result.save(
        OUTPUT_IMAGE,
        show_mesh=True,
        mesh_style="neon",
        mesh_color="#00FF88",
        mesh_thickness=1,
    )
    print(f"[detekface] Annotated mesh image successfully saved to: {saved_path}")


if __name__ == "__main__":
    main()
