"""Programmatic face cropping and capture example using detekface."""

from pathlib import Path
import sys

import detekface


def main() -> None:
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = str(Path(__file__).parent.parent / "tests" / "assets" / "sample_face.jpg")

    print(f"Detecting faces in: {image_path}")

    # 1. Detect faces
    result = detekface.detect(image_path, confidence_threshold=0.6)
    print(f"Found {result.count} face(s).")

    if not result.has_faces:
        print("No faces detected to capture.")
        return

    # 2. Crop and capture the largest detected face
    face = detekface.capture(result, save_dir="captures", filename="cropped_sample_face.jpg")

    print("\n--- Captured Face Details ---")
    print(f"Resolution : {face.width}x{face.height} pixels")
    print(f"Confidence : {face.confidence * 100:.1f}%")
    print(f"Saved Path : {face.path}")
    print(f"Coordinates: (x={face.bbox.x}, y={face.bbox.y}, w={face.bbox.width}, h={face.bbox.height})")


if __name__ == "__main__":
    main()
