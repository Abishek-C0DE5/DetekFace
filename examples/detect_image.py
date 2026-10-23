"""Example: Detecting faces in an image with detekface."""

from pathlib import Path
import sys

import detekface


def main() -> None:
    # Use the sample image from tests if no image is passed
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = str(Path(__file__).parent.parent / "tests" / "assets" / "sample_face.jpg")

    print(f"Running detekface on: {image_path}")

    try:
        result = detekface.detect(image_path, confidence_threshold=0.6)
    except detekface.DetekFaceError as e:
        print(f"Error running detekface: {e}")
        sys.exit(1)

    print(f"Detected {result.count} face(s) in image:\n")
    for i, face in enumerate(result.faces, start=1):
        bbox = face.bbox
        print(f"Face #{i}:")
        print(f"  Confidence : {face.confidence * 100:.2f}%")
        print(f"  BoundingBox: (x={bbox.x}, y={bbox.y}, width={bbox.width}, height={bbox.height})")
        print(f"  Coordinates: [x1={bbox.x1}, y1={bbox.y1}, x2={bbox.x2}, y2={bbox.y2}]")
        if face.landmarks:
            lm = face.landmarks
            print("  Landmarks  :")
            print(f"    Right Eye  : {lm.right_eye.as_tuple()}")
            print(f"    Left Eye   : {lm.left_eye.as_tuple()}")
            print(f"    Nose Tip   : {lm.nose_tip.as_tuple()}")
            print(f"    Right Mouth: {lm.right_mouth.as_tuple()}")
            print(f"    Left Mouth : {lm.left_mouth.as_tuple()}")
        print("-" * 40)


if __name__ == "__main__":
    main()
