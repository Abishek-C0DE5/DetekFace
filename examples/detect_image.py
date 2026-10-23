"""Example: Detecting faces in an image with detekface and rendering overlays."""

import argparse
from pathlib import Path
import sys

import detekface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect faces in an image, display count/coordinates, and optionally show or save annotated results.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=str(Path(__file__).parent.parent / "tests" / "assets" / "sample_face.jpg"),
        help="Path to an image file (default: tests/assets/sample_face.jpg).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to save annotated image (e.g. annotated.jpg).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated image in a GUI window.",
    )
    parser.add_argument(
        "--shape",
        choices=sorted(detekface.SUPPORTED_SHAPES),
        default="hud",
        help="Overlay shape style: rectangle, corners, circle, rounded, hud (default: hud).",
    )
    parser.add_argument(
        "--color",
        default="#00FF88",
        help="Overlay color hex string or name (default: #00FF88).",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=3,
        help="Stroke thickness (default: 3).",
    )
    parser.add_argument(
        "--show-confidence",
        action="store_true",
        help="Render confidence score badges near each face.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Running detekface on: {args.image}")

    try:
        result = detekface.detect(args.image, confidence_threshold=0.6)
    except detekface.DetekFaceError as e:
        print(f"Error running detekface: {e}")
        sys.exit(1)

    print(f"\nResult count: {result.count} face(s) found.")
    print(f"Result faces: {len(result.faces)} detection objects:")

    for i, face in enumerate(result.faces, start=1):
        bbox = face.bbox
        print(f"\nFace #{i}:")
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

    if args.save:
        saved_path = result.save(
            args.save,
            shape=args.shape,
            color=args.color,
            thickness=args.thickness,
            show_confidence=args.show_confidence,
        )
        print(f"\nSaved annotated image to: {saved_path}")

    if args.show:
        print("\nDisplaying annotated window (press any key or close window to exit)...")
        result.show(
            shape=args.shape,
            color=args.color,
            thickness=args.thickness,
            show_confidence=args.show_confidence,
        )


if __name__ == "__main__":
    main()
