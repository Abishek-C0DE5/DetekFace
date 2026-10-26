"""Example: Batch face detection across multiple images using detekface."""

import argparse
from pathlib import Path

import detekface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch detect faces in a directory or list of images with detekface.",
    )
    parser.add_argument(
        "images",
        nargs="*",
        default=[str(Path(__file__).parent.parent / "tests" / "assets" / "sample_face.jpg")],
        help="One or more image paths or a directory path (default: tests/assets/sample_face.jpg).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to save annotated images.",
    )
    parser.add_argument(
        "--crop-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to save unannotated face crops.",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(detekface.SUPPORTED_BACKENDS),
        default="yunet",
        help="Detection backend: 'yunet' (default) or 'mediapipe'.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Confidence threshold between 0.0 and 1.0 (default: 0.6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = args.images[0] if len(args.images) == 1 else args.images
    print(f"Running batch detection on: {input_paths} using backend '{args.backend}'")

    def progress(done: int, total: int) -> None:
        print(f"\rProgress: [{done}/{total}] {done / total * 100:.1f}%", end="")

    results = detekface.detect_many(
        input_paths,
        confidence_threshold=args.confidence,
        backend=args.backend,
        progress_callback=progress,
    )
    print("\n")

    print("=" * 60)
    print(" Batch Detection Summary")
    print("=" * 60)
    print(f"Total Processed : {len(results)}")
    print(f"Successful      : {len(results.successes)}")
    print(f"Failed          : {len(results.failures)}")
    print(f"Total Faces     : {results.total_faces}")
    print("=" * 60)

    # Save outputs if requested
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in results.successes:
            if item.result is not None:
                src_name = Path(str(item.source)).name
                item.result.save(out_dir / f"annotated_{src_name}", shape="hud", color="#00FF88")
        print(f"Saved annotated images to: {out_dir}")

    if args.crop_dir:
        crop_dir = Path(args.crop_dir)
        crop_dir.mkdir(parents=True, exist_ok=True)
        for item in results.successes:
            if item.has_faces and item.result is not None:
                src_stem = Path(str(item.source)).stem
                item.result.capture(save_dir=crop_dir, filename=f"crop_{src_stem}.jpg")
        print(f"Saved face crops to: {crop_dir}")


if __name__ == "__main__":
    main()
