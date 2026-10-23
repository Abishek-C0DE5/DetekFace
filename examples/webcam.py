"""Live webcam face detection example using detekface."""

import argparse
import sys

import detekface


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam face detection with detekface.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring of the webcam stream.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Face detection confidence threshold between 0.0 and 1.0 (default: 0.7).",
    )

    args = parser.parse_args()

    print("Starting detekface live webcam preview...")
    print(f"Camera index : {args.camera_index}")
    print(f"Mirror feed  : {not args.no_mirror}")
    print("Press 'q' or 'Escape' in the preview window to exit.\n")

    try:
        detekface.camera(
            camera_index=args.camera_index,
            mirror=not args.no_mirror,
            confidence_threshold=args.confidence,
        )
    except detekface.CameraError as e:
        print(f"Camera error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nWebcam preview terminated by user.")


if __name__ == "__main__":
    main()
