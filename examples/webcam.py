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
    parser.add_argument(
        "--capture-dir",
        type=str,
        default="captures",
        help="Directory where cropped face captures will be saved (default: 'captures').",
    )

    args = parser.parse_args()

    def handle_capture(captured: detekface.FaceCapture) -> None:
        print(f"\n[CALLBACK] Successfully captured face! Size: {captured.width}x{captured.height}px")
        if captured.path:
            print(f"[CALLBACK] File saved at: {captured.path}")

    print("Starting detekface live webcam preview...")
    print(f"Camera index : {args.camera_index}")
    print(f"Mirror feed  : {not args.no_mirror}")
    print(f"Capture dir  : {args.capture_dir}")
    print("Controls     : Press 'c' to capture face, 'q' or 'Escape' to exit.\n")

    try:
        detekface.camera(
            camera_index=args.camera_index,
            mirror=not args.no_mirror,
            confidence_threshold=args.confidence,
            capture_dir=args.capture_dir,
            on_capture=handle_capture,
        )
    except detekface.CameraError as e:
        print(f"Camera error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nWebcam preview terminated by user.")


if __name__ == "__main__":
    main()
