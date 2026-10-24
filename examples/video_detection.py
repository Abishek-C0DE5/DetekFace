"""Example: Detecting faces in a prerecorded video file with detekface."""

import argparse
import sys

import detekface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a prerecorded video file, detect faces frame by frame, and optionally save annotated video.",
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to the input video file (e.g. clip.mp4).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="PATH",
        help="Destination path to save the annotated video (e.g. annotated_clip.mp4).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run in headless mode without opening a display window.",
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
        help="Line stroke thickness in pixels (default: 3).",
    )
    parser.add_argument(
        "--show-confidence",
        action="store_true",
        help="Render detection confidence percentage badges near each face.",
    )
    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Render processing throughput FPS counter.",
    )
    parser.add_argument(
        "--show-count",
        action="store_true",
        help="Render face count banner.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process before stopping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Starting detekface video processing on: {args.video}")
    if args.output:
        print(f"Output will be saved to: {args.output}")

    try:
        saved_path = detekface.video(
            path=args.video,
            output_path=args.output,
            show=not args.no_show,
            shape=args.shape,
            color=args.color,
            thickness=args.thickness,
            show_confidence=args.show_confidence,
            show_fps=args.show_fps,
            show_count=args.show_count,
            max_frames=args.max_frames,
            on_capture=lambda face: print(f"[Captured] Saved face to: {face.path}"),
        )
        if saved_path:
            print(f"Annotated video successfully saved to: {saved_path}")
        else:
            print("Video processing finished.")
    except detekface.DetekFaceError as e:
        print(f"Error processing video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
