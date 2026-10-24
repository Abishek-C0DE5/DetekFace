"""Command-line interface for detekface."""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

import detekface
from detekface import __version__
from detekface.drawing import SUPPORTED_SHAPES
from detekface.exceptions import DetekFaceError


def build_parser() -> argparse.ArgumentParser:
    """Construct the detekface command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="detekface",
        description="DetekFace - Lightweight, high-performance face detection CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  detekface camera\n"
            "  detekface camera --backend mediapipe --shape hud\n"
            "  detekface image photo.jpg --output annotated.jpg\n"
            "  detekface image photo.jpg --crop-face face.jpg --overwrite\n"
            "  detekface batch ./images --output ./results\n"
            "  detekface video input.mp4 --output output.mp4\n"
        ),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program version number and exit.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Commands",
        description="Run 'detekface <command> --help' for detailed options on each command.",
        metavar="<command>",
    )

    # -------------------------------------------------------------------------
    # Subcommand: camera
    # -------------------------------------------------------------------------
    p_cam = subparsers.add_parser(
        "camera",
        help="Open live webcam face detection feed.",
        description="Stream real-time webcam feed with customizable face detection overlays.",
    )
    p_cam.add_argument(
        "--device",
        "-d",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    p_cam.add_argument(
        "--backend",
        choices=["yunet", "mediapipe"],
        default="yunet",
        help="Detection backend (default: yunet).",
    )
    p_cam.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.7,
        help="Minimum confidence threshold between 0.0 and 1.0 (default: 0.7).",
    )
    p_cam.add_argument(
        "--shape",
        "-s",
        choices=sorted(SUPPORTED_SHAPES),
        default="rectangle",
        help="Overlay shape style (default: rectangle).",
    )
    p_cam.add_argument(
        "--color",
        default="#00FF88",
        help="Overlay color as hex code (#00FF88) or color name (default: #00FF88).",
    )
    p_cam.add_argument(
        "--thickness",
        "-t",
        type=int,
        default=3,
        help="Overlay border thickness in pixels (default: 3).",
    )
    p_cam.add_argument(
        "--show-confidence",
        action="store_true",
        help="Display confidence percentage badges near detected faces.",
    )
    p_cam.add_argument(
        "--show-fps",
        action="store_true",
        help="Display live smoothed FPS counter overlay.",
    )
    p_cam.add_argument(
        "--show-count",
        action="store_true",
        help="Display total detected face count banner.",
    )
    p_cam.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirror flip of the webcam feed.",
    )
    p_cam.add_argument(
        "--capture-dir",
        default="captures",
        help="Directory to save face captures triggered via 'C' key (default: captures).",
    )
    p_cam.add_argument(
        "--window-title",
        default="detekface - Live Face Detection",
        help="Custom window title for display preview.",
    )
    p_cam.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Exit automatically after processing this many frames (for testing).",
    )
    p_cam.set_defaults(func=_handle_camera)

    # -------------------------------------------------------------------------
    # Subcommand: image
    # -------------------------------------------------------------------------
    p_img = subparsers.add_parser(
        "image",
        help="Detect faces in a single still image.",
        description="Inspect faces in a photo, print coordinates, and optionally export annotated images or crops.",
    )
    p_img.add_argument(
        "input",
        help="Path to the input image file.",
    )
    p_img.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to save the annotated output image.",
    )
    p_img.add_argument(
        "--crop-face",
        default=None,
        help="Path to save the cropped largest detected face.",
    )
    p_img.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output or crop files if they already exist.",
    )
    p_img.add_argument(
        "--backend",
        choices=["yunet", "mediapipe"],
        default="yunet",
        help="Detection backend (default: yunet).",
    )
    p_img.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (default: 0.7).",
    )
    p_img.add_argument(
        "--shape",
        "-s",
        choices=sorted(SUPPORTED_SHAPES),
        default="rectangle",
        help="Overlay shape style for saved images (default: rectangle).",
    )
    p_img.add_argument(
        "--color",
        default="#00FF88",
        help="Overlay color as hex or named color (default: #00FF88).",
    )
    p_img.add_argument(
        "--thickness",
        "-t",
        type=int,
        default=3,
        help="Overlay stroke thickness in pixels (default: 3).",
    )
    p_img.add_argument(
        "--show-confidence",
        action="store_true",
        help="Render confidence badges on annotated output.",
    )
    p_img.add_argument(
        "--show-count",
        action="store_true",
        help="Render face count banner on annotated output.",
    )
    p_img.add_argument(
        "--show",
        action="store_true",
        help="Display preview window of annotated image.",
    )
    p_img.set_defaults(func=_handle_image)

    # -------------------------------------------------------------------------
    # Subcommand: batch
    # -------------------------------------------------------------------------
    p_batch = subparsers.add_parser(
        "batch",
        help="Batch detect faces in a directory or file list.",
        description="Process multiple images sequentially with error reporting and summary statistics.",
    )
    p_batch.add_argument(
        "input",
        help="Directory or path containing images to process.",
    )
    p_batch.add_argument(
        "--output",
        "-o",
        default=None,
        help="Directory to save annotated images.",
    )
    p_batch.add_argument(
        "--crop-dir",
        default=None,
        help="Directory to save face crops.",
    )
    p_batch.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output directories.",
    )
    p_batch.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively discover images in subdirectories.",
    )
    p_batch.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately and raise if any image fails to load or detect.",
    )
    p_batch.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit with non-zero exit code if one or more batch items encountered errors.",
    )
    p_batch.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-image progress log.",
    )
    p_batch.add_argument(
        "--backend",
        choices=["yunet", "mediapipe"],
        default="yunet",
        help="Detection backend (default: yunet).",
    )
    p_batch.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (default: 0.7).",
    )
    p_batch.add_argument(
        "--shape",
        "-s",
        choices=sorted(SUPPORTED_SHAPES),
        default="rectangle",
        help="Overlay shape style for saved images (default: rectangle).",
    )
    p_batch.add_argument(
        "--color",
        default="#00FF88",
        help="Overlay color as hex or named color (default: #00FF88).",
    )
    p_batch.add_argument(
        "--thickness",
        "-t",
        type=int,
        default=3,
        help="Overlay stroke thickness in pixels (default: 3).",
    )
    p_batch.set_defaults(func=_handle_batch)

    # -------------------------------------------------------------------------
    # Subcommand: video
    # -------------------------------------------------------------------------
    p_vid = subparsers.add_parser(
        "video",
        help="Process prerecorded video files.",
        description="Process a video file frame-by-frame with face detection and optional video export.",
    )
    p_vid.add_argument(
        "input",
        help="Path to input video file (.mp4, .avi, etc.).",
    )
    p_vid.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to save exported annotated video.",
    )
    p_vid.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output video file if it already exists.",
    )
    p_vid.add_argument(
        "--no-preview",
        action="store_true",
        help="Run without displaying a live playback window.",
    )
    p_vid.add_argument(
        "--backend",
        choices=["yunet", "mediapipe"],
        default="yunet",
        help="Detection backend (default: yunet).",
    )
    p_vid.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (default: 0.7).",
    )
    p_vid.add_argument(
        "--shape",
        "-s",
        choices=sorted(SUPPORTED_SHAPES),
        default="rectangle",
        help="Overlay shape style (default: rectangle).",
    )
    p_vid.add_argument(
        "--color",
        default="#00FF88",
        help="Overlay color as hex or named color (default: #00FF88).",
    )
    p_vid.add_argument(
        "--thickness",
        "-t",
        type=int,
        default=3,
        help="Overlay stroke thickness in pixels (default: 3).",
    )
    p_vid.add_argument(
        "--show-confidence",
        action="store_true",
        help="Display confidence percentage badges on video.",
    )
    p_vid.add_argument(
        "--show-fps",
        action="store_true",
        help="Display smoothed FPS counter overlay on video.",
    )
    p_vid.add_argument(
        "--show-count",
        action="store_true",
        help="Display detected face count banner on video.",
    )
    p_vid.set_defaults(func=_handle_video)

    return parser


def _handle_camera(args: argparse.Namespace) -> int:
    try:
        detekface.camera(
            camera_index=args.device,
            mirror=not args.no_mirror,
            confidence_threshold=args.confidence,
            shape=args.shape,
            color=args.color,
            thickness=args.thickness,
            show_confidence=args.show_confidence,
            show_fps=args.show_fps,
            show_count=args.show_count,
            capture_dir=args.capture_dir,
            window_title=args.window_title,
            max_frames=args.max_frames,
            backend=args.backend,
        )
        return 0
    except DetekFaceError as err:
        sys.stderr.write(f"Error: {err.message}\n")
        if getattr(err, "hint", None):
            sys.stderr.write(f"Hint: {err.hint}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Unexpected error: {exc}\n")
        return 1


def _handle_image(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Error: Input image does not exist: '{input_path}'.\n")
        sys.stderr.write("Hint: Verify the file path and spelling.\n")
        return 1

    if args.output:
        out_path = Path(args.output)
        if out_path.exists() and not args.overwrite:
            sys.stderr.write(f"Error: Output file '{out_path}' already exists.\n")
            sys.stderr.write("Hint: Pass --overwrite to overwrite existing files.\n")
            return 1

    if args.crop_face:
        crop_path = Path(args.crop_face)
        if crop_path.exists() and not args.overwrite:
            sys.stderr.write(f"Error: Crop output file '{crop_path}' already exists.\n")
            sys.stderr.write("Hint: Pass --overwrite to overwrite existing files.\n")
            return 1

    try:
        result = detekface.detect(
            image=input_path,
            confidence_threshold=args.confidence,
            backend=args.backend,
        )

        print(f"Detected {result.count} face(s) in '{input_path}' (backend: {args.backend}).")
        for idx, face in enumerate(result.faces, start=1):
            box = face.bbox
            print(
                f"  Face #{idx}: Box(x={box.x}, y={box.y}, w={box.width}, h={box.height}), Conf={face.confidence:.1%}"
            )

        if args.output:
            saved_path = result.save(
                path=args.output,
                shape=args.shape,
                color=args.color,
                thickness=args.thickness,
                show_confidence=args.show_confidence,
            )
            print(f"Saved annotated image to: '{saved_path}'")

        if args.crop_face:
            if result.has_faces:
                crop = result.capture(save=False)
                crop.save(args.crop_face)
                print(f"Saved face crop to: '{args.crop_face}'")
            else:
                sys.stderr.write("Warning: No faces detected; skipping face crop output.\n")

        if args.show:
            result.show(
                title=f"DetekFace - {input_path.name}",
                shape=args.shape,
                color=args.color,
                thickness=args.thickness,
                show_confidence=args.show_confidence,
            )

        return 0
    except DetekFaceError as err:
        sys.stderr.write(f"Error: {err.message}\n")
        if getattr(err, "hint", None):
            sys.stderr.write(f"Hint: {err.hint}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Unexpected error: {exc}\n")
        return 1


def _handle_batch(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Error: Input path does not exist: '{input_path}'.\n")
        sys.stderr.write("Hint: Verify the directory or file path.\n")
        return 1

    out_dir = Path(args.output) if args.output else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    crop_dir = Path(args.crop_dir) if args.crop_dir else None
    if crop_dir:
        crop_dir.mkdir(parents=True, exist_ok=True)

    def progress_cb(item, current, total):
        if not args.quiet:
            status = "OK" if item.is_success else "FAIL"
            faces = f"{item.count} face(s)" if item.is_success else str(item.error)
            print(f"[{current}/{total}] {status} | {item.source} -> {faces}")

    try:
        results = detekface.detect_many(
            images=input_path,
            confidence_threshold=args.confidence,
            backend=args.backend,
            fail_fast=args.fail_fast,
            recursive=args.recursive,
            progress_callback=progress_cb if not args.quiet else None,
            keep_images=bool(args.output or args.crop_dir),
        )

        # Output annotations if output directory requested
        if out_dir:
            for item in results.successes:
                src_p = Path(item.source if isinstance(item.source, (str, Path)) else "image.jpg")
                dest = out_dir / f"annotated_{src_p.name}"
                if dest.exists() and not args.overwrite:
                    continue
                assert item.result is not None
                item.result.save(
                    path=dest,
                    shape=args.shape,
                    color=args.color,
                    thickness=args.thickness,
                )

        # Output crops if crop directory requested
        if crop_dir:
            for item in results.successes:
                if item.has_faces:
                    src_p = Path(item.source if isinstance(item.source, (str, Path)) else "image.jpg")
                    crop_dest = crop_dir / f"crop_{src_p.stem}.jpg"
                    if crop_dest.exists() and not args.overwrite:
                        continue
                    assert item.result is not None
                    item.result.capture(save_dir=crop_dir, filename=crop_dest.name, save=True)

        print("\nBatch Summary:")
        print(f"  Total Processed : {len(results)}")
        print(f"  Successful      : {len(results.successes)}")
        print(f"  Failed          : {len(results.failures)}")
        print(f"  Total Faces     : {results.total_faces}")

        if results.has_failures and args.fail_on_errors:
            return 1
        return 0
    except DetekFaceError as err:
        sys.stderr.write(f"Error: {err.message}\n")
        if getattr(err, "hint", None):
            sys.stderr.write(f"Hint: {err.hint}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Unexpected error: {exc}\n")
        return 1


def _handle_video(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Error: Input video does not exist: '{input_path}'.\n")
        sys.stderr.write("Hint: Check video path and extension.\n")
        return 1

    if args.output:
        out_p = Path(args.output)
        if out_p.exists() and not args.overwrite:
            sys.stderr.write(f"Error: Output file '{out_p}' already exists.\n")
            sys.stderr.write("Hint: Pass --overwrite to overwrite existing files.\n")
            return 1

    try:
        frames = detekface.video(
            source=input_path,
            output=args.output,
            show=not args.no_preview,
            shape=args.shape,
            color=args.color,
            thickness=args.thickness,
            show_confidence=args.show_confidence,
            show_fps=args.show_fps,
            show_count=args.show_count,
            confidence_threshold=args.confidence,
            backend=args.backend,
        )
        print(f"Successfully processed {frames} frames from '{input_path}'.")
        if args.output:
            print(f"Exported annotated video to '{args.output}'.")
        return 0
    except DetekFaceError as err:
        sys.stderr.write(f"Error: {err.message}\n")
        if getattr(err, "hint", None):
            sys.stderr.write(f"Hint: {err.hint}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Unexpected error: {exc}\n")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point returning shell exit code."""
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    if len(argv) == 0:
        parser.print_help()
        return 0

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else (0 if code is None else 1)

    if hasattr(args, "func"):
        res = args.func(args)
        return int(res) if isinstance(res, int) else 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
