import math
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import cv2

from detekface.capture import capture
from detekface.detector import FaceDetector
from detekface.drawing import (
    FPSCounter,
    draw_faces,
    draw_metrics_overlay,
    parse_color_to_bgr,
    validate_render_parameters,
)
from detekface.exceptions import (
    CaptureError,
    CaptureSaveError,
    ConfigurationError,
    InvalidVideoError,
    NoFaceDetectedError,
    VideoNotFoundError,
    VideoWriteError,
)
from detekface.types import FaceCapture

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def video(
    path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    confidence_threshold: float = 0.7,
    detector: Optional[FaceDetector] = None,
    shape: str = "rectangle",
    color: Union[str, Tuple[int, int, int]] = "#00FF88",
    thickness: int = 3,
    show_confidence: bool = False,
    show_fps: bool = False,
    show_count: bool = False,
    capture_dir: Optional[Union[str, Path]] = "captures",
    on_capture: Optional[Callable[[FaceCapture], None]] = None,
    window_title: str = "detekface - Video Face Detection",
    max_frames: Optional[int] = None,
    codec: Optional[str] = None,
    source: Optional[Union[str, Path]] = None,
    output: Optional[Union[str, Path]] = None,
    backend: str = "yunet",
    frame_skip: int = 0,
    input_size: Optional[Tuple[int, int]] = None,
) -> Optional[Path]:
    """Process a prerecorded video file frame by frame and detect faces.

    Renders customizable face overlays, optionally displays real-time preview,
    supports interactive face cropping via the 'c' key, and saves annotated output
    video preserving the source dimensions and frame rate.

    Controls (when show=True):
        - Press 'c' or 'C' to capture and save the largest detected face from the current frame.
        - Press 'q', 'Q', or 'Escape' to stop playback and exit early.

    Args:
        path: Path to the input video file (str or Path).
        output_path: Optional destination path to save the annotated video (e.g. 'annotated.mp4').
        show: Whether to display playback in an OpenCV GUI window (default True). Set False for headless.
        confidence_threshold: Minimum detection confidence score (0.0 to 1.0). Default is 0.7.
        detector: Optional pre-configured FaceDetector instance. Reuses existing engine if provided.
        shape: Face overlay shape style ('rectangle', 'corners', 'circle', 'rounded', 'hud').
        color: RGB hex string (e.g. '#00FF88') or documented color name. Default is '#00FF88'.
        thickness: Line stroke thickness in pixels (default 3).
        show_confidence: Whether to display confidence percentages near each face (default False).
        show_fps: Whether to display measured processing throughput FPS (default False).
        show_count: Whether to display current face count in a top banner (default False).
        capture_dir: Directory where face captures triggered via 'c' are saved (default 'captures').
        on_capture: Optional callback invoked with the FaceCapture object when 'c' is pressed.
        window_title: Window title for the OpenCV preview window.
        max_frames: Optional maximum number of frames to process before stopping.
        codec: Optional 4-character FourCC codec string (e.g. 'mp4v', 'MJPG'). If None,
               selected automatically based on output file extension.
        source: Backward-compatible alias for path.
        output: Backward-compatible alias for output_path.
        backend: Detection backend ('yunet' or 'mediapipe'). Default is 'yunet'.
        frame_skip: Number of intermediate frames to skip detection on to boost throughput (default 0).
        input_size: Optional inference resolution (width, height) for high-FPS detection scaling.

    Returns:
        The resolved Path of the saved output video file if output_path was provided, else None.

    Raises:
        VideoNotFoundError: If the input video file does not exist.
        InvalidVideoError: If the input or output file format is invalid, empty, or unreadable.
        ConfigurationError: If styling or display parameters are invalid.
        VideoWriteError: If creating or writing to the output video file fails.
    """
    if path is None and source is not None:
        path = source
    if output_path is None and output is not None:
        output_path = output

    if path is None or not isinstance(path, (str, Path)) or not str(path).strip():
        raise InvalidVideoError(
            message=f"Invalid video path: {path!r}.",
            hint="Provide a valid filepath string or Path to an existing video file.",
        )

    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise VideoNotFoundError(
            message=f"Video file not found at: '{file_path}'.",
            hint="Check the path for typos or verify the file exists on disk.",
        )
    if not file_path.is_file():
        raise InvalidVideoError(
            message=f"Specified video path is a directory, not a file: '{file_path}'.",
            hint="Provide a path to a video file (e.g. .mp4, .avi).",
        )

    # Validate render and display parameters
    valid_shape, valid_thickness = validate_render_parameters(shape, thickness)
    bgr_color = parse_color_to_bgr(color)

    if not isinstance(show, bool):
        raise ConfigurationError(
            message=f"show must be a boolean, got {type(show).__name__}.",
            hint="Set show=True for GUI preview or show=False for headless processing.",
        )
    if not isinstance(show_confidence, bool):
        raise ConfigurationError(
            message=f"show_confidence must be a boolean, got {type(show_confidence).__name__}.",
            hint="Set show_confidence=True or show_confidence=False.",
        )
    if not isinstance(show_fps, bool):
        raise ConfigurationError(
            message=f"show_fps must be a boolean, got {type(show_fps).__name__}.",
            hint="Set show_fps=True or show_fps=False.",
        )
    if not isinstance(show_count, bool):
        raise ConfigurationError(
            message=f"show_count must be a boolean, got {type(show_count).__name__}.",
            hint="Set show_count=True or show_count=False.",
        )

    if not isinstance(frame_skip, int) or isinstance(frame_skip, bool) or frame_skip < 0:
        raise ConfigurationError(
            message=f"frame_skip must be an integer >= 0, got {frame_skip!r}.",
            hint="Set frame_skip=0 for full processing, or frame_skip=1/2 to process every 2nd/3rd frame.",
        )

    # Validate output destination if requested
    target_output: Optional[Path] = None
    fourcc_code: str = "mp4v"

    if output_path is not None:
        if not isinstance(output_path, (str, Path)) or not str(output_path).strip():
            raise InvalidVideoError(
                message=f"Invalid output video path: {output_path!r}.",
                hint="Provide a valid destination filepath with a supported video extension.",
            )
        target_output = Path(output_path).expanduser().resolve()
        out_ext = target_output.suffix.lower()
        if not out_ext:
            raise InvalidVideoError(
                message=f"Output video path '{target_output}' has no file extension.",
                hint=f"Provide a supported video extension such as: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}.",
            )
        if out_ext not in SUPPORTED_VIDEO_EXTENSIONS:
            raise InvalidVideoError(
                message=f"Unsupported output video format '{out_ext}' in path: '{target_output}'.",
                hint=f"Supported video extensions are: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}.",
            )

        parent_dir = target_output.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise VideoWriteError(
                    message=f"Could not create parent directory for output video: '{parent_dir}'.",
                    hint=str(e),
                ) from e
        elif not parent_dir.is_dir():
            raise VideoWriteError(
                message=f"Parent path of output video is a file, not a directory: '{parent_dir}'.",
                hint="Specify an output video path inside a valid directory.",
            )

        if codec is not None:
            if not isinstance(codec, str) or len(codec) != 4:
                raise ConfigurationError(
                    message=f"Codec must be a 4-character FourCC string, got: {codec!r}.",
                    hint="Use codecs like 'mp4v', 'MJPG', or 'XVID'.",
                )
            fourcc_code = codec
        else:
            if out_ext == ".avi":
                fourcc_code = "MJPG"
            elif out_ext == ".webm":
                fourcc_code = "VP80"
            else:
                fourcc_code = "mp4v"

    if detector is None:
        if input_size is not None:
            detector = FaceDetector(
                backend=backend,
                confidence_threshold=confidence_threshold,
                input_size=input_size,
            )
        else:
            detector = FaceDetector(
                backend=backend,
                confidence_threshold=confidence_threshold,
            )

    cap = cv2.VideoCapture(str(file_path))
    writer: Optional[cv2.VideoWriter] = None

    fps_counter = FPSCounter(window_size=30)
    frames_processed = 0
    latest_result = None

    feedback_text: str = ""
    feedback_frames: int = 0
    feedback_color = (0, 255, 0)

    try:
        if not cap.isOpened():
            raise InvalidVideoError(
                message=f"Could not open or decode video file: '{file_path}'.",
                hint="Verify that the file is an uncorrupted video with a supported container and codec.",
            )

        # Source framerate for display pacing and output writing
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        try:
            video_fps = float(src_fps)
            if video_fps <= 0 or math.isnan(video_fps):
                video_fps = 25.0
        except (TypeError, ValueError):
            video_fps = 25.0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if frames_processed == 0:
                    raise InvalidVideoError(
                        message=f"Could not read any video frames from: '{file_path}'.",
                        hint="The video file may be empty or contain unsupported stream data.",
                    )
                # Normal end of video stream
                break

            current_fps = fps_counter.tick()

            # Initialize writer upon reading the first frame using precise dimensions
            if writer is None and target_output is not None:
                frame_h, frame_w = frame.shape[:2]
                try:
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
                    writer = cv2.VideoWriter(str(target_output), fourcc, video_fps, (frame_w, frame_h))
                except Exception as e:
                    raise VideoWriteError(
                        message=f"Failed to create VideoWriter for '{target_output}' with codec '{fourcc_code}'.",
                        hint=str(e),
                    ) from e

                if not writer.isOpened():
                    raise VideoWriteError(
                        message=f"OpenCV failed to open VideoWriter for '{target_output}' with codec '{fourcc_code}'.",
                        hint="Check output directory permissions and video codec compatibility.",
                    )

            # Keep pristine clean frame for detection and clean crops
            clean_frame = frame.copy()

            # Perform face detection (reusing detections on skipped frames)
            should_detect = (frame_skip == 0) or (frames_processed % (frame_skip + 1) == 0) or (latest_result is None)
            if should_detect:
                latest_result = detector.detect(clean_frame)
            result = latest_result

            # Draw overlays on display/output frame
            display_frame = frame
            draw_faces(
                display_frame,
                result.faces,
                shape=valid_shape,
                color=bgr_color,
                thickness=valid_thickness,
                show_confidence=show_confidence,
                copy=False,
            )

            # Render status metrics if requested
            if show_fps or show_count:
                draw_metrics_overlay(
                    display_frame,
                    fps=current_fps if show_fps else None,
                    face_count=result.count if show_count else None,
                    color_bgr=bgr_color,
                )

            # Write annotated frame to output file
            if writer is not None:
                writer.write(display_frame)

            # Display GUI preview if requested
            if show:
                if feedback_frames > 0:
                    cv2.putText(
                        display_frame,
                        feedback_text,
                        (20, 60 if (show_fps or show_count) else 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        feedback_color,
                        2,
                        cv2.LINE_AA,
                    )
                    feedback_frames -= 1

                cv2.imshow(window_title, display_frame)

                # Delay timed to match original source video framerate
                frame_delay = max(1, int(round(1000.0 / video_fps)))
                key = cv2.waitKey(frame_delay) & 0xFF

                # Early exit on 'q', 'Q', or Escape (27)
                if key in (ord("q"), ord("Q"), 27):
                    break

                # Capture largest face from pristine frame on 'c' or 'C'
                if key in (ord("c"), ord("C")):
                    try:
                        captured = capture(result, save_dir=capture_dir)
                        saved_name = captured.path.name if captured.path else "Face"
                        feedback_text = f"Captured: {saved_name}"
                        feedback_frames = 30
                        feedback_color = (0, 255, 0)
                        if on_capture is not None:
                            try:
                                on_capture(captured)
                            except Exception as cb_err:
                                feedback_text = "Callback failed!"
                                feedback_frames = 30
                                feedback_color = (0, 0, 255)
                                print(f"[detekface] Warning: on_capture callback raised an exception: {cb_err}")
                    except NoFaceDetectedError:
                        feedback_text = "No face detected!"
                        feedback_frames = 30
                        feedback_color = (0, 140, 255)
                    except CaptureSaveError as save_err:
                        feedback_text = "Save failed!"
                        feedback_frames = 30
                        feedback_color = (0, 0, 255)
                        print(f"[detekface] Error: Capture failed to save: {save_err}")
                    except CaptureError as cap_err:
                        feedback_text = "Capture error!"
                        feedback_frames = 30
                        feedback_color = (0, 0, 255)
                        print(f"[detekface] Error during capture: {cap_err}")

                # Check if window was closed via OS close button
                try:
                    if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    pass

            frames_processed += 1
            if max_frames is not None and frames_processed >= max_frames:
                break

    finally:
        # Guarantee resource release and window destruction
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    return target_output
