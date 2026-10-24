"""Live webcam capture, customizable face overlay rendering, and capture loop."""

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
    CameraNotFoundError,
    CameraReadError,
    CaptureError,
    CaptureSaveError,
    ConfigurationError,
    NoFaceDetectedError,
)
from detekface.types import FaceCapture


def camera(
    camera_index: int = 0,
    mirror: bool = True,
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
    window_title: str = "detekface - Live Face Detection",
    max_frames: Optional[int] = None,
    backend: str = "yunet",
) -> None:
    """Open a live webcam feed, detect faces in real time, and display customizable overlays.

    Controls:
        - Press 'c' or 'C' to capture and save the largest detected face.
        - Press 'q', 'Q', or 'Escape' to close the preview window.

    Args:
        camera_index: Hardware device index for the camera (default is 0).
        mirror: Whether to horizontally flip the live feed for a mirror-like experience (default True).
        confidence_threshold: Minimum detection confidence score (0.0 to 1.0). Default is 0.7.
        detector: Optional pre-configured FaceDetector instance. Reuses existing engine if provided.
        shape: Face overlay shape style. One of 'rectangle', 'corners', 'circle', 'rounded', 'hud'.
               Default is 'rectangle'.
        color: RGB hex string (e.g. '#00FF88') or documented color name. Default is '#00FF88'.
        thickness: Line stroke thickness in pixels (default 3).
        show_confidence: Whether to display confidence percentage near each detected face (default False).
        show_fps: Whether to display measured, smoothed real-time FPS throughput (default False).
        show_count: Whether to display the total number of faces in the current frame (default False).
        capture_dir: Directory where face captures triggered via 'C' are saved (default 'captures').
        on_capture: Optional callback function invoked with the FaceCapture object when 'C' is pressed.
        window_title: Window title for the OpenCV preview window.
        max_frames: Optional maximum number of frames to process before closing (useful for automated testing).
        backend: Detection backend ('yunet' or 'mediapipe'). Default is 'yunet'.

    Raises:
        ConfigurationError: If shape, color, thickness, or display flags are invalid.
        CameraNotFoundError: If the camera cannot be opened or is not detected.
        CameraReadError: If capturing a video frame from the camera fails.
    """
    valid_shape, valid_thickness = validate_render_parameters(shape, thickness)
    bgr_color = parse_color_to_bgr(color)

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

    if detector is None:
        detector = FaceDetector(backend=backend, confidence_threshold=confidence_threshold)

    cap = cv2.VideoCapture(camera_index)
    frames_processed = 0

    # Real-time smoothed FPS throughput tracker
    fps_counter = FPSCounter(window_size=30)

    # On-screen feedback state
    feedback_text: str = ""
    feedback_frames: int = 0
    feedback_color = (0, 255, 0)

    try:
        if not cap.isOpened():
            raise CameraNotFoundError(
                message=f"Could not open camera device with index {camera_index}.",
                hint=(
                    "Verify your webcam is connected, that camera access permissions are enabled "
                    "in your operating system settings, and that no other application is currently using it."
                ),
            )

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                raise CameraReadError(
                    message=f"Failed to read video frame from camera index {camera_index}.",
                    hint="The camera may have been disconnected or encountered a hardware/driver error.",
                )

            current_fps = fps_counter.tick()

            # Horizontally mirror the frame if requested
            if mirror:
                frame = cv2.flip(frame, 1)

            # Keep a pristine copy of the frame for detection and clean crops
            clean_frame = frame.copy()

            # Perform face detection using existing engine
            result = detector.detect(clean_frame)

            # Draw customizable overlays on the display frame
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

            # Draw status metrics (FPS and Face count)
            if show_fps or show_count:
                draw_metrics_overlay(
                    display_frame,
                    fps=current_fps if show_fps else None,
                    face_count=result.count if show_count else None,
                    color_bgr=bgr_color,
                )

            # Render capture feedback banner if active
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

            frames_processed += 1
            if max_frames is not None and frames_processed >= max_frames:
                break

            # Check keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Close preview on 'q', 'Q', or Escape (27)
            if key in (ord("q"), ord("Q"), 27):
                break

            # Capture largest face on 'c' or 'C'
            if key in (ord("c"), ord("C")):
                try:
                    captured = capture(result, save_dir=capture_dir)
                    saved_name = captured.path.name if captured.path else "Face"
                    feedback_text = f"Captured: {saved_name}"
                    feedback_frames = 30
                    feedback_color = (0, 255, 0)
                    print(f"[detekface] Captured face saved to: {captured.path}")

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
                    print("[detekface] Capture skipped: No face detected.")
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

    finally:
        # Guarantee camera release and window destruction even on exception
        cap.release()
        try:
            cv2.destroyWindow(window_title)
        except cv2.error:
            pass
