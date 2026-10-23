"""Live webcam capture and face detection display loop."""

from pathlib import Path
from typing import Callable, Optional, Union
import cv2

from detekface.capture import capture
from detekface.detector import FaceDetector
from detekface.exceptions import CameraNotFoundError, CameraReadError, NoFaceDetectedError
from detekface.types import FaceCapture


def camera(
    camera_index: int = 0,
    mirror: bool = True,
    confidence_threshold: float = 0.7,
    detector: Optional[FaceDetector] = None,
    capture_dir: Optional[Union[str, Path]] = "captures",
    on_capture: Optional[Callable[[FaceCapture], None]] = None,
    window_title: str = "detekface - Live Face Detection",
    max_frames: Optional[int] = None,
) -> None:
    """Open a live webcam feed, detect faces in real time, and display bounding boxes.

    Controls:
        - Press 'c' or 'C' to capture and save the largest detected face.
        - Press 'q', 'Q', or 'Escape' to close the preview window.

    Args:
        camera_index: Hardware device index for the camera (default is 0).
        mirror: Whether to horizontally flip the live feed for a mirror-like experience (default True).
        confidence_threshold: Minimum detection confidence score (0.0 to 1.0). Default is 0.7.
        detector: Optional pre-configured FaceDetector instance. Reuses existing engine if provided.
        capture_dir: Directory where face captures triggered via 'C' are saved (default 'captures').
        on_capture: Optional callback function invoked with the FaceCapture object when 'C' is pressed.
        window_title: Window title for the OpenCV preview window.
        max_frames: Optional maximum number of frames to process before closing (useful for automated testing).

    Raises:
        CameraNotFoundError: If the camera cannot be opened or is not detected.
        CameraReadError: If capturing a video frame from the camera fails.
    """
    if detector is None:
        detector = FaceDetector(confidence_threshold=confidence_threshold)

    cap = cv2.VideoCapture(camera_index)
    frames_processed = 0

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

            # Horizontally mirror the frame if requested
            if mirror:
                frame = cv2.flip(frame, 1)

            # Perform face detection using existing engine
            result = detector.detect(frame)

            # Draw simple green rectangular box around each detected face
            for face in result.faces:
                bbox = face.bbox
                cv2.rectangle(
                    frame,
                    (bbox.x1, bbox.y1),
                    (bbox.x2, bbox.y2),
                    (0, 255, 0),
                    2,
                )

            # Render feedback message if active
            if feedback_frames > 0:
                cv2.putText(
                    frame,
                    feedback_text,
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    feedback_color,
                    2,
                    cv2.LINE_AA,
                )
                feedback_frames -= 1

            cv2.imshow(window_title, frame)

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
                        on_capture(captured)

                except NoFaceDetectedError:
                    feedback_text = "No face detected!"
                    feedback_frames = 30
                    feedback_color = (0, 140, 255)
                    print("[detekface] Capture skipped: No face detected.")

    finally:
        # Guarantee camera release and window destruction even on exception
        cap.release()
        cv2.destroyAllWindows()
