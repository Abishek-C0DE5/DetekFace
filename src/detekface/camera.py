"""Live webcam capture, customizable face overlay rendering, and camera session management."""

from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple, Union
import cv2
import numpy as np

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
    CameraError,
    CameraNotFoundError,
    CameraReadError,
    CaptureError,
    CaptureSaveError,
    ConfigurationError,
    NoFaceDetectedError,
)
from detekface.types import DetectionResult, FaceCapture


class CameraSession:
    """Manages an active camera device session with explicit lifecycle control.

    Provides frame-by-frame face detection, frame iteration, programmatic face capture,
    and optional GUI preview rendering.

    Frame Ownership & Memory Semantics:
        - Each call to `read()` fetches the latest frame from the hardware device.
        - If `mirror=True`, the frame is horizontally flipped via cv2.flip.
        - A pristine copy (`clean_frame = frame.copy()`) is stored as `latest_frame`
          and attached to `DetectionResult.image`.
        - The consumer owns `DetectionResult.image`. It is guaranteed to be clean,
          unannotated, and unpolluted by any overlay drawing or annotations.
        - Overlays are rendered on separate copies when `render()` or `show()` is called.

    Lifecycle & Cleanup:
        - Can be used as a context manager (`with CameraSession(...) as session:`).
        - Can be opened and closed manually (`session.open()`, `session.close()`).
        - Closing is idempotent and guarantees `cap.release()` is called.
        - Windows created via `show()` are automatically destroyed on session exit.
    """

    def __init__(
        self,
        camera_index: int = 0,
        mirror: bool = True,
        confidence_threshold: float = 0.7,
        detector: Optional[FaceDetector] = None,
        backend: str = "yunet",
        window_title: Optional[str] = None,
        auto_open: bool = True,
        resolution: Optional[Tuple[int, int]] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.camera_index = camera_index
        self.mirror = mirror
        self.confidence_threshold = confidence_threshold
        self.backend = backend
        self.window_title = window_title or "detekface - Camera Session"
        self._window_opened = False

        if resolution is not None:
            if (
                not isinstance(resolution, (tuple, list))
                or len(resolution) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 32 for v in resolution)
            ):
                raise ConfigurationError(
                    message=f"Invalid camera resolution: {resolution}. Expected (width, height) >= 32.",
                    hint="Set resolution to a standard size like (1280, 720) or (640, 480), or None.",
                )
            self.resolution: Optional[Tuple[int, int]] = (int(resolution[0]), int(resolution[1]))
        else:
            self.resolution = None

        self.input_size = input_size
        if detector is None:
            if input_size is not None:
                self.detector = FaceDetector(
                    backend=backend,
                    confidence_threshold=confidence_threshold,
                    input_size=input_size,
                )
            else:
                self.detector = FaceDetector(
                    backend=backend,
                    confidence_threshold=confidence_threshold,
                )
        else:
            self.detector = detector

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_result: Optional[DetectionResult] = None
        self._fps_counter = FPSCounter(window_size=30)
        self._frames_read: int = 0

        if auto_open:
            self.open()

    @property
    def is_open(self) -> bool:
        """Return True if the camera device is currently open and active."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def is_active(self) -> bool:
        """Alias for is_open."""
        return self.is_open

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        """Return the unannotated pristine BGR frame from the most recent read."""
        return self._latest_frame

    @property
    def latest_result(self) -> Optional[DetectionResult]:
        """Return the DetectionResult from the most recent read."""
        return self._latest_result

    @property
    def fps(self) -> float:
        """Return current smoothed real-time processing FPS throughput."""
        return self._fps_counter.fps

    @property
    def frames_read(self) -> int:
        """Return the total number of frames successfully read in this session."""
        return self._frames_read

    def open(self) -> "CameraSession":
        """Open the camera capture hardware device.

        Raises:
            CameraNotFoundError: If the camera cannot be opened or is not detected.
        """
        if self._cap is not None and self._cap.isOpened():
            return self

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap.release()
            self._cap = None
            raise CameraNotFoundError(
                message=f"Could not open camera device with index {self.camera_index}.",
                hint=(
                    "Verify your webcam is connected, that camera access permissions are enabled "
                    "in your operating system settings, and that no other application is currently using it."
                ),
            )

        if self.resolution is not None:
            w, h = self.resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        self._cap = cap
        return self

    def close(self) -> None:
        """Release camera hardware resources and destroy any active session windows.

        This method is idempotent and safe to call multiple times.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if self._window_opened:
            try:
                cv2.destroyWindow(self.window_title)
            except cv2.error:
                pass
            self._window_opened = False

    def __enter__(self) -> "CameraSession":
        if not self.is_open:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def read_raw(self) -> np.ndarray:
        """Read the next video frame from the camera without running face detection.

        Returns:
            np.ndarray: Pristine BGR uint8 NumPy array (mirrored if mirror=True).

        Raises:
            CameraError: If the camera session is not open.
            CameraReadError: If reading a frame from the hardware fails.
        """
        if not self.is_open or self._cap is None:
            raise CameraError(
                message="Cannot read from a closed CameraSession.",
                hint="Call session.open() or use the session within a 'with detekface.open_camera():' context manager.",
            )

        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise CameraReadError(
                message=f"Failed to read video frame from camera index {self.camera_index}.",
                hint="The camera may have been disconnected or encountered a hardware/driver error.",
            )

        self._fps_counter.tick()
        if self.mirror:
            clean_frame = cv2.flip(frame, 1)
        else:
            clean_frame = frame.copy()

        self._latest_frame = clean_frame
        self._frames_read += 1
        return clean_frame

    def read(self) -> DetectionResult:
        """Read the next video frame from the camera and detect human faces.

        Returns:
            DetectionResult: Structured detection result containing detected faces and the unannotated image.

        Raises:
            CameraError: If the camera session is not open.
            CameraReadError: If reading a frame from the hardware fails.
        """
        clean_frame = self.read_raw()
        result = self.detector.detect(clean_frame)
        self._latest_result = result
        return result

    def capture(
        self,
        face_index: Optional[int] = None,
        save_dir: Optional[Union[str, Path]] = "captures",
        filename: Optional[str] = None,
        save: bool = True,
    ) -> FaceCapture:
        """Programmatically capture a detected face crop from the current session frame.

        Args:
            face_index: Optional index of the face to crop. Defaults to largest face by area.
            save_dir: Target directory for saving cropped file.
            filename: Custom filename. Defaults to timestamped name.
            save: Whether to write image to disk (default True).

        Returns:
            FaceCapture: Cropped face dataclass with unpolluted image array and metadata.

        Raises:
            CaptureError: If no frames have been read yet in this session.
            NoFaceDetectedError: If no faces were detected in the latest frame.
        """
        if self._latest_result is None:
            raise CaptureError(
                message="No frames have been read yet in this camera session.",
                hint="Call session.read() at least once before attempting to capture a face.",
            )
        return capture(
            self._latest_result,
            face_index=face_index,
            save_dir=save_dir,
            filename=filename,
            save=save,
        )

    def __iter__(self) -> "CameraSession":
        return self

    def __next__(self) -> DetectionResult:
        if not self.is_open:
            raise StopIteration
        try:
            return self.read()
        except (CameraReadError, CameraError):
            raise StopIteration

    def stream(self, max_frames: Optional[int] = None) -> Iterator[DetectionResult]:
        """Generator yielding DetectionResults on each frame.

        Args:
            max_frames: Optional maximum number of frames to yield before stopping.

        Yields:
            DetectionResult: Result for each consecutive camera frame.
        """
        count = 0
        while self.is_open:
            if max_frames is not None and count >= max_frames:
                break
            try:
                yield self.read()
                count += 1
            except (CameraReadError, CameraError):
                break

    def process(
        self,
        callback: Callable[[DetectionResult], Optional[bool]],
        max_frames: Optional[int] = None,
    ) -> int:
        """Process incoming camera frames sequentially using a callback function.

        Args:
            callback: Function accepting a DetectionResult. If it returns False, processing stops.
            max_frames: Optional maximum number of frames to process.

        Returns:
            int: Total number of frames successfully processed.
        """
        processed = 0
        for result in self.stream(max_frames=max_frames):
            should_continue = callback(result)
            processed += 1
            if should_continue is False:
                break
        return processed

    def render(
        self,
        shape: str = "rectangle",
        color: Union[str, Tuple[int, int, int]] = "#00FF88",
        thickness: int = 3,
        show_confidence: bool = False,
        show_fps: bool = False,
        show_count: bool = False,
    ) -> np.ndarray:
        """Render overlays on a copy of the latest frame.

        Returns:
            np.ndarray: Annotated BGR frame array.
        """
        if self._latest_frame is None or self._latest_result is None:
            raise CameraError(
                message="No frame available to render.",
                hint="Call session.read() before calling session.render().",
            )
        valid_shape, valid_thickness = validate_render_parameters(shape, thickness)
        bgr_color = parse_color_to_bgr(color)

        display_frame = self._latest_frame.copy()
        draw_faces(
            display_frame,
            self._latest_result.faces,
            shape=valid_shape,
            color=bgr_color,
            thickness=valid_thickness,
            show_confidence=show_confidence,
            copy=False,
        )
        if show_fps or show_count:
            draw_metrics_overlay(
                display_frame,
                fps=self.fps if show_fps else None,
                face_count=self._latest_result.count if show_count else None,
                color_bgr=bgr_color,
            )
        return display_frame

    def show(
        self,
        shape: str = "rectangle",
        color: Union[str, Tuple[int, int, int]] = "#00FF88",
        thickness: int = 3,
        show_confidence: bool = False,
        show_fps: bool = False,
        show_count: bool = False,
        window_title: Optional[str] = None,
        wait_ms: int = 1,
    ) -> int:
        """Render and display the latest frame in an OpenCV window.

        Args:
            shape: Overlay shape style.
            color: Color definition.
            thickness: Line stroke thickness.
            show_confidence: Display confidence tags.
            show_fps: Display smoothed FPS overlay.
            show_count: Display detected face count.
            window_title: Window title. Defaults to session window title.
            wait_ms: Delay in milliseconds for cv2.waitKey.

        Returns:
            int: Key code returned by cv2.waitKey.
        """
        title = window_title or self.window_title
        annotated = self.render(
            shape=shape,
            color=color,
            thickness=thickness,
            show_confidence=show_confidence,
            show_fps=show_fps,
            show_count=show_count,
        )
        cv2.imshow(title, annotated)
        self._window_opened = True
        return cv2.waitKey(wait_ms) & 0xFF


def open_camera(
    camera_index: int = 0,
    mirror: bool = True,
    confidence_threshold: float = 0.7,
    detector: Optional[FaceDetector] = None,
    backend: str = "yunet",
    window_title: Optional[str] = None,
    auto_open: bool = True,
    resolution: Optional[Tuple[int, int]] = None,
    input_size: Optional[Tuple[int, int]] = None,
) -> CameraSession:
    """Open and return a managed CameraSession context manager.

    Example:
        import detekface

        with detekface.open_camera(camera_index=0) as session:
            for result in session:
                print(f"Detected {result.count} faces")
                if result.has_faces:
                    face = session.capture()
                    print(f"Captured face to {face.path}")
                    break
    """
    return CameraSession(
        camera_index=camera_index,
        mirror=mirror,
        confidence_threshold=confidence_threshold,
        detector=detector,
        backend=backend,
        window_title=window_title,
        auto_open=auto_open,
        resolution=resolution,
        input_size=input_size,
    )


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
    resolution: Optional[Tuple[int, int]] = None,
    input_size: Optional[Tuple[int, int]] = None,
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
        resolution: Optional capture hardware resolution (width, height) to negotiate with camera.
        input_size: Optional inference resolution (width, height) for high-FPS detection scaling.

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

    session = CameraSession(
        camera_index=camera_index,
        mirror=mirror,
        confidence_threshold=confidence_threshold,
        detector=detector,
        backend=backend,
        window_title=window_title,
        auto_open=False,
        resolution=resolution,
        input_size=input_size,
    )

    feedback_text: str = ""
    feedback_frames: int = 0
    feedback_color = (0, 255, 0)
    frames_processed = 0

    try:
        session.open()

        while True:
            result = session.read()
            display_frame = session.latest_frame.copy()

            # Draw customizable overlays on the display frame
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
                    fps=session.fps if show_fps else None,
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
        session.close()
        try:
            cv2.destroyWindow(window_title)
        except cv2.error:
            pass
