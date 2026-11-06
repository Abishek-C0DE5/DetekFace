"""Live webcam capture, customizable face overlay rendering, and camera session management."""

from pathlib import Path
import time
from typing import Callable, Iterator, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from detekface.async_cam import AsyncCameraEngine, CameraMetrics
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
from detekface.types import DetectionResult, FaceCapture, FaceDetection
from detekface.face_shape import TemporalShapeTracker, draw_face_shape_overlay
from detekface.gestures import GestureConfig, GestureDetector, GestureEvent
from detekface.pose import TemporalPoseSmoother
from detekface.tracking import FaceTracker


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
        analyze_face_shape: bool = False,
        mesh: bool = False,
        track_faces: bool = False,
        estimate_pose: bool = False,
        target_fps: float = 60.0,
        async_mode: bool = False,
        stale_timeout_sec: float = 0.5,
        detect_gestures: bool = False,
        on_gesture: Optional[Callable[[GestureEvent], None]] = None,
        gestures: Optional[Sequence[str]] = None,
        gesture_config: Optional[GestureConfig] = None,
        show_gestures: bool = False,
        effect: Optional[str] = None,
        accessory: Optional[str] = None,
        accessory_opacity: float = 1.0,
        scale_adjust: float = 1.0,
        y_offset: float = 0.0,
        accessory_scale: Optional[float] = None,
        accessory_y_offset: Optional[float] = None,
        show_quality: bool = False,
        anonymize: Optional[str] = None,
        anonymize_strength: float = 0.5,
        anonymize_color: Union[str, Tuple[int, int, int]] = (0, 0, 0),
        anonymize_mask_shape: str = "box",
    ) -> None:
        self.camera_index = camera_index
        self.mirror = mirror
        self.confidence_threshold = confidence_threshold
        self.backend = backend
        self.window_title = window_title or "detekface - Camera Session"
        self._window_opened = False
        self.analyze_face_shape = bool(analyze_face_shape)
        self.mesh = bool(mesh)
        self.detect_gestures = bool(detect_gestures or on_gesture is not None)
        self.on_gesture = on_gesture
        self.gestures = gestures
        self.gesture_config = gesture_config
        self.show_gestures = bool(show_gestures)
        self.show_quality = bool(show_quality)
        self.anonymize = anonymize
        self.anonymize_strength = float(anonymize_strength)
        self.anonymize_color = anonymize_color
        self.anonymize_mask_shape = str(anonymize_mask_shape)

        eff = effect if effect is not None else accessory
        self.effect: Optional[str] = eff
        self.accessory_opacity: float = float(accessory_opacity)
        self.accessory_scale: float = float(accessory_scale if accessory_scale is not None else scale_adjust)
        self.accessory_y_offset: float = float(accessory_y_offset if accessory_y_offset is not None else y_offset)

        if self.effect is not None:
            from detekface.accessories import FaceAccessoryRenderer

            self._accessory_renderer = FaceAccessoryRenderer(
                accessory=self.effect,
                opacity=self.accessory_opacity,
                scale_adjust=self.accessory_scale,
                y_offset=self.accessory_y_offset,
                smooth=True,
            )
        else:
            self._accessory_renderer = None

        # Gestures and AR accessories benefit from tracking and pose estimation
        self.track_faces = bool(track_faces or self.detect_gestures or self.effect is not None)
        self.estimate_pose = bool(estimate_pose or self.detect_gestures or self.effect is not None)

        self._shape_tracker = TemporalShapeTracker() if self.analyze_face_shape else None
        self._tracker = FaceTracker() if self.track_faces else None
        self._pose_smoother = TemporalPoseSmoother() if self.estimate_pose else None
        self._gesture_detector = (
            GestureDetector(
                config=gesture_config,
                on_gesture=on_gesture,
                enabled_gestures=gestures,
            )
            if self.detect_gestures
            else None
        )

        if not isinstance(track_faces, bool):
            raise ConfigurationError(
                message=f"track_faces must be a boolean, got {type(track_faces).__name__}.",
                hint="Set track_faces=True or track_faces=False.",
            )
        if not isinstance(estimate_pose, bool):
            raise ConfigurationError(
                message=f"estimate_pose must be a boolean, got {type(estimate_pose).__name__}.",
                hint="Set estimate_pose=True or estimate_pose=False.",
            )

        if not isinstance(target_fps, (int, float)) or isinstance(target_fps, bool) or target_fps <= 0:
            raise ConfigurationError(
                message=f"target_fps must be a positive number, got {target_fps}.",
                hint="Provide a valid frame rate like 60.0 or 30.0.",
            )
        self.target_fps = float(target_fps)

        if not isinstance(async_mode, bool):
            raise ConfigurationError(
                message=f"async_mode must be a boolean, got {type(async_mode).__name__}.",
                hint="Set async_mode=True or async_mode=False.",
            )
        self.async_mode = bool(async_mode)

        if not isinstance(stale_timeout_sec, (int, float)) or isinstance(stale_timeout_sec, bool) or stale_timeout_sec <= 0:
            raise ConfigurationError(
                message=f"stale_timeout_sec must be a positive number, got {stale_timeout_sec}.",
                hint="Set stale_timeout_sec to a duration in seconds like 0.5.",
            )
        self.stale_timeout_sec = float(stale_timeout_sec)

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
            self.resolution: Optional[Tuple[int, int]] = (
                int(resolution[0]),
                int(resolution[1]),
            )
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
        self._async_engine: Optional[AsyncCameraEngine] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_result: Optional[DetectionResult] = None
        self._fps_counter = FPSCounter(window_size=30)
        self._frames_read: int = 0
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._actual_hardware_fps: float = 0.0
        self._last_inference_latency_ms: float = 0.0

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
        if self.async_mode and self._async_engine is not None:
            return self._async_engine.metrics.render_fps
        return self._fps_counter.fps

    @property
    def actual_width(self) -> int:
        """Negotiated camera hardware frame width."""
        return self._actual_width

    @property
    def actual_height(self) -> int:
        """Negotiated camera hardware frame height."""
        return self._actual_height

    @property
    def actual_hardware_fps(self) -> float:
        """Reported hardware capture FPS negotiated with device driver."""
        return self._actual_hardware_fps

    @property
    def metrics(self) -> CameraMetrics:
        """Real-time performance telemetry metrics and verified camera properties."""
        if self._async_engine is not None:
            return self._async_engine.metrics
        return CameraMetrics(
            capture_fps=self._fps_counter.fps,
            inference_fps=self._fps_counter.fps,
            render_fps=self._fps_counter.fps,
            inference_latency_ms=self._last_inference_latency_ms,
            actual_width=self._actual_width,
            actual_height=self._actual_height,
            hardware_fps=self._actual_hardware_fps,
        )

    @property
    def frames_read(self) -> int:
        """Return the total number of frames successfully read in this session."""
        return self._frames_read

    def open(self, cap: Optional[cv2.VideoCapture] = None) -> "CameraSession":
        """Open the camera capture hardware device.

        Args:
            cap: Optional pre-existing cv2.VideoCapture instance (for hardware testing or injection).

        Raises:
            CameraNotFoundError: If the camera cannot be opened or is not detected.
        """
        if self._cap is not None and self._cap.isOpened():
            return self

        if cap is not None:
            self._cap = cap
        else:
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
            self._cap = cap

        if self.resolution is not None:
            w, h = self.resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._actual_hardware_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        self._cap = cap

        if self.async_mode:
            self._async_engine = AsyncCameraEngine(
                camera_index=self.camera_index,
                mirror=self.mirror,
                detector=self.detector,
                target_fps=self.target_fps,
                resolution=self.resolution,
                analyze_face_shape=self.analyze_face_shape,
                mesh=self.mesh,
                track_faces=self.track_faces,
                estimate_pose=self.estimate_pose,
                detect_gestures=self.detect_gestures,
                on_gesture=self.on_gesture,
                gestures=self.gestures,
                gesture_config=self.gesture_config,
                stale_timeout_sec=self.stale_timeout_sec,
            )
            self._async_engine.start(cap=self._cap)

        return self

    def close(self) -> None:
        """Release camera hardware resources and destroy any active session windows.

        This method is idempotent and safe to call multiple times.
        """
        if self._async_engine is not None:
            self._async_engine.stop()
            self._async_engine = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if self._window_opened:
            try:
                cv2.destroyWindow(self.window_title)
            except cv2.error:
                pass
            self._window_opened = False

        if self._shape_tracker is not None:
            self._shape_tracker.reset()

        if self._tracker is not None:
            self._tracker.reset()

        if self._pose_smoother is not None:
            self._pose_smoother.reset()

        if self._gesture_detector is not None:
            self._gesture_detector.reset()

        self._latest_frame = None
        self._latest_result = None

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

        if self.async_mode and self._async_engine is not None:
            frame, _, _ = self._async_engine.get_latest_state(timeout=1.0)
            if frame is None:
                raise CameraReadError(
                    message=f"Failed to read video frame from camera index {self.camera_index}.",
                    hint="The camera may have been disconnected or encountered a hardware/driver error.",
                )
            self._latest_frame = frame
            self._frames_read += 1
            return frame

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
        if not self.is_open or self._cap is None:
            raise CameraError(
                message="Cannot read from a closed CameraSession.",
                hint="Call session.open() or use the session within a 'with detekface.open_camera():' context manager.",
            )

        if self.async_mode and self._async_engine is not None:
            frame, faces, _ = self._async_engine.get_latest_state(timeout=1.0)
            if frame is None:
                raise CameraReadError(
                    message=f"Failed to read video frame from camera index {self.camera_index}.",
                    hint="The camera may have been disconnected or encountered a hardware/driver error.",
                )
            self._latest_frame = frame
            self._frames_read += 1
            result = DetectionResult(faces=faces, image=frame)
            self._latest_result = result
            return result

        clean_frame = self.read_raw()
        t0 = time.perf_counter()
        result = self.detector.detect(
            clean_frame,
            analyze_face_shape=self.analyze_face_shape,
            estimate_pose=self.estimate_pose,
        )
        self._last_inference_latency_ms = (time.perf_counter() - t0) * 1000.0

        faces = result.faces
        if self.analyze_face_shape and self._shape_tracker is not None:
            faces = self._shape_tracker.update(faces)

        if self.estimate_pose and self._pose_smoother is not None:
            smoothed_pose_faces = []
            for face in faces:
                if face.head_pose is not None:
                    sp = self._pose_smoother.smooth(face.head_pose, track_id=face.track_id)
                    smoothed_pose_faces.append(
                        FaceDetection(
                            bbox=face.bbox,
                            confidence=face.confidence,
                            landmarks=face.landmarks,
                            face_shape=face.face_shape,
                            dense_landmarks=face.dense_landmarks,
                            track_id=face.track_id,
                            head_pose=sp,
                        )
                    )
                else:
                    smoothed_pose_faces.append(face)
            faces = smoothed_pose_faces

        if self.track_faces and self._tracker is not None:
            faces = self._tracker.update(faces)

        if self.detect_gestures and self._gesture_detector is not None:
            from dataclasses import replace

            active_events = self._gesture_detector.process(faces, clean_frame.shape[:2])
            if active_events:
                events_by_key = {}
                for ev in active_events:
                    k = ev.track_id if ev.track_id is not None else ev.face_index
                    events_by_key.setdefault(k, []).append(ev.name)

                faces_with_g = []
                for idx, face in enumerate(faces):
                    k = face.track_id if face.track_id is not None else idx
                    g_list = events_by_key.get(k)
                    if g_list:
                        faces_with_g.append(replace(face, gestures=g_list))
                    else:
                        faces_with_g.append(face)
                faces = faces_with_g

        result = DetectionResult(faces=faces, image=clean_frame)
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
        show_face_shape: bool = True,
        face_shape_effect: Optional[str] = "scan",
        show_shape_contour: bool = True,
        show_shape_measurements: bool = False,
        show_mesh: bool = False,
        mesh_style: str = "wireframe",
        mesh_color: Optional[Union[str, Tuple[int, int, int]]] = None,
        mesh_thickness: Optional[int] = None,
        mesh_regions: Optional[Sequence[str]] = None,
        mesh_draw_points: Optional[bool] = None,
        show_contour: Optional[bool] = None,
        show_scan: Optional[bool] = None,
        live_fx: bool = False,
        live_fx_style: str = "scan",
        show_gestures: bool = False,
        effect: Optional[str] = None,
        accessory: Optional[str] = None,
        accessory_opacity: Optional[float] = None,
        accessory_scale: Optional[float] = None,
        accessory_y_offset: Optional[float] = None,
        show_quality: bool = False,
        anonymize: Optional[str] = None,
        anonymize_strength: Optional[float] = None,
        anonymize_color: Optional[Union[str, Tuple[int, int, int]]] = None,
        anonymize_mask_shape: Optional[str] = None,
    ) -> np.ndarray:
        """Render overlays on a copy of the latest frame.

        Rendering Order:
            1. Facial mesh wireframe (base anatomical layer).
            2. Face shape contour silhouette (if enabled).
            3. Live FX scanner animations / laser beam sweep / reticle brackets.
            4. Primary bounding box / overlay shapes (rectangle, corners, circle, hud, rounded).
            5. Face shape classification badges & measurements.
            6. Top-level status HUD (FPS counter, face count, confidence tags).

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

        # Resolve aliases for show_contour and show_scan
        effective_show_contour = show_shape_contour if show_contour is None else bool(show_contour)
        effective_live_fx = live_fx or (show_scan is True)

        display_frame = self._latest_frame.copy()

        # Layer 0: Face Privacy Anonymization
        eff_anon = anonymize if anonymize is not None else self.anonymize
        if eff_anon is not None:
            from detekface.anonymize import anonymize_frame

            st = anonymize_strength if anonymize_strength is not None else self.anonymize_strength
            co = anonymize_color if anonymize_color is not None else self.anonymize_color
            sh = anonymize_mask_shape if anonymize_mask_shape is not None else self.anonymize_mask_shape
            anonymize_frame(
                display_frame,
                self._latest_result.faces,
                method=eff_anon,
                strength=st,
                color=co,
                mask_shape=sh,
                copy=False,
            )

        # Layer 1: Dense facial mesh wireframe
        if show_mesh:
            from detekface.mesh import draw_face_meshes

            m_col = parse_color_to_bgr(mesh_color) if mesh_color is not None else bgr_color
            draw_face_meshes(
                display_frame,
                self._latest_result.faces,
                style=mesh_style,
                color=m_col,
                thickness=mesh_thickness,
                draw_points=mesh_draw_points,
                regions=mesh_regions,
                copy=False,
            )

        # Layer 2: Live FX scanner & lock indicators
        if effective_live_fx:
            from detekface.live_fx import LiveFXRenderer

            fx_renderer = LiveFXRenderer(style=live_fx_style, color=bgr_color)
            fx_renderer.render(display_frame, faces=self._latest_result.faces, copy=False)

        # Layer 3: Primary face overlay shapes
        draw_faces(
            display_frame,
            self._latest_result.faces,
            shape=valid_shape,
            color=bgr_color,
            thickness=valid_thickness,
            show_confidence=show_confidence,
            show_tracking_ids=self.track_faces,
            show_pose=self.estimate_pose,
            show_gestures=show_gestures or self.show_gestures,
            copy=False,
        )

        # Layer 4: Face shape analysis badge and silhouette contour
        if show_face_shape and self.analyze_face_shape:
            scan_phase = (self._frames_read % 60) / 60.0
            eff_effect = face_shape_effect
            if show_scan is not None:
                eff_effect = "scan" if show_scan else None
            for face in self._latest_result.faces:
                if face.face_shape is not None:
                    draw_face_shape_overlay(
                        image=display_frame,
                        face=face,
                        effect=eff_effect,
                        scan_phase=scan_phase,
                        color=bgr_color,
                        thickness=valid_thickness,
                        show_contour=effective_show_contour,
                        show_measurements=show_shape_measurements,
                    )

        # Layer 5: Top-level status metrics (FPS and Face count)
        if show_fps or show_count:
            draw_metrics_overlay(
                display_frame,
                fps=self.fps if show_fps else None,
                face_count=self._latest_result.count if show_count else None,
                color_bgr=bgr_color,
            )

        # Layer 6: AR Face accessories
        eff = effect if effect is not None else (accessory if accessory is not None else self.effect)
        if eff is not None:
            op = accessory_opacity if accessory_opacity is not None else self.accessory_opacity
            sc = accessory_scale if accessory_scale is not None else self.accessory_scale
            yo = accessory_y_offset if accessory_y_offset is not None else self.accessory_y_offset
            if self._accessory_renderer is not None and eff == self.effect:
                display_frame = self._accessory_renderer.render(
                    display_frame,
                    self._latest_result.faces,
                    copy=False,
                )
            else:
                from detekface.accessories import render_accessory

                display_frame = render_accessory(
                    display_frame,
                    self._latest_result.faces,
                    accessory=eff,
                    opacity=op,
                    scale_adjust=sc,
                    y_offset=yo,
                    copy=False,
                )

        # Layer 7: Face Quality feedback
        if show_quality or self.show_quality:
            from detekface.quality import assess_face_quality

            for idx, face in enumerate(self._latest_result.faces):
                qm = assess_face_quality(display_frame, face, face_index=idx)
                bx1, by2 = int(face.bbox.x1), int(face.bbox.y2)
                badge_text = f"Q:{int(qm.quality_score * 100)}% | Sharp:{int(qm.sharpness)} | {qm.brightness_status.upper()}"
                b_col = (0, 0, 255) if qm.is_blurry else ((0, 165, 255) if qm.brightness_status != "optimal" else (0, 255, 128))
                cv2.putText(display_frame, badge_text, (bx1, by2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display_frame, badge_text, (bx1, by2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, b_col, 1, cv2.LINE_AA)

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
    analyze_face_shape: bool = False,
    track_faces: bool = False,
    estimate_pose: bool = False,
    target_fps: float = 60.0,
    async_mode: bool = False,
    stale_timeout_sec: float = 0.5,
    detect_gestures: bool = False,
    on_gesture: Optional[Callable[[GestureEvent], None]] = None,
    gestures: Optional[Sequence[str]] = None,
    gesture_config: Optional[GestureConfig] = None,
    show_gestures: bool = False,
    effect: Optional[str] = None,
    accessory: Optional[str] = None,
    accessory_opacity: float = 1.0,
    accessory_scale: float = 1.0,
    accessory_y_offset: float = 0.0,
) -> CameraSession:
    """Open and return a managed CameraSession context manager.

    Example:
        import detekface

        with detekface.open_camera(camera_index=0, analyze_face_shape=True) as session:
            for result in session:
                print(f"Detected {result.count} faces")
                for face in result.faces:
                    print(face.face_shape.label)
                if result.has_faces:
                    face = session.capture()
                    print(f"Captured face to {face.path}")
                    break
    """
    kwargs = {}
    if analyze_face_shape:
        kwargs["analyze_face_shape"] = True
    if track_faces:
        kwargs["track_faces"] = True
    if estimate_pose:
        kwargs["estimate_pose"] = True
    if target_fps != 60.0:
        kwargs["target_fps"] = target_fps
    if async_mode:
        kwargs["async_mode"] = True
    if stale_timeout_sec != 0.5:
        kwargs["stale_timeout_sec"] = stale_timeout_sec
    if detect_gestures or on_gesture is not None:
        kwargs["detect_gestures"] = True
        kwargs["on_gesture"] = on_gesture
        kwargs["gestures"] = gestures
        kwargs["gesture_config"] = gesture_config
        kwargs["show_gestures"] = show_gestures
    eff = effect if effect is not None else accessory
    if eff is not None:
        kwargs["effect"] = eff
        kwargs["accessory_opacity"] = accessory_opacity
        kwargs["accessory_scale"] = accessory_scale
        kwargs["accessory_y_offset"] = accessory_y_offset

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
        **kwargs,
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
    analyze_face_shape: bool = False,
    show_face_shape: bool = True,
    face_shape_effect: Optional[str] = "scan",
    show_shape_contour: bool = True,
    show_shape_measurements: bool = False,
    show_mesh: bool = False,
    mesh_style: str = "wireframe",
    mesh_color: Optional[Union[str, Tuple[int, int, int]]] = None,
    mesh_thickness: Optional[int] = None,
    mesh_regions: Optional[Sequence[str]] = None,
    mesh_draw_points: Optional[bool] = None,
    show_contour: Optional[bool] = None,
    show_scan: Optional[bool] = None,
    live_fx: bool = False,
    live_fx_style: str = "scan",
    target_fps: float = 60.0,
    async_mode: bool = False,
    stale_timeout_sec: float = 0.5,
    track_faces: bool = False,
    show_tracking_ids: bool = False,
    show_pose: bool = False,
    estimate_pose: bool = False,
    detect_gestures: bool = False,
    on_gesture: Optional[Callable[[GestureEvent], None]] = None,
    gestures: Optional[Sequence[str]] = None,
    gesture_config: Optional[GestureConfig] = None,
    show_gestures: bool = False,
    effect: Optional[str] = None,
    accessory: Optional[str] = None,
    accessory_opacity: float = 1.0,
    accessory_scale: float = 1.0,
    accessory_y_offset: float = 0.0,
    show_quality: bool = False,
    anonymize: Optional[str] = None,
    anonymize_strength: float = 0.5,
    anonymize_color: Union[str, Tuple[int, int, int]] = (0, 0, 0),
    anonymize_mask_shape: str = "box",
) -> None:
    """Open a live webcam feed, detect faces in real time, and display customizable overlays.

    Controls:
        - Press 'c' or 'C' to capture and save the largest detected face.
        - Press 'm' or 'M' to cycle dense facial mesh wireframe styles.
        - Press 'f' or 'F' to cycle Live FX scanner animation styles.
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
        analyze_face_shape: If True, executes dense facial landmarking and geometric shape classification.
        show_face_shape: If True and analyze_face_shape=True, renders shape classification badges.
        face_shape_effect: Biometric visual effect ('scan' for animated laser line, or None).
        show_shape_contour: If True, draws the facial silhouette contour mesh.
        show_shape_measurements: If True, displays live geometric proportion ratios.
        show_mesh: Whether to render dense facial landmark wireframe mesh.
        mesh_style: Mesh visualization style ('wireframe', 'neon', 'minimal', 'contour', 'points').
        mesh_color: Color for facial mesh lines (defaults to main overlay color).
        mesh_thickness: Line stroke width in pixels for mesh.
        mesh_regions: Anatomical facial regions to render.
        mesh_draw_points: Whether to render vertex dots.
        show_contour: Alias for show_shape_contour.
        show_scan: Convenience flag; when True enables Live FX laser scanner overlay.
        live_fx: Whether to render animated Live FX modern computer-vision scanner layer.
        live_fx_style: Live FX animation style ('scan', 'neon', 'wireframe', 'minimal').

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
    if not isinstance(analyze_face_shape, bool):
        raise ConfigurationError(
            message=f"analyze_face_shape must be a boolean, got {type(analyze_face_shape).__name__}.",
            hint="Set analyze_face_shape=True or analyze_face_shape=False.",
        )
    if not isinstance(show_face_shape, bool):
        raise ConfigurationError(
            message=f"show_face_shape must be a boolean, got {type(show_face_shape).__name__}.",
            hint="Set show_face_shape=True or show_face_shape=False.",
        )
    if not isinstance(show_mesh, bool):
        raise ConfigurationError(
            message=f"show_mesh must be a boolean, got {type(show_mesh).__name__}.",
            hint="Set show_mesh=True or show_mesh=False.",
        )

    if show_contour is not None and not isinstance(show_contour, bool):
        raise ConfigurationError(
            message=f"show_contour must be a boolean or None, got {type(show_contour).__name__}.",
            hint="Set show_contour=True or show_contour=False.",
        )
    if show_scan is not None and not isinstance(show_scan, bool):
        raise ConfigurationError(
            message=f"show_scan must be a boolean or None, got {type(show_scan).__name__}.",
            hint="Set show_scan=True or show_scan=False.",
        )

    effective_show_contour = show_shape_contour if show_contour is None else bool(show_contour)
    if show_scan is True:
        live_fx = True

    from detekface.mesh import validate_mesh_style

    if not isinstance(target_fps, (int, float)) or isinstance(target_fps, bool) or target_fps <= 0:
        raise ConfigurationError(
            message=f"target_fps must be a positive number, got {target_fps}.",
            hint="Set target_fps=60.0 or target_fps=30.0.",
        )
    if not isinstance(async_mode, bool):
        raise ConfigurationError(
            message=f"async_mode must be a boolean, got {type(async_mode).__name__}.",
            hint="Set async_mode=True or async_mode=False.",
        )
    if not isinstance(stale_timeout_sec, (int, float)) or isinstance(stale_timeout_sec, bool) or stale_timeout_sec <= 0:
        raise ConfigurationError(
            message=f"stale_timeout_sec must be a positive number, got {stale_timeout_sec}.",
            hint="Set stale_timeout_sec to a duration in seconds like 0.5.",
        )

    if not isinstance(track_faces, bool):
        raise ConfigurationError(
            message=f"track_faces must be a boolean, got {type(track_faces).__name__}.",
            hint="Set track_faces=True or track_faces=False.",
        )
    if not isinstance(show_tracking_ids, bool):
        raise ConfigurationError(
            message=f"show_tracking_ids must be a boolean, got {type(show_tracking_ids).__name__}.",
            hint="Set show_tracking_ids=True or show_tracking_ids=False.",
        )
    if not isinstance(show_pose, bool):
        raise ConfigurationError(
            message=f"show_pose must be a boolean, got {type(show_pose).__name__}.",
            hint="Set show_pose=True or show_pose=False.",
        )
    if not isinstance(estimate_pose, bool):
        raise ConfigurationError(
            message=f"estimate_pose must be a boolean, got {type(estimate_pose).__name__}.",
            hint="Set estimate_pose=True or estimate_pose=False.",
        )
    if not isinstance(detect_gestures, bool):
        raise ConfigurationError(
            message=f"detect_gestures must be a boolean, got {type(detect_gestures).__name__}.",
            hint="Set detect_gestures=True or detect_gestures=False.",
        )
    if not isinstance(show_gestures, bool):
        raise ConfigurationError(
            message=f"show_gestures must be a boolean, got {type(show_gestures).__name__}.",
            hint="Set show_gestures=True or show_gestures=False.",
        )
    if not isinstance(show_quality, bool):
        raise ConfigurationError(
            message=f"show_quality must be a boolean, got {type(show_quality).__name__}.",
            hint="Set show_quality=True or show_quality=False.",
        )

    if on_gesture is not None or show_gestures:
        detect_gestures = True

    if detect_gestures:
        track_faces = True
        estimate_pose = True

    if show_tracking_ids:
        track_faces = True
    if show_pose:
        estimate_pose = True

    valid_mesh_style = validate_mesh_style(mesh_style)

    if not isinstance(live_fx, bool):
        raise ConfigurationError(
            message=f"live_fx must be a boolean, got {type(live_fx).__name__}.",
            hint="Set live_fx=True or live_fx=False.",
        )

    from detekface.live_fx import LiveFXRenderer, validate_live_fx_style

    valid_live_fx_style = validate_live_fx_style(live_fx_style)

    if show_mesh or live_fx or detect_gestures:
        analyze_face_shape = True

    eff = effect if effect is not None else accessory
    current_accessory: Optional[str] = eff
    if current_accessory is not None:
        from detekface.accessories import FaceAccessoryRenderer

        accessory_renderer = FaceAccessoryRenderer(
            accessory=current_accessory,
            opacity=accessory_opacity,
            scale_adjust=accessory_scale,
            y_offset=accessory_y_offset,
            smooth=True,
        )
        track_faces = True
        estimate_pose = True
    else:
        accessory_renderer = None

    session_kwargs = {}
    if analyze_face_shape:
        session_kwargs["analyze_face_shape"] = True
    if show_mesh:
        session_kwargs["mesh"] = True
    if track_faces:
        session_kwargs["track_faces"] = True
    if estimate_pose:
        session_kwargs["estimate_pose"] = True
    if detect_gestures:
        session_kwargs["detect_gestures"] = True
        session_kwargs["on_gesture"] = on_gesture
        session_kwargs["gestures"] = gestures
        session_kwargs["gesture_config"] = gesture_config
        session_kwargs["show_gestures"] = show_gestures
    if current_accessory is not None:
        session_kwargs["effect"] = current_accessory
        session_kwargs["accessory_opacity"] = accessory_opacity
        session_kwargs["accessory_scale"] = accessory_scale
        session_kwargs["accessory_y_offset"] = accessory_y_offset
    if show_quality:
        session_kwargs["show_quality"] = True
    if anonymize is not None:
        session_kwargs["anonymize"] = anonymize
        session_kwargs["anonymize_strength"] = anonymize_strength
        session_kwargs["anonymize_color"] = anonymize_color
        session_kwargs["anonymize_mask_shape"] = anonymize_mask_shape

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
        target_fps=target_fps,
        async_mode=async_mode,
        stale_timeout_sec=stale_timeout_sec,
        **session_kwargs,
    )

    feedback_text: str = ""
    feedback_frames: int = 0
    feedback_color = (0, 255, 0)
    frames_processed = 0

    fx_renderer = LiveFXRenderer(style=valid_live_fx_style, color=bgr_color) if live_fx else None

    try:
        session.open()

        while True:
            t_loop_start = time.perf_counter()
            result = session.read()
            assert session.latest_frame is not None
            display_frame = session.latest_frame.copy()

            # Layer 0: Face Privacy Anonymization
            if anonymize is not None:
                from detekface.anonymize import anonymize_frame

                anonymize_frame(
                    display_frame,
                    result.faces,
                    method=anonymize,
                    strength=anonymize_strength,
                    color=anonymize_color,
                    mask_shape=anonymize_mask_shape,
                    copy=False,
                )

            # Layer 1: Dense facial mesh wireframe (base anatomical layer)
            if show_mesh:
                from detekface.mesh import draw_face_meshes

                m_col = parse_color_to_bgr(mesh_color) if mesh_color is not None else bgr_color
                draw_face_meshes(
                    display_frame,
                    result.faces,
                    style=valid_mesh_style,
                    color=m_col,
                    thickness=mesh_thickness,
                    draw_points=mesh_draw_points,
                    regions=mesh_regions,
                    copy=False,
                )

            # Layer 2: Live FX animated scanner & lock indicators
            if live_fx:
                if fx_renderer is None:
                    fx_renderer = LiveFXRenderer(style=valid_live_fx_style, color=bgr_color)
                fx_renderer.render(
                    display_frame,
                    faces=result.faces,
                    timestamp=time.perf_counter(),
                    copy=False,
                )

            # Layer 3: Primary face overlay shapes
            draw_faces(
                display_frame,
                result.faces,
                shape=valid_shape,
                color=bgr_color,
                thickness=valid_thickness,
                show_confidence=show_confidence,
                show_tracking_ids=show_tracking_ids,
                show_pose=show_pose,
                show_gestures=show_gestures,
                copy=False,
            )

            # Layer 4: Face shape biometric scanner badges and contour silhouette
            if show_face_shape and analyze_face_shape:
                scan_phase = (frames_processed % 60) / 60.0
                eff_effect = face_shape_effect
                if show_scan is not None:
                    eff_effect = "scan" if show_scan else None
                for face in result.faces:
                    if face.face_shape is not None:
                        draw_face_shape_overlay(
                            image=display_frame,
                            face=face,
                            effect=eff_effect,
                            scan_phase=scan_phase,
                            color=bgr_color,
                            thickness=valid_thickness,
                            show_contour=effective_show_contour,
                            show_measurements=show_shape_measurements,
                        )

            # Layer 5: Status metrics (FPS and Face count)
            if show_fps or show_count:
                draw_metrics_overlay(
                    display_frame,
                    fps=session.fps if show_fps else None,
                    face_count=result.count if show_count else None,
                    color_bgr=bgr_color,
                )

            # Layer 6: AR Face accessories
            if current_accessory is not None and accessory_renderer is not None:
                display_frame = accessory_renderer.render(
                    display_frame,
                    result.faces,
                    copy=False,
                )

            # Layer 7: Face Quality feedback
            if show_quality:
                from detekface.quality import assess_face_quality

                for idx, face in enumerate(result.faces):
                    qm = assess_face_quality(display_frame, face, face_index=idx)
                    bx1, by2 = int(face.bbox.x1), int(face.bbox.y2)
                    badge_text = f"Q:{int(qm.quality_score * 100)}% | Sharp:{int(qm.sharpness)} | {qm.brightness_status.upper()}"
                    b_col = (0, 0, 255) if qm.is_blurry else ((0, 165, 255) if qm.brightness_status != "optimal" else (0, 255, 128))
                    cv2.putText(display_frame, badge_text, (bx1, by2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(display_frame, badge_text, (bx1, by2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, b_col, 1, cv2.LINE_AA)

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
            if frames_processed == 0:
                try:
                    cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 1)
                except cv2.error:
                    pass

            frames_processed += 1
            if max_frames is not None and frames_processed >= max_frames:
                break

            # Check keyboard input with frame-rate throttling if async_mode
            if async_mode:
                elapsed = time.perf_counter() - t_loop_start
                target_interval = 1.0 / max(1.0, float(target_fps))
                delay_ms = max(1, int(round((target_interval - elapsed) * 1000.0)))
                key = cv2.waitKey(delay_ms) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            # Close preview on 'q', 'Q', or Escape (27)
            if key in (ord("q"), ord("Q"), 27):
                break

            # Toggle / cycle mesh visualization style on 'm' or 'M'
            if key in (ord("m"), ord("M")):
                from detekface.mesh import SUPPORTED_MESH_STYLES

                show_mesh = True
                curr_idx = (
                    SUPPORTED_MESH_STYLES.index(valid_mesh_style)
                    if valid_mesh_style in SUPPORTED_MESH_STYLES
                    else 0
                )
                valid_mesh_style = SUPPORTED_MESH_STYLES[(curr_idx + 1) % len(SUPPORTED_MESH_STYLES)]
                feedback_text = f"Mesh: {valid_mesh_style}"
                feedback_frames = 45
                feedback_color = (0, 255, 136)
                print(f"[detekface] Switched mesh style to: {valid_mesh_style}")

            # Toggle / cycle Live FX on 'f' or 'F'
            if key in (ord("f"), ord("F")):
                from detekface.live_fx import SUPPORTED_LIVE_FX_STYLES

                if not live_fx:
                    live_fx = True
                    feedback_text = f"Live FX: ON ({valid_live_fx_style})"
                else:
                    curr_fx_idx = (
                        SUPPORTED_LIVE_FX_STYLES.index(valid_live_fx_style)
                        if valid_live_fx_style in SUPPORTED_LIVE_FX_STYLES
                        else 0
                    )
                    next_idx = curr_fx_idx + 1
                    if next_idx >= len(SUPPORTED_LIVE_FX_STYLES):
                        live_fx = False
                        feedback_text = "Live FX: OFF"
                    else:
                        valid_live_fx_style = SUPPORTED_LIVE_FX_STYLES[next_idx]
                        if fx_renderer is not None:
                            fx_renderer.style = valid_live_fx_style
                        feedback_text = f"Live FX: {valid_live_fx_style}"

                feedback_frames = 45
                feedback_color = (0, 255, 136)
                print(f"[detekface] {feedback_text}")

            # Toggle / cycle AR face accessory on 'a' or 'A'
            if key in (ord("a"), ord("A")):
                from detekface.accessories import FaceAccessoryRenderer, SUPPORTED_ACCESSORIES

                acc_cycle = [None] + list(SUPPORTED_ACCESSORIES)
                try:
                    c_idx = acc_cycle.index(current_accessory)
                except ValueError:
                    c_idx = 0
                next_acc = acc_cycle[(c_idx + 1) % len(acc_cycle)]
                current_accessory = next_acc
                if current_accessory is not None:
                    accessory_renderer = FaceAccessoryRenderer(
                        accessory=current_accessory,
                        opacity=accessory_opacity,
                        scale_adjust=accessory_scale,
                        y_offset=accessory_y_offset,
                        smooth=True,
                    )
                    feedback_text = f"Accessory: {current_accessory}"
                else:
                    accessory_renderer = None
                    feedback_text = "Accessory: None"
                feedback_frames = 45
                feedback_color = (0, 255, 136)
                print(f"[detekface] {feedback_text}")

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
