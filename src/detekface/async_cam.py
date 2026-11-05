"""Asynchronous multi-threaded camera engine for detekface 0.3.0.

Provides decoupled, high-throughput, low-latency processing across 4 independent stages:
1. Frame Acquisition: Dedicated thread continuously fetching fresh hardware frames with timestamps.
2. Model Inference: Dedicated worker thread performing face detection and dense landmarking.
3. Temporal Tracking: Smoothing results and maintaining continuity across inference latency.
4. Preview Rendering: Running at monitor / display refresh rate (targeting 60 FPS) without lockstep inference stalling.

Performance & Memory Guarantees:
- Never unbounded queues: bounded buffer (default size 1) drops stale frames so inference always works on the freshest frame.
- Explicit lifecycle control: threads cleanly join on exit or exception.
- Stale detection timeout: if inference lags or fails, old detections fade/hide gracefully after staleness threshold.
- Accurate metrics: independently tracks capture FPS, inference FPS, and rendering FPS without fabrication.
"""

from dataclasses import dataclass
from queue import Empty, Queue
import threading
import time
from typing import Optional, Sequence, Tuple
import cv2
import numpy as np

from detekface.detector import FaceDetector
from detekface.drawing import FPSCounter
from detekface.exceptions import CameraNotFoundError, ConfigurationError
from detekface.face_shape import TemporalShapeTracker
from detekface.types import DetectionResult, FaceDetection


@dataclass(frozen=True)
class CameraMetrics:
    """Independent real-time performance telemetry metrics."""

    capture_fps: float
    inference_fps: float
    render_fps: float
    inference_latency_ms: float
    actual_width: int
    actual_height: int
    hardware_fps: float


@dataclass(frozen=True)
class TimestampedFrame:
    """Video frame bundled with high-resolution wall-clock acquisition metadata."""

    frame: np.ndarray
    timestamp: float
    frame_index: int


class AsyncCameraEngine:
    """Multi-threaded pipeline decoupling frame acquisition, inference, and rendering."""

    def __init__(
        self,
        camera_index: int = 0,
        mirror: bool = True,
        detector: Optional[FaceDetector] = None,
        target_fps: float = 60.0,
        resolution: Optional[Tuple[int, int]] = None,
        backend: str = "yunet",
        confidence_threshold: float = 0.7,
        input_size: Optional[Tuple[int, int]] = None,
        analyze_face_shape: bool = False,
        mesh: bool = False,
        stale_timeout_sec: float = 0.5,
        queue_maxsize: int = 1,
    ) -> None:
        self.camera_index = int(camera_index)
        self.mirror = bool(mirror)
        self.target_fps = max(1.0, float(target_fps))
        self.stale_timeout_sec = max(0.05, float(stale_timeout_sec))
        self.analyze_face_shape = bool(analyze_face_shape)
        self.mesh = bool(mesh)
        self._shape_tracker = TemporalShapeTracker() if self.analyze_face_shape else None

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

        if detector is None:
            self.detector = FaceDetector(
                backend=backend,
                confidence_threshold=confidence_threshold,
                input_size=input_size,
            )
        else:
            self.detector = detector

        # Threading state & bounded communication channels
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None

        # Bounded single-slot queues prevent stale frame build-up
        self._frame_queue: Queue[TimestampedFrame] = Queue(maxsize=max(1, queue_maxsize))
        self._latest_capture: Optional[TimestampedFrame] = None
        self._latest_capture_lock = threading.Lock()

        # Shared inference result cache
        self._latest_result: Optional[DetectionResult] = None
        self._latest_result_timestamp: float = 0.0
        self._result_lock = threading.Lock()

        # Independent FPS throughput counters
        self._capture_fps_counter = FPSCounter(window_size=30)
        self._inference_fps_counter = FPSCounter(window_size=30)
        self._render_fps_counter = FPSCounter(window_size=30)
        self._last_inference_latency_ms: float = 0.0

        # Hardware verified properties
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._actual_hardware_fps: float = 0.0

    @property
    def is_running(self) -> bool:
        """Return True if acquisition and inference threads are actively running."""
        return self._running

    @property
    def metrics(self) -> CameraMetrics:
        """Current independent FPS counters and hardware verification metrics."""
        return CameraMetrics(
            capture_fps=self._capture_fps_counter.fps,
            inference_fps=self._inference_fps_counter.fps,
            render_fps=self._render_fps_counter.fps,
            inference_latency_ms=self._last_inference_latency_ms,
            actual_width=self._actual_width,
            actual_height=self._actual_height,
            hardware_fps=self._actual_hardware_fps,
        )

    def start(self, cap: Optional[cv2.VideoCapture] = None) -> "AsyncCameraEngine":
        """Start hardware camera capture and background asynchronous inference threads."""
        if self._running:
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
                    hint="Verify your camera is connected and permissions are granted.",
                )
            self._cap = cap

        # 1. Request target resolution if specified
        if self.resolution is not None:
            w, h = self.resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # 2. Request target FPS (e.g. 60.0)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # 3. Verify actual negotiated properties
        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._actual_hardware_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)

        self._running = True

        # Launch dedicated frame acquisition thread
        self._capture_thread = threading.Thread(
            target=self._capture_worker,
            name="detekface-capture",
            daemon=True,
        )
        # Launch dedicated background inference thread
        self._inference_thread = threading.Thread(
            target=self._inference_worker,
            name="detekface-inference",
            daemon=True,
        )

        self._capture_thread.start()
        self._inference_thread.start()
        return self

    def stop(self) -> None:
        """Stop background threads, release camera hardware, and clean up channels."""
        self._running = False

        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None

        if self._inference_thread is not None and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=1.0)
            self._inference_thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        # Drain frame queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break

    def _capture_worker(self) -> None:
        """Continuously acquire fresh frames from hardware with zero stalling."""
        frame_idx = 0
        while self._running and self._cap is not None and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                # Brief sleep to avoid tight spin on transient hardware stalls
                time.sleep(0.005)
                continue

            now = time.perf_counter()
            self._capture_fps_counter.tick(now)

            if self.mirror:
                clean_frame = cv2.flip(frame, 1)
            else:
                clean_frame = frame.copy()

            ts_frame = TimestampedFrame(frame=clean_frame, timestamp=now, frame_index=frame_idx)
            frame_idx += 1

            # Update latest capture slot
            with self._latest_capture_lock:
                self._latest_capture = ts_frame

            # Feed inference queue; drop previous stale item if inference is busy
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except Empty:
                    pass
            try:
                self._frame_queue.put_nowait(ts_frame)
            except Exception:
                pass

    def _inference_worker(self) -> None:
        """Continuously fetch the latest frame and run model inference asynchronously."""
        while self._running:
            try:
                ts_frame = self._frame_queue.get(timeout=0.05)
            except Empty:
                continue

            t_start = time.perf_counter()
            try:
                res = self.detector.detect(
                    ts_frame.frame,
                    analyze_face_shape=self.analyze_face_shape,
                    mesh=self.mesh,
                )
                if self.analyze_face_shape and self._shape_tracker is not None:
                    smoothed_faces = self._shape_tracker.update(res.faces)
                    res = DetectionResult(faces=smoothed_faces, image=ts_frame.frame)
            except Exception:
                # Safely ignore transient frame inference faults without crashing preview
                res = DetectionResult(faces=[], image=ts_frame.frame)

            t_end = time.perf_counter()
            self._last_inference_latency_ms = (t_end - t_start) * 1000.0
            self._inference_fps_counter.tick(t_end)

            with self._result_lock:
                self._latest_result = res
                self._latest_result_timestamp = ts_frame.timestamp

    def get_latest_state(self, timeout: float = 0.0) -> Tuple[Optional[np.ndarray], Sequence[FaceDetection], float]:
        """Fetch the latest acquired frame and most recent valid face detections.

        Args:
            timeout: Optional seconds to wait for the initial frame if not yet arrived.

        Returns:
            Tuple containing:
            - Latest unannotated frame (or None if no frame has been read yet).
            - List of FaceDetection objects (empty if detections are excessively stale).
            - Acquisition timestamp of the returned frame.
        """
        if timeout > 0:
            t_wait_start = time.perf_counter()
            while self._running and self._latest_capture is None:
                if (time.perf_counter() - t_wait_start) >= timeout:
                    break
                time.sleep(0.005)

        now = time.perf_counter()
        self._render_fps_counter.tick(now)

        with self._latest_capture_lock:
            ts_frame = self._latest_capture

        if ts_frame is None:
            return None, [], now

        with self._result_lock:
            res = self._latest_result
            res_time = self._latest_result_timestamp

        # Check for excessive result staleness
        if res is not None and (now - res_time) <= self.stale_timeout_sec:
            faces = res.faces
        else:
            # Result is excessively stale or inference hasn't completed yet
            faces = []

        return ts_frame.frame, faces, ts_frame.timestamp

    def __enter__(self) -> "AsyncCameraEngine":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
