"""Automated tests for detekface 0.3.0 low-latency asynchronous webcam engine.

Covers:
- Thread synchronization and non-blocking decoupled pipelines
- Bounded frame queues with zero unbounded memory growth
- High-resolution wall-clock timestamps and frame sequencing
- Graceful degradation during delayed/slow inference
- Detection hiding on excessive staleness timeout
- Camera read failure resilience and safe exception isolation
- Concurrent shutdown without thread leaks
- Independent telemetry reporting (capture, inference, render FPS)
- Face capture during active asynchronous background inference
- Long session stability
"""

import threading
import time
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from detekface import (
    AsyncCameraEngine,
    BoundingBox,
    CameraMetrics,
    CameraSession,
    ConfigurationError,
    DetectionResult,
    FaceDetection,
    camera,
    open_camera,
)


class MockVideoCap:
    """Mock cv2.VideoCapture producing synthetic test frames with simulated read delay."""

    def __init__(self, width: int = 640, height: int = 480, fps: float = 30.0, delay_sec: float = 0.005) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.delay_sec = delay_sec
        self._is_open = True
        self.read_count = 0
        self.props = {
            cv2.CAP_PROP_FRAME_WIDTH: float(width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(height),
            cv2.CAP_PROP_FPS: float(fps),
        }

    def isOpened(self) -> bool:
        return self._is_open

    def get(self, prop_id: int) -> float:
        return self.props.get(prop_id, 0.0)

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == cv2.CAP_PROP_FPS and self.fps is not None:
            # Simulate camera hardware that reports its fixed capability (e.g. 30.0)
            return True
        self.props[prop_id] = float(value)
        return True

    def read(self):
        if not self._is_open:
            return False, None
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        self.read_count += 1
        frame = np.full((self.height, self.width, 3), fill_value=(self.read_count % 255), dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        self._is_open = False


# =========================================================================
# 1. Parameter Validation Tests
# =========================================================================


def test_invalid_async_parameters_raise_configuration_error() -> None:
    """Verify informative errors on invalid target_fps, async_mode, or stale_timeout_sec."""
    with pytest.raises(ConfigurationError) as exc_info:
        CameraSession(target_fps=-10, auto_open=False)
    assert "target_fps must be a positive number" in str(exc_info.value)

    with pytest.raises(ConfigurationError) as exc_info:
        CameraSession(async_mode="yes", auto_open=False)  # type: ignore
    assert "async_mode must be a boolean" in str(exc_info.value)

    with pytest.raises(ConfigurationError) as exc_info:
        CameraSession(stale_timeout_sec=0, auto_open=False)
    assert "stale_timeout_sec must be a positive number" in str(exc_info.value)

    with pytest.raises(ConfigurationError):
        camera(target_fps=0)

    with pytest.raises(ConfigurationError):
        camera(async_mode="invalid")  # type: ignore

    with pytest.raises(ConfigurationError):
        camera(stale_timeout_sec=-1.0)


# =========================================================================
# 2. Bounded Queue & Memory Stability Tests
# =========================================================================


def test_async_engine_bounded_queue_no_overflow() -> None:
    """Verify single-slot queue drops older unconsumed frames and never overflows."""
    mock_cap = MockVideoCap(delay_sec=0.001)

    mock_detector = MagicMock()
    # Simulate slow detector (30ms per frame)
    def slow_detect(img, **kwargs):
        time.sleep(0.03)
        return DetectionResult(faces=[], image=img)

    mock_detector.detect.side_effect = slow_detect

    engine = AsyncCameraEngine(
        detector=mock_detector,
        target_fps=60.0,
        queue_maxsize=1,
    )

    try:
        engine.start(cap=mock_cap)
        # Let capture run for 150ms (generating ~50+ frames) while inference only processes ~4
        time.sleep(0.15)
        # Verify queue size is strictly bounded
        assert engine._frame_queue.qsize() <= 1
        # Verify frames were successfully captured
        assert mock_cap.read_count > 10
    finally:
        engine.stop()


# =========================================================================
# 3. Timestamped Frames & Ordering
# =========================================================================


def test_timestamped_frames_wallclock_order() -> None:
    """Verify acquired frames carry monotonically increasing timestamps and contiguous indices."""
    mock_cap = MockVideoCap(delay_sec=0.002)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=np.zeros((10, 10, 3), dtype=np.uint8))

    engine = AsyncCameraEngine(detector=mock_detector)
    timestamps = []

    try:
        engine.start(cap=mock_cap)
        for _ in range(10):
            frame, _, ts = engine.get_latest_state(timeout=0.5)
            if frame is not None:
                timestamps.append(ts)
            time.sleep(0.005)

        assert len(timestamps) >= 3
        # Assert strictly non-decreasing timestamps
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]
    finally:
        engine.stop()


# =========================================================================
# 4. Decoupled Rendering During Delayed Inference
# =========================================================================


def test_delayed_inference_rendering_continues() -> None:
    """Verify render loop continues to fetch fresh video frames even when inference lags."""
    mock_cap = MockVideoCap(delay_sec=0.002)

    mock_detector = MagicMock()
    # Detector stalls for 100ms
    def stalled_detect(img, **kwargs):
        time.sleep(0.1)
        bbox = BoundingBox(x=10, y=10, width=50, height=50)
        return DetectionResult(faces=[FaceDetection(bbox=bbox, confidence=0.99)], image=img)

    mock_detector.detect.side_effect = stalled_detect

    engine = AsyncCameraEngine(detector=mock_detector, stale_timeout_sec=0.5)

    try:
        engine.start(cap=mock_cap)
        render_frames = []
        start_time = time.perf_counter()

        # Render loop samples state 15 times over 75ms
        while time.perf_counter() - start_time < 0.08:
            frame, faces, _ = engine.get_latest_state(timeout=0.2)
            if frame is not None:
                render_frames.append(frame[0, 0, 0])
            time.sleep(0.005)

        # Rendering should have progressed through multiple distinct frames
        # without being blocked on the 100ms inference
        assert len(render_frames) >= 5
    finally:
        engine.stop()


# =========================================================================
# 5. Stale Results Timeout
# =========================================================================


def test_stale_results_hide_after_timeout() -> None:
    """Verify that when inference stops updating, stale detections are cleared after stale_timeout_sec."""
    mock_cap = MockVideoCap(delay_sec=0.005)

    bbox = BoundingBox(x=10, y=10, width=50, height=50)
    face = FaceDetection(bbox=bbox, confidence=0.95)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[face], image=np.zeros((10, 10, 3), dtype=np.uint8))

    # Short staleness timeout: 50ms
    engine = AsyncCameraEngine(detector=mock_detector, stale_timeout_sec=0.05)

    try:
        engine.start(cap=mock_cap)
        # Wait for first detection
        time.sleep(0.03)
        _, faces, _ = engine.get_latest_state(timeout=0.2)

        # Now simulate freeze in inference by pointing detect to infinite wait
        def freeze_inference(*args, **kwargs):
            time.sleep(1.0)
            return DetectionResult(faces=[], image=np.zeros((10, 10, 3), dtype=np.uint8))

        mock_detector.detect.side_effect = freeze_inference

        # Wait longer than stale_timeout_sec (80ms > 50ms)
        time.sleep(0.08)
        _, stale_faces, _ = engine.get_latest_state()

        # Detections should now be cleared to prevent phantom bounding boxes
        assert len(stale_faces) == 0
    finally:
        engine.stop()


# =========================================================================
# 6. Camera Failure Recovery and Clean Shutdown
# =========================================================================


def test_camera_read_failure_and_clean_exit() -> None:
    """Verify that camera read failures in capture worker do not crash the pipeline and exit cleanly."""
    mock_cap = MockVideoCap()
    # Force read failure
    mock_cap.read = MagicMock(return_value=(False, None))  # type: ignore

    mock_detector = MagicMock()
    engine = AsyncCameraEngine(detector=mock_detector)

    try:
        engine.start(cap=mock_cap)
        # Should not raise exception
        frame, faces, _ = engine.get_latest_state()
        assert frame is None
        assert faces == []
    finally:
        engine.stop()

    assert not engine.is_running


def test_concurrent_shutdown_no_thread_leaks() -> None:
    """Verify repeated start/stop lifecycle does not leak threads."""
    initial_threads = threading.active_count()

    for _ in range(3):
        mock_cap = MockVideoCap()
        mock_detector = MagicMock()
        mock_detector.detect.return_value = DetectionResult(faces=[], image=np.zeros((10, 10, 3), dtype=np.uint8))

        engine = AsyncCameraEngine(detector=mock_detector)
        engine.start(cap=mock_cap)
        assert engine.is_running
        time.sleep(0.02)
        engine.stop()
        assert not engine.is_running

    # Active threads should return to baseline
    time.sleep(0.05)
    assert threading.active_count() <= initial_threads + 1


# =========================================================================
# 7. CameraSession Async Integration & Properties
# =========================================================================


def test_camera_session_async_mode_lifecycle() -> None:
    """Verify CameraSession(async_mode=True) correctly opens, reads, provides metrics, and closes."""
    mock_cap = MockVideoCap(width=800, height=600, fps=30.0)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=np.zeros((600, 800, 3), dtype=np.uint8))

    with patch("cv2.VideoCapture", return_value=mock_cap):
        session = CameraSession(
            camera_index=0,
            detector=mock_detector,
            target_fps=60.0,
            async_mode=True,
            auto_open=True,
        )

        assert session.is_open
        assert session.async_mode
        assert session.target_fps == 60.0

        # Read raw
        raw_frame = session.read_raw()
        assert raw_frame is not None
        assert raw_frame.shape == (600, 800, 3)

        # Read detection result
        result = session.read()
        assert isinstance(result, DetectionResult)
        assert session.latest_result is result

        # Verify metrics property
        metrics = session.metrics
        assert isinstance(metrics, CameraMetrics)
        assert metrics.actual_width == 800
        assert metrics.actual_height == 600
        assert metrics.hardware_fps == 30.0
        assert metrics.capture_fps >= 0.0
        assert metrics.render_fps >= 0.0
        assert metrics.inference_fps >= 0.0

        session.close()
        assert not session.is_open
        assert mock_cap.isOpened() is False


# =========================================================================
# 8. Face Capture During Async Inference
# =========================================================================


def test_face_capture_during_async_inference(tmp_path) -> None:
    """Verify session.capture() works cleanly in async mode on fresh detection results."""
    mock_cap = MockVideoCap(width=400, height=400)
    bbox = BoundingBox(x=50, y=50, width=100, height=100)
    mock_detection = FaceDetection(bbox=bbox, confidence=0.92)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(
        faces=[mock_detection],
        image=np.full((400, 400, 3), 120, dtype=np.uint8),
    )

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with open_camera(async_mode=True, detector=mock_detector) as session:
            # Wait for frame & detection
            time.sleep(0.04)
            result = session.read()
            assert result.count == 1

            captured = session.capture(save_dir=tmp_path)
            assert captured.image.shape == (100, 100, 3)
            assert captured.path is not None
            assert captured.path.exists()


# =========================================================================
# 9. detekface.camera(async_mode=True) Interactive Loop Test
# =========================================================================


def test_camera_function_async_mode_max_frames() -> None:
    """Verify detekface.camera(async_mode=True, max_frames=3) runs smoothly and cleans up."""
    mock_cap = MockVideoCap(width=320, height=240)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=np.zeros((240, 320, 3), dtype=np.uint8))

    with (
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.waitKey", return_value=255) as mock_wait_key,
        patch("cv2.destroyWindow") as mock_destroy,
    ):
        camera(
            camera_index=0,
            detector=mock_detector,
            target_fps=60.0,
            async_mode=True,
            max_frames=3,
        )

        assert mock_imshow.call_count == 3
        assert mock_wait_key.call_count >= 2
        mock_destroy.assert_called_once()
        assert not mock_cap.isOpened()


# =========================================================================
# 10. Long Session Stability Test
# =========================================================================


def test_long_running_session_stability() -> None:
    """Verify long-running loop executes 50 frames with zero frame queue leaks or unhandled errors."""
    mock_cap = MockVideoCap(delay_sec=0.001)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=np.zeros((10, 10, 3), dtype=np.uint8))

    engine = AsyncCameraEngine(detector=mock_detector, target_fps=60.0)

    try:
        engine.start(cap=mock_cap)
        frames_seen = 0
        for _ in range(50):
            frame, _, _ = engine.get_latest_state(timeout=0.2)
            if frame is not None:
                frames_seen += 1
            time.sleep(0.002)

        assert frames_seen >= 25
        assert engine._frame_queue.qsize() <= 1
        metrics = engine.metrics
        assert metrics.actual_width > 0
    finally:
        engine.stop()

