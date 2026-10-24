"""Tests for CameraSession and open_camera advanced camera integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from detekface import (
    BoundingBox,
    CameraError,
    CameraNotFoundError,
    CameraReadError,
    CameraSession,
    CaptureError,
    DetectionResult,
    FaceCapture,
    FaceDetection,
    NoFaceDetectedError,
    open_camera,
)


@pytest.fixture
def mock_cv2_cap():
    """Fixture providing a mocked cv2.VideoCapture instance."""
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap_cls.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        yield mock_cap, mock_cap_cls


def test_open_camera_context_manager_lifecycle(mock_cv2_cap):
    """Verify open_camera enters context, sets is_open=True, and releases on exit."""
    mock_cap, mock_cap_cls = mock_cv2_cap

    with open_camera(camera_index=2) as session:
        mock_cap_cls.assert_called_once_with(2)
        assert session.is_open is True
        assert session.is_active is True
        assert session.camera_index == 2

    assert session.is_open is False
    mock_cap.release.assert_called_once()


def test_explicit_open_and_close_idempotent(mock_cv2_cap):
    """Verify manual open() and close() calls work and close() is idempotent."""
    mock_cap, mock_cap_cls = mock_cv2_cap

    session = open_camera(camera_index=0, auto_open=False)
    assert session.is_open is False

    session.open()
    assert session.is_open is True
    mock_cap_cls.assert_called_once_with(0)

    # Calling open again when already open is a no-op
    session.open()
    assert mock_cap_cls.call_count == 1

    session.close()
    assert session.is_open is False
    mock_cap.release.assert_called_once()

    # Calling close multiple times is safe and idempotent
    session.close()
    assert mock_cap.release.call_count == 1


def test_camera_not_found_on_open(mock_cv2_cap):
    """Verify CameraNotFoundError is raised when camera device cannot be opened."""
    mock_cap, _ = mock_cv2_cap
    mock_cap.isOpened.return_value = False

    with pytest.raises(CameraNotFoundError) as exc_info:
        with open_camera(camera_index=1):
            pass

    assert "Could not open camera device with index 1" in str(exc_info.value)
    assert exc_info.value.hint is not None
    mock_cap.release.assert_called_once()


def test_read_structured_detections_and_raw_frames(mock_cv2_cap):
    """Verify session.read() and session.read_raw() return proper types and preserve pristine frames."""
    mock_cap, _ = mock_cv2_cap
    dummy_bgr = np.full((120, 160, 3), 42, dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_bgr)

    mock_detector = MagicMock()
    mock_bbox = BoundingBox(10, 10, 50, 50)
    mock_face = FaceDetection(bbox=mock_bbox, confidence=0.92)
    mock_detector.detect.return_value = DetectionResult(faces=[mock_face], image=dummy_bgr)

    with open_camera(detector=mock_detector, mirror=False) as session:
        # Read structured detection
        result = session.read()
        assert isinstance(result, DetectionResult)
        assert result.count == 1
        assert result.faces[0].confidence == pytest.approx(0.92)
        assert session.latest_result is result
        assert session.frames_read == 1

        # Check frame ownership & pristine state
        assert session.latest_frame is not None
        assert session.latest_frame.shape == (120, 160, 3)

        # Read raw frame without detection
        raw = session.read_raw()
        assert isinstance(raw, np.ndarray)
        assert raw.shape == (120, 160, 3)
        assert session.frames_read == 2


def test_reading_from_closed_session_raises_camera_error():
    """Verify reading from a closed session raises CameraError."""
    session = CameraSession(auto_open=False)
    assert session.is_open is False

    with pytest.raises(CameraError) as exc_info:
        session.read()
    assert "Cannot read from a closed CameraSession" in str(exc_info.value)

    with pytest.raises(CameraError):
        session.read_raw()


def test_camera_read_failure_raises_camera_read_error(mock_cv2_cap):
    """Verify CameraReadError is raised when frame reading fails."""
    mock_cap, _ = mock_cv2_cap
    mock_cap.read.return_value = (False, None)

    with open_camera() as session:
        with pytest.raises(CameraReadError) as exc_info:
            session.read()
        assert "Failed to read video frame" in str(exc_info.value)
        assert exc_info.value.hint is not None


def test_frame_iteration_protocol_and_early_exit(mock_cv2_cap):
    """Verify session can be used as an iterator and stops safely on break."""
    mock_cap, _ = mock_cv2_cap
    dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_bgr)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=dummy_bgr)

    read_count = 0
    with open_camera(detector=mock_detector) as session:
        for result in session:
            assert isinstance(result, DetectionResult)
            read_count += 1
            if read_count == 3:
                break

    assert read_count == 3
    # Device must be closed upon exiting context manager despite break
    assert session.is_open is False
    mock_cap.release.assert_called_once()


def test_stream_generator_max_frames(mock_cv2_cap):
    """Verify session.stream(max_frames=N) yields exactly N frames."""
    mock_cap, _ = mock_cv2_cap
    dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_bgr)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=dummy_bgr)

    with open_camera(detector=mock_detector) as session:
        results = list(session.stream(max_frames=4))
        assert len(results) == 4
        assert session.frames_read == 4


def test_process_callback_stops_when_returning_false(mock_cv2_cap):
    """Verify session.process(callback) passes frames and halts when callback returns False."""
    mock_cap, _ = mock_cv2_cap
    dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_bgr)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=dummy_bgr)

    calls = []

    def my_callback(result: DetectionResult):
        calls.append(result)
        if len(calls) == 2:
            return False  # Request stop
        return True

    with open_camera(detector=mock_detector) as session:
        processed = session.process(my_callback, max_frames=10)
        assert processed == 2
        assert len(calls) == 2


def test_programmatic_face_capture(mock_cv2_cap, tmp_path: Path):
    """Verify session.capture() captures face from latest frame and reuses capture utility."""
    mock_cap, _ = mock_cv2_cap
    dummy_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_bgr)

    bbox = BoundingBox(20, 20, 60, 60)
    face = FaceDetection(bbox=bbox, confidence=0.94)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[face], image=dummy_bgr)

    with open_camera(detector=mock_detector) as session:
        # Calling capture before read raises CaptureError
        with pytest.raises(CaptureError) as exc_early:
            session.capture()
        assert "No frames have been read yet" in str(exc_early.value)

        # Read a frame
        session.read()

        # Capture without saving to disk
        captured = session.capture(save=False)
        assert isinstance(captured, FaceCapture)
        assert captured.width == 60
        assert captured.height == 60
        assert captured.confidence == pytest.approx(0.94)

        # Capture with disk save
        saved = session.capture(save_dir=tmp_path, save=True)
        assert saved.path is not None
        assert saved.path.exists()


def test_programmatic_capture_zero_faces_raises_error(mock_cv2_cap):
    """Verify session.capture() raises NoFaceDetectedError when latest frame has no faces."""
    mock_cap, _ = mock_cv2_cap
    dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_bgr)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=dummy_bgr)

    with open_camera(detector=mock_detector) as session:
        session.read()
        with pytest.raises(NoFaceDetectedError) as exc_info:
            session.capture()
        assert "No faces detected" in str(exc_info.value)


def test_headless_by_default_and_optional_render(mock_cv2_cap):
    """Verify CameraSession does not touch cv2 window functions unless show() is called."""
    mock_cap, _ = mock_cv2_cap
    dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_bgr)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=dummy_bgr)

    with (
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.destroyWindow") as mock_destroy,
    ):
        with open_camera(detector=mock_detector) as session:
            session.read()
            # session.render creates annotated image without displaying a window
            annotated = session.render(shape="corners", color="#00FF88")
            assert isinstance(annotated, np.ndarray)
            mock_imshow.assert_not_called()

        # Headless session exit does not call destroyWindow
        mock_destroy.assert_not_called()


def test_show_window_and_cleanup_on_close(mock_cv2_cap):
    """Verify show() displays an OpenCV window and close() cleans it up."""
    mock_cap, _ = mock_cv2_cap
    dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_bgr)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=dummy_bgr)

    with (
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.waitKey", return_value=ord("a")),
        patch("cv2.destroyWindow") as mock_destroy,
    ):
        with open_camera(detector=mock_detector, window_title="TestWin") as session:
            session.read()
            key = session.show()
            assert key == ord("a")
            mock_imshow.assert_called_once()
            args, _ = mock_imshow.call_args
            assert args[0] == "TestWin"

        mock_destroy.assert_called_once_with("TestWin")


def test_session_cleanup_on_unhandled_exception(mock_cv2_cap):
    """Verify camera resource is released even when consumer raises an unhandled exception."""
    mock_cap, _ = mock_cv2_cap
    mock_cap.read.return_value = (True, np.zeros((50, 50, 3), dtype=np.uint8))

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=np.zeros((50, 50, 3), dtype=np.uint8))

    with pytest.raises(ValueError) as exc_info:
        with open_camera(detector=mock_detector) as session:
            session.read()
            raise ValueError("Consumer crash inside context block")

    assert "Consumer crash inside context block" in str(exc_info.value)
    assert session.is_open is False
    mock_cap.release.assert_called_once()
