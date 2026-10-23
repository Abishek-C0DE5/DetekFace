"""Tests for live webcam loop, frame processing, box drawing, and resource cleanup."""

from unittest.mock import MagicMock, call, patch
import cv2
import numpy as np
import pytest

from detekface import (
    BoundingBox,
    CameraError,
    CameraNotFoundError,
    CameraReadError,
    DetectionResult,
    FaceDetection,
    camera,
)


@pytest.fixture
def mock_cv2_camera():
    """Fixture providing mocked cv2 VideoCapture and GUI window functions."""
    with (
        patch("cv2.VideoCapture") as mock_cap_cls,
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.waitKey") as mock_wait_key,
        patch("cv2.destroyAllWindows") as mock_destroy,
    ):
        mock_cap = MagicMock()
        mock_cap_cls.return_value = mock_cap
        yield mock_cap, mock_imshow, mock_wait_key, mock_destroy


def test_unavailable_camera_raises_camera_not_found(mock_cv2_camera) -> None:
    """Verify CameraNotFoundError is raised and resources are cleaned up when camera cannot be opened."""
    mock_cap, _, _, mock_destroy = mock_cv2_camera
    mock_cap.isOpened.return_value = False

    with pytest.raises(CameraNotFoundError) as exc_info:
        camera(camera_index=99)

    assert "Could not open camera device with index 99" in str(exc_info.value)
    assert exc_info.value.hint is not None
    mock_cap.release.assert_called_once()
    mock_destroy.assert_called_once()


def test_failed_frame_read_raises_camera_read_error(mock_cv2_camera) -> None:
    """Verify CameraReadError is raised and cleanup occurs when reading a frame fails."""
    mock_cap, _, _, mock_destroy = mock_cv2_camera
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)

    with pytest.raises(CameraReadError) as exc_info:
        camera(camera_index=0)

    assert "Failed to read video frame" in str(exc_info.value)
    assert exc_info.value.hint is not None
    mock_cap.release.assert_called_once()
    mock_destroy.assert_called_once()


def test_cleanup_occurs_on_normal_exit_q_key(mock_cv2_camera) -> None:
    """Verify clean exit when 'q' key is pressed."""
    mock_cap, mock_imshow, mock_wait_key, mock_destroy = mock_cv2_camera
    mock_cap.isOpened.return_value = True

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_frame)
    # Simulate pressing 'q' on first waitKey
    mock_wait_key.return_value = ord("q")

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=dummy_frame)

    camera(camera_index=0, detector=mock_detector)

    mock_cap.release.assert_called_once()
    mock_destroy.assert_called_once()
    mock_imshow.assert_called_once()


def test_cleanup_occurs_on_normal_exit_escape_key(mock_cv2_camera) -> None:
    """Verify clean exit when Escape (27) is pressed."""
    mock_cap, _, mock_wait_key, mock_destroy = mock_cv2_camera
    mock_cap.isOpened.return_value = True

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_frame)
    mock_wait_key.return_value = 27  # Escape

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=dummy_frame)

    camera(camera_index=0, detector=mock_detector)

    mock_cap.release.assert_called_once()
    mock_destroy.assert_called_once()


def test_cleanup_occurs_on_unexpected_exception(mock_cv2_camera) -> None:
    """Verify resources are guaranteed to be released even if an unexpected exception occurs inside the loop."""
    mock_cap, _, _, mock_destroy = mock_cv2_camera
    mock_cap.isOpened.return_value = True

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_frame)

    mock_detector = MagicMock()
    mock_detector.detect.side_effect = RuntimeError("Simulated unexpected detector crash")

    with pytest.raises(RuntimeError) as exc_info:
        camera(camera_index=0, detector=mock_detector)

    assert "Simulated unexpected detector crash" in str(exc_info.value)
    mock_cap.release.assert_called_once()
    mock_destroy.assert_called_once()


def test_camera_draws_green_box_on_detection(mock_cv2_camera) -> None:
    """Verify that bounding boxes for detected faces are drawn as green rectangles on the frame."""
    mock_cap, mock_imshow, mock_wait_key, _ = mock_cv2_camera
    mock_cap.isOpened.return_value = True

    test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, test_frame)
    mock_wait_key.return_value = ord("q")

    mock_bbox = BoundingBox(x=20, y=30, width=50, height=60)
    mock_detection = FaceDetection(bbox=mock_bbox, confidence=0.95)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[mock_detection], image=test_frame)

    with patch("cv2.rectangle") as mock_rectangle:
        camera(camera_index=0, detector=mock_detector, mirror=False)

        # Expected top-left (20, 30) and bottom-right (20+50=70, 30+60=90), green color (0, 255, 0), thickness 2
        mock_rectangle.assert_called_once_with(
            test_frame,
            (20, 30),
            (70, 90),
            (0, 255, 0),
            2,
        )


def test_mirror_option_flips_frame(mock_cv2_camera) -> None:
    """Verify that mirror=True applies cv2.flip(frame, 1) and mirror=False does not."""
    mock_cap, _, mock_wait_key, _ = mock_cv2_camera
    mock_cap.isOpened.return_value = True

    test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, test_frame)
    mock_wait_key.return_value = ord("q")

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=test_frame)

    with patch("cv2.flip", return_value=test_frame) as mock_flip:
        camera(camera_index=0, detector=mock_detector, mirror=True)
        mock_flip.assert_called_once_with(test_frame, 1)

    with patch("cv2.flip", return_value=test_frame) as mock_flip_disabled:
        camera(camera_index=0, detector=mock_detector, mirror=False)
        mock_flip_disabled.assert_not_called()


# =========================================================================
# Hardware check (runs only if physical camera 0 is detected)
# =========================================================================

def test_real_hardware_webcam_execution_if_available() -> None:
    """Run real camera hardware check if camera device 0 is accessible.

    Opens camera 0, captures up to 2 frames, runs detection, and verifies clean release.
    """
    test_cap = cv2.VideoCapture(0)
    hardware_available = test_cap.isOpened()
    test_cap.release()

    if not hardware_available:
        pytest.skip("No physical webcam hardware detected on index 0.")

    # Run for exactly 2 frames so the test finishes automatically without user input
    camera(camera_index=0, mirror=True, max_frames=2)
