"""Unit and integration tests for face cropping, selection, capture, and camera callbacks."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from detekface import (
    BoundingBox,
    CaptureError,
    CaptureSaveError,
    DetectionResult,
    FaceCapture,
    FaceDetection,
    FaceLandmarks,
    InvalidCropError,
    InvalidImageError,
    NoFaceDetectedError,
    Point,
    camera,
    capture,
    detect,
    get_cache_dir,
)


def _create_dummy_detection(x: int, y: int, width: int, height: int, confidence: float = 0.9) -> FaceDetection:
    """Helper to create a FaceDetection instance with landmarks."""
    bbox = BoundingBox(x=x, y=y, width=width, height=height)
    landmarks = FaceLandmarks(
        right_eye=Point(x + 5, y + 5),
        left_eye=Point(x + 15, y + 5),
        nose_tip=Point(x + 10, y + 10),
        right_mouth=Point(x + 5, y + 15),
        left_mouth=Point(x + 15, y + 15),
    )
    return FaceDetection(bbox=bbox, confidence=confidence, landmarks=landmarks)


def test_capture_single_face(tmp_path: Path) -> None:
    """Verify that capturing a single face produces a valid FaceCapture object and saves to disk."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    face = _create_dummy_detection(x=10, y=10, width=40, height=50, confidence=0.92)
    result = DetectionResult(faces=[face], image=img)

    captured = capture(result, save_dir=tmp_path, filename="test_face.jpg")

    assert isinstance(captured, FaceCapture)
    assert captured.image.shape == (50, 40, 3)
    assert captured.confidence == pytest.approx(0.92)
    assert captured.bbox == face.bbox
    assert captured.path is not None
    assert captured.path.is_file()
    assert captured.path.name == "test_face.jpg"


def test_capture_chooses_largest_face_by_default(tmp_path: Path) -> None:
    """Verify that capture automatically selects the face with the largest area when multiple exist."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    small_face = _create_dummy_detection(x=10, y=10, width=20, height=20, confidence=0.99)  # area = 400
    med_face = _create_dummy_detection(x=50, y=50, width=40, height=40, confidence=0.85)    # area = 1600
    large_face = _create_dummy_detection(x=100, y=100, width=80, height=80, confidence=0.75) # area = 6400

    result = DetectionResult(faces=[small_face, large_face, med_face], image=img)

    captured = capture(result, save_dir=tmp_path, save=False)

    # Must select the large face
    assert captured.bbox == large_face.bbox
    assert captured.image.shape == (80, 80, 3)
    assert captured.confidence == pytest.approx(0.75)


def test_capture_specific_face_index(tmp_path: Path) -> None:
    """Verify selecting a specific face by index instead of the default largest face."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    face_0 = _create_dummy_detection(x=10, y=10, width=20, height=20)
    face_1 = _create_dummy_detection(x=50, y=50, width=60, height=60)
    result = DetectionResult(faces=[face_0, face_1], image=img)

    captured = capture(result, face_index=0, save=False)
    assert captured.bbox == face_0.bbox
    assert captured.image.shape == (20, 20, 3)


def test_capture_invalid_face_index_raises_error() -> None:
    """Verify that an out-of-range face index raises IndexError."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    face = _create_dummy_detection(10, 10, 20, 20)
    result = DetectionResult(faces=[face], image=img)

    with pytest.raises(IndexError):
        capture(result, face_index=5)


def test_capture_no_face_detected_raises_error() -> None:
    """Verify that invoking capture on a result with 0 faces raises NoFaceDetectedError."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    empty_result = DetectionResult(faces=[], image=img)

    with pytest.raises(NoFaceDetectedError) as exc_info:
        capture(empty_result)

    assert "No faces detected" in str(exc_info.value)
    assert exc_info.value.hint is not None


def test_capture_clamps_to_image_boundaries() -> None:
    """Verify that a bounding box partially extending outside the frame is clamped safely."""
    # Image is 100x100; box starts at x=-10, y=-10, width=50, height=60
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    face = _create_dummy_detection(x=-10, y=-10, width=50, height=60)
    result = DetectionResult(faces=[face], image=img)

    captured = capture(result, save=False)
    # Clamped bounds: x: 0 to 40 (width 40), y: 0 to 50 (height 50)
    assert captured.image.shape == (50, 40, 3)


def test_capture_completely_outside_boundary_raises_invalid_crop() -> None:
    """Verify that a bounding box entirely outside image dimensions raises InvalidCropError."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    outside_face = _create_dummy_detection(x=150, y=150, width=50, height=50)
    result = DetectionResult(faces=[outside_face], image=img)

    with pytest.raises(InvalidCropError) as exc_info:
        capture(result)
    assert "empty crop" in str(exc_info.value)


def test_capture_invalid_image_raises_error() -> None:
    """Verify InvalidImageError when passing an invalid result or empty image array."""
    face = _create_dummy_detection(10, 10, 20, 20)
    bad_result = DetectionResult(faces=[face], image=np.array([]))

    with pytest.raises(InvalidImageError):
        capture(bad_result)


def test_capture_file_write_failure_raises_save_error(tmp_path: Path) -> None:
    """Verify that file write failures from cv2.imwrite raise CaptureSaveError."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    face = _create_dummy_detection(10, 10, 20, 20)
    result = DetectionResult(faces=[face], image=img)

    with patch("cv2.imwrite", return_value=False):
        with pytest.raises(CaptureSaveError) as exc_info:
            capture(result, save_dir=tmp_path)
        assert "failed to write image" in str(exc_info.value)


def test_camera_keyboard_capture_with_callback(tmp_path: Path) -> None:
    """Verify that pressing 'c' in the camera loop triggers capture and invokes on_capture callback."""
    dummy_frame = np.full((200, 200, 3), 100, dtype=np.uint8)
    face = _create_dummy_detection(10, 10, 50, 50, confidence=0.88)
    mock_result = DetectionResult(faces=[face], image=dummy_frame)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = mock_result

    received_captures = []

    def capture_callback(c: FaceCapture) -> None:
        received_captures.append(c)

    with (
        patch("cv2.VideoCapture") as mock_cap_cls,
        patch("cv2.imshow"),
        patch("cv2.waitKey") as mock_wait_key,
        patch("cv2.destroyAllWindows"),
    ):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap_cls.return_value = mock_cap

        # First frame: press 'c' (trigger capture)
        # Second frame: press 'q' (exit)
        mock_wait_key.side_effect = [ord("c"), ord("q")]

        camera(
            camera_index=0,
            detector=mock_detector,
            capture_dir=tmp_path,
            on_capture=capture_callback,
        )

    assert len(received_captures) == 1
    cap = received_captures[0]
    assert isinstance(cap, FaceCapture)
    assert cap.bbox == face.bbox
    assert cap.image.shape == (50, 50, 3)
    assert cap.path is not None
    assert cap.path.is_file()


def test_camera_keyboard_capture_no_face_graceful_handling(tmp_path: Path) -> None:
    """Verify that pressing 'c' when no faces exist does not crash, does not save files, and loop continues."""
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    empty_result = DetectionResult(faces=[], image=dummy_frame)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = empty_result

    callback = MagicMock()

    with (
        patch("cv2.VideoCapture") as mock_cap_cls,
        patch("cv2.imshow"),
        patch("cv2.waitKey") as mock_wait_key,
        patch("cv2.destroyAllWindows"),
    ):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap_cls.return_value = mock_cap

        # Press 'C', then press 'q'
        mock_wait_key.side_effect = [ord("C"), ord("q")]

        camera(
            camera_index=0,
            detector=mock_detector,
            capture_dir=tmp_path,
            on_capture=callback,
        )

    # Callback must not have been called
    callback.assert_not_called()
    # No files should have been created in tmp_path
    assert list(tmp_path.glob("*.jpg")) == []


def test_real_model_detection_and_capture_integration(tmp_path: Path) -> None:
    """Integration test: Real model detection followed by programmatic capture on sample asset."""
    sample_img_path = Path(__file__).parent / "assets" / "sample_face.jpg"
    if not sample_img_path.is_file():
        pytest.skip("Test asset 'tests/assets/sample_face.jpg' not found.")

    cache_dir = get_cache_dir()
    if not list(cache_dir.glob("*.onnx")):
        pytest.skip("No cached model available for offline test.")

    result = detect(str(sample_img_path), confidence_threshold=0.6)
    assert result.has_faces is True

    captured = capture(result, save_dir=tmp_path, filename="real_face_crop.jpg")
    assert isinstance(captured, FaceCapture)
    assert captured.width > 50
    assert captured.height > 50
    assert captured.path is not None
    assert captured.path.is_file()
    assert captured.path.stat().st_size > 0
