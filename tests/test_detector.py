"""Tests for detekface still-image face detection engine, data models, and input validation."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from detekface import (
    BoundingBox,
    ConfigurationError,
    DetectionResult,
    FaceDetection,
    FaceDetector,
    FaceLandmarks,
    ImageNotFoundError,
    InvalidImageError,
    Point,
    detect,
    get_cache_dir,
    resolve_model,
)


@pytest.fixture
def mock_detector_engine():
    """Create a mock FaceDetector with controlled underlying OpenCV engine."""
    with patch("detekface.detector.resolve_model") as mock_resolve:
        mock_resolve.return_value = Path("dummy_model.onnx")
        with patch("cv2.FaceDetectorYN.create") as mock_create:
            mock_cv_detector = MagicMock()
            mock_create.return_value = mock_cv_detector
            detector = FaceDetector(model="dummy_model.onnx")
            yield detector, mock_cv_detector


def test_bounding_box_properties_and_methods() -> None:
    """Test BoundingBox coordinates, area, and representation methods."""
    bbox = BoundingBox(x=10, y=20, width=50, height=80)
    assert bbox.x1 == 10
    assert bbox.y1 == 20
    assert bbox.x2 == 60
    assert bbox.y2 == 100
    assert bbox.area == 4000
    assert bbox.as_tuple() == (10, 20, 50, 80)
    assert bbox.as_xyxy() == (10, 20, 60, 100)


def test_face_landmarks_methods() -> None:
    """Test FaceLandmarks list and tuple transformations."""
    landmarks = FaceLandmarks(
        right_eye=Point(10, 20),
        left_eye=Point(30, 20),
        nose_tip=Point(20, 30),
        right_mouth=Point(15, 40),
        left_mouth=Point(25, 40),
    )
    assert len(landmarks.as_list()) == 5
    tuples = landmarks.as_tuples()
    assert tuples == [(10, 20), (30, 20), (20, 30), (15, 40), (25, 40)]


def test_detection_result_container() -> None:
    """Test DetectionResult length, indexing, iteration, and representation."""
    bbox = BoundingBox(10, 10, 20, 20)
    detection = FaceDetection(bbox=bbox, confidence=0.9)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = DetectionResult(faces=[detection], image=img)

    assert result.count == 1
    assert result.has_faces is True
    assert len(result) == 1
    assert result[0] == detection
    assert [f for f in result] == [detection]
    assert "1 face(s) found in 100x100" in repr(result)


def test_empty_detection_output(mock_detector_engine) -> None:
    """Verify that when no faces are detected, an empty DetectionResult is returned."""
    detector, mock_cv = mock_detector_engine
    mock_cv.detect.return_value = (1, None)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detector.detect(img)

    assert result.count == 0
    assert result.has_faces is False
    assert len(result.faces) == 0


def test_blank_image_detection_real_model_if_cached() -> None:
    """Verify that running a real model on an all-black blank image yields 0 detected faces without errors."""
    cache_dir = get_cache_dir()
    if not list(cache_dir.glob("*.onnx")):
        pytest.skip("No cached model available for offline test.")

    blank_img = np.zeros((320, 320, 3), dtype=np.uint8)
    result = detect(blank_img)
    assert result.count == 0
    assert result.has_faces is False
    assert len(result) == 0


def test_single_face_detection_parsing(mock_detector_engine) -> None:
    """Verify exact parsing of coordinates, confidence, and 5 landmarks for a single detected face."""
    detector, mock_cv = mock_detector_engine
    # 15 elements: x, y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rmc_x, rmc_y, lmc_x, lmc_y, score
    raw_face = np.array([[
        15.4, 25.6, 120.1, 140.8,
        35.2, 50.1, 85.3, 51.4,
        60.0, 85.2,
        42.1, 115.3, 78.4, 116.2,
        0.945,
    ]], dtype=np.float32)

    mock_cv.detect.return_value = (1, raw_face)

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    result = detector.detect(img)

    assert result.count == 1
    face = result.faces[0]

    # Bounding box integer coordinates
    assert face.bbox.x == 15
    assert face.bbox.y == 26
    assert face.bbox.width == 120
    assert face.bbox.height == 141
    assert face.bbox.x2 == 15 + 120
    assert face.bbox.y2 == 26 + 141

    # Confidence
    assert pytest.approx(face.confidence, rel=1e-3) == 0.945

    # Landmarks
    assert face.landmarks is not None
    assert face.landmarks.right_eye.as_tuple() == (35, 50)
    assert face.landmarks.left_eye.as_tuple() == (85, 51)
    assert face.landmarks.nose_tip.as_tuple() == (60, 85)
    assert face.landmarks.right_mouth.as_tuple() == (42, 115)
    assert face.landmarks.left_mouth.as_tuple() == (78, 116)


def test_multiple_face_detection_parsing(mock_detector_engine) -> None:
    """Verify parsing multiple faces in a single image."""
    detector, mock_cv = mock_detector_engine
    raw_faces = np.array([
        [10, 10, 50, 50, 20, 20, 40, 20, 30, 30, 22, 40, 38, 40, 0.92],
        [80, 80, 40, 40, 85, 85, 95, 85, 90, 92, 86, 105, 94, 105, 0.88],
    ], dtype=np.float32)

    mock_cv.detect.return_value = (1, raw_faces)

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    result = detector.detect(img)

    assert result.count == 2
    assert result.faces[0].confidence == pytest.approx(0.92)
    assert result.faces[1].confidence == pytest.approx(0.88)


def test_missing_image_path_raises_not_found(mock_detector_engine) -> None:
    """Verify ImageNotFoundError when passing a non-existent filepath."""
    detector, _ = mock_detector_engine
    with pytest.raises(ImageNotFoundError) as exc_info:
        detector.detect("nonexistent_image_path_12345.jpg")
    assert "does not exist" in str(exc_info.value)
    assert exc_info.value.hint is not None


def test_directory_as_image_path_raises_invalid(tmp_path: Path, mock_detector_engine) -> None:
    """Verify InvalidImageError when passing a directory path instead of an image file."""
    detector, _ = mock_detector_engine
    with pytest.raises(InvalidImageError) as exc_info:
        detector.detect(tmp_path)
    assert "directory" in str(exc_info.value)


def test_unreadable_or_corrupt_image_file_raises_invalid(tmp_path: Path, mock_detector_engine) -> None:
    """Verify InvalidImageError when passing a corrupt/empty file as image."""
    detector, _ = mock_detector_engine
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"not a valid jpeg or png")

    with pytest.raises(InvalidImageError) as exc_info:
        detector.detect(corrupt_file)
    assert "Failed to decode" in str(exc_info.value)


def test_empty_numpy_array_raises_invalid(mock_detector_engine) -> None:
    """Verify InvalidImageError when passing an empty NumPy array."""
    detector, _ = mock_detector_engine
    empty_arr = np.array([])
    with pytest.raises(InvalidImageError) as exc_info:
        detector.detect(empty_arr)
    assert "is empty" in str(exc_info.value)


def test_unsupported_array_dimensions_raise_invalid(mock_detector_engine) -> None:
    """Verify InvalidImageError for 1D or 4D NumPy arrays."""
    detector, _ = mock_detector_engine
    arr_1d = np.zeros((100,), dtype=np.uint8)
    with pytest.raises(InvalidImageError) as exc_info:
        detector.detect(arr_1d)
    assert "Unsupported image array dimensions" in str(exc_info.value)

    arr_4d = np.zeros((1, 100, 100, 3), dtype=np.uint8)
    with pytest.raises(InvalidImageError):
        detector.detect(arr_4d)


def test_grayscale_2d_array_conversion(mock_detector_engine) -> None:
    """Verify that a 2D grayscale NumPy array is auto-converted to 3-channel BGR."""
    detector, mock_cv = mock_detector_engine
    mock_cv.detect.return_value = (1, None)

    gray = np.full((50, 50), 128, dtype=np.uint8)
    result = detector.detect(gray)
    assert result.image.shape == (50, 50, 3)


def test_bgra_4channel_array_conversion(mock_detector_engine) -> None:
    """Verify that a 4-channel BGRA NumPy array is auto-converted to 3-channel BGR."""
    detector, mock_cv = mock_detector_engine
    mock_cv.detect.return_value = (1, None)

    bgra = np.zeros((40, 60, 4), dtype=np.uint8)
    result = detector.detect(bgra)
    assert result.image.shape == (40, 60, 3)


def test_unsupported_input_type_raises_invalid(mock_detector_engine) -> None:
    """Verify InvalidImageError when passing invalid types (e.g. integer or list)."""
    detector, _ = mock_detector_engine
    with pytest.raises(InvalidImageError) as exc_info:
        detector.detect(12345)  # type: ignore
    assert "Unsupported image input type" in str(exc_info.value)


def test_threshold_validation_errors() -> None:
    """Verify ConfigurationError for invalid thresholds."""
    with patch("detekface.detector.resolve_model") as mock_resolve:
        mock_resolve.return_value = Path("dummy.onnx")

        # Confidence threshold out of bounds
        with pytest.raises(ConfigurationError):
            FaceDetector(confidence_threshold=-0.1)
        with pytest.raises(ConfigurationError):
            FaceDetector(confidence_threshold=1.5)
        with pytest.raises(ConfigurationError):
            FaceDetector(confidence_threshold="high")  # type: ignore

        # NMS threshold out of bounds
        with pytest.raises(ConfigurationError):
            FaceDetector(nms_threshold=-0.1)
        with pytest.raises(ConfigurationError):
            FaceDetector(nms_threshold=1.2)

        # Top-k non-positive
        with pytest.raises(ConfigurationError):
            FaceDetector(top_k=0)


# =========================================================================
# Integration test with real model & licensed sample asset
# =========================================================================

def test_real_model_detection_on_sample_asset() -> None:
    """Integration test: Real YuNet model face detection on the official OpenCV sample image.

    Requires local model file to be present in cache. If not yet cached, this test is skipped
    to ensure automated test suites remain strictly independent of network access.
    """
    sample_img_path = Path(__file__).parent / "assets" / "sample_face.jpg"
    if not sample_img_path.is_file():
        pytest.skip("Test asset 'tests/assets/sample_face.jpg' not found.")

    # Check if a model is present in the local cache
    cache_dir = get_cache_dir()
    cached_models = list(cache_dir.glob("*.onnx"))
    if not cached_models:
        pytest.skip(
            "No cached model found in local cache. "
            "Skipping real-model test to keep test run offline without downloading."
        )

    # Run real face detection
    result = detect(str(sample_img_path), confidence_threshold=0.6)

    assert result.has_faces is True
    assert result.count >= 1

    first_face = result.faces[0]
    assert first_face.confidence >= 0.8
    assert first_face.bbox.width > 50
    assert first_face.bbox.height > 50

    # Ensure bounding box is within image bounds
    img_h, img_w = result.image.shape[:2]
    assert 0 <= first_face.bbox.x <= img_w
    assert 0 <= first_face.bbox.y <= img_h

    # Ensure landmarks exist and are inside the face region
    assert first_face.landmarks is not None
    for pt in first_face.landmarks.as_list():
        assert 0 <= pt.x <= img_w
        assert 0 <= pt.y <= img_h
