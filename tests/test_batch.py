"""Tests for batch image face detection (detect_many, BatchItemResult, BatchDetectionResult)."""

from pathlib import Path
from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest

import detekface
from detekface import (
    BatchDetectionResult,
    BoundingBox,
    DetectionResult,
    FaceDetection,
    FaceDetector,
    ImageNotFoundError,
    InvalidImageError,
    detect_many,
)


@pytest.fixture
def sample_image_files(tmp_path: Path):
    """Create a temporary directory with valid and invalid sample images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Image 1 (100x100 BGR)
    img1_path = img_dir / "one.jpg"
    cv2.imwrite(str(img1_path), np.full((100, 100, 3), 120, dtype=np.uint8))

    # Image 2 (80x80 BGR)
    img2_path = img_dir / "two.png"
    cv2.imwrite(str(img2_path), np.full((80, 80, 3), 200, dtype=np.uint8))

    # Image 3 (corrupted / empty file)
    corrupt_path = img_dir / "corrupt.jpg"
    corrupt_path.write_bytes(b"not a valid image")

    # Subdirectory with an image
    sub_dir = img_dir / "nested"
    sub_dir.mkdir()
    sub_path = sub_dir / "nested_img.jpg"
    cv2.imwrite(str(sub_path), np.full((50, 50, 3), 80, dtype=np.uint8))

    return {
        "dir": img_dir,
        "one": img1_path,
        "two": img2_path,
        "corrupt": corrupt_path,
        "nested": sub_path,
    }


def test_empty_batch():
    """Verify detect_many with an empty input list returns an empty BatchDetectionResult."""
    results = detect_many([])
    assert isinstance(results, BatchDetectionResult)
    assert len(results) == 0
    assert results.successes == []
    assert results.failures == []
    assert results.total_faces == 0
    assert results.has_failures is False
    assert results.as_dict() == {}


def test_result_ordering_and_mapping(sample_image_files):
    """Verify detect_many preserves input ordering and associates results with inputs."""
    paths = [
        sample_image_files["one"],
        sample_image_files["two"],
    ]

    mock_detector = MagicMock()
    face1 = FaceDetection(bbox=BoundingBox(10, 10, 30, 30), confidence=0.91)
    face2 = FaceDetection(bbox=BoundingBox(5, 5, 20, 20), confidence=0.85)

    mock_detector.detect.side_effect = [
        DetectionResult(faces=[face1], image=np.zeros((10, 10, 3), dtype=np.uint8)),
        DetectionResult(faces=[face2], image=np.zeros((10, 10, 3), dtype=np.uint8)),
    ]

    results = detect_many(paths, detector=mock_detector)

    assert len(results) == 2
    assert results[0].source == sample_image_files["one"]
    assert results[0].is_success is True
    assert results[0].count == 1
    assert results[0].faces[0].confidence == pytest.approx(0.91)

    assert results[1].source == sample_image_files["two"]
    assert results[1].is_success is True
    assert results[1].count == 1
    assert results[1].faces[0].confidence == pytest.approx(0.85)

    assert results.total_faces == 2
    assert results.has_failures is False
    mapping = results.as_dict()
    assert mapping[sample_image_files["one"]] is results[0].result
    assert mapping[sample_image_files["two"]] is results[1].result


def test_mixed_valid_and_invalid_images_collect_errors(sample_image_files):
    """Verify collect-errors mode (default) reports errors without losing inputs."""
    missing_path = sample_image_files["dir"] / "does_not_exist.jpg"
    paths = [
        sample_image_files["one"],
        missing_path,
        sample_image_files["corrupt"],
    ]

    mock_detector = MagicMock()
    mock_detector.detect.side_effect = [
        DetectionResult(faces=[], image=np.zeros((10, 10, 3), dtype=np.uint8)),
        ImageNotFoundError(message="Not found", hint="Check path"),
        InvalidImageError(message="Corrupt", hint="Check file"),
    ]

    results = detect_many(paths, detector=mock_detector, fail_fast=False)

    assert len(results) == 3
    assert results.has_failures is True
    assert len(results.successes) == 1
    assert len(results.failures) == 2

    # Item 0: Success
    assert results[0].is_success is True
    assert results[0].error is None

    # Item 1: Missing file
    assert results[1].source == missing_path
    assert results[1].is_failure is True
    assert isinstance(results[1].error, ImageNotFoundError)
    assert results[1].result is None

    # Item 2: Corrupt file
    assert results[2].source == sample_image_files["corrupt"]
    assert results[2].is_failure is True
    assert isinstance(results[2].error, InvalidImageError)


def test_fail_fast_mode_raises_on_first_error(sample_image_files):
    """Verify fail_fast=True immediately raises on encountering an error."""
    missing_path = sample_image_files["dir"] / "missing.jpg"
    paths = [
        sample_image_files["one"],
        missing_path,
        sample_image_files["two"],
    ]

    mock_detector = MagicMock()
    mock_detector.detect.side_effect = [
        DetectionResult(faces=[], image=np.zeros((10, 10, 3), dtype=np.uint8)),
        ImageNotFoundError(message="File not found", hint="Check path"),
    ]

    with pytest.raises(ImageNotFoundError):
        detect_many(paths, detector=mock_detector, fail_fast=True)


def test_multiple_faces_aggregation():
    """Verify multiple faces per image are correctly reported and aggregated."""
    f1 = FaceDetection(bbox=BoundingBox(10, 10, 20, 20), confidence=0.9)
    f2 = FaceDetection(bbox=BoundingBox(40, 40, 30, 30), confidence=0.8)
    f3 = FaceDetection(bbox=BoundingBox(80, 80, 25, 25), confidence=0.95)

    mock_detector = MagicMock()
    mock_detector.detect.side_effect = [
        DetectionResult(faces=[f1, f2], image=np.zeros((10, 10, 3), dtype=np.uint8)),
        DetectionResult(faces=[f3], image=np.zeros((10, 10, 3), dtype=np.uint8)),
    ]

    results = detect_many(["img1.jpg", "img2.jpg"], detector=mock_detector)
    assert len(results) == 2
    assert results[0].count == 2
    assert results[1].count == 1
    assert results.total_faces == 3


def test_directory_discovery_flat_and_recursive(sample_image_files):
    """Verify directory input auto-discovers images flat and recursively."""
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=np.zeros((10, 10, 3), dtype=np.uint8))

    # Flat discovery (recursive=False) should find one.jpg, two.png, corrupt.jpg (3 files in root)
    flat_results = detect_many(sample_image_files["dir"], detector=mock_detector, recursive=False)
    flat_sources = [Path(r.source).name for r in flat_results]
    assert "one.jpg" in flat_sources
    assert "two.png" in flat_sources
    assert "nested_img.jpg" not in flat_sources

    # Recursive discovery (recursive=True) should include nested/nested_img.jpg
    rec_results = detect_many(sample_image_files["dir"], detector=mock_detector, recursive=True)
    rec_sources = [Path(r.source).name for r in rec_results]
    assert "nested_img.jpg" in rec_sources
    assert len(rec_results) == len(flat_results) + 1


def test_progress_callback_arities():
    """Verify progress callback works with 1, 2, or 3 arguments."""
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=np.zeros((10, 10, 3), dtype=np.uint8))

    # 3-arg callback: (item, current, total)
    cb_3_args = []

    def callback_3(item, cur, tot):
        cb_3_args.append((item.source, cur, tot))

    detect_many(["a.jpg", "b.jpg"], detector=mock_detector, progress_callback=callback_3)
    assert cb_3_args == [("a.jpg", 1, 2), ("b.jpg", 2, 2)]

    # 2-arg callback: (current, total)
    cb_2_args = []

    def callback_2(cur, tot):
        cb_2_args.append((cur, tot))

    detect_many(["a.jpg", "b.jpg"], detector=mock_detector, progress_callback=callback_2)
    assert cb_2_args == [(1, 2), (2, 2)]

    # 1-arg callback: (item)
    cb_1_arg = []

    def callback_1(item):
        cb_1_arg.append(item.source)

    detect_many(["a.jpg", "b.jpg"], detector=mock_detector, progress_callback=callback_1)
    assert cb_1_arg == ["a.jpg", "b.jpg"]


def test_keep_images_false_frees_memory():
    """Verify keep_images=False replaces large image array with empty array."""
    heavy_image = np.full((1000, 1000, 3), 255, dtype=np.uint8)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=heavy_image)

    results = detect_many(["img.jpg"], detector=mock_detector, keep_images=False)
    assert results[0].is_success is True
    # Image size should be 0 elements
    assert results[0].result.image.size == 0


def test_large_batch_of_small_images():
    """Verify sequential batch detection runs smoothly on a large batch (50 images)."""
    batch_size = 50
    dummy_images = [np.full((30, 30, 3), i, dtype=np.uint8) for i in range(batch_size)]

    mock_detector = MagicMock()
    mock_detector.detect.side_effect = lambda img: DetectionResult(
        faces=[FaceDetection(bbox=BoundingBox(0, 0, 10, 10), confidence=0.8)],
        image=img,
    )

    results = detect_many(dummy_images, detector=mock_detector)
    assert len(results) == batch_size
    assert results.total_faces == batch_size
    assert results.has_failures is False


def test_face_detector_detect_many_method():
    """Verify FaceDetector instance has .detect_many() matching standalone detect_many."""
    mock_detector = MagicMock(spec=FaceDetector)
    # Give it the actual method
    mock_detector.detect_many = FaceDetector.detect_many.__get__(mock_detector, FaceDetector)
    mock_detector.detect.return_value = DetectionResult(faces=[], image=np.zeros((10, 10, 3), dtype=np.uint8))

    results = mock_detector.detect_many(["test.jpg"])
    assert len(results) == 1
    assert results[0].source == "test.jpg"


def test_real_model_batch_detection_yunet():
    """Verify real YuNet model batch detection on sample assets."""
    asset_path = Path(__file__).parent / "assets" / "sample_face.jpg"
    if not asset_path.exists():
        pytest.skip("Asset sample_face.jpg not found.")

    results = detect_many([asset_path, asset_path], backend="yunet")
    assert len(results) == 2
    assert results[0].is_success is True
    assert results[1].is_success is True
    assert results.total_faces >= 2
    assert results.has_failures is False


@pytest.mark.optional_backend
def test_real_model_batch_detection_mediapipe():
    """Verify real MediaPipe model batch detection on sample assets."""
    asset_path = Path(__file__).parent / "assets" / "sample_face.jpg"
    if not asset_path.exists():
        pytest.skip("Asset sample_face.jpg not found.")

    try:
        results = detect_many([asset_path], backend="mediapipe")
        assert len(results) == 1
        assert results[0].is_success is True
        assert results.total_faces >= 1
    except detekface.DependencyNotFoundError:
        pytest.skip("MediaPipe not installed.")
