"""Automated tests for Milestone 8: public image API.

Tests DetectionResult methods (count, faces, show, save, annotate, draw, capture),
verifies original image immutability, format validation, crop purity, and reopening saved files.
"""

from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pytest

import detekface
from detekface.exceptions import (
    ConfigurationError,
    DetekFaceError,
    ImageSaveError,
    InvalidImageError,
)
from detekface.types import (
    BoundingBox,
    DetectionResult,
    FaceCapture,
    FaceDetection,
    FaceLandmarks,
    Point,
)


@pytest.fixture
def synthetic_face_image() -> np.ndarray:
    """Create a 300x400 synthetic BGR image with a white background and dark face square."""
    img = np.full((300, 400, 3), 240, dtype=np.uint8)
    # Face region: 100x100 square at (150, 100)
    img[100:200, 150:250] = (80, 80, 80)
    return img


@pytest.fixture
def sample_detection() -> FaceDetection:
    """A realistic FaceDetection object."""
    bbox = BoundingBox(x=150, y=100, width=100, height=100)
    landmarks = FaceLandmarks(
        right_eye=Point(175, 130),
        left_eye=Point(225, 130),
        nose_tip=Point(200, 155),
        right_mouth=Point(180, 180),
        left_mouth=Point(220, 180),
    )
    return FaceDetection(bbox=bbox, confidence=0.92, landmarks=landmarks)


@pytest.fixture
def single_face_result(synthetic_face_image, sample_detection) -> DetectionResult:
    return DetectionResult(faces=[sample_detection], image=synthetic_face_image)


@pytest.fixture
def multi_face_result(synthetic_face_image, sample_detection) -> DetectionResult:
    face2 = FaceDetection(
        bbox=BoundingBox(x=20, y=20, width=50, height=50),
        confidence=0.85,
        landmarks=None,
    )
    return DetectionResult(faces=[sample_detection, face2], image=synthetic_face_image)


@pytest.fixture
def zero_face_result(synthetic_face_image) -> DetectionResult:
    return DetectionResult(faces=[], image=synthetic_face_image)


# -------------------------------------------------------------------------
# Check 1 & Check 3: Properties, Distinguishability, Immutability
# -------------------------------------------------------------------------


def test_result_count_and_faces_properties(zero_face_result, single_face_result, multi_face_result):
    """Test result.count accurately reflects faces and result.faces exposes them."""
    assert zero_face_result.count == 0
    assert len(zero_face_result) == 0
    assert not zero_face_result.has_faces
    assert zero_face_result.faces == []

    assert single_face_result.count == 1
    assert len(single_face_result) == 1
    assert single_face_result.has_faces
    assert len(single_face_result.faces) == 1
    assert single_face_result[0].bbox.width == 100

    assert multi_face_result.count == 2
    assert len(multi_face_result) == 2
    assert multi_face_result.has_faces
    assert len(multi_face_result.faces) == 2
    assert [f.confidence for f in multi_face_result] == [0.92, 0.85]


def test_original_image_remains_unmodified_and_distinguishable(single_face_result):
    """Verify original image is not modified by annotate, draw, or save operations."""
    original_copy = single_face_result.image.copy()

    # Access original_image property
    assert single_face_result.original_image is single_face_result.image

    # Generate annotations
    annotated = single_face_result.annotate(shape="hud", color="#00FF88", thickness=3)

    # Annotated array must be distinct from original image
    assert id(annotated) != id(single_face_result.image)
    assert not np.array_equal(annotated, single_face_result.image)

    # Original image array must remain 100% bitwise identical
    assert np.array_equal(single_face_result.image, original_copy)
    assert np.array_equal(single_face_result.original_image, original_copy)


def test_annotate_and_draw_styling_options(single_face_result):
    """Verify explicit shapes, colors, and confidence badges can be specified."""
    # Test all 5 shapes
    for shape in ("rectangle", "corners", "circle", "rounded", "hud"):
        out = single_face_result.annotate(shape=shape, color="cyan", thickness=2, show_confidence=True)
        assert isinstance(out, np.ndarray)
        assert out.shape == single_face_result.image.shape
        # Check that pixels were drawn
        assert not np.array_equal(out, single_face_result.image)

    # draw() alias works identically
    drawn = single_face_result.draw(shape="rounded", color="red", thickness=2)
    assert isinstance(drawn, np.ndarray)

    # annotated property works with default options
    default_annotated = single_face_result.annotated
    assert isinstance(default_annotated, np.ndarray)

    # Invalid styling raises ConfigurationError
    with pytest.raises(ConfigurationError):
        single_face_result.annotate(shape="invalid_shape")
    with pytest.raises(ConfigurationError):
        single_face_result.annotate(color="invalid_color")
    with pytest.raises(ConfigurationError):
        single_face_result.annotate(thickness=0)


def test_annotate_zero_faces_returns_unmodified_copy(zero_face_result):
    """Zero faces returns an identical copy of the image without overlays."""
    annotated = zero_face_result.annotate()
    assert id(annotated) != id(zero_face_result.image)
    assert np.array_equal(annotated, zero_face_result.image)


# -------------------------------------------------------------------------
# Check 2: Save and Reopen Saved Files
# -------------------------------------------------------------------------


@pytest.mark.parametrize("extension", [".jpg", ".png", ".bmp", ".webp"])
def test_save_and_reopen_various_formats(single_face_result, tmp_path, extension):
    """Test saving annotated image in various formats, reopening, and verifying dimensions."""
    out_file = tmp_path / f"annotated_output{extension}"
    saved_path = single_face_result.save(out_file, shape="hud", color="#00FF88", thickness=2)

    assert saved_path == out_file.resolve()
    assert out_file.exists()
    assert out_file.stat().st_size > 0

    # Reopen saved file with OpenCV
    reopened = cv2.imread(str(saved_path))
    assert reopened is not None
    assert reopened.shape == (300, 400, 3)

    # Reopened image must differ from original raw image due to annotations
    assert not np.array_equal(reopened, single_face_result.image)


def test_save_zero_faces_writes_valid_image(zero_face_result, tmp_path):
    """Verify save() on zero-faces result writes a valid file with expected dimensions."""
    out_file = tmp_path / "zero_faces.jpg"
    saved_path = zero_face_result.save(out_file)

    assert saved_path.exists()
    reopened = cv2.imread(str(saved_path))
    assert reopened is not None
    assert reopened.shape == (300, 400, 3)


def test_save_creates_nested_parent_directories(single_face_result, tmp_path):
    """Verify save() creates parent directories if they do not exist."""
    nested_file = tmp_path / "a" / "b" / "c" / "annotated.jpg"
    assert not nested_file.parent.exists()

    saved_path = single_face_result.save(nested_file)
    assert saved_path.exists()
    assert nested_file.parent.exists()


# -------------------------------------------------------------------------
# Check 4: Invalid Output Paths and Formats
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "   ",
        None,
        12345,
    ],
)
def test_save_invalid_path_types_raise_error(single_face_result, bad_path):
    """Test empty, None, and non-string/Path arguments raise InvalidImageError."""
    with pytest.raises(InvalidImageError):
        single_face_result.save(bad_path)


@pytest.mark.parametrize(
    "bad_filename",
    [
        "output_no_extension",
        "output.txt",
        "output.pdf",
        "output.csv",
        "output.xyz",
        "output.mp4",
    ],
)
def test_save_unsupported_extensions_raise_error(single_face_result, tmp_path, bad_filename):
    """Test paths without extensions or with unsupported extensions raise InvalidImageError."""
    target = tmp_path / bad_filename
    with pytest.raises(InvalidImageError) as exc_info:
        single_face_result.save(target)
    assert "Unsupported" in str(exc_info.value) or "extension" in str(exc_info.value)


def test_save_parent_is_file_raises_image_save_error(single_face_result, tmp_path):
    """Test saving inside a path where parent is an existing file raises ImageSaveError."""
    dummy_file = tmp_path / "blocker.txt"
    dummy_file.write_text("I am a file")

    invalid_target = dummy_file / "output.jpg"
    with pytest.raises(ImageSaveError):
        single_face_result.save(invalid_target)


def test_save_opencv_failure_raises_image_save_error(single_face_result, tmp_path):
    """Test that a failure in cv2.imwrite raises an actionable ImageSaveError."""
    out_file = tmp_path / "failed.jpg"
    with patch("cv2.imwrite", return_value=False):
        with pytest.raises(ImageSaveError) as exc_info:
            single_face_result.save(out_file)
        assert "OpenCV failed to write" in str(exc_info.value)


# -------------------------------------------------------------------------
# Crop Purity Requirement: Crops from Original Image, Not Drawn Boxes
# -------------------------------------------------------------------------


def test_captured_crops_come_from_original_image_not_annotated(single_face_result, tmp_path):
    """Verify that face crops are extracted strictly from the unannotated original image."""
    # First, render multiple overlays and save annotated image
    _ = single_face_result.annotate(shape="hud", color="#00FF88", thickness=5)
    single_face_result.save(tmp_path / "annotated.jpg", shape="hud", color="#00FF88", thickness=5)

    # Perform capture via detekface.capture()
    capture1 = detekface.capture(single_face_result, save=False)

    # Perform capture via result.capture() convenience method
    capture2 = single_face_result.capture(save=False)

    # In single_face_result, the face bbox is at [100:200, 150:250] which is solid grey (80, 80, 80)
    # If the crop had overlays, it would contain bright neon green (136, 255, 0)
    assert np.all(capture1.image == (80, 80, 80))
    assert np.all(capture2.image == (80, 80, 80))

    # Compare directly to clean original image slice
    expected_clean_slice = single_face_result.image[100:200, 150:250]
    assert np.array_equal(capture1.image, expected_clean_slice)
    assert np.array_equal(capture2.image, expected_clean_slice)


def test_detection_result_capture_method_features(multi_face_result, tmp_path):
    """Verify result.capture() convenience method behaves identically to detekface.capture()."""
    # Default captures largest face (100x100 face)
    cap = multi_face_result.capture(save_dir=tmp_path, filename="largest.jpg")
    assert isinstance(cap, FaceCapture)
    assert cap.width == 100
    assert cap.height == 100
    assert cap.path.exists()

    # Capture specific face index
    cap_second = multi_face_result.capture(face_index=1, save=False)
    assert cap_second.width == 50
    assert cap_second.height == 50


# -------------------------------------------------------------------------
# Window GUI show() Method Tests
# -------------------------------------------------------------------------


def test_show_window_keypress_closure(single_face_result):
    """Test result.show() displays window and returns keycode on user keypress."""
    with (
        patch("cv2.namedWindow") as mock_named_win,
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.waitKey", return_value=ord("q")),
        patch("cv2.destroyWindow") as mock_destroy,
    ):
        key = single_face_result.show(title="My Face Window", shape="corners", color="cyan")
        assert key == ord("q")
        mock_named_win.assert_called_once_with("My Face Window", cv2.WINDOW_AUTOSIZE)
        mock_imshow.assert_called_once()
        mock_destroy.assert_called_once_with("My Face Window")


def test_show_window_close_button_x(single_face_result):
    """Test result.show() exits cleanly when user clicks the window close button."""
    with (
        patch("cv2.namedWindow"),
        patch("cv2.imshow"),
        patch("cv2.waitKey", side_effect=[-1, -1]),
        patch("cv2.getWindowProperty", return_value=0),  # Window closed / invisible
        patch("cv2.destroyWindow") as mock_destroy,
    ):
        key = single_face_result.show(title="Close Me Window")
        assert key == -1
        mock_destroy.assert_called_once_with("Close Me Window")


def test_show_window_with_delay(single_face_result):
    """Test result.show() with delay > 0 waits for specified delay and closes."""
    with (
        patch("cv2.namedWindow"),
        patch("cv2.imshow"),
        patch("cv2.waitKey", return_value=-1) as mock_wait_key,
        patch("cv2.destroyWindow") as mock_destroy,
    ):
        key = single_face_result.show(delay=100)
        assert key == -1
        mock_wait_key.assert_called_once_with(100)
        mock_destroy.assert_called_once()


def test_show_window_gui_error_raises_detekface_error(single_face_result):
    """Test that cv2.error in GUI display raises a helpful DetekFaceError."""
    with (
        patch("cv2.namedWindow"),
        patch("cv2.imshow", side_effect=cv2.error("Display server unavailable")),
        patch("cv2.destroyWindow"),
    ):
        with pytest.raises(DetekFaceError) as exc_info:
            single_face_result.show()
        assert "Failed to display GUI window" in str(exc_info.value)


# -------------------------------------------------------------------------
# Integration: Real Model Image Detection, Save, and Reopen
# -------------------------------------------------------------------------


@pytest.mark.network
def test_real_model_detection_save_and_reopen_cycle(tmp_path):
    """Integration check: detect face on sample asset, annotate, save, and reopen."""
    asset_path = Path(__file__).parent / "assets" / "sample_face.jpg"
    if not asset_path.exists():
        pytest.skip(f"Asset file '{asset_path}' not found.")

    # 1. detect()
    result = detekface.detect(asset_path, confidence_threshold=0.6)
    assert result.count >= 1
    assert len(result.faces) == result.count

    # 2. save() with HUD overlay and confidence
    annotated_file = tmp_path / "real_annotated.jpg"
    saved = result.save(annotated_file, shape="hud", color="#00FF88", show_confidence=True)
    assert saved.exists()

    # 3. Reopen and verify
    reopened = cv2.imread(str(saved))
    assert reopened is not None
    assert reopened.shape == result.image.shape

    # 4. capture() from result remains pristine
    clean_crop = result.capture(save=False)
    assert clean_crop.width > 0
    assert clean_crop.height > 0
