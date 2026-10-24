"""Unit tests for customizable face overlay rendering, color parsing, and styling."""

from pathlib import Path
import numpy as np
import pytest

from detekface import (
    BoundingBox,
    ConfigurationError,
    DetectionResult,
    FaceDetection,
    capture,
    draw_face_overlay,
    draw_faces,
    parse_color_to_bgr,
)


def _create_sample_face(
    x: int = 20,
    y: int = 20,
    width: int = 40,
    height: int = 50,
    confidence: float = 0.95,
) -> FaceDetection:
    """Helper to create a sample FaceDetection."""
    bbox = BoundingBox(x=x, y=y, width=width, height=height)
    return FaceDetection(bbox=bbox, confidence=confidence)


def test_color_hex_to_bgr_conversion() -> None:
    """Verify exact RGB hex string conversion to OpenCV BGR."""
    # #00FF88 -> R=00, G=255, B=136 -> BGR=(136, 255, 0)
    assert parse_color_to_bgr("#00FF88") == (136, 255, 0)
    assert parse_color_to_bgr("00FF88") == (136, 255, 0)

    # #FF0000 -> Red -> BGR=(0, 0, 255)
    assert parse_color_to_bgr("#FF0000") == (0, 0, 255)

    # #0000FF -> Blue -> BGR=(255, 0, 0)
    assert parse_color_to_bgr("#0000FF") == (255, 0, 0)

    # 3-digit shorthand #F00 -> #FF0000 -> BGR=(0, 0, 255)
    assert parse_color_to_bgr("#F00") == (0, 0, 255)


def test_color_named_conversion() -> None:
    """Verify documented color names map to correct OpenCV BGR tuples."""
    assert parse_color_to_bgr("green") == (0, 255, 0)
    assert parse_color_to_bgr("red") == (0, 0, 255)
    assert parse_color_to_bgr("blue") == (255, 0, 0)
    assert parse_color_to_bgr("cyan") == (255, 255, 0)
    assert parse_color_to_bgr("yellow") == (0, 255, 255)
    assert parse_color_to_bgr("white") == (255, 255, 255)
    assert parse_color_to_bgr("black") == (0, 0, 0)


def test_color_rgb_tuple_conversion() -> None:
    """Verify that an RGB tuple is converted to BGR."""
    assert parse_color_to_bgr((10, 20, 30)) == (30, 20, 10)


def test_invalid_color_raises_configuration_error() -> None:
    """Verify that malformed hex strings or unknown color names raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        parse_color_to_bgr("#GG0011")  # invalid hex digits
    with pytest.raises(ConfigurationError):
        parse_color_to_bgr("#12345")  # invalid length
    with pytest.raises(ConfigurationError):
        parse_color_to_bgr("rainbow_color")  # unknown name
    with pytest.raises(ConfigurationError):
        parse_color_to_bgr(12345)  # invalid type


def test_invalid_shape_raises_configuration_error() -> None:
    """Verify that unsupported shape names raise ConfigurationError."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    face = _create_sample_face()

    with pytest.raises(ConfigurationError) as exc_info:
        draw_face_overlay(img, face, shape="hexagon")
    assert "Unsupported overlay shape" in str(exc_info.value)
    assert "hexagon" in str(exc_info.value)


def test_invalid_thickness_raises_configuration_error() -> None:
    """Verify that thickness < 1 or non-integer thickness raises ConfigurationError."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    face = _create_sample_face()

    with pytest.raises(ConfigurationError):
        draw_face_overlay(img, face, thickness=0)
    with pytest.raises(ConfigurationError):
        draw_face_overlay(img, face, thickness=-2)
    with pytest.raises(ConfigurationError):
        draw_face_overlay(img, face, thickness="thick")  # type: ignore


@pytest.mark.parametrize("shape", ["rectangle", "corners", "circle", "rounded", "hud"])
def test_all_five_rendering_styles(shape: str) -> None:
    """Verify that all 5 supported shapes render non-black pixels onto an image with requested color."""
    base_img = np.zeros((120, 120, 3), dtype=np.uint8)
    face = _create_sample_face(x=20, y=20, width=60, height=60)

    # Use neon_green (#00FF88) -> BGR = (136, 255, 0)
    rendered = draw_face_overlay(base_img, face, shape=shape, color="#00FF88", thickness=2, copy=True)

    # Verify pixels were drawn
    assert rendered.shape == base_img.shape
    assert rendered.max() > 0

    # Verify the target color is present in the rendered image
    has_target_color = np.any(np.all(rendered == np.array([136, 255, 0]), axis=-1))
    assert has_target_color, f"Target BGR color (136, 255, 0) was not found in rendered {shape} overlay"


def test_draw_does_not_modify_original_image() -> None:
    """Verify that draw_face_overlay and draw_faces with copy=True preserve the original image unmodified."""
    orig = np.zeros((80, 80, 3), dtype=np.uint8)
    face = _create_sample_face(x=10, y=10, width=30, height=30)

    rendered = draw_face_overlay(orig, face, shape="corners", color="#FF0000", copy=True)

    # Rendered copy has painted pixels
    assert rendered.max() > 0
    # Original remains pure untouched black zeros
    assert orig.max() == 0


def test_draw_multiple_faces_independently() -> None:
    """Verify that draw_faces renders multiple faces onto the image."""
    base_img = np.zeros((200, 200, 3), dtype=np.uint8)
    face1 = _create_sample_face(x=10, y=10, width=30, height=30)
    face2 = _create_sample_face(x=100, y=100, width=40, height=40)

    rendered = draw_faces(base_img, [face1, face2], shape="rounded", color="cyan", thickness=2)

    # Face 1 region has drawn pixels
    assert rendered[10:40, 10:40].max() > 0
    # Face 2 region has drawn pixels
    assert rendered[100:140, 100:140].max() > 0


def test_out_of_bounds_bounding_box_handling() -> None:
    """Verify safe clipping for partially or completely out-of-bounds bounding boxes."""
    base_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Partially out-of-bounds box: clamped gracefully without error
    partial_face = _create_sample_face(x=-20, y=-20, width=50, height=50)
    rendered = draw_face_overlay(base_img, partial_face, shape="rectangle", color="yellow")
    assert rendered.max() > 0

    # Completely out-of-bounds box: skipped gracefully without modifying pixels
    outside_face = _create_sample_face(x=200, y=200, width=50, height=50)
    rendered_outside = draw_face_overlay(base_img, outside_face, shape="rectangle", color="yellow")
    assert rendered_outside.max() == 0


def test_capture_crop_is_not_polluted_by_drawn_overlays(tmp_path: Path) -> None:
    """Verify that captured face crops remain clean and unpolluted by visual overlays."""
    # Create clean image
    clean_img = np.full((100, 100, 3), 100, dtype=np.uint8)
    face = _create_sample_face(x=20, y=20, width=40, height=40)
    result = DetectionResult(faces=[face], image=clean_img)

    # Perform capture from the DetectionResult
    captured = capture(result, save_dir=tmp_path, filename="clean_crop.jpg")

    # Now draw on a separate display copy
    display_copy = draw_faces(clean_img, result.faces, shape="hud", color="#00FF88", thickness=3)

    # The display copy has painted overlay lines
    assert np.any(display_copy != clean_img)

    # The captured crop is completely identical to the clean source image area (no green HUD lines)
    expected_crop = clean_img[20:60, 20:60]
    np.testing.assert_array_equal(captured.image, expected_crop)
