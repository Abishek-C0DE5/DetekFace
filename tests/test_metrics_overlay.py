"""Unit tests for optional webcam information overlays: confidence labels, FPS calculations, and face count."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from detekface import (
    BoundingBox,
    ConfigurationError,
    DetectionResult,
    FaceCapture,
    FaceDetection,
    FPSCounter,
    camera,
    capture,
    draw_face_overlay,
    draw_faces,
    draw_metrics_overlay,
    format_confidence,
    format_face_count,
    format_fps,
)


def _create_sample_face(x: int = 25, y: int = 35, width: int = 40, height: int = 50, confidence: float = 0.945) -> FaceDetection:
    """Helper to create a sample FaceDetection."""
    bbox = BoundingBox(x=x, y=y, width=width, height=height)
    return FaceDetection(bbox=bbox, confidence=confidence)


def test_face_count_formatting() -> None:
    """Verify exact face count formatting for zero, singular, and plural counts."""
    assert format_face_count(0) == "0 Faces"
    assert format_face_count(1) == "1 Face"
    assert format_face_count(2) == "2 Faces"
    assert format_face_count(10) == "10 Faces"


def test_confidence_formatting() -> None:
    """Verify confidence percentage string formatting."""
    assert format_confidence(0.945) == "94.5%"
    assert format_confidence(1.0) == "100.0%"
    assert format_confidence(0.5) == "50.0%"
    assert format_confidence(0.0) == "0.0%"


def test_fps_formatting() -> None:
    """Verify FPS throughput string formatting."""
    assert format_fps(29.97) == "30.0 FPS"
    assert format_fps(60.12) == "60.1 FPS"
    assert format_fps(0.0) == "0.0 FPS"


def test_fps_counter_calculation_and_smoothing() -> None:
    """Verify FPSCounter sliding window timing calculation and reset."""
    counter = FPSCounter(window_size=10)

    # With fewer than 2 timestamps, FPS must be 0.0
    assert counter.fps == 0.0
    counter.tick(current_time=1.0)
    assert counter.fps == 0.0

    # Simulate 10 frames at 30 FPS (interval = 1/30 = ~0.03333s)
    t = 1.0
    for _ in range(10):
        t += 1.0 / 30.0
        fps_val = counter.tick(current_time=t)

    assert pytest.approx(fps_val, rel=1e-2) == 30.0
    assert pytest.approx(counter.fps, rel=1e-2) == 30.0

    # Test reset
    counter.reset()
    assert counter.fps == 0.0


def test_metrics_overlay_hidden_when_disabled() -> None:
    """Verify that draw_metrics_overlay draws zero pixels when both fps and face_count are None."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_metrics_overlay(img, fps=None, face_count=None)
    # Entire image must remain untouched
    assert img.max() == 0


def test_metrics_overlay_renders_text_when_enabled() -> None:
    """Verify that draw_metrics_overlay draws status text when fps or face_count is supplied."""
    img_fps = np.zeros((100, 200, 3), dtype=np.uint8)
    draw_metrics_overlay(img_fps, fps=30.0, face_count=None)
    assert img_fps.max() > 0

    img_count = np.zeros((100, 200, 3), dtype=np.uint8)
    draw_metrics_overlay(img_count, fps=None, face_count=2)
    assert img_count.max() > 0

    img_both = np.zeros((100, 200, 3), dtype=np.uint8)
    draw_metrics_overlay(img_both, fps=28.5, face_count=1)
    assert img_both.max() > 0


@pytest.mark.parametrize("shape", ["rectangle", "corners", "circle", "rounded", "hud"])
def test_all_overlay_styles_with_confidence_flag(shape: str) -> None:
    """Verify every shape renders properly both with and without show_confidence enabled."""
    base_no_conf = np.zeros((150, 150, 3), dtype=np.uint8)
    base_with_conf = np.zeros((150, 150, 3), dtype=np.uint8)
    face = _create_sample_face(x=30, y=30, width=50, height=50, confidence=0.89)

    out_no_conf = draw_face_overlay(base_no_conf, face, shape=shape, show_confidence=False, copy=True)
    out_with_conf = draw_face_overlay(base_with_conf, face, shape=shape, show_confidence=True, copy=True)

    # Both must render without errors
    assert out_no_conf.max() > 0
    assert out_with_conf.max() > 0

    # Enabling confidence renders additional badge pixels
    assert np.count_nonzero(out_with_conf) > np.count_nonzero(out_no_conf)


def test_camera_display_flags_validation() -> None:
    """Verify ConfigurationError when passing non-boolean display flags to camera()."""
    with patch("detekface.camera.validate_render_parameters", return_value=("rectangle", 3)):
        with patch("detekface.camera.parse_color_to_bgr", return_value=(0, 255, 0)):
            with pytest.raises(ConfigurationError):
                camera(show_confidence="true")  # type: ignore

            with pytest.raises(ConfigurationError):
                camera(show_fps=1)  # type: ignore

            with pytest.raises(ConfigurationError):
                camera(show_count=None)  # type: ignore


def test_camera_information_overlays_integration(tmp_path: Path) -> None:
    """Verify camera() integrates show_confidence, show_fps, and show_count correctly."""
    dummy_frame = np.full((200, 200, 3), 50, dtype=np.uint8)
    face = _create_sample_face(x=20, y=20, width=50, height=50, confidence=0.92)
    mock_result = DetectionResult(faces=[face], image=dummy_frame)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = mock_result

    with (
        patch("cv2.VideoCapture") as mock_cap_cls,
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.waitKey", return_value=ord("q")),
        patch("cv2.destroyAllWindows"),
        patch("detekface.camera.draw_metrics_overlay") as mock_metrics_draw,
    ):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap_cls.return_value = mock_cap

        camera(
            camera_index=0,
            detector=mock_detector,
            show_confidence=True,
            show_fps=True,
            show_count=True,
            max_frames=1,
        )

        mock_metrics_draw.assert_called_once()
        args, kwargs = mock_metrics_draw.call_args
        # kwargs should receive fps and face_count
        assert kwargs["face_count"] == 1
        assert "fps" in kwargs


def test_metrics_do_not_alter_bounding_boxes_or_saved_crops(tmp_path: Path) -> None:
    """Verify that bounding box coordinates and saved face crops are completely unaltered by metric rendering."""
    clean_image = np.full((120, 120, 3), 70, dtype=np.uint8)
    face = _create_sample_face(x=20, y=30, width=40, height=50, confidence=0.91)
    result = DetectionResult(faces=[face], image=clean_image)

    orig_bbox = (face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height)

    # Perform capture
    captured = capture(result, save_dir=tmp_path, filename="unaltered_face.jpg")

    # Render all overlays including confidence, metrics, HUD
    display_image = clean_image.copy()
    draw_faces(display_image, result.faces, shape="hud", color="#00FF88", show_confidence=True)
    draw_metrics_overlay(display_image, fps=30.0, face_count=1)

    # 1. Bounding box coordinates remain untouched
    assert (face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height) == orig_bbox

    # 2. Captured face image array contains pure unmodified pixels from clean_image without any text or lines
    expected_crop = clean_image[30:80, 20:60]
    np.testing.assert_array_equal(captured.image, expected_crop)
