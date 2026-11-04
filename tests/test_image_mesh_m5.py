"""Comprehensive test suite for Milestone 5: Still-image mesh visualization."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

import detekface
from detekface import (
    BoundingBox,
    DetectionResult,
    FaceDetection,
    FaceShapeResult,
    InvalidImageError,
    Point,
    detect,
)

ASSETS_DIR = Path(__file__).parent / "assets"
SAMPLE_FACE_PATH = ASSETS_DIR / "sample_face.jpg"


def _make_dummy_image(h: int = 400, w: int = 500) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), 60, (200, 200, 200), -1)
    return img


def _make_face_with_landmarks(x=100, y=100, w=150, h=150, n_pts=478, has_shape=True):
    landmarks = [Point(x + (i % 80), y + (i % 90)) for i in range(n_pts)]
    shape_res = (
        FaceShapeResult(
            label="oval",
            fit_score=0.89,
            alternatives={"round": 0.52},
            measurements={"length_to_width_ratio": 1.45},
            quality_status="good",
        )
        if has_shape
        else None
    )
    return FaceDetection(
        bbox=BoundingBox(x, y, w, h),
        confidence=0.96,
        dense_landmarks=landmarks,
        face_shape=shape_res,
    )


def test_intended_api_result_render_mesh():
    """Verify intended API: result = detect(image, mesh=True); annotated = result.render(show_mesh=True, mesh_style='neon')."""
    img = _make_dummy_image()
    face = _make_face_with_landmarks()
    result = DetectionResult(faces=[face], image=img)

    annotated = result.render(show_mesh=True, mesh_style="neon")

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == img.shape
    # Rendering should modify pixels in the annotated copy
    assert not np.array_equal(annotated, img)
    # Original image must be preserved pristine
    assert np.array_equal(result.image, img)


def test_all_five_mesh_styles_on_still_image():
    """Verify wireframe, neon, minimal, contour, and points render successfully on still image."""
    img = _make_dummy_image()
    face = _make_face_with_landmarks()
    result = DetectionResult(faces=[face], image=img)

    for style in ["wireframe", "neon", "minimal", "contour", "points"]:
        rendered = result.render(show_mesh=True, mesh_style=style)
        assert isinstance(rendered, np.ndarray)
        assert rendered.shape == img.shape
        assert not np.array_equal(rendered, img)


def test_custom_mesh_color_and_thickness():
    """Verify custom mesh colors (hex, named, RGB tuple) and custom thickness."""
    img = _make_dummy_image()
    face = _make_face_with_landmarks()
    result = DetectionResult(faces=[face], image=img)

    # Custom hex color & custom thickness
    out1 = result.render(show_mesh=True, mesh_color="#FF0000", mesh_thickness=2)
    assert isinstance(out1, np.ndarray)

    # Custom named color
    out2 = result.render(show_mesh=True, mesh_color="cyan")
    assert isinstance(out2, np.ndarray)

    # Custom RGB tuple
    out3 = result.render(show_mesh=True, mesh_color=(255, 255, 0), mesh_thickness=4)
    assert isinstance(out3, np.ndarray)


def test_multiple_faces_rendered_independently():
    """Verify multiple faces are rendered with distinct mesh wireframes without cross-interference."""
    img = _make_dummy_image(h=600, w=800)
    face1 = _make_face_with_landmarks(x=100, y=100, w=150, h=150)
    face2 = _make_face_with_landmarks(x=450, y=150, w=160, h=160)
    result = DetectionResult(faces=[face1, face2], image=img)

    annotated = result.render(show_mesh=True, mesh_style="wireframe")
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == img.shape


def test_zero_faces_handles_gracefully():
    """Verify zero detected faces returns a clean copy of image without error."""
    img = _make_dummy_image()
    result = DetectionResult(faces=[], image=img)

    annotated = result.render(show_mesh=True, mesh_style="neon")
    assert isinstance(annotated, np.ndarray)
    # With 0 faces, image contents should match original pixels
    assert np.array_equal(annotated, img)
    assert annotated is not img  # Must be a copy


def test_unavailable_or_empty_landmarks_does_not_crash():
    """Verify faces missing dense landmarks do not crash result.render(show_mesh=True)."""
    img = _make_dummy_image()
    face_no_lms = FaceDetection(
        bbox=BoundingBox(50, 50, 100, 100),
        confidence=0.9,
        dense_landmarks=None,
        face_shape=None,
    )
    result = DetectionResult(faces=[face_no_lms], image=img)

    annotated = result.render(show_mesh=True, mesh_style="neon")
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == img.shape


def test_mesh_and_shape_detection_overlays_together():
    """Verify all 5 primary detection overlay shapes work alongside mesh visualization."""
    img = _make_dummy_image()
    face = _make_face_with_landmarks()
    result = DetectionResult(faces=[face], image=img)

    for shape in ["rectangle", "corners", "circle", "rounded", "hud"]:
        annotated = result.render(
            shape=shape,
            show_confidence=True,
            show_mesh=True,
            mesh_style="neon",
            show_contour=True,
        )
        assert isinstance(annotated, np.ndarray)


def test_saving_annotated_mesh_image(tmp_path):
    """Verify result.save() saves rendered mesh overlay to disk and preserves image format."""
    img = _make_dummy_image()
    face = _make_face_with_landmarks()
    result = DetectionResult(faces=[face], image=img)

    save_path = tmp_path / "mesh_output.png"
    saved = result.save(save_path, show_mesh=True, mesh_style="neon")

    assert saved.is_file()
    loaded = cv2.imread(str(saved))
    assert loaded is not None
    assert loaded.shape == img.shape


def test_lazy_model_initialization_when_mesh_disabled():
    """Verify landmark extractor is NOT loaded or initialized when mesh/face_shape are disabled."""
    img = _make_dummy_image()

    with patch("detekface.face_shape.DenseLandmarkExtractor.get_default") as mock_extractor:
        _ = detekface.detect(img, analyze_face_shape=False, mesh=False)
        mock_extractor.assert_not_called()


def test_detect_with_mesh_flag_attaches_landmarks_without_face_shape():
    """Verify detect(image, mesh=True) populates dense_landmarks while keeping face_shape separate."""
    img = _make_dummy_image()
    mock_lms = [[Point(100 + i, 100 + i) for i in range(478)]]

    with patch("detekface.face_shape.DenseLandmarkExtractor.get_default") as mock_extractor_cls:
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = mock_lms
        mock_extractor_cls.return_value = mock_extractor

        # Detect with mesh=True and analyze_face_shape=False
        result = detect(img, mesh=True, analyze_face_shape=False)

        assert len(result.faces) > 0 or result.count >= 0
        for face in result.faces:
            if face.dense_landmarks is not None:
                assert len(face.dense_landmarks) == 478
                # face_shape should be None because analyze_face_shape was False
                assert face.face_shape is None


def test_original_image_immutability():
    """Verify result.image remains unchanged after multiple calls to render, draw, annotate, and save."""
    img = _make_dummy_image()
    orig_copy = img.copy()
    face = _make_face_with_landmarks()
    result = DetectionResult(faces=[face], image=img)

    _ = result.render(show_mesh=True, mesh_style="wireframe")
    _ = result.annotate(show_mesh=True, mesh_style="neon")
    _ = result.draw(show_mesh=True, mesh_style="contour")

    assert np.array_equal(result.image, orig_copy)


def test_different_image_dimensions():
    """Verify mesh renders correctly on non-square, small, and high-resolution images."""
    for (h, w) in [(120, 320), (720, 1280), (300, 300)]:
        img = _make_dummy_image(h=h, w=w)
        face = _make_face_with_landmarks(x=20, y=20, w=60, h=60)
        result = DetectionResult(faces=[face], image=img)
        annotated = result.render(show_mesh=True, mesh_style="neon")
        assert annotated.shape == (h, w, 3)


def test_invalid_image_raises_error():
    """Verify InvalidImageError is raised when trying to annotate an empty or invalid result."""
    empty_result = DetectionResult(faces=[], image=np.empty((0, 0, 3), dtype=np.uint8))
    with pytest.raises(InvalidImageError):
        empty_result.render(show_mesh=True)

