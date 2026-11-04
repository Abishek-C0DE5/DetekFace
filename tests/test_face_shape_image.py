"""Tests for Mode 1: Still-Image Face Shape Analysis."""

from pathlib import Path
import numpy as np

import detekface
from detekface.face_shape import (
    SUPPORTED_FACE_SHAPES,
    analyze_face_geometry,
)
from detekface.types import Point

ASSETS_DIR = Path(__file__).parent / "assets"
SAMPLE_FACE_PATH = ASSETS_DIR / "sample_face.jpg"


def test_still_image_face_shape_analysis_public_api():
    """Verify Mode 1 public API detekface.detect(..., analyze_face_shape=True)."""
    assert SAMPLE_FACE_PATH.is_file(), f"Missing test asset: {SAMPLE_FACE_PATH}"

    result = detekface.detect(SAMPLE_FACE_PATH, analyze_face_shape=True)

    assert result.has_faces is True
    assert result.count >= 1

    face = result.faces[0]
    assert face.face_shape is not None
    shape = face.face_shape

    # Verify structured fields
    assert isinstance(shape.label, str)
    assert shape.label in SUPPORTED_FACE_SHAPES
    assert isinstance(shape.fit_score, float)
    assert 0.0 <= shape.fit_score <= 1.0

    # Verify alternatives
    assert isinstance(shape.alternatives, dict)
    for alt_name, alt_score in shape.alternatives.items():
        assert alt_name in SUPPORTED_FACE_SHAPES
        assert 0.0 <= alt_score <= 1.0

    # Verify measurements dictionary
    assert isinstance(shape.measurements, dict)
    for expected_key in (
        "face_length",
        "cheekbone_width",
        "forehead_width",
        "jaw_width",
        "length_to_width_ratio",
        "forehead_to_jaw_ratio",
        "cheek_to_jaw_ratio",
        "jaw_angle",
    ):
        assert expected_key in shape.measurements
        assert shape.measurements[expected_key] > 0.0

    # Verify quality_status and warnings / reasons alias
    assert isinstance(shape.quality_status, str)
    assert isinstance(shape.warnings, list)
    assert shape.reasons == shape.warnings

    # Verify dense landmarks attached (478 points)
    assert face.dense_landmarks is not None
    assert len(face.dense_landmarks) >= 468
    for pt in face.dense_landmarks[:10]:
        assert isinstance(pt, Point)
        assert pt.x >= 0
        assert pt.y >= 0


def test_default_detection_leaves_face_shape_none():
    """Verify that default detect() does NOT run heavy face shape analysis."""
    result = detekface.detect(SAMPLE_FACE_PATH, analyze_face_shape=False)
    assert result.has_faces is True
    for face in result.faces:
        assert face.face_shape is None
        assert face.dense_landmarks is None


def test_annotate_with_face_shape_overlay():
    """Verify DetectionResult.annotate() draws face shape badge and contour."""
    result = detekface.detect(SAMPLE_FACE_PATH, analyze_face_shape=True)
    orig = result.image.copy()

    annotated = result.annotate(
        show_face_shape=True,
        show_shape_contour=True,
        show_shape_measurements=True,
    )

    assert annotated.shape == orig.shape
    assert annotated.dtype == np.uint8
    # Visual overlay should have modified pixel values
    assert not np.array_equal(annotated, orig)


def test_face_shape_on_blank_image():
    """Verify face shape analysis on an image without any faces returns empty results gracefully."""
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    result = detekface.detect(blank, analyze_face_shape=True)
    assert result.count == 0
    assert len(result.faces) == 0


def test_undetermined_fallback_on_extreme_yaw():
    """Verify that extreme head rotation yields 'undetermined' shape with cautionary warning."""
    # Create synthetic landmarks with severe yaw asymmetry (nose shifted far to the left)
    landmarks = [Point(100, 100) for _ in range(478)]
    # Forehead (10) and chin (152)
    landmarks[10] = Point(200, 50)
    landmarks[152] = Point(200, 250)
    # Cheeks
    landmarks[234] = Point(120, 150)
    landmarks[454] = Point(280, 150)
    # Temples and jaw
    landmarks[54] = Point(140, 70)
    landmarks[284] = Point(260, 70)
    landmarks[58] = Point(140, 210)
    landmarks[288] = Point(260, 210)
    # Nose and eyes: place nose very close to right eye (extreme yaw)
    landmarks[1] = Point(125, 130)  # Nose tip
    landmarks[33] = Point(120, 110)  # Right eye
    landmarks[263] = Point(270, 110)  # Left eye

    res = analyze_face_geometry(landmarks, image_shape=(300, 400))
    assert res.label == "undetermined"
    assert res.fit_score == 0.0
    assert res.quality_status == "rejected"
    assert any("yaw" in w.lower() for w in res.warnings)
    assert len(res.reasons) > 0


def test_insufficient_landmarks_fallback():
    """Verify that passing insufficient landmarks returns undetermined result."""
    lms = [Point(10, 10) for _ in range(50)]  # Only 50 landmarks
    res = analyze_face_geometry(lms, image_shape=(200, 200))
    assert res.label == "undetermined"
    assert res.fit_score == 0.0
    assert res.quality_status == "rejected"
    assert any("insufficient" in w.lower() for w in res.warnings)


def test_batch_detection_with_face_shape_analysis():
    """Verify batch detection with analyze_face_shape=True."""
    batch_results = detekface.detect_many(
        [SAMPLE_FACE_PATH],
        analyze_face_shape=True,
    )
    assert len(batch_results) == 1
    assert batch_results[0].is_success is True
    res = batch_results[0].result
    assert res is not None
    assert res.has_faces is True
    assert res.faces[0].face_shape is not None
    assert res.faces[0].face_shape.label in SUPPORTED_FACE_SHAPES
