"""Tests verifying the shared engine architecture and parity between Mode 1 and Mode 2."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import pytest

import detekface
from detekface.exceptions import DependencyNotFoundError
from detekface.face_shape import (
    DenseLandmarkExtractor,
    analyze_face_geometry,
)

ASSETS_DIR = Path(__file__).parent / "assets"
SAMPLE_FACE_PATH = ASSETS_DIR / "sample_face.jpg"


def test_shared_engine_classification_parity():
    """Verify that still-image Mode 1 and webcam Mode 2 use the exact same underlying analysis engine.

    Given identical landmarks and frame, both modes MUST return identical shape labels,
    fit scores, alternatives, and measurements.
    """
    assert SAMPLE_FACE_PATH.is_file()
    img = cv2.imread(str(SAMPLE_FACE_PATH))

    # 1. Extract landmarks directly using shared extractor
    extractor = DenseLandmarkExtractor.get_default()
    landmark_sets = extractor.extract(img)
    assert len(landmark_sets) >= 1
    shared_lms = landmark_sets[0]

    # Compute reference result from shared geometric analysis
    h, w = img.shape[:2]
    reference_result = analyze_face_geometry(shared_lms, (h, w))

    # 2. Mode 1: detect(image, analyze_face_shape=True)
    res_mode1 = detekface.detect(img, analyze_face_shape=True)
    assert res_mode1.has_faces is True
    face_mode1 = res_mode1.faces[0]
    assert face_mode1.face_shape is not None

    assert face_mode1.face_shape.label == reference_result.label
    assert face_mode1.face_shape.fit_score == reference_result.fit_score
    assert face_mode1.face_shape.measurements == reference_result.measurements
    assert face_mode1.face_shape.alternatives == reference_result.alternatives

    # 3. Mode 2: CameraSession read on the identical frame
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, img)
        mock_cap_cls.return_value = mock_cap

        with detekface.open_camera(mirror=False, analyze_face_shape=True) as session:
            # Read multiple frames so temporal tracker stabilizes to classified state
            for _ in range(4):
                cam_res = session.read()

            cam_face = cam_res.faces[0]
            assert cam_face.face_shape is not None
            # Classification label must match Mode 1 exactly
            assert cam_face.face_shape.label == reference_result.label
            assert cam_face.face_shape.measurements == reference_result.measurements


def test_mediapipe_missing_dependency_guidance():
    """Verify that attempting face shape analysis without mediapipe raises actionable DependencyNotFoundError."""
    with patch.dict("sys.modules", {"mediapipe": None}):
        with pytest.raises(DependencyNotFoundError) as exc_info:
            DenseLandmarkExtractor()

        err_msg = str(exc_info.value)
        assert "mediapipe" in err_msg.lower()
        assert "detekface[mediapipe]" in err_msg.lower()
