"""Tests for Mode 2: Real-Time Webcam Face Shape Scanner."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from detekface.camera import camera, open_camera
from detekface.exceptions import ConfigurationError
from detekface.face_shape import (
    TemporalShapeTracker,
    draw_face_shape_overlay,
)
from detekface.types import BoundingBox, DetectionResult, FaceDetection, FaceShapeResult, Point

ASSETS_DIR = Path(__file__).parent / "assets"
SAMPLE_FACE_PATH = ASSETS_DIR / "sample_face.jpg"


def _make_dummy_frame():
    # Load actual sample face image if available or create dummy BGR frame
    if SAMPLE_FACE_PATH.is_file():
        return cv2.imread(str(SAMPLE_FACE_PATH))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 80, (200, 200, 200), -1)
    return frame


def test_camera_session_with_face_shape_analysis():
    """Verify CameraSession operates with analyze_face_shape=True."""
    dummy_frame = _make_dummy_frame()

    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap_cls.return_value = mock_cap

        with open_camera(camera_index=0, analyze_face_shape=True, auto_open=True) as session:
            assert session.is_open is True
            assert session.analyze_face_shape is True
            assert session._shape_tracker is not None

            # Read frame and perform detection
            result = session.read()
            assert isinstance(result, DetectionResult)
            assert session.latest_result is result
            assert session.latest_frame is not None

            # Render frame with scanner overlay
            rendered = session.render(
                show_face_shape=True,
                face_shape_effect="scan",
                show_shape_contour=True,
                show_shape_measurements=True,
            )
            assert isinstance(rendered, np.ndarray)
            assert rendered.shape == session.latest_frame.shape


def test_temporal_shape_tracker_smoothing():
    """Verify TemporalShapeTracker tracks faces and smooths classifications over frames."""
    tracker = TemporalShapeTracker(window_size=5)

    bbox = BoundingBox(x=100, y=100, width=150, height=200)

    # Frame 1 & 2: Initial frames state is 'analyzing'
    for f in range(2):
        shape_res = FaceShapeResult(
            label="oval",
            fit_score=0.85,
            alternatives={"round": 0.60},
            measurements={"length_to_width_ratio": 1.35},
            quality_status="good",
        )
        face = FaceDetection(bbox=bbox, confidence=0.9, face_shape=shape_res)
        smoothed = tracker.update([face])
        assert len(smoothed) == 1
        assert smoothed[0].face_shape.label == "analyzing"
        assert smoothed[0].face_shape.quality_status == "analyzing"

    # Frame 3 onwards: Classified state is established
    for f in range(3):
        shape_res = FaceShapeResult(
            label="oval",
            fit_score=0.88,
            alternatives={"round": 0.58},
            measurements={"length_to_width_ratio": 1.36},
            quality_status="good",
        )
        face = FaceDetection(bbox=bbox, confidence=0.92, face_shape=shape_res)
        smoothed = tracker.update([face])
        assert len(smoothed) == 1
        assert smoothed[0].face_shape.label == "oval"
        assert smoothed[0].face_shape.quality_status == "good"
        assert smoothed[0].face_shape.fit_score > 0.80


def test_camera_scanner_visual_effects():
    """Verify draw_face_shape_overlay draws contour, scanline, and status badges."""
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    shape_res = FaceShapeResult(
        label="square",
        fit_score=0.91,
        alternatives={"round": 0.65},
        measurements={"length_to_width_ratio": 1.05, "jaw_angle": 115.0},
        quality_status="good",
    )
    face = FaceDetection(
        bbox=BoundingBox(50, 50, 200, 200),
        confidence=0.95,
        face_shape=shape_res,
        dense_landmarks=[Point(50 + i % 200, 50 + i % 200) for i in range(478)],
    )

    out = image.copy()
    draw_face_shape_overlay(
        image=out,
        face=face,
        effect="scan",
        scan_phase=0.5,
        color=(0, 255, 136),
        thickness=2,
        show_contour=True,
        show_measurements=True,
    )

    # Pixel contents should be rendered
    assert np.any(out > 0)


def test_camera_function_with_face_shape_parameters(monkeypatch):
    """Verify camera() entry point accepts face shape arguments and cleans up."""
    dummy_frame = _make_dummy_frame()

    with (
        patch("cv2.VideoCapture") as mock_cap_cls,
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.waitKey", return_value=ord("q")),
        patch("cv2.destroyWindow"),
    ):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap_cls.return_value = mock_cap

        # Call detekface.camera with max_frames=2 and analyze_face_shape=True
        camera(
            camera_index=0,
            max_frames=2,
            analyze_face_shape=True,
            show_face_shape=True,
            face_shape_effect="scan",
        )

        assert mock_imshow.called
        assert mock_cap.release.called


def test_camera_parameter_validation():
    """Verify camera() rejects invalid face shape boolean parameters."""
    with pytest.raises(ConfigurationError):
        camera(analyze_face_shape="not_a_bool")  # type: ignore

    with pytest.raises(ConfigurationError):
        camera(show_face_shape=123)  # type: ignore
