"""Tests for performance optimizations, coordinate transformations, frame skipping, and resolution controls."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

import detekface
from detekface.backends.yunet import YuNetBackend
from detekface.camera import CameraSession, camera, open_camera
from detekface.detector import (
    FaceDetector,
    _scale_point,
    _transform_detections_to_original_scale,
    _validate_input_size,
)
from detekface.exceptions import ConfigurationError
from detekface.types import BoundingBox, FaceDetection, FaceLandmarks, Point
from detekface.video import video


# =========================================================================
# 1. Coordinate Transformation & Clamping
# =========================================================================

def test_scale_point_scaling_and_clamping():
    """Verify landmark point scaling and bounding clamping."""
    pt = Point(x=10, y=20)
    # Scale 2x with boundaries 100x100
    scaled = _scale_point(pt, scale_x=2.0, scale_y=2.5, max_w=100, max_h=100)
    assert scaled is not None
    assert scaled.x == 20
    assert scaled.y == 50

    # Test None input returns None
    assert _scale_point(None, scale_x=2.0, scale_y=2.0, max_w=100, max_h=100) is None

    # Test clamping out-of-bounds point
    oob_pt = Point(x=150, y=200)
    clamped = _scale_point(oob_pt, scale_x=2.0, scale_y=2.0, max_w=100, max_h=100)
    assert clamped.x == 99
    assert clamped.y == 99


def test_transform_detections_to_original_scale():
    """Verify bounding box and all landmarks scale accurately from downscaled inference resolution."""
    raw_face = FaceDetection(
        bbox=BoundingBox(x=10, y=10, width=20, height=30),
        confidence=0.92,
        landmarks=FaceLandmarks(
            right_eye=Point(15, 15),
            left_eye=Point(25, 15),
            nose_tip=Point(20, 22),
            right_mouth=Point(16, 32),
            left_mouth=Point(24, 32),
            mouth_center=Point(20, 32),
            right_ear=Point(10, 18),
            left_ear=Point(30, 18),
        ),
    )

    # Scale 320x240 -> 1280x720 (scale_x = 4.0, scale_y = 3.0)
    scaled = _transform_detections_to_original_scale(
        faces=[raw_face],
        scale_x=4.0,
        scale_y=3.0,
        orig_w=1280,
        orig_h=720,
    )

    assert len(scaled) == 1
    face = scaled[0]
    assert face.confidence == 0.92
    assert face.bbox.x == 40
    assert face.bbox.y == 30
    assert face.bbox.width == 80
    assert face.bbox.height == 90

    assert face.landmarks.right_eye == Point(60, 45)
    assert face.landmarks.left_eye == Point(100, 45)
    assert face.landmarks.nose_tip == Point(80, 66)
    assert face.landmarks.right_mouth == Point(64, 96)
    assert face.landmarks.left_mouth == Point(96, 96)
    assert face.landmarks.mouth_center == Point(80, 96)
    assert face.landmarks.right_ear == Point(40, 54)
    assert face.landmarks.left_ear == Point(120, 54)


def test_transform_detections_clamping_edge_cases():
    """Verify bounding boxes overflowing original image edges are strictly clamped."""
    raw_face = FaceDetection(
        bbox=BoundingBox(x=90, y=90, width=50, height=50),
        confidence=0.85,
    )
    # Image is 100x100, scale is 1.0 -> box exceeds right & bottom
    scaled = _transform_detections_to_original_scale(
        faces=[raw_face],
        scale_x=1.0,
        scale_y=1.0,
        orig_w=100,
        orig_h=100,
    )
    face = scaled[0]
    assert face.bbox.x == 90
    assert face.bbox.y == 90
    assert face.bbox.x + face.bbox.width <= 100
    assert face.bbox.y + face.bbox.height <= 100
    assert face.landmarks is None


# =========================================================================
# 2. Input Size Validation & Configuration
# =========================================================================

def test_validate_input_size_valid():
    """Verify valid input sizes are normalized to (int, int)."""
    assert _validate_input_size(None) is None
    assert _validate_input_size((640, 480)) == (640, 480)
    assert _validate_input_size([320, 240]) == (320, 240)


@pytest.mark.parametrize(
    "invalid_size",
    [
        (640,),
        (640, 480, 3),
        "640x480",
        (640, -10),
        (0, 480),
        (10, 10),  # Less than minimum 32
        (640.5, 480),
        (True, False),
    ],
)
def test_validate_input_size_invalid(invalid_size):
    """Verify invalid input size formats raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        _validate_input_size(invalid_size)


def test_detector_input_size_property():
    """Verify FaceDetector correctly stores and exposes the input_size property."""
    detector = FaceDetector(input_size=(640, 480))
    assert detector.input_size == (640, 480)

    detector_default = FaceDetector()
    assert detector_default.input_size is None


# =========================================================================
# 3. High-Resolution Face Crop Preservation with Scaled Inference
# =========================================================================

def test_high_resolution_face_crop_preservation():
    """Verify that when input_size is scaled down, the returned image is pristine full resolution."""
    # Create a 1920x1080 image with a distinctive pattern
    orig_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    orig_img[100:300, 200:400] = [10, 20, 200]  # distinct rectangle

    # Mock backend detect to return a face at 640x360 coords:
    # 640x360 coordinates: box at (x=67, y=33, w=67, h=67) corresponds to ~ (200, 100, 200, 200) in 1080p
    mock_backend = MagicMock()
    mock_backend.name = "mock"
    mock_backend.detect.return_value = [
        FaceDetection(
            bbox=BoundingBox(x=67, y=33, width=67, height=67),
            confidence=0.95,
            landmarks=None,
        )
    ]

    detector = FaceDetector(input_size=(640, 360))
    detector._backend = mock_backend

    res = detector.detect(orig_img)

    # 1. Result image must be the original 1920x1080 image (not downscaled 640x360)
    assert res.image.shape == (1080, 1920, 3)

    # 2. Rescaled box must match original 1080p coordinate space
    face = res.faces[0]
    assert 195 <= face.bbox.x <= 205
    assert 95 <= face.bbox.y <= 105
    assert 195 <= face.bbox.width <= 205
    assert 195 <= face.bbox.height <= 205

    # 3. Face crop from pristine image must be full-resolution
    captured = detekface.capture(res, save=False)
    crop = captured.image
    assert crop.shape[0] == face.bbox.height
    assert crop.shape[1] == face.bbox.width
    assert crop.shape[0] > 100  # Full size, not 67px!


# =========================================================================
# 4. Redundant setInputSize Caching
# =========================================================================

def test_yunet_set_input_size_caching():
    """Verify YuNetBackend caches input size to avoid redundant setInputSize calls."""
    backend = YuNetBackend()
    backend._detector = MagicMock()
    backend._detector.detect.return_value = (1, None)
    backend._current_input_size = None

    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

    backend.detect(frame1)
    assert backend._current_input_size == (640, 480)
    assert backend._detector.setInputSize.call_count == 1

    # Second call with same dimensions must NOT call setInputSize again
    backend.detect(frame2)
    assert backend._detector.setInputSize.call_count == 1

    # Third call with new dimensions MUST call setInputSize
    frame3 = np.zeros((720, 1280, 3), dtype=np.uint8)
    backend.detect(frame3)
    assert backend._detector.setInputSize.call_count == 2
    assert backend._current_input_size == (1280, 720)


# =========================================================================
# 5. Video Frame Skipping
# =========================================================================

def test_video_frame_skipping(tmp_path: Path):
    """Verify video(frame_skip=N) skips inference on intermediate frames while writing all frames."""
    vid_file = tmp_path / "skip_test.mp4"
    out_file = tmp_path / "skip_out.mp4"

    # Create 9 frames video
    writer = cv2.VideoWriter(str(vid_file), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (100, 100))
    for _ in range(9):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()

    mock_detector = MagicMock()
    mock_detector.detect.return_value = detekface.types.DetectionResult(faces=[], image=np.zeros((100, 100, 3), dtype=np.uint8))

    # frame_skip=2: detect on frame 0, 3, 6 (total 3 detections for 9 frames)
    video(
        path=vid_file,
        output_path=out_file,
        detector=mock_detector,
        frame_skip=2,
        show=False,
    )

    assert mock_detector.detect.call_count == 3
    assert out_file.exists()


@pytest.mark.parametrize("invalid_skip", [-1, -5, "1", 1.5, True])
def test_video_frame_skip_invalid(invalid_skip, tmp_path: Path):
    """Verify invalid frame_skip raises ConfigurationError."""
    vid_file = tmp_path / "dummy.mp4"
    vid_file.write_bytes(b"dummy")

    with pytest.raises(ConfigurationError):
        video(path=vid_file, frame_skip=invalid_skip, show=False)


# =========================================================================
# 6. Camera Resolution & Memory Semantics
# =========================================================================

def test_camera_resolution_negotiation():
    """Verify CameraSession sets CAP_PROP_FRAME_WIDTH and CAP_PROP_FRAME_HEIGHT when resolution is provided."""
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap

        session = CameraSession(
            camera_index=0,
            resolution=(1280, 720),
            auto_open=True,
        )

        assert session.resolution == (1280, 720)
        # Verify resolution set calls
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 720)


@pytest.mark.parametrize("invalid_res", [(1280,), (1280, -10), (10, 10), "1280x720", (True, 720)])
def test_camera_resolution_validation(invalid_res):
    """Verify invalid camera resolution raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        CameraSession(camera_index=0, resolution=invalid_res, auto_open=False)


def test_camera_read_raw_mirror_no_double_copy():
    """Verify read_raw with mirror=True returns independent flipped frame."""
    frame = np.ones((100, 100, 3), dtype=np.uint8)
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = mock_cap

        session = CameraSession(mirror=True, auto_open=True)
        raw = session.read_raw()
        assert raw.shape == (100, 100, 3)
        # Verify modifying returned frame does not affect subsequent operations
        raw[0, 0] = [255, 255, 255]
        assert raw is session.latest_frame


def test_open_camera_and_camera_pass_parameters():
    """Verify open_camera and camera pass resolution and input_size properly."""
    with patch("detekface.camera.CameraSession") as mock_session_cls:
        open_camera(resolution=(640, 480), input_size=(320, 240), auto_open=False)
        mock_session_cls.assert_called_once_with(
            camera_index=0,
            mirror=True,
            confidence_threshold=0.7,
            detector=None,
            backend="yunet",
            window_title=None,
            auto_open=False,
            resolution=(640, 480),
            input_size=(320, 240),
        )
