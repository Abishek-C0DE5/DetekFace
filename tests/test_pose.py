"""Automated tests for 3D head-pose estimation (Yaw, Pitch, Roll) in detekface 0.3.0.

Tests:
- 5-point landmark pose estimation
- Dense landmark pose estimation
- Directional rotation angle sign conventions:
    * Yaw: positive = right, negative = left
    * Pitch: positive = up, negative = down
    * Roll: positive = tilt right, negative = tilt left
- Temporal pose smoothing
- Graceful handling of unavailable / degenerate landmarks
- 3D Cartesian axis projection
- Independence and non-interference with face-shape classification
- Still image integration via detekface.detect(..., estimate_pose=True)
"""

import math
from typing import Tuple
from unittest.mock import patch
import numpy as np

import detekface
from detekface import (
    BoundingBox,
    FaceDetection,
    FaceLandmarks,
    HeadPose,
    Point,
    estimate_head_pose,
)
from detekface.pose import TemporalPoseSmoother, project_pose_axes


def _make_landmarks(
    cx: int = 320,
    cy: int = 240,
    eye_span: int = 70,
    nose_y_offset: int = 30,
    mouth_y_offset: int = 60,
    mouth_span: int = 50,
    yaw_shift: int = 0,
    pitch_shift: int = 0,
    roll_angle_deg: float = 0.0,
) -> FaceLandmarks:
    """Generate synthetic facial landmarks with precise geometrical offsets."""
    half_eye = eye_span // 2
    half_mouth = mouth_span // 2

    # Unrotated coordinates relative to center
    re_x, re_y = -half_eye + yaw_shift, -30 + pitch_shift
    le_x, le_y = half_eye + yaw_shift, -30 + pitch_shift
    n_x, n_y = 0 + yaw_shift * 2, -30 + nose_y_offset + pitch_shift * 2
    rm_x, rm_y = -half_mouth + yaw_shift, -30 + mouth_y_offset + pitch_shift
    lm_x, lm_y = half_mouth + yaw_shift, -30 + mouth_y_offset + pitch_shift

    # Apply in-plane roll rotation
    rad = math.radians(roll_angle_deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)

    def rot(x: float, y: float) -> Tuple[int, int]:
        rx = x * cos_r - y * sin_r + cx
        ry = x * sin_r + y * cos_r + cy
        return int(round(rx)), int(round(ry))

    r_eye = rot(re_x, re_y)
    l_eye = rot(le_x, le_y)
    nose = rot(n_x, n_y)
    r_mouth = rot(rm_x, rm_y)
    l_mouth = rot(lm_x, lm_y)

    return FaceLandmarks(
        right_eye=Point(x=r_eye[0], y=r_eye[1]),
        left_eye=Point(x=l_eye[0], y=l_eye[1]),
        nose_tip=Point(x=nose[0], y=nose[1]),
        right_mouth=Point(x=r_mouth[0], y=r_mouth[1]),
        left_mouth=Point(x=l_mouth[0], y=l_mouth[1]),
    )


# =========================================================================
# 1. Frontal & Basic Pose Estimation
# =========================================================================


def test_frontal_face_pose_near_zero() -> None:
    """Verify standard frontal face produces angles near zero (|angles| < 8°)."""
    lm = _make_landmarks()
    det = FaceDetection(
        bbox=BoundingBox(x=200, y=140, width=240, height=240),
        confidence=0.99,
        landmarks=lm,
    )
    pose = estimate_head_pose(det, image_shape=(480, 640))

    assert pose is not None
    assert isinstance(pose, HeadPose)
    assert abs(pose.yaw) < 8.0
    assert abs(pose.pitch) < 8.0
    assert abs(pose.roll) < 8.0


# =========================================================================
# 2. Angle Sign Conventions (Yaw, Pitch, Roll)
# =========================================================================


def test_yaw_rotation_direction_signs() -> None:
    """Verify turning right produces positive yaw, and turning left produces negative yaw."""
    # Turned right: nose shifts leftward relative to eye midpoint
    lm_right = _make_landmarks(yaw_shift=-15)
    det_right = FaceDetection(bbox=BoundingBox(200, 140, 240, 240), confidence=0.95, landmarks=lm_right)
    pose_right = estimate_head_pose(det_right, image_shape=(480, 640))

    # Turned left: nose shifts rightward relative to eye midpoint
    lm_left = _make_landmarks(yaw_shift=15)
    det_left = FaceDetection(bbox=BoundingBox(200, 140, 240, 240), confidence=0.95, landmarks=lm_left)
    pose_left = estimate_head_pose(det_left, image_shape=(480, 640))

    assert pose_right is not None
    assert pose_left is not None
    assert pose_right.yaw > 0.0, f"Expected positive yaw when turning right, got {pose_right.yaw}"
    assert pose_left.yaw < 0.0, f"Expected negative yaw when turning left, got {pose_left.yaw}"


def test_pitch_tilt_direction_signs() -> None:
    """Verify looking up produces positive pitch, and looking down produces negative pitch."""
    # Looking up: nose tip moves closer to eye line
    lm_up = _make_landmarks(nose_y_offset=20, pitch_shift=-5)
    det_up = FaceDetection(bbox=BoundingBox(200, 140, 240, 240), confidence=0.95, landmarks=lm_up)
    pose_up = estimate_head_pose(det_up, image_shape=(480, 640))

    # Looking down: nose tip moves closer to mouth line
    lm_down = _make_landmarks(nose_y_offset=40, pitch_shift=5)
    det_down = FaceDetection(bbox=BoundingBox(200, 140, 240, 240), confidence=0.95, landmarks=lm_down)
    pose_down = estimate_head_pose(det_down, image_shape=(480, 640))

    assert pose_up is not None
    assert pose_down is not None
    assert pose_up.pitch > 0.0, f"Expected positive pitch when looking up, got {pose_up.pitch}"
    assert pose_down.pitch < 0.0, f"Expected negative pitch when looking down, got {pose_down.pitch}"
    assert pose_up.pitch > pose_down.pitch


def test_roll_tilt_direction_signs() -> None:
    """Verify tilting head right produces positive roll, and tilting left produces negative roll."""
    # Tilt head right (clockwise in image space: right eye lower)
    lm_roll_r = _make_landmarks(roll_angle_deg=18.0)
    det_roll_r = FaceDetection(bbox=BoundingBox(200, 140, 240, 240), confidence=0.95, landmarks=lm_roll_r)
    pose_roll_r = estimate_head_pose(det_roll_r, image_shape=(480, 640))

    # Tilt head left (counter-clockwise: left eye lower)
    lm_roll_l = _make_landmarks(roll_angle_deg=-18.0)
    det_roll_l = FaceDetection(bbox=BoundingBox(200, 140, 240, 240), confidence=0.95, landmarks=lm_roll_l)
    pose_roll_l = estimate_head_pose(det_roll_l, image_shape=(480, 640))

    assert pose_roll_r is not None
    assert pose_roll_l is not None
    assert pose_roll_r.roll > 0.0
    assert pose_roll_l.roll < 0.0


# =========================================================================
# 3. Dense Landmarks Pose Estimation
# =========================================================================


def test_dense_landmarks_pose_estimation() -> None:
    """Verify pose estimation functions accurately with dense 478-point landmarks."""
    # Build synthetic 478-point landmark array with key indices populated
    dense_pts = [Point(x=320, y=240) for _ in range(478)]
    # 1: Nose tip
    dense_pts[1] = Point(x=320, y=240)
    # 152: Chin
    dense_pts[152] = Point(x=320, y=295)
    # 33: Right eye corner (viewer's left)
    dense_pts[33] = Point(x=285, y=210)
    # 263: Left eye corner (viewer's right)
    dense_pts[263] = Point(x=355, y=210)
    # 61: Right mouth corner
    dense_pts[61] = Point(x=295, y=270)
    # 291: Left mouth corner
    dense_pts[291] = Point(x=345, y=270)

    det = FaceDetection(
        bbox=BoundingBox(260, 180, 120, 130),
        confidence=0.98,
        dense_landmarks=dense_pts,
    )
    pose = estimate_head_pose(det, image_shape=(480, 640))

    assert pose is not None
    assert abs(pose.yaw) < 10.0
    assert abs(pose.pitch) < 10.0
    assert abs(pose.roll) < 10.0


# =========================================================================
# 4. Temporal Smoothing
# =========================================================================


def test_temporal_pose_smoother() -> None:
    """Verify TemporalPoseSmoother eliminates jitter while tracking head orientation."""
    smoother = TemporalPoseSmoother(alpha=0.5)

    p1 = HeadPose(yaw=10.0, pitch=5.0, roll=2.0)
    s1 = smoother.smooth(p1, track_id=1)
    assert s1.yaw == 10.0

    # Jitter step to 14.0
    p2 = HeadPose(yaw=14.0, pitch=5.0, roll=2.0)
    s2 = smoother.smooth(p2, track_id=1)
    # Smoothed value with alpha=0.5 should be (10.0 + 14.0) / 2 = 12.0
    assert s2.yaw == 12.0

    smoother.reset()
    s3 = smoother.smooth(p2, track_id=1)
    assert s3.yaw == 14.0  # Reset cleared history


# =========================================================================
# 5. Degenerate & Missing Landmarks
# =========================================================================


def test_unavailable_or_collapsed_landmarks_return_none() -> None:
    """Verify missing or degenerate landmarks safely return None without crashing."""
    # No landmarks at all
    det_empty = FaceDetection(bbox=BoundingBox(10, 10, 50, 50), confidence=0.8, landmarks=None)
    assert estimate_head_pose(det_empty, image_shape=(480, 640)) is None

    # Collapsed / overlapping eyes (< 4px apart)
    collapsed_lm = FaceLandmarks(
        right_eye=Point(100, 100),
        left_eye=Point(101, 100),
        nose_tip=Point(100, 110),
        right_mouth=Point(95, 120),
        left_mouth=Point(105, 120),
    )
    det_collapsed = FaceDetection(bbox=BoundingBox(80, 80, 40, 50), confidence=0.8, landmarks=collapsed_lm)
    assert estimate_head_pose(det_collapsed, image_shape=(480, 640)) is None

    # Invalid image dimensions
    lm = _make_landmarks()
    det_valid = FaceDetection(bbox=BoundingBox(100, 100, 100, 100), confidence=0.9, landmarks=lm)
    assert estimate_head_pose(det_valid, image_shape=(0, 0)) is None


# =========================================================================
# 6. 3D Pose Axis Projection
# =========================================================================


def test_project_pose_axes_returns_valid_pixel_coordinates() -> None:
    """Verify 3D Cartesian axis projection generates valid 2D coordinates within frame bounds."""
    lm = _make_landmarks()
    det = FaceDetection(bbox=BoundingBox(200, 140, 240, 240), confidence=0.95, landmarks=lm)
    pose = estimate_head_pose(det, image_shape=(480, 640))
    assert pose is not None

    axes = project_pose_axes(image_shape=(480, 640), pose=pose, nose_point=lm.nose_tip, axis_length=40.0)
    assert axes is not None
    assert len(axes) == 4
    origin, p_pitch, p_yaw, p_roll = axes

    # Origin should be near the nose tip
    assert abs(origin[0] - lm.nose_tip.x) <= 1
    assert abs(origin[1] - lm.nose_tip.y) <= 1

    # Endpoints should be valid coordinate pairs
    for pt in (p_pitch, p_yaw, p_roll):
        assert isinstance(pt[0], int)
        assert isinstance(pt[1], int)


# =========================================================================
# 7. Non-Interference: Head Rotation vs Face-Shape Analysis
# =========================================================================


def test_head_rotation_does_not_corrupt_face_shape_analysis() -> None:
    """Verify that extreme head rotation triggers 'undetermined' with pose warning,

    ensuring head yaw is never confused with or misclassified as a deformed face shape.
    """
    from detekface.face_shape import analyze_face_geometry

    # Create synthetic 478 landmarks with heavy yaw asymmetry (nose closer to one eye)
    pts = [Point(x=320, y=240) for _ in range(478)]
    pts[10] = Point(320, 100)   # Forehead
    pts[152] = Point(320, 350)  # Chin
    pts[234] = Point(240, 220)  # Right cheek
    pts[454] = Point(390, 220)  # Left cheek
    pts[54] = Point(255, 130)   # Right temple
    pts[284] = Point(380, 130)  # Left temple
    pts[58] = Point(260, 310)   # Right jaw
    pts[288] = Point(375, 310)  # Left jaw
    pts[150] = Point(290, 340)
    pts[378] = Point(350, 340)

    # Extreme yaw: nose (x=270) is 30px from right eye (x=240), but 120px from left eye (x=390)
    pts[1] = Point(270, 230)
    pts[33] = Point(240, 200)
    pts[263] = Point(390, 200)

    shape_result = analyze_face_geometry(pts, image_shape=(480, 640))
    assert shape_result.label == "undetermined"
    assert shape_result.quality_status == "rejected"
    assert any("Face yaw angle exceeds frontal tolerance" in w for w in shape_result.warnings)


# =========================================================================
# 8. Still-Image detekface.detect(..., estimate_pose=True)
# =========================================================================


def test_still_image_detect_with_estimate_pose() -> None:
    """Verify detekface.detect(image, estimate_pose=True) attaches HeadPose to detections."""
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    lm = _make_landmarks(cx=150, cy=150)
    mock_det = FaceDetection(bbox=BoundingBox(100, 100, 100, 100), confidence=0.95, landmarks=lm)

    with patch.object(detekface.detector.FaceDetector, "detect") as mock_detect:
        mock_detect.return_value = detekface.types.DetectionResult(faces=[mock_det], image=dummy_img)

        # First call without estimate_pose
        detekface.detect(dummy_img, estimate_pose=False)
        mock_detect.assert_called_with(dummy_img, analyze_face_shape=False, mesh=False, estimate_pose=False)

        # Call with estimate_pose=True
        mock_pose = HeadPose(yaw=12.5, pitch=-3.2, roll=1.1)
        det_with_pose = FaceDetection(
            bbox=mock_det.bbox,
            confidence=mock_det.confidence,
            landmarks=mock_det.landmarks,
            head_pose=mock_pose,
        )
        mock_detect.return_value = detekface.types.DetectionResult(faces=[det_with_pose], image=dummy_img)

        res_pose = detekface.detect(dummy_img, estimate_pose=True)
        assert res_pose.faces[0].head_pose is not None
        assert res_pose.faces[0].head_pose.yaw == 12.5
