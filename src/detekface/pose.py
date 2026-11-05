"""Head-pose estimation engine for detekface 0.3.0.

Computes 3D head rotation angles (Yaw, Pitch, Roll) in degrees using facial
landmarks and Perspective-n-Point (PnP) pose solving against a canonical 3D
facial anatomical model.

Coordinate Conventions & Angle Definitions:
- **Yaw (Left/Right Rotation)**:
  Rotation around the vertical Y-axis.
  - Positive (> 0°): Subject turning to their right (viewer's left).
  - Negative (< 0°): Subject turning to their left (viewer's right).
- **Pitch (Up/Down Tilt)**:
  Rotation around the lateral X-axis.
  - Positive (> 0°): Subject tilting head upward (looking up).
  - Negative (< 0°): Subject tilting head downward (looking down).
- **Roll (In-Plane Lateral Tilt)**:
  Rotation around the optical Z-axis.
  - Positive (> 0°): Subject tilting head toward their right shoulder.
  - Negative (< 0°): Subject tilting head toward their left shoulder.

Works across still images, live webcam feeds, and video streams.
Supports both primary 5-point facial landmarks (YuNet / MediaPipe) and
dense 468/478-point facial meshes.
Includes temporal smoothing for webcam/video feeds and 3D axis projection.
"""

import math
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from detekface.types import FaceDetection, HeadPose, Point

# Canonical 3D anthropometric face model points (in millimetres, nose tip at origin)
# X: viewer right (+), Y: image down (+), Z: into scene (+)
CANONICAL_6_POINT_MODEL_5PT = np.array(
    [
        [-35.0, -35.0, 35.0],  # Right eye center (subject's right, viewer's left)
        [35.0, -35.0, 35.0],   # Left eye center (subject's left, viewer's right)
        [0.0, 0.0, 0.0],       # Nose tip (origin)
        [-25.0, 35.0, 25.0],   # Right mouth corner
        [25.0, 35.0, 25.0],    # Left mouth corner
        [0.0, 35.0, 25.0],     # Mouth center
    ],
    dtype=np.float64,
)

# Canonical 6-point model for dense landmarks (includes chin)
CANONICAL_6_POINT_MODEL_DENSE = np.array(
    [
        [0.0, 0.0, 0.0],       # Nose tip (index 1)
        [0.0, 65.0, 25.0],     # Chin (index 152)
        [-35.0, -35.0, 35.0],  # Right eye outer corner (index 33)
        [35.0, -35.0, 35.0],   # Left eye outer corner (index 263)
        [-25.0, 35.0, 25.0],   # Right mouth corner (index 61)
        [25.0, 35.0, 25.0],    # Left mouth corner (index 291)
    ],
    dtype=np.float64,
)


def _get_camera_matrix(image_shape: Tuple[int, int]) -> np.ndarray:
    """Construct an intrinsic camera matrix based on image dimensions."""
    h, w = image_shape[:2]
    focal_length = float(w)
    center_x = float(w) / 2.0
    center_y = float(h) / 2.0
    return np.array(
        [
            [focal_length, 0.0, center_x],
            [0.0, focal_length, center_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_head_pose(
    detection: FaceDetection,
    image_shape: Tuple[int, int],
    camera_matrix: Optional[np.ndarray] = None,
) -> Optional[HeadPose]:
    """Estimate 3D head pose (Yaw, Pitch, Roll) from facial landmarks.

    Uses dense landmarks if present; otherwise falls back to 5-point landmarks.

    Args:
        detection: FaceDetection containing 2D facial landmarks or dense landmarks.
        image_shape: Image dimensions tuple (height, width).
        camera_matrix: Optional custom 3x3 camera intrinsic matrix.

    Returns:
        Optional[HeadPose]: Estimated angles in degrees, or None if landmarks are insufficient.
    """
    if image_shape[0] <= 0 or image_shape[1] <= 0:
        return None

    cam_matrix = _get_camera_matrix(image_shape) if camera_matrix is None else camera_matrix
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_points: Optional[np.ndarray] = None
    model_points: Optional[np.ndarray] = None

    # Priority 1: Dense landmarks (468/478 points) if available
    if detection.dense_landmarks is not None and len(detection.dense_landmarks) >= 292:
        try:
            dl = detection.dense_landmarks
            # Indices: 1 (nose), 152 (chin), 33 (right eye), 263 (left eye), 61 (right mouth), 291 (left mouth)
            image_points = np.array(
                [
                    [dl[1].x, dl[1].y],
                    [dl[152].x, dl[152].y],
                    [dl[33].x, dl[33].y],
                    [dl[263].x, dl[263].y],
                    [dl[61].x, dl[61].y],
                    [dl[291].x, dl[291].y],
                ],
                dtype=np.float64,
            )
            model_points = CANONICAL_6_POINT_MODEL_DENSE
        except (IndexError, AttributeError):
            image_points = None

    # Priority 2: Standard 5-point landmarks (extended to 6 points with mouth center)
    if image_points is None and detection.landmarks is not None:
        lm = detection.landmarks
        if (
            lm.right_eye is not None
            and lm.left_eye is not None
            and lm.nose_tip is not None
            and lm.right_mouth is not None
            and lm.left_mouth is not None
        ):
            eye_dist = math.hypot(lm.left_eye.x - lm.right_eye.x, lm.left_eye.y - lm.right_eye.y)
            if eye_dist >= 4.0:
                mouth_cx = (lm.right_mouth.x + lm.left_mouth.x) / 2.0
                mouth_cy = (lm.right_mouth.y + lm.left_mouth.y) / 2.0
                image_points = np.array(
                    [
                        [lm.right_eye.x, lm.right_eye.y],
                        [lm.left_eye.x, lm.left_eye.y],
                        [lm.nose_tip.x, lm.nose_tip.y],
                        [lm.right_mouth.x, lm.right_mouth.y],
                        [lm.left_mouth.x, lm.left_mouth.y],
                        [mouth_cx, mouth_cy],
                    ],
                    dtype=np.float64,
                )
                model_points = CANONICAL_6_POINT_MODEL_5PT

    if image_points is None or model_points is None:
        return None

    try:
        success, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or rvec is None or tvec is None:
            return None
    except cv2.error:
        return None

    # Convert rotation vector to 3x3 rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Decompose rotation matrix into Euler angles
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        raw_pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        raw_yaw = math.atan2(-rmat[2, 0], sy)
        raw_roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        raw_pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        raw_yaw = math.atan2(-rmat[2, 0], sy)
        raw_roll = 0.0

    # Convert radians to degrees
    pitch = -math.degrees(raw_pitch)
    yaw = math.degrees(raw_yaw)
    roll = math.degrees(raw_roll)

    # Normalize roll to [-90, +90]
    if roll > 90.0:
        roll -= 180.0
    elif roll < -90.0:
        roll += 180.0

    # Clamp angles to physically plausible human range [-90°, +90°]
    pitch = max(-90.0, min(90.0, pitch))
    yaw = max(-90.0, min(90.0, yaw))
    roll = max(-90.0, min(90.0, roll))

    return HeadPose(
        yaw=round(yaw, 2),
        pitch=round(pitch, 2),
        roll=round(roll, 2),
        rotation_vector=rvec,
        translation_vector=tvec,
    )


class TemporalPoseSmoother:
    """Stateful temporal smoother for head-pose angles in video and camera feeds."""

    def __init__(self, alpha: float = 0.70) -> None:
        """Initialize pose smoother.

        Args:
            alpha: Weight of fresh observation (0.0 to 1.0). Higher = more responsive.
        """
        self.alpha = max(0.1, min(1.0, float(alpha)))
        self._history: Dict[int, Tuple[float, float, float]] = {}

    def smooth(self, pose: HeadPose, track_id: Optional[int] = None) -> HeadPose:
        """Apply exponential moving average to head pose angles.

        Args:
            pose: Freshly estimated HeadPose instance.
            track_id: Optional temporal track ID. If None, uses fallback slot 0.

        Returns:
            HeadPose: Smoothed HeadPose instance with jitter eliminated.
        """
        key = track_id if track_id is not None else 0

        if key in self._history:
            prev_yaw, prev_pitch, prev_roll = self._history[key]
            new_yaw = self.alpha * pose.yaw + (1.0 - self.alpha) * prev_yaw
            new_pitch = self.alpha * pose.pitch + (1.0 - self.alpha) * prev_pitch
            new_roll = self.alpha * pose.roll + (1.0 - self.alpha) * prev_roll
        else:
            new_yaw, new_pitch, new_roll = pose.yaw, pose.pitch, pose.roll

        self._history[key] = (new_yaw, new_pitch, new_roll)

        return HeadPose(
            yaw=round(new_yaw, 2),
            pitch=round(new_pitch, 2),
            roll=round(new_roll, 2),
            rotation_vector=pose.rotation_vector,
            translation_vector=pose.translation_vector,
        )

    def reset(self) -> None:
        """Clear recorded pose histories."""
        self._history.clear()


def project_pose_axes(
    image_shape: Tuple[int, int],
    pose: HeadPose,
    nose_point: Point,
    axis_length: float = 40.0,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """Project 3D Cartesian orientation axes onto 2D image coordinates.

    Args:
        image_shape: (height, width) of the target image.
        pose: HeadPose instance containing rotation and translation vectors.
        nose_point: 2D pixel coordinates of the nose tip.
        axis_length: Length in millimetres of the projected 3D axes.

    Returns:
        Optional tuple containing:
        - Origin point (nose tip): (x, y)
        - X-axis endpoint (Pitch axis / Red): (x, y)
        - Y-axis endpoint (Yaw axis / Green): (x, y)
        - Z-axis endpoint (Roll/Depth axis / Blue): (x, y)
    """
    if pose.rotation_vector is None or pose.translation_vector is None:
        rad_yaw = math.radians(pose.yaw)
        rad_pitch = math.radians(pose.pitch)
        rad_roll = math.radians(pose.roll)

        nx, ny = nose_point.x, nose_point.y
        x_pt = (int(round(nx + axis_length * math.cos(rad_roll))), int(round(ny + axis_length * math.sin(rad_roll))))
        y_pt = (int(round(nx - axis_length * math.sin(rad_yaw))), int(round(ny - axis_length * math.cos(rad_pitch))))
        z_pt = (int(round(nx - axis_length * math.sin(rad_yaw))), int(round(ny + axis_length * math.sin(rad_pitch))))
        return ((nx, ny), x_pt, y_pt, z_pt)

    cam_matrix = _get_camera_matrix(image_shape)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    axis_3d = np.array(
        [
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float64,
    )

    try:
        pts_2d, _ = cv2.projectPoints(
            axis_3d,
            pose.rotation_vector,
            pose.translation_vector,
            cam_matrix,
            dist_coeffs,
        )
        origin = (int(round(nose_point.x)), int(round(nose_point.y)))
        p_x = (int(round(pts_2d[0][0][0])), int(round(pts_2d[0][0][1])))
        p_y = (int(round(pts_2d[1][0][0])), int(round(pts_2d[1][0][1])))
        p_z = (int(round(pts_2d[2][0][0])), int(round(pts_2d[2][0][1])))
        return (origin, p_x, p_y, p_z)
    except Exception:
        return None
