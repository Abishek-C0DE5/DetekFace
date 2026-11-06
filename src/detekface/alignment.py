"""Reusable face alignment module for detekface.

Provides robust facial tilt estimation, aspect-ratio preserving affine straightening,
configurable padding, consistent output dimension cropping, safe boundary handling,
and complete transformation metadata.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from detekface.exceptions import InvalidImageError, InvalidSettingError
from detekface.types import (
    BoundingBox,
    FaceDetection,
    FaceLandmarks,
    Point,
    _validate_and_prepare_save_path,
)


@dataclass
class AlignedFace:
    """Cropped and aligned face with full transformation metadata.

    Attributes:
        image: Straightened and cropped face image as BGR uint8 NumPy array.
        angle: Estimated face tilt angle in degrees (positive = clockwise roll).
        rotation_applied: The counter-rotation applied to straighten the face in degrees (-angle).
        center: Rotation center (cx, cy) in original image coordinates.
        scale: Applied uniform scaling factor (1.0 = original scale).
        transform_matrix: 2x3 affine transformation matrix used for cv2.warpAffine.
        original_bbox: BoundingBox of the face in original image space.
        face: The source FaceDetection object, if available.
        landmarks: Transformed 5-point facial landmarks mapped into the aligned crop coordinate space.
        dense_landmarks: Optional transformed dense facial landmarks mapped into crop coordinates.
    """

    image: np.ndarray
    angle: float
    rotation_applied: float
    center: Tuple[float, float]
    scale: float
    transform_matrix: np.ndarray
    original_bbox: BoundingBox
    face: Optional[FaceDetection] = None
    landmarks: Optional[FaceLandmarks] = None
    dense_landmarks: Optional[np.ndarray] = None

    @property
    def width(self) -> int:
        """Width of the aligned face crop in pixels."""
        return int(self.image.shape[1]) if isinstance(self.image, np.ndarray) and self.image.ndim >= 2 else 0

    @property
    def height(self) -> int:
        """Height of the aligned face crop in pixels."""
        return int(self.image.shape[0]) if isinstance(self.image, np.ndarray) and self.image.ndim >= 2 else 0

    @property
    def rgb_image(self) -> np.ndarray:
        """Aligned face image converted to RGB format."""
        if not isinstance(self.image, np.ndarray) or self.image.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def save(self, path: Union[str, Path]) -> Path:
        """Save the aligned face crop to disk."""
        resolved = _validate_and_prepare_save_path(path)
        success = cv2.imwrite(str(resolved), self.image)
        if not success:
            raise InvalidImageError(
                message=f"Failed to write aligned face crop to: '{resolved}'.",
                hint="Verify directory permissions and valid image path.",
            )
        return resolved


def estimate_face_rotation(face: FaceDetection) -> float:
    """Estimate face roll tilt angle in degrees from facial landmarks or 3D head pose.

    Returns:
        float: Tilt angle in degrees.
               Positive indicates clockwise tilt (viewer perspective: right eye higher, left eye lower).
               Negative indicates counter-clockwise tilt.
               0.0 indicates upright face.
    """
    if face.landmarks is not None:
        re = face.landmarks.right_eye
        le = face.landmarks.left_eye
        if re is not None and le is not None:
            # OpenCV coordinate system: y increases downward.
            # dx = left_eye.x - right_eye.x (subject's left eye is on viewer's right)
            # dy = left_eye.y - right_eye.y
            dx = float(le.x - re.x)
            dy = float(le.y - re.y)
            if abs(dx) > 1e-5 or abs(dy) > 1e-5:
                # Math atan2 gives angle from horizontal.
                # If dy > 0 (left eye lower than right eye), face is tilted clockwise.
                return float(math.degrees(math.atan2(dy, dx)))

    if face.head_pose is not None:
        return float(face.head_pose.roll)

    return 0.0


def transform_point(pt: Point, matrix: np.ndarray) -> Point:
    """Transform a single 2D Point using a 2x3 affine matrix."""
    vec = np.array([pt.x, pt.y, 1.0], dtype=np.float64)
    tx = float(matrix[0, 0] * vec[0] + matrix[0, 1] * vec[1] + matrix[0, 2])
    ty = float(matrix[1, 0] * vec[0] + matrix[1, 1] * vec[1] + matrix[1, 2])
    return Point(x=float(round(tx, 2)), y=float(round(ty, 2)))


def transform_landmarks(landmarks: FaceLandmarks, matrix: np.ndarray) -> FaceLandmarks:
    """Transform canonical 5-point facial landmarks into crop space using affine matrix."""
    return FaceLandmarks(
        right_eye=transform_point(landmarks.right_eye, matrix) if landmarks.right_eye is not None else None,
        left_eye=transform_point(landmarks.left_eye, matrix) if landmarks.left_eye is not None else None,
        nose_tip=transform_point(landmarks.nose_tip, matrix) if landmarks.nose_tip is not None else None,
        right_mouth=transform_point(landmarks.right_mouth, matrix) if landmarks.right_mouth is not None else None,
        left_mouth=transform_point(landmarks.left_mouth, matrix) if landmarks.left_mouth is not None else None,
    )


def transform_dense_landmarks(dense_pts: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform dense landmark array of shape (N, 2) or (N, 3) using 2x3 affine matrix."""
    if dense_pts is None or len(dense_pts) == 0:
        return np.empty((0, 2), dtype=np.float32)

    xy = dense_pts[:, :2].astype(np.float64)
    ones = np.ones((len(xy), 1), dtype=np.float64)
    homo = np.hstack([xy, ones])  # (N, 3)
    transformed_xy = (matrix @ homo.T).T  # (N, 2)

    if dense_pts.shape[1] >= 3:
        # Preserve original Z depth
        return np.hstack([transformed_xy.astype(np.float32), dense_pts[:, 2:3]])
    return transformed_xy.astype(np.float32)


def align_face(
    image: np.ndarray,
    face: FaceDetection,
    output_size: Optional[Tuple[int, int]] = (112, 112),
    padding: float = 0.25,
    center_on: str = "eyes",
    border_mode: int = cv2.BORDER_REFLECT_101,
    border_value: Tuple[int, int, int] = (0, 0, 0),
) -> AlignedFace:
    """Align and crop a detected face using landmark-guided rotation and aspect-ratio preservation.

    Straightens tilted faces by aligning the eye line horizontally, scales uniformly to preserve
    facial proportions without stretching, and uses border padding reflection to prevent boundary artifacts.

    Args:
        image: Original input image array (BGR uint8).
        face: FaceDetection object containing bounding box and landmarks.
        output_size: Desired (width, height) output dimensions in pixels, or None to retain natural face dimensions.
        padding: Fractional padding around the face bounding box (e.g. 0.25 adds 25% margin).
        center_on: Anchor point for cropping; 'eyes' (canonical portrait centering) or 'face' (box center).
        border_mode: OpenCV border interpolation mode (e.g. cv2.BORDER_REFLECT_101, cv2.BORDER_CONSTANT).
        border_value: BGR color used when border_mode is cv2.BORDER_CONSTANT.

    Returns:
        AlignedFace: Straightened, cropped face array and transformation metadata.

    Raises:
        InvalidImageError: If the input image is invalid or empty.
        InvalidSettingError: If output_size or padding settings are invalid.
    """
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim < 2:
        raise InvalidImageError(
            message="Cannot align face from an invalid or empty image array.",
            hint="Ensure input image is a valid 2D/3D NumPy array.",
        )

    if padding < 0.0:
        raise InvalidSettingError(
            message=f"Padding must be non-negative, got: {padding}.",
            hint="Specify padding >= 0.0, e.g. padding=0.25 for 25% boundary margin.",
        )

    if output_size is not None:
        if (
            not isinstance(output_size, (tuple, list))
            or len(output_size) != 2
            or int(output_size[0]) <= 0
            or int(output_size[1]) <= 0
        ):
            raise InvalidSettingError(
                message=f"output_size must be a tuple of two positive integers (width, height), got: {output_size}.",
                hint="Pass a tuple like (112, 112) or (224, 224), or None to keep natural face dimensions.",
            )
        out_w = int(output_size[0])
        out_h = int(output_size[1])
    else:
        out_w = None
        out_h = None

    bbox = face.bbox
    tilt_deg = estimate_face_rotation(face)

    # In OpenCV's getRotationMatrix2D coordinate system:
    # M = [[cos(theta), sin(theta)], [-sin(theta), cos(theta)]]
    # For a vector (dx, dy) with angle theta = atan2(dy, dx),
    # dy' = -sin(theta)*dx + cos(theta)*dy = 0.
    # Therefore, rotating by tilt_deg levels the eye line horizontally.
    rotation_to_straighten = tilt_deg

    # Determine center of rotation in source image
    has_eyes = (
        face.landmarks is not None
        and face.landmarks.right_eye is not None
        and face.landmarks.left_eye is not None
    )

    if str(center_on).strip().lower() == "eyes" and has_eyes:
        re = face.landmarks.right_eye
        le = face.landmarks.left_eye
        src_cx = float(re.x + le.x) / 2.0
        src_cy = float(re.y + le.y) / 2.0
    else:
        src_cx = float(bbox.x + bbox.width / 2.0)
        src_cy = float(bbox.y + bbox.height / 2.0)

    # Calculate padded source face dimensions
    padded_src_w = float(bbox.width) * (1.0 + 2.0 * padding)
    padded_src_h = float(bbox.height) * (1.0 + 2.0 * padding)

    if out_w is not None and out_h is not None:
        # Uniform scaling to preserve aspect ratio without stretching
        scale_x = float(out_w) / max(1.0, padded_src_w)
        scale_y = float(out_h) / max(1.0, padded_src_h)
        uniform_scale = min(scale_x, scale_y)

        # Destination center
        if str(center_on).strip().lower() == "eyes" and has_eyes:
            # Canonical portrait placement: eyes placed at 38% height
            dst_cx = float(out_w) * 0.5
            dst_cy = float(out_h) * 0.38
        else:
            dst_cx = float(out_w) * 0.5
            dst_cy = float(out_h) * 0.5
        dst_w = out_w
        dst_h = out_h
    else:
        uniform_scale = 1.0
        dst_w = int(round(padded_src_w))
        dst_h = int(round(padded_src_h))
        dst_cx = float(dst_w) * 0.5
        dst_cy = float(dst_h) * 0.5

    # Compute affine transformation matrix
    # cv2.getRotationMatrix2D produces rotation around (src_cx, src_cy)
    # Note: cv2.getRotationMatrix2D rotates counter-clockwise for positive angles.
    # In OpenCV image coordinates (+y down), rotating by straightening_deg (-tilt_deg)
    # correctly levels the eye line.
    rot_mat = cv2.getRotationMatrix2D(center=(src_cx, src_cy), angle=rotation_to_straighten, scale=uniform_scale)

    # Adjust translation component so (src_cx, src_cy) maps to (dst_cx, dst_cy)
    # M [src_cx, src_cy, 1]^T currently equals [src_cx, src_cy]^T
    # So we add (dst_cx - src_cx, dst_cy - src_cy)
    rot_mat[0, 2] += (dst_cx - src_cx)
    rot_mat[1, 2] += (dst_cy - src_cy)

    # Warp image with boundary handling
    warped = cv2.warpAffine(
        image,
        rot_mat,
        (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=border_value,
    )

    # Transform landmarks into crop coordinate space
    transformed_5pt = None
    if face.landmarks is not None:
        transformed_5pt = transform_landmarks(face.landmarks, rot_mat)

    transformed_dense = None
    if face.dense_landmarks is not None:
        transformed_dense = transform_dense_landmarks(face.dense_landmarks, rot_mat)

    return AlignedFace(
        image=warped,
        angle=float(round(tilt_deg, 2)),
        rotation_applied=float(round(rotation_to_straighten, 2)),
        center=(float(round(src_cx, 2)), float(round(src_cy, 2))),
        scale=float(round(uniform_scale, 4)),
        transform_matrix=rot_mat,
        original_bbox=bbox,
        face=face,
        landmarks=transformed_5pt,
        dense_landmarks=transformed_dense,
    )


def align_faces(
    image: np.ndarray,
    faces: Sequence[FaceDetection],
    output_size: Optional[Tuple[int, int]] = (112, 112),
    padding: float = 0.25,
    center_on: str = "eyes",
    border_mode: int = cv2.BORDER_REFLECT_101,
    border_value: Tuple[int, int, int] = (0, 0, 0),
) -> List[AlignedFace]:
    """Align multiple detected faces from an image.

    Args:
        image: Original input image array (BGR uint8).
        faces: Sequence of FaceDetection objects.
        output_size: Output crop size tuple (width, height), or None.
        padding: Padding factor around face bounding boxes.
        center_on: 'eyes' or 'face'.
        border_mode: OpenCV border mode.
        border_value: Border fill color if constant border mode.

    Returns:
        List[AlignedFace]: Ordered list of aligned face crops.
    """
    results: List[AlignedFace] = []
    for face in faces:
        aligned = align_face(
            image=image,
            face=face,
            output_size=output_size,
            padding=padding,
            center_on=center_on,
            border_mode=border_mode,
            border_value=border_value,
        )
        results.append(aligned)
    return results


class FaceAligner:
    """Reusable face alignment engine with pre-configured parameters."""

    def __init__(
        self,
        output_size: Optional[Tuple[int, int]] = (112, 112),
        padding: float = 0.25,
        center_on: str = "eyes",
        border_mode: int = cv2.BORDER_REFLECT_101,
        border_value: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.output_size = output_size
        self.padding = padding
        self.center_on = center_on
        self.border_mode = border_mode
        self.border_value = border_value

    def align(self, image: np.ndarray, face: FaceDetection) -> AlignedFace:
        """Align a single face detection."""
        return align_face(
            image=image,
            face=face,
            output_size=self.output_size,
            padding=self.padding,
            center_on=self.center_on,
            border_mode=self.border_mode,
            border_value=self.border_value,
        )

    def align_all(self, image: np.ndarray, faces: Sequence[FaceDetection]) -> List[AlignedFace]:
        """Align all face detections in an image."""
        return align_faces(
            image=image,
            faces=faces,
            output_size=self.output_size,
            padding=self.padding,
            center_on=self.center_on,
            border_mode=self.border_mode,
            border_value=self.border_value,
        )
