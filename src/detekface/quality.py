"""Face quality assessment module for detekface.

Measures technical capture quality (blur/sharpness, illumination, relative face scale,
framing, head-pose suitability, and landmark completeness) and returns structured
actionable warnings and metrics without making subjective judgements.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from detekface.exceptions import (
    ImageNotFoundError,
    InvalidImageError,
)
from detekface.types import (
    BoundingBox,
    DetectionResult,
    FaceDetection,
    HeadPose,
)

QUALITY_DISCLAIMER: str = (
    "Quality metrics assess technical capture conditions (sharpness, illumination, "
    "framing, resolution, and pose), NOT human identity, attractiveness, or personal worth. "
    "Appropriate quality thresholds are strictly application-dependent."
)


@dataclass
class FaceQualityMetrics:
    """Detailed technical quality measurements for an individual face detection.

    Attributes:
        face_index: Zero-based index of this face within the image.
        bbox: Bounding box of the evaluated face.
        sharpness: Laplacian variance score (higher indicates crisper focus).
        is_blurry: True if sharpness score is below the blur threshold.
        brightness: Mean grayscale luminance in range [0, 255].
        brightness_status: Illumination assessment ('underexposed', 'optimal', or 'overexposed').
        face_size_ratio: Fraction of total image area occupied by the face bounding box.
        face_width: Bounding box width in pixels.
        face_height: Bounding box height in pixels.
        is_too_small: True if face dimensions are below minimum recommended resolution.
        framing: Framing evaluation ('centered', 'off_center', or 'truncated').
        is_truncated: True if the face bounding box touches or is clipped by image borders.
        margins: Pixel distances to image boundaries as (left, top, right, bottom).
        head_pose: 3D head pose estimate (yaw, pitch, roll) if available.
        is_frontal: True if face is approximately frontal facing.
        pose_suitability: Descriptive pose status ('frontal', 'turned_left', 'turned_right', 'looking_up', 'looking_down', 'tilted', or 'profile').
        landmark_completeness: Fraction of required facial landmarks detected in range [0.0, 1.0].
        missing_landmarks: List of names of expected landmarks that were not localized.
        warnings: Actionable, objective recommendations to improve capture conditions.
        quality_score: Heuristic technical suitability score between 0.0 and 1.0.
        disclaimer: Explicit ethical and technical disclaimer.
    """

    face_index: int
    bbox: BoundingBox
    sharpness: float
    is_blurry: bool
    brightness: float
    brightness_status: str
    face_size_ratio: float
    face_width: int
    face_height: int
    is_too_small: bool
    framing: str
    is_truncated: bool
    margins: Tuple[int, int, int, int]
    head_pose: Optional[HeadPose]
    is_frontal: bool
    pose_suitability: str
    landmark_completeness: float
    missing_landmarks: List[str]
    warnings: List[str]
    quality_score: float
    disclaimer: str = QUALITY_DISCLAIMER


@dataclass
class ImageQualityResult:
    """Structured assessment report evaluating face capture quality across an image.

    Attributes:
        image_size: Image dimensions as (width, height) in pixels.
        face_count: Number of faces detected.
        faces: Individual quality measurements for each detected face.
        primary_face: Quality measurements for the most prominent (largest) face, or None.
        warnings: Consolidated list of all actionable warnings across all faces.
        overall_score: Aggregate technical quality score (0.0 to 1.0).
        disclaimer: Explicit ethical and technical disclaimer.
    """

    image_size: Tuple[int, int]
    face_count: int
    faces: Sequence[FaceQualityMetrics] = field(default_factory=list)
    primary_face: Optional[FaceQualityMetrics] = None
    warnings: List[str] = field(default_factory=list)
    overall_score: float = 0.0
    disclaimer: str = QUALITY_DISCLAIMER

    def __len__(self) -> int:
        return len(self.faces)

    def __iter__(self):
        return iter(self.faces)

    def __getitem__(self, index: int) -> FaceQualityMetrics:
        return self.faces[index]

    def __bool__(self) -> bool:
        return self.face_count > 0

    def summary(self) -> str:
        """Generate a clean, human-readable summary of the quality analysis."""
        lines = [
            f"Face Quality Report ({self.face_count} face{'s' if self.face_count != 1 else ''} evaluated)",
            f"Image Size: {self.image_size[0]}x{self.image_size[1]} px",
            f"Overall Technical Score: {self.overall_score:.2f} / 1.00",
        ]
        if self.primary_face is not None:
            pf = self.primary_face
            lines.extend(
                [
                    f"- Sharpness: {pf.sharpness:.1f} ({'Blurry' if pf.is_blurry else 'Sharp'})",
                    f"- Brightness: {pf.brightness:.1f} ({pf.brightness_status.capitalize()})",
                    f"- Resolution: {pf.face_width}x{pf.face_height} px ({pf.face_size_ratio * 100:.1f}% of image)",
                    f"- Framing: {pf.framing.capitalize()}{' (Border Clipped)' if pf.is_truncated else ''}",
                    f"- Pose: {pf.pose_suitability.capitalize()}{' (Frontal)' if pf.is_frontal else ''}",
                    f"- Landmarks: {pf.landmark_completeness * 100:.0f}% localized",
                ]
            )

        if self.warnings:
            lines.append("Warnings & Actionable Feedback:")
            for w in self.warnings:
                lines.append(f"  * {w}")
        else:
            lines.append("No technical capture warnings identified.")

        lines.append(f"Note: {self.disclaimer}")
        return "\n".join(lines)


def calculate_sharpness(face_roi_gray: np.ndarray) -> float:
    """Compute focus sharpness using the variance of the Laplacian operator."""
    if face_roi_gray.size == 0 or face_roi_gray.shape[0] < 4 or face_roi_gray.shape[1] < 4:
        return 0.0
    lap = cv2.Laplacian(face_roi_gray, cv2.CV_64F)
    return float(lap.var())


def calculate_brightness(face_roi_gray: np.ndarray) -> float:
    """Compute mean illumination level in range [0, 255]."""
    if face_roi_gray.size == 0:
        return 0.0
    return float(np.mean(face_roi_gray))


def evaluate_pose(face: FaceDetection) -> Tuple[bool, str, Optional[HeadPose]]:
    """Determine head pose orientation suitability from 3D pose or 5-point landmarks."""
    hp = face.head_pose
    if hp is not None:
        yaw = float(hp.yaw)
        pitch = float(hp.pitch)
        roll = float(hp.roll)

        is_frontal = abs(yaw) <= 20.0 and abs(pitch) <= 20.0 and abs(roll) <= 20.0

        if is_frontal:
            status = "frontal"
        elif abs(yaw) > 40.0:
            status = "profile"
        elif yaw > 20.0:
            status = "turned_left"  # subject turned left from viewer's perspective
        elif yaw < -20.0:
            status = "turned_right"
        elif pitch > 20.0:
            status = "looking_up"
        elif pitch < -20.0:
            status = "looking_down"
        elif abs(roll) > 20.0:
            status = "tilted"
        else:
            status = "near_frontal"

        return is_frontal, status, hp

    # Fallback when 3D pose is unavailable: infer from 5-point landmarks
    if face.landmarks is not None:
        re = face.landmarks.right_eye
        le = face.landmarks.left_eye
        nt = face.landmarks.nose_tip
        if re is not None and le is not None and nt is not None:
            # Check horizontal asymmetry of nose relative to eyes
            eye_dist = abs(le.x - re.x)
            if eye_dist > 5:
                d_re_nose = abs(nt.x - re.x)
                d_le_nose = abs(le.x - nt.x)
                asym = abs(d_re_nose - d_le_nose) / eye_dist
                is_frontal = asym < 0.25
                status = "frontal" if is_frontal else "turned"
                return is_frontal, status, None

    return False, "unknown", None


def check_landmark_completeness(face: FaceDetection) -> Tuple[float, List[str]]:
    """Verify presence of key 5-point facial landmarks."""
    expected = ["right_eye", "left_eye", "nose_tip", "right_mouth", "left_mouth"]
    if face.landmarks is None:
        return 0.0, expected

    missing = []
    for name in expected:
        val = getattr(face.landmarks, name, None)
        if val is None:
            missing.append(name)

    present_count = len(expected) - len(missing)
    fraction = float(present_count) / float(len(expected))
    return fraction, missing


def assess_face_quality(
    image: np.ndarray,
    face: FaceDetection,
    face_index: int = 0,
    blur_threshold: float = 80.0,
    min_face_size: int = 48,
    min_area_ratio: float = 0.005,
    min_brightness: float = 50.0,
    max_brightness: float = 200.0,
    border_margin: int = 4,
) -> FaceQualityMetrics:
    """Assess technical capture quality for an individual face detection.

    Args:
        image: Original full image array (BGR uint8).
        face: FaceDetection object containing bounding box and landmarks.
        face_index: Positional index of this face in the image.
        blur_threshold: Laplacian variance threshold below which face is considered blurry.
        min_face_size: Minimum width and height in pixels for high-quality recognition.
        min_area_ratio: Minimum fraction of total image area.
        min_brightness: Minimum mean luminance threshold for adequate illumination.
        max_brightness: Maximum mean luminance threshold before overexposure warning.
        border_margin: Minimum margin in pixels to image edge before truncation warning.

    Returns:
        FaceQualityMetrics: Documented measurements and actionable warnings.
    """
    img_h, img_w = image.shape[:2]
    total_area = float(img_w * img_h)
    bbox = face.bbox

    # 1. Image boundary margins and framing
    margin_left = max(0, int(bbox.x1))
    margin_top = max(0, int(bbox.y1))
    margin_right = max(0, int(img_w - bbox.x2))
    margin_bottom = max(0, int(img_h - bbox.y2))
    margins = (margin_left, margin_top, margin_right, margin_bottom)

    is_truncated = (
        margin_left <= border_margin
        or margin_top <= border_margin
        or margin_right <= border_margin
        or margin_bottom <= border_margin
    )

    center_x = (bbox.x1 + bbox.x2) / 2.0
    center_y = (bbox.y1 + bbox.y2) / 2.0
    img_cx = img_w / 2.0
    img_cy = img_h / 2.0
    off_center_dist = math.hypot(center_x - img_cx, center_y - img_cy)
    max_radius = math.hypot(img_cx, img_cy)
    norm_dist = off_center_dist / max(1.0, max_radius)

    if is_truncated:
        framing = "truncated"
    elif norm_dist < 0.25:
        framing = "centered"
    else:
        framing = "off_center"

    # 2. Extract clamped face ROI for sharpness and illumination analysis
    clamped_bbox = bbox.clamp(max_width=img_w, max_height=img_h)
    x1, y1, x2, y2 = clamped_bbox.x1, clamped_bbox.y1, clamped_bbox.x2, clamped_bbox.y2

    if x2 > x1 and y2 > y1:
        face_roi = image[y1:y2, x1:x2]
        if face_roi.ndim == 3:
            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            face_roi_gray = face_roi
    else:
        face_roi_gray = np.empty((0, 0), dtype=np.uint8)

    # 3. Blur / Sharpness
    sharpness = calculate_sharpness(face_roi_gray)
    is_blurry = sharpness < blur_threshold

    # 4. Brightness
    brightness = calculate_brightness(face_roi_gray)
    if brightness < min_brightness:
        brightness_status = "underexposed"
    elif brightness > max_brightness:
        brightness_status = "overexposed"
    else:
        brightness_status = "optimal"

    # 5. Face size relative to image
    face_area = float(bbox.width * bbox.height)
    face_size_ratio = face_area / max(1.0, total_area)
    is_too_small = (
        bbox.width < min_face_size
        or bbox.height < min_face_size
        or face_size_ratio < min_area_ratio
    )

    # 6. Pose suitability
    is_frontal, pose_suitability, head_pose = evaluate_pose(face)

    # 7. Landmark completeness
    landmark_completeness, missing_landmarks = check_landmark_completeness(face)

    # 8. Compile actionable warnings
    warnings: List[str] = []
    if is_blurry:
        warnings.append(
            f"Face appears soft or blurry (sharpness score {sharpness:.1f} < threshold {blur_threshold:.1f}). "
            "Consider holding the camera steady or improving lighting."
        )
    if brightness_status == "underexposed":
        warnings.append(
            f"Face is underexposed (luminance {brightness:.1f} < {min_brightness:.1f}). "
            "Consider facing toward a direct light source."
        )
    elif brightness_status == "overexposed":
        warnings.append(
            f"Face is overexposed / washed out (luminance {brightness:.1f} > {max_brightness:.1f}). "
            "Consider reducing harsh backlighting or glare."
        )
    if is_too_small:
        warnings.append(
            f"Face resolution is low ({bbox.width}x{bbox.height} px, {face_size_ratio * 100:.2f}% of frame). "
            f"Recommended minimum is {min_face_size}x{min_face_size} px for reliable processing."
        )
    if is_truncated:
        clipped_sides = []
        if margin_left <= border_margin:
            clipped_sides.append("left")
        if margin_top <= border_margin:
            clipped_sides.append("top")
        if margin_right <= border_margin:
            clipped_sides.append("right")
        if margin_bottom <= border_margin:
            clipped_sides.append("bottom")
        warnings.append(
            f"Face is clipped by image boundary on {', '.join(clipped_sides)} side(s). "
            "Reframe subject fully within the camera view."
        )
    if not is_frontal:
        if head_pose is not None:
            warnings.append(
                f"Head pose is angled ({pose_suitability}: yaw={head_pose.yaw:.1f}°, pitch={head_pose.pitch:.1f}°, roll={head_pose.roll:.1f}°). "
                "Frontal head orientation is recommended for consistent recognition."
            )
        else:
            warnings.append(
                f"Head pose appears non-frontal ({pose_suitability}). "
                "Facing directly toward the camera is recommended."
            )
    if missing_landmarks:
        warnings.append(
            f"Facial landmarks are incomplete ({len(missing_landmarks)} missing: {', '.join(missing_landmarks)})."
        )

    # 9. Compute composite technical suitability score (0.0 to 1.0)
    sharp_subscore = min(1.0, sharpness / max(1.0, blur_threshold * 1.5))
    bright_subscore = 1.0 - min(1.0, abs(brightness - 128.0) / 128.0)
    size_subscore = min(1.0, bbox.width / max(1.0, float(min_face_size * 2)))
    framing_subscore = 0.4 if is_truncated else (1.0 if framing == "centered" else 0.8)
    pose_subscore = 1.0 if is_frontal else 0.6
    lm_subscore = landmark_completeness

    score = (
        0.25 * sharp_subscore
        + 0.20 * bright_subscore
        + 0.20 * size_subscore
        + 0.15 * framing_subscore
        + 0.10 * pose_subscore
        + 0.10 * lm_subscore
    )

    return FaceQualityMetrics(
        face_index=face_index,
        bbox=bbox,
        sharpness=float(round(sharpness, 2)),
        is_blurry=is_blurry,
        brightness=float(round(brightness, 2)),
        brightness_status=brightness_status,
        face_size_ratio=float(round(face_size_ratio, 4)),
        face_width=int(bbox.width),
        face_height=int(bbox.height),
        is_too_small=is_too_small,
        framing=framing,
        is_truncated=is_truncated,
        margins=margins,
        head_pose=head_pose,
        is_frontal=is_frontal,
        pose_suitability=pose_suitability,
        landmark_completeness=float(round(landmark_completeness, 2)),
        missing_landmarks=missing_landmarks,
        warnings=warnings,
        quality_score=float(round(score, 3)),
    )


def check_face_quality(
    image: Union[str, Path, np.ndarray, DetectionResult],
    confidence_threshold: float = 0.7,
    backend: str = "yunet",
    blur_threshold: float = 80.0,
    min_face_size: int = 48,
    min_area_ratio: float = 0.005,
    min_brightness: float = 50.0,
    max_brightness: float = 200.0,
    border_margin: int = 4,
) -> ImageQualityResult:
    """Assess face capture quality for an image or DetectionResult.

    Measures sharpness, illumination, face size relative to the frame, border framing,
    pose suitability, and landmark completeness.

    Args:
        image: Path to image file (str or Path), BGR NumPy array, or pre-computed DetectionResult.
        confidence_threshold: Minimum detection confidence score (0.0 to 1.0). Default 0.7.
        backend: Face detector backend ('yunet' or 'mediapipe'). Default 'yunet'.
        blur_threshold: Laplacian variance cutoff for sharpness warning (default 80.0).
        min_face_size: Minimum pixel dimension for face size warning (default 48).
        min_area_ratio: Minimum face area relative to image (default 0.005 = 0.5%).
        min_brightness: Minimum mean luminance cutoff (default 50.0).
        max_brightness: Maximum mean luminance cutoff (default 200.0).
        border_margin: Boundary padding in pixels for truncation check (default 4).

    Returns:
        ImageQualityResult: Structured report containing individual face metrics and actionable warnings.

    Raises:
        ImageNotFoundError: If the image file path does not exist.
        InvalidImageError: If the input image format or array is invalid.
    """
    if isinstance(image, DetectionResult):
        det_result = image
        img_array = det_result.image
    else:
        # Load or validate image array
        if isinstance(image, (str, Path)):
            path_obj = Path(image)
            if not path_obj.exists():
                raise ImageNotFoundError(
                    message=f"Image file not found at: '{path_obj}'.",
                    hint="Verify that the file path is correct and exists on disk.",
                )
            img_array = cv2.imread(str(path_obj))
            if img_array is None:
                raise InvalidImageError(
                    message=f"Failed to read image at '{path_obj}'.",
                    hint="Ensure the file is a valid and uncorrupted image.",
                )
        elif isinstance(image, np.ndarray):
            if image.size == 0 or image.ndim < 2:
                raise InvalidImageError(
                    message="Cannot assess quality of an empty or invalid NumPy image array.",
                    hint="Verify that the image array contains valid pixel data.",
                )
            img_array = image
        else:
            raise InvalidImageError(
                message=f"Unsupported image input type: {type(image)}.",
                hint="Pass a file path (str or Path), NumPy array (BGR), or DetectionResult.",
            )

        from detekface import detect

        det_result = detect(
            image=img_array,
            confidence_threshold=confidence_threshold,
            backend=backend,
            estimate_pose=True,
        )

    img_h, img_w = img_array.shape[:2]
    faces = det_result.faces

    if len(faces) == 0:
        return ImageQualityResult(
            image_size=(img_w, img_h),
            face_count=0,
            faces=[],
            primary_face=None,
            warnings=["No faces detected in the provided image."],
            overall_score=0.0,
        )

    assessed_faces: List[FaceQualityMetrics] = []
    all_warnings: List[str] = []

    for idx, face in enumerate(faces):
        m = assess_face_quality(
            image=img_array,
            face=face,
            face_index=idx,
            blur_threshold=blur_threshold,
            min_face_size=min_face_size,
            min_area_ratio=min_area_ratio,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
            border_margin=border_margin,
        )
        assessed_faces.append(m)
        for w in m.warnings:
            all_warnings.append(f"Face #{idx + 1}: {w}" if len(faces) > 1 else w)

    # Primary face is largest face by bounding box area
    primary = max(assessed_faces, key=lambda f: f.bbox.area)
    avg_score = float(np.mean([f.quality_score for f in assessed_faces]))

    return ImageQualityResult(
        image_size=(img_w, img_h),
        face_count=len(assessed_faces),
        faces=assessed_faces,
        primary_face=primary,
        warnings=all_warnings,
        overall_score=float(round(avg_score, 3)),
    )
