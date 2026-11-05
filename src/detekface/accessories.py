"""AR face accessories and virtual props rendering engine for detekface 0.3.0.

Provides modular augmented-reality 2D/3D accessories anchored to facial landmarks:
1. Sunglasses (anchored to eyes and nose bridge)
2. Crown (anchored to forehead / upper cranium)
3. Face mask (anchored to nose bridge, cheeks, and chin)
4. Futuristic visor (anchored to eye line and temples with cybernetic glow)
5. Halo (floating angel ring above head with radial aura)

Supports custom transparent PNG assets, automatic scaling, rotation, subpixel
alignment, temporal smoothing across frames, and multiple faces.
"""

from pathlib import Path
import math
from typing import Dict, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from detekface.exceptions import ImageNotFoundError, InvalidImageError, InvalidSettingError
from detekface.types import FaceDetection

# Canonical Accessory Identifiers
ACCESSORY_SUNGLASSES = "sunglasses"
ACCESSORY_CROWN = "crown"
ACCESSORY_FACE_MASK = "face_mask"
ACCESSORY_FUTURISTIC_VISOR = "futuristic_visor"
ACCESSORY_HALO = "halo"

SUPPORTED_ACCESSORIES: Tuple[str, ...] = (
    ACCESSORY_SUNGLASSES,
    ACCESSORY_CROWN,
    ACCESSORY_FACE_MASK,
    ACCESSORY_FUTURISTIC_VISOR,
    ACCESSORY_HALO,
)

# Friendly aliases mapping to canonical accessory names
ACCESSORY_ALIASES: Dict[str, str] = {
    "sunglasses": ACCESSORY_SUNGLASSES,
    "glasses": ACCESSORY_SUNGLASSES,
    "shades": ACCESSORY_SUNGLASSES,
    "crown": ACCESSORY_CROWN,
    "tiara": ACCESSORY_CROWN,
    "royal_crown": ACCESSORY_CROWN,
    "face_mask": ACCESSORY_FACE_MASK,
    "mask": ACCESSORY_FACE_MASK,
    "surgical_mask": ACCESSORY_FACE_MASK,
    "medical_mask": ACCESSORY_FACE_MASK,
    "futuristic_visor": ACCESSORY_FUTURISTIC_VISOR,
    "visor": ACCESSORY_FUTURISTIC_VISOR,
    "cyber_visor": ACCESSORY_FUTURISTIC_VISOR,
    "cyberpunk_visor": ACCESSORY_FUTURISTIC_VISOR,
    "halo": ACCESSORY_HALO,
    "angel_halo": ACCESSORY_HALO,
    "head_accessory": ACCESSORY_HALO,
}

_ASSET_CACHE: Dict[str, np.ndarray] = {}
_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "accessories"


def normalize_accessory_name(name: str) -> Optional[str]:
    """Normalize accessory name or alias to canonical form."""
    clean = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return ACCESSORY_ALIASES.get(clean)


def get_accessory_asset_path(name: str) -> Path:
    """Return local filesystem path to bundled accessory PNG."""
    canonical = normalize_accessory_name(name) or name
    return _ASSET_DIR / f"{canonical}.png"


def load_accessory_asset(source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load and validate an RGBA transparent PNG accessory image.

    Args:
        source: Canonical name, filesystem path, or RGBA NumPy array.

    Returns:
        np.ndarray: RGBA uint8 image array (height x width x 4).

    Raises:
        InvalidSettingError: If the accessory name or asset format is invalid.
    """
    if isinstance(source, np.ndarray):
        if source.ndim != 3 or source.shape[2] != 4:
            raise InvalidSettingError(
                message=f"Custom accessory array must be RGBA (4 channels), got shape {source.shape}.",
                hint="Provide a valid 4-channel RGBA NumPy array with an alpha channel.",
            )
        return source

    cache_key = str(source)
    if cache_key in _ASSET_CACHE:
        return _ASSET_CACHE[cache_key]

    target_path = Path(source)
    if not target_path.exists():
        # Check bundled assets
        canonical = normalize_accessory_name(str(source))
        if canonical is None:
            if isinstance(source, Path) or any(str(source).lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")):
                raise ImageNotFoundError(
                    message=f"Accessory image file not found: '{source}'.",
                    hint="Verify that the path to the accessory image exists.",
                )
            raise InvalidSettingError(
                message=f"Unknown accessory: '{source}'.",
                hint=f"Supported accessories are: {', '.join(sorted(SUPPORTED_ACCESSORIES))}.",
            )
        target_path = get_accessory_asset_path(canonical)

    if not target_path.exists():
        raise InvalidSettingError(
            message=f"Accessory asset not found at '{target_path}'.",
            hint="Verify the asset file exists or reinstall the package.",
        )

    img = cv2.imread(str(target_path), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        raise InvalidSettingError(
            message=f"Accessory asset at '{target_path}' is not a valid 4-channel RGBA image.",
            hint="Provide a PNG image with a transparency alpha channel.",
        )

    _ASSET_CACHE[cache_key] = img
    return img


def compute_accessory_transform(
    face: FaceDetection,
    accessory: str,
    image_shape: Tuple[int, int],
    scale_adjust: float = 1.0,
    y_offset: float = 0.0,
) -> Optional[Tuple[float, float, float, float]]:
    """Compute (center_x, center_y, scale, angle_deg) transformation for an accessory.

    Args:
        face: FaceDetection containing bounding box, landmarks, and head pose.
        accessory: Canonical accessory identifier.
        image_shape: Dimensions of the destination image (height, width).
        scale_adjust: Multiplicative user scale adjustment factor (default 1.0).
        y_offset: Vertical offset adjustment factor relative to face height (default 0.0).

    Returns:
        Optional[Tuple[float, float, float, float]]: (cx, cy, scale, angle_deg) or None if unreliable.
    """
    bbox = face.bbox
    if bbox.width <= 4 or bbox.height <= 4:
        return None

    # Resolve rotation angle (degrees)
    # Eye landmark slope directly reflects the physical eye line tilt in image coordinates
    angle_deg = 0.0
    if face.landmarks is not None and face.landmarks.left_eye and face.landmarks.right_eye:
        le = face.landmarks.left_eye
        re = face.landmarks.right_eye
        # re is viewer's left (smaller x), le is viewer's right (larger x)
        # In image coordinates (y down), positive slope dy requires negative OpenCV angle to tilt right side down
        angle_deg = -math.degrees(math.atan2(le.y - re.y, le.x - re.x))
    elif face.head_pose is not None:
        angle_deg = -face.head_pose.roll

    # Profile view check: if subject is turned away past 55°, skip or attenuate
    if face.head_pose is not None and abs(face.head_pose.yaw) > 55.0:
        return None

    # Face center and dimensions
    face_cx = float(bbox.x + bbox.width / 2.0)
    w_box = float(bbox.width)
    h_box = float(bbox.height)

    # Resolve eye positions if available
    eye_left: Optional[Tuple[float, float]] = None
    eye_right: Optional[Tuple[float, float]] = None
    if face.landmarks is not None and face.landmarks.left_eye and face.landmarks.right_eye:
        eye_left = (float(face.landmarks.left_eye.x), float(face.landmarks.left_eye.y))
        eye_right = (float(face.landmarks.right_eye.x), float(face.landmarks.right_eye.y))
    elif face.dense_landmarks and len(face.dense_landmarks) >= 388:
        dl = face.dense_landmarks
        eye_left = ((dl[263].x + dl[362].x) / 2.0, (dl[263].y + dl[362].y) / 2.0)
        eye_right = ((dl[33].x + dl[133].x) / 2.0, (dl[33].y + dl[133].y) / 2.0)

    # -------------------------------------------------------------------------
    # 1. Sunglasses
    # -------------------------------------------------------------------------
    if accessory == ACCESSORY_SUNGLASSES:
        if eye_left is not None and eye_right is not None:
            cx = (eye_left[0] + eye_right[0]) / 2.0
            cy = (eye_left[1] + eye_right[1]) / 2.0
            eye_dist = math.hypot(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1])
            # Lens distance in 512px asset is 221.5px. Scale directly to inter-pupillary distance:
            scale = (eye_dist / 221.5) * scale_adjust
        else:
            cx = face_cx
            cy = bbox.y + h_box * 0.35
            scale = (w_box * 0.90 / 512.0) * scale_adjust

        cy += y_offset * h_box
        return (cx, cy, scale, angle_deg)

    # -------------------------------------------------------------------------
    # 2. Crown
    # -------------------------------------------------------------------------
    elif accessory == ACCESSORY_CROWN:
        cx = face_cx
        # Forehead / hairline position
        if face.dense_landmarks and len(face.dense_landmarks) >= 11:
            top_y = float(face.dense_landmarks[10].y)
        else:
            top_y = float(bbox.y)

        # Crown width proportional to face width
        scale = (w_box * 0.95 / 512.0) * scale_adjust
        # Crown asset height is 300, base is at y=283, center is at 150 (dist = 133px)
        cy = top_y + (h_box * 0.04) - (130.0 * scale) + (y_offset * h_box)
        return (cx, cy, scale, angle_deg)

    # -------------------------------------------------------------------------
    # 3. Face Mask
    # -------------------------------------------------------------------------
    elif accessory == ACCESSORY_FACE_MASK:
        cx = face_cx
        if face.landmarks is not None and face.landmarks.nose_tip:
            eye_y = (eye_left[1] + eye_right[1]) / 2.0 if (eye_left and eye_right) else bbox.y + h_box * 0.35
            nose_bridge_y = (eye_y + face.landmarks.nose_tip.y) / 2.0
            chin_y = float(bbox.y + h_box)
            desired_height = max(chin_y - nose_bridge_y, h_box * 0.55)
            # Mask asset height is 266px active area
            scale = (desired_height / 266.0) * 1.15 * scale_adjust
            cy = (nose_bridge_y + chin_y) / 2.0 + (y_offset * h_box)
        elif face.dense_landmarks and len(face.dense_landmarks) >= 153:
            nose_bridge_y = float(face.dense_landmarks[6].y)
            chin_y = float(face.dense_landmarks[152].y)
            desired_height = max(chin_y - nose_bridge_y, h_box * 0.55)
            scale = (desired_height / 266.0) * 1.15 * scale_adjust
            cy = (nose_bridge_y + chin_y) / 2.0 + (y_offset * h_box)
        else:
            cy = bbox.y + h_box * 0.70 + (y_offset * h_box)
            scale = (w_box * 0.95 / 512.0) * scale_adjust

        return (cx, cy, scale, angle_deg)

    # -------------------------------------------------------------------------
    # 4. Futuristic Visor
    # -------------------------------------------------------------------------
    elif accessory == ACCESSORY_FUTURISTIC_VISOR:
        if eye_left is not None and eye_right is not None:
            cx = (eye_left[0] + eye_right[0]) / 2.0
            cy = (eye_left[1] + eye_right[1]) / 2.0
            eye_dist = math.hypot(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1])
            # Visor lens distance in 512px asset is 216.5px:
            scale = (eye_dist / 216.5) * scale_adjust
        else:
            cx = face_cx
            cy = bbox.y + h_box * 0.36
            scale = (w_box * 0.95 / 512.0) * scale_adjust

        cy += y_offset * h_box
        return (cx, cy, scale, angle_deg)

    # -------------------------------------------------------------------------
    # 5. Halo
    # -------------------------------------------------------------------------
    elif accessory == ACCESSORY_HALO:
        cx = face_cx
        scale = (w_box * 0.95 / 512.0) * scale_adjust
        cy = float(bbox.y) - (h_box * 0.15) - (35.0 * scale) + (y_offset * h_box)
        eff_angle = angle_deg
        if face.head_pose is not None:
            eff_angle += face.head_pose.pitch * 0.15
        return (cx, cy, scale, eff_angle)

    # Fallback generic head accessory
    else:
        cx = face_cx
        cy = float(bbox.y) + (y_offset * h_box)
        scale = (w_box / 512.0) * scale_adjust
        return (cx, cy, scale, angle_deg)


def overlay_rgba(
    background: np.ndarray,
    overlay: np.ndarray,
    center: Tuple[float, float],
    scale: float,
    angle_deg: float,
    opacity: float = 1.0,
) -> np.ndarray:
    """Affine transform and alpha-blend an RGBA overlay onto a BGR image array.

    Args:
        background: Destination BGR uint8 NumPy image array (modified in-place or returned).
        overlay: Source RGBA uint8 image array (height x width x 4).
        center: Destination (x, y) coordinates where the center of the overlay will land.
        scale: Multiplicative scale factor.
        angle_deg: Rotation angle in degrees (counter-clockwise).
        opacity: Global alpha multiplier (0.0 to 1.0).

    Returns:
        np.ndarray: Blended BGR image.
    """
    if scale <= 1e-4 or opacity <= 1e-4:
        return background

    bg_h, bg_w = background.shape[:2]
    ov_h, ov_w = overlay.shape[:2]

    # Center of source overlay asset
    src_cx = ov_w / 2.0
    src_cy = ov_h / 2.0
    dst_cx, dst_cy = center

    # Construct affine matrix mapping source overlay center directly to destination (dst_cx, dst_cy)
    # R_matrix rotates and scales about (src_cx, src_cy)
    rot_mat = cv2.getRotationMatrix2D((src_cx, src_cy), angle_deg, scale)
    # Translate from (src_cx, src_cy) to (dst_cx, dst_cy)
    rot_mat[0, 2] += dst_cx - src_cx
    rot_mat[1, 2] += dst_cy - src_cy

    # Transform the 4 corners of overlay to destination coordinates to determine minimal bounding ROI
    corners = np.array(
        [[0.0, 0.0], [float(ov_w), 0.0], [float(ov_w), float(ov_h)], [0.0, float(ov_h)]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    dst_corners = cv2.transform(corners, rot_mat).reshape(-1, 2)

    x0 = int(math.floor(float(dst_corners[:, 0].min())))
    y0 = int(math.floor(float(dst_corners[:, 1].min())))
    x1 = int(math.ceil(float(dst_corners[:, 0].max())))
    y1 = int(math.ceil(float(dst_corners[:, 1].max())))

    # Clip to background canvas boundaries
    bx0 = max(0, x0)
    by0 = max(0, y0)
    bx1 = min(bg_w, x1)
    by1 = min(bg_h, y1)

    if bx1 <= bx0 or by1 <= by0:
        return background

    roi_w = bx1 - bx0
    roi_h = by1 - by0

    # Adjust affine matrix to project into local ROI canvas
    local_rot = rot_mat.copy()
    local_rot[0, 2] -= float(bx0)
    local_rot[1, 2] -= float(by0)

    # Warp only into local ROI image bounds (dramatically lowers memory bandwidth & blend operations)
    warped_roi = cv2.warpAffine(
        overlay,
        local_rot,
        (roi_w, roi_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    alpha_channel = warped_roi[:, :, 3]
    if not np.any(alpha_channel > 0):
        return background

    # Vectorized alpha blending strictly inside the accessory ROI
    eff_alpha = (alpha_channel.astype(np.float32) / 255.0) * float(max(0.0, min(1.0, opacity)))
    eff_alpha_3ch = np.dstack([eff_alpha, eff_alpha, eff_alpha])

    foreground_bgr = warped_roi[:, :, :3].astype(np.float32)
    background_roi = background[by0:by1, bx0:bx1].astype(np.float32)

    blended = foreground_bgr * eff_alpha_3ch + background_roi * (1.0 - eff_alpha_3ch)
    background[by0:by1, bx0:bx1] = np.clip(blended, 0, 255).astype(np.uint8)
    return background


class AccessorySmoother:
    """Adaptive temporal smoothing filter eliminating jitter while preserving responsive movement."""

    def __init__(self, alpha: float = 0.35, deadband_px: float = 1.0, adaptive: bool = True) -> None:
        self.alpha = max(0.01, min(1.0, float(alpha)))
        self.deadband_px = float(deadband_px)
        self.adaptive = bool(adaptive)
        self._tracks: Dict[Union[int, str], Tuple[float, float, float, float]] = {}

    def smooth(
        self,
        track_id: Optional[Union[int, str]],
        transform: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        """Apply adaptive temporal smoothing to (cx, cy, scale, angle)."""
        if track_id is None:
            return transform

        cx, cy, s, a = transform
        if track_id not in self._tracks:
            self._tracks[track_id] = transform
            return transform

        prev_cx, prev_cy, prev_s, prev_a = self._tracks[track_id]

        dist = math.hypot(cx - prev_cx, cy - prev_cy)
        # Deadband: if movement is tiny subpixel jitter, hold position steady
        if dist < self.deadband_px:
            cx = prev_cx
            cy = prev_cy

        # Adaptive alpha: smooth during stationary posture, responsive during rapid movement
        if self.adaptive:
            eff_alpha = min(0.85, self.alpha + (dist / 80.0) * 0.5)
        else:
            eff_alpha = self.alpha

        # Handle angle wraparound if needed
        diff_a = a - prev_a
        if diff_a > 180.0:
            diff_a -= 360.0
        elif diff_a < -180.0:
            diff_a += 360.0

        new_cx = eff_alpha * cx + (1.0 - eff_alpha) * prev_cx
        new_cy = eff_alpha * cy + (1.0 - eff_alpha) * prev_cy
        new_s = eff_alpha * s + (1.0 - eff_alpha) * prev_s
        new_a = prev_a + eff_alpha * diff_a

        smoothed = (new_cx, new_cy, new_s, new_a)
        self._tracks[track_id] = smoothed
        return smoothed

    def reset(self) -> None:
        """Clear smoothed history."""
        self._tracks.clear()


class FaceAccessoryRenderer:
    """Renderer for applying AR face accessories and virtual props to images and video frames."""

    def __init__(
        self,
        accessory: Union[str, Path, np.ndarray],
        opacity: float = 1.0,
        scale_adjust: float = 1.0,
        y_offset: float = 0.0,
        smooth: bool = True,
    ) -> None:
        """Initialize FaceAccessoryRenderer.

        Args:
            accessory: Canonical accessory identifier, custom path, or RGBA NumPy array.
            opacity: Opacity multiplier (0.0 to 1.0). Default 1.0.
            scale_adjust: Additional scale multiplier (default 1.0).
            y_offset: Additional vertical offset factor (default 0.0).
            smooth: Whether to apply temporal smoothing across video frames (default True).
        """
        self.accessory_name = str(accessory) if isinstance(accessory, (str, Path)) else "custom"
        self.canonical = normalize_accessory_name(self.accessory_name) or self.accessory_name
        self.asset = load_accessory_asset(accessory)
        self.opacity = float(max(0.0, min(1.0, opacity)))
        self.scale_adjust = float(scale_adjust)
        self.y_offset = float(y_offset)
        self.smoother = AccessorySmoother() if smooth else None

    def reset(self) -> None:
        """Reset temporal smoothing history."""
        if self.smoother is not None:
            self.smoother.reset()

    def render(
        self,
        image: np.ndarray,
        faces: Sequence[FaceDetection],
        copy: bool = True,
    ) -> np.ndarray:
        """Render the configured accessory onto all detected faces in the image.

        Args:
            image: Source BGR image NumPy array.
            faces: Sequence of detected FaceDetection instances.
            copy: If True (default), creates a copy of the image.

        Returns:
            np.ndarray: Annotated BGR image array.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise InvalidImageError(
                message="Cannot render accessory on empty or invalid image array.",
                hint="Provide a valid BGR NumPy uint8 image array.",
            )

        out = image.copy() if copy else image
        if not faces:
            return out

        img_h, img_w = out.shape[:2]

        for idx, face in enumerate(faces):
            transform = compute_accessory_transform(
                face=face,
                accessory=self.canonical,
                image_shape=(img_h, img_w),
                scale_adjust=self.scale_adjust,
                y_offset=self.y_offset,
            )
            if transform is None:
                continue

            track_id = face.track_id if face.track_id is not None else idx
            if self.smoother is not None:
                transform = self.smoother.smooth(track_id, transform)

            cx, cy, scale, angle = transform
            overlay_rgba(
                background=out,
                overlay=self.asset,
                center=(cx, cy),
                scale=scale,
                angle_deg=angle,
                opacity=self.opacity,
            )

        return out


def render_accessory(
    image: np.ndarray,
    faces: Sequence[FaceDetection],
    accessory: Union[str, Path, np.ndarray],
    opacity: float = 1.0,
    scale_adjust: float = 1.0,
    y_offset: float = 0.0,
    copy: bool = True,
) -> np.ndarray:
    """Render an AR face accessory onto detected faces in an image.

    Args:
        image: Source BGR image array.
        faces: Sequence of detected faces.
        accessory: Canonical accessory name ('sunglasses', 'crown', 'face_mask',
            'futuristic_visor', 'halo') or path to custom RGBA PNG.
        opacity: Overlay alpha opacity (0.0 to 1.0). Default is 1.0.
        scale_adjust: Multiplicative size factor (default 1.0).
        y_offset: Vertical shift adjustment (default 0.0).
        copy: Whether to render on a copy of the image (default True).

    Returns:
        np.ndarray: BGR image array with rendered accessory.
    """
    renderer = FaceAccessoryRenderer(
        accessory=accessory,
        opacity=opacity,
        scale_adjust=scale_adjust,
        y_offset=y_offset,
        smooth=False,
    )
    return renderer.render(image, faces, copy=copy)
