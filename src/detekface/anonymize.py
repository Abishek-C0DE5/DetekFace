"""Face privacy and anonymization module for detekface.

Provides robust, local, offline face anonymization using blur, pixelation, or solid masking,
with configurable strength, aspect-preserving canvas protection, strict source file overwrite
protection, and fail-closed privacy safety mechanisms.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from detekface.drawing import parse_color_to_bgr
from detekface.exceptions import (
    ImageNotFoundError,
    ImageSaveError,
    InvalidImageError,
    InvalidSettingError,
)
from detekface.types import (
    BoundingBox,
    DetectionResult,
    FaceDetection,
    _validate_and_prepare_save_path,
)

PRIVACY_DISCLAIMER: str = (
    "Automated face detection is heuristic and probabilistic; algorithms may fail to "
    "detect faces due to extreme angles, motion blur, partial occlusions, poor lighting, "
    "or low sensor resolution. Automated anonymization cannot guarantee 100% complete "
    "privacy protection in high-risk scenarios. For privacy-critical applications, consider "
    "manual verification or enabling fail_closed=True."
)

SUPPORTED_ANONYMIZE_METHODS = {
    "blur",
    "gaussian",
    "gaussian_blur",
    "pixelate",
    "pixelation",
    "mosaic",
    "solid",
    "mask",
    "censor",
    "blackout",
}


def normalize_anonymize_method(method: str) -> str:
    """Normalize user-supplied method string to 'blur', 'pixelate', or 'solid'."""
    clean = str(method).strip().lower().replace("-", "_")
    if clean in {"blur", "gaussian", "gaussian_blur"}:
        return "blur"
    if clean in {"pixelate", "pixelation", "mosaic"}:
        return "pixelate"
    if clean in {"solid", "mask", "censor", "blackout"}:
        return "solid"

    raise InvalidSettingError(
        message=f"Unsupported anonymization method: '{method}'.",
        hint=f"Supported methods are: 'blur', 'pixelate', 'solid' (or aliases {sorted(SUPPORTED_ANONYMIZE_METHODS)}).",
    )


@dataclass
class AnonymizationResult:
    """Structured result of a face anonymization operation.

    Attributes:
        image: The anonymized BGR uint8 NumPy array (dimensions match original).
        original_image: Unmodified original BGR uint8 NumPy array.
        faces_anonymized: Total count of faces successfully masked/anonymized.
        face_boxes: List of BoundingBox regions that were anonymized.
        method: Normalized anonymization method used ('blur', 'pixelate', 'solid').
        fail_closed_triggered: True if fail-closed mechanism anonymized the full frame.
        source_path: Path to the source image file if loaded from disk, otherwise None.
        path: Path where the anonymized image was written, or None if kept in memory.
        disclaimer: Explicit privacy and detection limitation disclaimer.
    """

    image: np.ndarray
    original_image: np.ndarray
    faces_anonymized: int
    face_boxes: List[BoundingBox] = field(default_factory=list)
    method: str = "blur"
    fail_closed_triggered: bool = False
    source_path: Optional[Path] = None
    path: Optional[Path] = None
    disclaimer: str = PRIVACY_DISCLAIMER

    @property
    def width(self) -> int:
        """Width of the image in pixels."""
        return int(self.image.shape[1]) if isinstance(self.image, np.ndarray) and self.image.ndim >= 2 else 0

    @property
    def height(self) -> int:
        """Height of the image in pixels."""
        return int(self.image.shape[0]) if isinstance(self.image, np.ndarray) and self.image.ndim >= 2 else 0

    @property
    def rgb_image(self) -> np.ndarray:
        """The anonymized image converted to RGB format."""
        if not isinstance(self.image, np.ndarray) or self.image.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def save(self, path: Optional[Union[str, Path]] = None, overwrite: bool = False) -> Path:
        """Save the anonymized image to disk with strict source-file overwrite protection.

        Args:
            path: Destination file path. If None, uses self.path or raises an error.
            overwrite: Must be True to overwrite an existing file, especially the source image.

        Returns:
            Resolved Path of the saved image.

        Raises:
            InvalidSettingError: If path is missing or attempting to overwrite source without overwrite=True.
            ImageSaveError: If saving fails.
        """
        target = path or self.path
        if target is None:
            raise InvalidSettingError(
                message="No destination path specified to save the anonymized image.",
                hint="Pass a valid file path to save(), e.g. result.save('anonymized.jpg').",
            )

        resolved_target = _validate_and_prepare_save_path(target)

        # Never overwrite source file by default
        if self.source_path is not None:
            try:
                is_same_file = resolved_target.resolve() == self.source_path.resolve()
            except Exception:
                is_same_file = False

            if is_same_file and not overwrite:
                raise ImageSaveError(
                    message=f"Refusing to overwrite original source file: '{resolved_target}'.",
                    hint="Specify a different output path, or pass overwrite=True if overwriting is intended.",
                )

        if resolved_target.exists() and not overwrite:
            raise ImageSaveError(
                message=f"Target file already exists: '{resolved_target}'.",
                hint="Specify a different filename or pass overwrite=True to replace it.",
            )

        success = cv2.imwrite(str(resolved_target), self.image)
        if not success:
            raise ImageSaveError(
                message=f"OpenCV failed to write anonymized image to: '{resolved_target}'.",
                hint="Verify directory permissions and valid image format.",
            )

        self.path = resolved_target
        return resolved_target

    def __repr__(self) -> str:
        fc = "full image [FAIL-CLOSED]" if self.fail_closed_triggered else f"{self.faces_anonymized} face(s)"
        return f"<AnonymizationResult: {fc} anonymized using '{self.method}'>"


def apply_blur_roi(
    roi: np.ndarray,
    strength: float = 0.5,
    kernel_size: Optional[int] = None,
) -> np.ndarray:
    """Apply Gaussian blur to an image ROI."""
    h, w = roi.shape[:2]
    if h <= 0 or w <= 0:
        return roi

    if kernel_size is not None:
        k = int(kernel_size)
        if k <= 0:
            raise InvalidSettingError(
                message=f"kernel_size must be positive, got {kernel_size}.",
                hint="Specify an odd integer >= 3, e.g. kernel_size=25.",
            )
        if k % 2 == 0:
            k += 1
    else:
        # Dynamic kernel size proportional to face ROI dimensions and strength
        dim = max(h, w)
        # strength in [0.0, 1.0] maps to [3, dim]
        k = int(round(3 + (dim - 3) * np.clip(strength, 0.05, 1.0)))
        if k % 2 == 0:
            k += 1
        k = max(3, k)

    sigma = k / 3.0
    return cv2.GaussianBlur(roi, (k, k), sigma)


def apply_pixelate_roi(
    roi: np.ndarray,
    strength: float = 0.5,
    pixel_size: Optional[int] = None,
) -> np.ndarray:
    """Apply mosaic pixelation to an image ROI."""
    h, w = roi.shape[:2]
    if h <= 0 or w <= 0:
        return roi

    if pixel_size is not None:
        block = int(pixel_size)
        if block <= 0:
            raise InvalidSettingError(
                message=f"pixel_size must be positive, got {pixel_size}.",
                hint="Specify an integer >= 1, e.g. pixel_size=12.",
            )
    else:
        # Block size proportional to face dimensions and strength
        dim = min(h, w)
        # strength 0.05 to 1.0 maps block size from 2 up to dim // 2
        max_block = max(2, dim // 2)
        block = max(2, int(round(2 + (max_block - 2) * np.clip(strength, 0.05, 1.0))))

    # Downscale then upscale with nearest-neighbor interpolation
    down_w = max(1, w // block)
    down_h = max(1, h // block)
    small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_solid_roi(
    roi: np.ndarray,
    color: Union[str, Tuple[int, int, int]] = (0, 0, 0),
) -> np.ndarray:
    """Fill an image ROI with a solid BGR color."""
    bgr_color = parse_color_to_bgr(color)
    solid = np.zeros_like(roi)
    solid[:] = bgr_color
    return solid


def anonymize_roi(
    roi: np.ndarray,
    method: str = "blur",
    strength: float = 0.5,
    kernel_size: Optional[int] = None,
    pixel_size: Optional[int] = None,
    color: Union[str, Tuple[int, int, int]] = (0, 0, 0),
    mask_shape: str = "box",
) -> np.ndarray:
    """Anonymize a single ROI using the specified method and mask shape."""
    norm_method = normalize_anonymize_method(method)
    if norm_method == "blur":
        processed = apply_blur_roi(roi, strength=strength, kernel_size=kernel_size)
    elif norm_method == "pixelate":
        processed = apply_pixelate_roi(roi, strength=strength, pixel_size=pixel_size)
    else:  # solid
        processed = apply_solid_roi(roi, color=color)

    clean_shape = str(mask_shape).strip().lower()
    if clean_shape in {"ellipse", "oval", "circle"}:
        h, w = roi.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
        result = roi.copy()
        result[mask > 0] = processed[mask > 0]
        return result

    return processed


def anonymize_frame(
    image: np.ndarray,
    faces: Sequence[FaceDetection],
    method: str = "blur",
    strength: float = 0.5,
    kernel_size: Optional[int] = None,
    pixel_size: Optional[int] = None,
    color: Union[str, Tuple[int, int, int]] = (0, 0, 0),
    mask_shape: str = "box",
    padding: float = 0.0,
    copy: bool = True,
) -> np.ndarray:
    """Anonymize detected faces in an image array in place or on a copy.

    Args:
        image: Original BGR uint8 NumPy image array.
        faces: Sequence of FaceDetection objects.
        method: Anonymization technique ('blur', 'pixelate', 'solid').
        strength: Normalized intensity in [0.0, 1.0].
        kernel_size: Optional explicit odd kernel size for blur.
        pixel_size: Optional explicit block pixel size for pixelation.
        color: Solid mask color (name, hex string, or RGB tuple). Default black.
        mask_shape: 'box' for bounding rectangle or 'ellipse' for inscribed oval.
        padding: Fractional padding margin around bounding box (e.g. 0.1 adds 10%).
        copy: Whether to clone the image before modifying.

    Returns:
        np.ndarray: Modified BGR image array with anonymized face regions.
    """
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim < 2:
        raise InvalidImageError(
            message="Cannot anonymize an invalid or empty image array.",
            hint="Provide a valid 2D/3D uint8 NumPy array.",
        )

    out = image.copy() if copy else image
    img_h, img_w = out.shape[:2]

    for face in faces:
        bbox = face.bbox
        if padding > 0.0:
            pad_w = int(round(bbox.width * padding))
            pad_h = int(round(bbox.height * padding))
            x1 = max(0, int(bbox.x1) - pad_w)
            y1 = max(0, int(bbox.y1) - pad_h)
            x2 = min(img_w, int(bbox.x2) + pad_w)
            y2 = min(img_h, int(bbox.y2) + pad_h)
        else:
            x1 = max(0, int(bbox.x1))
            y1 = max(0, int(bbox.y1))
            x2 = min(img_w, int(bbox.x2))
            y2 = min(img_h, int(bbox.y2))

        if x2 > x1 and y2 > y1:
            roi = out[y1:y2, x1:x2]
            out[y1:y2, x1:x2] = anonymize_roi(
                roi=roi,
                method=method,
                strength=strength,
                kernel_size=kernel_size,
                pixel_size=pixel_size,
                color=color,
                mask_shape=mask_shape,
            )

    return out


def anonymize(
    image: Union[str, Path, np.ndarray, DetectionResult],
    method: str = "blur",
    strength: float = 0.5,
    kernel_size: Optional[int] = None,
    pixel_size: Optional[int] = None,
    color: Union[str, Tuple[int, int, int]] = (0, 0, 0),
    mask_shape: str = "box",
    padding: float = 0.0,
    confidence_threshold: float = 0.7,
    backend: str = "yunet",
    fail_closed: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> AnonymizationResult:
    """Anonymize visible faces in an image using blur, pixelation, or solid masking.

    Processes images locally and offline. Preserves original dimensions without modifying
    source files unless explicitly requested.

    Args:
        image: Path to an image file (str or Path), NumPy array (BGR uint8), or DetectionResult.
        method: Anonymization technique ('blur', 'pixelate', 'solid'). Default is 'blur'.
        strength: Normalized intensity (0.0 to 1.0). Default is 0.5.
        kernel_size: Optional explicit odd blur kernel size in pixels.
        pixel_size: Optional explicit pixelation mosaic block size in pixels.
        color: Solid mask color name, hex code, or RGB tuple. Default is black (0, 0, 0).
        mask_shape: Mask geometry: 'box' for bounding rectangle or 'ellipse' for inscribed oval.
        padding: Fractional expansion margin around detected faces (e.g. 0.1 for 10% padding).
        confidence_threshold: Minimum detection confidence score (0.0 to 1.0). Default is 0.7.
        backend: Detection backend ('yunet' or 'mediapipe'). Default is 'yunet'.
        fail_closed: If True, anonymizes the entire image canvas if face detection fails or detects 0 faces.
        output_path: Optional destination path to save the anonymized image.
        overwrite: If True, allows writing to output_path even if it matches the input file.

    Returns:
        AnonymizationResult: Structured report containing anonymized image and metadata.

    Raises:
        ImageNotFoundError: If the input image file does not exist.
        InvalidImageError: If the image cannot be decoded or is empty.
        InvalidSettingError: If parameters are invalid.
        ImageSaveError: If output_path matches the source file and overwrite=False.
    """
    source_path_obj: Optional[Path] = None
    norm_method = normalize_anonymize_method(method)

    if padding < 0.0:
        raise InvalidSettingError(
            message=f"padding must be non-negative, got {padding}.",
            hint="Specify padding >= 0.0, e.g. padding=0.1 for 10% margin.",
        )

    # 1. Resolve image input
    if isinstance(image, DetectionResult):
        img_array = image.image
        det_result = image
        source_path_obj = None
    elif isinstance(image, (str, Path)):
        source_path_obj = Path(image)
        if not source_path_obj.exists():
            raise ImageNotFoundError(
                message=f"Image file not found at: '{source_path_obj}'.",
                hint="Verify that the file path is correct and exists on disk.",
            )
        img_array = cv2.imread(str(source_path_obj))
        if img_array is None:
            raise InvalidImageError(
                message=f"Failed to decode image from: '{source_path_obj}'.",
                hint="Ensure the file is a valid, uncorrupted image.",
            )
        det_result = None
    elif isinstance(image, np.ndarray):
        if image.size == 0 or image.ndim < 2:
            raise InvalidImageError(
                message="Cannot anonymize an empty or invalid NumPy image array.",
                hint="Verify that the image array contains valid pixel data.",
            )
        img_array = image
        det_result = None
    else:
        raise InvalidImageError(
            message=f"Unsupported image input type: {type(image)}.",
            hint="Pass a file path (str or Path), NumPy array (BGR), or DetectionResult.",
        )

    clean_original = img_array.copy()

    # 2. Perform detection if not pre-computed
    detection_failed = False
    if det_result is None:
        try:
            from detekface import detect

            det_result = detect(
                image=clean_original,
                confidence_threshold=confidence_threshold,
                backend=backend,
            )
        except Exception:
            detection_failed = True
            det_result = None

    faces = det_result.faces if det_result is not None else []

    # 3. Handle fail-closed behavior
    should_fail_closed = fail_closed and (detection_failed or len(faces) == 0)

    if should_fail_closed:
        # Privacy-critical fallback: anonymize the entire image canvas
        anonymized_img = anonymize_roi(
            roi=clean_original.copy(),
            method=norm_method,
            strength=strength,
            kernel_size=kernel_size,
            pixel_size=pixel_size,
            color=color,
            mask_shape="box",
        )
        res = AnonymizationResult(
            image=anonymized_img,
            original_image=clean_original,
            faces_anonymized=0,
            face_boxes=[],
            method=norm_method,
            fail_closed_triggered=True,
            source_path=source_path_obj,
        )
    elif detection_failed:
        # Detection failed and fail_closed=False -> raise error
        raise InvalidImageError(
            message="Face detection failed during privacy anonymization.",
            hint="Verify model files and input image or enable fail_closed=True to protect the frame.",
        )
    else:
        # Standard anonymization of detected faces
        anonymized_img = anonymize_frame(
            image=clean_original,
            faces=faces,
            method=norm_method,
            strength=strength,
            kernel_size=kernel_size,
            pixel_size=pixel_size,
            color=color,
            mask_shape=mask_shape,
            padding=padding,
            copy=True,
        )
        face_boxes = [f.bbox for f in faces]
        res = AnonymizationResult(
            image=anonymized_img,
            original_image=clean_original,
            faces_anonymized=len(faces),
            face_boxes=face_boxes,
            method=norm_method,
            fail_closed_triggered=False,
            source_path=source_path_obj,
        )

    # 4. Optional output saving with overwrite protection
    if output_path is not None:
        res.save(output_path, overwrite=overwrite)

    return res

