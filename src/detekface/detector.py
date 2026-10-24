"""Core face detection engine and detection pipeline."""

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from detekface.exceptions import (
    ConfigurationError,
    ImageNotFoundError,
    InvalidImageError,
)
from detekface.models import resolve_model
from detekface.types import (
    BoundingBox,
    DetectionResult,
    FaceDetection,
    FaceLandmarks,
    Point,
)


def _validate_threshold(name: str, value: float, min_val: float = 0.0, max_val: float = 1.0) -> None:
    """Validate that a threshold is a float within [min_val, max_val]."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigurationError(
            message=f"Threshold '{name}' must be a number, got {type(value).__name__}.",
            hint=f"Provide a float between {min_val} and {max_val} (e.g. 0.7).",
        )
    if not (min_val <= float(value) <= max_val):
        raise ConfigurationError(
            message=f"Threshold '{name}' out of range: {value}.",
            hint=f"The value must be between {min_val} and {max_val}.",
        )


def _validate_input_size(
    input_size: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """Validate optional input_size tuple (width, height)."""
    if input_size is None:
        return None
    if (
        not isinstance(input_size, (tuple, list))
        or len(input_size) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 32 for v in input_size)
    ):
        raise ConfigurationError(
            message=f"Invalid input_size: {input_size}. Expected a tuple of 2 positive integers (width, height) >= 32.",
            hint="Set input_size to a resolution tuple like (640, 480) or (320, 240), or leave as None.",
        )
    return (int(input_size[0]), int(input_size[1]))


def _load_and_validate_image(image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Validate and convert image input to a 3-channel BGR uint8 NumPy array."""
    if isinstance(image_input, (str, Path)):
        path = Path(image_input).expanduser()
        if not path.exists():
            raise ImageNotFoundError(
                message=f"Image file does not exist: {path}",
                hint="Check the filepath for typos or verify the file is accessible.",
            )
        if not path.is_file():
            raise InvalidImageError(
                message=f"Specified path is a directory, not a file: {path}",
                hint="Provide a path to an image file (e.g. .jpg, .png).",
            )
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise InvalidImageError(
                message=f"Failed to decode image from file: {path}",
                hint="Ensure the file is a valid, uncorrupted image (e.g. JPEG, PNG, BMP).",
            )
        return img

    if isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise InvalidImageError(
                message="Image NumPy array is empty (0 elements).",
                hint="Ensure your image array contains image pixel data.",
            )

        # Normalize data type to uint8
        img = image_input
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                if not np.all(np.isfinite(img)):
                    raise InvalidImageError(
                        message="Image array contains NaN or infinite values.",
                        hint="Clean or impute non-finite values before passing to detector.",
                    )
                if np.max(img) <= 1.0:
                    img = (img * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)

        # Normalize channels to 3-channel BGR
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3:
            channels = img.shape[2]
            if channels == 3:
                return img
            if channels == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if channels == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            raise InvalidImageError(
                message=f"Image array has {channels} channels; expected 1, 3 (BGR), or 4 (BGRA).",
                hint="Convert your image array to standard 3-channel BGR format.",
            )

        raise InvalidImageError(
            message=f"Unsupported image array dimensions: {img.ndim}D.",
            hint="Input array must be 2D (grayscale) or 3D (height, width, channels).",
        )

    raise InvalidImageError(
        message=f"Unsupported image input type: '{type(image_input).__name__}'.",
        hint="Pass a valid filepath (str or Path) or a NumPy array in BGR format.",
    )


def _parse_yunet_detections(raw_faces: Optional[np.ndarray]) -> List[FaceDetection]:
    """Parse the raw (N, 15) array returned by cv2.FaceDetectorYN into structured FaceDetection objects."""
    if raw_faces is None or len(raw_faces) == 0:
        return []

    detections: List[FaceDetection] = []
    for row in raw_faces:
        if len(row) < 15:
            continue

        x = int(round(float(row[0])))
        y = int(round(float(row[1])))
        width = int(round(float(row[2])))
        height = int(round(float(row[3])))
        bbox = BoundingBox(x=x, y=y, width=width, height=height)

        # Facial landmarks (right eye, left eye, nose tip, right mouth, left mouth)
        right_eye = Point(x=int(round(float(row[4]))), y=int(round(float(row[5]))))
        left_eye = Point(x=int(round(float(row[6]))), y=int(round(float(row[7]))))
        nose_tip = Point(x=int(round(float(row[8]))), y=int(round(float(row[9]))))
        right_mouth = Point(x=int(round(float(row[10]))), y=int(round(float(row[11]))))
        left_mouth = Point(x=int(round(float(row[12]))), y=int(round(float(row[13]))))

        landmarks = FaceLandmarks(
            right_eye=right_eye,
            left_eye=left_eye,
            nose_tip=nose_tip,
            right_mouth=right_mouth,
            left_mouth=left_mouth,
        )

        confidence = float(row[14])

        detections.append(
            FaceDetection(
                bbox=bbox,
                confidence=confidence,
                landmarks=landmarks,
            )
        )

    return detections


def _scale_point(pt: Optional[Point], scale_x: float, scale_y: float, max_w: int, max_h: int) -> Optional[Point]:
    """Rescale a single landmark point and clamp to image bounds."""
    if pt is None:
        return None
    px = max(0, min(int(round(pt.x * scale_x)), max_w - 1))
    py = max(0, min(int(round(pt.y * scale_y)), max_h - 1))
    return Point(x=px, y=py)


def _transform_detections_to_original_scale(
    faces: List[FaceDetection],
    scale_x: float,
    scale_y: float,
    orig_w: int,
    orig_h: int,
) -> List[FaceDetection]:
    """Rescale detections from inference resolution back to original image resolution."""
    if not faces:
        return []

    scaled: List[FaceDetection] = []
    for face in faces:
        raw_x = face.bbox.x * scale_x
        raw_y = face.bbox.y * scale_y
        raw_w = face.bbox.width * scale_x
        raw_h = face.bbox.height * scale_y

        x = max(0, min(int(round(raw_x)), orig_w - 1))
        y = max(0, min(int(round(raw_y)), orig_h - 1))
        w = max(1, min(int(round(raw_w)), orig_w - x))
        h = max(1, min(int(round(raw_h)), orig_h - y))
        bbox = BoundingBox(x=x, y=y, width=w, height=h)

        lm = face.landmarks
        if lm is not None:
            new_landmarks = FaceLandmarks(
                right_eye=_scale_point(lm.right_eye, scale_x, scale_y, orig_w, orig_h),
                left_eye=_scale_point(lm.left_eye, scale_x, scale_y, orig_w, orig_h),
                nose_tip=_scale_point(lm.nose_tip, scale_x, scale_y, orig_w, orig_h),
                right_mouth=_scale_point(lm.right_mouth, scale_x, scale_y, orig_w, orig_h),
                left_mouth=_scale_point(lm.left_mouth, scale_x, scale_y, orig_w, orig_h),
                mouth_center=_scale_point(lm.mouth_center, scale_x, scale_y, orig_w, orig_h),
                right_ear=_scale_point(lm.right_ear, scale_x, scale_y, orig_w, orig_h),
                left_ear=_scale_point(lm.left_ear, scale_x, scale_y, orig_w, orig_h),
            )
        else:
            new_landmarks = None

        scaled.append(
            FaceDetection(
                bbox=bbox,
                confidence=face.confidence,
                landmarks=new_landmarks,
            )
        )
    return scaled


class FaceDetector:
    """Unified face detector wrapper with automatic model loading and input handling.

    Supports multiple detection backends (default 'yunet', optional 'mediapipe').
    """

    def __init__(
        self,
        model: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        cache_dir: Optional[Union[str, Path]] = None,
        backend: str = "yunet",
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Initialize the FaceDetector.

        Args:
            model: Optional model name, alias, or custom local model path.
            confidence_threshold: Minimum detection confidence score (0.0 to 1.0).
            nms_threshold: Non-maximum suppression IoU threshold (0.0 to 1.0, YuNet only).
            top_k: Maximum candidate faces to retain prior to NMS (YuNet only).
            cache_dir: Optional custom directory for caching models.
            backend: Face detection backend ('yunet' or 'mediapipe'). Default is 'yunet'.
            input_size: Optional fixed resolution (width, height) for model inference.
                When specified, frames are downscaled/upscaled for model inference and
                detected coordinates are mapped accurately back to original image space.
        """
        from detekface.backends import get_backend

        clean_backend = str(backend).strip().lower()
        _validate_threshold("confidence_threshold", confidence_threshold, 0.0, 1.0)
        self._input_size = _validate_input_size(input_size)
        model_to_pass = model
        if clean_backend == "yunet":
            _validate_threshold("nms_threshold", nms_threshold, 0.0, 1.0)
            if not isinstance(top_k, int) or top_k < 1:
                raise ConfigurationError(
                    message=f"top_k must be a positive integer >= 1, got {top_k}.",
                    hint="Set top_k to a value like 5000.",
                )
            model_to_pass = resolve_model(model, cache_dir=cache_dir)

        self._confidence_threshold = float(confidence_threshold)
        self._nms_threshold = float(nms_threshold)
        self._top_k = int(top_k)

        self._backend = get_backend(
            backend=backend,
            model=model_to_pass,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            cache_dir=cache_dir,
        )

        # Expose underlying engine attributes for backward-compatibility with tests & internals
        self._detector = getattr(self._backend, "_detector", None)
        self._model_path = getattr(self._backend, "_model_path", None)

    @property
    def backend_name(self) -> str:
        """Name of the active detection backend ('yunet' or 'mediapipe')."""
        return self._backend.name

    @property
    def backend(self):
        """Active detection backend adapter instance."""
        return self._backend

    @property
    def model_path(self) -> Optional[Path]:
        """The resolved filesystem path of the active model, or None if in-memory."""
        return getattr(self._backend, "model_path", None)

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence threshold for detected faces."""
        return self._confidence_threshold

    @property
    def nms_threshold(self) -> float:
        """IoU threshold for non-maximum suppression."""
        return self._nms_threshold

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        """Configured inference resolution (width, height), or None for native resolution."""
        return self._input_size

    def detect(self, image: Union[str, Path, np.ndarray]) -> DetectionResult:
        """Perform face detection on a single image.

        Args:
            image: A filesystem path to an image file, or a NumPy array in BGR format.

        Returns:
            DetectionResult containing detected FaceDetection instances and the original BGR image.
        """
        bgr_image = _load_and_validate_image(image)
        height, width = bgr_image.shape[:2]

        if height <= 0 or width <= 0:
            raise InvalidImageError(
                message=f"Image dimensions are invalid: {width}x{height}.",
                hint="Ensure the image has non-zero width and height.",
            )

        if self._input_size is not None and (width, height) != self._input_size:
            infer_w, infer_h = self._input_size
            inference_image = cv2.resize(bgr_image, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
            raw_faces = self._backend.detect(inference_image)
            scale_x = width / float(infer_w)
            scale_y = height / float(infer_h)
            faces = _transform_detections_to_original_scale(raw_faces, scale_x, scale_y, orig_w=width, orig_h=height)
        else:
            faces = self._backend.detect(bgr_image)

        return DetectionResult(faces=faces, image=bgr_image)

    def detect_many(
        self,
        images: Union[str, Path, Sequence[Union[str, Path, np.ndarray]]],
        fail_fast: bool = False,
        progress_callback: Optional[Callable] = None,
        recursive: bool = False,
        keep_images: bool = True,
    ):
        """Perform face detection across multiple images or a directory.

        Args:
            images: File path, directory path, NumPy array, or sequence thereof.
            fail_fast: If True, stop and raise immediately upon encountering an error.
            progress_callback: Optional callable invoked after each processed image.
            recursive: If True and directory is passed, recursively scan subdirectories.
            keep_images: If False, drops heavy image arrays from returned results to save memory.

        Returns:
            BatchDetectionResult: List of BatchItemResult objects preserving input ordering.
        """
        from detekface.batch import run_batch_detection

        return run_batch_detection(
            detector=self,
            images=images,
            fail_fast=fail_fast,
            progress_callback=progress_callback,
            recursive=recursive,
            keep_images=keep_images,
        )
