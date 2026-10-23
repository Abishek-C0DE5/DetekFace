"""Core face detection engine and detection pipeline."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from detekface.exceptions import (
    ConfigurationError,
    ImageNotFoundError,
    InvalidImageError,
    ModelCompatibilityError,
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
            if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
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


class FaceDetector:
    """YuNet face detector wrapper with automatic model loading and input handling.

    Provides high-level face detection with structured bounding boxes and facial landmarks.
    """

    def __init__(
        self,
        model: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the FaceDetector.

        Args:
            model: Optional model name, alias, or custom local .onnx path.
            confidence_threshold: Minimum detection confidence score (0.0 to 1.0).
            nms_threshold: Non-maximum suppression IoU threshold (0.0 to 1.0).
            top_k: Maximum number of candidate faces to retain prior to NMS.
            cache_dir: Optional custom directory for caching models.
        """
        _validate_threshold("confidence_threshold", confidence_threshold, 0.0, 1.0)
        _validate_threshold("nms_threshold", nms_threshold, 0.0, 1.0)
        if not isinstance(top_k, int) or top_k < 1:
            raise ConfigurationError(
                message=f"top_k must be a positive integer >= 1, got {top_k}.",
                hint="Set top_k to a value like 5000.",
            )

        self._confidence_threshold = float(confidence_threshold)
        self._nms_threshold = float(nms_threshold)
        self._top_k = int(top_k)

        self._model_path = resolve_model(model, cache_dir=cache_dir)

        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(self._model_path),
                "",
                (320, 320),
                score_threshold=self._confidence_threshold,
                nms_threshold=self._nms_threshold,
                top_k=self._top_k,
            )
        except cv2.error as e:
            raise ModelCompatibilityError(
                message=f"Failed to create OpenCV FaceDetectorYN with model: {self._model_path.name}",
                hint=str(e),
            ) from e

        if self._detector is None:
            raise ModelCompatibilityError(
                message=f"OpenCV returned None when initializing FaceDetectorYN with '{self._model_path.name}'.",
                hint="Verify that the ONNX model architecture matches YuNet specifications.",
            )

    @property
    def model_path(self) -> Path:
        """The resolved filesystem path of the active ONNX model."""
        return self._model_path

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence threshold for detected faces."""
        return self._confidence_threshold

    @property
    def nms_threshold(self) -> float:
        """IoU threshold for non-maximum suppression."""
        return self._nms_threshold

    def detect(self, image: Union[str, Path, np.ndarray]) -> DetectionResult:
        """Perform face detection on a single image.

        Args:
            image: A filesystem path to an image file, or a NumPy array in BGR format.

        Returns:
            DetectionResult containing detected FaceDetection instances and the BGR image array.
        """
        bgr_image = _load_and_validate_image(image)
        height, width = bgr_image.shape[:2]

        if height <= 0 or width <= 0:
            raise InvalidImageError(
                message=f"Image dimensions are invalid: {width}x{height}.",
                hint="Ensure the image has non-zero width and height.",
            )

        # Set input dimensions for inference
        self._detector.setInputSize((width, height))

        retval, raw_faces = self._detector.detect(bgr_image)
        faces = _parse_yunet_detections(raw_faces)

        return DetectionResult(faces=faces, image=bgr_image)
