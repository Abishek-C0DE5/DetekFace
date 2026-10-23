"""detekface - A clean, reusable Python face detection package powered by YuNet and OpenCV.

This package provides beginner-friendly face detection interfaces with sensible defaults,
clean dataclass outputs, and robust error handling.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np

from detekface.camera import camera
from detekface.capture import capture
from detekface.detector import FaceDetector
from detekface.drawing import (
    NAMED_COLORS,
    SUPPORTED_SHAPES,
    draw_face_overlay,
    draw_faces,
    parse_color_to_bgr,
)
from detekface.exceptions import (
    CameraError,
    CameraNotFoundError,
    CameraReadError,
    CaptureError,
    CaptureSaveError,
    ConfigurationError,
    DetekFaceError,
    ImageError,
    ImageNotFoundError,
    InvalidCropError,
    InvalidImageError,
    ModelCompatibilityError,
    ModelCorruptionError,
    ModelDownloadError,
    ModelError,
    ModelNotFoundError,
    NoFaceDetectedError,
)
from detekface.models import (
    OFFICIAL_MODELS,
    ModelMetadata,
    get_cache_dir,
    resolve_model,
    verify_model_initialization,
)
from detekface.types import (
    BoundingBox,
    DetectionResult,
    FaceCapture,
    FaceDetection,
    FaceLandmarks,
    Point,
)

__version__: str = "0.1.0"

_default_detector: Optional[FaceDetector] = None


def detect(
    image: Union[str, Path, np.ndarray],
    confidence_threshold: float = 0.7,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
    model: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> DetectionResult:
    """Detect faces in a still image.

    Args:
        image: A filesystem path to an image file (str or Path), or a NumPy array in BGR format.
        confidence_threshold: Minimum detection confidence score (0.0 to 1.0). Default is 0.7.
        nms_threshold: Non-maximum suppression IoU threshold (0.0 to 1.0). Default is 0.3.
        top_k: Maximum candidate faces to retain prior to NMS. Default is 5000.
        model: Optional model name, alias, or custom local .onnx path.
        cache_dir: Optional custom directory to store/retrieve cached models.

    Returns:
        DetectionResult containing detected faces and the original BGR image.
    """
    global _default_detector

    # Reuse cached default detector for identical standard configurations
    is_standard = (
        model is None
        and cache_dir is None
        and confidence_threshold == 0.7
        and nms_threshold == 0.3
        and top_k == 5000
    )

    if is_standard:
        if _default_detector is None:
            _default_detector = FaceDetector(
                confidence_threshold=0.7,
                nms_threshold=0.3,
                top_k=5000,
            )
        return _default_detector.detect(image)

    detector = FaceDetector(
        model=model,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k,
        cache_dir=cache_dir,
    )
    return detector.detect(image)


__all__: List[str] = [
    "__version__",
    "detect",
    "camera",
    "capture",
    "draw_faces",
    "draw_face_overlay",
    "parse_color_to_bgr",
    "SUPPORTED_SHAPES",
    "NAMED_COLORS",
    "FaceDetector",
    "BoundingBox",
    "Point",
    "FaceLandmarks",
    "FaceDetection",
    "DetectionResult",
    "FaceCapture",
    "DetekFaceError",
    "ModelError",
    "ModelNotFoundError",
    "ModelDownloadError",
    "ModelCorruptionError",
    "ModelCompatibilityError",
    "ImageError",
    "ImageNotFoundError",
    "InvalidImageError",
    "ConfigurationError",
    "CameraError",
    "CameraNotFoundError",
    "CameraReadError",
    "CaptureError",
    "NoFaceDetectedError",
    "InvalidCropError",
    "CaptureSaveError",
    "ModelMetadata",
    "OFFICIAL_MODELS",
    "get_cache_dir",
    "resolve_model",
    "verify_model_initialization",
]
