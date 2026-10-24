"""detekface - A clean, reusable Python face detection package powered by YuNet and OpenCV.

This package provides beginner-friendly face detection interfaces with sensible defaults,
clean dataclass outputs, and robust error handling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from detekface.camera import camera
from detekface.capture import capture
from detekface.detector import FaceDetector
from detekface.drawing import (
    FPSCounter,
    NAMED_COLORS,
    SUPPORTED_SHAPES,
    draw_face_overlay,
    draw_faces,
    draw_metrics_overlay,
    format_confidence,
    format_face_count,
    format_fps,
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
    ImageSaveError,
    InvalidCropError,
    InvalidImageError,
    InvalidSettingError,
    InvalidVideoError,
    ModelCompatibilityError,
    ModelCorruptionError,
    ModelDownloadError,
    ModelError,
    ModelNotFoundError,
    NoFaceDetectedError,
    VideoError,
    VideoNotFoundError,
    VideoWriteError,
)
from detekface.models import (
    OFFICIAL_MODELS,
    ModelMetadata,
    get_cache_dir,
    resolve_model,
    verify_model_initialization,
)
from detekface.types import (
    SUPPORTED_IMAGE_EXTENSIONS,
    BoundingBox,
    DetectionResult,
    FaceCapture,
    FaceDetection,
    FaceLandmarks,
    Point,
)
from detekface.video import SUPPORTED_VIDEO_EXTENSIONS, video

__version__: str = "0.1.0"

_detector_cache: Dict[Tuple, FaceDetector] = {}


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
    cache_key = (
        str(model) if model is not None else None,
        float(confidence_threshold),
        float(nms_threshold),
        int(top_k),
        str(cache_dir) if cache_dir is not None else None,
    )
    detector = _detector_cache.get(cache_key)
    if detector is None:
        detector = FaceDetector(
            model=model,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            cache_dir=cache_dir,
        )
        _detector_cache[cache_key] = detector

    return detector.detect(image)


__all__: List[str] = [
    "__version__",
    "detect",
    "camera",
    "capture",
    "video",
    "draw_faces",
    "draw_face_overlay",
    "draw_metrics_overlay",
    "parse_color_to_bgr",
    "format_confidence",
    "format_face_count",
    "format_fps",
    "FPSCounter",
    "SUPPORTED_SHAPES",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "SUPPORTED_VIDEO_EXTENSIONS",
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
    "ImageSaveError",
    "ConfigurationError",
    "InvalidSettingError",
    "CameraError",
    "CameraNotFoundError",
    "CameraReadError",
    "CaptureError",
    "NoFaceDetectedError",
    "InvalidCropError",
    "CaptureSaveError",
    "VideoError",
    "VideoNotFoundError",
    "InvalidVideoError",
    "VideoWriteError",
    "ModelMetadata",
    "OFFICIAL_MODELS",
    "get_cache_dir",
    "resolve_model",
    "verify_model_initialization",
]
