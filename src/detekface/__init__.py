"""detekface - A clean, reusable Python face detection package powered by YuNet and OpenCV.

This package provides beginner-friendly face detection interfaces with sensible defaults,
clean dataclass outputs, and robust error handling.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from detekface.batch import BatchDetectionResult, BatchItemResult
from detekface.camera import CameraSession, camera, open_camera
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
    DependencyNotFoundError,
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
from detekface.backends import (
    DEFAULT_BACKEND,
    SUPPORTED_BACKENDS,
    BaseFaceBackend,
    get_backend,
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
    backend: str = "yunet",
    input_size: Optional[Tuple[int, int]] = None,
) -> DetectionResult:
    """Detect faces in a still image.

    Args:
        image: A filesystem path to an image file (str or Path), or a NumPy array in BGR format.
        confidence_threshold: Minimum detection confidence score (0.0 to 1.0). Default is 0.7.
        nms_threshold: Non-maximum suppression IoU threshold (0.0 to 1.0, YuNet only). Default is 0.3.
        top_k: Maximum candidate faces to retain prior to NMS (YuNet only). Default is 5000.
        model: Optional model name, alias, or custom local model path.
        cache_dir: Optional custom directory to store/retrieve cached models.
        backend: Detection backend ('yunet' or 'mediapipe'). Default is 'yunet'.
        input_size: Optional inference resolution tuple (width, height) for performance scaling.

    Returns:
        DetectionResult containing detected faces and the original BGR image.
    """
    clean_input_size = (int(input_size[0]), int(input_size[1])) if input_size is not None else None
    cache_key = (
        str(backend).strip().lower(),
        str(model) if model is not None else None,
        float(confidence_threshold),
        float(nms_threshold),
        int(top_k),
        str(cache_dir) if cache_dir is not None else None,
        clean_input_size,
    )
    detector = _detector_cache.get(cache_key)
    if detector is None:
        if input_size is not None:
            detector = FaceDetector(
                model=model,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                top_k=top_k,
                cache_dir=cache_dir,
                backend=backend,
                input_size=input_size,
            )
        else:
            detector = FaceDetector(
                model=model,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                top_k=top_k,
                cache_dir=cache_dir,
                backend=backend,
            )
        _detector_cache[cache_key] = detector

    return detector.detect(image)


def detect_many(
    images: Union[str, Path, Sequence[Union[str, Path, np.ndarray]]],
    confidence_threshold: float = 0.7,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
    model: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    backend: str = "yunet",
    detector: Optional[FaceDetector] = None,
    fail_fast: bool = False,
    progress_callback: Optional[Callable] = None,
    recursive: bool = False,
    keep_images: bool = True,
    input_size: Optional[Tuple[int, int]] = None,
) -> BatchDetectionResult:
    """Detect faces across a batch of images or an image directory.

    Args:
        images: A filesystem path, directory, NumPy array, or sequence of inputs.
        confidence_threshold: Minimum detection confidence score (0.0 to 1.0). Default is 0.7.
        nms_threshold: Non-maximum suppression IoU threshold (0.0 to 1.0, YuNet only). Default is 0.3.
        top_k: Maximum candidate faces to retain prior to NMS (YuNet only). Default is 5000.
        model: Optional model name, alias, or custom local model path.
        cache_dir: Optional custom directory to store/retrieve cached models.
        backend: Detection backend ('yunet' or 'mediapipe'). Default is 'yunet'.
        detector: Optional pre-configured FaceDetector instance to reuse.
        fail_fast: If True, stop and raise immediately upon the first failure.
                   If False (default), collect errors into BatchItemResult.error.
        progress_callback: Optional callable called after processing each image.
        recursive: If True, scans subdirectories when a directory path is passed.
        keep_images: If False, drops heavy image pixel arrays from returned results to conserve RAM.
        input_size: Optional inference resolution tuple (width, height) for performance scaling.

    Returns:
        BatchDetectionResult containing ordered BatchItemResult objects.
    """
    if detector is None:
        clean_input_size = (int(input_size[0]), int(input_size[1])) if input_size is not None else None
        cache_key = (
            str(backend).strip().lower(),
            str(model) if model is not None else None,
            float(confidence_threshold),
            float(nms_threshold),
            int(top_k),
            str(cache_dir) if cache_dir is not None else None,
            clean_input_size,
        )
        detector = _detector_cache.get(cache_key)
        if detector is None:
            if input_size is not None:
                detector = FaceDetector(
                    model=model,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                    top_k=top_k,
                    cache_dir=cache_dir,
                    backend=backend,
                    input_size=input_size,
                )
            else:
                detector = FaceDetector(
                    model=model,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                    top_k=top_k,
                    cache_dir=cache_dir,
                    backend=backend,
                )
            _detector_cache[cache_key] = detector

    from detekface.batch import run_batch_detection

    return run_batch_detection(
        detector=detector,
        images=images,
        fail_fast=fail_fast,
        progress_callback=progress_callback,
        recursive=recursive,
        keep_images=keep_images,
    )


__all__: List[str] = [
    "__version__",
    "detect",
    "detect_many",
    "BatchItemResult",
    "BatchDetectionResult",
    "camera",
    "open_camera",
    "CameraSession",
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
    "DependencyNotFoundError",
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
    "BaseFaceBackend",
    "SUPPORTED_BACKENDS",
    "DEFAULT_BACKEND",
    "get_backend",
]
