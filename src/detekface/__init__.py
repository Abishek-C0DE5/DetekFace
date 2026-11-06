"""detekface - A clean, reusable Python face detection package powered by YuNet and OpenCV.

This package provides beginner-friendly face detection interfaces with sensible defaults,
clean dataclass outputs, and robust error handling.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from detekface.async_cam import AsyncCameraEngine, CameraMetrics, TimestampedFrame
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
    draw_pose_axes,
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
    FaceShapeResult,
    GestureEvent,
    HeadPose,
    Point,
)
from detekface.tracking import FaceTracker, compute_iou
from detekface.pose import (
    TemporalPoseSmoother,
    estimate_head_pose,
    project_pose_axes,
)
from detekface.gestures import (
    ALL_GESTURES,
    GESTURE_BOTH_EYES_CLOSED,
    GESTURE_HEAD_TILT,
    GESTURE_HEAD_TURN_LEFT,
    GESTURE_HEAD_TURN_RIGHT,
    GESTURE_LEFT_EYE_BLINK,
    GESTURE_MOUTH_OPENING,
    GESTURE_RIGHT_EYE_BLINK,
    GestureConfig,
    GestureDetector,
    compute_eye_aspect_ratio,
    compute_mouth_aspect_ratio,
    normalize_gesture_name,
)
from detekface.face_shape import (
    SUPPORTED_FACE_SHAPES,
    DenseLandmarkExtractor,
    analyze_face_geometry,
    draw_face_shape_overlay,
)
from detekface.mesh import (
    FACEMESH_CONTOURS,
    FACEMESH_FACE_OVAL,
    FACEMESH_LEFT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_LIPS,
    FACEMESH_NOSE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW,
    SUPPORTED_MESH_REGIONS,
    SUPPORTED_MESH_STYLES,
    FaceMeshRenderer,
    convert_landmarks_to_pixels,
    draw_facial_mesh,
    draw_face_meshes,
    validate_mesh_style,
)
from detekface.live_fx import (
    LiveFXRenderer,
    SUPPORTED_LIVE_FX_STYLES,
    validate_live_fx_style,
)
from detekface.video import SUPPORTED_VIDEO_EXTENSIONS, video
from detekface.accessories import (
    ACCESSORY_ALIASES,
    ACCESSORY_CROWN,
    ACCESSORY_FACE_MASK,
    ACCESSORY_FUTURISTIC_VISOR,
    ACCESSORY_HALO,
    ACCESSORY_SUNGLASSES,
    SUPPORTED_ACCESSORIES,
    AccessorySmoother,
    FaceAccessoryRenderer,
    compute_accessory_transform,
    load_accessory_asset,
    overlay_rgba,
    render_accessory,
)
from detekface.alignment import (
    AlignedFace,
    FaceAligner,
    align_face,
    align_faces,
    estimate_face_rotation,
)
from detekface.quality import (
    FaceQualityMetrics,
    ImageQualityResult,
    QUALITY_DISCLAIMER,
    assess_face_quality,
    check_face_quality,
)
from detekface.anonymize import (
    PRIVACY_DISCLAIMER,
    SUPPORTED_ANONYMIZE_METHODS,
    AnonymizationResult,
    anonymize,
    anonymize_frame,
    anonymize_roi,
    normalize_anonymize_method,
)

__version__: str = "0.3.0"

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
    analyze_face_shape: bool = False,
    mesh: bool = False,
    estimate_pose: bool = False,
    detect_gestures: bool = False,
    track_faces: bool = False,
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
        analyze_face_shape: If True, performs dense landmark localization and geometric
            face shape classification (oval, round, square, heart, diamond, oblong, undetermined).
        mesh: If True, performs dense facial landmark localization to enable mesh rendering.

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

    detect_kwargs = {
        "analyze_face_shape": analyze_face_shape,
        "mesh": mesh,
        "estimate_pose": estimate_pose,
    }
    if detect_gestures:
        detect_kwargs["detect_gestures"] = detect_gestures
    if track_faces:
        detect_kwargs["track_faces"] = track_faces

    return detector.detect(
        image,
        **detect_kwargs,
    )


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
    analyze_face_shape: bool = False,
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
        analyze_face_shape: If True, executes face shape analysis on all detected faces.

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
        analyze_face_shape=analyze_face_shape,
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
    "AsyncCameraEngine",
    "CameraMetrics",
    "TimestampedFrame",
    "capture",
    "video",
    "FaceTracker",
    "compute_iou",
    "HeadPose",
    "estimate_head_pose",
    "TemporalPoseSmoother",
    "project_pose_axes",
    "draw_pose_axes",
    "GestureDetector",
    "GestureConfig",
    "GestureEvent",
    "GESTURE_LEFT_EYE_BLINK",
    "GESTURE_RIGHT_EYE_BLINK",
    "GESTURE_BOTH_EYES_CLOSED",
    "GESTURE_MOUTH_OPENING",
    "GESTURE_HEAD_TILT",
    "GESTURE_HEAD_TURN_LEFT",
    "GESTURE_HEAD_TURN_RIGHT",
    "ALL_GESTURES",
    "compute_eye_aspect_ratio",
    "compute_mouth_aspect_ratio",
    "normalize_gesture_name",
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
    "FaceShapeResult",
    "SUPPORTED_FACE_SHAPES",
    "draw_face_shape_overlay",
    "DenseLandmarkExtractor",
    "analyze_face_geometry",
    "FaceMeshRenderer",
    "draw_facial_mesh",
    "draw_face_meshes",
    "convert_landmarks_to_pixels",
    "SUPPORTED_MESH_REGIONS",
    "SUPPORTED_MESH_STYLES",
    "validate_mesh_style",
    "LiveFXRenderer",
    "SUPPORTED_LIVE_FX_STYLES",
    "validate_live_fx_style",
    "FACEMESH_CONTOURS",
    "FACEMESH_FACE_OVAL",
    "FACEMESH_LEFT_EYE",
    "FACEMESH_RIGHT_EYE",
    "FACEMESH_LEFT_EYEBROW",
    "FACEMESH_RIGHT_EYEBROW",
    "FACEMESH_LIPS",
    "FACEMESH_NOSE",
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
    "ACCESSORY_SUNGLASSES",
    "ACCESSORY_CROWN",
    "ACCESSORY_FACE_MASK",
    "ACCESSORY_FUTURISTIC_VISOR",
    "ACCESSORY_HALO",
    "SUPPORTED_ACCESSORIES",
    "ACCESSORY_ALIASES",
    "FaceAccessoryRenderer",
    "AccessorySmoother",
    "load_accessory_asset",
    "compute_accessory_transform",
    "overlay_rgba",
    "render_accessory",
    "AlignedFace",
    "FaceAligner",
    "align_face",
    "align_faces",
    "estimate_face_rotation",
    "FaceQualityMetrics",
    "ImageQualityResult",
    "QUALITY_DISCLAIMER",
    "assess_face_quality",
    "check_face_quality",
    "AnonymizationResult",
    "PRIVACY_DISCLAIMER",
    "SUPPORTED_ANONYMIZE_METHODS",
    "anonymize",
    "anonymize_frame",
    "anonymize_roi",
    "normalize_anonymize_method",
]
