"""detekface - A clean, reusable Python face detection package powered by YuNet and OpenCV.

This package provides beginner-friendly face detection interfaces with sensible defaults,
clean dataclass outputs, and robust error handling.
"""

from detekface.exceptions import (
    DetekFaceError,
    ModelCompatibilityError,
    ModelCorruptionError,
    ModelDownloadError,
    ModelError,
    ModelNotFoundError,
)
from detekface.models import (
    OFFICIAL_MODELS,
    ModelMetadata,
    get_cache_dir,
    resolve_model,
    verify_model_initialization,
)

__version__: str = "0.1.0"

__all__ = [
    "__version__",
    "DetekFaceError",
    "ModelError",
    "ModelNotFoundError",
    "ModelDownloadError",
    "ModelCorruptionError",
    "ModelCompatibilityError",
    "ModelMetadata",
    "OFFICIAL_MODELS",
    "get_cache_dir",
    "resolve_model",
    "verify_model_initialization",
]
