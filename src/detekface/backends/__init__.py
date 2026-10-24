"""Detection backend registry and adapter factory.

Provides unified backend selection for 'yunet' (default) and 'mediapipe' (optional).
"""

from pathlib import Path
from typing import Optional, Tuple, Union

from detekface.backends.base import BaseFaceBackend
from detekface.backends.yunet import YuNetBackend
from detekface.exceptions import InvalidSettingError

SUPPORTED_BACKENDS: Tuple[str, ...] = ("yunet", "mediapipe")
DEFAULT_BACKEND: str = "yunet"


def get_backend(
    backend: str = "yunet",
    model: Optional[Union[str, Path]] = None,
    confidence_threshold: float = 0.7,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
    cache_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> BaseFaceBackend:
    """Create and return a configured face detection backend instance.

    Args:
        backend: Name of the backend ('yunet' or 'mediapipe'). Default is 'yunet'.
        model: Optional model name, alias, or local file path.
        confidence_threshold: Minimum detection confidence score (0.0 to 1.0).
        nms_threshold: Non-maximum suppression IoU threshold (0.0 to 1.0, YuNet only).
        top_k: Maximum candidate faces to retain prior to NMS (YuNet only).
        cache_dir: Optional custom directory to cache models.
        **kwargs: Additional backend-specific parameters.

    Returns:
        An instantiated BaseFaceBackend adapter.

    Raises:
        InvalidSettingError: If the backend name is unknown or invalid.
        DependencyNotFoundError: If an optional backend (e.g. 'mediapipe') is not installed.
    """
    if not isinstance(backend, str) or not backend.strip():
        raise InvalidSettingError(
            message=f"Invalid backend type or empty backend name: {backend!r}.",
            hint=f"Supported backends are: {', '.join(sorted(SUPPORTED_BACKENDS))}.",
        )

    clean_backend = backend.strip().lower()

    if clean_backend == "yunet":
        return YuNetBackend(
            model=model,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            cache_dir=cache_dir,
        )

    if clean_backend == "mediapipe":
        from detekface.backends.mediapipe import MediaPipeBackend

        return MediaPipeBackend(
            model=model,
            confidence_threshold=confidence_threshold,
            cache_dir=cache_dir,
            **kwargs,
        )

    raise InvalidSettingError(
        message=f"Unsupported detection backend '{backend}'.",
        hint=f"Supported backends are: {', '.join(sorted(SUPPORTED_BACKENDS))}. Default is '{DEFAULT_BACKEND}'.",
    )


__all__ = [
    "BaseFaceBackend",
    "YuNetBackend",
    "SUPPORTED_BACKENDS",
    "DEFAULT_BACKEND",
    "get_backend",
]
