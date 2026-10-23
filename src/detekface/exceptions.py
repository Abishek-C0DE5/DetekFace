"""Custom exception hierarchy for detekface.

Provides beginner-friendly, structured exceptions with actionable remediation guidance.
"""

from typing import Optional


class DetekFaceError(Exception):
    """Base exception for all detekface runtime errors."""

    def __init__(self, message: str, hint: Optional[str] = None) -> None:
        self.message = message
        self.hint = hint
        full_message = f"{message}\nHint: {hint}" if hint else message
        super().__init__(full_message)


class ModelError(DetekFaceError):
    """Base exception for model resolution, download, and verification errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a specified model file cannot be found on disk."""
    pass


class ModelDownloadError(ModelError):
    """Raised when an official model file cannot be downloaded from remote sources."""
    pass


class ModelCorruptionError(ModelError):
    """Raised when a model file fails checksum or integrity validation."""
    pass


class ModelCompatibilityError(ModelError):
    """Raised when a model file is incompatible with the installed OpenCV runtime."""
    pass


class ImageError(DetekFaceError):
    """Base exception for image loading, format, and decoding errors."""
    pass


class ImageNotFoundError(ImageError):
    """Raised when an image file cannot be found at the specified path."""
    pass


class InvalidImageError(ImageError):
    """Raised when an image file or NumPy array is corrupt, empty, or has unsupported dimensions."""
    pass


class ConfigurationError(DetekFaceError):
    """Raised when invalid thresholds or configuration parameters are provided."""
    pass
