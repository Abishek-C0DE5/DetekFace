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


class ImageSaveError(ImageError):
    """Raised when writing or encoding an annotated image file to disk fails."""
    pass


class ConfigurationError(DetekFaceError):
    """Raised when invalid thresholds or configuration parameters are provided."""
    pass


class InvalidSettingError(ConfigurationError):
    """Raised when an invalid overlay shape, color, or configuration setting is provided."""
    pass


class CameraError(DetekFaceError):
    """Base exception for camera capture and video streaming errors."""
    pass


class CameraNotFoundError(CameraError):
    """Raised when the specified camera device cannot be opened or is not detected."""
    pass


class CameraReadError(CameraError):
    """Raised when reading a frame from the camera fails or the stream disconnects."""
    pass


class CaptureError(DetekFaceError):
    """Base exception for face cropping, capturing, and image saving operations."""
    pass


class NoFaceDetectedError(CaptureError):
    """Raised when capture is invoked on a detection result containing no faces."""
    pass


class InvalidCropError(CaptureError):
    """Raised when a face bounding box results in an empty or invalid image crop."""
    pass


class CaptureSaveError(CaptureError):
    """Raised when saving a cropped face image to disk fails."""
    pass


class VideoError(DetekFaceError):
    """Base exception for video processing, decoding, and encoding errors."""
    pass


class VideoNotFoundError(VideoError):
    """Raised when the specified video file cannot be found."""
    pass


class InvalidVideoError(VideoError):
    """Raised when a video file is corrupt, empty, unreadable, or has an unsupported codec."""
    pass


class VideoWriteError(VideoError):
    """Raised when writing or encoding an output video fails."""
    pass
