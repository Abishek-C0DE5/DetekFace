"""Base adapter interface for face detection backends.

Standardizes face detection across diverse inference engines (YuNet, MediaPipe, etc.)
into detekface's canonical result dataclasses.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from detekface.types import FaceDetection


class BaseFaceBackend(ABC):
    """Abstract base adapter for face detection backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the detection backend."""
        pass

    @property
    @abstractmethod
    def confidence_threshold(self) -> float:
        """Minimum confidence threshold for detected faces."""
        pass

    @abstractmethod
    def detect(self, bgr_image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in a normalized 3-channel BGR uint8 NumPy array.

        Args:
            bgr_image: Image array in BGR format (height, width, 3).

        Returns:
            List of normalized FaceDetection objects with bounding boxes,
            confidence scores, and backend-specific landmarks.
        """
        pass

    def close(self) -> None:
        """Release any backend-specific resources or sessions."""
        pass

    def __enter__(self) -> "BaseFaceBackend":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
