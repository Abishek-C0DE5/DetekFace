"""Data structures representing detected faces, bounding boxes, landmarks, and results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box coordinates of a detected face in image pixel space.

    Attributes:
        x: Horizontal pixel coordinate of the top-left corner.
        y: Vertical pixel coordinate of the top-left corner.
        width: Width of the bounding box in pixels.
        height: Height of the bounding box in pixels.
    """

    x: int
    y: int
    width: int
    height: int

    @property
    def x1(self) -> int:
        """Top-left horizontal coordinate."""
        return self.x

    @property
    def y1(self) -> int:
        """Top-left vertical coordinate."""
        return self.y

    @property
    def x2(self) -> int:
        """Bottom-right horizontal coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom-right vertical coordinate."""
        return self.y + self.height

    @property
    def area(self) -> int:
        """Area of the bounding box in square pixels."""
        return max(0, self.width) * max(0, self.height)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return (x, y, width, height) format."""
        return (self.x, self.y, self.width, self.height)

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) format."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class Point:
    """A 2D coordinate representing a key point in image pixel space."""

    x: int
    y: int

    def as_tuple(self) -> Tuple[int, int]:
        """Return (x, y) tuple."""
        return (self.x, self.y)


@dataclass(frozen=True)
class FaceLandmarks:
    """The 5 key facial landmarks detected by YuNet.

    Attributes:
        right_eye: Center coordinates of the subject's right eye (viewer's left).
        left_eye: Center coordinates of the subject's left eye (viewer's right).
        nose_tip: Coordinates of the nose tip.
        right_mouth: Coordinates of the subject's right mouth corner.
        left_mouth: Coordinates of the subject's left mouth corner.
    """

    right_eye: Point
    left_eye: Point
    nose_tip: Point
    right_mouth: Point
    left_mouth: Point

    def as_list(self) -> List[Point]:
        """Return landmarks as a list in canonical order."""
        return [
            self.right_eye,
            self.left_eye,
            self.nose_tip,
            self.right_mouth,
            self.left_mouth,
        ]

    def as_tuples(self) -> List[Tuple[int, int]]:
        """Return landmarks as a list of (x, y) integer tuples."""
        return [p.as_tuple() for p in self.as_list()]


@dataclass(frozen=True)
class FaceDetection:
    """A single face detected in an image.

    Attributes:
        bbox: The bounding box encompassing the face.
        confidence: Detection score between 0.0 and 1.0.
        landmarks: 5 facial key points if available, otherwise None.
    """

    bbox: BoundingBox
    confidence: float
    landmarks: Optional[FaceLandmarks] = None

    @property
    def x(self) -> int:
        """Top-left x pixel coordinate."""
        return self.bbox.x

    @property
    def y(self) -> int:
        """Top-left y pixel coordinate."""
        return self.bbox.y

    @property
    def width(self) -> int:
        """Bounding box pixel width."""
        return self.bbox.width

    @property
    def height(self) -> int:
        """Bounding box pixel height."""
        return self.bbox.height

    @property
    def x1(self) -> int:
        """Top-left horizontal coordinate."""
        return self.bbox.x1

    @property
    def y1(self) -> int:
        """Top-left vertical coordinate."""
        return self.bbox.y1

    @property
    def x2(self) -> int:
        """Bottom-right horizontal coordinate."""
        return self.bbox.x2

    @property
    def y2(self) -> int:
        """Bottom-right vertical coordinate."""
        return self.bbox.y2


@dataclass
class DetectionResult:
    """Result returned by the face detector.

    Attributes:
        faces: Sequence of detected FaceDetection instances.
        image: The original image array in BGR format.
    """

    faces: Sequence[FaceDetection]
    image: np.ndarray

    @property
    def count(self) -> int:
        """Total number of detected faces."""
        return len(self.faces)

    @property
    def has_faces(self) -> bool:
        """True if one or more faces were detected."""
        return len(self.faces) > 0

    def __len__(self) -> int:
        return len(self.faces)

    def __iter__(self) -> Iterator[FaceDetection]:
        return iter(self.faces)

    def __getitem__(self, index: int) -> FaceDetection:
        return self.faces[index]

    def __repr__(self) -> str:
        img_h, img_w = self.image.shape[:2] if isinstance(self.image, np.ndarray) and self.image.ndim >= 2 else (0, 0)
        return f"<DetectionResult: {len(self.faces)} face(s) found in {img_w}x{img_h} image>"


@dataclass
class FaceCapture:
    """A cropped face image captured from a detection result.

    Attributes:
        image: The cropped face image as a BGR uint8 NumPy array.
        path: Filesystem Path where the cropped image was saved, or None if not saved.
        bbox: BoundingBox of the face in the original image coordinates.
        confidence: Detection confidence score (0.0 to 1.0).
        landmarks: Facial landmarks if available, otherwise None.
    """

    image: np.ndarray
    path: Optional[Path]
    bbox: BoundingBox
    confidence: float
    landmarks: Optional[FaceLandmarks] = None

    @property
    def width(self) -> int:
        """Width of the cropped face in pixels."""
        return self.image.shape[1] if isinstance(self.image, np.ndarray) and self.image.ndim >= 2 else 0

    @property
    def height(self) -> int:
        """Height of the cropped face in pixels."""
        return self.image.shape[0] if isinstance(self.image, np.ndarray) and self.image.ndim >= 2 else 0

    def __repr__(self) -> str:
        saved_info = f", saved to '{self.path}'" if self.path else ""
        return (
            f"<FaceCapture: {self.width}x{self.height}px, "
            f"confidence={self.confidence:.2f}{saved_info}>"
        )
