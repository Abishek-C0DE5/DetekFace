"""Data structures representing detected faces, bounding boxes, landmarks, and results."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from detekface.exceptions import DetekFaceError, ImageSaveError, InvalidImageError

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


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

    @property
    def original_image(self) -> np.ndarray:
        """The unmodified original image array."""
        return self.image

    @property
    def rgb_image(self) -> np.ndarray:
        """The original image array converted from BGR to RGB format."""
        if not isinstance(self.image, np.ndarray) or self.image.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def annotate(
        self,
        shape: str = "rectangle",
        color: Union[str, Tuple[int, int, int]] = "#00FF88",
        thickness: int = 3,
        show_confidence: bool = False,
    ) -> np.ndarray:
        """Render face overlay annotations on a copy of the original image.

        Args:
             shape: Overlay shape ('rectangle', 'corners', 'circle', 'rounded', 'hud').
             color: RGB hex code (e.g. '#00FF88'), color name ('cyan', 'green', etc.), or RGB tuple.
             thickness: Line stroke thickness in pixels (default 3).
             show_confidence: Whether to draw confidence percentage badges near each face.

        Returns:
             A new NumPy array (BGR uint8) containing the annotated image with overlays.
        """
        if not isinstance(self.image, np.ndarray) or self.image.size == 0:
            raise InvalidImageError(
                message="DetectionResult does not contain a valid image array to annotate.",
                hint="Verify that the detection result contains valid image data.",
            )

        from detekface.drawing import draw_faces

        return draw_faces(
            image=self.image,
            faces=self.faces,
            shape=shape,
            color=color,
            thickness=thickness,
            show_confidence=show_confidence,
            copy=True,
        )

    def draw(
        self,
        shape: str = "rectangle",
        color: Union[str, Tuple[int, int, int]] = "#00FF88",
        thickness: int = 3,
        show_confidence: bool = False,
    ) -> np.ndarray:
        """Alias for `annotate()` to obtain the rendered image array."""
        return self.annotate(
            shape=shape,
            color=color,
            thickness=thickness,
            show_confidence=show_confidence,
        )

    @property
    def annotated(self) -> np.ndarray:
        """Annotated image array with default overlay styling."""
        return self.annotate()

    def show(
        self,
        title: str = "DetekFace Detection",
        shape: str = "rectangle",
        color: Union[str, Tuple[int, int, int]] = "#00FF88",
        thickness: int = 3,
        show_confidence: bool = False,
        delay: int = 0,
    ) -> int:
        """Display the annotated image in an OpenCV GUI window until closed or dismissed.

        Args:
            title: Title string of the GUI display window.
            shape: Overlay shape ('rectangle', 'corners', 'circle', 'rounded', 'hud').
            color: Overlay color name, hex string, or RGB tuple.
            thickness: Stroke thickness.
            show_confidence: Whether to display detection confidence labels.
            delay: Milliseconds to wait before closing (if > 0). Default is 0, which
                   waits until the user presses a key or closes the window.

        Returns:
            The integer keycode of the pressed key, or -1 if the window was closed.
        """
        annotated_frame = self.annotate(
            shape=shape,
            color=color,
            thickness=thickness,
            show_confidence=show_confidence,
        )

        try:
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(title, annotated_frame)
        except cv2.error as e:
            raise DetekFaceError(
                message=f"Failed to display GUI window '{title}'.",
                hint="Ensure an active graphical display environment is available.",
            ) from e

        try:
            if delay > 0:
                raw_key = cv2.waitKey(delay)
                return (raw_key & 0xFF) if raw_key != -1 else -1

            # Polling loop allows clean exit when user presses a key OR clicks OS window close button
            while True:
                raw_key = cv2.waitKey(50)
                if raw_key != -1:
                    return raw_key & 0xFF
                try:
                    prop = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
                    if prop < 1:
                        return -1
                except cv2.error:
                    return -1
        finally:
            try:
                cv2.destroyWindow(title)
            except cv2.error:
                pass

    def save(
        self,
        path: Union[str, Path],
        shape: str = "rectangle",
        color: Union[str, Tuple[int, int, int]] = "#00FF88",
        thickness: int = 3,
        show_confidence: bool = False,
    ) -> Path:
        """Save the annotated image with overlays to the requested disk path.

        Args:
            path: Target file path (str or Path) with a supported image extension.
            shape: Overlay shape ('rectangle', 'corners', 'circle', 'rounded', 'hud').
            color: Overlay color name, hex string, or RGB tuple.
            thickness: Stroke thickness in pixels.
            show_confidence: Whether to draw confidence scores.

        Returns:
            The resolved Path to the saved image file.

        Raises:
            InvalidImageError: If the path is empty, invalid, or has an unsupported extension.
            ImageSaveError: If the directory cannot be created or saving the file fails.
        """
        if path is None or not isinstance(path, (str, Path)) or not str(path).strip():
            raise InvalidImageError(
                message=f"Invalid save path: {path!r}.",
                hint="Provide a valid filepath string or Path with a supported image extension.",
            )

        target_path = Path(path).expanduser()
        ext = target_path.suffix.lower()

        if not ext:
            raise InvalidImageError(
                message=f"Save path '{target_path}' has no file extension.",
                hint=f"Provide a file extension such as: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}.",
            )

        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            raise InvalidImageError(
                message=f"Unsupported image file format '{ext}' in path: '{target_path}'.",
                hint=f"Supported formats are: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}.",
            )

        parent_dir = target_path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise ImageSaveError(
                    message=f"Could not create parent directory for save path: '{parent_dir}'.",
                    hint=str(e),
                ) from e
        elif not parent_dir.is_dir():
            raise ImageSaveError(
                message=f"Parent path of save target is a file, not a directory: '{parent_dir}'.",
                hint="Specify a destination path inside a valid directory.",
            )

        annotated_frame = self.annotate(
            shape=shape,
            color=color,
            thickness=thickness,
            show_confidence=show_confidence,
        )

        resolved_path = target_path.resolve()
        try:
            success = cv2.imwrite(str(resolved_path), annotated_frame)
            if not success:
                raise ImageSaveError(
                    message=f"OpenCV failed to write image to disk at: '{resolved_path}'.",
                    hint="Verify directory write permissions and available disk space.",
                )
        except cv2.error as e:
            raise ImageSaveError(
                message=f"OpenCV encountered an error while writing: '{resolved_path}'.",
                hint=str(e),
            ) from e

        return resolved_path

    def capture(
        self,
        face_index: Optional[int] = None,
        save_dir: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        save: bool = True,
    ) -> "FaceCapture":
        """Crop and capture a detected face from this DetectionResult.

        Crops strictly from the original, unannotated image to ensure the captured
        face is never polluted with drawn overlays.

        Args:
             face_index: Index of face to capture (default None chooses largest face).
             save_dir: Directory where the face crop is saved (default 'captures').
             filename: Optional custom filename.
             save: Whether to save crop to disk (default True).

        Returns:
             FaceCapture containing the cropped image and metadata.
        """
        from detekface.capture import capture

        return capture(
            result=self,
            face_index=face_index,
            save_dir=save_dir,
            filename=filename,
            save=save,
        )

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

    @property
    def box(self) -> BoundingBox:
        """BoundingBox alias for bbox."""
        return self.bbox

    @property
    def rgb_image(self) -> np.ndarray:
        """The cropped face image converted from BGR to RGB format."""
        if not isinstance(self.image, np.ndarray) or self.image.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def __repr__(self) -> str:
        saved_info = f", saved to '{self.path}'" if self.path else ""
        return (
            f"<FaceCapture: {self.width}x{self.height}px, "
            f"confidence={self.confidence:.2f}{saved_info}>"
        )
