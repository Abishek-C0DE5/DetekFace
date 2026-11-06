"""Data structures representing detected faces, bounding boxes, landmarks, and results."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from detekface.exceptions import DetekFaceError, ImageSaveError, InvalidImageError

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def _validate_and_prepare_save_path(path: Union[str, Path]) -> Path:
    """Validate image save path and create parent directories if necessary."""
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

    return target_path.resolve()


@dataclass(frozen=True)
class Point:
    """A 2D coordinate representing a key point in image pixel space.

    Attributes:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.
    """

    x: int
    y: int

    def as_tuple(self) -> Tuple[int, int]:
        """Return (x, y) tuple."""
        return (self.x, self.y)

    def as_int_tuple(self) -> Tuple[int, int]:
        """Return (x, y) rounded integer tuple."""
        return (int(round(self.x)), int(round(self.y)))


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

    @property
    def center(self) -> Point:
        """Center coordinate of the bounding box."""
        return Point(x=self.x + self.width // 2, y=self.y + self.height // 2)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return (x, y, width, height) format."""
        return (self.x, self.y, self.width, self.height)

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) format."""
        return (self.x1, self.y1, self.x2, self.y2)

    def as_ltrb(self) -> Tuple[int, int, int, int]:
        """Return (left, top, right, bottom) format (alias for as_xyxy)."""
        return self.as_xyxy()

    def clamp(self, max_width: int, max_height: int) -> "BoundingBox":
        """Return a new BoundingBox clamped to image boundaries [0, max_width] and [0, max_height]."""
        x1 = max(0, min(self.x1, max_width))
        y1 = max(0, min(self.y1, max_height))
        x2 = max(0, min(self.x2, max_width))
        y2 = max(0, min(self.y2, max_height))
        return BoundingBox(x=x1, y=y1, width=max(0, x2 - x1), height=max(0, y2 - y1))


@dataclass(frozen=True)
class FaceLandmarks:
    """Facial keypoints detected by face detection backends.

    Supplied landmarks by backend:
        - YuNet: right_eye, left_eye, nose_tip, right_mouth, left_mouth.
        - MediaPipe: right_eye, left_eye, nose_tip, mouth_center, right_ear, left_ear.

    Landmarks not provided by a given backend remain None. Missing landmarks are
    never fabricated or interpolated.

    Attributes:
        right_eye: Center coordinates of the subject's right eye (viewer's left).
        left_eye: Center coordinates of the subject's left eye (viewer's right).
        nose_tip: Coordinates of the nose tip.
        right_mouth: Coordinates of the subject's right mouth corner (YuNet).
        left_mouth: Coordinates of the subject's left mouth corner (YuNet).
        mouth_center: Coordinates of the mouth center (MediaPipe).
        right_ear: Coordinates of the subject's right ear tragion (MediaPipe).
        left_ear: Coordinates of the subject's left ear tragion (MediaPipe).
    """

    right_eye: Optional[Point] = None
    left_eye: Optional[Point] = None
    nose_tip: Optional[Point] = None
    right_mouth: Optional[Point] = None
    left_mouth: Optional[Point] = None
    mouth_center: Optional[Point] = None
    right_ear: Optional[Point] = None
    left_ear: Optional[Point] = None

    @property
    def mouth_left(self) -> Optional[Point]:
        """Alias for left_mouth (subject's left, viewer's right)."""
        return self.left_mouth

    @property
    def mouth_right(self) -> Optional[Point]:
        """Alias for right_mouth (subject's right, viewer's left)."""
        return self.right_mouth

    def as_list(self) -> List[Point]:
        """Return all available (non-None) landmarks as a list in canonical order."""
        ordered = [
            self.right_eye,
            self.left_eye,
            self.nose_tip,
            self.right_mouth,
            self.left_mouth,
            self.mouth_center,
            self.right_ear,
            self.left_ear,
        ]
        return [p for p in ordered if p is not None]

    def as_tuples(self) -> List[Tuple[int, int]]:
        """Return all available (non-None) landmarks as a list of (x, y) integer tuples."""
        return [p.as_tuple() for p in self.as_list()]


@dataclass(frozen=True)
class FaceShapeResult:
    """Structured result of face shape classification and facial geometric analysis.

    Supported shape categories:
        - 'oval': Balanced proportions, cheekbones widest, gently curved jawline.
        - 'round': Length and width nearly equal, soft curved angles, rounded jaw.
        - 'square': Strong angular jawline, jaw width comparable to cheekbones and forehead.
        - 'heart': Broad forehead and prominent cheekbones tapering to a narrow chin.
        - 'diamond': Cheekbones widest with narrower forehead and tapering chin.
        - 'oblong': Noticeably longer face length with uniform width.
        - 'undetermined': Analysis uncertain due to extreme pose, occlusion, or low resolution.

    Attributes:
        label: Classified face shape category name.
        fit_score: Heuristic geometric fit score (0.0 to 1.0) indicating classification strength.
        alternatives: Relative heuristic fit scores for candidate shape categories.
        measurements: Geometric dimensions (pixels) and proportional ratios.
        quality_status: Reliability assessment ('good', 'warning', 'rejected', etc.).
        warnings: List of cautionary notes explaining degraded accuracy or undetermined status.
    """

    label: str
    fit_score: float
    alternatives: Dict[str, float]
    measurements: Dict[str, float]
    quality_status: str = "good"
    warnings: List[str] = field(default_factory=list)

    @property
    def reasons(self) -> List[str]:
        """Alias for warnings."""
        return self.warnings


@dataclass(frozen=True)
class HeadPose:
    """Estimated 3D head pose orientation angles (in degrees).

    Coordinate convention:
        - yaw: Left/right head rotation in degrees.
               Positive = turning to subject's right (viewer's left).
               Negative = turning to subject's left (viewer's right).
        - pitch: Up/down head tilt in degrees.
               Positive = looking upward.
               Negative = looking downward.
        - roll: In-plane lateral tilt in degrees.
               Positive = tilting head toward subject's right shoulder.
               Negative = tilting head toward subject's left shoulder.
    """

    yaw: float
    pitch: float
    roll: float
    rotation_vector: Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return (yaw, pitch, roll) in degrees."""
        return (self.yaw, self.pitch, self.roll)


@dataclass(frozen=True)
class GestureEvent:
    """Event emitted when an observable facial gesture is detected.

    Attributes:
        name: Canonical gesture name (e.g. 'mouth_opening', 'left_eye_blink').
        track_id: Temporary tracking identifier of the face (if tracking is active).
        face_index: Positional index of the face within the current frame detections.
        timestamp: Wall-clock timestamp (seconds) when the gesture event occurred.
        confidence: Estimated confidence / intensity of the gesture (0.0 to 1.0).
        duration: Duration (seconds) the gesture state was observed.
        state: Event trigger state ('action', 'start', 'end').
        metadata: Underlying metrics dictionary (e.g. angles, aspect ratios).
    """

    name: str
    track_id: Optional[int] = None
    face_index: int = 0
    timestamp: float = 0.0
    confidence: float = 1.0
    duration: float = 0.0
    state: str = "action"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FaceDetection:
    """A single face detected in an image.

    Attributes:
        bbox: The bounding box encompassing the face.
        confidence: Detection score between 0.0 and 1.0.
        landmarks: 5 facial key points if available, otherwise None.
        face_shape: FaceShapeResult if analyze_face_shape=True, otherwise None.
        dense_landmarks: Sequence of dense facial keypoints (e.g. 478 points) if extracted, otherwise None.
        track_id: Temporary tracking identifier for video/camera continuity (not persistent identity).
        head_pose: Estimated 3D head pose orientation (yaw, pitch, roll in degrees) if enabled.
        gestures: List of active gesture names currently observed on this face.
        blendshapes: Dictionary mapping blendshape category names to confidence scores (0.0 to 1.0).
    """

    bbox: BoundingBox
    confidence: float
    landmarks: Optional[FaceLandmarks] = None
    face_shape: Optional[FaceShapeResult] = None
    dense_landmarks: Optional[List[Point]] = None
    track_id: Optional[int] = None
    head_pose: Optional[HeadPose] = None
    gestures: Optional[List[str]] = None
    blendshapes: Optional[Dict[str, float]] = None

    @property
    def box(self) -> BoundingBox:
        """BoundingBox alias for bbox."""
        return self.bbox

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
        show_face_shape: bool = False,
        show_shape_contour: bool = True,
        show_shape_measurements: bool = False,
        show_mesh: bool = False,
        mesh_style: str = "wireframe",
        mesh_color: Optional[Union[str, Tuple[int, int, int]]] = None,
        mesh_thickness: Optional[int] = None,
        mesh_regions: Optional[Sequence[str]] = None,
        mesh_draw_points: Optional[bool] = None,
        show_contour: Optional[bool] = None,
        live_fx: bool = False,
        live_fx_style: str = "scan",
        live_fx_timestamp: Optional[float] = None,
        effect: Optional[str] = None,
        accessory: Optional[str] = None,
        accessory_opacity: float = 1.0,
        accessory_scale: float = 1.0,
        accessory_y_offset: float = 0.0,
    ) -> np.ndarray:
        """Render face overlay annotations on a copy of the original image.

        Args:
             shape: Overlay shape ('rectangle', 'corners', 'circle', 'rounded', 'hud').
             color: RGB hex code (e.g. '#00FF88'), color name ('cyan', 'green', etc.), or RGB tuple.
             thickness: Line stroke thickness in pixels (default 3).
             show_confidence: Whether to draw confidence percentage badges near each face.
             show_face_shape: Whether to draw face shape classification badge and biometric overlays.
             show_shape_contour: Whether to draw the facial silhouette contour mesh.
             show_shape_measurements: Whether to display key geometric measurement ratios.
             show_mesh: Whether to render the dense facial landmark wireframe mesh.
             mesh_style: Visualization style ('wireframe', 'neon', 'minimal', 'contour', 'points').
             mesh_color: Color for facial mesh lines (defaults to main overlay color).
             mesh_thickness: Line stroke thickness for facial mesh.
             mesh_regions: Anatomical mesh regions to render ('contours', 'eyes', 'lips', etc.).
             mesh_draw_points: Whether to render vertex keypoint dots.
             show_contour: Alias for show_shape_contour.
             live_fx: Whether to render animated Live FX modern computer-vision scanner layer.
             live_fx_style: Live FX animation style ('scan', 'neon', 'wireframe', 'minimal').
             live_fx_timestamp: Optional clock timestamp in seconds for deterministic animation testing.

        Returns:
             A new NumPy array (BGR uint8) containing the annotated image with overlays.
        """
        if not isinstance(self.image, np.ndarray) or self.image.size == 0:
            raise InvalidImageError(
                message="DetectionResult does not contain a valid image array to annotate.",
                hint="Verify that the detection result contains valid image data.",
            )

        from detekface.drawing import draw_faces, parse_color_to_bgr

        effective_contour = show_shape_contour if show_contour is None else bool(show_contour)

        # Layer 1: Dense facial mesh wireframe (drawn first as base layer)
        annotated = self.image.copy()
        if show_mesh:
            from detekface.mesh import draw_face_meshes

            m_col = mesh_color if mesh_color is not None else color
            annotated = draw_face_meshes(
                image=annotated,
                faces=self.faces,
                style=mesh_style,
                color=m_col,
                thickness=mesh_thickness,
                draw_points=mesh_draw_points,
                regions=mesh_regions,
                copy=False,
            )

        # Layer 2: Live FX modern computer-vision scanner
        if live_fx:
            from detekface.live_fx import LiveFXRenderer

            fx_renderer = LiveFXRenderer(style=live_fx_style, color=color)
            annotated = fx_renderer.render(
                annotated,
                faces=self.faces,
                timestamp=live_fx_timestamp,
                copy=False,
            )

        # Layer 3: Primary face overlay shapes
        if shape is not None and str(shape).strip().lower() != "none":
            annotated = draw_faces(
                image=annotated,
                faces=self.faces,
                shape=shape,
                color=color,
                thickness=thickness,
                show_confidence=show_confidence,
                copy=False,
            )

        # Layer 4: Face shape biometric badges & silhouette contour
        if show_face_shape:
            from detekface.face_shape import draw_face_shape_overlay

            bgr_col = parse_color_to_bgr(color)
            for face in self.faces:
                if face.face_shape is not None:
                    draw_face_shape_overlay(
                        image=annotated,
                        face=face,
                        effect=None,
                        color=bgr_col,
                        thickness=thickness,
                        show_contour=effective_contour,
                        show_measurements=show_shape_measurements,
                    )

        # Layer 5: AR Face accessories
        eff = effect if effect is not None else accessory
        if eff is not None:
            from detekface.accessories import render_accessory

            annotated = render_accessory(
                image=annotated,
                faces=self.faces,
                accessory=eff,
                opacity=accessory_opacity,
                scale_adjust=accessory_scale,
                y_offset=accessory_y_offset,
                copy=False,
            )

        return annotated

    def draw(
        self,
        shape: str = "rectangle",
        color: Union[str, Tuple[int, int, int]] = "#00FF88",
        thickness: int = 3,
        show_confidence: bool = False,
        show_face_shape: bool = False,
        show_shape_contour: bool = True,
        show_shape_measurements: bool = False,
        show_mesh: bool = False,
        mesh_style: str = "wireframe",
        mesh_color: Optional[Union[str, Tuple[int, int, int]]] = None,
        mesh_thickness: Optional[int] = None,
        mesh_regions: Optional[Sequence[str]] = None,
        mesh_draw_points: Optional[bool] = None,
        show_contour: Optional[bool] = None,
        live_fx: bool = False,
        live_fx_style: str = "scan",
        live_fx_timestamp: Optional[float] = None,
        effect: Optional[str] = None,
        accessory: Optional[str] = None,
        accessory_opacity: float = 1.0,
        accessory_scale: float = 1.0,
        accessory_y_offset: float = 0.0,
    ) -> np.ndarray:
        """Alias for `annotate()` to obtain the rendered image array."""
        return self.annotate(
            shape=shape,
            color=color,
            thickness=thickness,
            show_confidence=show_confidence,
            show_face_shape=show_face_shape,
            show_shape_contour=show_shape_contour,
            show_shape_measurements=show_shape_measurements,
            show_mesh=show_mesh,
            mesh_style=mesh_style,
            mesh_color=mesh_color,
            mesh_thickness=mesh_thickness,
            mesh_regions=mesh_regions,
            mesh_draw_points=mesh_draw_points,
            show_contour=show_contour,
            live_fx=live_fx,
            live_fx_style=live_fx_style,
            live_fx_timestamp=live_fx_timestamp,
            effect=effect,
            accessory=accessory,
            accessory_opacity=accessory_opacity,
            accessory_scale=accessory_scale,
            accessory_y_offset=accessory_y_offset,
        )

    def render(
        self,
        shape: str = "rectangle",
        color: Union[str, Tuple[int, int, int]] = "#00FF88",
        thickness: int = 3,
        show_confidence: bool = False,
        show_face_shape: bool = False,
        show_shape_contour: bool = True,
        show_shape_measurements: bool = False,
        show_mesh: bool = False,
        mesh_style: str = "wireframe",
        mesh_color: Optional[Union[str, Tuple[int, int, int]]] = None,
        mesh_thickness: Optional[int] = None,
        mesh_regions: Optional[Sequence[str]] = None,
        mesh_draw_points: Optional[bool] = None,
        show_contour: Optional[bool] = None,
        live_fx: bool = False,
        live_fx_style: str = "scan",
        live_fx_timestamp: Optional[float] = None,
        effect: Optional[str] = None,
        accessory: Optional[str] = None,
        accessory_opacity: float = 1.0,
        accessory_scale: float = 1.0,
        accessory_y_offset: float = 0.0,
    ) -> np.ndarray:
        """Alias for `annotate()` to obtain the rendered image array."""
        return self.annotate(
            shape=shape,
            color=color,
            thickness=thickness,
            show_confidence=show_confidence,
            show_face_shape=show_face_shape,
            show_shape_contour=show_shape_contour,
            show_shape_measurements=show_shape_measurements,
            show_mesh=show_mesh,
            mesh_style=mesh_style,
            mesh_color=mesh_color,
            mesh_thickness=mesh_thickness,
            mesh_regions=mesh_regions,
            mesh_draw_points=mesh_draw_points,
            show_contour=show_contour,
            live_fx=live_fx,
            live_fx_style=live_fx_style,
            live_fx_timestamp=live_fx_timestamp,
            effect=effect,
            accessory=accessory,
            accessory_opacity=accessory_opacity,
            accessory_scale=accessory_scale,
            accessory_y_offset=accessory_y_offset,
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
        show_face_shape: bool = False,
        show_shape_contour: bool = True,
        show_shape_measurements: bool = False,
        show_mesh: bool = False,
        mesh_style: str = "wireframe",
        mesh_color: Optional[Union[str, Tuple[int, int, int]]] = None,
        mesh_thickness: Optional[int] = None,
        mesh_regions: Optional[Sequence[str]] = None,
        mesh_draw_points: Optional[bool] = None,
        show_contour: Optional[bool] = None,
        live_fx: bool = False,
        live_fx_style: str = "scan",
        live_fx_timestamp: Optional[float] = None,
        effect: Optional[str] = None,
        accessory: Optional[str] = None,
        accessory_opacity: float = 1.0,
        accessory_scale: float = 1.0,
        accessory_y_offset: float = 0.0,
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
            show_face_shape: Whether to render face shape classification overlays.
            show_shape_contour: Whether to render facial silhouette contour mesh.
            show_shape_measurements: Whether to print key proportion ratios.
            show_mesh: Whether to render dense facial landmark wireframe mesh.
            mesh_style: Visualization style ('wireframe', 'neon', 'minimal', 'contour', 'points').
            mesh_color: Color for facial mesh lines.
            mesh_thickness: Line stroke thickness for facial mesh.
            mesh_regions: Anatomical mesh regions to render ('contours', 'eyes', etc.).
            mesh_draw_points: Whether to render vertex keypoint dots.

        Returns:
            The integer keycode of the pressed key, or -1 if the window was closed.
        """
        annotated_frame = self.annotate(
            shape=shape,
            color=color,
            thickness=thickness,
            show_confidence=show_confidence,
            show_face_shape=show_face_shape,
            show_shape_contour=show_shape_contour,
            show_shape_measurements=show_shape_measurements,
            show_mesh=show_mesh,
            mesh_style=mesh_style,
            mesh_color=mesh_color,
            mesh_thickness=mesh_thickness,
            mesh_regions=mesh_regions,
            mesh_draw_points=mesh_draw_points,
            show_contour=show_contour,
            live_fx=live_fx,
            live_fx_style=live_fx_style,
            live_fx_timestamp=live_fx_timestamp,
            effect=effect,
            accessory=accessory,
            accessory_opacity=accessory_opacity,
            accessory_scale=accessory_scale,
            accessory_y_offset=accessory_y_offset,
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
        show_face_shape: bool = False,
        show_shape_contour: bool = True,
        show_shape_measurements: bool = False,
        show_mesh: bool = False,
        mesh_style: str = "wireframe",
        mesh_color: Optional[Union[str, Tuple[int, int, int]]] = None,
        mesh_thickness: Optional[int] = None,
        mesh_regions: Optional[Sequence[str]] = None,
        mesh_draw_points: Optional[bool] = None,
        show_contour: Optional[bool] = None,
        live_fx: bool = False,
        live_fx_style: str = "scan",
        live_fx_timestamp: Optional[float] = None,
        effect: Optional[str] = None,
        accessory: Optional[str] = None,
        accessory_opacity: float = 1.0,
        accessory_scale: float = 1.0,
        accessory_y_offset: float = 0.0,
    ) -> Path:
        """Save the annotated image with overlays to the requested disk path.

        Args:
            path: Target file path (str or Path) with a supported image extension.
            shape: Overlay shape ('rectangle', 'corners', 'circle', 'rounded', 'hud').
            color: Overlay color name, hex string, or RGB tuple.
            thickness: Stroke thickness in pixels.
            show_confidence: Whether to draw confidence scores.
            show_face_shape: Whether to render face shape classification overlays.
            show_shape_contour: Whether to render facial silhouette contour mesh.
            show_shape_measurements: Whether to print key proportion ratios.
            show_mesh: Whether to render dense facial landmark wireframe mesh.
            mesh_style: Visualization style ('wireframe', 'neon', 'minimal', 'contour', 'points').
            mesh_color: Color for facial mesh lines.
            mesh_thickness: Line stroke thickness for facial mesh.
            mesh_regions: Anatomical mesh regions to render ('contours', 'eyes', etc.).
            mesh_draw_points: Whether to render vertex keypoint dots.

        Returns:
            The resolved Path to the saved image file.

        Raises:
            InvalidImageError: If the path is empty, invalid, or has an unsupported extension.
            ImageSaveError: If the directory cannot be created or saving the file fails.
        """
        resolved_path = _validate_and_prepare_save_path(path)

        annotated_frame = self.annotate(
            shape=shape,
            color=color,
            thickness=thickness,
            show_confidence=show_confidence,
            show_face_shape=show_face_shape,
            show_shape_contour=show_shape_contour,
            show_shape_measurements=show_shape_measurements,
            show_mesh=show_mesh,
            mesh_style=mesh_style,
            mesh_color=mesh_color,
            mesh_thickness=mesh_thickness,
            mesh_regions=mesh_regions,
            mesh_draw_points=mesh_draw_points,
            show_contour=show_contour,
            live_fx=live_fx,
            live_fx_style=live_fx_style,
            live_fx_timestamp=live_fx_timestamp,
            effect=effect,
            accessory=accessory,
            accessory_opacity=accessory_opacity,
            accessory_scale=accessory_scale,
            accessory_y_offset=accessory_y_offset,
        )

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

    def align(
        self,
        face_index: Optional[int] = None,
        output_size: Optional[Tuple[int, int]] = (112, 112),
        padding: float = 0.25,
        center_on: str = "eyes",
        border_mode: int = cv2.BORDER_REFLECT_101,
        border_value: Tuple[int, int, int] = (0, 0, 0),
    ):
        """Straighten and crop a detected face using landmark alignment.

        Args:
            face_index: Index of the face to align (default None selects largest face).
            output_size: Output (width, height) in pixels, or None to keep natural face dimensions.
            padding: Additional fractional margin around face bounding box (default 0.25).
            center_on: 'eyes' for portrait eye centering or 'face' for bbox centering.
            border_mode: OpenCV border interpolation mode.
            border_value: Border fill color if constant border mode.

        Returns:
            AlignedFace: Straightened, cropped face with full transformation metadata.
        """
        if not self.has_faces:
            from detekface.exceptions import NoFaceDetectedError

            raise NoFaceDetectedError(
                message="Cannot align face: no faces were detected in this image.",
                hint="Verify that the image contains a clear face.",
            )

        if face_index is None:
            # Largest face by bounding box area
            target_face = max(self.faces, key=lambda f: f.bbox.area)
        else:
            if not 0 <= face_index < len(self.faces):
                raise IndexError(
                    f"Face index {face_index} is out of range for {len(self.faces)} detected face(s)."
                )
            target_face = self.faces[face_index]

        from detekface.alignment import align_face

        return align_face(
            image=self.image,
            face=target_face,
            output_size=output_size,
            padding=padding,
            center_on=center_on,
            border_mode=border_mode,
            border_value=border_value,
        )

    def aligned_faces(
        self,
        output_size: Optional[Tuple[int, int]] = (112, 112),
        padding: float = 0.25,
        center_on: str = "eyes",
        border_mode: int = cv2.BORDER_REFLECT_101,
        border_value: Tuple[int, int, int] = (0, 0, 0),
    ):
        """Align and crop all detected faces in this result.

        Returns:
            List[AlignedFace]: Ordered list of aligned face crops.
        """
        from detekface.alignment import align_faces

        return align_faces(
            image=self.image,
            faces=self.faces,
            output_size=output_size,
            padding=padding,
            center_on=center_on,
            border_mode=border_mode,
            border_value=border_value,
        )

    def check_quality(
        self,
        blur_threshold: float = 80.0,
        min_face_size: int = 48,
        min_area_ratio: float = 0.005,
        min_brightness: float = 50.0,
        max_brightness: float = 200.0,
        border_margin: int = 4,
    ):
        """Evaluate technical capture quality for detected faces in this result.

        Returns:
            ImageQualityResult: Structured report with sharpness, lighting, framing,
                                pose suitability, and actionable warnings.
        """
        from detekface.quality import check_face_quality

        return check_face_quality(
            image=self,
            blur_threshold=blur_threshold,
            min_face_size=min_face_size,
            min_area_ratio=min_area_ratio,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
            border_margin=border_margin,
        )

    def anonymize(
        self,
        method: str = "blur",
        strength: float = 0.5,
        kernel_size: Optional[int] = None,
        pixel_size: Optional[int] = None,
        color: Union[str, Tuple[int, int, int]] = (0, 0, 0),
        mask_shape: str = "box",
        padding: float = 0.0,
        fail_closed: bool = False,
        output_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ):
        """Anonymize detected faces in this result using blur, pixelation, or solid masking.

        Args:
            method: 'blur', 'pixelate', or 'solid' (default 'blur').
            strength: Intensity in [0.0, 1.0] (default 0.5).
            kernel_size: Optional explicit odd blur kernel size.
            pixel_size: Optional explicit pixelation mosaic block size.
            color: Solid mask color (default black).
            mask_shape: 'box' or 'ellipse'.
            padding: Additional fractional margin around bounding boxes.
            fail_closed: If True and 0 faces were detected, anonymizes the entire frame.
            output_path: Optional file path to save the anonymized image.
            overwrite: Whether to overwrite existing destination files.

        Returns:
            AnonymizationResult: Anonymized image array and operation metadata.
        """
        from detekface.anonymize import anonymize

        return anonymize(
            image=self,
            method=method,
            strength=strength,
            kernel_size=kernel_size,
            pixel_size=pixel_size,
            color=color,
            mask_shape=mask_shape,
            padding=padding,
            fail_closed=fail_closed,
            output_path=output_path,
            overwrite=overwrite,
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

    def save(self, path: Union[str, Path]) -> Path:
        """Save the cropped face image to disk.

        Args:
            path: Target file path (str or Path) with a supported image extension.

        Returns:
            The resolved Path to the saved face image file.

        Raises:
            InvalidImageError: If the path is empty, invalid, or has an unsupported extension.
            ImageSaveError: If the directory cannot be created or saving the file fails.
        """
        if not isinstance(self.image, np.ndarray) or self.image.size == 0:
            raise InvalidImageError(
                message="FaceCapture does not contain a valid image array to save.",
                hint="Verify that the captured face contains valid image data.",
            )

        resolved_path = _validate_and_prepare_save_path(path)
        try:
            success = cv2.imwrite(str(resolved_path), self.image)
            if not success:
                raise ImageSaveError(
                    message=f"OpenCV failed to write face crop to disk at: '{resolved_path}'.",
                    hint="Verify directory write permissions and available disk space.",
                )
        except cv2.error as e:
            raise ImageSaveError(
                message=f"OpenCV encountered an error while writing face crop: '{resolved_path}'.",
                hint=str(e),
            ) from e

        self.path = resolved_path
        return resolved_path

    def show(self, title: str = "Captured Face", delay: int = 0) -> int:
        """Display the cropped face image in an OpenCV GUI window until closed or dismissed.

        Args:
            title: Title string of the GUI display window.
            delay: Milliseconds to wait before closing (if > 0). Default is 0 (waits indefinitely).

        Returns:
            The integer keycode of the pressed key, or -1 if the window was closed.
        """
        if not isinstance(self.image, np.ndarray) or self.image.size == 0:
            raise InvalidImageError(
                message="FaceCapture does not contain a valid image array to display.",
                hint="Verify that the captured face contains valid image data.",
            )

        try:
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(title, self.image)
        except cv2.error as e:
            raise DetekFaceError(
                message=f"Failed to display GUI window '{title}'.",
                hint="Ensure an active graphical display environment is available.",
            ) from e

        try:
            if delay > 0:
                raw_key = cv2.waitKey(delay)
                return (raw_key & 0xFF) if raw_key != -1 else -1

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

    def __repr__(self) -> str:
        saved_info = f", saved to '{self.path}'" if self.path else ""
        return f"<FaceCapture: {self.width}x{self.height}px, confidence={self.confidence:.2f}{saved_info}>"
