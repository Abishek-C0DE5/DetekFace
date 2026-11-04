"""Facial mesh wireframe rendering engine for dense facial landmarks.

Visualizes 2D planar projections of dense facial surface keypoints and topological
connections across anatomical facial features (face oval, eyes, eyebrows, nose, lips).

IMPORTANT ARCHITECTURAL & ANATOMICAL NOTE:
This rendering component visualizes 2D planar projections of detected surface
facial landmark keypoints on image planes. It is a visual topological wireframe,
NOT a complete anatomical 3D craniofacial reconstruction or volumetric soft-tissue model.

Coordinate Conventions:
- Normalized coordinates: Floats x, y in [0.0, 1.0] where (0, 0) is top-left and (1, 1) is bottom-right.
- Image pixel space: Integers 0 <= x < image_width and 0 <= y < image_height.
- Both Point instances, (x, y) tuples, and normalized landmark models are supported and
  automatically mapped and clamped to the destination image bounds.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import cv2
import numpy as np

from detekface.drawing import parse_color_to_bgr
from detekface.exceptions import InvalidImageError, InvalidSettingError
from detekface.types import FaceDetection, Point

# -----------------------------------------------------------------------------
# Canonical MediaPipe FaceMesh Topological Connections
# -----------------------------------------------------------------------------

FACEMESH_FACE_OVAL: Tuple[Tuple[int, int], ...] = (
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
)

FACEMESH_LEFT_EYE: Tuple[Tuple[int, int], ...] = (
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
)

FACEMESH_RIGHT_EYE: Tuple[Tuple[int, int], ...] = (
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),
)

FACEMESH_LEFT_EYEBROW: Tuple[Tuple[int, int], ...] = (
    (276, 283), (283, 282), (282, 295), (295, 285),
    (300, 293), (293, 334), (334, 296), (296, 336),
)

FACEMESH_RIGHT_EYEBROW: Tuple[Tuple[int, int], ...] = (
    (46, 53), (53, 52), (52, 65), (65, 55),
    (70, 63), (63, 105), (105, 66), (66, 107),
)

FACEMESH_LIPS: Tuple[Tuple[int, int], ...] = (
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40),
    (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270),
    (270, 409), (409, 291), (78, 95), (95, 88), (88, 178), (178, 87),
    (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312),
    (312, 311), (311, 310), (310, 415), (415, 308),
)

FACEMESH_NOSE: Tuple[Tuple[int, int], ...] = (
    (168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 1),
    (1, 19), (19, 94), (94, 2), (98, 97), (97, 2), (2, 326),
    (326, 327), (327, 294), (294, 278), (278, 344), (344, 440),
    (440, 275), (275, 4), (4, 45), (45, 220), (220, 115),
    (115, 48), (48, 64), (64, 98),
)

# Canonical full contours combination
FACEMESH_CONTOURS: Tuple[Tuple[int, int], ...] = tuple(
    dict.fromkeys(
        FACEMESH_FACE_OVAL
        + FACEMESH_LEFT_EYE
        + FACEMESH_RIGHT_EYE
        + FACEMESH_LEFT_EYEBROW
        + FACEMESH_RIGHT_EYEBROW
        + FACEMESH_LIPS
        + FACEMESH_NOSE
    )
)

SUPPORTED_MESH_REGIONS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "contours": FACEMESH_CONTOURS,
    "oval": FACEMESH_FACE_OVAL,
    "eyes": tuple(dict.fromkeys(FACEMESH_LEFT_EYE + FACEMESH_RIGHT_EYE)),
    "left_eye": FACEMESH_LEFT_EYE,
    "right_eye": FACEMESH_RIGHT_EYE,
    "eyebrows": tuple(dict.fromkeys(FACEMESH_LEFT_EYEBROW + FACEMESH_RIGHT_EYEBROW)),
    "left_eyebrow": FACEMESH_LEFT_EYEBROW,
    "right_eyebrow": FACEMESH_RIGHT_EYEBROW,
    "lips": FACEMESH_LIPS,
    "nose": FACEMESH_NOSE,
}

SUPPORTED_MESH_STYLES: Tuple[str, ...] = (
    "wireframe",
    "neon",
    "minimal",
    "contour",
    "points",
)

# Canonical anchor keypoints emphasized in neon / cyber visualization style
NEON_KEY_LANDMARKS: Tuple[int, ...] = (
    10, 152, 234, 454, 33, 133, 362, 263, 1, 4, 61, 291, 0, 17, 468, 473
)


def validate_mesh_style(style: str) -> str:
    """Validate mesh visualization style name.

    Raises:
        InvalidSettingError: If the style name is invalid or unsupported.
    """
    if not isinstance(style, str):
        raise InvalidSettingError(
            message=f"Mesh style must be a string, got {type(style).__name__}.",
            hint=f"Supported mesh styles are: {sorted(SUPPORTED_MESH_STYLES)}.",
        )
    clean_style = style.strip().lower()
    if clean_style not in SUPPORTED_MESH_STYLES:
        raise InvalidSettingError(
            message=f"Unsupported mesh style: '{style}'.",
            hint=f"Supported mesh styles are: {sorted(SUPPORTED_MESH_STYLES)}.",
        )
    return clean_style


def _extract_coord(pt: Any) -> Optional[Tuple[float, float]]:
    """Extract (x, y) float coordinates from Point, dict, tuple, or object."""
    if pt is None:
        return None
    if isinstance(pt, Point):
        x, y = float(pt.x), float(pt.y)
    elif isinstance(pt, (tuple, list)):
        if len(pt) < 2:
            return None
        x, y = float(pt[0]), float(pt[1])
    elif isinstance(pt, dict):
        if "x" not in pt or "y" not in pt:
            return None
        x, y = float(pt["x"]), float(pt["y"])
    elif hasattr(pt, "x") and hasattr(pt, "y"):
        x, y = float(pt.x), float(pt.y)
    else:
        return None

    if not (math.isfinite(x) and math.isfinite(y)):
        return None

    return (x, y)


def convert_landmarks_to_pixels(
    landmarks: Optional[Sequence[Any]],
    image_width: int,
    image_height: int,
    normalized: Optional[bool] = None,
) -> List[Optional[Tuple[int, int]]]:
    """Map dense landmarks (normalized or pixel space) to pixel coordinates within image bounds.

    Args:
        landmarks: Sequence of landmark coordinates (Point, tuple, dict, or normalized object).
        image_width: Target image width in pixels.
        image_height: Target image height in pixels.
        normalized: Explicitly specify whether coordinates are normalized [0.0, 1.0].
            If None (default), auto-detects based on coordinate value ranges.

    Returns:
        List of (x, y) integer pixel coordinates clamped to [0, width - 1] and [0, height - 1],
        with None representing invalid or missing coordinates.
    """
    if not landmarks or image_width <= 0 or image_height <= 0:
        return []

    # Extract raw float coordinates safely
    extracted: List[Optional[Tuple[float, float]]] = [_extract_coord(pt) for pt in landmarks]
    valid_coords = [c for c in extracted if c is not None]

    if not valid_coords:
        return [None for _ in landmarks]

    # Detect normalized coordinates [0.0, 1.0] if not explicitly specified
    if normalized is not None:
        is_normalized = bool(normalized)
    else:
        # If all valid coordinates lie within a typical normalized range [-1.5, 2.5]
        # and at least one coordinate is a non-integer or <= 1.0, treat as normalized.
        is_normalized = all(-1.5 <= c[0] <= 2.5 and -1.5 <= c[1] <= 2.5 for c in valid_coords) and any(
            c[0] <= 1.0 and c[1] <= 1.0 for c in valid_coords
        )

    mapped: List[Optional[Tuple[int, int]]] = []
    max_x = image_width - 1
    max_y = image_height - 1

    for c in extracted:
        if c is None:
            mapped.append(None)
            continue
        raw_x, raw_y = c
        if is_normalized:
            px = int(round(raw_x * image_width))
            py = int(round(raw_y * image_height))
        else:
            px = int(round(raw_x))
            py = int(round(raw_y))

        # Clamp strictly within image boundaries
        clamped_x = max(0, min(max_x, px))
        clamped_y = max(0, min(max_y, py))
        mapped.append((clamped_x, clamped_y))

    return mapped


class FaceMeshRenderer:
    """Configurable 2D facial landmark wireframe and visual styling renderer.

    Visualization Styles:
        - 'wireframe': Connected canonical facial landmark wireframe lines.
        - 'neon': Highlighted mesh lines, prominent landmark points, and emphasized contours.
        - 'minimal': Ultra-clean, thin mesh contours with zero visual clutter.
        - 'contour': Emphasizes primary anatomical facial contours (oval, eyes, nose, lips).
        - 'points': Displays facial landmark dots without interconnecting lines.

    Attributes:
        style: Visualization style ('wireframe', 'neon', 'minimal', 'contour', 'points').
        color: Primary line color as hex string (#00FF88) or BGR tuple.
        thickness: Line stroke width in pixels (>= 1, or None for style-adaptive).
        draw_points: Whether to render circular keypoint dots at vertex positions.
        point_radius: Radius of vertex dots in pixels (or None for style-adaptive).
        point_color: Color of vertex dots.
        regions: Specific anatomical regions to render ('contours', 'eyes', 'lips', etc.).
    """

    def __init__(
        self,
        style: str = "wireframe",
        color: Union[str, Tuple[int, int, int]] = "#00FF88",
        thickness: Optional[int] = None,
        draw_points: Optional[bool] = None,
        point_radius: Optional[int] = None,
        point_color: Optional[Union[str, Tuple[int, int, int]]] = None,
        regions: Optional[Sequence[str]] = None,
    ) -> None:
        self.style = validate_mesh_style(style)
        self.color = parse_color_to_bgr(color)

        if thickness is not None:
            if not isinstance(thickness, int) or isinstance(thickness, bool) or thickness < 1:
                raise InvalidSettingError(
                    message=f"Mesh thickness must be an integer >= 1, got: {thickness}.",
                    hint="Use an integer such as 1, 2, or 3.",
                )
            self.thickness: Optional[int] = thickness
        else:
            self.thickness = None

        self.draw_points = bool(draw_points) if draw_points is not None else None

        if point_radius is not None:
            if not isinstance(point_radius, int) or isinstance(point_radius, bool) or point_radius < 1:
                raise InvalidSettingError(
                    message=f"Mesh point_radius must be an integer >= 1, got: {point_radius}.",
                    hint="Use an integer such as 1, 2, or 3.",
                )
            self.point_radius: Optional[int] = point_radius
        else:
            self.point_radius = None

        self.point_color = parse_color_to_bgr(point_color) if point_color else None
        self._user_regions = regions
        self.regions = self._resolve_regions(regions, self.style)

    def _resolve_regions(
        self,
        regions: Optional[Sequence[str]],
        style: str,
    ) -> Set[Tuple[int, int]]:
        """Resolve requested region names into a unified set of connection index pairs."""
        if style == "points" and regions is None:
            return set()

        if regions is None:
            return set(FACEMESH_CONTOURS)

        selected_connections: Set[Tuple[int, int]] = set()
        for reg in regions:
            reg_clean = str(reg).strip().lower()
            if reg_clean in SUPPORTED_MESH_REGIONS:
                selected_connections.update(SUPPORTED_MESH_REGIONS[reg_clean])

        return selected_connections if selected_connections else set(FACEMESH_CONTOURS)

    def render_landmarks(
        self,
        image: np.ndarray,
        landmarks: Optional[Sequence[Any]],
        copy: bool = True,
    ) -> np.ndarray:
        """Render facial mesh for a single set of landmarks.

        Args:
            image: Source image array (BGR uint8).
            landmarks: Sequence of dense facial keypoints.
            copy: If True (default), renders onto a copy of the image.

        Returns:
            Annotated BGR NumPy image array.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise InvalidImageError(
                message="Cannot render mesh on invalid or empty image array.",
                hint="Provide a valid NumPy uint8 image array.",
            )

        out = image.copy() if copy else image

        if not landmarks:
            return out

        h, w = out.shape[:2]
        pixel_points = convert_landmarks_to_pixels(landmarks, image_width=w, image_height=h)
        total_pts = len(pixel_points)

        # Estimate face dimension from valid coordinates for adaptive scaling
        valid_pixel_pts = [p for p in pixel_points if p is not None]
        if valid_pixel_pts:
            xs = [p[0] for p in valid_pixel_pts]
            ys = [p[1] for p in valid_pixel_pts]
            face_span = max(max(xs) - min(xs), max(ys) - min(ys))
        else:
            face_span = min(w, h)

        # 1. Determine effective thickness
        if self.thickness is not None:
            line_thickness = self.thickness
        elif self.style in ("neon", "contour"):
            line_thickness = max(1, min(3, int(round(face_span / 180.0))))
        elif self.style == "minimal":
            line_thickness = 1
        else:
            line_thickness = 1

        # 2. Determine effective draw_points flag
        if self.draw_points is not None:
            should_draw_points = self.draw_points
        elif self.style in ("points", "neon"):
            should_draw_points = True
        else:
            should_draw_points = False

        # 3. Determine effective point radius
        if self.point_radius is not None:
            pt_radius = self.point_radius
        elif self.style == "points":
            pt_radius = max(1, min(4, int(round(face_span / 120.0))))
        elif self.style == "neon":
            pt_radius = max(1, min(3, int(round(face_span / 160.0))))
        else:
            pt_radius = 1

        # 4. Determine effective point color
        if self.point_color is not None:
            pt_col = self.point_color
        elif self.style == "neon":
            pt_col = (255, 255, 255)  # Bright high-contrast white accent
        else:
            pt_col = self.color

        # 5. Draw connection lines
        for start_idx, end_idx in self.regions:
            if start_idx < total_pts and end_idx < total_pts:
                p1 = pixel_points[start_idx]
                p2 = pixel_points[end_idx]
                if p1 is not None and p2 is not None:
                    # Emphasize face oval and lips slightly in neon style
                    if self.style == "neon" and (start_idx, end_idx) in FACEMESH_FACE_OVAL:
                        cv2.line(out, p1, p2, self.color, line_thickness + 1, cv2.LINE_AA)
                    else:
                        cv2.line(out, p1, p2, self.color, line_thickness, cv2.LINE_AA)

        # 6. Draw landmark points
        if should_draw_points:
            if self.style == "neon" and self.draw_points is None:
                # In neon style without explicit draw_points=True, emphasize key anatomical anchors
                for idx in NEON_KEY_LANDMARKS:
                    if idx < total_pts and pixel_points[idx] is not None:
                        cv2.circle(out, pixel_points[idx], pt_radius + 1, pt_col, -1, cv2.LINE_AA)
            else:
                for pt in pixel_points:
                    if pt is not None:
                        cv2.circle(out, pt, pt_radius, pt_col, -1, cv2.LINE_AA)

        return out

    def render_faces(
        self,
        image: np.ndarray,
        faces: Sequence[FaceDetection],
        copy: bool = True,
    ) -> np.ndarray:
        """Render facial meshes across multiple detected faces independently.

        Args:
            image: Source image array (BGR uint8).
            faces: Sequence of FaceDetection instances.
            copy: If True (default), renders onto a copy of the image.

        Returns:
            Annotated BGR NumPy image array.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise InvalidImageError(
                message="Cannot render mesh on invalid or empty image array.",
                hint="Provide a valid NumPy uint8 image array.",
            )

        out = image.copy() if copy else image

        for face in faces:
            if face.dense_landmarks is not None:
                self.render_landmarks(out, face.dense_landmarks, copy=False)

        return out


def draw_facial_mesh(
    image: np.ndarray,
    landmarks: Optional[Sequence[Any]],
    style: str = "wireframe",
    color: Union[str, Tuple[int, int, int]] = "#00FF88",
    thickness: Optional[int] = None,
    draw_points: Optional[bool] = None,
    point_radius: Optional[int] = None,
    point_color: Optional[Union[str, Tuple[int, int, int]]] = None,
    regions: Optional[Sequence[str]] = None,
    copy: bool = True,
) -> np.ndarray:
    """Render 2D facial mesh wireframe for a single set of landmarks.

    Args:
        image: Source BGR image NumPy array.
        landmarks: Dense sequence of facial keypoints.
        style: Mesh visualization style ('wireframe', 'neon', 'minimal', 'contour', 'points').
        color: Wireframe stroke color (hex or RGB/BGR tuple). Default '#00FF88'.
        thickness: Line stroke width in pixels.
        draw_points: Whether to render vertex points as small circles.
        point_radius: Vertex point radius in pixels.
        point_color: Vertex point color.
        regions: Specific anatomical regions to render ('contours', 'eyes', 'lips', etc.).
        copy: Whether to render on a copy of the image (default True, original untouched).

    Returns:
        Rendered BGR image NumPy array.
    """
    renderer = FaceMeshRenderer(
        style=style,
        color=color,
        thickness=thickness,
        draw_points=draw_points,
        point_radius=point_radius,
        point_color=point_color,
        regions=regions,
    )
    return renderer.render_landmarks(image=image, landmarks=landmarks, copy=copy)


def draw_face_meshes(
    image: np.ndarray,
    faces: Sequence[FaceDetection],
    style: str = "wireframe",
    color: Union[str, Tuple[int, int, int]] = "#00FF88",
    thickness: Optional[int] = None,
    draw_points: Optional[bool] = None,
    point_radius: Optional[int] = None,
    point_color: Optional[Union[str, Tuple[int, int, int]]] = None,
    regions: Optional[Sequence[str]] = None,
    copy: bool = True,
) -> np.ndarray:
    """Render 2D facial meshes for multiple detected faces independently.

    Args:
        image: Source BGR image NumPy array.
        faces: Sequence of FaceDetection instances containing dense_landmarks.
        style: Mesh visualization style ('wireframe', 'neon', 'minimal', 'contour', 'points').
        color: Wireframe stroke color.
        thickness: Line stroke width in pixels.
        draw_points: Whether to render vertex dots.
        point_radius: Vertex dot radius.
        point_color: Vertex dot color.
        regions: Feature regions to render.
        copy: Whether to copy image before rendering (default True).

    Returns:
        Rendered BGR image NumPy array.
    """
    renderer = FaceMeshRenderer(
        style=style,
        color=color,
        thickness=thickness,
        draw_points=draw_points,
        point_radius=point_radius,
        point_color=point_color,
        regions=regions,
    )
    return renderer.render_faces(image=image, faces=faces, copy=copy)
