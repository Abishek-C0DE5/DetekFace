"""Face overlay rendering engine providing customizable visual styles and metric overlays.

Supports rectangle, corners, circle, rounded, and hud overlays with robust RGB/hex-to-BGR
color conversion, confidence labeling, FPS calculation, and face count status metrics.
"""

from collections import deque
import math
import time
from typing import Deque, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from detekface.exceptions import ConfigurationError
from detekface.types import BoundingBox, DetectionResult, FaceDetection

# Documented named color mapping (RGB values)
NAMED_COLORS: Dict[str, Tuple[int, int, int]] = {
    "green": (0, 255, 0),
    "neon_green": (0, 255, 136),
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "cyan": (0, 255, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "lime": (50, 205, 50),
    "teal": (0, 128, 128),
    "purple": (128, 0, 128),
}

SUPPORTED_SHAPES = {"rectangle", "corners", "circle", "rounded", "hud"}


class FPSCounter:
    """Sliding-window frames-per-second calculator providing stable, smoothed FPS throughput."""

    def __init__(self, window_size: int = 30) -> None:
        self._window_size = max(2, window_size)
        self._timestamps: Deque[float] = deque(maxlen=self._window_size)

    def tick(self, current_time: Optional[float] = None) -> float:
        """Record a frame timestamp and return the current smoothed FPS."""
        now = time.perf_counter() if current_time is None else float(current_time)
        self._timestamps.append(now)
        return self.fps

    @property
    def fps(self) -> float:
        """Current smoothed FPS based on recent frame intervals."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        """Clear recorded timestamps."""
        self._timestamps.clear()


def format_face_count(count: int) -> str:
    """Format face count for display."""
    if count == 1:
        return "1 Face"
    return f"{count} Faces"


def format_fps(fps_value: float) -> str:
    """Format FPS metric for display."""
    return f"{fps_value:.1f} FPS"


def format_confidence(confidence: float) -> str:
    """Format confidence score as a percentage."""
    return f"{confidence * 100:.1f}%"


def parse_color_to_bgr(color: Union[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """Convert a color name, RGB tuple, or hex string to an OpenCV BGR tuple.

    Args:
        color: A hex string (e.g. '#00FF88'), a documented color name (e.g. 'green', 'neon_green'),
               or an (R, G, B) integer tuple.

    Returns:
        A tuple of (B, G, R) integer values in [0, 255] for OpenCV.

    Raises:
        ConfigurationError: If the color format or name is not recognized.
    """
    if isinstance(color, (tuple, list)):
        if len(color) != 3 or not all(isinstance(v, int) and 0 <= v <= 255 for v in color):
            raise ConfigurationError(
                message=f"Color tuple must contain 3 integers in range 0-255, got: {color}.",
                hint="Provide a tuple like (0, 255, 136) or a hex string like '#00FF88'.",
            )
        # Assumed input is RGB -> convert to BGR
        r, g, b = color
        return (int(b), int(g), int(r))

    if not isinstance(color, str):
        raise ConfigurationError(
            message=f"Invalid color type '{type(color).__name__}'.",
            hint="Provide a hex string (e.g. '#00FF88') or a color name.",
        )

    clean_str = color.strip().lower()

    # Check named colors
    if clean_str in NAMED_COLORS:
        r, g, b = NAMED_COLORS[clean_str]
        return (b, g, r)

    # Check hex string
    hex_str = clean_str.lstrip("#")
    if len(hex_str) == 3:
        # Expand 3-digit hex: e.g. '0f8' -> '00ff88'
        hex_str = "".join(ch * 2 for ch in hex_str)

    if len(hex_str) == 6:
        try:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            return (b, g, r)
        except ValueError as e:
            raise ConfigurationError(
                message=f"Invalid hex color string: '{color}'.",
                hint="Hex string must contain valid hexadecimal characters (0-9, A-F).",
            ) from e

    raise ConfigurationError(
        message=f"Unrecognized color specification: '{color}'.",
        hint=(
            f"Available color names: {sorted(NAMED_COLORS.keys())}\n"
            "Or provide an RGB hex string such as '#00FF88' or '#00FF00'."
        ),
    )


def validate_render_parameters(shape: str, thickness: int) -> Tuple[str, int]:
    """Validate shape name and thickness value."""
    if not isinstance(shape, str):
        raise ConfigurationError(
            message=f"Shape must be a string, got {type(shape).__name__}.",
            hint=f"Supported shapes are: {sorted(SUPPORTED_SHAPES)}.",
        )

    clean_shape = shape.strip().lower()
    if clean_shape not in SUPPORTED_SHAPES:
        raise ConfigurationError(
            message=f"Unsupported overlay shape: '{shape}'.",
            hint=f"Supported shapes are: {sorted(SUPPORTED_SHAPES)}.",
        )

    if not isinstance(thickness, int) or isinstance(thickness, bool) or thickness < 1:
        raise ConfigurationError(
            message=f"Thickness must be an integer >= 1, got: {thickness}.",
            hint="Use an integer such as 1, 2, 3, or 4.",
        )

    return clean_shape, thickness


def draw_text_badge(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (25, 25, 25),
    font_scale: float = 0.45,
    thickness: int = 1,
    padding: int = 4,
) -> None:
    """Draw text with a readable dark background badge."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    img_h, img_w = img.shape[:2]

    # Clamp position within image
    x = max(padding, min(x, img_w - tw - padding))
    y = max(th + padding, min(y, img_h - padding))

    # Background rectangle
    cv2.rectangle(
        img,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color,
        cv2.FILLED,
    )
    # Foreground text
    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_metrics_overlay(
    image: np.ndarray,
    fps: Optional[float] = None,
    face_count: Optional[int] = None,
    color_bgr: Tuple[int, int, int] = (136, 255, 0),
) -> None:
    """Draw FPS throughput and face count in a top-left status banner."""
    badges = []
    if fps is not None:
        badges.append(format_fps(fps))
    if face_count is not None:
        badges.append(format_face_count(face_count))

    if not badges:
        return

    status_str = " | ".join(badges)
    draw_text_badge(
        image,
        status_str,
        (15, 26),
        text_color=color_bgr,
        bg_color=(20, 20, 20),
        font_scale=0.52,
        thickness=1,
        padding=5,
    )


def _draw_rectangle(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color_bgr: Tuple[int, int, int], thickness: int) -> None:
    """Draw a classic rectangle box."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thickness)


def _draw_corners(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color_bgr: Tuple[int, int, int], thickness: int) -> None:
    """Draw tactical corner brackets on the four corners of the bounding box."""
    w = x2 - x1
    h = y2 - y1
    c_len = max(6, min(w, h) // 4)

    # Top-Left
    cv2.line(img, (x1, y1), (x1 + c_len, y1), color_bgr, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + c_len), color_bgr, thickness)

    # Top-Right
    cv2.line(img, (x2, y1), (x2 - c_len, y1), color_bgr, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + c_len), color_bgr, thickness)

    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + c_len, y2), color_bgr, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - c_len), color_bgr, thickness)

    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - c_len, y2), color_bgr, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - c_len), color_bgr, thickness)


def _draw_circle(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color_bgr: Tuple[int, int, int], thickness: int) -> None:
    """Draw a geometric circle overlay encompassing the detected face.

    Note: This is a geometric overlay indicator around the face, not a facial contour.
    """
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = x2 - x1
    h = y2 - y1
    radius = max(8, int(round(max(w, h) / 2 * 1.05)))
    cv2.circle(img, (cx, cy), radius, color_bgr, thickness)


def _draw_rounded(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color_bgr: Tuple[int, int, int], thickness: int) -> None:
    """Draw a rounded rectangle with smooth circular corners."""
    w = x2 - x1
    h = y2 - y1
    r = max(4, min(w // 5, h // 5, 20))

    # Four straight sides
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color_bgr, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color_bgr, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color_bgr, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color_bgr, thickness)

    # Four rounded corner arcs
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color_bgr, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color_bgr, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color_bgr, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color_bgr, thickness)


def _draw_hud(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    face: FaceDetection,
    color_bgr: Tuple[int, int, int],
    thickness: int,
    show_confidence: bool = False,
) -> None:
    """Draw a technical HUD overlay with reticle brackets, crosshair ticks, and optional target badge."""
    w = x2 - x1
    h = y2 - y1

    # Tactical corner brackets
    _draw_corners(img, x1, y1, x2, y2, color_bgr, thickness)

    # Crosshair tick marks on center edges
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    tick_len = max(4, min(w, h) // 12)

    cv2.line(img, (cx, y1), (cx, y1 + tick_len), color_bgr, max(1, thickness - 1))
    cv2.line(img, (cx, y2), (cx, y2 - tick_len), color_bgr, max(1, thickness - 1))
    cv2.line(img, (x1, cy), (x1 + tick_len, cy), color_bgr, max(1, thickness - 1))
    cv2.line(img, (x2, cy), (x2 - tick_len, cy), color_bgr, max(1, thickness - 1))

    # Badge rendering when confidence is enabled
    if show_confidence:
        score_text = f"TARGET {format_confidence(face.confidence)}"
        draw_text_badge(img, score_text, (x1, max(16, y1 - 6)), text_color=color_bgr, font_scale=0.42, padding=3)


def draw_face_overlay(
    image: np.ndarray,
    face: FaceDetection,
    shape: str = "rectangle",
    color: Union[str, Tuple[int, int, int]] = "#00FF88",
    thickness: int = 3,
    show_confidence: bool = False,
    copy: bool = True,
) -> np.ndarray:
    """Draw a face overlay on an image for a single face detection.

    Args:
        image: Source BGR image NumPy array.
        face: FaceDetection instance to render.
        shape: One of 'rectangle', 'corners', 'circle', 'rounded', 'hud'. Default is 'rectangle'.
        color: RGB hex string (e.g. '#00FF88') or documented color name. Default is '#00FF88'.
        thickness: Line stroke thickness in pixels (default 3).
        show_confidence: Whether to display the detection confidence score near the face (default False).
        copy: Whether to copy the image before drawing (default True, ensuring the original is untouched).

    Returns:
        NumPy array with the overlay rendered.
    """
    valid_shape, valid_thickness = validate_render_parameters(shape, thickness)
    bgr_color = parse_color_to_bgr(color)

    out = image.copy() if copy else image
    img_h, img_w = out.shape[:2]

    # Clip coordinates to image boundaries
    bbox = face.bbox
    x1 = max(0, min(bbox.x1, img_w - 1))
    y1 = max(0, min(bbox.y1, img_h - 1))
    x2 = max(0, min(bbox.x2, img_w - 1))
    y2 = max(0, min(bbox.y2, img_h - 1))

    if x2 <= x1 or y2 <= y1:
        # Out-of-bounds or zero-area box, skip drawing
        return out

    if valid_shape == "rectangle":
        _draw_rectangle(out, x1, y1, x2, y2, bgr_color, valid_thickness)
    elif valid_shape == "corners":
        _draw_corners(out, x1, y1, x2, y2, bgr_color, valid_thickness)
    elif valid_shape == "circle":
        _draw_circle(out, x1, y1, x2, y2, bgr_color, valid_thickness)
    elif valid_shape == "rounded":
        _draw_rounded(out, x1, y1, x2, y2, bgr_color, valid_thickness)
    elif valid_shape == "hud":
        _draw_hud(out, x1, y1, x2, y2, face, bgr_color, valid_thickness, show_confidence=show_confidence)

    # Render confidence label for non-HUD shapes if requested
    if show_confidence and valid_shape != "hud":
        conf_text = format_confidence(face.confidence)
        label_y = max(18, y1 - 6)
        draw_text_badge(out, conf_text, (x1, label_y), text_color=bgr_color, font_scale=0.42, padding=3)

    return out


def draw_faces(
    image: np.ndarray,
    faces: Sequence[FaceDetection],
    shape: str = "rectangle",
    color: Union[str, Tuple[int, int, int]] = "#00FF88",
    thickness: int = 3,
    show_confidence: bool = False,
    copy: bool = True,
) -> np.ndarray:
    """Draw face overlays for multiple detected faces onto an image.

    Renders each face independently without altering the original image when copy=True.

    Args:
        image: Source BGR image NumPy array.
        faces: Sequence of FaceDetection instances to render.
        shape: One of 'rectangle', 'corners', 'circle', 'rounded', 'hud'. Default is 'rectangle'.
        color: RGB hex string (e.g. '#00FF88') or documented color name. Default is '#00FF88'.
        thickness: Line stroke thickness in pixels (default 3).
        show_confidence: Whether to render confidence percentages near each face (default False).
        copy: Whether to copy the image before drawing (default True).

    Returns:
        NumPy array containing all rendered face overlays.
    """
    valid_shape, valid_thickness = validate_render_parameters(shape, thickness)
    bgr_color = parse_color_to_bgr(color)

    out = image.copy() if copy else image
    for face in faces:
        draw_face_overlay(
            out,
            face,
            shape=valid_shape,
            color=bgr_color,
            thickness=valid_thickness,
            show_confidence=show_confidence,
            copy=False,
        )

    return out
