"""Animated Live FX visualization engine for detekface.

Provides a modern computer-vision scanner overlay layer on top of dense facial landmarks:
- Timestamp-based animated horizontal laser scanning line
- Highlighted facial contour & connected landmark mesh
- Prominent anatomical keypoint markers
- Smooth face-locking HUD brackets and acquisition reticles
- Technical telemetry labels ([FACE LOCKED], [SCANNING], [TARGET ACQUIRED])
- Multi-face tracking with temporal smoothing and grace-period occlusion recovery
- Independent styling: 'scan', 'neon', 'wireframe', 'minimal'
- Clean distinction between visual scanner effects and biometric shape analysis
"""

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import cv2
import numpy as np

from detekface.drawing import parse_color_to_bgr
from detekface.exceptions import InvalidImageError, InvalidSettingError
from detekface.mesh import (
    FACEMESH_CONTOURS,
    FACEMESH_FACE_OVAL,
    FACEMESH_LEFT_EYE,
    FACEMESH_LIPS,
    FACEMESH_RIGHT_EYE,
    NEON_KEY_LANDMARKS,
    convert_landmarks_to_pixels,
)
from detekface.types import BoundingBox, FaceDetection

SUPPORTED_LIVE_FX_STYLES: Tuple[str, ...] = (
    "scan",
    "neon",
    "wireframe",
    "minimal",
)


def validate_live_fx_style(style: str) -> str:
    """Validate Live FX visualization style name.

    Raises:
        InvalidSettingError: If the style name is invalid or unsupported.
    """
    if not isinstance(style, str):
        raise InvalidSettingError(
            message=f"Live FX style must be a string, got {type(style).__name__}.",
            hint=f"Supported live FX styles are: {sorted(SUPPORTED_LIVE_FX_STYLES)}.",
        )
    clean = style.strip().lower()
    if clean not in SUPPORTED_LIVE_FX_STYLES:
        raise InvalidSettingError(
            message=f"Unsupported live FX style: '{style}'.",
            hint=f"Supported live FX styles are: {sorted(SUPPORTED_LIVE_FX_STYLES)}.",
        )
    return clean


class LiveFXTrackedFace:
    """Stateful temporal tracker for an individual detected face across animation frames."""

    def __init__(
        self,
        face_id: int,
        detection: FaceDetection,
        timestamp: float,
    ) -> None:
        self.face_id = face_id
        self.bbox: BoundingBox = detection.bbox
        self.dense_landmarks: Optional[Sequence[Any]] = detection.dense_landmarks
        self.confidence: float = detection.confidence
        self.face_shape = detection.face_shape

        self.first_seen: float = timestamp
        self.last_seen: float = timestamp
        self.lock_progress: float = 0.0  # Ramps from 0.0 to 1.0
        self.disappeared_frames: int = 0
        self.phase_offset: float = (face_id * 0.35) % 1.0

    def update(self, detection: FaceDetection, timestamp: float) -> None:
        """Update tracker with freshly detected face data and advance lock progress."""
        # Exponential smoothing on bounding box to prevent visual jitter
        alpha = 0.65
        self.bbox = BoundingBox(
            x=int(round(alpha * detection.bbox.x + (1 - alpha) * self.bbox.x)),
            y=int(round(alpha * detection.bbox.y + (1 - alpha) * self.bbox.y)),
            width=int(round(alpha * detection.bbox.width + (1 - alpha) * self.bbox.width)),
            height=int(round(alpha * detection.bbox.height + (1 - alpha) * self.bbox.height)),
        )
        self.dense_landmarks = detection.dense_landmarks
        self.confidence = detection.confidence
        self.face_shape = detection.face_shape

        dt = max(0.0, timestamp - self.last_seen)
        self.last_seen = timestamp
        self.disappeared_frames = 0

        # Ramp lock progress smoothly over ~0.25 seconds
        self.lock_progress = min(1.0, self.lock_progress + dt / 0.25)


class LiveFXRenderer:
    """Reusable Live FX animation renderer for real-time camera and still-image scanner overlays.

    Attributes:
        style: Visualization mode ('scan', 'neon', 'wireframe', 'minimal').
        color: Base accent color as hex code or BGR tuple.
        scan_speed: Scanning vertical cycle frequency in Hz (cycles per second).
        show_labels: Whether to render technical telemetry badges.
        show_lock_reticle: Whether to render animated HUD corner brackets.
        grace_period_sec: Time in seconds a briefly lost face is retained before fading.
    """

    def __init__(
        self,
        style: str = "scan",
        color: Union[str, Tuple[int, int, int]] = "#00FF88",
        scan_speed: float = 0.75,
        show_labels: bool = True,
        show_lock_reticle: bool = True,
        grace_period_sec: float = 0.25,
    ) -> None:
        self.style = validate_live_fx_style(style)
        self.color = parse_color_to_bgr(color)
        self.scan_speed = max(0.1, float(scan_speed))
        self.show_labels = bool(show_labels)
        self.show_lock_reticle = bool(show_lock_reticle)
        self.grace_period_sec = max(0.05, float(grace_period_sec))

        self._tracks: Dict[int, LiveFXTrackedFace] = {}
        self._next_id: int = 1
        self._last_time: Optional[float] = None

    def reset(self) -> None:
        """Reset internal face tracks and timer state."""
        self._tracks.clear()
        self._next_id = 1
        self._last_time = None

    def _match_faces(self, faces: Sequence[FaceDetection], current_time: float) -> None:
        """Associate detections with existing tracks using center-distance heuristic."""
        if not faces:
            # Mark all existing tracks as temporarily lost and purge expired tracks
            for track in self._tracks.values():
                track.disappeared_frames += 1
            dead_tracks = [
                tid for tid, t in self._tracks.items() if (current_time - t.last_seen) > self.grace_period_sec
            ]
            for tid in dead_tracks:
                del self._tracks[tid]
            return

        unmatched_detections = list(faces)
        matched_tracks: Set[int] = set()

        for track_id, track in list(self._tracks.items()):
            tx = track.bbox.x + track.bbox.width / 2.0
            ty = track.bbox.y + track.bbox.height / 2.0
            t_diag = math.hypot(track.bbox.width, track.bbox.height)

            best_idx: Optional[int] = None
            best_dist = float("inf")

            for i, det in enumerate(unmatched_detections):
                dx = (det.bbox.x + det.bbox.width / 2.0) - tx
                dy = (det.bbox.y + det.bbox.height / 2.0) - ty
                dist = math.hypot(dx, dy)
                if dist < best_dist and dist < t_diag * 0.75:
                    best_dist = dist
                    best_idx = i

            if best_idx is not None:
                matched_det = unmatched_detections.pop(best_idx)
                track.update(matched_det, current_time)
                matched_tracks.add(track_id)
            else:
                track.disappeared_frames += 1

        # Purge tracks that exceed grace period
        dead_tracks = [
            tid
            for tid, t in self._tracks.items()
            if tid not in matched_tracks and (current_time - t.last_seen) > self.grace_period_sec
        ]
        for tid in dead_tracks:
            del self._tracks[tid]

        # Create new tracks for remaining unmatched detections
        for det in unmatched_detections:
            self._tracks[self._next_id] = LiveFXTrackedFace(
                face_id=self._next_id,
                detection=det,
                timestamp=current_time,
            )
            self._next_id += 1

    def render(
        self,
        image: np.ndarray,
        faces: Optional[Sequence[FaceDetection]] = None,
        timestamp: Optional[float] = None,
        copy: bool = True,
    ) -> np.ndarray:
        """Render animated Live FX scanning visualizer onto an image or video frame.

        Args:
            image: Source BGR image NumPy array.
            faces: Current frame detections. If None, continues rendering existing tracks.
            timestamp: Clock time in seconds (defaults to time.perf_counter()).
            copy: Whether to render on a copy of the image (default True).

        Returns:
            Annotated BGR NumPy image array.
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise InvalidImageError(
                message="Cannot render Live FX on invalid or empty image array.",
                hint="Provide a valid NumPy uint8 image array.",
            )

        out = image.copy() if copy else image
        now = time.perf_counter() if timestamp is None else float(timestamp)
        self._last_time = now

        if faces is not None:
            self._match_faces(faces, now)

        # Render all active tracks within grace period
        for track in list(self._tracks.values()):
            if (now - track.last_seen) <= self.grace_period_sec:
                self._draw_track_fx(out, track, now)

        return out

    def _draw_track_fx(
        self,
        image: np.ndarray,
        track: LiveFXTrackedFace,
        current_time: float,
    ) -> None:
        """Render Live FX visual layers for a single tracked face."""
        h, w = image.shape[:2]
        bbox = track.bbox

        # Clamp box to image bounds
        x1 = max(0, min(bbox.x1, w - 1))
        y1 = max(0, min(bbox.y1, h - 1))
        x2 = max(0, min(bbox.x2, w - 1))
        y2 = max(0, min(bbox.y2, h - 1))
        bw = x2 - x1
        bh = y2 - y1

        if bw <= 4 or bh <= 4:
            return

        # 1. Subtle Face-Locking Reticle Corners
        if self.show_lock_reticle:
            self._draw_lock_reticle(image, x1, y1, x2, y2, track.lock_progress)

        # 2. Dense Facial Mesh & Contour Elements (if landmarks are present)
        pixel_points: Optional[List[Optional[Tuple[int, int]]]] = None
        if track.dense_landmarks:
            pixel_points = convert_landmarks_to_pixels(track.dense_landmarks, image_width=w, image_height=h)
            self._draw_mesh_layers(image, pixel_points, bw)

        # 3. Animated Horizontal Laser Scanning Line (for 'scan' and 'neon' styles)
        if self.style in ("scan", "neon"):
            # Continuous sinusoidal vertical sweep
            phase = (current_time * self.scan_speed + track.phase_offset) % 1.0
            t = (math.sin(phase * 2.0 * math.pi) + 1.0) / 2.0
            scan_y = int(round(y1 + t * bh))
            self._draw_scanning_beam(image, x1, x2, scan_y, h)

        # 4. Small Technical HUD Telemetry Badges
        if self.show_labels:
            self._draw_technical_labels(image, x1, y1, x2, y2, track)

    def _draw_lock_reticle(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        lock_progress: float,
    ) -> None:
        """Render technical HUD corner brackets that converge as lock is acquired."""
        bw = x2 - x1
        bh = y2 - y1
        bracket_len = max(6, min(bw, bh) // 5)

        # Reticle margin expands slightly when locking (animated convergence)
        reticle_offset = int(round((1.0 - lock_progress) * 8.0))
        rx1 = max(0, x1 - reticle_offset)
        ry1 = max(0, y1 - reticle_offset)
        rx2 = min(image.shape[1] - 1, x2 + reticle_offset)
        ry2 = min(image.shape[0] - 1, y2 + reticle_offset)

        # Reticle color: amber while acquiring, primary accent when locked
        if lock_progress < 0.95:
            reticle_color = (0, 200, 255)  # Amber / orange acquiring lock
        else:
            reticle_color = self.color

        th = 2 if self.style != "minimal" else 1

        # Top-Left
        cv2.line(image, (rx1, ry1), (rx1 + bracket_len, ry1), reticle_color, th, cv2.LINE_AA)
        cv2.line(image, (rx1, ry1), (rx1, ry1 + bracket_len), reticle_color, th, cv2.LINE_AA)
        # Top-Right
        cv2.line(image, (rx2, ry1), (rx2 - bracket_len, ry1), reticle_color, th, cv2.LINE_AA)
        cv2.line(image, (rx2, ry1), (rx2, ry1 + bracket_len), reticle_color, th, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(image, (rx1, ry2), (rx1 + bracket_len, ry2), reticle_color, th, cv2.LINE_AA)
        cv2.line(image, (rx1, ry2), (rx1, ry2 - bracket_len), reticle_color, th, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(image, (rx2, ry2), (rx2 - bracket_len, ry2), reticle_color, th, cv2.LINE_AA)
        cv2.line(image, (rx2, ry2), (rx2, ry2 - bracket_len), reticle_color, th, cv2.LINE_AA)

    def _draw_mesh_layers(
        self,
        image: np.ndarray,
        pixel_points: List[Optional[Tuple[int, int]]],
        face_span: int,
    ) -> None:
        """Render topological wireframe, contours, and anchor points according to style."""
        total_pts = len(pixel_points)

        # Style-specific line selection
        if self.style == "minimal":
            connections = FACEMESH_FACE_OVAL + FACEMESH_LEFT_EYE + FACEMESH_RIGHT_EYE + FACEMESH_LIPS
            line_thickness = 1
        elif self.style in ("scan", "wireframe"):
            connections = FACEMESH_CONTOURS
            line_thickness = 1
        elif self.style == "neon":
            connections = FACEMESH_CONTOURS
            line_thickness = max(1, min(2, int(round(face_span / 180.0))))
        else:
            connections = FACEMESH_CONTOURS
            line_thickness = 1

        # Draw wireframe lines
        for s_idx, e_idx in connections:
            if s_idx < total_pts and e_idx < total_pts:
                p1 = pixel_points[s_idx]
                p2 = pixel_points[e_idx]
                if p1 is not None and p2 is not None:
                    # Emphasize face oval in neon/scan
                    if self.style in ("neon", "scan") and (s_idx, e_idx) in FACEMESH_FACE_OVAL:
                        cv2.line(image, p1, p2, self.color, line_thickness + 1, cv2.LINE_AA)
                    else:
                        cv2.line(image, p1, p2, self.color, line_thickness, cv2.LINE_AA)

        # Draw selected key landmark points for neon and scan styles
        if self.style in ("neon", "scan"):
            pt_radius = max(1, min(3, int(round(face_span / 160.0))))
            for idx in NEON_KEY_LANDMARKS:
                if idx < total_pts and pixel_points[idx] is not None:
                    # High-contrast white center
                    cv2.circle(image, pixel_points[idx], pt_radius + 1, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_scanning_beam(
        self,
        image: np.ndarray,
        x1: int,
        x2: int,
        scan_y: int,
        img_h: int,
    ) -> None:
        """Render high-contrast anti-aliased horizontal laser beam across face bounds."""
        if 0 <= scan_y < img_h:
            # Core bright laser ray
            cv2.line(image, (x1, scan_y), (x2, scan_y), (255, 255, 255), 1, cv2.LINE_AA)

            # Secondary accent glow lines immediately above and below
            for dy in (-1, 1):
                gy = scan_y + dy
                if 0 <= gy < img_h:
                    cv2.line(image, (x1 + 4, gy), (x2 - 4, gy), self.color, 1, cv2.LINE_AA)

    def _draw_technical_labels(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        track: LiveFXTrackedFace,
    ) -> None:
        """Render small telemetry tags: [FACE LOCKED] / [SCANNING] and shape analysis."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        font_thick = 1

        # Primary status tag
        if track.lock_progress >= 0.95:
            status_text = f"FACE LOCKED #{track.face_id}"
            badge_color = self.color
        else:
            status_text = f"SCANNING #{track.face_id}"
            badge_color = (0, 200, 255)

        # Compute text size
        (tw, th), baseline = cv2.getTextSize(status_text, font, font_scale, font_thick)
        label_x = x1
        label_y = max(th + 4, y1 - 8)

        # Technical dark background plate
        bg_x1 = max(0, label_x - 3)
        bg_y1 = max(0, label_y - th - 3)
        bg_x2 = min(image.shape[1] - 1, label_x + tw + 3)
        bg_y2 = min(image.shape[0] - 1, label_y + baseline + 1)

        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), badge_color, 1)
        cv2.putText(image, status_text, (label_x, label_y), font, font_scale, badge_color, font_thick, cv2.LINE_AA)

        # Biometric Face Shape result tag (if analyzed)
        if track.face_shape is not None:
            label_name = track.face_shape.label.upper()
            if label_name == "ANALYZING":
                shape_text = "GEOMETRY: ANALYZING..."
                shape_color = (0, 215, 255)
            elif label_name == "UNDETERMINED":
                shape_text = "GEOMETRY: UNDETERMINED"
                shape_color = (0, 140, 255)
            else:
                pct = int(round(track.face_shape.fit_score * 100))
                shape_text = f"SHAPE: {label_name} ({pct}%)"
                shape_color = self.color

            (stw, sth), sbase = cv2.getTextSize(shape_text, font, font_scale, font_thick)
            s_y = min(image.shape[0] - 6, y2 + sth + 8)

            sbg_x1 = max(0, x1 - 3)
            sbg_y1 = max(0, s_y - sth - 3)
            sbg_x2 = min(image.shape[1] - 1, x1 + stw + 3)
            sbg_y2 = min(image.shape[0] - 1, s_y + sbase + 1)

            cv2.rectangle(image, (sbg_x1, sbg_y1), (sbg_x2, sbg_y2), (20, 20, 20), -1)
            cv2.rectangle(image, (sbg_x1, sbg_y1), (sbg_x2, sbg_y2), shape_color, 1)
            cv2.putText(image, shape_text, (x1, s_y), font, font_scale, shape_color, font_thick, cv2.LINE_AA)
