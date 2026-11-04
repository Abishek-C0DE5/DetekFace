"""Face shape analysis and geometric classification engine.

Provides dense facial landmark extraction (478 points via MediaPipe Face Landmarker),
geometric contour measurement, heuristic face shape classification (oval, round, square,
heart, diamond, oblong, undetermined), temporal tracking/smoothing for webcam feeds,
and futuristic biometric scanning HUD visualization.

Both Mode 1 (still images) and Mode 2 (real-time webcam) share this identical engine.
"""

from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from detekface.exceptions import DependencyNotFoundError
from detekface.models import resolve_mediapipe_model
from detekface.types import BoundingBox, FaceDetection, FaceShapeResult, Point

# Canonical MediaPipe Face Mesh silhouette / oval contour landmark indices (36 perimeter points)
FACEMESH_FACE_OVAL: Tuple[int, ...] = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)

# Supported face shape categories
SUPPORTED_FACE_SHAPES: Tuple[str, ...] = (
    "oval",
    "round",
    "square",
    "heart",
    "diamond",
    "oblong",
    "undetermined",
)


class DenseLandmarkExtractor:
    """Lazy-loaded dense facial landmark extractor using MediaPipe Face Landmarker.

    Extracts 478 3D facial landmarks per detected face.
    Requires the optional 'mediapipe' dependency.
    """

    _instance: Optional["DenseLandmarkExtractor"] = None

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        num_faces: int = 10,
        min_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp  # type: ignore[import-untyped]
            from mediapipe.tasks.python import vision  # type: ignore[import-untyped]
            from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore[import-untyped]
        except ImportError as e:
            raise DependencyNotFoundError(
                message="Face shape analysis requires the optional 'mediapipe' package.",
                hint='Install it with: pip install "detekface[mediapipe]"',
            ) from e

        self._mp = mp
        self._resolved_model_path = resolve_mediapipe_model(
            model=model_path,
            cache_dir=cache_dir,
            model_name="face_landmarker",
        )

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._resolved_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    @classmethod
    def get_default(cls, cache_dir: Optional[Union[str, Path]] = None) -> "DenseLandmarkExtractor":
        """Get or create singleton DenseLandmarkExtractor instance."""
        if cls._instance is None:
            cls._instance = cls(cache_dir=cache_dir)
        return cls._instance

    def extract(self, bgr_image: np.ndarray) -> List[List[Point]]:
        """Extract dense landmarks from an image for all visible faces.

        Args:
            bgr_image: BGR format NumPy uint8 image array.

        Returns:
            List of landmark lists, each containing 478 Point instances.
        """
        if not isinstance(bgr_image, np.ndarray) or bgr_image.size == 0:
            return []

        h, w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        detection_result = self._landmarker.detect(mp_image)

        all_faces: List[List[Point]] = []
        if not detection_result.face_landmarks:
            return all_faces

        for face_lms in detection_result.face_landmarks:
            points: List[Point] = [Point(x=int(round(lm.x * w)), y=int(round(lm.y * h))) for lm in face_lms]
            all_faces.append(points)

        return all_faces


def _euclidean_dist(p1: Point, p2: Point) -> float:
    """Euclidean distance between two Point coordinates."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def _calculate_angle_deg(a: Point, b: Point, c: Point) -> float:
    """Calculate the interior angle ABC in degrees with vertex at point B."""
    v1 = np.array([a.x - b.x, a.y - b.y], dtype=np.float64)
    v2 = np.array([c.x - b.x, c.y - b.y], dtype=np.float64)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 180.0
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_angle)))


def _gaussian_fit(val: float, target: float, sigma: float) -> float:
    """Compute Gaussian similarity score between 0.0 and 1.0."""
    return float(math.exp(-0.5 * ((val - target) / max(1e-5, sigma)) ** 2))


def analyze_face_geometry(
    landmarks: Sequence[Point],
    image_shape: Tuple[int, int],
) -> FaceShapeResult:
    """Perform geometric analysis and face shape classification from dense facial landmarks.

    Args:
        landmarks: Dense sequence of facial keypoints (expects at least 468/478 points).
        image_shape: Image dimensions as (height, width).

    Returns:
        FaceShapeResult with classified label, fit score, alternatives, measurements, and warnings.
    """
    img_h, img_w = image_shape
    warnings: List[str] = []

    if len(landmarks) < 468:
        return FaceShapeResult(
            label="undetermined",
            fit_score=0.0,
            alternatives={s: 0.0 for s in SUPPORTED_FACE_SHAPES if s != "undetermined"},
            measurements={},
            quality_status="rejected",
            warnings=["Insufficient facial landmarks available for geometric analysis."],
        )

    # Key landmark indices:
    # 10: Trichion / top forehead center
    # 152: Menton / lowest chin point
    # 234: Right cheekbone (viewer left)
    # 454: Left cheekbone (viewer right)
    # 54, 284: Right and left upper forehead / temple boundary
    # 58, 288: Right and left jaw gonion angles
    # 150, 378: Chin taper points
    # 1: Nose tip
    # 33: Right eye outer corner, 263: Left eye outer corner
    p_forehead_top = landmarks[10]
    p_chin = landmarks[152]
    p_cheek_r = landmarks[234]
    p_cheek_l = landmarks[454]
    p_temple_r = landmarks[54]
    p_temple_l = landmarks[284]
    p_jaw_r = landmarks[58]
    p_jaw_l = landmarks[288]
    p_chin_r = landmarks[150]
    p_chin_l = landmarks[378]
    p_nose = landmarks[1]
    p_eye_r = landmarks[33]
    p_eye_l = landmarks[263]

    # 1. Pose Quality Assessment (Yaw and Head Tilt)
    # Distance from nose tip to right eye vs left eye
    dist_nose_eyer = _euclidean_dist(p_nose, p_eye_r)
    dist_nose_eyel = _euclidean_dist(p_nose, p_eye_l)
    total_eye_dist = dist_nose_eyer + dist_nose_eyel
    if total_eye_dist > 0:
        yaw_asymmetry = abs(dist_nose_eyer - dist_nose_eyel) / total_eye_dist
    else:
        yaw_asymmetry = 0.0

    # If yaw asymmetry exceeds 0.28 (approx > 25 degrees yaw), frontal proportions are invalid
    if yaw_asymmetry > 0.28:
        warnings.append(
            "Face yaw angle exceeds frontal tolerance (>25 deg); face shape analysis requires a near-frontal view."
        )
        return FaceShapeResult(
            label="undetermined",
            fit_score=0.0,
            alternatives={s: 0.0 for s in SUPPORTED_FACE_SHAPES if s != "undetermined"},
            measurements={
                "yaw_asymmetry": round(yaw_asymmetry, 3),
            },
            quality_status="rejected",
            warnings=warnings,
        )

    # 2. Key Facial Measurements (in pixels)
    face_length = _euclidean_dist(p_forehead_top, p_chin)
    cheekbone_width = _euclidean_dist(p_cheek_r, p_cheek_l)
    forehead_width = _euclidean_dist(p_temple_r, p_temple_l)
    jaw_width = _euclidean_dist(p_jaw_r, p_jaw_l)
    chin_width = _euclidean_dist(p_chin_r, p_chin_l)

    # Resolution warning for very small faces
    if cheekbone_width < 45 or face_length < 45:
        warnings.append("Low face resolution may reduce geometric measurement precision.")

    # Guard against division by zero
    safe_cw = max(1.0, cheekbone_width)
    safe_jw = max(1.0, jaw_width)
    safe_fw = max(1.0, forehead_width)

    # 3. Geometric Proportion Ratios
    length_to_width = face_length / safe_cw
    forehead_to_jaw = forehead_width / safe_jw
    cheek_to_jaw = cheekbone_width / safe_jw
    cheek_to_forehead = cheekbone_width / safe_fw
    chin_to_jaw = chin_width / safe_jw

    # 4. Jaw Angles
    # Angle at right jaw vertex (234 -> 58 -> 152) and left jaw vertex (454 -> 288 -> 152)
    angle_r = _calculate_angle_deg(p_cheek_r, p_jaw_r, p_chin)
    angle_l = _calculate_angle_deg(p_cheek_l, p_jaw_l, p_chin)
    jaw_angle = (angle_r + angle_l) / 2.0

    measurements: Dict[str, float] = {
        "face_length": round(face_length, 1),
        "cheekbone_width": round(cheekbone_width, 1),
        "forehead_width": round(forehead_width, 1),
        "jaw_width": round(jaw_width, 1),
        "chin_width": round(chin_width, 1),
        "length_to_width_ratio": round(length_to_width, 2),
        "forehead_to_jaw_ratio": round(forehead_to_jaw, 2),
        "cheek_to_jaw_ratio": round(cheek_to_jaw, 2),
        "cheek_to_forehead_ratio": round(cheek_to_forehead, 2),
        "jaw_angle": round(jaw_angle, 1),
    }

    # 5. Heuristic Geometric Scoring for the 6 Canonical Face Shapes
    # Oval: balanced length (> width), cheekbones widest, soft curved jaw, forehead slightly wider than jaw
    score_oval = (
        _gaussian_fit(length_to_width, 1.38, 0.16) * 0.40
        + _gaussian_fit(forehead_to_jaw, 1.08, 0.15) * 0.25
        + _gaussian_fit(cheek_to_jaw, 1.15, 0.15) * 0.20
        + _gaussian_fit(jaw_angle, 138.0, 15.0) * 0.15
    )

    # Round: length ≈ width, cheekbones widest, soft rounded jawline
    score_round = (
        _gaussian_fit(length_to_width, 1.08, 0.14) * 0.45
        + _gaussian_fit(forehead_to_jaw, 1.00, 0.15) * 0.20
        + _gaussian_fit(cheek_to_jaw, 1.16, 0.16) * 0.20
        + _gaussian_fit(jaw_angle, 142.0, 16.0) * 0.15
    )

    # Square: length ≈ width, strong wide jaw (jaw width ≈ cheekbone width ≈ forehead), sharp angular jawline
    score_square = (
        _gaussian_fit(length_to_width, 1.12, 0.15) * 0.35
        + _gaussian_fit(cheek_to_jaw, 1.02, 0.08) * 0.30
        + _gaussian_fit(forehead_to_jaw, 0.98, 0.10) * 0.20
        + _gaussian_fit(jaw_angle, 116.0, 14.0) * 0.15
    )

    # Oblong / Rectangle: length >> width (ratio >= 1.5), uniform width along forehead, cheeks, and jaw
    score_oblong = (
        _gaussian_fit(length_to_width, 1.62, 0.18) * 0.50
        + _gaussian_fit(cheek_to_jaw, 1.05, 0.12) * 0.25
        + _gaussian_fit(forehead_to_jaw, 1.00, 0.12) * 0.25
    )

    # Heart: wide forehead, cheekbones prominent, tapering down to a pointed/narrow chin
    score_heart = (
        _gaussian_fit(length_to_width, 1.34, 0.16) * 0.25
        + _gaussian_fit(forehead_to_jaw, 1.28, 0.18) * 0.35
        + _gaussian_fit(cheek_to_jaw, 1.25, 0.18) * 0.25
        + _gaussian_fit(chin_to_jaw, 0.30, 0.10) * 0.15
    )

    # Diamond: cheekbones widest, narrow forehead and narrow jaw / pointed chin
    score_diamond = (
        _gaussian_fit(length_to_width, 1.36, 0.16) * 0.25
        + _gaussian_fit(cheek_to_forehead, 1.26, 0.16) * 0.35
        + _gaussian_fit(cheek_to_jaw, 1.28, 0.16) * 0.25
        + _gaussian_fit(forehead_to_jaw, 1.02, 0.15) * 0.15
    )

    raw_scores: Dict[str, float] = {
        "oval": max(0.0, min(1.0, score_oval)),
        "round": max(0.0, min(1.0, score_round)),
        "square": max(0.0, min(1.0, score_square)),
        "heart": max(0.0, min(1.0, score_heart)),
        "diamond": max(0.0, min(1.0, score_diamond)),
        "oblong": max(0.0, min(1.0, score_oblong)),
    }

    # Sort shapes by fit score descending
    sorted_shapes = sorted(raw_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_shape, best_score = sorted_shapes[0]

    # Fallback to undetermined if confidence is low
    if best_score < 0.35:
        warnings.append("Geometric proportions did not strongly match any canonical face shape.")
        return FaceShapeResult(
            label="undetermined",
            fit_score=round(best_score, 2),
            alternatives={k: round(v, 2) for k, v in sorted_shapes},
            measurements=measurements,
            quality_status="uncertain",
            warnings=warnings,
        )

    alternatives = {k: round(v, 2) for k, v in sorted_shapes if k != best_shape}

    quality_status = "good" if not warnings else "warning"

    return FaceShapeResult(
        label=best_shape,
        fit_score=round(best_score, 2),
        alternatives=alternatives,
        measurements=measurements,
        quality_status=quality_status,
        warnings=warnings,
    )


def match_and_attach_face_shapes(
    faces: Sequence[FaceDetection],
    landmark_sets: List[List[Point]],
    image_shape: Tuple[int, int],
    skip_shape_analysis: bool = False,
) -> List[FaceDetection]:
    """Associate extracted dense landmarks with detected face bounding boxes and perform analysis.

    Args:
        faces: Sequence of detected FaceDetection objects from the primary detector.
        landmark_sets: Extracted dense landmarks (one list of 478 points per localized face).
        image_shape: Dimensions of the image as (height, width).
        skip_shape_analysis: If True, attaches dense landmarks for mesh visualization without
            calculating geometric face shape proportions or setting face_shape.

    Returns:
        List of FaceDetection objects with face_shape and dense_landmarks attached.
    """
    if not faces:
        return []

    # If no landmarks were found across the image
    if not landmark_sets:
        enriched: List[FaceDetection] = []
        for face in faces:
            shape_result = None
            if not skip_shape_analysis:
                shape_result = FaceShapeResult(
                    label="undetermined",
                    fit_score=0.0,
                    alternatives={s: 0.0 for s in SUPPORTED_FACE_SHAPES if s != "undetermined"},
                    measurements={},
                    quality_status="rejected",
                    warnings=["No dense facial landmarks were detected on this face."],
                )
            enriched.append(replace(face, face_shape=shape_result, dense_landmarks=None))
        return enriched

    # Match each face bounding box to the closest landmark cluster center
    unassigned_lms = list(landmark_sets)
    enriched_faces: List[FaceDetection] = []

    for face in faces:
        face_center_x = face.bbox.x + face.bbox.width / 2.0
        face_center_y = face.bbox.y + face.bbox.height / 2.0

        best_idx: Optional[int] = None
        min_dist = float("inf")

        for idx, lms in enumerate(unassigned_lms):
            # Compute centroid of landmark silhouette
            pts_x = [lms[i].x for i in FACEMESH_FACE_OVAL]
            pts_y = [lms[i].y for i in FACEMESH_FACE_OVAL]
            lm_cx = sum(pts_x) / len(pts_x)
            lm_cy = sum(pts_y) / len(pts_y)

            dist = math.hypot(face_center_x - lm_cx, face_center_y - lm_cy)
            # Center must be reasonably inside or near the bounding box
            if dist < min_dist and dist < max(face.bbox.width, face.bbox.height):
                min_dist = dist
                best_idx = idx

        if best_idx is not None:
            matched_lms = unassigned_lms.pop(best_idx)
            shape_result = None
            if not skip_shape_analysis:
                shape_result = analyze_face_geometry(matched_lms, image_shape)
            enriched_faces.append(replace(face, face_shape=shape_result, dense_landmarks=matched_lms))
        else:
            # Fallback if no matching landmark set
            shape_result = None
            if not skip_shape_analysis:
                shape_result = FaceShapeResult(
                    label="undetermined",
                    fit_score=0.0,
                    alternatives={s: 0.0 for s in SUPPORTED_FACE_SHAPES if s != "undetermined"},
                    measurements={},
                    quality_status="rejected",
                    warnings=["Could not associate dense facial landmarks with this face."],
                )
            enriched_faces.append(replace(face, face_shape=shape_result, dense_landmarks=None))

    return enriched_faces


class TemporalShapeTracker:
    """Temporal face shape tracker and smoothing filter for real-time video feeds.

    Maintains a rolling window of recent shape predictions per face across frames,
    eliminating flickering between shape classifications, tracking identities via IoU,
    and generating scan-line animation state for biometric HUD rendering.
    """

    def __init__(self, window_size: int = 10, iou_threshold: float = 0.3) -> None:
        self.window_size = max(1, window_size)
        self.iou_threshold = iou_threshold
        # Map tracker_id -> dict of tracking history
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_track_id = 0
        self._frame_index = 0

    def _compute_iou(self, b1: BoundingBox, b2: BoundingBox) -> float:
        x1 = max(b1.x, b2.x)
        y1 = max(b1.y, b2.y)
        x2 = min(b1.x + b1.width, b2.x + b2.width)
        y2 = min(b1.y + b1.height, b2.y + b2.height)

        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        intersection = w * h
        union = b1.area + b2.area - intersection
        return intersection / float(union) if union > 0 else 0.0

    def update(self, faces: Sequence[FaceDetection]) -> List[FaceDetection]:
        """Update tracker with new frame detections and return smoothed results."""
        self._frame_index += 1
        updated_faces: List[FaceDetection] = []
        matched_track_ids = set()

        for face in faces:
            # Find best matching existing track by IoU
            best_track_id: Optional[int] = None
            best_iou = self.iou_threshold

            for track_id, track in self._tracks.items():
                if track_id in matched_track_ids:
                    continue
                iou = self._compute_iou(face.bbox, track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is None:
                # Assign new track
                best_track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[best_track_id] = {
                    "bbox": face.bbox,
                    "history": [],
                    "last_seen": self._frame_index,
                }

            matched_track_ids.add(best_track_id)
            track = self._tracks[best_track_id]
            track["bbox"] = face.bbox
            track["last_seen"] = self._frame_index

            if face.face_shape is not None:
                track["history"].append(face.face_shape)
                if len(track["history"]) > self.window_size:
                    track["history"].pop(0)

                # Temporal smoothing across window
                hist = track["history"]
                # Count valid shape predictions
                valid_preds = [res for res in hist if res.label != "undetermined"]

                if len(hist) < 3:
                    # Still gathering initial frames
                    smoothed_shape = FaceShapeResult(
                        label="analyzing",
                        fit_score=face.face_shape.fit_score,
                        alternatives=face.face_shape.alternatives,
                        measurements=face.face_shape.measurements,
                        quality_status="analyzing",
                        warnings=["Analyzing face geometry across initial frames..."],
                    )
                elif not valid_preds:
                    smoothed_shape = face.face_shape
                else:
                    # Score accumulator across history
                    shape_votes: Dict[str, float] = {}
                    for item in valid_preds:
                        shape_votes[item.label] = shape_votes.get(item.label, 0.0) + item.fit_score

                    top_shape = max(shape_votes.items(), key=lambda kv: kv[1])[0]
                    avg_fit = sum(item.fit_score for item in valid_preds if item.label == top_shape) / max(
                        1, sum(1 for item in valid_preds if item.label == top_shape)
                    )

                    smoothed_shape = FaceShapeResult(
                        label=top_shape,
                        fit_score=round(avg_fit, 2),
                        alternatives=face.face_shape.alternatives,
                        measurements=face.face_shape.measurements,
                        quality_status="good",
                        warnings=[],
                    )

                updated_faces.append(replace(face, face_shape=smoothed_shape))
            else:
                updated_faces.append(face)

        # Purge stale tracks not seen for > 15 frames
        stale_ids = [tid for tid, t in self._tracks.items() if self._frame_index - t["last_seen"] > 15]
        for tid in stale_ids:
            del self._tracks[tid]

        return updated_faces


def draw_face_shape_overlay(
    image: np.ndarray,
    face: FaceDetection,
    effect: Optional[str] = "scan",
    scan_phase: float = 0.0,
    color: Tuple[int, int, int] = (0, 255, 136),
    thickness: int = 2,
    show_contour: bool = True,
    show_measurements: bool = False,
) -> None:
    """Render face shape contour mesh, status badge, and biometric scanning HUD onto an image.

    Args:
        image: Target image array (modified in place).
        face: FaceDetection containing face_shape and optional dense_landmarks.
        effect: Effect style, e.g. "scan" or None.
        scan_phase: Continuous float in [0.0, 1.0] representing scanline cycle position.
        color: BGR line color.
        thickness: Stroke thickness.
        show_contour: Whether to draw facial silhouette contour lines.
        show_measurements: Whether to print key proportion ratios below the badge.
    """
    if face.face_shape is None:
        return

    bbox = face.bbox
    shape_result = face.face_shape

    # 1. Draw facial silhouette contour mesh if dense landmarks are present
    if show_contour and face.dense_landmarks:
        lms = face.dense_landmarks
        poly_pts = np.array(
            [[lms[i].x, lms[i].y] for i in FACEMESH_FACE_OVAL],
            dtype=np.int32,
        )
        if len(poly_pts) >= 3:
            # Draw smooth translucent contour
            overlay = image.copy()
            cv2.polylines(
                overlay, [poly_pts], isClosed=True, color=color, thickness=max(1, thickness - 1), lineType=cv2.LINE_AA
            )
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # 2. Futuristic Scanning Effect
    if effect == "scan" and bbox.height > 0:
        # scan_phase in [0.0, 1.0]: oscillates smoothly down and up using sine wave
        t = (math.sin(scan_phase * 2 * math.pi) + 1.0) / 2.0
        scan_y = int(bbox.y + t * bbox.height)

        # Draw glowing laser line across bounding box
        x1, x2 = max(0, bbox.x), min(image.shape[1], bbox.x + bbox.width)
        if 0 <= scan_y < image.shape[0]:
            cv2.line(image, (x1, scan_y), (x2, scan_y), (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            # Glowing halo lines above and below
            for dy, alpha in [(1, 0.5), (-1, 0.5), (2, 0.25), (-2, 0.25)]:
                gy = scan_y + dy
                if 0 <= gy < image.shape[0]:
                    overlay = image.copy()
                    cv2.line(overlay, (x1, gy), (x2, gy), color, thickness=1, lineType=cv2.LINE_AA)
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # 3. Shape Classification HUD Badge
    label_str = shape_result.label.upper()
    if label_str == "ANALYZING":
        badge_text = "[ SCANNING GEOMETRY... ]"
        badge_color = (0, 215, 255)  # Gold/Amber
    elif label_str == "UNDETERMINED":
        badge_text = "[ SHAPE: UNDETERMINED ]"
        badge_color = (0, 140, 255)  # Orange
    else:
        fit_pct = int(round(shape_result.fit_score * 100))
        badge_text = f"[ SHAPE: {label_str} | FIT: {fit_pct}% ]"
        badge_color = color

    # Draw badge background box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    text_thick = 1
    (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, text_thick)

    badge_x = max(5, bbox.x)
    badge_y = max(th + 10, bbox.y - 12)

    bg_x1 = badge_x - 4
    bg_y1 = badge_y - th - 4
    bg_x2 = badge_x + tw + 4
    bg_y2 = badge_y + baseline + 2

    # Draw semi-transparent background plate
    sub_img = image[max(0, bg_y1) : min(image.shape[0], bg_y2), max(0, bg_x1) : min(image.shape[1], bg_x2)]
    if sub_img.size > 0:
        dark_rect = np.zeros_like(sub_img)
        cv2.addWeighted(dark_rect, 0.75, sub_img, 0.25, 0, sub_img)

    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), badge_color, 1)
    cv2.putText(image, badge_text, (badge_x, badge_y), font, font_scale, badge_color, text_thick, cv2.LINE_AA)

    # Optional: draw measurement ratios
    if show_measurements and shape_result.measurements:
        m = shape_result.measurements
        ratio_str = f"L/W: {m.get('length_to_width_ratio', 0):.2f} | JAW: {m.get('jaw_angle', 0):.0f}deg"
        cv2.putText(
            image,
            ratio_str,
            (badge_x, bg_y2 + 14),
            font,
            0.40,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
