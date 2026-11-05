"""Temporary multi-face tracking engine for detekface 0.3.0.

Provides frame-to-frame temporal face association assigning stable, temporary
track IDs (e.g., Face ID 1, Face ID 2) to detected faces across live camera,
video streams, and image sequences.

Matching & Continuity Strategy:
1. Multi-cue Association: Combines bounding box IoU (Intersection over Union),
   centroid motion velocity, and landmark proximity.
2. Motion Prediction: Carries forward spatial velocity to anticipate position
   during rapid head movement.
3. Occlusion & Missed Detection Recovery: Maintains an active grace period
   (default 15-30 frames) for temporarily occluded or missed detections,
   reconnecting the original track ID when the face re-emerges.
4. Clean Lifecycle: Automatically expires stale tracks and provides explicit
   reset functionality between sessions.

IMPORTANT COMPLIANCE & PRIVACY NOTICE:
Temporary tracking IDs represent transient spatial-temporal motion tracklets only.
They do NOT perform persistent identity verification, facial recognition, or
cross-session biometric profiling.
"""

import math
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from detekface.types import BoundingBox, FaceDetection


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute Intersection-over-Union (IoU) overlap between two bounding boxes.

    Args:
        box1: First bounding box.
        box2: Second bounding box.

    Returns:
        float: IoU score ranging from 0.0 (no overlap) to 1.0 (identical).
    """
    xA = max(box1.x1, box2.x1)
    yA = max(box1.y1, box2.y1)
    xB = min(box1.x2, box2.x2)
    yB = min(box1.y2, box2.y2)

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    area1 = max(1, box1.width * box1.height)
    area2 = max(1, box2.width * box2.height)
    union_area = float(area1 + area2 - inter_area)

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_landmark_similarity(det1: FaceDetection, det2: FaceDetection) -> float:
    """Compute normalized geometric distance between primary facial landmarks.

    Returns:
        float: Similarity score between 0.0 (far) and 1.0 (coincident).
    """
    if det1.landmarks is None or det2.landmarks is None:
        return 0.5  # Neutral fallback when landmarks unavailable

    pts1 = det1.landmarks.as_list()
    pts2 = det2.landmarks.as_list()

    if not pts1 or not pts2 or len(pts1) != len(pts2):
        return 0.5

    diag1 = math.hypot(det1.bbox.width, det1.bbox.height)
    diag2 = math.hypot(det2.bbox.width, det2.bbox.height)
    avg_diag = max(1.0, (diag1 + diag2) / 2.0)

    total_dist = 0.0
    for p1, p2 in zip(pts1, pts2):
        total_dist += math.hypot(p1.x - p2.x, p1.y - p2.y)

    mean_dist = total_dist / len(pts1)
    norm_dist = mean_dist / avg_diag
    # Convert normalized distance to similarity (1.0 at 0 dist, approaching 0 at 0.5 diag dist)
    return max(0.0, 1.0 - norm_dist * 2.0)


class TrackedFace:
    """Stateful temporal representation of an actively tracked human face."""

    def __init__(self, track_id: int, detection: FaceDetection, timestamp: float) -> None:
        self.track_id: int = track_id
        self.bbox: BoundingBox = detection.bbox
        self.confidence: float = detection.confidence
        self.landmarks = detection.landmarks
        self.dense_landmarks = detection.dense_landmarks
        self.face_shape = detection.face_shape
        self.head_pose = detection.head_pose

        self.first_seen: float = timestamp
        self.last_seen: float = timestamp
        self.disappeared_frames: int = 0
        self.hits: int = 1

        # Centroid velocity: (dx, dy) per frame
        self.vx: float = 0.0
        self.vy: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        """Current bounding box center coordinates."""
        return (self.bbox.x + self.bbox.width / 2.0, self.bbox.y + self.bbox.height / 2.0)

    def predict_center(self) -> Tuple[float, float]:
        """Anticipate position in next frame using current motion velocity."""
        cx, cy = self.center
        return (cx + self.vx, cy + self.vy)

    def update(self, detection: FaceDetection, timestamp: float, alpha: float = 0.75) -> None:
        """Update tracker with fresh detection observations using exponential smoothing."""
        old_cx, old_cy = self.center
        new_cx = detection.bbox.x + detection.bbox.width / 2.0
        new_cy = detection.bbox.y + detection.bbox.height / 2.0

        # Update motion velocity
        self.vx = 0.6 * (new_cx - old_cx) + 0.4 * self.vx
        self.vy = 0.6 * (new_cy - old_cy) + 0.4 * self.vy

        # Smooth bounding box geometry to eliminate camera quantization jitter
        smooth_x = int(round(alpha * detection.bbox.x + (1.0 - alpha) * self.bbox.x))
        smooth_y = int(round(alpha * detection.bbox.y + (1.0 - alpha) * self.bbox.y))
        smooth_w = int(round(alpha * detection.bbox.width + (1.0 - alpha) * self.bbox.width))
        smooth_h = int(round(alpha * detection.bbox.height + (1.0 - alpha) * self.bbox.height))

        self.bbox = BoundingBox(x=smooth_x, y=smooth_y, width=smooth_w, height=smooth_h)
        self.confidence = detection.confidence
        self.landmarks = detection.landmarks
        self.dense_landmarks = detection.dense_landmarks
        self.face_shape = detection.face_shape
        self.head_pose = detection.head_pose

        self.last_seen = timestamp
        self.disappeared_frames = 0
        self.hits += 1


class FaceTracker:
    """Multi-face temporary tracking engine for real-time video and camera streams."""

    def __init__(
        self,
        max_disappeared_frames: int = 20,
        min_iou_threshold: float = 0.2,
        max_center_distance_ratio: float = 0.9,
    ) -> None:
        """Initialize the face tracking engine.

        Args:
            max_disappeared_frames: Maximum frames a track may be missing before expiring.
            min_iou_threshold: Minimum IoU overlap required for spatial matching.
            max_center_distance_ratio: Maximum centroid displacement relative to face size.
        """
        self.max_disappeared_frames = max(1, int(max_disappeared_frames))
        self.min_iou_threshold = max(0.0, float(min_iou_threshold))
        self.max_center_distance_ratio = max(0.1, float(max_center_distance_ratio))

        self._tracks: Dict[int, TrackedFace] = {}
        self._next_id: int = 1

    @property
    def active_tracks_count(self) -> int:
        """Number of active tracked faces."""
        return len(self._tracks)

    def reset(self) -> None:
        """Reset all active tracks and reset track identifier sequence."""
        self._tracks.clear()
        self._next_id = 1

    def update(
        self,
        detections: Sequence[FaceDetection],
        timestamp: Optional[float] = None,
    ) -> List[FaceDetection]:
        """Match incoming face detections to existing temporal tracks.

        Assigns a stable `track_id` to each detection, registers newly entering faces,
        and manages grace periods for temporarily occluded or missed detections.

        Args:
            detections: Sequence of FaceDetection instances from current frame.
            timestamp: Optional clock timestamp in seconds (defaults to time.perf_counter()).

        Returns:
            List[FaceDetection]: New FaceDetection objects populated with assigned `track_id`.
        """
        now = time.perf_counter() if timestamp is None else float(timestamp)

        # Case 1: No detections in current frame
        if not detections:
            expired_ids = []
            for tid, track in self._tracks.items():
                track.disappeared_frames += 1
                if track.disappeared_frames > self.max_disappeared_frames:
                    expired_ids.append(tid)
            for tid in expired_ids:
                del self._tracks[tid]
            return []

        # Case 2: No existing tracks -> initialize all incoming detections
        if not self._tracks:
            output = []
            for det in detections:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = TrackedFace(track_id=tid, detection=det, timestamp=now)
                # Attach assigned track_id to detection copy
                output.append(self._attach_track_id(det, tid))
            return output

        # Case 3: Match existing tracks against incoming detections
        track_ids = list(self._tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(detections)

        # Compute affinity matrix: higher score = better match (0.0 to 1.0)
        affinity_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            track = self._tracks[tid]
            pred_cx, pred_cy = track.predict_center()
            t_diag = math.hypot(track.bbox.width, track.bbox.height)

            for j, det in enumerate(detections):
                d_cx = det.bbox.x + det.bbox.width / 2.0
                d_cy = det.bbox.y + det.bbox.height / 2.0
                dist = math.hypot(pred_cx - d_cx, pred_cy - d_cy)

                # Normalized centroid distance score (1.0 = identical center, 0.0 = beyond threshold)
                dist_ratio = dist / max(1.0, t_diag)
                if dist_ratio > self.max_center_distance_ratio:
                    center_score = 0.0
                else:
                    center_score = 1.0 - (dist_ratio / self.max_center_distance_ratio)

                # Bounding box IoU score
                iou_score = compute_iou(track.bbox, det.bbox)

                # Landmark similarity score
                landmark_score = compute_landmark_similarity(
                    FaceDetection(bbox=track.bbox, confidence=track.confidence, landmarks=track.landmarks),
                    det,
                )

                # Weighted affinity score
                affinity = 0.45 * iou_score + 0.35 * center_score + 0.20 * landmark_score

                # Gating: must satisfy minimum IoU or close centroid proximity
                if iou_score >= self.min_iou_threshold or dist_ratio <= 0.45:
                    affinity_matrix[i, j] = affinity
                else:
                    affinity_matrix[i, j] = 0.0

        # Greedy assignment in descending order of affinity
        matched_tracks: Set[int] = set()
        matched_detections: Set[int] = set()
        matches: List[Tuple[int, int]] = []

        # Find matching pairs with affinity > 0
        flat_indices = np.argsort(-affinity_matrix.ravel())
        for idx in flat_indices:
            score = affinity_matrix.ravel()[idx]
            if score <= 0.05:
                break
            i = idx // num_dets
            j = idx % num_dets
            if i in matched_tracks or j in matched_detections:
                continue
            matched_tracks.add(i)
            matched_detections.add(j)
            matches.append((track_ids[i], j))

        output_map: Dict[int, FaceDetection] = {}

        # 1. Update matched tracks
        for tid, det_idx in matches:
            det = detections[det_idx]
            self._tracks[tid].update(det, now)
            output_map[det_idx] = self._attach_track_id(det, tid)

        # 2. Increment disappeared count for unmatched existing tracks
        expired_ids = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self._tracks[tid].disappeared_frames += 1
                if self._tracks[tid].disappeared_frames > self.max_disappeared_frames:
                    expired_ids.append(tid)

        for tid in expired_ids:
            del self._tracks[tid]

        # 3. Create new tracks for unmatched incoming detections (newly entering faces)
        for j in range(num_dets):
            if j not in matched_detections:
                det = detections[j]
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = TrackedFace(track_id=tid, detection=det, timestamp=now)
                output_map[j] = self._attach_track_id(det, tid)

        # Return results in the exact same order as input detections
        return [output_map[i] for i in range(num_dets)]

    @staticmethod
    def _attach_track_id(detection: FaceDetection, track_id: int) -> FaceDetection:
        """Create a new FaceDetection copy with track_id attached."""
        return FaceDetection(
            bbox=detection.bbox,
            confidence=detection.confidence,
            landmarks=detection.landmarks,
            face_shape=detection.face_shape,
            dense_landmarks=detection.dense_landmarks,
            track_id=int(track_id),
            head_pose=detection.head_pose,
        )
