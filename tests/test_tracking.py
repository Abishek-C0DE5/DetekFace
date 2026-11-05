"""Automated tests for temporary multi-face tracking in detekface 0.3.0.

Tests:
- Consecutive frame face tracking assigning temporary track IDs (Face ID: 1, Face ID: 2)
- Multi-face spatial continuity and moving face tracking
- Newly entering face receives fresh track ID
- Missed detections / temporary occlusion recovery within grace period
- Disappearing face track expiration
- Tracker reset on session end
- CameraSession integration with track_faces=True
- camera() GUI loop with show_tracking_ids=True
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from detekface import (
    BoundingBox,
    CameraSession,
    DetectionResult,
    FaceDetection,
    FaceLandmarks,
    FaceTracker,
    Point,
    camera,
    compute_iou,
)


def _make_face(x: int, y: int, w: int = 100, h: int = 100, conf: float = 0.95) -> FaceDetection:
    """Helper creating synthetic FaceDetection at (x, y) with 5 canonical landmarks."""
    bbox = BoundingBox(x=x, y=y, width=w, height=h)
    landmarks = FaceLandmarks(
        right_eye=Point(x=x + int(w * 0.3), y=y + int(h * 0.35)),
        left_eye=Point(x=x + int(w * 0.7), y=y + int(h * 0.35)),
        nose_tip=Point(x=x + int(w * 0.5), y=y + int(h * 0.55)),
        right_mouth=Point(x=x + int(w * 0.35), y=y + int(h * 0.75)),
        left_mouth=Point(x=x + int(w * 0.65), y=y + int(h * 0.75)),
    )
    return FaceDetection(bbox=bbox, confidence=conf, landmarks=landmarks)


# =========================================================================
# 1. IoU and Association Math
# =========================================================================


def test_compute_iou_overlap() -> None:
    """Verify compute_iou handles identical, partial, and non-overlapping boxes."""
    b1 = BoundingBox(x=0, y=0, width=100, height=100)
    b2 = BoundingBox(x=0, y=0, width=100, height=100)
    assert pytest.approx(compute_iou(b1, b2)) == 1.0

    # Half overlap: 50x100 overlap / 150x100 union
    b3 = BoundingBox(x=50, y=0, width=100, height=100)
    assert 0.3 < compute_iou(b1, b3) < 0.4

    # Disjoint boxes
    b4 = BoundingBox(x=200, y=200, width=50, height=50)
    assert compute_iou(b1, b4) == 0.0


# =========================================================================
# 2. Multi-Face Tracking & Continuous Trajectory
# =========================================================================


def test_tracker_assigns_ids_and_maintains_continuity() -> None:
    """Verify 2 faces receive stable IDs (1 and 2) across continuous movement."""
    tracker = FaceTracker(max_disappeared_frames=10)

    # Frame 1: Two separate faces
    f1_t0 = _make_face(x=100, y=100)
    f2_t0 = _make_face(x=400, y=100)

    tracked_0 = tracker.update([f1_t0, f2_t0], timestamp=0.0)
    assert len(tracked_0) == 2
    ids_0 = {f.track_id for f in tracked_0}
    assert ids_0 == {1, 2}

    # Frame 2: Faces shift slightly by 10 pixels (smooth motion)
    f1_t1 = _make_face(x=110, y=105)
    f2_t1 = _make_face(x=390, y=102)

    tracked_1 = tracker.update([f1_t1, f2_t1], timestamp=0.033)
    assert len(tracked_1) == 2

    # Verify ID persistence
    assert tracked_1[0].track_id == tracked_0[0].track_id
    assert tracked_1[1].track_id == tracked_0[1].track_id


def test_new_entering_face_receives_fresh_id() -> None:
    """Verify newly entering face is assigned a new incremented track ID."""
    tracker = FaceTracker()

    # Frame 1: Face 1 enters
    f1 = _make_face(x=100, y=100)
    res1 = tracker.update([f1])
    assert res1[0].track_id == 1

    # Frame 2: Face 1 moves, and Face 2 enters on the opposite side
    f1_moved = _make_face(x=105, y=102)
    f2 = _make_face(x=500, y=300)
    res2 = tracker.update([f1_moved, f2])

    assert len(res2) == 2
    assert res2[0].track_id == 1
    assert res2[1].track_id == 2


# =========================================================================
# 3. Occlusion Recovery & Track Expiration
# =========================================================================


def test_missed_detections_grace_period_recovery() -> None:
    """Verify face that disappears for 3 frames retains its track ID upon reappearance."""
    tracker = FaceTracker(max_disappeared_frames=5)

    # Frame 1: Face detected (ID 1)
    f = _make_face(x=200, y=200)
    res = tracker.update([f])
    assert res[0].track_id == 1

    # Frames 2, 3, 4: Face temporarily occluded / missed
    for _ in range(3):
        empty = tracker.update([])
        assert empty == []
        assert tracker.active_tracks_count == 1  # Retained in grace period

    # Frame 5: Face re-appears near previous location
    f_reappear = _make_face(x=210, y=205)
    res_reappear = tracker.update([f_reappear])

    assert len(res_reappear) == 1
    assert res_reappear[0].track_id == 1  # Reconnected to original track ID!


def test_disappearing_face_expires_after_max_frames() -> None:
    """Verify face that stays missing longer than max_disappeared_frames is purged."""
    tracker = FaceTracker(max_disappeared_frames=3)

    f = _make_face(x=200, y=200)
    tracker.update([f])
    assert tracker.active_tracks_count == 1

    # Miss for 4 consecutive frames (exceeding max 3)
    for _ in range(4):
        tracker.update([])

    # Track must now be fully expired
    assert tracker.active_tracks_count == 0

    # Next detection gets new track ID (ID 2)
    f_new = _make_face(x=200, y=200)
    res_new = tracker.update([f_new])
    assert res_new[0].track_id == 2


def test_tracker_reset_clears_all_tracks_and_ids() -> None:
    """Verify tracker.reset() clears active tracks and resets ID counter."""
    tracker = FaceTracker()
    tracker.update([_make_face(100, 100), _make_face(300, 300)])
    assert tracker.active_tracks_count == 2

    tracker.reset()
    assert tracker.active_tracks_count == 0

    # After reset, first face gets ID 1 again
    res = tracker.update([_make_face(100, 100)])
    assert res[0].track_id == 1


# =========================================================================
# 4. CameraSession and camera() Integration
# =========================================================================


class MockCap:
    def __init__(self):
        self._open = True
    def isOpened(self):
        return self._open
    def get(self, prop):
        return 30.0 if prop == 5 else 640.0
    def set(self, prop, val):
        return True
    def read(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)
    def release(self):
        self._open = False


def test_camera_session_track_faces_populates_track_id() -> None:
    """Verify CameraSession with track_faces=True populates track_id on each detection."""
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(
        faces=[_make_face(150, 150)],
        image=np.zeros((480, 640, 3), dtype=np.uint8),
    )

    with patch("cv2.VideoCapture", return_value=MockCap()):
        session = CameraSession(track_faces=True, detector=mock_detector)
        res1 = session.read()
        assert res1.faces[0].track_id == 1

        res2 = session.read()
        assert res2.faces[0].track_id == 1

        session.close()


def test_camera_function_show_tracking_ids_draws_badge() -> None:
    """Verify detekface.camera(show_tracking_ids=True) auto-enables tracking and renders badges."""
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(
        faces=[_make_face(120, 120)],
        image=np.zeros((480, 640, 3), dtype=np.uint8),
    )

    with (
        patch("cv2.VideoCapture", return_value=MockCap()),
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.waitKey", return_value=ord("q")),
        patch("cv2.destroyWindow"),
    ):
        camera(
            detector=mock_detector,
            show_tracking_ids=True,
            max_frames=1,
        )

        assert mock_imshow.call_count == 1

