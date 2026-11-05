"""Automated test suite for real-time face gesture detection in detekface 0.3.0.

Tests:
1. Blink duration (noise rejection vs valid transient blink vs held closure)
2. Wink detection (left eye blink vs right eye blink isolation)
3. Mouth opening (onset detection, hold non-spam, closure reset)
4. Head movement (head turn left, head turn right, head tilt via HeadPose)
5. Debounce filter (suppression of rapid re-triggering within cooldown window)
6. Repeated events (successful re-triggering after debounce window has elapsed)
7. Multi-face independent tracking (independent gesture states across track IDs)
8. Missing and degenerate landmarks (graceful handling without crashes)
9. Dense landmark EAR / MAR calculation (geometric verification)
10. Configurable gesture selection and custom thresholds
11. Safe user callback exception isolation
12. Camera integration options
"""

from typing import List
from unittest.mock import MagicMock
import pytest

from detekface import (
    BoundingBox,
    FaceDetection,
    GestureConfig,
    GestureDetector,
    GestureEvent,
    HeadPose,
    Point,
    compute_eye_aspect_ratio,
    compute_mouth_aspect_ratio,
    normalize_gesture_name,
)
from detekface.gestures import (
    GESTURE_BOTH_EYES_CLOSED,
    GESTURE_HEAD_TILT,
    GESTURE_HEAD_TURN_LEFT,
    GESTURE_HEAD_TURN_RIGHT,
    GESTURE_LEFT_EYE_BLINK,
    GESTURE_MOUTH_OPENING,
    GESTURE_RIGHT_EYE_BLINK,
)


@pytest.fixture
def synthetic_open_face() -> FaceDetection:
    """A synthetic face with neutral open eyes, closed mouth, and centered head pose."""
    return FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.98,
        track_id=1,
        blendshapes={"eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05, "jawOpen": 0.05},
        head_pose=HeadPose(yaw=0.0, pitch=0.0, roll=0.0),
    )


def _make_dense_points(ear_left: float = 0.30, ear_right: float = 0.30, mar: float = 0.10) -> List[Point]:
    """Generate 478 synthetic Point landmarks calibrated to given EAR and MAR values."""
    pts = [Point(x=100 + (i % 50), y=100 + (i // 50)) for i in range(478)]

    # Left eye: corners 263, 362 (width = 40)
    # Upper: 386, 387; Lower: 374, 373
    pts[263] = Point(x=160, y=120)
    pts[362] = Point(x=120, y=120)
    v_left = int(round(ear_left * 40))
    pts[386] = Point(x=140, y=120 - v_left // 2)
    pts[387] = Point(x=140, y=120 - v_left // 2)
    pts[374] = Point(x=140, y=120 + v_left // 2)
    pts[373] = Point(x=140, y=120 + v_left // 2)

    # Right eye: corners 33, 133 (width = 40)
    # Upper: 159, 160; Lower: 145, 144
    pts[33] = Point(x=40, y=120)
    pts[133] = Point(x=80, y=120)
    v_right = int(round(ear_right * 40))
    pts[159] = Point(x=60, y=120 - v_right // 2)
    pts[160] = Point(x=60, y=120 - v_right // 2)
    pts[145] = Point(x=60, y=120 + v_right // 2)
    pts[144] = Point(x=60, y=120 + v_right // 2)

    # Inner lips: corners 61, 291 (width = 50)
    # Upper inner: 13, lower inner: 14
    pts[61] = Point(x=75, y=170)
    pts[291] = Point(x=125, y=170)
    v_mouth = int(round(mar * 50))
    pts[13] = Point(x=100, y=170 - v_mouth // 2)
    pts[14] = Point(x=100, y=170 + v_mouth // 2)

    return pts


# =============================================================================
# 1. Blink Duration Tests
# =============================================================================


def test_blink_duration_filtering() -> None:
    """Verify that short sensor glitches are rejected and valid blinks trigger once."""
    detector = GestureDetector(
        config=GestureConfig(min_blink_duration=0.06, max_blink_duration=0.50, debounce_seconds=0.20)
    )

    # Frame 1: Eyes open at t=0.00
    f1 = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        track_id=1,
        blendshapes={"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0},
    )
    events = detector.process([f1], (480, 640), timestamp=0.00)
    assert len(events) == 0

    # Frame 2: Eyes glitch closed for only 0.02s at t=0.02 (shorter than min_blink_duration)
    f2 = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        track_id=1,
        blendshapes={"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9},
    )
    events = detector.process([f2], (480, 640), timestamp=0.02)
    assert len(events) == 0

    # Frame 3: Eyes open at t=0.04 (duration was only 0.02s -> noise rejected)
    events = detector.process([f1], (480, 640), timestamp=0.04)
    assert len(events) == 0

    # Frame 4: Valid blink starts at t=0.10
    events = detector.process([f2], (480, 640), timestamp=0.10)
    assert len(events) == 0

    # Frame 5: Eyes reopen at t=0.25 (duration = 0.15s, valid blink!)
    events = detector.process([f1], (480, 640), timestamp=0.25)
    assert len(events) == 1
    assert events[0].name == GESTURE_BOTH_EYES_CLOSED
    assert events[0].track_id == 1
    assert pytest.approx(events[0].duration, abs=0.02) == 0.15


# =============================================================================
# 2. Wink Detection Tests
# =============================================================================


def test_wink_detection_left_and_right() -> None:
    """Verify left eye blink (left wink) and right eye blink (right wink) isolate properly."""
    detector = GestureDetector(
        config=GestureConfig(min_blink_duration=0.05, max_blink_duration=0.50, debounce_seconds=0.15)
    )

    open_face = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        track_id=1,
        blendshapes={"eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05},
    )
    left_wink_face = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        track_id=1,
        blendshapes={"eyeBlinkLeft": 0.85, "eyeBlinkRight": 0.05},
    )
    right_wink_face = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        track_id=1,
        blendshapes={"eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.85},
    )

    # 1. Left Wink
    detector.process([open_face], (480, 640), timestamp=0.00)
    detector.process([left_wink_face], (480, 640), timestamp=0.10)
    evs_left = detector.process([open_face], (480, 640), timestamp=0.25)

    assert len(evs_left) == 1
    assert evs_left[0].name == GESTURE_LEFT_EYE_BLINK
    assert evs_left[0].track_id == 1

    # 2. Right Wink after cooldown
    detector.process([open_face], (480, 640), timestamp=0.50)
    detector.process([right_wink_face], (480, 640), timestamp=0.60)
    evs_right = detector.process([open_face], (480, 640), timestamp=0.75)

    assert len(evs_right) == 1
    assert evs_right[0].name == GESTURE_RIGHT_EYE_BLINK
    assert evs_right[0].track_id == 1


# =============================================================================
# 3. Mouth Opening & Anti-Spam Hold Tests
# =============================================================================


def test_mouth_opening_onset_and_hold_prevention() -> None:
    """Verify mouth opening triggers once on sustained opening and does NOT spam while held."""
    events_received: List[GestureEvent] = []

    detector = GestureDetector(
        config=GestureConfig(min_mouth_duration=0.06, debounce_seconds=0.25),
        on_gesture=lambda ev: events_received.append(ev),
    )

    closed = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        track_id=1,
        blendshapes={"jawOpen": 0.05},
    )
    opened = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        track_id=1,
        blendshapes={"jawOpen": 0.75},
    )

    # t=0.00: Closed mouth
    detector.process([closed], (480, 640), timestamp=0.00)
    assert len(events_received) == 0

    # t=0.02: Mouth begins opening (pending)
    detector.process([opened], (480, 640), timestamp=0.02)
    assert len(events_received) == 0

    # t=0.09: Sustained for 0.07s >= min_mouth_duration (0.06s) -> TRIGGERS ONCE!
    detector.process([opened], (480, 640), timestamp=0.09)
    assert len(events_received) == 1
    assert events_received[0].name == GESTURE_MOUTH_OPENING
    assert events_received[0].state == "action"

    # t=0.12 - t=0.60: Mouth held open across 15 consecutive frames -> 0 extra events!
    for step in range(15):
        t = 0.12 + step * 0.03
        detector.process([opened], (480, 640), timestamp=t)
    assert len(events_received) == 1, "Expected zero duplicate callbacks while mouth is held open"

    # t=0.70: Mouth closes
    detector.process([closed], (480, 640), timestamp=0.70)
    assert len(events_received) == 1

    # t=1.00: Mouth opens again past debounce cooldown -> TRIGGERS SECOND EVENT!
    detector.process([opened], (480, 640), timestamp=1.00)
    detector.process([opened], (480, 640), timestamp=1.08)
    assert len(events_received) == 2
    assert events_received[1].name == GESTURE_MOUTH_OPENING


# =============================================================================
# 4. Head Movement Tests
# =============================================================================


def test_head_turn_and_tilt_gestures() -> None:
    """Verify head turn left, head turn right, and head tilt trigger from 3D HeadPose."""
    detector = GestureDetector(
        config=GestureConfig(
            head_turn_yaw_threshold=18.0,
            head_tilt_roll_threshold=15.0,
            min_head_gesture_duration=0.05,
            debounce_seconds=0.20,
        )
    )

    neutral = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        track_id=1,
        head_pose=HeadPose(yaw=0.0, pitch=0.0, roll=0.0),
    )
    turn_left = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        track_id=1,
        head_pose=HeadPose(yaw=-25.0, pitch=2.0, roll=1.0),
    )
    turn_right = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        track_id=1,
        head_pose=HeadPose(yaw=28.0, pitch=-1.0, roll=-2.0),
    )
    tilt = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        track_id=1,
        head_pose=HeadPose(yaw=2.0, pitch=0.0, roll=22.0),
    )

    # 1. Turn Left
    detector.process([neutral], (480, 640), timestamp=0.00)
    detector.process([turn_left], (480, 640), timestamp=0.02)
    evs_left = detector.process([turn_left], (480, 640), timestamp=0.09)
    assert len(evs_left) == 1
    assert evs_left[0].name == GESTURE_HEAD_TURN_LEFT

    # Hold turned left -> no spam
    evs_hold = detector.process([turn_left], (480, 640), timestamp=0.15)
    assert len(evs_hold) == 0

    # 2. Return to neutral, then Turn Right
    detector.process([neutral], (480, 640), timestamp=0.35)
    detector.process([turn_right], (480, 640), timestamp=0.45)
    evs_right = detector.process([turn_right], (480, 640), timestamp=0.52)
    assert len(evs_right) == 1
    assert evs_right[0].name == GESTURE_HEAD_TURN_RIGHT

    # 3. Return to neutral, then Tilt
    detector.process([neutral], (480, 640), timestamp=0.80)
    detector.process([tilt], (480, 640), timestamp=0.90)
    evs_tilt = detector.process([tilt], (480, 640), timestamp=0.97)
    assert len(evs_tilt) == 1
    assert evs_tilt[0].name == GESTURE_HEAD_TILT


# =============================================================================
# 5. Debounce & Repeated Events Tests
# =============================================================================


def test_debounce_suppresses_rapid_retrigger() -> None:
    """Verify actions re-occurring within the debounce cooldown are safely suppressed."""
    detector = GestureDetector(
        config=GestureConfig(min_mouth_duration=0.04, debounce_seconds=0.30)
    )

    closed = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        track_id=1,
        blendshapes={"jawOpen": 0.05},
    )
    opened = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        track_id=1,
        blendshapes={"jawOpen": 0.70},
    )

    # First opening at t=0.00 -> 0.05 (fires)
    detector.process([closed], (480, 640), timestamp=0.00)
    detector.process([opened], (480, 640), timestamp=0.01)
    evs1 = detector.process([opened], (480, 640), timestamp=0.06)
    assert len(evs1) == 1

    # Closes at t=0.10
    detector.process([closed], (480, 640), timestamp=0.10)

    # Rapid re-opening at t=0.15 -> 0.20 (only 0.14s after last trigger < 0.30s debounce)
    detector.process([opened], (480, 640), timestamp=0.15)
    evs2 = detector.process([opened], (480, 640), timestamp=0.20)
    assert len(evs2) == 0, "Expected rapid re-trigger within 0.30s debounce to be suppressed"

    # Closes at t=0.25
    detector.process([closed], (480, 640), timestamp=0.25)

    # Re-opening at t=0.45 (> 0.30s since 0.06) -> FIRES CLEANLY
    detector.process([opened], (480, 640), timestamp=0.45)
    evs3 = detector.process([opened], (480, 640), timestamp=0.51)
    assert len(evs3) == 1
    assert evs3[0].name == GESTURE_MOUTH_OPENING


# =============================================================================
# 6. Multiple Faces Tests
# =============================================================================


def test_multi_face_independent_gesture_tracking() -> None:
    """Verify multiple faces are tracked independently without cross-face gesture interference."""
    detector = GestureDetector(
        config=GestureConfig(min_mouth_duration=0.04, min_blink_duration=0.05, debounce_seconds=0.20)
    )

    # Face 1 is winking left; Face 2 has mouth open
    f1_wink = FaceDetection(
        bbox=BoundingBox(50, 100, 120, 120),
        confidence=0.95,
        track_id=1,
        blendshapes={"eyeBlinkLeft": 0.85, "eyeBlinkRight": 0.05, "jawOpen": 0.05},
    )
    f1_open = FaceDetection(
        bbox=BoundingBox(50, 100, 120, 120),
        confidence=0.95,
        track_id=1,
        blendshapes={"eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05, "jawOpen": 0.05},
    )
    f2_mouth = FaceDetection(
        bbox=BoundingBox(300, 100, 120, 120),
        confidence=0.95,
        track_id=2,
        blendshapes={"eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05, "jawOpen": 0.80},
    )

    # Initial frame
    detector.process([f1_open, f2_mouth], (480, 640), timestamp=0.00)
    # Face 1 closes left eye, Face 2 keeps mouth open
    detector.process([f1_wink, f2_mouth], (480, 640), timestamp=0.02)

    # At t=0.06: Face 2 mouth opening triggers!
    evs = detector.process([f1_wink, f2_mouth], (480, 640), timestamp=0.06)
    assert len(evs) == 1
    assert evs[0].name == GESTURE_MOUTH_OPENING
    assert evs[0].track_id == 2

    # At t=0.15: Face 1 reopens left eye -> Left wink triggers!
    evs_f1 = detector.process([f1_open, f2_mouth], (480, 640), timestamp=0.15)
    assert len(evs_f1) == 1
    assert evs_f1[0].name == GESTURE_LEFT_EYE_BLINK
    assert evs_f1[0].track_id == 1


# =============================================================================
# 7. Dense Landmarks Geometric EAR / MAR Tests
# =============================================================================


def test_dense_landmarks_ear_and_mar() -> None:
    """Verify EAR and MAR algorithms with calibrated geometric landmarks."""
    # 1. Open eyes (EAR ~ 0.30) & Closed mouth (MAR ~ 0.10)
    open_pts = _make_dense_points(ear_left=0.30, ear_right=0.30, mar=0.10)
    assert compute_eye_aspect_ratio(open_pts, "left") >= 0.25
    assert compute_eye_aspect_ratio(open_pts, "right") >= 0.25
    assert compute_mouth_aspect_ratio(open_pts) <= 0.15

    # 2. Closed eyes (EAR ~ 0.10) & Open mouth (MAR ~ 0.50)
    closed_pts = _make_dense_points(ear_left=0.10, ear_right=0.10, mar=0.50)
    assert compute_eye_aspect_ratio(closed_pts, "left") < 0.20
    assert compute_eye_aspect_ratio(closed_pts, "right") < 0.20
    assert compute_mouth_aspect_ratio(closed_pts) > 0.35

    # 3. Test through GestureDetector
    detector = GestureDetector(config=GestureConfig(min_mouth_duration=0.05))
    f_open = FaceDetection(bbox=BoundingBox(100, 100, 150, 150), confidence=0.95, dense_landmarks=open_pts)
    f_mouth = FaceDetection(bbox=BoundingBox(100, 100, 150, 150), confidence=0.95, dense_landmarks=closed_pts)

    detector.process([f_open], (480, 640), timestamp=0.00)
    detector.process([f_mouth], (480, 640), timestamp=0.02)
    evs = detector.process([f_mouth], (480, 640), timestamp=0.08)

    assert len(evs) == 1
    assert evs[0].name == GESTURE_MOUTH_OPENING


# =============================================================================
# 8. Missing & Degenerate Landmarks Tests
# =============================================================================


def test_missing_and_degenerate_landmarks_handling() -> None:
    """Verify faces without landmarks or blendshapes are handled safely without crashing."""
    detector = GestureDetector()

    # Empty face detection (no landmarks, no blendshapes, no pose)
    bare_face = FaceDetection(bbox=BoundingBox(100, 100, 150, 150), confidence=0.95)
    evs = detector.process([bare_face], (480, 640), timestamp=0.00)
    assert evs == []

    # Insufficient dense landmarks (< 292 points)
    short_lms = [Point(100, 100)] * 50
    short_face = FaceDetection(bbox=BoundingBox(100, 100, 150, 150), confidence=0.95, dense_landmarks=short_lms)
    evs2 = detector.process([short_face], (480, 640), timestamp=0.05)
    assert evs2 == []


# =============================================================================
# 9. Gesture Selection & Aliasing Tests
# =============================================================================


def test_selective_gesture_configuration_and_aliases() -> None:
    """Verify only explicitly requested gestures trigger and aliases map correctly."""
    assert normalize_gesture_name("blink_left") == GESTURE_LEFT_EYE_BLINK
    assert normalize_gesture_name("mouth_open") == GESTURE_MOUTH_OPENING
    assert normalize_gesture_name("tilt") == GESTURE_HEAD_TILT

    # Enable ONLY mouth opening
    detector = GestureDetector(enabled_gestures=["mouth_opening"])

    # Face with both mouth open and head turned
    face = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        track_id=1,
        blendshapes={"jawOpen": 0.80},
        head_pose=HeadPose(yaw=30.0, pitch=0.0, roll=0.0),
    )

    detector.process([face], (480, 640), timestamp=0.00)
    evs = detector.process([face], (480, 640), timestamp=0.10)

    # Should only contain mouth_opening; head_turn_right was excluded
    assert len(evs) == 1
    assert evs[0].name == GESTURE_MOUTH_OPENING


# =============================================================================
# 10. Callback Exception Safety Test
# =============================================================================


def test_user_callback_exception_does_not_crash_pipeline() -> None:
    """Verify a faulty user callback raising an error does not crash the processing pipeline."""
    faulty_callback = MagicMock(side_effect=RuntimeError("User callback error!"))

    detector = GestureDetector(
        config=GestureConfig(min_mouth_duration=0.02),
        on_gesture=faulty_callback,
    )

    f_open = FaceDetection(
        bbox=BoundingBox(100, 100, 150, 150),
        confidence=0.95,
        blendshapes={"jawOpen": 0.80},
    )

    detector.process([f_open], (480, 640), timestamp=0.00)
    # This must NOT raise RuntimeError
    evs = detector.process([f_open], (480, 640), timestamp=0.05)

    assert len(evs) == 1
    assert faulty_callback.called

