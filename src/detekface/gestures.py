"""Real-time face gesture detection engine for detekface 0.3.0.

Provides an event-based system for detecting observable facial actions:
- Left eye blink / wink
- Right eye blink / wink
- Both eyes closed
- Mouth opening
- Head tilt
- Head turning left
- Head turning right

=============================================================================
ETHICAL & SCIENTIFIC DISCLAIMER:
Gestures detected by this module are mathematical approximations of observable
surface geometric motion and facial landmarks. They MUST NOT be interpreted as
inferring human emotions, psychological states, intent, truthfulness, cognitive
load, personality traits, or medical/neurological conditions.
Gesture detection is strictly intended for human-computer interaction (HCI),
hands-free user interfaces, and accessibility controls.
=============================================================================
"""

from dataclasses import dataclass
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from detekface.pose import estimate_head_pose
from detekface.types import FaceDetection, GestureEvent, Point

# Canonical Gesture Names
GESTURE_LEFT_EYE_BLINK = "left_eye_blink"
GESTURE_RIGHT_EYE_BLINK = "right_eye_blink"
GESTURE_BOTH_EYES_CLOSED = "both_eyes_closed"
GESTURE_MOUTH_OPENING = "mouth_opening"
GESTURE_HEAD_TILT = "head_tilt"
GESTURE_HEAD_TURN_LEFT = "head_turn_left"
GESTURE_HEAD_TURN_RIGHT = "head_turn_right"

ALL_GESTURES: Tuple[str, ...] = (
    GESTURE_LEFT_EYE_BLINK,
    GESTURE_RIGHT_EYE_BLINK,
    GESTURE_BOTH_EYES_CLOSED,
    GESTURE_MOUTH_OPENING,
    GESTURE_HEAD_TILT,
    GESTURE_HEAD_TURN_LEFT,
    GESTURE_HEAD_TURN_RIGHT,
)

# Friendly aliases mapping to canonical names
GESTURE_ALIASES: Dict[str, str] = {
    "left_eye_blink": GESTURE_LEFT_EYE_BLINK,
    "blink_left": GESTURE_LEFT_EYE_BLINK,
    "wink_left": GESTURE_LEFT_EYE_BLINK,
    "left_blink": GESTURE_LEFT_EYE_BLINK,
    "left_wink": GESTURE_LEFT_EYE_BLINK,
    "right_eye_blink": GESTURE_RIGHT_EYE_BLINK,
    "blink_right": GESTURE_RIGHT_EYE_BLINK,
    "wink_right": GESTURE_RIGHT_EYE_BLINK,
    "right_blink": GESTURE_RIGHT_EYE_BLINK,
    "right_wink": GESTURE_RIGHT_EYE_BLINK,
    "both_eyes_closed": GESTURE_BOTH_EYES_CLOSED,
    "eyes_closed": GESTURE_BOTH_EYES_CLOSED,
    "close_eyes": GESTURE_BOTH_EYES_CLOSED,
    "mouth_opening": GESTURE_MOUTH_OPENING,
    "mouth_open": GESTURE_MOUTH_OPENING,
    "jaw_open": GESTURE_MOUTH_OPENING,
    "head_tilt": GESTURE_HEAD_TILT,
    "tilt": GESTURE_HEAD_TILT,
    "head_tilt_left": GESTURE_HEAD_TILT,
    "head_tilt_right": GESTURE_HEAD_TILT,
    "head_turn_left": GESTURE_HEAD_TURN_LEFT,
    "turn_left": GESTURE_HEAD_TURN_LEFT,
    "look_left": GESTURE_HEAD_TURN_LEFT,
    "head_turn_right": GESTURE_HEAD_TURN_RIGHT,
    "turn_right": GESTURE_HEAD_TURN_RIGHT,
    "look_right": GESTURE_HEAD_TURN_RIGHT,
}


def normalize_gesture_name(name: str) -> Optional[str]:
    """Normalize a gesture name or alias to its canonical string."""
    clean = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return GESTURE_ALIASES.get(clean)


# Canonical 468/478 Landmark Indices for Eye Aspect Ratio (EAR) & Mouth Aspect Ratio (MAR)
# MediaPipe FaceMesh topological indices:
# Left eye (subject's left, viewer's right):
#   Corners: 263 (outer), 362 (inner)
#   Upper eyelid: 386, 387
#   Lower eyelid: 374, 373
LEFT_EYE_CORNERS = (263, 362)
LEFT_EYE_UPPER = (386, 387)
LEFT_EYE_LOWER = (374, 373)

# Right eye (subject's right, viewer's left):
#   Corners: 33 (outer), 133 (inner)
#   Upper eyelid: 159, 160
#   Lower eyelid: 145, 144
RIGHT_EYE_CORNERS = (33, 133)
RIGHT_EYE_UPPER = (159, 160)
RIGHT_EYE_LOWER = (145, 144)

# Inner lips for MAR:
#   Upper inner lip: 13
#   Lower inner lip: 14
#   Corners: 61 (right), 291 (left)
MOUTH_CORNERS = (61, 291)
MOUTH_INNER_LIPS = (13, 14)


def compute_eye_aspect_ratio(
    dense_landmarks: Sequence[Point],
    eye: str = "left",
) -> float:
    """Compute Eye Aspect Ratio (EAR) using canonical 6-point eyelid landmarks.

    EAR = (||p_upper1 - p_lower1|| + ||p_upper2 - p_lower2||) / (2 * ||p_corner1 - p_corner2||)

    Args:
        dense_landmarks: Sequence of at least 388 dense landmark Points.
        eye: 'left' (subject's left eye) or 'right' (subject's right eye).

    Returns:
        float: Calculated EAR ratio (typically ~0.25-0.35 when open, <0.20 when closed).
    """
    if len(dense_landmarks) < 388:
        return 0.30

    if eye == "left":
        c1, c2 = LEFT_EYE_CORNERS
        u1, u2 = LEFT_EYE_UPPER
        l1, l2 = LEFT_EYE_LOWER
    else:
        c1, c2 = RIGHT_EYE_CORNERS
        u1, u2 = RIGHT_EYE_UPPER
        l1, l2 = RIGHT_EYE_LOWER

    try:
        p_c1, p_c2 = dense_landmarks[c1], dense_landmarks[c2]
        p_u1, p_u2 = dense_landmarks[u1], dense_landmarks[u2]
        p_l1, p_l2 = dense_landmarks[l1], dense_landmarks[l2]

        h_dist = math.hypot(p_c1.x - p_c2.x, p_c1.y - p_c2.y)
        if h_dist <= 1e-4:
            return 0.30

        v1 = math.hypot(p_u1.x - p_l1.x, p_u1.y - p_l1.y)
        v2 = math.hypot(p_u2.x - p_l2.x, p_u2.y - p_l2.y)
        return (v1 + v2) / (2.0 * h_dist)
    except (IndexError, AttributeError):
        return 0.30


def compute_mouth_aspect_ratio(dense_landmarks: Sequence[Point]) -> float:
    """Compute Mouth Aspect Ratio (MAR) using canonical inner lip landmarks.

    MAR = ||p_upper_lip - p_lower_lip|| / ||p_corner_left - p_corner_right||

    Args:
        dense_landmarks: Sequence of at least 292 dense landmark Points.

    Returns:
        float: Calculated MAR ratio (typically <0.15 when closed, >0.35 when open).
    """
    if len(dense_landmarks) < 292:
        return 0.0

    try:
        c1, c2 = MOUTH_CORNERS
        idx_upper, idx_lower = MOUTH_INNER_LIPS
        p_c1, p_c2 = dense_landmarks[c1], dense_landmarks[c2]
        p_u, p_l = dense_landmarks[idx_upper], dense_landmarks[idx_lower]

        h_dist = math.hypot(p_c1.x - p_c2.x, p_c1.y - p_c2.y)
        if h_dist <= 1e-4:
            return 0.0

        v_dist = math.hypot(p_u.x - p_l.x, p_u.y - p_l.y)
        return v_dist / h_dist
    except (IndexError, AttributeError):
        return 0.0


@dataclass
class GestureConfig:
    """Configurable thresholds, durations, and debounce parameters for gesture detection.

    Attributes:
        enabled_gestures: Optional set of gesture names to detect. If None, all are enabled.
        blink_ear_threshold: EAR threshold below which an eye is considered closed.
        blink_blendshape_threshold: Blendshape confidence threshold for eyeBlinkLeft / eyeBlinkRight.
        min_blink_duration: Minimum seconds an eye must be closed to register as a blink (avoids frame noise).
        max_blink_duration: Maximum seconds for a blink before it is treated as held closure.
        both_eyes_closed_min_duration: Minimum seconds both eyes must be closed to fire both_eyes_closed.
        mouth_open_mar_threshold: MAR threshold above which mouth is considered open.
        mouth_open_blendshape_threshold: Blendshape confidence threshold for jawOpen.
        min_mouth_duration: Minimum seconds mouth must remain open before firing event.
        head_turn_yaw_threshold: Head yaw angle in degrees required to fire head_turn_left / right.
        head_tilt_roll_threshold: Head roll angle in degrees required to fire head_tilt.
        min_head_gesture_duration: Minimum seconds head pose must be sustained to fire gesture.
        debounce_seconds: Cooldown period in seconds before the same gesture can fire again on the same face.
    """

    enabled_gestures: Optional[Set[str]] = None
    blink_ear_threshold: float = 0.20
    blink_blendshape_threshold: float = 0.45
    min_blink_duration: float = 0.04
    max_blink_duration: float = 0.60
    both_eyes_closed_min_duration: float = 0.20
    mouth_open_mar_threshold: float = 0.35
    mouth_open_blendshape_threshold: float = 0.35
    min_mouth_duration: float = 0.08
    head_turn_yaw_threshold: float = 18.0
    head_tilt_roll_threshold: float = 15.0
    min_head_gesture_duration: float = 0.08
    debounce_seconds: float = 0.25

    def is_enabled(self, gesture_name: str) -> bool:
        """Check whether a given canonical gesture name is enabled."""
        if self.enabled_gestures is None:
            return True
        canonical = normalize_gesture_name(gesture_name) or gesture_name
        return canonical in self.enabled_gestures or gesture_name in self.enabled_gestures


class _ContinuousState:
    """Internal state tracker for a continuous gesture (mouth open, head turn, head tilt)."""

    def __init__(self) -> None:
        self.is_active: bool = False
        self.pending_start: Optional[float] = None
        self.last_trigger_time: float = -999.0

    def update(
        self,
        condition: bool,
        current_time: float,
        min_duration: float,
        debounce: float,
    ) -> bool:
        """Update state. Returns True exactly once when the gesture triggers."""
        if condition:
            if self.is_active:
                # Already active and held: DO NOT trigger again while held
                return False

            if self.pending_start is None:
                self.pending_start = current_time
                return False

            # Check if condition has been sustained for minimum duration
            if current_time - self.pending_start >= min_duration:
                # Check debounce cooldown
                if current_time - self.last_trigger_time >= debounce:
                    self.is_active = True
                    self.last_trigger_time = current_time
                    self.pending_start = None
                    return True
                # Still within debounce cooldown
                return False

            return False
        else:
            # Condition returned to neutral
            self.is_active = False
            self.pending_start = None
            return False


class _TrackGestureState:
    """Maintains state machines for all gestures for a single face track."""

    def __init__(self) -> None:
        self.last_seen: float = 0.0
        # Continuous gestures
        self.mouth_state = _ContinuousState()
        self.head_turn_left_state = _ContinuousState()
        self.head_turn_right_state = _ContinuousState()
        self.head_tilt_state = _ContinuousState()
        self.both_eyes_state = _ContinuousState()

        # Transient blink tracking
        self.left_eye_close_start: Optional[float] = None
        self.right_eye_close_start: Optional[float] = None
        self.both_eyes_close_start: Optional[float] = None
        self.last_left_blink_time: float = -999.0
        self.last_right_blink_time: float = -999.0
        self.last_both_blink_time: float = -999.0


class GestureDetector:
    """Event-based face gesture detection and temporal filtering engine.

    Evaluates observable gestures per face using blendshapes, dense landmarks,
    and 3D head pose. Tracks gesture states over time to ensure robust temporal
    filtering, debouncing, and prevention of duplicate spam events while held.
    """

    def __init__(
        self,
        config: Optional[GestureConfig] = None,
        on_gesture: Optional[Callable[[GestureEvent], None]] = None,
        enabled_gestures: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize GestureDetector.

        Args:
            config: Optional custom GestureConfig. If None, default thresholds are used.
            on_gesture: Optional callback invoked immediately whenever a gesture event triggers.
            enabled_gestures: Optional sequence of gesture names / aliases to enable.
        """
        self.config = config or GestureConfig()
        if enabled_gestures is not None:
            resolved_set: Set[str] = set()
            for g in enabled_gestures:
                canon = normalize_gesture_name(g)
                if canon:
                    resolved_set.add(canon)
            self.config.enabled_gestures = resolved_set

        self.on_gesture = on_gesture
        # Map track_id (or face key) -> _TrackGestureState
        self._states: Dict[Union[int, str], _TrackGestureState] = {}
        self._last_cleanup_time: float = time.perf_counter()

    def reset(self) -> None:
        """Reset all internal tracking states and history."""
        self._states.clear()

    def _cleanup_stale_tracks(self, current_time: float, max_idle: float = 2.5) -> None:
        """Prune tracks that have not been observed recently to prevent memory leaks."""
        if current_time - self._last_cleanup_time < 1.0:
            return
        self._last_cleanup_time = current_time
        stale_keys = [
            k for k, state in self._states.items() if current_time - state.last_seen > max_idle
        ]
        for k in stale_keys:
            del self._states[k]

    def process(
        self,
        faces: Sequence[FaceDetection],
        image_shape: Tuple[int, int],
        timestamp: Optional[float] = None,
    ) -> List[GestureEvent]:
        """Process detected faces and emit gesture events for newly observed actions.

        Args:
            faces: Sequence of FaceDetection objects in the current frame.
            image_shape: Image dimensions as (height, width).
            timestamp: Optional wall-clock timestamp in seconds (defaults to time.perf_counter()).

        Returns:
            List of GestureEvent instances triggered in this frame.
        """
        now = time.perf_counter() if timestamp is None else float(timestamp)
        self._cleanup_stale_tracks(now)

        if not faces:
            return []

        triggered_events: List[GestureEvent] = []

        for idx, face in enumerate(faces):
            # Resolve stable key for face (track_id if available; fallback to positional index)
            track_key: Union[int, str] = face.track_id if face.track_id is not None else f"face_{idx}"

            if track_key not in self._states:
                self._states[track_key] = _TrackGestureState()

            state = self._states[track_key]
            state.last_seen = now

            # -----------------------------------------------------------------
            # 1. Evaluate Observable State Metrics
            # -----------------------------------------------------------------
            left_closed: Optional[bool] = None
            right_closed: Optional[bool] = None
            mouth_open: Optional[bool] = None
            mar_val: Optional[float] = None
            ear_l_val: Optional[float] = None
            ear_r_val: Optional[float] = None

            # Priority 1: Blendshapes if already provided
            if face.blendshapes:
                bs = face.blendshapes
                if "eyeBlinkLeft" in bs:
                    left_closed = bs["eyeBlinkLeft"] >= self.config.blink_blendshape_threshold
                    ear_l_val = round(float(bs["eyeBlinkLeft"]), 3)
                if "eyeBlinkRight" in bs:
                    right_closed = bs["eyeBlinkRight"] >= self.config.blink_blendshape_threshold
                    ear_r_val = round(float(bs["eyeBlinkRight"]), 3)
                if "jawOpen" in bs:
                    mouth_open = bs["jawOpen"] >= self.config.mouth_open_blendshape_threshold
                    mar_val = round(float(bs["jawOpen"]), 3)

            # Priority 2: Dense Landmarks geometric ratios (EAR & MAR)
            if (left_closed is None or right_closed is None or mouth_open is None) and face.dense_landmarks:
                dl = face.dense_landmarks
                if len(dl) >= 388:
                    if left_closed is None:
                        ear_l = compute_eye_aspect_ratio(dl, eye="left")
                        left_closed = ear_l < self.config.blink_ear_threshold
                        ear_l_val = round(ear_l, 3)
                    if right_closed is None:
                        ear_r = compute_eye_aspect_ratio(dl, eye="right")
                        right_closed = ear_r < self.config.blink_ear_threshold
                        ear_r_val = round(ear_r, 3)
                if mouth_open is None and len(dl) >= 292:
                    mar = compute_mouth_aspect_ratio(dl)
                    mouth_open = mar > self.config.mouth_open_mar_threshold
                    mar_val = round(mar, 3)

            # Priority 3: 3D Head Pose
            pose = face.head_pose
            if pose is None and face.landmarks is not None:
                # Estimate pose without duplicate inference
                pose = estimate_head_pose(face, image_shape)

            # -----------------------------------------------------------------
            # 2. Continuous Gestures Evaluation (Mouth & Head Movements)
            # -----------------------------------------------------------------

            # Gesture: Mouth Opening
            if self.config.is_enabled(GESTURE_MOUTH_OPENING) and mouth_open is not None:
                triggered = state.mouth_state.update(
                    condition=mouth_open,
                    current_time=now,
                    min_duration=self.config.min_mouth_duration,
                    debounce=self.config.debounce_seconds,
                )
                if triggered:
                    ev = GestureEvent(
                        name=GESTURE_MOUTH_OPENING,
                        track_id=face.track_id,
                        face_index=idx,
                        timestamp=now,
                        confidence=1.0,
                        duration=round(now - (state.mouth_state.last_trigger_time - self.config.min_mouth_duration), 3),
                        state="action",
                        metadata={"mar": mar_val},
                    )
                    triggered_events.append(ev)

            # Gesture: Head Movement (Turn Left, Turn Right, Tilt)
            if pose is not None:
                # Head turn left (subject turning left: yaw < -threshold)
                if self.config.is_enabled(GESTURE_HEAD_TURN_LEFT):
                    turn_left_cond = pose.yaw < -self.config.head_turn_yaw_threshold
                    if state.head_turn_left_state.update(
                        condition=turn_left_cond,
                        current_time=now,
                        min_duration=self.config.min_head_gesture_duration,
                        debounce=self.config.debounce_seconds,
                    ):
                        ev = GestureEvent(
                            name=GESTURE_HEAD_TURN_LEFT,
                            track_id=face.track_id,
                            face_index=idx,
                            timestamp=now,
                            confidence=round(min(1.0, abs(pose.yaw) / 45.0), 2),
                            duration=self.config.min_head_gesture_duration,
                            state="action",
                            metadata={"yaw": pose.yaw, "pitch": pose.pitch, "roll": pose.roll},
                        )
                        triggered_events.append(ev)

                # Head turn right (subject turning right: yaw > +threshold)
                if self.config.is_enabled(GESTURE_HEAD_TURN_RIGHT):
                    turn_right_cond = pose.yaw > self.config.head_turn_yaw_threshold
                    if state.head_turn_right_state.update(
                        condition=turn_right_cond,
                        current_time=now,
                        min_duration=self.config.min_head_gesture_duration,
                        debounce=self.config.debounce_seconds,
                    ):
                        ev = GestureEvent(
                            name=GESTURE_HEAD_TURN_RIGHT,
                            track_id=face.track_id,
                            face_index=idx,
                            timestamp=now,
                            confidence=round(min(1.0, abs(pose.yaw) / 45.0), 2),
                            duration=self.config.min_head_gesture_duration,
                            state="action",
                            metadata={"yaw": pose.yaw, "pitch": pose.pitch, "roll": pose.roll},
                        )
                        triggered_events.append(ev)

                # Head tilt (|roll| > threshold)
                if self.config.is_enabled(GESTURE_HEAD_TILT):
                    tilt_cond = abs(pose.roll) > self.config.head_tilt_roll_threshold
                    if state.head_tilt_state.update(
                        condition=tilt_cond,
                        current_time=now,
                        min_duration=self.config.min_head_gesture_duration,
                        debounce=self.config.debounce_seconds,
                    ):
                        ev = GestureEvent(
                            name=GESTURE_HEAD_TILT,
                            track_id=face.track_id,
                            face_index=idx,
                            timestamp=now,
                            confidence=round(min(1.0, abs(pose.roll) / 35.0), 2),
                            duration=self.config.min_head_gesture_duration,
                            state="action",
                            metadata={"roll": pose.roll, "yaw": pose.yaw, "pitch": pose.pitch},
                        )
                        triggered_events.append(ev)

            # -----------------------------------------------------------------
            # 3. Eye Gestures Evaluation (Blink, Wink, Both Eyes Closed)
            # -----------------------------------------------------------------
            if left_closed is not None and right_closed is not None:
                both_closed = left_closed and right_closed
                left_only_closed = left_closed and not right_closed
                right_only_closed = right_closed and not left_closed

                # Continuous both_eyes_closed (held closure >= both_eyes_closed_min_duration)
                if self.config.is_enabled(GESTURE_BOTH_EYES_CLOSED):
                    both_eyes_triggered = state.both_eyes_state.update(
                        condition=both_closed,
                        current_time=now,
                        min_duration=self.config.both_eyes_closed_min_duration,
                        debounce=self.config.debounce_seconds,
                    )
                    if both_eyes_triggered:
                        ev = GestureEvent(
                            name=GESTURE_BOTH_EYES_CLOSED,
                            track_id=face.track_id,
                            face_index=idx,
                            timestamp=now,
                            confidence=1.0,
                            duration=self.config.both_eyes_closed_min_duration,
                            state="action",
                            metadata={"ear_left": ear_l_val, "ear_right": ear_r_val},
                        )
                        triggered_events.append(ev)

                # Transient both eyes closure tracking for quick blinks
                if both_closed:
                    if state.both_eyes_close_start is None:
                        state.both_eyes_close_start = now
                    # Invalidate single-eye wink candidates when both eyes close
                    state.left_eye_close_start = None
                    state.right_eye_close_start = None
                else:
                    if state.both_eyes_close_start is not None:
                        duration = now - state.both_eyes_close_start
                        state.both_eyes_close_start = None
                        # If eyes closed and reopened within blink window and didn't already fire held
                        if (
                            self.config.is_enabled(GESTURE_BOTH_EYES_CLOSED)
                            and self.config.min_blink_duration <= duration < self.config.both_eyes_closed_min_duration
                            and now - state.last_both_blink_time >= self.config.debounce_seconds
                        ):
                            state.last_both_blink_time = now
                            ev = GestureEvent(
                                name=GESTURE_BOTH_EYES_CLOSED,
                                track_id=face.track_id,
                                face_index=idx,
                                timestamp=now,
                                confidence=1.0,
                                duration=round(duration, 3),
                                state="action",
                                metadata={"ear_left": ear_l_val, "ear_right": ear_r_val},
                            )
                            triggered_events.append(ev)

                # Left Eye Blink / Wink (left closed, right open)
                if self.config.is_enabled(GESTURE_LEFT_EYE_BLINK):
                    if left_only_closed:
                        if state.left_eye_close_start is None:
                            state.left_eye_close_start = now
                    else:
                        if state.left_eye_close_start is not None:
                            duration = now - state.left_eye_close_start
                            state.left_eye_close_start = None
                            if (
                                self.config.min_blink_duration <= duration <= self.config.max_blink_duration
                                and now - state.last_left_blink_time >= self.config.debounce_seconds
                            ):
                                state.last_left_blink_time = now
                                ev = GestureEvent(
                                    name=GESTURE_LEFT_EYE_BLINK,
                                    track_id=face.track_id,
                                    face_index=idx,
                                    timestamp=now,
                                    confidence=1.0,
                                    duration=round(duration, 3),
                                    state="action",
                                    metadata={"ear_left": ear_l_val, "ear_right": ear_r_val},
                                )
                                triggered_events.append(ev)

                # Right Eye Blink / Wink (right closed, left open)
                if self.config.is_enabled(GESTURE_RIGHT_EYE_BLINK):
                    if right_only_closed:
                        if state.right_eye_close_start is None:
                            state.right_eye_close_start = now
                    else:
                        if state.right_eye_close_start is not None:
                            duration = now - state.right_eye_close_start
                            state.right_eye_close_start = None
                            if (
                                self.config.min_blink_duration <= duration <= self.config.max_blink_duration
                                and now - state.last_right_blink_time >= self.config.debounce_seconds
                            ):
                                state.last_right_blink_time = now
                                ev = GestureEvent(
                                    name=GESTURE_RIGHT_EYE_BLINK,
                                    track_id=face.track_id,
                                    face_index=idx,
                                    timestamp=now,
                                    confidence=1.0,
                                    duration=round(duration, 3),
                                    state="action",
                                    metadata={"ear_left": ear_l_val, "ear_right": ear_r_val},
                                )
                                triggered_events.append(ev)

        # Dispatch callbacks if user provided on_gesture
        if self.on_gesture is not None:
            for ev in triggered_events:
                try:
                    self.on_gesture(ev)
                except Exception:
                    # User callback exceptions must never crash camera preview
                    pass

        return triggered_events
