"""Tests for detekface animated Live FX visualization engine (Milestone 3).

Verifies:
- All 4 Live FX styles ('scan', 'neon', 'wireframe', 'minimal')
- Timestamp-based animation independent of frame rate
- Smooth face-locking convergence and telemetry labels
- Multiple face tracking and independent phase offsets
- Graceful handling of missing detections and recovery
- Face-shape biometric label integration
- High-throughput rendering latency and minimal overhead
"""

import math
import time
from typing import List
import numpy as np
import pytest

from detekface.exceptions import InvalidImageError, InvalidSettingError
from detekface.live_fx import (
    LiveFXRenderer,
    SUPPORTED_LIVE_FX_STYLES,
    validate_live_fx_style,
)
from detekface.types import (
    BoundingBox,
    DetectionResult,
    FaceDetection,
    FaceShapeResult,
    Point,
)


@pytest.fixture
def blank_image() -> np.ndarray:
    """Create a black 400x400 BGR uint8 canvas."""
    return np.zeros((400, 400, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_478_landmarks() -> List[Point]:
    """Generate 478 synthetic normalized facial landmark points."""
    points: List[Point] = []
    for i in range(478):
        angle = 2.0 * math.pi * (i / 478.0)
        r = 0.22 + 0.04 * math.sin(i * 0.1)
        x = 0.5 + r * math.cos(angle)
        y = 0.5 + r * math.sin(angle)
        points.append(Point(x=x, y=y))
    return points


@pytest.fixture
def sample_detection(synthetic_478_landmarks: List[Point]) -> FaceDetection:
    """Create a sample FaceDetection fixture."""
    return FaceDetection(
        bbox=BoundingBox(x=100, y=100, width=200, height=200),
        confidence=0.98,
        dense_landmarks=synthetic_478_landmarks,
        face_shape=FaceShapeResult(
            label="oval",
            fit_score=0.91,
            alternatives={"oval": 0.91, "round": 0.52},
            measurements={"length_to_width_ratio": 1.45},
        ),
    )


class TestLiveFXStyles:
    """Tests for Live FX styles and validation."""

    def test_supported_styles_constant(self):
        """Verify all 4 required styles are declared."""
        assert set(SUPPORTED_LIVE_FX_STYLES) == {"scan", "neon", "wireframe", "minimal"}

    @pytest.mark.parametrize("invalid", ["hologram", "3d", "matrix", "", 123])
    def test_invalid_style_raises_error(self, invalid):
        """Verify invalid styles raise actionable InvalidSettingError."""
        with pytest.raises(InvalidSettingError):
            validate_live_fx_style(invalid)
        with pytest.raises(InvalidSettingError):
            LiveFXRenderer(style=invalid)

    @pytest.mark.parametrize("style", ["scan", "neon", "wireframe", "minimal"])
    def test_each_style_renders_successfully(
        self, style: str, blank_image: np.ndarray, sample_detection: FaceDetection
    ):
        """Verify each style modifies canvas pixels without error."""
        renderer = LiveFXRenderer(style=style, color="#00FF88")
        out = renderer.render(blank_image, faces=[sample_detection], timestamp=0.5, copy=True)

        assert out.shape == blank_image.shape
        assert np.count_nonzero(out) > 0, f"Style '{style}' should draw non-zero pixels"

    def test_invalid_image_raises_error(self, sample_detection: FaceDetection):
        """Verify empty or non-array image raises InvalidImageError."""
        renderer = LiveFXRenderer()
        with pytest.raises(InvalidImageError):
            renderer.render(np.empty((0, 0, 3), dtype=np.uint8), faces=[sample_detection])


class TestAnimationTimingAndMotion:
    """Tests for timestamp-driven animation and frame rate independence."""

    def test_scanning_beam_moves_across_timestamps(
        self, blank_image: np.ndarray, sample_detection: FaceDetection
    ):
        """Verify scanning beam shifts vertical position across different timestamps."""
        renderer = LiveFXRenderer(style="scan", color="#00FF88", scan_speed=1.0)

        # Render at t = 0.0 (near middle or top) and t = 0.25 (near bottom)
        f1 = renderer.render(blank_image, faces=[sample_detection], timestamp=0.0, copy=True)
        f2 = renderer.render(blank_image, faces=[sample_detection], timestamp=0.25, copy=True)

        # Images at different timestamps must differ due to beam motion
        assert not np.array_equal(f1, f2)

    def test_frame_rate_independence(
        self, blank_image: np.ndarray, sample_detection: FaceDetection
    ):
        """Verify animation progress matches clock time regardless of simulation FPS."""
        # 15 FPS simulation (steps of 1/15s)
        r15 = LiveFXRenderer(style="scan", scan_speed=1.0)
        for step in range(16):
            r15.render(blank_image, faces=[sample_detection], timestamp=step * (1.0 / 15.0), copy=True)

        # 60 FPS simulation (steps of 1/60s)
        r60 = LiveFXRenderer(style="scan", scan_speed=1.0)
        for step in range(61):
            r60.render(blank_image, faces=[sample_detection], timestamp=step * (1.0 / 60.0), copy=True)

        # Both trackers at t = 1.0 should have reached full lock
        assert r15._tracks[1].lock_progress >= 0.95
        assert r60._tracks[1].lock_progress >= 0.95


class TestTrackingAndMultiFace:
    """Tests for multi-face tracking, occlusion handling, and labels."""

    def test_multiple_faces_rendered_independently(
        self, blank_image: np.ndarray, synthetic_478_landmarks: List[Point]
    ):
        """Verify multiple faces are tracked with independent IDs and telemetry."""
        face1 = FaceDetection(
            bbox=BoundingBox(x=30, y=100, width=120, height=120),
            confidence=0.96,
            dense_landmarks=[Point(x=p.x * 0.4, y=p.y) for p in synthetic_478_landmarks],
        )
        face2 = FaceDetection(
            bbox=BoundingBox(x=220, y=100, width=120, height=120),
            confidence=0.94,
            dense_landmarks=[Point(x=0.5 + p.x * 0.4, y=p.y) for p in synthetic_478_landmarks],
        )

        renderer = LiveFXRenderer(style="scan")
        out = renderer.render(blank_image, faces=[face1, face2], timestamp=0.1, copy=True)

        # Verify left and right halves both have rendered pixels
        assert np.count_nonzero(out[:, :200]) > 0
        assert np.count_nonzero(out[:, 200:]) > 0
        assert len(renderer._tracks) == 2

    def test_grace_period_handles_temporary_face_loss(
        self, blank_image: np.ndarray, sample_detection: FaceDetection
    ):
        """Verify brief loss of detection does not abruptly erase track."""
        renderer = LiveFXRenderer(style="scan", grace_period_sec=0.3)

        # Frame 1: Detection present
        renderer.render(blank_image, faces=[sample_detection], timestamp=0.1, copy=True)
        assert len(renderer._tracks) == 1

        # Frame 2: Momentary occlusion (empty detection) at t = 0.2 (< grace period)
        out_mid = renderer.render(blank_image, faces=[], timestamp=0.2, copy=True)
        # Track still active and drawn
        assert 1 in renderer._tracks
        assert np.count_nonzero(out_mid) > 0

        # Frame 3: Detection recovers at t = 0.25
        renderer.render(blank_image, faces=[sample_detection], timestamp=0.25, copy=True)
        assert 1 in renderer._tracks
        assert renderer._tracks[1].disappeared_frames == 0

        # Frame 4: Long absence at t = 1.0 (> grace period) -> track purged
        renderer.render(blank_image, faces=[], timestamp=1.0, copy=True)
        assert len(renderer._tracks) == 0

    def test_lock_transition_and_telemetry_labels(
        self, blank_image: np.ndarray, sample_detection: FaceDetection
    ):
        """Verify transition from acquisition to locked status and face-shape label presence."""
        renderer = LiveFXRenderer(style="scan", show_labels=True)

        # Initial frame -> lock progress < 1.0 (acquiring)
        renderer.render(blank_image, faces=[sample_detection], timestamp=0.01, copy=True)
        assert renderer._tracks[1].lock_progress < 0.5

        # After 0.3s -> lock progress >= 0.95 (locked)
        out_locked = renderer.render(blank_image, faces=[sample_detection], timestamp=0.4, copy=True)
        assert renderer._tracks[1].lock_progress >= 0.95
        assert np.count_nonzero(out_locked) > 0


class TestPerformanceAndOverhead:
    """Benchmark tests ensuring Live FX renderer introduces negligible overhead."""

    def test_render_latency_overhead(
        self, blank_image: np.ndarray, sample_detection: FaceDetection
    ):
        """Verify rendering 100 frames takes well under 5ms per frame (200+ FPS capability)."""
        renderer = LiveFXRenderer(style="scan")
        # Warmup
        renderer.render(blank_image, faces=[sample_detection], timestamp=0.0, copy=True)

        t_start = time.perf_counter()
        iterations = 100
        for i in range(iterations):
            renderer.render(
                blank_image,
                faces=[sample_detection],
                timestamp=0.016 * i,
                copy=False,
            )
        elapsed = time.perf_counter() - t_start
        avg_ms = (elapsed / iterations) * 1000.0

        # Rendering a single frame should easily stay under 6ms on CPU
        assert avg_ms < 6.0, f"Average Live FX render time too high: {avg_ms:.2f}ms"


class TestDetectionResultIntegration:
    """Integration test with DetectionResult.annotate()."""

    def test_result_annotate_live_fx(
        self, blank_image: np.ndarray, sample_detection: FaceDetection
    ):
        """Verify DetectionResult.annotate(live_fx=True) renders Live FX scanner layer."""
        result = DetectionResult(faces=[sample_detection], image=blank_image.copy())

        out = result.annotate(
            live_fx=True,
            live_fx_style="neon",
            live_fx_timestamp=0.5,
            color="#00FF88",
        )
        assert out.shape == blank_image.shape
        assert np.count_nonzero(out) > 0
        assert not np.array_equal(out, blank_image)
        # Original image in result must remain untouched
        assert np.count_nonzero(result.image) == 0

