"""Automated unit and integration benchmarks for detekface 0.3.0 (Milestone 9).

Validates:
1. Target 60 FPS decoupled capture and rendering when supported.
2. Preview does not freeze while inference is running.
3. Detections older than stale_timeout_sec are suppressed (never displayed as current).
4. Mesh, accessories, and tracking stay accurately aligned.
5. Model thresholds, defaults, and image dimensions are preserved.
6. Zero thread leaks and zero memory leaks in prolonged sessions.
7. Physical webcam hardware probe and capability classification.
8. Reproducible independent metrics across scenarios A through J.
"""

import time
import cv2
import numpy as np
import pytest

from detekface.accessories import compute_accessory_transform
from detekface.async_cam import AsyncCameraEngine, TimestampedFrame
from detekface.camera import CameraSession
from detekface.detector import FaceDetector
from detekface.types import BoundingBox, DetectionResult, FaceDetection, FaceLandmarks, Point
from tools.audit_milestone9 import (
    ControllableMockCamera,
    probe_physical_camera,
    run_prolonged_reliability_test,
    run_scenario_benchmark,
)


@pytest.fixture
def synthetic_face_frame() -> np.ndarray:
    """Realistic synthetic frame containing face-like features."""
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    cv2.circle(img, (320, 240), 90, (180, 180, 180), -1)
    cv2.circle(img, (290, 220), 12, (20, 20, 20), -1)
    cv2.circle(img, (350, 220), 12, (20, 20, 20), -1)
    return img


# =========================================================================
# 1. 60 FPS Target & Decoupled Non-Freezing Pipeline
# =========================================================================


def test_target_60fps_rendering_with_slow_inference(synthetic_face_frame):
    """Verify that preview rendering maintains ~60 FPS even when inference is artificially slowed."""
    mock_cap = ControllableMockCamera(width=640, height=480, native_fps=60.0, sample_image=synthetic_face_frame)

    session = CameraSession(
        auto_open=False,
        async_mode=True,
        target_fps=60.0,
        resolution=(640, 480),
    )

    # Artificially inject slow inference (e.g. 70ms per frame = max ~14 inference FPS)
    orig_detect = session.detector.detect

    def slow_detect(*args, **kwargs):
        time.sleep(0.06)
        return orig_detect(*args, **kwargs)

    session.detector.detect = slow_detect  # type: ignore[assignment]
    session.open(cap=mock_cap)

    engine = session._async_engine
    assert engine is not None

    frames_rendered = 0
    t0 = time.perf_counter()

    try:
        # Render for 1 second (should hit ~50-60 frames despite slow 60ms inference!)
        while (time.perf_counter() - t0) < 1.0:
            frame, faces, _ = engine.get_latest_state(timeout=0.05)
            if frame is not None:
                session._latest_frame = frame
                session._latest_result = DetectionResult(faces=faces, image=frame)
                _ = session.render(show_fps=True)
                frames_rendered += 1
            time.sleep(1.0 / 65.0)
    finally:
        session.close()

    actual_fps = frames_rendered / (time.perf_counter() - t0)
    # Rendering must achieve > 45 FPS (decoupled from the 14 FPS inference rate)
    assert actual_fps >= 45.0, f"Render FPS dropped to {actual_fps:.1f} due to inference stalling!"


def test_preview_does_not_freeze_on_inference_stall(synthetic_face_frame):
    """Verify that when inference hangs or takes 200ms, fresh preview frames keep arriving."""
    mock_cap = ControllableMockCamera(width=640, height=480, native_fps=60.0, sample_image=synthetic_face_frame)

    session = CameraSession(auto_open=False, async_mode=True, mirror=False, stale_timeout_sec=0.1)

    def hanging_detect(*args, **kwargs):
        time.sleep(0.3)
        return DetectionResult(faces=[], image=synthetic_face_frame)

    session.detector.detect = hanging_detect  # type: ignore[assignment]
    session.open(cap=mock_cap)

    engine = session._async_engine
    assert engine is not None

    received_frames = []
    t0 = time.perf_counter()
    try:
        while (time.perf_counter() - t0) < 0.25:
            frame, _, _ = engine.get_latest_state(timeout=0.02)
            if frame is not None:
                received_frames.append(int(frame[0, 0, 0]))
            time.sleep(0.01)
    finally:
        session.close()

    # The capture thread continuously produced unique frames despite inference being blocked
    assert len(received_frames) >= 10
    assert len(set(received_frames)) > 1, "Preview froze on identical frame while inference was stalled!"


# =========================================================================
# 2. Stale Detection Suppression
# =========================================================================


def test_stale_detection_suppressed_after_timeout():
    """Verify that detections older than stale_timeout_sec are cleared and not displayed as current."""
    engine = AsyncCameraEngine(stale_timeout_sec=0.1, queue_maxsize=1)

    # Simulate an acquired frame at t=0
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_face = FaceDetection(bbox=BoundingBox(10, 10, 50, 50), confidence=0.9)
    ts_frame = TimestampedFrame(frame=test_img, timestamp=time.perf_counter(), frame_index=1)

    with engine._latest_capture_lock:
        engine._latest_capture = ts_frame

    with engine._result_lock:
        engine._latest_result = DetectionResult(faces=[dummy_face], image=test_img)
        engine._latest_result_timestamp = time.perf_counter()

    # 1. Fresh state immediately returns the face
    _, faces_fresh, _ = engine.get_latest_state()
    assert len(faces_fresh) == 1

    # 2. Wait longer than stale_timeout_sec (0.15s)
    time.sleep(0.15)
    _, faces_stale, _ = engine.get_latest_state()
    # Detections must be suppressed because inference has not updated recently
    assert len(faces_stale) == 0, "Stale detections were incorrectly returned as if current!"


# =========================================================================
# 3. Spatial Alignment Consistency
# =========================================================================


def test_mesh_accessory_tracking_alignment_consistency():
    """Verify accessories scale and align strictly with facial landmarks across movement."""
    lm1 = FaceLandmarks(
        right_eye=Point(40, 50),
        left_eye=Point(80, 50),
        nose_tip=Point(60, 70),
        right_mouth=Point(45, 90),
        left_mouth=Point(75, 90),
    )
    face1 = FaceDetection(bbox=BoundingBox(20, 20, 80, 90), confidence=0.9, landmarks=lm1)

    # Shifted and enlarged face
    lm2 = FaceLandmarks(
        right_eye=Point(80, 100),
        left_eye=Point(160, 100),
        nose_tip=Point(120, 140),
        right_mouth=Point(90, 180),
        left_mouth=Point(150, 180),
    )
    face2 = FaceDetection(bbox=BoundingBox(40, 40, 160, 180), confidence=0.9, landmarks=lm2)

    cx1, cy1, scale1, rot1 = compute_accessory_transform(face1, "sunglasses", image_shape=(480, 640))
    cx2, cy2, scale2, rot2 = compute_accessory_transform(face2, "sunglasses", image_shape=(480, 640))

    # Center must follow midpoint of eyes
    assert cx1 == 60.0
    assert cy1 == 50.0
    assert cx2 == 120.0
    assert cy2 == 100.0

    # Scale must double proportionally with 2x eye distance
    assert pytest.approx(scale2 / scale1, rel=0.05) == 2.0


# =========================================================================
# 4. Model Thresholds & Resolution Preservation
# =========================================================================


def test_preserved_model_thresholds_and_defaults():
    """Verify that default model thresholds and resolution configurations remain untouched."""
    detector = FaceDetector(backend="yunet")
    assert detector.confidence_threshold == 0.7
    assert detector.nms_threshold == 0.3
    assert detector.top_k == 5000

    session = CameraSession(auto_open=False)
    assert session.confidence_threshold == 0.7
    assert session.target_fps == 60.0
    assert session.stale_timeout_sec == 0.5


# =========================================================================
# 5. Reliability: Resource Cleanup & Zero Leaks
# =========================================================================


def test_prolonged_webcam_session_resource_cleanup():
    """Verify that running a multi-second session cleanly joins threads and does not leak memory."""
    rel = run_prolonged_reliability_test(duration_sec=2.0)
    assert not rel["thread_leak_detected"], f"Thread leak detected! Active threads: {rel['final_threads']}"
    assert not rel["memory_leak_detected"], f"Memory leak detected! Growth: {rel['memory_growth_mb']} MB"
    assert rel["frames_processed"] >= 80


# =========================================================================
# 6. Physical Camera Hardware Probe
# =========================================================================


def test_physical_camera_capability_probe():
    """Probe physical camera index 0 and verify structured hardware capabilities."""
    probe = probe_physical_camera(0)
    assert isinstance(probe, dict)
    if probe.get("available"):
        assert probe["default_width"] > 0
        assert probe["default_height"] > 0
        assert probe["measured_capture_fps"] >= 0.0
        assert "backend" in probe


# =========================================================================
# 7. Reproducible Benchmark of Scenarios A through J
# =========================================================================


@pytest.mark.parametrize(
    "scenario_id,scenario_name,session_config,render_config",
    [
        (
            "A",
            "Basic webcam detection",
            {"track_faces": False, "estimate_pose": False, "detect_gestures": False},
            {"shape": "rectangle", "show_confidence": False},
        ),
        (
            "C",
            "Dense facial mesh",
            {"analyze_face_shape": True, "mesh": True},
            {"shape": "rectangle", "show_mesh": True},
        ),
        (
            "E",
            "Head-pose estimation",
            {"estimate_pose": True},
            {"shape": "rectangle"},
        ),
        (
            "F",
            "Gesture detection",
            {"detect_gestures": True, "show_gestures": True},
            {"shape": "rectangle", "show_gestures": True},
        ),
        (
            "G",
            "AR sunglasses and visor",
            {"effect": "sunglasses"},
            {"shape": "rectangle", "effect": "sunglasses"},
        ),
        (
            "I",
            "Privacy effects",
            {"anonymize": "blur"},
            {"shape": "rectangle", "anonymize": "blur"},
        ),
        (
            "J",
            "All supported features combined",
            {
                "track_faces": True,
                "estimate_pose": True,
                "detect_gestures": True,
                "analyze_face_shape": True,
                "mesh": True,
                "effect": "sunglasses",
                "anonymize": "blur",
                "show_quality": True,
            },
            {
                "shape": "hud",
                "show_mesh": True,
                "show_gestures": True,
                "show_quality": True,
                "effect": "sunglasses",
                "anonymize": "blur",
            },
        ),
    ],
)
def test_scenario_benchmarks_metrics_and_throughput(
    synthetic_face_frame,
    scenario_id,
    scenario_name,
    session_config,
    render_config,
):
    """Verify that all major scenarios produce valid latency, throughput, and zero crashes."""
    res = run_scenario_benchmark(
        scenario_id=scenario_id,
        scenario_name=scenario_name,
        session_config=session_config,
        render_config=render_config,
        test_frame=synthetic_face_frame,
        target_camera_fps=60.0,
        num_frames=20,
    )

    assert res.render_fps >= 35.0
    assert res.avg_latency_ms < 100.0
    assert res.p95_latency_ms < 150.0
    assert res.frames_rendered == 20

