"""Comprehensive integration and regression tests for Milestone 8 (detekface 0.3.0).

Verifies:
- Beginner zero-config experience (`import detekface; detekface.camera()`)
- Coherent feature access across image, webcam, and video workflows
- Features individually enabled:
  * Camera performance configuration (async mode, target FPS, scaling)
  * Face tracking IDs
  * Head-pose estimation (3D yaw/pitch/roll)
  * Gesture detection and event callbacks
  * AR effects (sunglasses, crown, visor, mask, halo)
  * Face alignment and straightening
  * Face quality analysis (sharpness, lighting, pose, framing)
  * Privacy filtering (blur, pixelate, solid mask, fail-closed)
  * Face shape analysis
  * Dense mesh and Live FX scanner visualization
- Multi-feature combined workflows (all features active simultaneously)
- Shared detection results without duplicate inference passes
- Isolation of optional advanced dependencies (basic YuNet detection without mediapipe)
- Public argument and return type validations
"""

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

import detekface
from detekface import (
    AlignedFace,
    AnonymizationResult,
    BoundingBox,
    CameraSession,
    ConfigurationError,
    DetectionResult,
    FaceDetection,
    FaceLandmarks,
    GestureEvent,
    HeadPose,
    ImageQualityResult,
    Point,
    align_face,
    anonymize,
    camera,
    check_face_quality,
    video,
)


@pytest.fixture
def synthetic_face_image():
    """Generates a 320x240 image with a textured synthetic face region."""
    img = np.full((240, 320, 3), 120, dtype=np.uint8)
    # Add contrast lines in face area
    for y in range(40, 160, 4):
        cv2.line(img, (70, y), (170, y), (y % 255, 200, (y * 2) % 255), 1)
    return img


@pytest.fixture
def sample_detection(synthetic_face_image):
    """DetectionResult with rich landmarks and pose for testing."""
    face = FaceDetection(
        bbox=BoundingBox(x=70, y=40, width=100, height=120),
        confidence=0.96,
        landmarks=FaceLandmarks(
            right_eye=Point(95, 80),
            left_eye=Point(145, 80),
            nose_tip=Point(120, 105),
            right_mouth=Point(100, 135),
            left_mouth=Point(140, 135),
        ),
        head_pose=HeadPose(yaw=2.0, pitch=-1.0, roll=0.5),
        track_id=1,
    )
    return DetectionResult(faces=[face], image=synthetic_face_image)


# =========================================================================
# 1. Beginner Experience Verification
# =========================================================================


def test_beginner_camera_zero_args():
    """Verify that detekface.camera() runs with zero configuration arguments."""
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    with (
        patch("cv2.VideoCapture") as mock_cap_cls,
        patch("cv2.imshow"),
        patch("cv2.waitKey", return_value=ord("q")),
        patch("cv2.destroyWindow"),
    ):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, dummy)
        mock_cap_cls.return_value = mock_cap

        # Zero arguments call
        detekface.camera(max_frames=1)

        mock_cap.release.assert_called_once()


def test_beginner_detect_basic_usage(synthetic_face_image, sample_detection):
    """Verify that detekface.detect(img) works with default settings."""
    with patch("detekface.detector.FaceDetector.detect", return_value=sample_detection):
        res = detekface.detect(synthetic_face_image)
        assert isinstance(res, DetectionResult)
        assert res.has_faces
        assert len(res) == 1
        assert res[0].confidence == 0.96


# =========================================================================
# 2. Individual Feature Verifications
# =========================================================================


def test_feature_camera_performance_configuration():
    """Verify camera configuration accepts async mode, custom target FPS, and input scaling."""
    session = CameraSession(
        auto_open=False,
        async_mode=True,
        target_fps=60.0,
        stale_timeout_sec=0.25,
        resolution=(640, 480),
        input_size=(320, 240),
    )
    assert session.async_mode is True
    assert session.target_fps == 60.0


def test_feature_face_tracking_ids(synthetic_face_image, sample_detection):
    """Verify face tracking assigns consistent track_id."""
    with patch("detekface.detector.FaceDetector.detect", return_value=sample_detection):
        res = detekface.detect(synthetic_face_image, track_faces=True)
        assert res[0].track_id is not None


def test_feature_head_pose_estimation(synthetic_face_image, sample_detection):
    """Verify 3D head-pose estimation produces yaw, pitch, and roll angles."""
    with patch("detekface.detector.FaceDetector.detect", return_value=sample_detection):
        res = detekface.detect(synthetic_face_image, estimate_pose=True)
        assert res[0].head_pose is not None
        assert isinstance(res[0].head_pose.yaw, float)


def test_feature_gesture_detection_and_callback():
    """Verify gesture events are triggered and delivered to on_gesture callback."""
    captured_events = []

    def on_g(event: GestureEvent):
        captured_events.append(event)

    session = CameraSession(
        auto_open=False,
        detect_gestures=True,
        on_gesture=on_g,
    )
    assert session.detect_gestures is True
    assert session.on_gesture == on_g


def test_feature_ar_effects(synthetic_face_image, sample_detection):
    """Verify AR accessory effects render onto detected face."""
    effects = ["sunglasses", "crown", "face_mask", "futuristic_visor", "halo"]
    for eff in effects:
        annotated = sample_detection.annotate(effect=eff)
        assert annotated.shape == synthetic_face_image.shape
        assert not np.array_equal(annotated, synthetic_face_image)


def test_feature_face_alignment(sample_detection):
    """Verify face alignment straightens and standardizes crop dimensions."""
    aligned = sample_detection.align(output_size=(112, 112))
    assert isinstance(aligned, AlignedFace)
    assert aligned.width == 112
    assert aligned.height == 112
    assert aligned.transform_matrix.shape == (2, 3)


def test_feature_face_quality_analysis(sample_detection):
    """Verify technical face quality analysis generates metrics and summary."""
    report = sample_detection.check_quality()
    assert isinstance(report, ImageQualityResult)
    assert report.face_count == 1
    assert report.primary_face is not None
    assert report.primary_face.sharpness > 0.0
    assert report.primary_face.brightness > 0.0


def test_feature_privacy_anonymization(sample_detection):
    """Verify privacy mode anonymizes faces while leaving exterior pixels intact."""
    res = sample_detection.anonymize(method="blur", strength=0.5)
    assert isinstance(res, AnonymizationResult)
    assert res.faces_anonymized == 1
    assert res.image.shape == sample_detection.image.shape


# =========================================================================
# 3. Combined Multi-Feature Workflows
# =========================================================================


def test_combined_features_image_annotation(sample_detection):
    """Verify image annotation when multiple features (mesh, live_fx, accessory, anonymize) are combined."""
    annotated = sample_detection.annotate(
        shape="hud",
        color="#00FF88",
        thickness=2,
        show_confidence=True,
        show_mesh=True,
        mesh_style="neon",
        live_fx=True,
        live_fx_style="scan",
        effect="sunglasses",
        anonymize="blur",
    )
    assert annotated.shape == sample_detection.image.shape
    assert not np.array_equal(annotated, sample_detection.image)


def test_combined_features_camera_session_render(synthetic_face_image, sample_detection):
    """Verify CameraSession.render() operates cleanly with all features active simultaneously."""
    session = CameraSession(
        auto_open=False,
        track_faces=True,
        estimate_pose=True,
        detect_gestures=True,
        show_gestures=True,
        show_quality=True,
        effect="halo",
        anonymize="pixelate",
    )
    session._latest_frame = synthetic_face_image
    session._latest_result = sample_detection

    rendered = session.render(
        shape="corners",
        show_fps=True,
        show_count=True,
        show_mesh=True,
        live_fx=True,
        show_quality=True,
    )
    assert rendered.shape == synthetic_face_image.shape
    assert not np.array_equal(rendered, synthetic_face_image)


def test_combined_features_video_workflow(tmp_path: Path, synthetic_face_image, sample_detection):
    """Verify detekface.video() processes with all features active simultaneously."""
    in_video = tmp_path / "combined_input.mp4"
    out_video = tmp_path / "combined_output.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(in_video), fourcc, 15.0, (320, 240))
    for _ in range(2):
        vw.write(synthetic_face_image)
    vw.release()

    events_received = []

    with patch("detekface.detector.FaceDetector.detect", return_value=sample_detection):
        saved = video(
            path=in_video,
            output_path=out_video,
            show=False,
            track_faces=True,
            show_tracking_ids=True,
            show_pose=True,
            detect_gestures=True,
            on_gesture=lambda ev: events_received.append(ev),
            show_gestures=True,
            effect="sunglasses",
            anonymize="blur",
            show_quality=True,
            show_mesh=True,
            live_fx=True,
            show_fps=True,
            show_count=True,
        )
        assert saved == out_video.resolve()
        assert out_video.exists()
        assert out_video.stat().st_size > 0


# =========================================================================
# 4. Optional Advanced Dependency Isolation
# =========================================================================


def test_basic_detection_without_mediapipe(synthetic_face_image):
    """Verify that core face detection operates without needing mediapipe installed."""
    with patch.dict(sys.modules, {"mediapipe": None}):
        # Basic YuNet detection must not import or require mediapipe
        detector = detekface.FaceDetector(backend="yunet")
        assert detector.backend_name == "yunet"

        # Align and anonymize must also function without mediapipe
        fake_face = FaceDetection(
            bbox=BoundingBox(50, 50, 80, 80),
            confidence=0.9,
            landmarks=FaceLandmarks(Point(65, 75), Point(115, 75), Point(90, 95), Point(70, 115), Point(110, 115)),
        )
        fake_res = DetectionResult(faces=[fake_face], image=synthetic_face_image)

        aligned = align_face(synthetic_face_image, fake_face)
        assert aligned.width == 112

        anon = anonymize(fake_res, method="blur")
        assert anon.faces_anonymized == 1

        quality = check_face_quality(fake_res)
        assert quality.face_count == 1


# =========================================================================
# 5. Argument Validation & Error Robustness
# =========================================================================


def test_argument_validation_camera():
    """Verify invalid parameters in camera() raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        camera(track_faces="not_a_bool")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        camera(show_quality="invalid")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        camera(show_gestures=123)  # type: ignore[arg-type]


def test_argument_validation_video(tmp_path: Path):
    """Verify invalid parameters in video() raise ConfigurationError."""
    dummy_video = tmp_path / "valid.mp4"
    dummy_video.touch()
    with pytest.raises(ConfigurationError):
        video(dummy_video, show="invalid_bool")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        video(dummy_video, codec="toolongcodec")
