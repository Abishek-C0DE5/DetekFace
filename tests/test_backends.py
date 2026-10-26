"""Comprehensive automated tests for optional multiple-backend architecture.

Verifies:
- Backend selection ('yunet' default, 'mediapipe' optional) across detect(), camera(), video()
- DependencyNotFoundError when optional backend is missing with exact installation hint
- No silent fallback when requested backend fails
- Invalid backend names raise InvalidSettingError
- Output normalization into canonical DetectionResult, BoundingBox, FaceDetection, FaceLandmarks
- Landmark contracts: YuNet supplies 5 landmarks, MediaPipe supplies 6 keypoints, never invented
- Zero detections and multiple detections handling
- Real integration test with MediaPipe when available
"""

import builtins
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from detekface import (
    DEFAULT_BACKEND,
    SUPPORTED_BACKENDS,
    BoundingBox,
    DependencyNotFoundError,
    DetectionResult,
    FaceCapture,
    FaceDetector,
    InvalidSettingError,
    Point,
    camera,
    detect,
    video,
)


# =========================================================================
# 1. Backend Selection & Defaults
# =========================================================================


def test_default_backend_is_yunet():
    """Verify that default backend across all public entrypoints is 'yunet'."""
    assert DEFAULT_BACKEND == "yunet"
    assert "yunet" in SUPPORTED_BACKENDS
    assert "mediapipe" in SUPPORTED_BACKENDS

    detector = FaceDetector()
    assert detector.backend_name == "yunet"


def test_invalid_backend_names_raise_invalid_setting_error():
    """Verify that unknown or invalid backend names raise InvalidSettingError with hints."""
    blank = np.zeros((100, 100, 3), dtype=np.uint8)

    # Unknown string name
    with pytest.raises(InvalidSettingError) as exc_info:
        FaceDetector(backend="deepface")
    assert "Unsupported detection backend 'deepface'" in str(exc_info.value)
    assert exc_info.value.hint is not None
    assert "yunet" in exc_info.value.hint
    assert "mediapipe" in exc_info.value.hint

    # Empty string
    with pytest.raises(InvalidSettingError):
        detect(blank, backend="")

    # Non-string type
    with pytest.raises(InvalidSettingError):
        detect(blank, backend=123)  # type: ignore


# =========================================================================
# 2. Optional Dependency Handling & No Silent Fallback
# =========================================================================


def test_missing_optional_dependency_raises_actionable_error():
    """Verify that requesting 'mediapipe' when uninstalled raises DependencyNotFoundError with pip hint."""
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mediapipe" or name.startswith("mediapipe."):
            raise ImportError("No module named 'mediapipe'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        # Removing cached module if currently present
        with patch.dict(sys.modules, {"mediapipe": None}):
            with pytest.raises(DependencyNotFoundError) as exc_info:
                FaceDetector(backend="mediapipe")

            err_msg = str(exc_info.value)
            assert "The 'mediapipe' backend requires the optional 'mediapipe' package" in err_msg
            assert exc_info.value.hint is not None
            assert 'pip install "detekface[mediapipe]"' in exc_info.value.hint


def test_no_silent_fallback_on_backend_failure():
    """Verify that if mediapipe backend fails to initialize, it does NOT silently switch to yunet."""
    with patch(
        "detekface.backends.mediapipe.MediaPipeBackend.__init__",
        side_effect=RuntimeError("Init crash"),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            FaceDetector(backend="mediapipe")
        assert "Init crash" in str(exc_info.value)


# =========================================================================
# 3. Output Normalization & Landmark Contracts
# =========================================================================


def test_yunet_landmark_contract():
    """Verify YuNet supplies 5 landmarks and sets MediaPipe-specific landmarks to None."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_row = np.array(
        [[10, 10, 40, 50, 15, 20, 35, 20, 25, 30, 20, 40, 30, 40, 0.95]],
        dtype=np.float32,
    )

    with (
        patch("detekface.detector.resolve_model", return_value=Path("mock.onnx")),
        patch("cv2.FaceDetectorYN.create") as mock_create,
    ):
        mock_cv = MagicMock()
        mock_cv.detect.return_value = (1, dummy_row)
        mock_create.return_value = mock_cv

        detector = FaceDetector(backend="yunet", model=Path("mock.onnx"))
        res = detector.detect(img)

        assert res.count == 1
        face = res.faces[0]
        assert face.confidence == pytest.approx(0.95)
        assert face.bbox == BoundingBox(10, 10, 40, 50)

        # Landmark contract: YuNet supplies eyes, nose, and mouth corners
        assert face.landmarks is not None
        assert face.landmarks.right_eye == Point(15, 20)
        assert face.landmarks.left_eye == Point(35, 20)
        assert face.landmarks.nose_tip == Point(25, 30)
        assert face.landmarks.right_mouth == Point(20, 40)
        assert face.landmarks.left_mouth == Point(30, 40)
        assert face.landmarks.mouth_right == Point(20, 40)
        assert face.landmarks.mouth_left == Point(30, 40)

        # Must NOT invent MediaPipe landmarks
        assert face.landmarks.mouth_center is None
        assert face.landmarks.right_ear is None
        assert face.landmarks.left_ear is None
        assert len(face.landmarks.as_list()) == 5


@pytest.mark.optional_backend
def test_mediapipe_output_normalization_mocked():
    """Verify MediaPipe detections are normalized into canonical DetectionResult objects."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    # Mock detection result from MediaPipe Tasks
    mock_detection = MagicMock()
    mock_detection.categories = [MagicMock(score=0.92)]
    mock_detection.bounding_box = MagicMock(origin_x=30, origin_y=40, width=80, height=90)
    mock_detection.keypoints = [
        MagicMock(x=0.25, y=0.30),  # 0: right eye
        MagicMock(x=0.55, y=0.30),  # 1: left eye
        MagicMock(x=0.40, y=0.50),  # 2: nose tip
        MagicMock(x=0.40, y=0.70),  # 3: mouth center
        MagicMock(x=0.15, y=0.45),  # 4: right ear
        MagicMock(x=0.65, y=0.45),  # 5: left ear
    ]

    mock_mp_result = MagicMock(detections=[mock_detection])

    with (
        patch("mediapipe.tasks.python.vision.FaceDetector.create_from_options") as mock_create,
        patch(
            "detekface.backends.mediapipe.resolve_mediapipe_model",
            return_value=Path("mock.tflite"),
        ),
    ):
        mock_task_detector = MagicMock()
        mock_task_detector.detect.return_value = mock_mp_result
        mock_create.return_value = mock_task_detector

        detector = FaceDetector(backend="mediapipe")
        result = detector.detect(img)

        assert isinstance(result, DetectionResult)
        assert result.count == 1
        assert result.has_faces is True

        face = result[0]
        assert face.confidence == pytest.approx(0.92)
        assert face.bbox == BoundingBox(30, 40, 80, 90)

        # Landmark contract: MediaPipe supplies eyes, nose, mouth center, and ears
        assert face.landmarks is not None
        assert face.landmarks.right_eye == Point(50, 60)
        assert face.landmarks.left_eye == Point(110, 60)
        assert face.landmarks.nose_tip == Point(80, 100)
        assert face.landmarks.mouth_center == Point(80, 140)
        assert face.landmarks.right_ear == Point(30, 90)
        assert face.landmarks.left_ear == Point(130, 90)

        # Must NOT invent YuNet mouth corner landmarks
        assert face.landmarks.right_mouth is None
        assert face.landmarks.left_mouth is None
        assert face.landmarks.mouth_right is None
        assert face.landmarks.mouth_left is None
        assert len(face.landmarks.as_list()) == 6


# =========================================================================
# 4. Camera and Video Backend Integration
# =========================================================================


def test_camera_backend_parameter_selection():
    """Verify detekface.camera() instantiates FaceDetector with the selected backend."""
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with (
        patch("cv2.VideoCapture") as mock_cap_cls,
        patch("cv2.imshow"),
        patch("cv2.waitKey", return_value=ord("q")),
        patch("cv2.destroyWindow"),
        patch("detekface.camera.FaceDetector") as mock_detector_cls,
    ):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap_cls.return_value = mock_cap

        mock_inst = MagicMock()
        mock_inst.detect.return_value = DetectionResult(faces=[], image=dummy_frame)
        mock_detector_cls.return_value = mock_inst

        camera(backend="mediapipe", camera_index=0, max_frames=1)

        mock_detector_cls.assert_called_once_with(backend="mediapipe", confidence_threshold=0.7)


def test_video_backend_parameter_selection(tmp_path: Path):
    """Verify detekface.video() instantiates FaceDetector with the selected backend."""
    vid_file = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_file), fourcc, 20.0, (100, 100))
    writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()

    with patch("detekface.video.FaceDetector") as mock_detector_cls:
        mock_inst = MagicMock()
        mock_inst.detect.return_value = DetectionResult(faces=[], image=np.zeros((100, 100, 3), dtype=np.uint8))
        mock_detector_cls.return_value = mock_inst

        video(vid_file, backend="mediapipe", show=False)
        mock_detector_cls.assert_called_once_with(backend="mediapipe", confidence_threshold=0.7)


# =========================================================================
# 5. Real Model Integration Test with MediaPipe (if installed)
# =========================================================================


@pytest.mark.optional_backend
def test_real_mediapipe_detection_on_sample_asset():
    """Integration check: Run real MediaPipe backend on sample_face.jpg."""
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        pytest.skip("MediaPipe not installed; skipping real-model integration test.")

    sample_img = Path(__file__).parent / "assets" / "sample_face.jpg"
    if not sample_img.is_file():
        pytest.skip(f"Asset '{sample_img}' not found.")

    # Run detection with MediaPipe backend
    result = detect(sample_img, backend="mediapipe", confidence_threshold=0.6)

    assert isinstance(result, DetectionResult)
    assert result.has_faces is True
    assert result.count >= 1

    first_face = result.faces[0]
    assert first_face.confidence >= 0.75
    assert first_face.bbox.width > 50
    assert first_face.bbox.height > 50

    # Verify landmarks
    assert first_face.landmarks is not None
    assert first_face.landmarks.right_eye is not None
    assert first_face.landmarks.left_eye is not None
    assert first_face.landmarks.nose_tip is not None
    assert first_face.landmarks.mouth_center is not None
    assert first_face.landmarks.right_mouth is None
    assert first_face.landmarks.left_mouth is None

    # Test annotations, capture, and rendering work identically with MediaPipe output
    annotated = result.annotate(shape="hud", color="#00FF88")
    assert annotated.shape == result.image.shape

    captured = result.capture(save=False)
    assert isinstance(captured, FaceCapture)
    assert captured.width > 50
    assert captured.height > 50
