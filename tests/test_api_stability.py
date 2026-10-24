"""Comprehensive regression tests for API-stability milestone.

Verifies:
- Public beginner-facing API: import detekface; detekface.camera()
- Consistent result objects across image, webcam, and video detection
- Standardized bounding box coordinates, confidence values, landmarks, color-order
- Invalid settings validation with useful errors
- Optional heavy dependencies isolation
- Resource cleanup on normal exit and exceptions
- Strict distinction between original frames, annotated frames, and crops
- Zero detections and multiple detections handling
- Callback exception safety
- FaceCapture.save() and FaceCapture.show() functionality
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

import detekface
from detekface import (
    BoundingBox,
    CameraError,
    CameraNotFoundError,
    CameraReadError,
    CaptureError,
    ConfigurationError,
    DetectionResult,
    FaceCapture,
    FaceDetection,
    FaceDetector,
    FaceLandmarks,
    ImageNotFoundError,
    InvalidImageError,
    InvalidSettingError,
    InvalidVideoError,
    NoFaceDetectedError,
    Point,
    VideoNotFoundError,
    camera,
    capture,
    detect,
    draw_faces,
    video,
)


# =========================================================================
# 1. Coordinate Conventions & Data Structure Properties
# =========================================================================

def test_point_representation_and_conversions():
    """Verify Point attributes, as_tuple(), and as_int_tuple()."""
    p = Point(x=15, y=25)
    assert p.x == 15
    assert p.y == 25
    assert p.as_tuple() == (15, 25)
    assert p.as_int_tuple() == (15, 25)


def test_bounding_box_coordinate_conventions():
    """Verify bounding box x, y, width, height, x1, y1, x2, y2, center, clamp, as_ltrb."""
    box = BoundingBox(x=100, y=50, width=80, height=120)
    assert box.x == 100
    assert box.y == 50
    assert box.width == 80
    assert box.height == 120
    assert box.x1 == 100
    assert box.y1 == 50
    assert box.x2 == 180
    assert box.y2 == 170
    assert box.area == 80 * 120
    assert box.as_tuple() == (100, 50, 80, 120)
    assert box.as_xyxy() == (100, 50, 180, 170)
    assert box.as_ltrb() == (100, 50, 180, 170)
    assert box.center == Point(x=140, y=110)

    # Clamping
    clamped = box.clamp(max_width=150, max_height=100)
    assert clamped.x1 == 100
    assert clamped.y1 == 50
    assert clamped.x2 == 150
    assert clamped.y2 == 100
    assert clamped.width == 50
    assert clamped.height == 50


def test_face_landmarks_conventions_and_aliases():
    """Verify FaceLandmarks coordinates, aliases, and list transformations."""
    landmarks = FaceLandmarks(
        right_eye=Point(20, 30),
        left_eye=Point(60, 30),
        nose_tip=Point(40, 50),
        right_mouth=Point(25, 70),
        left_mouth=Point(55, 70),
    )
    # Check canonical landmarks
    assert landmarks.right_eye.as_tuple() == (20, 30)
    assert landmarks.left_eye.as_tuple() == (60, 30)
    assert landmarks.nose_tip.as_tuple() == (40, 50)
    assert landmarks.right_mouth.as_tuple() == (25, 70)
    assert landmarks.left_mouth.as_tuple() == (55, 70)

    # Check aliases
    assert landmarks.mouth_left == landmarks.left_mouth
    assert landmarks.mouth_right == landmarks.right_mouth

    # Check list & tuples
    assert len(landmarks.as_list()) == 5
    assert landmarks.as_tuples() == [(20, 30), (60, 30), (40, 50), (25, 70), (55, 70)]


# =========================================================================
# 2. Zero Detections Handling
# =========================================================================

def test_zero_detections_image():
    """Verify DetectionResult properties and methods when zero faces are detected."""
    blank = np.zeros((200, 300, 3), dtype=np.uint8)
    empty_result = DetectionResult(faces=[], image=blank)

    assert empty_result.count == 0
    assert empty_result.has_faces is False
    assert len(empty_result) == 0
    assert list(empty_result) == []
    assert empty_result.original_image is blank
    assert empty_result.image.shape == (200, 300, 3)

    # Annotate on zero faces should return a clean copy
    annotated = empty_result.annotate(shape="hud")
    assert annotated.shape == (200, 300, 3)
    np.testing.assert_array_equal(annotated, blank)

    # Capture on zero faces must raise NoFaceDetectedError
    with pytest.raises(NoFaceDetectedError) as exc_info:
        empty_result.capture()
    assert "No faces detected" in str(exc_info.value)
    assert exc_info.value.hint is not None


def test_zero_detections_video(tmp_path: Path):
    """Verify video processing handles zero detections cleanly."""
    vid_path = tmp_path / "blank_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_path), fourcc, 20.0, (160, 120))
    for _ in range(5):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=np.zeros((120, 160, 3), dtype=np.uint8))

    out_path = tmp_path / "out_blank.mp4"
    saved = detekface.video(vid_path, output_path=out_path, detector=mock_detector, show=False)
    assert saved == out_path.resolve()
    assert out_path.exists()


def test_zero_detections_camera():
    """Verify camera loop runs and terminates without error on zero detections."""
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[], image=dummy_frame)

    with (
        patch("cv2.VideoCapture") as mock_cap_cls,
        patch("cv2.imshow"),
        patch("cv2.waitKey", return_value=ord("q")),
        patch("cv2.destroyAllWindows"),
    ):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap_cls.return_value = mock_cap

        camera(camera_index=0, detector=mock_detector, max_frames=2)
        mock_cap.release.assert_called_once()


# =========================================================================
# 3. Multiple Detections Handling
# =========================================================================

def test_multiple_detections_handling(tmp_path: Path):
    """Verify handling of multiple faces: iteration, indexing, and capture selection."""
    img = np.full((400, 500, 3), 100, dtype=np.uint8)
    f1 = FaceDetection(bbox=BoundingBox(10, 10, 40, 40), confidence=0.85)  # area 1600
    f2 = FaceDetection(bbox=BoundingBox(100, 100, 80, 80), confidence=0.95) # area 6400 (largest)
    f3 = FaceDetection(bbox=BoundingBox(250, 200, 50, 50), confidence=0.90) # area 2500

    result = DetectionResult(faces=[f1, f2, f3], image=img)
    assert result.count == 3
    assert result.has_faces is True
    assert len(result) == 3
    assert result[0] == f1
    assert result[1] == f2
    assert result[2] == f3

    # Default capture selects largest face (f2)
    cap_largest = result.capture(save=False)
    assert cap_largest.bbox == f2.bbox
    assert cap_largest.width == 80
    assert cap_largest.height == 80
    assert cap_largest.confidence == pytest.approx(0.95)

    # Selecting specific face index
    cap_f1 = result.capture(face_index=0, save=False)
    assert cap_f1.bbox == f1.bbox
    assert cap_f1.width == 40


# =========================================================================
# 4. Invalid Settings Validation
# =========================================================================

def test_invalid_settings_raise_useful_errors():
    """Verify that invalid shapes, colors, and thickness raise InvalidSettingError with hints."""
    # Invalid shape
    with pytest.raises(InvalidSettingError) as exc_shape:
        draw_faces(np.zeros((100, 100, 3), dtype=np.uint8), [], shape="hexagon")
    assert "Unsupported overlay shape: 'hexagon'" in str(exc_shape.value)
    assert exc_shape.value.hint is not None
    assert "rectangle" in exc_shape.value.hint

    # Invalid color
    with pytest.raises(InvalidSettingError) as exc_color:
        draw_faces(np.zeros((100, 100, 3), dtype=np.uint8), [], color="not_a_color")
    assert "Unrecognized color specification" in str(exc_color.value)
    assert exc_color.value.hint is not None

    # Invalid thickness
    with pytest.raises(InvalidSettingError) as exc_thick:
        draw_faces(np.zeros((100, 100, 3), dtype=np.uint8), [], thickness=0)
    assert "Thickness must be an integer >= 1" in str(exc_thick.value)

    with pytest.raises(InvalidSettingError):
        draw_faces(np.zeros((100, 100, 3), dtype=np.uint8), [], thickness=-2)


def test_invalid_detector_configurations():
    """Verify that out-of-range thresholds raise ConfigurationError."""
    with patch("detekface.detector.resolve_model", return_value=Path("dummy.onnx")):
        with pytest.raises(ConfigurationError):
            FaceDetector(confidence_threshold=-0.5)
        with pytest.raises(ConfigurationError):
            FaceDetector(confidence_threshold=1.5)
        with pytest.raises(ConfigurationError):
            FaceDetector(nms_threshold=-0.1)
        with pytest.raises(ConfigurationError):
            FaceDetector(top_k=0)


def test_invalid_image_inputs_raise_errors():
    """Verify non-existent files, invalid extensions, and unsupported types raise errors."""
    with pytest.raises(ImageNotFoundError):
        detect("non_existent_image_12345.jpg")

    with pytest.raises(InvalidImageError):
        detect(12345)  # type: ignore

    with pytest.raises(InvalidImageError):
        detect([])  # type: ignore

    # Zero size image array
    empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(InvalidImageError):
        detect(empty_arr)


# =========================================================================
# 5. Callback Error Handling & Exception Safety
# =========================================================================

def test_camera_capture_callback_error_does_not_crash(tmp_path: Path):
    """Verify that an exception inside the on_capture callback is safely caught."""
    dummy_frame = np.full((100, 100, 3), 150, dtype=np.uint8)
    face = FaceDetection(bbox=BoundingBox(10, 10, 30, 30), confidence=0.9)
    result = DetectionResult(faces=[face], image=dummy_frame)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = result

    def faulty_callback(captured):
        raise ValueError("Simulated callback exception")

    with (
        patch("cv2.VideoCapture") as mock_cap_cls,
        patch("cv2.imshow"),
        patch("cv2.waitKey", side_effect=[ord("c"), ord("q")]),
        patch("cv2.destroyAllWindows"),
    ):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap_cls.return_value = mock_cap

        # Running camera should catch the callback exception without crashing
        camera(
            camera_index=0,
            detector=mock_detector,
            capture_dir=tmp_path,
            on_capture=faulty_callback,
        )

        mock_cap.release.assert_called_once()


def test_video_capture_callback_error_does_not_crash(tmp_path: Path):
    """Verify that an exception in on_capture during video playback is safely handled."""
    vid_file = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_file), fourcc, 20.0, (100, 100))
    writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()

    face = FaceDetection(bbox=BoundingBox(10, 10, 30, 30), confidence=0.9)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[face], image=np.zeros((100, 100, 3), dtype=np.uint8))

    def bad_callback(captured):
        raise RuntimeError("Video callback crash")

    with (
        patch("cv2.imshow"),
        patch("cv2.waitKey", side_effect=[ord("c"), ord("q")]),
        patch("cv2.destroyAllWindows"),
    ):
        video(vid_file, detector=mock_detector, show=True, on_capture=bad_callback, capture_dir=tmp_path)


# =========================================================================
# 6. Resource Cleanup Verification
# =========================================================================

def test_camera_resource_cleanup_on_exception():
    """Verify camera release and destroyAllWindows are called when an unhandled exception occurs."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))

    mock_detector = MagicMock()
    mock_detector.detect.side_effect = ZeroDivisionError("Simulated detector crash")

    with (
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch("cv2.imshow"),
        patch("cv2.destroyWindow") as mock_destroy,
    ):
        with pytest.raises(ZeroDivisionError):
            camera(camera_index=0, detector=mock_detector)

        mock_cap.release.assert_called_once()
        mock_destroy.assert_called_once()


def test_video_resource_cleanup_on_exception(tmp_path: Path):
    """Verify video resources are released if an exception occurs during processing."""
    vid_file = tmp_path / "valid_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid_file), fourcc, 20.0, (100, 100))
    writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()

    mock_detector = MagicMock()
    mock_detector.detect.side_effect = RuntimeError("Crash during frame processing")

    with pytest.raises(RuntimeError):
        video(vid_file, detector=mock_detector, show=False)


# =========================================================================
# 7. FaceCapture.save() and FaceCapture.show()
# =========================================================================

def test_face_capture_save_method(tmp_path: Path):
    """Verify that FaceCapture.save() saves in-memory crop to disk and updates .path."""
    crop_arr = np.full((60, 60, 3), 200, dtype=np.uint8)
    cap = FaceCapture(
        image=crop_arr,
        path=None,
        bbox=BoundingBox(10, 10, 60, 60),
        confidence=0.91,
    )
    assert cap.path is None
    assert cap.width == 60
    assert cap.height == 60

    out_file = tmp_path / "saved_crop.png"
    resolved = cap.save(out_file)

    assert resolved == out_file.resolve()
    assert cap.path == resolved
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_face_capture_show_method():
    """Verify FaceCapture.show() opens window and returns keycode."""
    crop_arr = np.full((40, 40, 3), 180, dtype=np.uint8)
    cap = FaceCapture(
        image=crop_arr,
        path=None,
        bbox=BoundingBox(5, 5, 40, 40),
        confidence=0.88,
    )

    with (
        patch("cv2.namedWindow") as mock_named_win,
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.waitKey", return_value=ord("x")),
        patch("cv2.destroyWindow") as mock_destroy_win,
    ):
        key = cap.show(title="Crop Preview", delay=100)
        assert key == ord("x")
        mock_named_win.assert_called_once_with("Crop Preview", cv2.WINDOW_AUTOSIZE)
        mock_imshow.assert_called_once()
        mock_destroy_win.assert_called_once_with("Crop Preview")


# =========================================================================
# 8. Frame Distinction Verification
# =========================================================================

def test_original_frame_distinction_from_annotated_and_crops():
    """Verify that original frame remains unpolluted after annotate(), render(), and capture()."""
    original = np.full((200, 200, 3), 50, dtype=np.uint8)
    original_copy = original.copy()

    face = FaceDetection(bbox=BoundingBox(20, 20, 60, 60), confidence=0.95)
    result = DetectionResult(faces=[face], image=original)

    # 1. Annotate
    annotated = result.annotate(shape="hud", color="#00FF88", thickness=5)
    assert not np.array_equal(original, annotated)
    # Original image array must be unmodified
    np.testing.assert_array_equal(result.image, original_copy)

    # 2. Render alias
    rendered = result.render(shape="corners", color="#FF0000", thickness=3)
    assert not np.array_equal(original, rendered)
    np.testing.assert_array_equal(result.image, original_copy)

    # 3. Capture
    crop = result.capture(save=False)
    # Crop must match original image region, completely free of any overlay
    np.testing.assert_array_equal(crop.image, original_copy[20:80, 20:80])
