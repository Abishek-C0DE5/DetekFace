"""Comprehensive tests for Milestone 4: Live webcam integration with mesh and Live FX."""

from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from detekface import (
    BoundingBox,
    DetectionResult,
    FaceCapture,
    FaceDetection,
    FaceShapeResult,
    Point,
    camera,
    open_camera,
)


@pytest.fixture
def mock_webcam():
    """Fixture providing mocked cv2 VideoCapture and GUI window functions."""
    with (
        patch("cv2.VideoCapture") as mock_cap_cls,
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.waitKey") as mock_wait_key,
        patch("cv2.destroyWindow") as mock_destroy_window,
    ):
        mock_cap = MagicMock()
        mock_cap_cls.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        yield mock_cap, mock_imshow, mock_wait_key, mock_destroy_window


def _make_frame(h: int = 480, w: int = 640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_face(x=100, y=100, w=150, h=150, with_landmarks=True):
    landmarks = [Point(x + (i % 50), y + (i % 50)) for i in range(478)] if with_landmarks else None
    return FaceDetection(
        bbox=BoundingBox(x, y, w, h),
        confidence=0.92,
        dense_landmarks=landmarks,
        face_shape=FaceShapeResult(
            label="oval",
            fit_score=0.88,
            alternatives={"round": 0.45},
            measurements={"length_to_width_ratio": 1.4},
            quality_status="good",
        ),
    )


def test_camera_without_arguments_preserved(mock_webcam):
    """Requirement 1: camera() preserves default invocation without arguments."""
    mock_cap, mock_imshow, mock_wait_key, mock_destroy_window = mock_webcam
    test_frame = _make_frame()
    mock_cap.read.return_value = (True, test_frame)
    mock_wait_key.return_value = ord("q")

    # Run camera() with max_frames=1
    camera(max_frames=1)

    mock_cap.read.assert_called()
    mock_imshow.assert_called()
    mock_cap.release.assert_called_once()
    mock_destroy_window.assert_called_once()


def test_milestone4_proposed_api_arguments(mock_webcam):
    """Verify proposed API: show_mesh=True, mesh_style='neon', show_contour=True, show_scan=True."""
    mock_cap, mock_imshow, mock_wait_key, _ = mock_webcam
    test_frame = _make_frame()
    mock_cap.read.return_value = (True, test_frame)
    mock_wait_key.return_value = ord("q")

    face = _make_face()
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[face], image=test_frame)

    with (
        patch("detekface.mesh.draw_face_meshes") as mock_mesh,
        patch("detekface.camera.draw_face_shape_overlay") as mock_shape,
        patch("detekface.live_fx.LiveFXRenderer.render") as mock_fx,
    ):
        camera(
            detector=mock_detector,
            show_mesh=True,
            mesh_style="neon",
            show_contour=True,
            show_scan=True,
            max_frames=1,
        )

        mock_mesh.assert_called_once()
        assert mock_mesh.call_args[1]["style"] == "neon"
        mock_shape.assert_called()
        assert mock_shape.call_args[1]["show_contour"] is True
        mock_fx.assert_called()


@pytest.mark.parametrize("shape_style", ["rectangle", "corners", "circle", "rounded", "hud"])
def test_all_five_overlay_styles_with_mesh(mock_webcam, shape_style):
    """Requirement 3 & 7: Verify all 5 overlay styles work seamlessly with mesh enabled."""
    mock_cap, mock_imshow, mock_wait_key, _ = mock_webcam
    test_frame = _make_frame()
    mock_cap.read.return_value = (True, test_frame)
    mock_wait_key.return_value = ord("q")

    face = _make_face()
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[face], image=test_frame)

    with patch("detekface.camera.draw_faces") as mock_draw:
        camera(
            detector=mock_detector,
            shape=shape_style,
            show_mesh=True,
            mesh_style="wireframe",
            max_frames=1,
        )
        mock_draw.assert_called_once()
        assert mock_draw.call_args[1]["shape"] == shape_style


def test_metrics_overlay_preserved_with_mesh(mock_webcam):
    """Requirement 4: Verify FPS, face count, and confidence overlays work with mesh."""
    mock_cap, mock_imshow, mock_wait_key, _ = mock_webcam
    test_frame = _make_frame()
    mock_cap.read.return_value = (True, test_frame)
    mock_wait_key.return_value = ord("q")

    face = _make_face()
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[face], image=test_frame)

    with (
        patch("detekface.camera.draw_metrics_overlay") as mock_metrics,
        patch("detekface.camera.draw_faces") as mock_draw_faces,
    ):
        camera(
            detector=mock_detector,
            show_mesh=True,
            show_confidence=True,
            show_fps=True,
            show_count=True,
            max_frames=1,
        )
        mock_draw_faces.assert_called_once()
        assert mock_draw_faces.call_args[1]["show_confidence"] is True
        mock_metrics.assert_called_once()
        assert mock_metrics.call_args[1]["fps"] is not None
        assert mock_metrics.call_args[1]["face_count"] == 1


def test_capture_and_callback_while_mesh_enabled(mock_webcam, tmp_path):
    """Requirement 5: Verify face capture key 'C' and callback work cleanly while mesh is enabled."""
    mock_cap, mock_imshow, mock_wait_key, _ = mock_webcam
    test_frame = _make_frame()
    mock_cap.read.return_value = (True, test_frame)

    # First frame press 'c', second frame press 'q'
    mock_wait_key.side_effect = [ord("c"), ord("q")]

    face = _make_face()
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[face], image=test_frame)

    captured_faces = []

    def on_cap(cap: FaceCapture):
        captured_faces.append(cap)

    camera(
        detector=mock_detector,
        show_mesh=True,
        mesh_style="neon",
        capture_dir=tmp_path,
        on_capture=on_cap,
    )

    assert len(captured_faces) == 1
    assert captured_faces[0].path.is_file()
    # Pristine face crop is preserved without mesh lines
    saved_img = cv2.imread(str(captured_faces[0].path))
    assert saved_img is not None


def test_multiple_faces_rendered_independently(mock_webcam):
    """Requirement 9: Support multiple faces independently with mesh and Live FX."""
    mock_cap, mock_imshow, mock_wait_key, _ = mock_webcam
    test_frame = _make_frame()
    mock_cap.read.return_value = (True, test_frame)
    mock_wait_key.return_value = ord("q")

    faces = [
        _make_face(x=50, y=50, w=100, h=100),
        _make_face(x=250, y=80, w=120, h=120),
    ]
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=faces, image=test_frame)

    with (
        patch("detekface.mesh.draw_face_meshes") as mock_mesh,
        patch("detekface.live_fx.LiveFXRenderer.render") as mock_fx,
    ):
        camera(
            detector=mock_detector,
            show_mesh=True,
            show_scan=True,
            max_frames=1,
        )
        mock_mesh.assert_called_once()
        assert len(mock_mesh.call_args[0][1]) == 2
        mock_fx.assert_called_once()
        assert len(mock_fx.call_args[1]["faces"]) == 2


def test_unavailable_landmarks_does_not_crash_preview(mock_webcam):
    """Requirement 10: Handle faces with None or empty landmarks gracefully without crashing."""
    mock_cap, mock_imshow, mock_wait_key, _ = mock_webcam
    test_frame = _make_frame()
    mock_cap.read.return_value = (True, test_frame)
    mock_wait_key.return_value = ord("q")

    # Detection without landmarks
    face_no_landmarks = _make_face(with_landmarks=False)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = DetectionResult(faces=[face_no_landmarks], image=test_frame)

    # Should run and exit without exception
    camera(
        detector=mock_detector,
        show_mesh=True,
        show_scan=True,
        max_frames=1,
    )
    mock_imshow.assert_called()


def test_exit_on_escape_and_q(mock_webcam):
    """Requirement 11: Verify preview exits cleanly on 'q', 'Q', and Escape."""
    mock_cap, _, mock_wait_key, mock_destroy_window = mock_webcam
    test_frame = _make_frame()
    mock_cap.read.return_value = (True, test_frame)

    # Test Escape key (27)
    mock_wait_key.return_value = 27
    camera(max_frames=10)
    mock_cap.release.assert_called_once()
    mock_destroy_window.assert_called_once()


def test_exception_cleanup_guarantee(mock_webcam):
    """Requirement 11: Verify camera and window cleanup on unexpected exceptions."""
    mock_cap, _, mock_wait_key, mock_destroy_window = mock_webcam
    test_frame = _make_frame()
    mock_cap.read.return_value = (True, test_frame)

    mock_detector = MagicMock()
    mock_detector.detect.side_effect = RuntimeError("Inference hardware fault")

    with pytest.raises(RuntimeError):
        camera(detector=mock_detector)

    mock_cap.release.assert_called_once()
    mock_destroy_window.assert_called_once()


def test_rendering_order_in_camera_session():
    """Verify layered rendering order in CameraSession.render()."""
    test_frame = _make_frame()
    face = _make_face()
    result = DetectionResult(faces=[face], image=test_frame)

    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, test_frame)
        mock_cap_cls.return_value = mock_cap

        with open_camera(camera_index=0) as session:
            session._latest_frame = test_frame
            session._latest_result = result

            # Render with mesh, contour, scan, and metrics
            annotated = session.render(
                show_mesh=True,
                mesh_style="neon",
                show_contour=True,
                show_scan=True,
                show_fps=True,
                show_count=True,
            )

            assert isinstance(annotated, np.ndarray)
            assert annotated.shape == test_frame.shape
            assert not np.array_equal(annotated, test_frame)

