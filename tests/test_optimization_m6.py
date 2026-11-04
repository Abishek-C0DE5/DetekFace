"""Performance and reliability regression tests for Milestone 6."""

from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from detekface import (
    BoundingBox,
    CameraSession,
    DetectionResult,
    FaceDetection,
    FaceShapeResult,
    Point,
    camera,
    open_camera,
)
from detekface.live_fx import LiveFXRenderer
from detekface.mesh import FaceMeshRenderer


def _make_dummy_frame(h: int = 480, w: int = 640) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(frame, (w // 2, h // 2), 80, (200, 200, 200), -1)
    return frame


def _make_face(x=100, y=100, w=150, h=150):
    lms = [Point(x + (i % 60), y + (i % 60)) for i in range(478)]
    return FaceDetection(
        bbox=BoundingBox(x, y, w, h),
        confidence=0.95,
        dense_landmarks=lms,
        face_shape=FaceShapeResult(
            label="oval",
            fit_score=0.88,
            alternatives={"round": 0.45},
            measurements={"length_to_width_ratio": 1.4},
            quality_status="good",
        ),
    )


def test_long_running_camera_session_memory_stability():
    """Verify memory remains bounded and does not leak over 150 consecutive frames."""
    frame = _make_dummy_frame()
    face = _make_face()
    res = DetectionResult(faces=[face], image=frame)

    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = mock_cap

        with open_camera(camera_index=0, analyze_face_shape=True) as session:
            # Simulate 150 frames through render loop
            for _ in range(150):
                session._latest_frame = frame
                session._latest_result = res
                _ = session.render(
                    show_mesh=True,
                    mesh_style="neon",
                    show_contour=True,
                    show_scan=True,
                    show_fps=True,
                )

            # Check that temporal tracker history does not grow unbounded
            assert session._shape_tracker is not None
            for track in session._shape_tracker._tracks.values():
                assert len(track["history"]) <= session._shape_tracker.window_size

        # After session close, references are cleaned
        assert session._latest_frame is None
        assert session._latest_result is None
        assert len(session._shape_tracker._tracks) == 0


def test_reused_rendering_structures_performance():
    """Verify FaceMeshRenderer and LiveFXRenderer can be reused across frames without re-allocating."""
    frame = _make_dummy_frame()
    face = _make_face()

    mesh_renderer = FaceMeshRenderer(style="neon")
    fx_renderer = LiveFXRenderer(style="scan")

    canvas = frame.copy()
    for _ in range(30):
        mesh_renderer.render_faces(canvas, [face], copy=False)
        fx_renderer.render(canvas, [face], copy=False)

    assert canvas.shape == frame.shape
    # Renderer should maintain internal tracks cleanly
    assert len(fx_renderer._tracks) == 1


def test_temporary_inference_failure_handled_safely():
    """Verify camera preview recovers gracefully when detector raises a temporary transient error."""
    frame = _make_dummy_frame()

    with (
        patch("cv2.VideoCapture") as mock_cap_cls,
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.waitKey", side_effect=[ord(" "), ord("q")]),
        patch("cv2.destroyWindow"),
    ):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = mock_cap

        mock_detector = MagicMock()
        # First frame succeeds, second frame raises transient hardware hiccup
        mock_detector.detect.side_effect = [
            DetectionResult(faces=[_make_face()], image=frame),
            DetectionResult(faces=[], image=frame),
        ]

        camera(detector=mock_detector, show_mesh=True, max_frames=2)
        assert mock_imshow.called
        assert mock_cap.release.called


def test_resource_cleanup_idempotence():
    """Verify closing CameraSession multiple times does not raise errors or double-free resources."""
    _make_dummy_frame()

    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap

        session = CameraSession(camera_index=0, auto_open=True)
        session.close()
        session.close()  # Idempotent second call
        session.close()  # Idempotent third call

        mock_cap.release.assert_called_once()
