"""Automated tests for Milestone 9: prerecorded video file face detection.

Tests detekface.video() processing, output video generation, dimension & framerate preservation,
frame count verification, cleanup, error handling, and capture functionality.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

import detekface
from detekface.detector import FaceDetector
from detekface.exceptions import (
    ConfigurationError,
    InvalidVideoError,
    VideoNotFoundError,
    VideoWriteError,
)
from detekface.types import BoundingBox, DetectionResult, FaceDetection


def generate_synthetic_video(
    path: Path,
    num_frames: int = 15,
    width: int = 320,
    height: int = 240,
    fps: float = 20.0,
) -> Path:
    """Helper to create a short, valid synthetic video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), 200, dtype=np.uint8)
            # Draw synthetic face-like dark rectangle
            frame[60:160, 100:200] = (80, 80, 80)
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture
def synthetic_video_path(tmp_path: Path) -> Path:
    vid_file = tmp_path / "test_clip.mp4"
    return generate_synthetic_video(vid_file, num_frames=12, width=320, height=240, fps=20.0)


# -------------------------------------------------------------------------
# Check 1 & Check 3: Video Processing, Dimensions, Output, Cleanup
# -------------------------------------------------------------------------


def test_video_headless_processing(synthetic_video_path):
    """Test detekface.video() processes a video file end-to-end headlessly."""
    result = detekface.video(synthetic_video_path, show=False)
    assert result is None


def test_video_annotated_output_creation_and_dimensions(synthetic_video_path, tmp_path):
    """Test generating annotated output video, verifying dimensions, frame count, and cleanup."""
    out_video = tmp_path / "annotated_out.mp4"

    saved_path = detekface.video(
        path=synthetic_video_path,
        output_path=out_video,
        show=False,
        shape="hud",
        color="#00FF88",
        thickness=2,
        show_confidence=True,
        show_fps=True,
        show_count=True,
    )

    assert saved_path == out_video.resolve()
    assert out_video.exists()
    assert out_video.stat().st_size > 0

    # Reopen saved output video and inspect properties
    cap = cv2.VideoCapture(str(out_video))
    try:
        assert cap.isOpened()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps = cap.get(cv2.CAP_PROP_FPS)

        assert width == 320
        assert height == 240
        assert abs(out_fps - 20.0) < 1.5

        # Count frames
        read_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            read_frames += 1

        assert read_frames == 12
    finally:
        cap.release()


def test_video_max_frames_limiting(synthetic_video_path, tmp_path):
    """Verify max_frames limits the number of processed and written frames."""
    out_video = tmp_path / "limited_frames.mp4"

    detekface.video(
        path=synthetic_video_path,
        output_path=out_video,
        show=False,
        max_frames=5,
    )

    assert out_video.exists()
    cap = cv2.VideoCapture(str(out_video))
    try:
        frames = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames += 1
        assert frames == 5
    finally:
        cap.release()


def test_video_creates_nested_output_directory(synthetic_video_path, tmp_path):
    """Verify video creates parent directories for output video."""
    nested_out = tmp_path / "deep" / "nested" / "dir" / "out.mp4"
    assert not nested_out.parent.exists()

    saved = detekface.video(synthetic_video_path, output_path=nested_out, show=False)
    assert saved.exists()
    assert nested_out.parent.exists()


# -------------------------------------------------------------------------
# Check 2: Invalid Paths, Read Failures, and Error Handling
# -------------------------------------------------------------------------


def test_nonexistent_video_path_raises_not_found():
    """Verify nonexistent path raises VideoNotFoundError."""
    with pytest.raises(VideoNotFoundError) as exc_info:
        detekface.video("nonexistent_video_path_12345.mp4", show=False)
    assert "not found" in str(exc_info.value).lower()


def test_directory_as_video_path_raises_invalid(tmp_path):
    """Verify passing a directory path raises InvalidVideoError."""
    with pytest.raises(InvalidVideoError) as exc_info:
        detekface.video(tmp_path, show=False)
    assert "directory" in str(exc_info.value).lower()


@pytest.mark.parametrize("invalid_path", ["", "   ", None, 1234])
def test_invalid_input_path_types_raise_invalid_video(invalid_path):
    """Verify empty or non-string/Path inputs raise InvalidVideoError."""
    with pytest.raises(InvalidVideoError):
        detekface.video(invalid_path, show=False)


def test_empty_zero_byte_video_file_raises_invalid(tmp_path):
    """Verify an empty (0-byte) video file raises InvalidVideoError."""
    empty_file = tmp_path / "empty.mp4"
    empty_file.write_bytes(b"")

    with pytest.raises(InvalidVideoError):
        detekface.video(empty_file, show=False)


def test_simulated_frame_read_failure_on_first_frame(synthetic_video_path):
    """Verify simulated read failure on frame 0 raises InvalidVideoError and cleans up."""
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cap_cls.return_value = mock_cap

        with pytest.raises(InvalidVideoError) as exc_info:
            detekface.video(synthetic_video_path, show=False)

        assert "Could not read any video frames" in str(exc_info.value)
        mock_cap.release.assert_called_once()


def test_simulated_read_exception_guarantees_cleanup(synthetic_video_path):
    """Verify unexpected exception during read triggers cap.release() in finally."""
    with patch("cv2.VideoCapture") as mock_cap_cls:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = RuntimeError("Hardware I/O error")
        mock_cap_cls.return_value = mock_cap

        with pytest.raises(RuntimeError):
            detekface.video(synthetic_video_path, show=False)

        mock_cap.release.assert_called_once()


# -------------------------------------------------------------------------
# Output Path & Codec Validation
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_ext",
    [
        "out_no_extension",
        "out.txt",
        "out.jpg",
        "out.pdf",
        "out.xyz",
    ],
)
def test_unsupported_output_video_extensions_raise_error(synthetic_video_path, tmp_path, bad_ext):
    """Verify unsupported video output extensions raise InvalidVideoError."""
    bad_out = tmp_path / bad_ext
    with pytest.raises(InvalidVideoError) as exc_info:
        detekface.video(synthetic_video_path, output_path=bad_out, show=False)
    assert "Unsupported" in str(exc_info.value) or "extension" in str(exc_info.value)


def test_output_parent_is_file_raises_video_write_error(synthetic_video_path, tmp_path):
    """Verify saving to a path where parent is a file raises VideoWriteError."""
    dummy_file = tmp_path / "blocker.txt"
    dummy_file.write_text("file")

    bad_out = dummy_file / "out.mp4"
    with pytest.raises(VideoWriteError):
        detekface.video(synthetic_video_path, output_path=bad_out, show=False)


def test_invalid_codec_string_raises_configuration_error(synthetic_video_path, tmp_path):
    """Verify non-4-character codec raises ConfigurationError."""
    bad_out = tmp_path / "out.mp4"
    with pytest.raises(ConfigurationError):
        detekface.video(synthetic_video_path, output_path=bad_out, codec="TOO_LONG", show=False)


def test_video_writer_open_failure_raises_video_write_error(synthetic_video_path, tmp_path):
    """Verify VideoWriter failing to open raises VideoWriteError."""
    bad_out = tmp_path / "out.mp4"
    with patch("cv2.VideoWriter") as mock_writer_cls:
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = False
        mock_writer_cls.return_value = mock_writer

        with pytest.raises(VideoWriteError) as exc_info:
            detekface.video(synthetic_video_path, output_path=bad_out, show=False)

        assert "OpenCV failed to open VideoWriter" in str(exc_info.value)


# -------------------------------------------------------------------------
# GUI, Keyboard Controls, and Cleanup
# -------------------------------------------------------------------------


def test_video_gui_early_exit_on_q_key(synthetic_video_path):
    """Verify pressing 'q' breaks loop and destroys OpenCV windows."""
    with (
        patch("cv2.imshow") as mock_imshow,
        patch("cv2.waitKey", return_value=ord("q")),
        patch("cv2.destroyAllWindows") as mock_destroy,
    ):
        detekface.video(synthetic_video_path, show=True)
        mock_imshow.assert_called_once()
        mock_destroy.assert_called_once()


def test_video_gui_window_close_x_button(synthetic_video_path):
    """Verify closing GUI window via 'X' button stops playback cleanly."""
    with (
        patch("cv2.imshow"),
        patch("cv2.waitKey", return_value=-1),
        patch("cv2.getWindowProperty", return_value=0),  # Window closed
        patch("cv2.destroyAllWindows") as mock_destroy,
    ):
        detekface.video(synthetic_video_path, show=True)
        mock_destroy.assert_called_once()


def test_video_interactive_capture_key(synthetic_video_path, tmp_path):
    """Verify pressing 'c' during video playback captures face cleanly and triggers callback."""
    fake_face = FaceDetection(
        bbox=BoundingBox(x=100, y=60, width=100, height=100),
        confidence=0.91,
    )
    fake_result = DetectionResult(faces=[fake_face], image=np.full((240, 320, 3), 80, dtype=np.uint8))

    callback_capture = []

    def on_capture_callback(face):
        callback_capture.append(face)

    with (
        patch.object(FaceDetector, "detect", return_value=fake_result),
        patch("cv2.imshow"),
        patch("cv2.waitKey", side_effect=[ord("c"), ord("q")]),
        patch("cv2.destroyAllWindows"),
    ):
        detekface.video(
            synthetic_video_path,
            show=True,
            capture_dir=tmp_path,
            on_capture=on_capture_callback,
        )

        assert len(callback_capture) == 1
        cap = callback_capture[0]
        assert cap.width == 100
        assert cap.height == 100
        assert cap.path.exists()


# -------------------------------------------------------------------------
# Display Flags Validation
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag_name, bad_value",
    [
        ("show", "true"),
        ("show_confidence", 1),
        ("show_fps", None),
        ("show_count", "yes"),
    ],
)
def test_display_flags_validation(synthetic_video_path, flag_name, bad_value):
    """Verify non-boolean display flags raise ConfigurationError."""
    kwargs = {flag_name: bad_value}
    with pytest.raises(ConfigurationError):
        detekface.video(synthetic_video_path, **kwargs)


def test_video_source_and_output_keyword_aliases(synthetic_video_path, tmp_path):
    """Verify that source and output keyword aliases work properly."""
    out_file = tmp_path / "alias_out.mp4"
    res = detekface.video(source=synthetic_video_path, output=out_file, show=False, max_frames=2)
    assert res is not None
    assert res.is_file()
