"""Tests for detekface command-line interface (CLI)."""

import io
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from detekface import BoundingBox, DetectionResult, FaceDetection
from detekface.cli import build_parser, main


@pytest.fixture
def sample_test_image(tmp_path: Path) -> Path:
    """Create a temporary test image file."""
    img_path = tmp_path / "test_face.jpg"
    cv2.imwrite(str(img_path), np.full((120, 120, 3), 150, dtype=np.uint8))
    return img_path


def test_cli_help_output(capsys):
    """Verify detekface --help outputs usage, commands, and examples with exit code 0."""
    code = main(["--help"])
    assert code == 0
    captured = capsys.readouterr()
    assert "DetekFace - Lightweight, high-performance face detection CLI." in captured.out
    assert "camera" in captured.out
    assert "image" in captured.out
    assert "batch" in captured.out
    assert "video" in captured.out


def test_cli_no_args_prints_help(capsys):
    """Verify running detekface without arguments displays help and exits cleanly."""
    code = main([])
    assert code == 0
    captured = capsys.readouterr()
    assert "usage: detekface" in captured.out


def test_cli_version(capsys):
    """Verify detekface --version outputs current version."""
    code = main(["--version"])
    assert code == 0
    captured = capsys.readouterr()
    assert "detekface 0.1.0" in captured.out


def test_cli_subcommand_helps(capsys):
    """Verify --help works for camera, image, batch, and video subcommands."""
    for cmd in ["camera", "image", "batch", "video"]:
        code = main([cmd, "--help"])
        assert code == 0
        captured = capsys.readouterr()
        assert f"usage: detekface {cmd}" in captured.out


def test_cli_invalid_subcommand_exits_nonzero(capsys):
    """Verify unknown subcommands exit with code 2."""
    code = main(["unknown_command"])
    assert code == 2


def test_cli_camera_mocked():
    """Verify detekface camera invokes detekface.camera without physical webcam."""
    with patch("detekface.camera") as mock_cam:
        code = main(["camera", "--device", "2", "--backend", "yunet", "--shape", "hud", "--show-fps"])
        assert code == 0
        mock_cam.assert_called_once()
        _, kwargs = mock_cam.call_args
        assert kwargs["camera_index"] == 2
        assert kwargs["backend"] == "yunet"
        assert kwargs["shape"] == "hud"
        assert kwargs["show_fps"] is True


def test_cli_image_detection_and_output(sample_test_image: Path, tmp_path: Path, capsys):
    """Verify detekface image detects faces and exports annotated output."""
    out_file = tmp_path / "output_annotated.jpg"

    face = FaceDetection(bbox=BoundingBox(10, 10, 40, 40), confidence=0.95)
    mock_res = DetectionResult(faces=[face], image=np.zeros((100, 100, 3), dtype=np.uint8))

    with patch("detekface.detect", return_value=mock_res):
        code = main(["image", str(sample_test_image), "--output", str(out_file), "--shape", "rounded"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Detected 1 face(s)" in captured.out
        assert "Conf=95.0%" in captured.out
        assert out_file.exists()


def test_cli_image_overwrite_protection(sample_test_image: Path, tmp_path: Path, capsys):
    """Verify CLI prevents overwriting existing files unless --overwrite is specified."""
    existing_out = tmp_path / "already_exists.jpg"
    existing_out.write_text("existing content")

    # Attempting to write to existing file without --overwrite should fail
    code = main(["image", str(sample_test_image), "--output", str(existing_out)])
    assert code == 1
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert "Pass --overwrite" in captured.err

    # With --overwrite, it should succeed
    face = FaceDetection(bbox=BoundingBox(10, 10, 20, 20), confidence=0.9)
    mock_res = DetectionResult(faces=[face], image=np.zeros((100, 100, 3), dtype=np.uint8))
    with patch("detekface.detect", return_value=mock_res):
        code_overwrite = main(["image", str(sample_test_image), "--output", str(existing_out), "--overwrite"])
        assert code_overwrite == 0


def test_cli_image_missing_input_file(tmp_path: Path, capsys):
    """Verify non-existent input files report clear error and nonzero exit code."""
    missing = tmp_path / "non_existent.jpg"
    code = main(["image", str(missing)])
    assert code == 1
    captured = capsys.readouterr()
    assert "Input image does not exist" in captured.err


def test_cli_image_face_crop(sample_test_image: Path, tmp_path: Path, capsys):
    """Verify --crop-face exports cropped face image."""
    crop_file = tmp_path / "crop.jpg"
    face = FaceDetection(bbox=BoundingBox(10, 10, 30, 30), confidence=0.92)
    mock_res = DetectionResult(faces=[face], image=np.zeros((100, 100, 3), dtype=np.uint8))

    with patch("detekface.detect", return_value=mock_res):
        code = main(["image", str(sample_test_image), "--crop-face", str(crop_file)])
        assert code == 0
        captured = capsys.readouterr()
        assert f"Saved face crop to: '{crop_file}'" in captured.out
        assert crop_file.exists()


def test_cli_batch_processing(tmp_path: Path, capsys):
    """Verify detekface batch processes directory and displays summary statistics."""
    batch_dir = tmp_path / "batch_in"
    batch_dir.mkdir()
    img1 = batch_dir / "img1.jpg"
    img2 = batch_dir / "img2.jpg"
    cv2.imwrite(str(img1), np.zeros((50, 50, 3), dtype=np.uint8))
    cv2.imwrite(str(img2), np.zeros((50, 50, 3), dtype=np.uint8))

    out_dir = tmp_path / "batch_out"

    code = main(["batch", str(batch_dir), "--output", str(out_dir), "--recursive"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Batch Summary:" in captured.out
    assert "Total Processed : 2" in captured.out


def test_cli_video_mocked(tmp_path: Path):
    """Verify detekface video command invokes detekface.video with correct arguments."""
    vid_file = tmp_path / "test_vid.mp4"
    vid_file.write_bytes(b"dummy video data")

    with patch("detekface.video", return_value=120) as mock_vid:
        code = main(["video", str(vid_file), "--no-preview", "--output", "annotated.mp4"])
        assert code == 0
        mock_vid.assert_called_once()
        _, kwargs = mock_vid.call_args
        assert kwargs["source"] == vid_file
        assert kwargs["output"] == "annotated.mp4"
        assert kwargs["show"] is False
