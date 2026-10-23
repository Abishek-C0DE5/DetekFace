"""Unit tests for detekface model resolution, caching, checksum verification, and error handling.

All tests are designed to run fully offline without external network dependencies.
"""

import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error

import cv2
import pytest

from detekface.exceptions import (
    ModelCompatibilityError,
    ModelCorruptionError,
    ModelDownloadError,
    ModelNotFoundError,
)
from detekface.models import (
    OFFICIAL_MODELS,
    compute_file_sha256,
    download_model,
    get_cache_dir,
    resolve_model,
    select_default_model,
    verify_model_initialization,
)


def test_official_model_metadata_completeness() -> None:
    """Ensure all official models record exact model name, filename, sha256, URLs, license, and author."""
    assert "yunet_2026may" in OFFICIAL_MODELS
    assert "yunet_2023mar" in OFFICIAL_MODELS

    for key, model in OFFICIAL_MODELS.items():
        assert model.name == key
        assert model.filename.endswith(".onnx")
        assert len(model.sha256) == 64
        assert model.size_bytes > 0
        assert model.license == "MIT"
        assert "Shiqi Yu" in model.author
        assert len(model.urls) >= 2
        for url in model.urls:
            assert url.startswith("https://")


def test_select_default_model_by_opencv_version() -> None:
    """Verify that OpenCV 5.x selects yunet_2026may and OpenCV 4.x selects yunet_2023mar."""
    model_v5 = select_default_model("5.0.0")
    assert model_v5.name == "yunet_2026may"

    model_v4 = select_default_model("4.8.0")
    assert model_v4.name == "yunet_2023mar"

    # Testing currently installed runtime
    model_current = select_default_model()
    expected_key = "yunet_2026may" if cv2.__version__.startswith("5.") else "yunet_2023mar"
    assert model_current.name == expected_key


def test_resolve_user_supplied_local_path(tmp_path: Path) -> None:
    """Verify that a valid user-supplied local file path is returned directly."""
    dummy_model = tmp_path / "custom_model.onnx"
    dummy_model.write_bytes(b"dummy onnx weights content")

    resolved = resolve_model(dummy_model)
    assert resolved == dummy_model.resolve()

    resolved_str = resolve_model(str(dummy_model))
    assert resolved_str == dummy_model.resolve()


def test_resolve_empty_user_supplied_file(tmp_path: Path) -> None:
    """Verify that an empty local file raises ModelCorruptionError with a helpful hint."""
    empty_model = tmp_path / "empty.onnx"
    empty_model.write_bytes(b"")

    with pytest.raises(ModelCorruptionError) as exc_info:
        resolve_model(empty_model)
    assert "is empty" in str(exc_info.value)
    assert exc_info.value.hint is not None


def test_resolve_nonexistent_local_path() -> None:
    """Verify that a non-existent explicit path raises ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError) as exc_info:
        resolve_model("nonexistent/path/to/model.onnx")
    assert "does not exist" in str(exc_info.value)
    assert exc_info.value.hint is not None


def test_resolve_unknown_model_name() -> None:
    """Verify that an unrecognized model alias raises ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError) as exc_info:
        resolve_model("nonexistent_model_alias")
    assert "Unknown model identifier" in str(exc_info.value)


def test_cache_reuse_without_download(tmp_path: Path) -> None:
    """Verify that when a valid model is already in cache, no network requests are made."""
    metadata = OFFICIAL_MODELS["yunet_2026may"]
    cached_file = tmp_path / metadata.filename
    dummy_content = b"valid pre-existing model content"
    cached_file.write_bytes(dummy_content)

    dummy_sha256 = compute_file_sha256(cached_file)

    # Patch metadata sha256 to match our dummy cached file
    with patch.dict(OFFICIAL_MODELS, {
        "yunet_2026may": MagicMock(
            name=metadata.name,
            filename=metadata.filename,
            sha256=dummy_sha256,
            urls=metadata.urls,
        )
    }):
        with patch("urllib.request.urlopen") as mock_urlopen:
            result = resolve_model("yunet_2026may", cache_dir=tmp_path)
            assert result == cached_file
            mock_urlopen.assert_not_called()


def test_cached_file_corruption_detection(tmp_path: Path) -> None:
    """Verify that a cached file with an invalid checksum raises ModelCorruptionError and is removed."""
    metadata = OFFICIAL_MODELS["yunet_2026may"]
    corrupt_file = tmp_path / metadata.filename
    corrupt_file.write_bytes(b"corrupted cached bytes")

    with pytest.raises(ModelCorruptionError) as exc_info:
        resolve_model("yunet_2026may", cache_dir=tmp_path, verify_checksum=True)

    assert "integrity validation" in str(exc_info.value)
    # Ensure the corrupt file was purged
    assert not corrupt_file.exists()


def test_mock_download_success(tmp_path: Path) -> None:
    """Verify that download succeeds and validates checksum when remote returns valid data."""
    metadata = OFFICIAL_MODELS["yunet_2026may"]
    dummy_payload = b"test onnx stream content"
    payload_sha256 = hashlib.sha256(dummy_payload).hexdigest()

    target_file = tmp_path / metadata.filename

    mock_response = MagicMock()
    mock_response.read.side_effect = [dummy_payload, b""]
    mock_response.__enter__.return_value = mock_response

    mock_meta = MagicMock(
        name=metadata.name,
        filename=metadata.filename,
        sha256=payload_sha256,
        urls=("https://mock.trusted.org/model.onnx",),
    )

    with patch("urllib.request.urlopen", return_value=mock_response):
        downloaded = download_model(mock_meta, target_file)
        assert downloaded == target_file
        assert target_file.is_file()
        assert target_file.read_bytes() == dummy_payload


def test_mock_download_checksum_failure(tmp_path: Path) -> None:
    """Verify that a download returning mismatched checksum raises ModelCorruptionError and deletes temp files."""
    metadata = OFFICIAL_MODELS["yunet_2026may"]
    dummy_payload = b"bad corrupted content"

    target_file = tmp_path / metadata.filename

    mock_response = MagicMock()
    mock_response.read.side_effect = [dummy_payload, b""]
    mock_response.__enter__.return_value = mock_response

    mock_meta = MagicMock(
        name=metadata.name,
        filename=metadata.filename,
        sha256="0000000000000000000000000000000000000000000000000000000000000000",
        urls=("https://mock.trusted.org/model.onnx",),
    )

    with patch("urllib.request.urlopen", return_value=mock_response):
        with pytest.raises(ModelCorruptionError) as exc_info:
            download_model(mock_meta, target_file)
        assert "Checksum validation failed" in str(exc_info.value)
        assert not target_file.exists()


def test_mock_download_network_failure(tmp_path: Path) -> None:
    """Verify that network failures on all URLs raise ModelDownloadError with helpful instructions."""
    target_file = tmp_path / "model.onnx"

    mock_meta = MagicMock(
        name="test_model",
        filename="model.onnx",
        sha256="1234",
        urls=("https://url1.com/m.onnx", "https://url2.com/m.onnx"),
    )

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network unreachable")):
        with pytest.raises(ModelDownloadError) as exc_info:
            download_model(mock_meta, target_file)

        assert "Failed to download model" in str(exc_info.value)
        assert "check your internet connection" in str(exc_info.value.hint)


def test_verify_model_initialization_invalid_file(tmp_path: Path) -> None:
    """Verify that an invalid/non-ONNX model file passed to initialization raises ModelCompatibilityError."""
    fake_model = tmp_path / "corrupt.onnx"
    fake_model.write_bytes(b"This is definitely not an ONNX model")

    with pytest.raises(ModelCompatibilityError) as exc_info:
        verify_model_initialization(fake_model)

    assert "OpenCV failed to initialize FaceDetectorYN" in str(exc_info.value)
    assert exc_info.value.hint is not None


def test_verify_model_initialization_nonexistent_file() -> None:
    """Verify that verifying a non-existent file raises ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError):
        verify_model_initialization("does_not_exist.onnx")


def test_real_model_initialization_if_cached() -> None:
    """If an official model is already cached locally, verify real OpenCV FaceDetectorYN initialization offline."""
    meta = select_default_model()
    cache_path = get_cache_dir() / meta.filename
    if cache_path.is_file():
        verify_model_initialization(cache_path)


def test_get_cache_dir_custom_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that DETEKFACE_CACHE_DIR environment variable is respected."""
    custom_dir = tmp_path / "custom_cache"
    monkeypatch.setenv("DETEKFACE_CACHE_DIR", str(custom_dir))
    cache_dir = get_cache_dir()
    assert cache_dir == custom_dir
    assert custom_dir.is_dir()
