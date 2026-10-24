"""Model management, caching, verification, and resolution for detekface.

Handles safe resolution of YuNet ONNX face detection models with automatic cache
management, integrity verification, and OpenCV compatibility detection.
"""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, Union
import urllib.error
import urllib.request

import cv2

from detekface.exceptions import (
    ModelCompatibilityError,
    ModelCorruptionError,
    ModelDownloadError,
    ModelNotFoundError,
)


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata describing an official face detection model."""

    name: str
    filename: str
    sha256: str
    size_bytes: int
    min_opencv_version: str
    license: str
    author: str
    description: str
    urls: Tuple[str, ...]


OFFICIAL_MODELS: Dict[str, ModelMetadata] = {
    "yunet_2026may": ModelMetadata(
        name="yunet_2026may",
        filename="face_detection_yunet_2026may.onnx",
        sha256="ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
        size_bytes=229738,
        min_opencv_version="5.0.0",
        license="MIT",
        author="Shiqi Yu (libfacedetection) & OpenCV Zoo",
        description="YuNet model with dynamic input shape for OpenCV 5.x graph engine.",
        urls=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx",
            "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx",
        ),
    ),
    "yunet_2023mar": ModelMetadata(
        name="yunet_2023mar",
        filename="face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        size_bytes=232589,
        min_opencv_version="4.5.4",
        license="MIT",
        author="Shiqi Yu (libfacedetection) & OpenCV Zoo",
        description="YuNet model with fixed input shape for OpenCV 4.x DNN engine.",
        urls=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ),
    ),
}

# Alias mapping for user convenience
MODEL_ALIASES: Dict[str, str] = {
    "yunet": "default",
    "yunet_2026may": "yunet_2026may",
    "yunet_2026": "yunet_2026may",
    "yunet_2023mar": "yunet_2023mar",
    "yunet_2023": "yunet_2023mar",
    "face_detection_yunet_2026may.onnx": "yunet_2026may",
    "face_detection_yunet_2023mar.onnx": "yunet_2023mar",
}


def get_cache_dir() -> Path:
    """Return the base directory used to cache downloaded models.

    Respects the DETEKFACE_CACHE_DIR environment variable if specified, otherwise
    defaults to standard OS-appropriate cache locations.
    """
    env_dir = os.environ.get("DETEKFACE_CACHE_DIR")
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
    elif sys.platform.startswith("win"):
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        path = base / "detekface" / "models"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
        path = base / "detekface" / "models"

    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute the SHA256 hexadecimal digest of a file.

    Reads in 64KB blocks to maintain low memory overhead.
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def select_default_model(opencv_version: Optional[str] = None) -> ModelMetadata:
    """Select the best-compatible YuNet model based on the installed OpenCV version.

    OpenCV 5.x utilizes the modern ONNX Runtime graph engine and prefers dynamic shape models
    (yunet_2026may). OpenCV 4.x operates reliably with the fixed shape variant (yunet_2023mar).
    """
    version = opencv_version if opencv_version is not None else cv2.__version__
    major_str = version.split(".")[0] if "." in version else version
    try:
        major = int(major_str)
    except ValueError:
        major = 4

    if major >= 5:
        return OFFICIAL_MODELS["yunet_2026may"]
    return OFFICIAL_MODELS["yunet_2023mar"]


def download_model(
    metadata: ModelMetadata,
    destination_path: Path,
    chunk_size: int = 65536,
    timeout: float = 30.0,
) -> Path:
    """Download an official model from trusted sources with integrity verification.

    Downloads to a temporary file before atomically moving to the destination path.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination_path.with_suffix(f"{destination_path.suffix}.tmp.{os.getpid()}")

    last_error: Optional[Exception] = None

    for url in metadata.urls:
        hasher = hashlib.sha256()
        total_downloaded = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "detekface/0.1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                with open(temp_path, "wb") as f:
                    while chunk := response.read(chunk_size):
                        f.write(chunk)
                        hasher.update(chunk)
                        total_downloaded += len(chunk)

            calculated_sha256 = hasher.hexdigest()
            if calculated_sha256.lower() != metadata.sha256.lower():
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                last_error = ModelCorruptionError(
                    message=f"Checksum validation failed for downloaded model {metadata.filename}.",
                    hint=(
                        f"Expected SHA256: {metadata.sha256}\n"
                        f"Calculated SHA256: {calculated_sha256}\n"
                        f"The downloaded file was corrupted or incomplete ({total_downloaded} bytes received)."
                    ),
                )
                continue

            # Atomic replace
            temp_path.replace(destination_path)
            return destination_path

        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            last_error = e
            continue

    # Exhausted all URLs without a successful download. Prefer surfacing a
    # checksum corruption (actionable: re-download) over a generic network error.
    if isinstance(last_error, ModelCorruptionError):
        raise last_error
    raise ModelDownloadError(
        message=f"Failed to download model '{metadata.name}' ({metadata.filename}) from all official sources.",
        hint=(
            f"Please check your internet connection or download '{metadata.filename}' manually from:\n"
            f"  {metadata.urls[0]}\n"
            f"and place it at: {destination_path} or pass the path directly to detekface."
        ),
    ) from last_error


def resolve_model(
    model_name_or_path: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    verify_checksum: bool = True,
) -> Path:
    """Resolve a valid local model file path.

    Can resolve from:
    1. A user-supplied local file path (Path or str pointing to an existing file).
    2. An official model alias (e.g. 'yunet_2026may', 'yunet_2023mar', 'default').
    3. Automatic selection if model_name_or_path is None.

    Downloaded models are stored in cache_dir and validated against official SHA256 checksums.
    """
    if model_name_or_path is not None:
        candidate_path = Path(model_name_or_path).expanduser()
        if candidate_path.is_file():
            # Validate that local file is readable and non-empty
            if candidate_path.stat().st_size == 0:
                raise ModelCorruptionError(
                    message=f"Model file '{candidate_path}' is empty (0 bytes).",
                    hint="Provide a valid ONNX model file.",
                )
            return candidate_path.resolve()

        # Check if user specified a non-existent explicit path
        str_name = str(model_name_or_path).strip()
        is_path_like = (
            any(sep in str_name for sep in ("/", "\\"))
            or str_name.endswith(".onnx")
            or candidate_path.exists()
        )
        normalized_key = str_name.lower()
        if is_path_like and normalized_key not in MODEL_ALIASES:
            raise ModelNotFoundError(
                message=f"Specified model file does not exist: {candidate_path}",
                hint="Check the filepath for typos or pass a recognized model name (e.g. 'default', 'yunet_2026may').",
            )

    # Resolve official model metadata
    if model_name_or_path is None or str(model_name_or_path).lower() in ("default", "yunet"):
        metadata = select_default_model()
    else:
        alias = MODEL_ALIASES.get(str(model_name_or_path).lower())
        if alias is None or alias not in OFFICIAL_MODELS:
            raise ModelNotFoundError(
                message=f"Unknown model identifier: '{model_name_or_path}'.",
                hint=f"Available models are: {list(OFFICIAL_MODELS.keys())} or pass a valid local .onnx filepath.",
            )
        metadata = OFFICIAL_MODELS[alias]

    base_cache = Path(cache_dir).expanduser().resolve() if cache_dir else get_cache_dir()
    base_cache.mkdir(parents=True, exist_ok=True)
    destination = base_cache / metadata.filename

    # If already cached, check integrity
    if destination.is_file() and not force_download:
        if destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
        elif verify_checksum:
            current_sha256 = compute_file_sha256(destination)
            if current_sha256.lower() != metadata.sha256.lower():
                destination.unlink(missing_ok=True)
                raise ModelCorruptionError(
                    message=f"Cached model '{metadata.filename}' failed integrity validation.",
                    hint="The corrupt cached file has been purged. Please rerun to download a fresh copy.",
                )
            return destination
        else:
            return destination

    # Download model
    return download_model(metadata, destination)


def verify_model_initialization(
    model_path: Union[str, Path],
    input_size: Tuple[int, int] = (320, 320),
) -> None:
    """Verify that a model file can be instantiated by OpenCV's FaceDetectorYN.

    Raises ModelCompatibilityError with actionable guidance if OpenCV fails to load the model.
    """
    path = Path(model_path).resolve()
    if not path.is_file():
        raise ModelNotFoundError(f"Cannot verify non-existent model file: {path}")

    try:
        detector = cv2.FaceDetectorYN.create(
            str(path),
            "",
            input_size,
            score_threshold=0.8,
            nms_threshold=0.3,
            top_k=5000,
        )
        if detector is None:
            raise ModelCompatibilityError(
                message=f"OpenCV returned None when initializing FaceDetectorYN with '{path.name}'.",
                hint="Verify that the ONNX model architecture matches YuNet specifications.",
            )
    except cv2.error as e:
        raise ModelCompatibilityError(
            message=f"OpenCV failed to initialize FaceDetectorYN with model '{path.name}': {e}",
            hint=(
                "Ensure your model matches your OpenCV version. OpenCV 5.x prefers dynamic-shape models "
                "(yunet_2026may.onnx), while OpenCV 4.x uses fixed-shape models (yunet_2023mar.onnx)."
            ),
        ) from e
