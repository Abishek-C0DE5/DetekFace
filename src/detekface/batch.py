"""Batch image face detection with error isolation, progress tracking, and input mapping."""

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union
import numpy as np

from detekface.types import DetectionResult, FaceDetection, SUPPORTED_IMAGE_EXTENSIONS


@dataclass
class BatchItemResult:
    """Detection result for an individual item within a batch.

    Attributes:
        source: The original input path (str or Path) or NumPy array.
        result: DetectionResult if face detection succeeded; None if an error occurred.
        error: Caught Exception instance if processing failed; None on success.
    """

    source: Union[str, Path, np.ndarray]
    result: Optional[DetectionResult] = None
    error: Optional[Exception] = None

    @property
    def is_success(self) -> bool:
        """Return True if image loading and face detection succeeded."""
        return self.error is None and self.result is not None

    @property
    def is_failure(self) -> bool:
        """Return True if an exception occurred while processing this item."""
        return self.error is not None

    @property
    def faces(self) -> List[FaceDetection]:
        """List of detected faces if successful, or an empty list if failed."""
        if self.result is not None:
            return list(self.result.faces)
        return []

    @property
    def count(self) -> int:
        """Number of detected faces if successful, or 0 if failed."""
        if self.result is not None:
            return self.result.count
        return 0

    @property
    def has_faces(self) -> bool:
        """Return True if at least one face was detected."""
        return self.count > 0


class BatchDetectionResult(list):
    """List subclass holding BatchItemResult instances with batch-level conveniences.

    Preserves the exact ordering of input images while providing quick access to
    successes, failures, and aggregate face counts.
    """

    @property
    def successes(self) -> List[BatchItemResult]:
        """Return all successfully processed batch items."""
        return [item for item in self if item.is_success]

    @property
    def failures(self) -> List[BatchItemResult]:
        """Return all failed batch items."""
        return [item for item in self if item.is_failure]

    @property
    def total_faces(self) -> int:
        """Total number of faces detected across all successful batch items."""
        return sum(item.count for item in self.successes)

    @property
    def has_failures(self) -> bool:
        """Return True if any batch item failed."""
        return any(item.is_failure for item in self)

    def as_dict(self) -> Dict[Union[str, Path, int], Optional[DetectionResult]]:
        """Return a dictionary mapping input sources to their DetectionResult (or None if failed)."""
        mapping: Dict[Union[str, Path, int], Optional[DetectionResult]] = {}
        for idx, item in enumerate(self):
            key: Union[str, Path, int]
            if isinstance(item.source, (str, Path)):
                key = item.source
            else:
                key = idx
            mapping[key] = item.result
        return mapping


def _dispatch_progress(
    callback: Callable,
    item: BatchItemResult,
    current: int,
    total: int,
) -> None:
    """Invoke progress callback supporting multiple flexible parameter arities."""
    try:
        sig = inspect.signature(callback)
        param_count = len(sig.parameters)
        if param_count == 1:
            callback(item)
        elif param_count == 2:
            callback(current, total)
        else:
            callback(item, current, total)
    except (ValueError, TypeError):
        try:
            callback(item, current, total)
        except TypeError:
            callback(current, total)


def _resolve_batch_inputs(
    images: Union[str, Path, Sequence[Union[str, Path, np.ndarray]]],
    recursive: bool = False,
) -> List[Union[str, Path, np.ndarray]]:
    """Resolve batch inputs, discovering files in directories when provided.

    Args:
        images: A file path, directory path, NumPy array, or sequence thereof.
        recursive: Whether to recursively scan subdirectories for images.

    Returns:
        List of resolved inputs in deterministic order.
    """
    if isinstance(images, (str, Path)):
        p = Path(images)
        if p.is_dir():
            if recursive:
                return [
                    f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                ]
            return [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
        return [images]

    resolved: List[Union[str, Path, np.ndarray]] = []
    for item in images:
        if isinstance(item, (str, Path)):
            p = Path(item)
            if p.is_dir():
                if recursive:
                    resolved.extend(
                        [
                            f
                            for f in sorted(p.rglob("*"))
                            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                        ]
                    )
                else:
                    resolved.extend(
                        [
                            f
                            for f in sorted(p.iterdir())
                            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                        ]
                    )
            else:
                resolved.append(item)
        else:
            resolved.append(item)
    return resolved


def run_batch_detection(
    detector,
    images: Union[str, Path, Sequence[Union[str, Path, np.ndarray]]],
    fail_fast: bool = False,
    progress_callback: Optional[Callable] = None,
    recursive: bool = False,
    keep_images: bool = True,
) -> BatchDetectionResult:
    """Execute sequential batch face detection over resolved inputs.

    Args:
        detector: FaceDetector instance used to process images.
        images: Single path, directory, NumPy array, or sequence of inputs.
        fail_fast: If True, raise the first encountered error immediately.
                   If False (default), collect errors into BatchItemResult.error.
        progress_callback: Optional callable called after processing each image.
        recursive: Whether to search subdirectories when a directory is passed.
        keep_images: If False, discards pixel arrays from DetectionResult to save RAM.

    Returns:
        BatchDetectionResult containing ordered BatchItemResult items.
    """
    inputs = _resolve_batch_inputs(images, recursive=recursive)
    total = len(inputs)
    results = BatchDetectionResult()

    for idx, input_item in enumerate(inputs, start=1):
        try:
            det_result = detector.detect(input_item)
            if not keep_images:
                det_result = DetectionResult(
                    faces=det_result.faces,
                    image=np.empty((0, 0, 3), dtype=np.uint8),
                )
            item_result = BatchItemResult(source=input_item, result=det_result, error=None)
        except Exception as exc:
            if fail_fast:
                raise
            item_result = BatchItemResult(source=input_item, result=None, error=exc)

        results.append(item_result)

        if progress_callback is not None:
            _dispatch_progress(progress_callback, item_result, idx, total)

    return results
