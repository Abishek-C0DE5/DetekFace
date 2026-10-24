"""YuNet face detection backend powered by OpenCV's FaceDetectorYN.

Supplied landmarks:
    - right_eye (Point)
    - left_eye (Point)
    - nose_tip (Point)
    - right_mouth (Point)
    - left_mouth (Point)

YuNet does NOT supply mouth_center, right_ear, or left_ear; these remain None.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from detekface.backends.base import BaseFaceBackend
from detekface.exceptions import ConfigurationError, ModelCompatibilityError
from detekface.models import resolve_model
from detekface.types import BoundingBox, FaceDetection, FaceLandmarks, Point


def _validate_threshold(name: str, value: float, min_val: float = 0.0, max_val: float = 1.0) -> None:
    """Validate that a threshold is a float within [min_val, max_val]."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigurationError(
            message=f"Threshold '{name}' must be a number, got {type(value).__name__}.",
            hint=f"Provide a float between {min_val} and {max_val} (e.g. 0.7).",
        )
    if not (min_val <= float(value) <= max_val):
        raise ConfigurationError(
            message=f"Threshold '{name}' out of range: {value}.",
            hint=f"The value must be between {min_val} and {max_val}.",
        )


def _parse_yunet_detections(raw_faces: Optional[np.ndarray]) -> List[FaceDetection]:
    """Parse the raw (N, 15) array returned by cv2.FaceDetectorYN into structured FaceDetection objects."""
    if raw_faces is None or len(raw_faces) == 0:
        return []

    detections: List[FaceDetection] = []
    for row in raw_faces:
        if len(row) < 15:
            continue

        x = int(round(float(row[0])))
        y = int(round(float(row[1])))
        width = int(round(float(row[2])))
        height = int(round(float(row[3])))
        bbox = BoundingBox(x=x, y=y, width=width, height=height)

        # Facial landmarks (right eye, left eye, nose tip, right mouth, left mouth)
        right_eye = Point(x=int(round(float(row[4]))), y=int(round(float(row[5]))))
        left_eye = Point(x=int(round(float(row[6]))), y=int(round(float(row[7]))))
        nose_tip = Point(x=int(round(float(row[8]))), y=int(round(float(row[9]))))
        right_mouth = Point(x=int(round(float(row[10]))), y=int(round(float(row[11]))))
        left_mouth = Point(x=int(round(float(row[12]))), y=int(round(float(row[13]))))

        landmarks = FaceLandmarks(
            right_eye=right_eye,
            left_eye=left_eye,
            nose_tip=nose_tip,
            right_mouth=right_mouth,
            left_mouth=left_mouth,
        )

        confidence = float(row[14])

        detections.append(
            FaceDetection(
                bbox=bbox,
                confidence=confidence,
                landmarks=landmarks,
            )
        )

    return detections


class YuNetBackend(BaseFaceBackend):
    """YuNet face detection backend implementation using OpenCV."""

    def __init__(
        self,
        model: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        _validate_threshold("confidence_threshold", confidence_threshold, 0.0, 1.0)
        _validate_threshold("nms_threshold", nms_threshold, 0.0, 1.0)
        if not isinstance(top_k, int) or top_k < 1:
            raise ConfigurationError(
                message=f"top_k must be a positive integer >= 1, got {top_k}.",
                hint="Set top_k to a value like 5000.",
            )

        self._confidence_threshold = float(confidence_threshold)
        self._nms_threshold = float(nms_threshold)
        self._top_k = int(top_k)
        if isinstance(model, Path):
            self._model_path = model
        else:
            self._model_path = resolve_model(model, cache_dir=cache_dir)

        try:
            self._detector = cv2.FaceDetectorYN.create(
                str(self._model_path),
                "",
                (320, 320),
                score_threshold=self._confidence_threshold,
                nms_threshold=self._nms_threshold,
                top_k=self._top_k,
            )
        except cv2.error as e:
            raise ModelCompatibilityError(
                message=f"Failed to create OpenCV FaceDetectorYN with model: {self._model_path.name}",
                hint=str(e),
            ) from e

        if self._detector is None:
            raise ModelCompatibilityError(
                message=f"OpenCV returned None when initializing FaceDetectorYN with '{self._model_path.name}'.",
                hint="Verify that the ONNX model architecture matches YuNet specifications.",
            )
        self._current_input_size: Optional[Tuple[int, int]] = (320, 320)

    @property
    def name(self) -> str:
        return "yunet"

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @property
    def nms_threshold(self) -> float:
        return self._nms_threshold

    @property
    def model_path(self) -> Path:
        return self._model_path

    def detect(self, bgr_image: np.ndarray) -> List[FaceDetection]:
        height, width = bgr_image.shape[:2]
        target_size = (width, height)
        if self._current_input_size != target_size:
            self._detector.setInputSize(target_size)
            self._current_input_size = target_size
        _, raw_faces = self._detector.detect(bgr_image)
        return _parse_yunet_detections(raw_faces)
