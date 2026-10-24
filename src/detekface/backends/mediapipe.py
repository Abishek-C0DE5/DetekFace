"""MediaPipe face detection backend powered by Google MediaPipe.

Supplied landmarks:
    - right_eye (Point)
    - left_eye (Point)
    - nose_tip (Point)
    - mouth_center (Point)
    - right_ear (Point)
    - left_ear (Point)

MediaPipe does NOT supply separate right_mouth or left_mouth corner coordinates;
these remain None. Landmarks are never invented or interpolated.

Installation:
    pip install "detekface[mediapipe]"
"""

from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np

from detekface.backends.base import BaseFaceBackend
from detekface.exceptions import DependencyNotFoundError, ModelError
from detekface.models import resolve_mediapipe_model
from detekface.types import BoundingBox, FaceDetection, FaceLandmarks, Point


class MediaPipeBackend(BaseFaceBackend):
    """MediaPipe face detection backend implementation."""

    def __init__(
        self,
        model: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.7,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        self._confidence_threshold = float(confidence_threshold)
        self._cache_dir = cache_dir

        # Attempt to import mediapipe dynamically
        try:
            import mediapipe as mp
        except ImportError as e:
            raise DependencyNotFoundError(
                message="The 'mediapipe' backend requires the optional 'mediapipe' package.",
                hint='Install it with: pip install "detekface[mediapipe]"',
            ) from e

        self._mp = mp
        self._detector = None
        self._is_tasks_api = False
        self._model_path: Optional[Path] = None

        # Check for Tasks API (MediaPipe 0.10.x / 1.x+)
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            # Resolve MediaPipe model file
            self._model_path = resolve_mediapipe_model(model, cache_dir=cache_dir)

            options = vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(self._model_path)),
                min_detection_confidence=self._confidence_threshold,
            )
            self._detector = vision.FaceDetector.create_from_options(options)
            self._is_tasks_api = True
        except (ImportError, AttributeError):
            # Fallback for legacy MediaPipe versions with mp.solutions.face_detection
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                self._detector = mp.solutions.face_detection.FaceDetection(
                    min_detection_confidence=self._confidence_threshold,
                    model_selection=0,
                )
                self._is_tasks_api = False
            else:
                raise DependencyNotFoundError(
                    message="Could not initialize MediaPipe FaceDetector using Tasks or Solutions API.",
                    hint='Ensure a compatible version is installed: pip install "detekface[mediapipe]"',
                )

    @property
    def name(self) -> str:
        return "mediapipe"

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @property
    def model_path(self) -> Optional[Path]:
        return self._model_path

    def detect(self, bgr_image: np.ndarray) -> List[FaceDetection]:
        height, width = bgr_image.shape[:2]
        if height <= 0 or width <= 0:
            return []

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        if self._is_tasks_api:
            return self._detect_tasks(rgb_image, width, height)
        return self._detect_solutions(rgb_image, width, height)

    def _detect_tasks(self, rgb_image: np.ndarray, width: int, height: int) -> List[FaceDetection]:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_image)
        results = self._detector.detect(mp_image)

        if not results.detections:
            return []

        detections: List[FaceDetection] = []
        for d in results.detections:
            # Score
            confidence = float(d.categories[0].score) if d.categories else 0.0

            # Bounding box in pixel coordinates
            bb = d.bounding_box
            x = max(0, min(int(bb.origin_x), width))
            y = max(0, min(int(bb.origin_y), height))
            w = max(0, min(int(bb.width), width - x))
            h = max(0, min(int(bb.height), height - y))
            bbox = BoundingBox(x=x, y=y, width=w, height=h)

            # Keypoints: MediaPipe Face Detection supplies 6 keypoints:
            # 0: right eye, 1: left eye, 2: nose tip, 3: mouth center, 4: right ear, 5: left ear
            landmarks = None
            if hasattr(d, "keypoints") and len(d.keypoints) >= 6:
                right_eye = Point(int(round(d.keypoints[0].x * width)), int(round(d.keypoints[0].y * height)))
                left_eye = Point(int(round(d.keypoints[1].x * width)), int(round(d.keypoints[1].y * height)))
                nose_tip = Point(int(round(d.keypoints[2].x * width)), int(round(d.keypoints[2].y * height)))
                mouth_center = Point(int(round(d.keypoints[3].x * width)), int(round(d.keypoints[3].y * height)))
                right_ear = Point(int(round(d.keypoints[4].x * width)), int(round(d.keypoints[4].y * height)))
                left_ear = Point(int(round(d.keypoints[5].x * width)), int(round(d.keypoints[5].y * height)))

                landmarks = FaceLandmarks(
                    right_eye=right_eye,
                    left_eye=left_eye,
                    nose_tip=nose_tip,
                    mouth_center=mouth_center,
                    right_ear=right_ear,
                    left_ear=left_ear,
                    right_mouth=None,
                    left_mouth=None,
                )

            detections.append(FaceDetection(bbox=bbox, confidence=confidence, landmarks=landmarks))

        return detections

    def _detect_solutions(self, rgb_image: np.ndarray, width: int, height: int) -> List[FaceDetection]:
        results = self._detector.process(rgb_image)
        if not results.detections:
            return []

        detections: List[FaceDetection] = []
        for d in results.detections:
            confidence = float(d.score[0]) if d.score else 0.0

            rel_box = d.location_data.relative_bounding_box
            x = max(0, min(int(rel_box.xmin * width), width))
            y = max(0, min(int(rel_box.ymin * height), height))
            w = max(0, min(int(rel_box.width * width), width - x))
            h = max(0, min(int(rel_box.height * height), height - y))
            bbox = BoundingBox(x=x, y=y, width=w, height=h)

            landmarks = None
            keypoints = getattr(d.location_data, "relative_keypoints", [])
            if len(keypoints) >= 6:
                right_eye = Point(int(round(keypoints[0].x * width)), int(round(keypoints[0].y * height)))
                left_eye = Point(int(round(keypoints[1].x * width)), int(round(keypoints[1].y * height)))
                nose_tip = Point(int(round(keypoints[2].x * width)), int(round(keypoints[2].y * height)))
                mouth_center = Point(int(round(keypoints[3].x * width)), int(round(keypoints[3].y * height)))
                right_ear = Point(int(round(keypoints[4].x * width)), int(round(keypoints[4].y * height)))
                left_ear = Point(int(round(keypoints[5].x * width)), int(round(keypoints[5].y * height)))

                landmarks = FaceLandmarks(
                    right_eye=right_eye,
                    left_eye=left_eye,
                    nose_tip=nose_tip,
                    mouth_center=mouth_center,
                    right_ear=right_ear,
                    left_ear=left_ear,
                    right_mouth=None,
                    left_mouth=None,
                )

            detections.append(FaceDetection(bbox=bbox, confidence=confidence, landmarks=landmarks))

        return detections

    def close(self) -> None:
        if self._detector is not None and hasattr(self._detector, "close"):
            try:
                self._detector.close()
            except Exception:
                pass
