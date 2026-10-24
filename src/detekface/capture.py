"""Face cropping and capture functionality from detection results."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np

from detekface.exceptions import (
    CaptureSaveError,
    InvalidCropError,
    InvalidImageError,
    NoFaceDetectedError,
)
from detekface.types import DetectionResult, FaceCapture, FaceDetection


def capture(
    result: DetectionResult,
    face_index: Optional[int] = None,
    save_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None,
    save: bool = True,
) -> FaceCapture:
    """Crop and capture a detected face from a DetectionResult.

    Selects the largest face by bounding-box area by default, or the face at `face_index`
    if specified. Clamps coordinates to image boundaries, verifies non-empty crops, and
    optionally saves the cropped face to disk.

    Args:
        result: A DetectionResult containing the original BGR image and detected faces.
        face_index: Optional index of the specific face to capture. If None, captures the largest face.
        save_dir: Directory where the cropped face image will be saved (default is 'captures').
        filename: Optional custom filename (e.g. 'my_face.jpg'). Default is timestamp-based.
        save: Whether to save the cropped face image to disk (default True).

    Returns:
        FaceCapture containing the cropped image array, saved path, bounding box, and confidence.

    Raises:
        InvalidImageError: If the input result does not contain a valid image array.
        NoFaceDetectedError: If no faces were detected in the DetectionResult.
        IndexError: If face_index is outside the range of detected faces.
        InvalidCropError: If the bounding box lies outside the image, yielding an empty crop.
        CaptureSaveError: If saving the cropped image file fails.
    """
    if not isinstance(result, DetectionResult) or not isinstance(result.image, np.ndarray) or result.image.size == 0:
        raise InvalidImageError(
            message="DetectionResult does not contain a valid image array.",
            hint="Pass a valid DetectionResult returned by detekface.detect().",
        )

    if not result.has_faces or len(result.faces) == 0:
        raise NoFaceDetectedError(
            message="No faces detected in the provided DetectionResult.",
            hint="Ensure the image or frame contains a clearly visible face above the confidence threshold.",
        )

    img_h, img_w = result.image.shape[:2]

    def _visible_area(face: FaceDetection) -> int:
        vx1 = max(0, min(face.bbox.x1, img_w))
        vy1 = max(0, min(face.bbox.y1, img_h))
        vx2 = max(0, min(face.bbox.x2, img_w))
        vy2 = max(0, min(face.bbox.y2, img_h))
        return max(0, vx2 - vx1) * max(0, vy2 - vy1)

    # Face selection: largest by visible area, or user-selected index
    if face_index is None:
        target_face: FaceDetection = max(
            result.faces,
            key=lambda f: (_visible_area(f), f.bbox.area, f.confidence),
        )
    else:
        if not isinstance(face_index, int) or face_index < 0 or face_index >= len(result.faces):
            raise IndexError(
                f"Face index {face_index} is out of range. Only {len(result.faces)} face(s) were detected."
            )
        target_face = result.faces[face_index]

    bbox = target_face.bbox

    # Clamp coordinates strictly within image boundaries
    x1 = max(0, min(bbox.x1, img_w))
    y1 = max(0, min(bbox.y1, img_h))
    x2 = max(0, min(bbox.x2, img_w))
    y2 = max(0, min(bbox.y2, img_h))

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w <= 0 or crop_h <= 0:
        raise InvalidCropError(
            message=(
                f"Bounding box (x={bbox.x}, y={bbox.y}, w={bbox.width}, h={bbox.height}) "
                f"lies outside image boundaries ({img_w}x{img_h}), yielding an empty crop."
            ),
            hint="Verify bounding box coordinates are inside the image area.",
        )

    cropped_image = result.image[y1:y2, x1:x2].copy()
    saved_path: Optional[Path] = None

    if save:
        target_dir = Path(save_dir).expanduser().resolve() if save_dir else Path("captures").resolve()
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise CaptureSaveError(
                message=f"Could not create capture directory: {target_dir}",
                hint=str(e),
            ) from e

        if filename is not None and str(filename).strip():
            clean_name = str(filename).strip()
            if not any(clean_name.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp")):
                clean_name = f"{clean_name}.jpg"
        else:
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            clean_name = f"face_capture_{now_str}.jpg"

        saved_path = target_dir / clean_name

        try:
            success = cv2.imwrite(str(saved_path), cropped_image)
            if not success:
                raise CaptureSaveError(
                    message=f"OpenCV failed to write image to disk at: {saved_path}",
                    hint="Verify directory write permissions and available disk space.",
                )
        except cv2.error as e:
            raise CaptureSaveError(
                message=f"Failed to write image file: {saved_path}",
                hint=str(e),
            ) from e

    return FaceCapture(
        image=cropped_image,
        path=saved_path,
        bbox=target_face.bbox,
        confidence=target_face.confidence,
        landmarks=target_face.landmarks,
    )
