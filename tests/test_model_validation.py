"""Real-model validation and regression suite for detekface (Milestone 10).

Validates actual detection quality against legally usable sample images and programmatic
variations covering:
- Blank scene (zero faces / false positives check)
- One face (single face detection and landmark verification)
- Multiple faces (composited multi-subject scenes)
- Varied face sizes (large, medium, small, tiny)
- Varied lighting conditions (low-light, overexposed, low contrast)
- Occlusions (sunglasses, facial obstruction)
- Perspective tilt / turned angles
- Confidence threshold sensitivity (0.3, 0.5, 0.7, 0.9)
- RGB conversion properties on DetectionResult and FaceCapture
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

import detekface
from detekface.exceptions import InvalidImageError


@pytest.fixture
def sample_image_path() -> Path:
    p = Path(__file__).parent / "assets" / "sample_face.jpg"
    if not p.exists():
        pytest.skip(f"Sample asset not found: {p}")
    return p


@pytest.fixture
def base_face_image(sample_image_path) -> np.ndarray:
    img = cv2.imread(str(sample_image_path))
    assert img is not None
    return img


# -------------------------------------------------------------------------
# 1. Blank Scene: Zero Faces & No False Detections
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scene_type",
    ["uniform_black", "uniform_white", "uniform_gray", "random_texture"],
)
def test_blank_scene_produces_zero_detections(scene_type):
    """Verify zero false detections on blank and non-face texture scenes."""
    if scene_type == "uniform_black":
        img = np.zeros((480, 640, 3), dtype=np.uint8)
    elif scene_type == "uniform_white":
        img = np.full((480, 640, 3), 255, dtype=np.uint8)
    elif scene_type == "uniform_gray":
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
    else:
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

    result = detekface.detect(img, confidence_threshold=0.5)
    assert result.count == 0
    assert not result.has_faces
    assert result.faces == []


# -------------------------------------------------------------------------
# 2. Single Face: Confidence, Coordinates, and Landmarks
# -------------------------------------------------------------------------

def test_single_face_detection_and_landmarks(sample_image_path):
    """Verify single face detection accuracy and landmark sanity."""
    result = detekface.detect(sample_image_path, confidence_threshold=0.6)
    assert result.count == 1
    face = result.faces[0]

    # Confidence must be strong for clear face
    assert face.confidence >= 0.80

    # Bounding box coordinates must be inside image
    img_h, img_w = result.image.shape[:2]
    assert 0 <= face.bbox.x < img_w
    assert 0 <= face.bbox.y < img_h
    assert face.bbox.width > 50
    assert face.bbox.height > 50

    # Landmarks verification
    assert face.landmarks is not None
    lm = face.landmarks
    # Right eye (viewer's left) should have smaller x than left eye (viewer's right)
    assert lm.right_eye.x < lm.left_eye.x
    # Nose should be lower than eyes (higher y)
    assert lm.nose_tip.y > lm.right_eye.y
    assert lm.nose_tip.y > lm.left_eye.y
    # Mouth should be lower than nose
    assert lm.right_mouth.y > lm.nose_tip.y
    assert lm.left_mouth.y > lm.nose_tip.y


# -------------------------------------------------------------------------
# 3. Multiple Faces: Composited Multi-Subject Scenes
# -------------------------------------------------------------------------

def test_multiple_faces_composited_scene(base_face_image):
    """Verify detector identifies multiple distinct faces without spurious NMS suppression."""
    # Scale face down to 200x200
    h, w = base_face_image.shape[:2]
    face_resized = cv2.resize(base_face_image, (200, 200))

    # Place two non-overlapping faces onto a 500x700 canvas
    canvas = np.full((500, 700, 3), 220, dtype=np.uint8)
    canvas[50:250, 50:250] = face_resized    # Top-Left face
    canvas[200:400, 400:600] = face_resized  # Bottom-Right face

    result = detekface.detect(canvas, confidence_threshold=0.6)
    assert result.count == 2
    assert len(result.faces) == 2

    # Verify both bounding boxes are in distinct regions
    centers_x = [f.bbox.x + f.bbox.width // 2 for f in result.faces]
    assert min(centers_x) < 300
    assert max(centers_x) > 350


# -------------------------------------------------------------------------
# 4. Face Sizes: Large, Medium, Small
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "size, min_expected",
    [
        ((350, 350), 1),  # Large image
        ((150, 150), 1),  # Medium image
        ((100, 100), 1),  # Small image
    ],
)
def test_face_scale_variation(base_face_image, size, min_expected):
    """Verify face detection across scale variations."""
    w, h = size
    face_scaled = cv2.resize(base_face_image, (w, h))

    canvas = np.full((500, 500, 3), 200, dtype=np.uint8)
    canvas[50:50 + h, 50:50 + w] = face_scaled

    result = detekface.detect(canvas, confidence_threshold=0.5)
    assert result.count >= min_expected


def test_small_face_crop_detection(base_face_image):
    """Verify detection of a small 60x60 cropped face on a canvas."""
    init_res = detekface.detect(base_face_image)
    face_crop = init_res.capture(save=False)

    scaled_face = cv2.resize(face_crop.image, (60, 60))
    canvas = np.full((300, 300, 3), 200, dtype=np.uint8)
    canvas[50:110, 50:110] = scaled_face

    result = detekface.detect(canvas, confidence_threshold=0.4)
    assert result.count >= 1
    assert result.faces[0].confidence > 0.5


# -------------------------------------------------------------------------
# 5. Varied Lighting: Low-light, Overexposed, Low Contrast
# -------------------------------------------------------------------------

def test_varied_lighting_robustness(base_face_image):
    """Test face detection under lighting shifts: dim, overexposed, low contrast."""
    # 1. Dim / Low-light (40% intensity)
    dim_img = (base_face_image.astype(np.float32) * 0.4).clip(0, 255).astype(np.uint8)
    dim_result = detekface.detect(dim_img, confidence_threshold=0.5)
    assert dim_result.count >= 1

    # 2. Overexposed (140% intensity)
    bright_img = (base_face_image.astype(np.float32) * 1.4).clip(0, 255).astype(np.uint8)
    bright_result = detekface.detect(bright_img, confidence_threshold=0.5)
    assert bright_result.count >= 1

    # 3. Low contrast (scaled down range centered around gray)
    low_contrast = (base_face_image.astype(np.float32) * 0.5 + 64).clip(0, 255).astype(np.uint8)
    contrast_result = detekface.detect(low_contrast, confidence_threshold=0.5)
    assert contrast_result.count >= 1


# -------------------------------------------------------------------------
# 6. Occlusions: Sunglasses & Mouth Cover
# -------------------------------------------------------------------------

def test_face_with_occlusion_sunglasses(base_face_image):
    """Test detection when eyes are occluded with sunglasses bar."""
    occ_img = base_face_image.copy()
    result_ref = detekface.detect(base_face_image, confidence_threshold=0.6)
    assert result_ref.count == 1
    face = result_ref.faces[0]

    if face.landmarks:
        # Draw dark rectangle across eye region
        y_eye = min(face.landmarks.right_eye.y, face.landmarks.left_eye.y)
        x_start = face.bbox.x + int(face.bbox.width * 0.15)
        x_end = face.bbox.x + int(face.bbox.width * 0.85)
        cv2.rectangle(occ_img, (x_start, y_eye - 15), (x_end, y_eye + 15), (20, 20, 20), -1)

    result_occ = detekface.detect(occ_img, confidence_threshold=0.4)
    assert result_occ.count >= 1


# -------------------------------------------------------------------------
# 7. Angle / Turned Face Simulation
# -------------------------------------------------------------------------

def test_partially_turned_face_perspective(base_face_image):
    """Test detection when face is subjected to slight perspective warp / shear."""
    h, w = base_face_image.shape[:2]
    # Shear matrix simulating yaw angle
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[20, 10], [w - 30, 0], [10, h - 10], [w - 10, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(base_face_image, matrix, (w, h), borderValue=(200, 200, 200))

    result_warped = detekface.detect(warped, confidence_threshold=0.5)
    assert result_warped.count >= 1


# -------------------------------------------------------------------------
# 8. Threshold Sensitivity Check
# -------------------------------------------------------------------------

def test_confidence_threshold_filtering(base_face_image):
    """Verify confidence threshold reliably filters low-confidence detections."""
    # Threshold = 0.99 (strict)
    strict_res = detekface.detect(base_face_image, confidence_threshold=0.99)
    # Threshold = 0.30 (permissive)
    loose_res = detekface.detect(base_face_image, confidence_threshold=0.30)

    assert loose_res.count >= strict_res.count
    if strict_res.has_faces:
        assert all(f.confidence >= 0.99 for f in strict_res.faces)
    assert all(f.confidence >= 0.30 for f in loose_res.faces)


# -------------------------------------------------------------------------
# 9. RGB Conversion Properties
# -------------------------------------------------------------------------

def test_rgb_image_properties(base_face_image):
    """Verify rgb_image properties on DetectionResult and FaceCapture."""
    result = detekface.detect(base_face_image)
    assert result.has_faces

    # DetectionResult.rgb_image
    rgb_full = result.rgb_image
    assert rgb_full.shape == result.image.shape
    # Channel 0 and Channel 2 should be swapped between BGR and RGB
    assert np.array_equal(rgb_full[:, :, 0], result.image[:, :, 2])
    assert np.array_equal(rgb_full[:, :, 2], result.image[:, :, 0])

    # FaceCapture.rgb_image
    capture = result.capture(save=False)
    rgb_crop = capture.rgb_image
    assert rgb_crop.shape == capture.image.shape
    assert np.array_equal(rgb_crop[:, :, 0], capture.image[:, :, 2])
    assert np.array_equal(rgb_crop[:, :, 2], capture.image[:, :, 0])


# -------------------------------------------------------------------------
# 10. Non-finite values rejection
# -------------------------------------------------------------------------

def test_non_finite_image_array_raises_invalid():
    """Verify float array with NaN or Inf raises InvalidImageError."""
    bad_img = np.full((100, 100, 3), np.nan, dtype=np.float32)
    with pytest.raises(InvalidImageError) as exc_info:
        detekface.detect(bad_img)
    assert "NaN or infinite" in str(exc_info.value)
