"""Unit tests for detekface.quality (Milestone 6 Feature B).

Verifies:
- Blur/sharpness detection (sharp vs artificially blurred image)
- Illumination/brightness analysis (optimal, underexposed, overexposed)
- Face size relative to image (large vs undersized faces)
- Framing analysis (centered, off-center, border truncation/clipping)
- Head pose orientation suitability (frontal vs profile/turned/tilted)
- Landmark completeness verification (full vs missing keypoints)
- Multiple independent faces assessment
- check_face_quality() top-level API (paths, arrays, DetectionResult)
- Error handling (ImageNotFoundError, InvalidImageError, zero faces)
- Technical disclaimer and non-judgmental wording
- Optional camera show_quality integration
"""

import cv2
import numpy as np
import pytest

from detekface import (
    BoundingBox,
    CameraSession,
    DetectionResult,
    FaceDetection,
    FaceLandmarks,
    HeadPose,
    ImageNotFoundError,
    ImageQualityResult,
    InvalidImageError,
    Point,
    QUALITY_DISCLAIMER,
    assess_face_quality,
    check_face_quality,
)


@pytest.fixture
def high_quality_face_and_image():
    """Generates a sharp, well-lit, centered synthetic portrait."""
    # 400x400 image with gradient background
    img = np.full((400, 400, 3), 130, dtype=np.uint8)

    # Draw textured face pattern inside 150x180 region at (125, 110)
    # Adding high-frequency edges for strong Laplacian sharpness
    for i in range(125, 275, 4):
        cv2.line(img, (i, 110), (i, 290), (i % 255, (i * 2) % 255, 200), 1)

    face = FaceDetection(
        bbox=BoundingBox(x=125, y=110, width=150, height=180),
        confidence=0.98,
        landmarks=FaceLandmarks(
            right_eye=Point(165, 160),
            left_eye=Point(235, 160),
            nose_tip=Point(200, 200),
            right_mouth=Point(175, 240),
            left_mouth=Point(225, 240),
        ),
        head_pose=HeadPose(yaw=2.0, pitch=-1.5, roll=0.5),
    )
    return img, face


@pytest.fixture
def sample_image():
    """A 300x300 synthetic test image."""
    return np.full((300, 300, 3), 128, dtype=np.uint8)


# =========================================================================
# 1. Blur & Sharpness Tests
# =========================================================================


def test_quality_sharpness_detection(high_quality_face_and_image):
    """Sharp face should have high sharpness score and is_blurry=False."""
    img, face = high_quality_face_and_image
    metrics = assess_face_quality(img, face, blur_threshold=80.0)

    assert metrics.sharpness > 80.0
    assert metrics.is_blurry is False


def test_quality_blur_warning(high_quality_face_and_image):
    """Artificially blurred face should be flagged with is_blurry=True and an actionable warning."""
    img, face = high_quality_face_and_image
    blurred_img = cv2.GaussianBlur(img, (31, 31), 10.0)

    metrics = assess_face_quality(blurred_img, face, blur_threshold=80.0)

    assert metrics.sharpness < 80.0
    assert metrics.is_blurry is True
    assert any("blurry" in w.lower() for w in metrics.warnings)


# =========================================================================
# 2. Illumination & Brightness Tests
# =========================================================================


def test_quality_optimal_brightness(high_quality_face_and_image):
    """Adequately illuminated face ROI reports brightness_status='optimal'."""
    img, face = high_quality_face_and_image
    metrics = assess_face_quality(img, face)

    assert 50.0 <= metrics.brightness <= 200.0
    assert metrics.brightness_status == "optimal"


def test_quality_underexposed_face():
    """Dark/underexposed image triggers 'underexposed' status and lighting recommendation."""
    dark_img = np.full((300, 300, 3), 25, dtype=np.uint8)
    face = FaceDetection(
        bbox=BoundingBox(x=80, y=80, width=140, height=140),
        confidence=0.9,
    )
    metrics = assess_face_quality(dark_img, face, min_brightness=50.0)

    assert metrics.brightness < 50.0
    assert metrics.brightness_status == "underexposed"
    assert any("underexposed" in w.lower() for w in metrics.warnings)


def test_quality_overexposed_face():
    """Washed out/overexposed image triggers 'overexposed' status and glare recommendation."""
    bright_img = np.full((300, 300, 3), 245, dtype=np.uint8)
    face = FaceDetection(
        bbox=BoundingBox(x=80, y=80, width=140, height=140),
        confidence=0.9,
    )
    metrics = assess_face_quality(bright_img, face, max_brightness=200.0)

    assert metrics.brightness > 200.0
    assert metrics.brightness_status == "overexposed"
    assert any("overexposed" in w.lower() for w in metrics.warnings)


# =========================================================================
# 3. Face Size & Framing Tests
# =========================================================================


def test_quality_face_size_evaluation():
    """Small/distant face dimensions trigger is_too_small=True."""
    large_canvas = np.full((1000, 1000, 3), 120, dtype=np.uint8)
    tiny_face = FaceDetection(
        bbox=BoundingBox(x=200, y=200, width=32, height=36),
        confidence=0.8,
    )
    metrics = assess_face_quality(large_canvas, tiny_face, min_face_size=48)

    assert metrics.is_too_small is True
    assert metrics.face_width == 32
    assert metrics.face_height == 36
    assert any("resolution is low" in w.lower() for w in metrics.warnings)


def test_quality_border_truncation():
    """Face bounding box touching image border triggers is_truncated=True and clipping warning."""
    img = np.full((300, 300, 3), 120, dtype=np.uint8)
    # Face touching left edge x=0
    face_clipped = FaceDetection(
        bbox=BoundingBox(x=1, y=50, width=100, height=100),
        confidence=0.85,
    )
    metrics = assess_face_quality(img, face_clipped, border_margin=4)

    assert metrics.is_truncated is True
    assert metrics.framing == "truncated"
    assert any("clipped by image boundary" in w.lower() for w in metrics.warnings)


# =========================================================================
# 4. Head Pose Suitability Tests
# =========================================================================


def test_quality_frontal_pose(high_quality_face_and_image):
    """Frontal face pose within tolerance is rated is_frontal=True."""
    img, face = high_quality_face_and_image
    metrics = assess_face_quality(img, face)

    assert metrics.is_frontal is True
    assert metrics.pose_suitability == "frontal"


def test_quality_turned_profile_pose(high_quality_face_and_image):
    """Extreme yaw angle (profile view) triggers non-frontal warning."""
    img, _ = high_quality_face_and_image
    turned_face = FaceDetection(
        bbox=BoundingBox(x=100, y=100, width=120, height=120),
        confidence=0.9,
        head_pose=HeadPose(yaw=48.0, pitch=4.0, roll=-2.0),
    )
    metrics = assess_face_quality(img, turned_face)

    assert metrics.is_frontal is False
    assert metrics.pose_suitability == "profile"
    assert any("head pose is angled" in w.lower() for w in metrics.warnings)


# =========================================================================
# 5. Landmark Completeness Tests
# =========================================================================


def test_quality_full_landmarks(high_quality_face_and_image):
    """All 5 landmarks present yields landmark_completeness=1.0 with no missing."""
    img, face = high_quality_face_and_image
    metrics = assess_face_quality(img, face)

    assert metrics.landmark_completeness == 1.0
    assert len(metrics.missing_landmarks) == 0


def test_quality_missing_landmarks():
    """Incomplete landmarks are reported with fraction and missing names."""
    img = np.full((300, 300, 3), 120, dtype=np.uint8)
    partial_face = FaceDetection(
        bbox=BoundingBox(50, 50, 100, 100),
        confidence=0.85,
        landmarks=FaceLandmarks(
            right_eye=Point(70, 80),
            left_eye=Point(130, 80),
            nose_tip=Point(100, 105),
            right_mouth=None,
            left_mouth=None,
        ),
    )
    metrics = assess_face_quality(img, partial_face)

    assert pytest.approx(metrics.landmark_completeness) == 3.0 / 5.0
    assert "right_mouth" in metrics.missing_landmarks
    assert "left_mouth" in metrics.missing_landmarks
    assert any("landmarks are incomplete" in w.lower() for w in metrics.warnings)


# =========================================================================
# 6. check_face_quality() Public API & Multi-Face Tests
# =========================================================================


def test_check_face_quality_with_detection_result(high_quality_face_and_image):
    """check_face_quality accepts pre-computed DetectionResult."""
    img, face = high_quality_face_and_image
    det_res = DetectionResult(faces=[face], image=img)

    report = check_face_quality(det_res)
    assert isinstance(report, ImageQualityResult)
    assert report.face_count == 1
    assert report.primary_face is not None
    assert report.overall_score > 0.0
    assert QUALITY_DISCLAIMER in report.disclaimer


def test_check_face_quality_multiple_faces(sample_image):
    """Multiple faces are evaluated independently with distinct metrics."""
    f1 = FaceDetection(
        bbox=BoundingBox(20, 20, 80, 80),
        confidence=0.9,
        landmarks=FaceLandmarks(Point(40, 50), Point(80, 50), Point(60, 70), Point(45, 85), Point(75, 85)),
    )
    f2 = FaceDetection(
        bbox=BoundingBox(180, 180, 100, 100),
        confidence=0.92,
        landmarks=FaceLandmarks(Point(200, 210), Point(260, 210), Point(230, 240), Point(210, 260), Point(250, 260)),
    )
    det_res = DetectionResult(faces=[f1, f2], image=sample_image)

    report = check_face_quality(det_res)
    assert report.face_count == 2
    assert len(report.faces) == 2
    assert report.primary_face == report.faces[1]  # f2 has larger area (100x100 vs 80x80)


def test_check_face_quality_zero_faces():
    """Image with zero detections returns face_count=0 and informative warning."""
    blank_img = np.zeros((200, 200, 3), dtype=np.uint8)
    det_res = DetectionResult(faces=[], image=blank_img)

    report = check_face_quality(det_res)
    assert report.face_count == 0
    assert report.primary_face is None
    assert len(report.warnings) >= 1
    assert "No faces detected" in report.warnings[0]
    assert bool(report) is False


def test_check_face_quality_summary_formatting(high_quality_face_and_image):
    """summary() produces human-readable text including disclaimer."""
    img, face = high_quality_face_and_image
    det_res = DetectionResult(faces=[face], image=img)

    report = check_face_quality(det_res)
    summary_text = report.summary()

    assert "Face Quality Report" in summary_text
    assert "Sharpness" in summary_text
    assert "Brightness" in summary_text
    assert "Framing" in summary_text
    assert QUALITY_DISCLAIMER in summary_text


def test_check_face_quality_file_not_found():
    """Missing file raises ImageNotFoundError."""
    with pytest.raises(ImageNotFoundError):
        check_face_quality("non_existent_file_12345.jpg")


def test_check_face_quality_invalid_image():
    """Invalid array type raises InvalidImageError."""
    with pytest.raises(InvalidImageError):
        check_face_quality(np.empty((0, 0, 3), dtype=np.uint8))


# =========================================================================
# 7. DetectionResult & CameraSession Integration Tests
# =========================================================================


def test_detection_result_check_quality_method(high_quality_face_and_image):
    """DetectionResult.check_quality() calls check_face_quality cleanly."""
    img, face = high_quality_face_and_image
    det_res = DetectionResult(faces=[face], image=img)

    report = det_res.check_quality()
    assert isinstance(report, ImageQualityResult)
    assert report.face_count == 1


def test_camera_session_render_show_quality(high_quality_face_and_image):
    """CameraSession.render(show_quality=True) renders quality HUD overlays."""
    img, face = high_quality_face_and_image
    det_res = DetectionResult(faces=[face], image=img)

    session = CameraSession(auto_open=False, show_quality=True)
    session._latest_frame = img
    session._latest_result = det_res

    rendered = session.render(show_quality=True)
    assert rendered.shape == img.shape
    # Frame should have drawn quality badge text
    assert not np.array_equal(rendered, img)
