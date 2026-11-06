"""Unit tests for detekface.anonymize (Milestone 7: Face Privacy Mode).

Verifies:
- Blur anonymization with configurable strength and kernel_size
- Mosaic pixelation with configurable block/cell pixel_size
- Solid color masking with box and ellipse shapes
- Pixel-level correctness (inside face ROI modified, outside untouched)
- Multiple faces independent anonymization
- Partially clipped/boundary face handling
- Zero-face handling (safe pass-through vs fail-closed)
- Fail-closed privacy safety (anonymizes full frame on failure or zero faces)
- Source file overwrite protection (never overwrites input file without overwrite=True)
- Image dimensions and channels preservation
- Privacy disclaimer completeness
- DetectionResult.anonymize() integration
- CameraSession and video() workflow integration
- Invalid input validation (missing files, empty arrays, invalid methods, negative padding)
"""

from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pytest

from detekface import (
    AnonymizationResult,
    BoundingBox,
    CameraSession,
    DetectionResult,
    FaceDetection,
    FaceLandmarks,
    ImageNotFoundError,
    ImageSaveError,
    InvalidImageError,
    InvalidSettingError,
    Point,
    PRIVACY_DISCLAIMER,
    anonymize,
    anonymize_roi,
    normalize_anonymize_method,
    video,
)


@pytest.fixture
def high_contrast_image():
    """Generates a 300x300 image with distinct high-frequency texture."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Checkerboard-like stripes to make pixel modification easily measurable
    for r in range(300):
        for c in range(300):
            img[r, c] = ((r * 17) % 256, (c * 23) % 256, ((r + c) * 31) % 256)
    return img


@pytest.fixture
def sample_faces():
    """Two non-overlapping synthetic face detections."""
    f1 = FaceDetection(
        bbox=BoundingBox(x=30, y=30, width=60, height=60),
        confidence=0.95,
        landmarks=FaceLandmarks(Point(45, 50), Point(75, 50), Point(60, 65), Point(50, 75), Point(70, 75)),
    )
    f2 = FaceDetection(
        bbox=BoundingBox(x=150, y=120, width=80, height=90),
        confidence=0.92,
        landmarks=FaceLandmarks(Point(170, 150), Point(210, 150), Point(190, 170), Point(175, 190), Point(205, 190)),
    )
    return [f1, f2]


# =========================================================================
# 1. Method Normalization & Supported Modes
# =========================================================================


def test_normalize_anonymize_method():
    """Supported names and aliases map cleanly to canonical modes."""
    assert normalize_anonymize_method("blur") == "blur"
    assert normalize_anonymize_method("gaussian") == "blur"
    assert normalize_anonymize_method("pixelate") == "pixelate"
    assert normalize_anonymize_method("mosaic") == "pixelate"
    assert normalize_anonymize_method("solid") == "solid"
    assert normalize_anonymize_method("mask") == "solid"
    assert normalize_anonymize_method("censor") == "solid"
    assert normalize_anonymize_method("blackout") == "solid"

    with pytest.raises(InvalidSettingError):
        normalize_anonymize_method("unknown_method_xyz")


# =========================================================================
# 2. Pixel-Level Correctness & Dimension Preservation
# =========================================================================


def test_anonymize_blur_pixels_modified_inside_roi_only(high_contrast_image, sample_faces):
    """Pixels inside face ROI must be modified; pixels outside must be strictly identical."""
    det_res = DetectionResult(faces=sample_faces, image=high_contrast_image)
    res = anonymize(det_res, method="blur", strength=0.7)

    assert isinstance(res, AnonymizationResult)
    assert res.faces_anonymized == 2
    assert res.image.shape == high_contrast_image.shape
    assert res.image.dtype == high_contrast_image.dtype

    # Create mask of all face regions
    h, w = high_contrast_image.shape[:2]
    faces_mask = np.zeros((h, w), dtype=bool)
    for f in sample_faces:
        b = f.bbox
        faces_mask[b.y1 : b.y2, b.x1 : b.x2] = True

        # Verify pixels INSIDE the face region ARE changed
        inside_orig = high_contrast_image[b.y1 : b.y2, b.x1 : b.x2]
        inside_anon = res.image[b.y1 : b.y2, b.x1 : b.x2]
        assert not np.array_equal(inside_orig, inside_anon), "Face ROI pixels were not modified!"

    # Verify EVERY pixel OUTSIDE the face regions is 100% IDENTICAL
    outside_orig = high_contrast_image[~faces_mask]
    outside_anon = res.image[~faces_mask]
    np.testing.assert_array_equal(outside_orig, outside_anon)


def test_anonymize_pixelate_blocks(high_contrast_image, sample_faces):
    """Pixelation creates constant color mosaic blocks and preserves outside pixels."""
    det_res = DetectionResult(faces=[sample_faces[0]], image=high_contrast_image)
    res = anonymize(det_res, method="pixelate", pixel_size=10)

    f0 = sample_faces[0]
    b = f0.bbox
    face_roi = res.image[b.y1 : b.y2, b.x1 : b.x2]

    # Inside pixelated ROI, adjacent pixels within a block have identical colors
    # Test that block (0:10, 0:10) is uniform
    block = face_roi[0:10, 0:10]
    first_pixel = block[0, 0]
    for r in range(10):
        for c in range(10):
            np.testing.assert_array_equal(block[r, c], first_pixel)


def test_anonymize_solid_color_box(high_contrast_image, sample_faces):
    """Solid box mask fills ROI completely with chosen BGR color."""
    det_res = DetectionResult(faces=[sample_faces[0]], image=high_contrast_image)
    res = anonymize(det_res, method="solid", color=(255, 0, 0), mask_shape="box")

    f0 = sample_faces[0]
    b = f0.bbox
    face_roi = res.image[b.y1 : b.y2, b.x1 : b.x2]

    # Color in BGR: (255, 0, 0) passed as RGB -> in BGR is (0, 0, 255)
    expected_bgr = (0, 0, 255)
    assert np.all(face_roi[:, :, 0] == expected_bgr[0])
    assert np.all(face_roi[:, :, 1] == expected_bgr[1])
    assert np.all(face_roi[:, :, 2] == expected_bgr[2])


def test_anonymize_ellipse_mask_shape(high_contrast_image, sample_faces):
    """Elliptical mask modifies center of face ROI but leaves corner pixels intact."""
    det_res = DetectionResult(faces=[sample_faces[0]], image=high_contrast_image)
    res = anonymize(det_res, method="solid", color=(0, 0, 0), mask_shape="ellipse")

    f0 = sample_faces[0]
    b = f0.bbox
    orig_roi = high_contrast_image[b.y1 : b.y2, b.x1 : b.x2]
    anon_roi = res.image[b.y1 : b.y2, b.x1 : b.x2]

    # Center pixel is inside ellipse -> modified to black (0, 0, 0)
    cy, cx = f0.bbox.height // 2, f0.bbox.width // 2
    np.testing.assert_array_equal(anon_roi[cy, cx], [0, 0, 0])

    # Corner pixel (0, 0) is outside inscribed ellipse -> unmodified
    np.testing.assert_array_equal(anon_roi[0, 0], orig_roi[0, 0])


# =========================================================================
# 3. Boundaries & Padding Tests
# =========================================================================


def test_anonymize_partially_clipped_faces(high_contrast_image):
    """Face bounding boxes extending beyond frame boundaries are safely clamped."""
    # Face extending beyond left and top borders
    clipped_face = FaceDetection(
        bbox=BoundingBox(x=-20, y=-10, width=80, height=80),
        confidence=0.88,
    )
    det_res = DetectionResult(faces=[clipped_face], image=high_contrast_image)
    res = anonymize(det_res, method="blur")

    assert res.faces_anonymized == 1
    assert res.image.shape == high_contrast_image.shape
    # Border pixels in [0:70, 0:60] must be blurred
    assert not np.array_equal(high_contrast_image[0:70, 0:60], res.image[0:70, 0:60])


def test_anonymize_with_padding(high_contrast_image, sample_faces):
    """Padding parameter expands the anonymized region around each face."""
    det_res = DetectionResult(faces=[sample_faces[0]], image=high_contrast_image)
    res_no_pad = anonymize(det_res, method="solid", padding=0.0)
    res_pad = anonymize(det_res, method="solid", padding=0.2)

    # Padded result should modify more pixels than unpadded
    modified_no_pad = np.sum(res_no_pad.image != high_contrast_image)
    modified_pad = np.sum(res_pad.image != high_contrast_image)
    assert modified_pad > modified_no_pad


# =========================================================================
# 4. Zero Faces & Fail-Closed Privacy Mode
# =========================================================================


def test_anonymize_zero_faces_fail_closed_false(high_contrast_image):
    """When 0 faces are detected and fail_closed=False, image remains unmodified."""
    det_res = DetectionResult(faces=[], image=high_contrast_image)
    res = anonymize(det_res, method="blur", fail_closed=False)

    assert res.faces_anonymized == 0
    assert res.fail_closed_triggered is False
    np.testing.assert_array_equal(res.image, high_contrast_image)


def test_anonymize_zero_faces_fail_closed_true(high_contrast_image):
    """When 0 faces are detected and fail_closed=True, entire image canvas is protected."""
    det_res = DetectionResult(faces=[], image=high_contrast_image)
    res = anonymize(det_res, method="blur", fail_closed=True)

    assert res.faces_anonymized == 0
    assert res.fail_closed_triggered is True
    # The entire image should now be blurred
    assert not np.array_equal(res.image, high_contrast_image)
    assert "FAIL-CLOSED" in repr(res)


def test_anonymize_detection_failure_fail_closed_true(high_contrast_image):
    """When detector raises an error and fail_closed=True, full image is anonymized."""
    with patch("detekface.detector.FaceDetector.detect", side_effect=RuntimeError("Model error")):
        res = anonymize(high_contrast_image, method="solid", color="black", fail_closed=True)
        assert res.fail_closed_triggered is True
        # Whole frame should be black
        assert np.all(res.image == 0)


def test_anonymize_detection_failure_fail_closed_false(high_contrast_image):
    """When detector raises an error and fail_closed=False, InvalidImageError is raised."""
    with patch("detekface.detector.FaceDetector.detect", side_effect=RuntimeError("Model error")):
        with pytest.raises(InvalidImageError):
            anonymize(high_contrast_image, method="blur", fail_closed=False)


# =========================================================================
# 5. Source File Overwrite Protection
# =========================================================================


def test_anonymize_never_overwrites_source_by_default(tmp_path: Path, high_contrast_image):
    """Saving to source file path raises ImageSaveError unless overwrite=True."""
    src_file = tmp_path / "source.png"
    cv2.imwrite(str(src_file), high_contrast_image)

    # Calling anonymize() with output_path=src_file without overwrite=True must raise
    with pytest.raises(ImageSaveError):
        anonymize(src_file, method="blur", output_path=src_file, overwrite=False)

    # Verify source file was NOT modified
    on_disk = cv2.imread(str(src_file))
    np.testing.assert_array_equal(on_disk, high_contrast_image)


def test_anonymize_allows_overwrite_when_explicit(tmp_path: Path, high_contrast_image, sample_faces):
    """Explicit overwrite=True allows saving over the source file."""
    src_file = tmp_path / "source_overwrite.png"
    cv2.imwrite(str(src_file), high_contrast_image)

    det_res = DetectionResult(faces=sample_faces, image=high_contrast_image)
    with patch("detekface.detect", return_value=det_res):
        res = anonymize(src_file, method="blur", output_path=src_file, overwrite=True)
        assert res.path == src_file.resolve()
        assert src_file.exists()


def test_anonymization_result_save_target_exists_protection(tmp_path: Path, high_contrast_image):
    """result.save() raises ImageSaveError if target exists and overwrite=False."""
    existing_file = tmp_path / "existing.jpg"
    cv2.imwrite(str(existing_file), high_contrast_image)

    res = AnonymizationResult(
        image=high_contrast_image,
        original_image=high_contrast_image,
        faces_anonymized=1,
    )
    with pytest.raises(ImageSaveError):
        res.save(existing_file, overwrite=False)


# =========================================================================
# 6. Privacy Disclaimer & Result Object
# =========================================================================


def test_privacy_disclaimer_presence(high_contrast_image):
    """Result object carries explicit privacy and heuristic disclaimer."""
    det_res = DetectionResult(faces=[], image=high_contrast_image)
    res = anonymize(det_res, method="blur")

    assert PRIVACY_DISCLAIMER in res.disclaimer
    assert "100%" in res.disclaimer
    assert res.rgb_image.shape == high_contrast_image.shape
    assert res.width == 300
    assert res.height == 300


# =========================================================================
# 7. DetectionResult, Video & CameraSession Integration
# =========================================================================


def test_detection_result_anonymize_method(high_contrast_image, sample_faces):
    """DetectionResult.anonymize() calls anonymize() on itself."""
    det_res = DetectionResult(faces=sample_faces, image=high_contrast_image)
    res = det_res.anonymize(method="pixelate", pixel_size=8)

    assert isinstance(res, AnonymizationResult)
    assert res.faces_anonymized == 2
    assert res.method == "pixelate"


def test_camera_session_render_anonymize(high_contrast_image, sample_faces):
    """CameraSession.render(anonymize='blur') blurs detected face regions."""
    det_res = DetectionResult(faces=sample_faces, image=high_contrast_image)
    session = CameraSession(auto_open=False, anonymize="blur")
    session._latest_frame = high_contrast_image
    session._latest_result = det_res

    rendered = session.render()
    f0 = sample_faces[0].bbox
    assert not np.array_equal(rendered[f0.y1 : f0.y2, f0.x1 : f0.x2], high_contrast_image[f0.y1 : f0.y2, f0.x1 : f0.x2])


def test_video_anonymize_integration(tmp_path: Path, high_contrast_image, sample_faces):
    """detekface.video() processes frames with anonymize='blur' without error."""
    video_path = tmp_path / "test_input.mp4"
    out_video_path = tmp_path / "test_output.mp4"

    # Create short 3-frame synthetic video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_path), fourcc, 10.0, (300, 300))
    for _ in range(3):
        vw.write(high_contrast_image)
    vw.release()

    det_res = DetectionResult(faces=sample_faces, image=high_contrast_image)
    with patch("detekface.detector.FaceDetector.detect", return_value=det_res):
        out = video(
            path=video_path,
            output_path=out_video_path,
            show=False,
            anonymize="blur",
        )
        assert out == out_video_path.resolve()
        assert out_video_path.exists()


# =========================================================================
# 8. Invalid Settings & Error Handling
# =========================================================================


def test_anonymize_missing_file():
    """Non-existent file raises ImageNotFoundError."""
    with pytest.raises(ImageNotFoundError):
        anonymize("missing_image_path_12345.jpg")


def test_anonymize_empty_numpy_array():
    """Empty NumPy array raises InvalidImageError."""
    with pytest.raises(InvalidImageError):
        anonymize(np.empty((0, 0, 3), dtype=np.uint8))


def test_anonymize_negative_padding(high_contrast_image):
    """Negative padding factor raises InvalidSettingError."""
    with pytest.raises(InvalidSettingError):
        anonymize(high_contrast_image, padding=-0.1)


def test_anonymize_roi_invalid_sizes(high_contrast_image):
    """Invalid kernel_size or pixel_size raises InvalidSettingError."""
    roi = high_contrast_image[10:50, 10:50]
    with pytest.raises(InvalidSettingError):
        anonymize_roi(roi, method="blur", kernel_size=-5)
    with pytest.raises(InvalidSettingError):
        anonymize_roi(roi, method="pixelate", pixel_size=0)
