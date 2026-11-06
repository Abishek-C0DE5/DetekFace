"""Unit tests for detekface.alignment (Milestone 6 Feature A).

Verifies:
- Facial landmark rotation estimation (upright, clockwise, counter-clockwise tilt)
- Face straightening (aligned eyes are horizontally level)
- Output size control (fixed sizes (112, 112), (224, 224), and natural size output_size=None)
- Aspect ratio preservation (strictly uniform scaling, no stretching)
- Padding customization (0.0, 0.25, 0.5)
- Boundary handling (faces near edges/borders with reflection padding)
- Coordinate transformations for 5-point and dense landmarks
- Multi-face alignment with align_faces()
- Reusable FaceAligner class
- DetectionResult.align() and DetectionResult.aligned_faces() integration
- Error handling (empty images, invalid sizes, negative padding, empty detections)
"""

import math
from pathlib import Path
import numpy as np
import pytest

from detekface import (
    AlignedFace,
    BoundingBox,
    DetectionResult,
    FaceAligner,
    FaceDetection,
    FaceLandmarks,
    HeadPose,
    InvalidImageError,
    InvalidSettingError,
    NoFaceDetectedError,
    Point,
    align_face,
    align_faces,
    estimate_face_rotation,
)


@pytest.fixture
def sample_face_upright():
    """FaceDetection with level eyes at y=100."""
    return FaceDetection(
        bbox=BoundingBox(x=50, y=50, width=100, height=120),
        confidence=0.95,
        landmarks=FaceLandmarks(
            right_eye=Point(80, 100),
            left_eye=Point(120, 100),
            nose_tip=Point(100, 120),
            right_mouth=Point(85, 145),
            left_mouth=Point(115, 145),
        ),
    )


@pytest.fixture
def sample_face_tilted_clockwise():
    """FaceDetection tilted 30 degrees clockwise (left eye lower than right eye)."""
    angle_deg = 30.0
    angle_rad = math.radians(angle_deg)
    cx, cy = 100.0, 100.0
    dist = 20.0  # half eye distance

    re_x = cx - dist * math.cos(angle_rad)
    re_y = cy - dist * math.sin(angle_rad)
    le_x = cx + dist * math.cos(angle_rad)
    le_y = cy + dist * math.sin(angle_rad)

    return FaceDetection(
        bbox=BoundingBox(x=50, y=50, width=100, height=120),
        confidence=0.92,
        landmarks=FaceLandmarks(
            right_eye=Point(re_x, re_y),
            left_eye=Point(le_x, le_y),
            nose_tip=Point(cx, cy + 20),
            right_mouth=Point(re_x, cy + 45),
            left_mouth=Point(le_x, cy + 45),
        ),
    )


@pytest.fixture
def sample_image():
    """A 300x300 synthetic test image with colored quadrants."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:150, :150] = (255, 0, 0)
    img[:150, 150:] = (0, 255, 0)
    img[150:, :150] = (0, 0, 255)
    img[150:, 150:] = (255, 255, 0)
    return img


# =========================================================================
# 1. Rotation Estimation Tests
# =========================================================================


def test_estimate_face_rotation_upright(sample_face_upright):
    """Upright face should have tilt angle close to 0.0."""
    angle = estimate_face_rotation(sample_face_upright)
    assert abs(angle) < 1e-4


def test_estimate_face_rotation_clockwise(sample_face_tilted_clockwise):
    """Clockwise tilted face should have positive tilt angle around ~30 degrees."""
    angle = estimate_face_rotation(sample_face_tilted_clockwise)
    assert pytest.approx(angle, abs=0.5) == 30.0


def test_estimate_face_rotation_counter_clockwise():
    """Counter-clockwise tilted face (left eye higher than right eye) has negative tilt angle."""
    face = FaceDetection(
        bbox=BoundingBox(50, 50, 100, 100),
        confidence=0.9,
        landmarks=FaceLandmarks(
            right_eye=Point(80, 110),
            left_eye=Point(120, 90),
            nose_tip=Point(100, 120),
            right_mouth=Point(85, 140),
            left_mouth=Point(115, 130),
        ),
    )
    angle = estimate_face_rotation(face)
    assert angle < 0.0
    assert pytest.approx(angle, abs=0.5) == -26.565


def test_estimate_face_rotation_fallback_to_head_pose():
    """When landmarks are absent, rotation estimation falls back to head_pose.roll."""
    face = FaceDetection(
        bbox=BoundingBox(50, 50, 100, 100),
        confidence=0.9,
        landmarks=None,
        head_pose=HeadPose(yaw=5.0, pitch=-2.0, roll=17.5),
    )
    angle = estimate_face_rotation(face)
    assert pytest.approx(angle, abs=1e-3) == 17.5


def test_estimate_face_rotation_fallback_to_zero():
    """When neither landmarks nor head pose is available, rotation is 0.0."""
    face = FaceDetection(
        bbox=BoundingBox(50, 50, 100, 100),
        confidence=0.9,
        landmarks=None,
        head_pose=None,
    )
    assert estimate_face_rotation(face) == 0.0


# =========================================================================
# 2. Face Alignment & Straightening Tests
# =========================================================================


def test_align_face_straightening(sample_image, sample_face_tilted_clockwise):
    """Face alignment must level the eyes horizontally in the aligned crop."""
    aligned = align_face(sample_image, sample_face_tilted_clockwise, output_size=(112, 112))

    assert isinstance(aligned, AlignedFace)
    assert aligned.image.shape == (112, 112, 3)
    assert aligned.width == 112
    assert aligned.height == 112
    assert pytest.approx(aligned.angle, abs=0.5) == 30.0
    assert pytest.approx(aligned.rotation_applied, abs=0.5) == 30.0

    # In the aligned coordinate space, the eyes should now be level (y coordinates equal)
    assert aligned.landmarks is not None
    assert aligned.landmarks.right_eye is not None
    assert aligned.landmarks.left_eye is not None

    re_y = aligned.landmarks.right_eye.y
    le_y = aligned.landmarks.left_eye.y
    assert abs(re_y - le_y) < 0.5, f"Eyes not level: right_eye.y={re_y}, left_eye.y={le_y}"


def test_align_face_aspect_ratio_preservation(sample_image, sample_face_upright):
    """Alignment must preserve aspect ratio and avoid non-uniform stretching."""
    aligned = align_face(sample_image, sample_face_upright, output_size=(224, 224), padding=0.25)

    M = aligned.transform_matrix
    # For a similarity transform, M = [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
    # Horizontal scaling s_x = sqrt(M[0,0]^2 + M[0,1]^2)
    # Vertical scaling s_y = sqrt(M[1,0]^2 + M[1,1]^2)
    s_x = math.hypot(M[0, 0], M[0, 1])
    s_y = math.hypot(M[1, 0], M[1, 1])

    assert pytest.approx(s_x, rel=1e-3) == s_y, f"Aspect ratio distorted: s_x={s_x}, s_y={s_y}"
    assert pytest.approx(aligned.scale, rel=1e-3) == s_x


def test_align_face_output_sizes(sample_image, sample_face_upright):
    """Alignment must produce consistently sized face crops when output_size is specified."""
    sizes = [(112, 112), (224, 224), (160, 160), (128, 256)]
    for w, h in sizes:
        res = align_face(sample_image, sample_face_upright, output_size=(w, h))
        assert res.image.shape == (h, w, 3)
        assert res.width == w
        assert res.height == h


def test_align_face_natural_size_none(sample_image, sample_face_upright):
    """When output_size=None, crop size is determined by padded face bbox without resizing."""
    padding = 0.2
    res = align_face(sample_image, sample_face_upright, output_size=None, padding=padding)
    expected_w = int(round(sample_face_upright.bbox.width * (1.0 + 2 * padding)))
    expected_h = int(round(sample_face_upright.bbox.height * (1.0 + 2 * padding)))

    assert res.width == expected_w
    assert res.height == expected_h
    assert res.scale == 1.0


def test_align_face_boundary_handling():
    """Alignment handles faces on or outside image boundaries without crashing."""
    small_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Face at top-left boundary touching x=0, y=0
    face_edge = FaceDetection(
        bbox=BoundingBox(x=0, y=0, width=80, height=80),
        confidence=0.88,
        landmarks=FaceLandmarks(
            right_eye=Point(20, 25),
            left_eye=Point(60, 25),
            nose_tip=Point(40, 45),
            right_mouth=Point(25, 65),
            left_mouth=Point(55, 65),
        ),
    )

    aligned = align_face(small_img, face_edge, output_size=(112, 112), padding=0.5)
    assert aligned.image.shape == (112, 112, 3)
    assert not np.isnan(aligned.image).any()


def test_align_face_dense_landmarks(sample_image, sample_face_upright):
    """Alignment correctly transforms dense landmark coordinates into crop space."""
    # Attach synthetic 3D dense landmarks
    dense_pts = np.array(
        [
            [80.0, 100.0, 0.0],
            [120.0, 100.0, 0.0],
            [100.0, 120.0, 5.0],
        ],
        dtype=np.float32,
    )
    face_with_dense = FaceDetection(
        bbox=sample_face_upright.bbox,
        confidence=0.95,
        landmarks=sample_face_upright.landmarks,
        dense_landmarks=dense_pts,
    )

    aligned = align_face(sample_image, face_with_dense, output_size=(112, 112))
    assert aligned.dense_landmarks is not None
    assert aligned.dense_landmarks.shape == (3, 3)
    # Z coordinate should remain intact
    assert aligned.dense_landmarks[2, 2] == 5.0


def test_align_face_rgb_property_and_save(sample_image, sample_face_upright, tmp_path: Path):
    """AlignedFace provides rgb_image and save() method."""
    aligned = align_face(sample_image, sample_face_upright, output_size=(112, 112))
    assert aligned.rgb_image.shape == (112, 112, 3)

    out_file = tmp_path / "aligned_face.png"
    saved = aligned.save(out_file)
    assert saved.exists()
    assert saved.stat().st_size > 0


# =========================================================================
# 3. Multi-Face & Class Interface Tests
# =========================================================================


def test_align_faces_multiple(sample_image, sample_face_upright, sample_face_tilted_clockwise):
    """align_faces() aligns all detected faces in sequence."""
    faces = [sample_face_upright, sample_face_tilted_clockwise]
    aligned_list = align_faces(sample_image, faces, output_size=(112, 112))

    assert len(aligned_list) == 2
    assert aligned_list[0].width == 112
    assert aligned_list[1].width == 112
    assert pytest.approx(aligned_list[0].angle, abs=0.5) == 0.0
    assert pytest.approx(aligned_list[1].angle, abs=0.5) == 30.0


def test_face_aligner_class(sample_image, sample_face_upright, sample_face_tilted_clockwise):
    """FaceAligner encapsulates alignment configuration cleanly."""
    aligner = FaceAligner(output_size=(160, 160), padding=0.3)
    single = aligner.align(sample_image, sample_face_upright)
    assert single.width == 160
    assert single.height == 160

    multiple = aligner.align_all(sample_image, [sample_face_upright, sample_face_tilted_clockwise])
    assert len(multiple) == 2
    assert multiple[0].width == 160
    assert multiple[1].width == 160


# =========================================================================
# 4. DetectionResult Integration Tests
# =========================================================================


def test_detection_result_align_method(sample_image, sample_face_upright):
    """DetectionResult.align() aligns the primary or specified face."""
    res = DetectionResult(faces=[sample_face_upright], image=sample_image)
    aligned = res.align(output_size=(112, 112))
    assert isinstance(aligned, AlignedFace)
    assert aligned.width == 112


def test_detection_result_aligned_faces_method(sample_image, sample_face_upright, sample_face_tilted_clockwise):
    """DetectionResult.aligned_faces() aligns all detected faces."""
    res = DetectionResult(faces=[sample_face_upright, sample_face_tilted_clockwise], image=sample_image)
    crops = res.aligned_faces(output_size=(112, 112))
    assert len(crops) == 2


def test_detection_result_align_no_faces_raises(sample_image):
    """DetectionResult.align() raises NoFaceDetectedError when result contains no faces."""
    res = DetectionResult(faces=[], image=sample_image)
    with pytest.raises(NoFaceDetectedError):
        res.align()


def test_detection_result_align_out_of_range_index(sample_image, sample_face_upright):
    """DetectionResult.align(face_index=99) raises IndexError."""
    res = DetectionResult(faces=[sample_face_upright], image=sample_image)
    with pytest.raises(IndexError):
        res.align(face_index=99)


# =========================================================================
# 5. Invalid Settings & Error Handling
# =========================================================================


def test_align_invalid_image(sample_face_upright):
    """Invalid or empty image array raises InvalidImageError."""
    with pytest.raises(InvalidImageError):
        align_face(np.empty((0, 0, 3), dtype=np.uint8), sample_face_upright)


def test_align_negative_padding(sample_image, sample_face_upright):
    """Negative padding raises InvalidSettingError."""
    with pytest.raises(InvalidSettingError):
        align_face(sample_image, sample_face_upright, padding=-0.1)


def test_align_invalid_output_size(sample_image, sample_face_upright):
    """Invalid output_size dimensions raise InvalidSettingError."""
    with pytest.raises(InvalidSettingError):
        align_face(sample_image, sample_face_upright, output_size=(-10, 112))
    with pytest.raises(InvalidSettingError):
        align_face(sample_image, sample_face_upright, output_size=(112,))
