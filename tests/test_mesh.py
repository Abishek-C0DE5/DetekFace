"""Tests for detekface facial mesh rendering engine (Milestone 1).

Verifies:
- Canonical mesh topology and regions
- Coordinate mapping across multiple image resolutions
- Boundary clipping and clamping
- Missing, incomplete, NaN, and invalid landmark safety
- Synthetic landmark fixture rendering
- Independent multiple face rendering
- Strict preservation of source images
- Clean isolation ensuring face crops are never polluted by mesh lines
"""

import math
from pathlib import Path
from typing import List
import numpy as np
import pytest

from detekface.capture import capture
from detekface.drawing import SUPPORTED_SHAPES
from detekface.exceptions import InvalidImageError, InvalidSettingError
from detekface.mesh import (
    FACEMESH_CONTOURS,
    FACEMESH_FACE_OVAL,
    FACEMESH_LEFT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_LIPS,
    FACEMESH_NOSE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW,
    SUPPORTED_MESH_REGIONS,
    SUPPORTED_MESH_STYLES,
    FaceMeshRenderer,
    convert_landmarks_to_pixels,
    draw_face_meshes,
    draw_facial_mesh,
    validate_mesh_style,
)
from detekface.types import BoundingBox, DetectionResult, FaceDetection, Point


@pytest.fixture
def blank_image_bgr() -> np.ndarray:
    """Create a black 400x400 BGR uint8 image array."""
    return np.zeros((400, 400, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_478_landmarks() -> List[Point]:
    """Generate 478 synthetic normalized facial landmark points.

    Arranged in an oval/mesh-like distribution centered at (0.5, 0.5) with radius 0.25.
    """
    points: List[Point] = []
    for i in range(478):
        angle = 2.0 * math.pi * (i / 478.0)
        # Radial modulation to give non-trivial geometry
        r = 0.25 + 0.05 * math.sin(i * 0.1)
        x = 0.5 + r * math.cos(angle)
        y = 0.5 + r * math.sin(angle)
        points.append(Point(x=x, y=y))
    return points


class TestMeshTopology:
    """Tests for facial mesh topological definitions."""

    def test_topological_connection_constants(self):
        """Verify connection tuples are populated with valid non-negative integer indices."""
        for name, connections in [
            ("FACEMESH_FACE_OVAL", FACEMESH_FACE_OVAL),
            ("FACEMESH_LEFT_EYE", FACEMESH_LEFT_EYE),
            ("FACEMESH_RIGHT_EYE", FACEMESH_RIGHT_EYE),
            ("FACEMESH_LEFT_EYEBROW", FACEMESH_LEFT_EYEBROW),
            ("FACEMESH_RIGHT_EYEBROW", FACEMESH_RIGHT_EYEBROW),
            ("FACEMESH_LIPS", FACEMESH_LIPS),
            ("FACEMESH_NOSE", FACEMESH_NOSE),
            ("FACEMESH_CONTOURS", FACEMESH_CONTOURS),
        ]:
            assert len(connections) > 0, f"{name} should not be empty"
            for start, end in connections:
                assert isinstance(start, int) and start >= 0
                assert isinstance(end, int) and end >= 0
                assert start != end

    def test_supported_mesh_regions(self):
        """Verify region dictionary maps expected keys to connection sets."""
        expected_keys = {
            "contours",
            "oval",
            "eyes",
            "left_eye",
            "right_eye",
            "eyebrows",
            "left_eyebrow",
            "right_eyebrow",
            "lips",
            "nose",
        }
        for k in expected_keys:
            assert k in SUPPORTED_MESH_REGIONS
            assert len(SUPPORTED_MESH_REGIONS[k]) > 0


class TestCoordinateConversion:
    """Tests for convert_landmarks_to_pixels across resolutions and coordinates."""

    @pytest.mark.parametrize(
        "width,height",
        [(100, 100), (640, 480), (1280, 720), (1920, 1080), (3840, 2160)],
    )
    def test_normalized_landmarks_scaling(self, width: int, height: int):
        """Verify normalized coordinates [0.0, 1.0] map correctly to pixels across resolutions."""
        landmarks = [
            Point(x=0.0, y=0.0),
            Point(x=0.5, y=0.5),
            Point(x=1.0, y=1.0),
        ]
        pixels = convert_landmarks_to_pixels(landmarks, image_width=width, image_height=height)

        assert len(pixels) == 3
        # Top-left corner
        assert pixels[0] == (0, 0)
        # Center
        expected_cx = max(0, min(width - 1, int(round(0.5 * width))))
        expected_cy = max(0, min(height - 1, int(round(0.5 * height))))
        assert pixels[1] == (expected_cx, expected_cy)
        # Bottom-right clamped corner
        assert pixels[2] == (width - 1, height - 1)

    def test_pixel_coordinates_handling(self):
        """Verify absolute pixel coordinates are retained and clamped."""
        landmarks = [
            Point(x=50, y=60),
            (120, 180),
            {"x": 200, "y": 250},
        ]
        pixels = convert_landmarks_to_pixels(landmarks, image_width=300, image_height=300)
        assert pixels == [(50, 60), (120, 180), (200, 250)]

    def test_boundary_clipping(self):
        """Verify out-of-bounds coordinates (negative or exceeding image dimensions) clamp safely."""
        w, h = 200, 150
        landmarks = [
            (-0.5, 0.5),  # negative normalized x
            (0.5, 1.5),  # normalized y exceeding 1.0
            (-100, 50),  # negative pixel x
            (50, 500),  # pixel y exceeding height
        ]
        pixels = convert_landmarks_to_pixels(landmarks, image_width=w, image_height=h)

        assert len(pixels) == 4
        for pt in pixels:
            assert pt is not None
            x, y = pt
            assert 0 <= x <= w - 1
            assert 0 <= y <= h - 1

    def test_missing_and_invalid_landmarks(self):
        """Verify missing, NaN, Inf, None, or malformed entries are converted safely to None."""
        landmarks = [
            Point(x=0.2, y=0.2),
            None,
            (float("nan"), 0.5),
            (0.5, float("inf")),
            "invalid_string",
            (),
            Point(x=0.8, y=0.8),
        ]
        pixels = convert_landmarks_to_pixels(landmarks, image_width=100, image_height=100)
        assert len(pixels) == 7
        assert pixels[0] is not None
        assert pixels[1] is None
        assert pixels[2] is None
        assert pixels[3] is None
        assert pixels[4] is None
        assert pixels[5] is None
        assert pixels[6] is not None

    def test_empty_or_zero_dimensions(self):
        """Verify empty input or non-positive dimensions return empty list."""
        assert convert_landmarks_to_pixels([], 100, 100) == []
        assert convert_landmarks_to_pixels(None, 100, 100) == []
        assert convert_landmarks_to_pixels([(0.5, 0.5)], 0, 100) == []
        assert convert_landmarks_to_pixels([(0.5, 0.5)], 100, -5) == []


class TestMeshRenderer:
    """Tests for FaceMeshRenderer and helper drawing functions."""

    def test_render_landmarks_modifies_pixels(self, blank_image_bgr: np.ndarray, synthetic_478_landmarks: List[Point]):
        """Verify rendering modifies output pixels where wireframe lines are drawn."""
        renderer = FaceMeshRenderer(color="#00FF00", thickness=2)
        out = renderer.render_landmarks(blank_image_bgr, synthetic_478_landmarks, copy=True)

        assert out.shape == blank_image_bgr.shape
        # Ensure that lines were actually drawn (non-zero green channel)
        assert np.count_nonzero(out[:, :, 1]) > 0

    def test_original_image_strictly_preserved(self, blank_image_bgr: np.ndarray, synthetic_478_landmarks: List[Point]):
        """Verify that render_landmarks does not mutate the source image when copy=True."""
        original_snapshot = blank_image_bgr.copy()
        _ = draw_facial_mesh(blank_image_bgr, synthetic_478_landmarks, color="#FF0000", copy=True)

        assert np.array_equal(blank_image_bgr, original_snapshot)
        assert np.count_nonzero(blank_image_bgr) == 0

    def test_draw_points_option(self, blank_image_bgr: np.ndarray, synthetic_478_landmarks: List[Point]):
        """Verify draw_points=True draws vertex dots."""
        out_no_pts = draw_facial_mesh(blank_image_bgr, synthetic_478_landmarks, draw_points=False)
        out_with_pts = draw_facial_mesh(blank_image_bgr, synthetic_478_landmarks, draw_points=True, point_radius=2)

        # More non-zero pixels when vertices are rendered as dots
        assert np.count_nonzero(out_with_pts) > np.count_nonzero(out_no_pts)

    def test_selective_regions(self, blank_image_bgr: np.ndarray, synthetic_478_landmarks: List[Point]):
        """Verify selective region rendering draws fewer lines than all contours."""
        out_all = draw_facial_mesh(blank_image_bgr, synthetic_478_landmarks, regions=["contours"])
        out_lips = draw_facial_mesh(blank_image_bgr, synthetic_478_landmarks, regions=["lips"])

        assert np.count_nonzero(out_lips) > 0
        assert np.count_nonzero(out_all) > np.count_nonzero(out_lips)

    def test_incomplete_landmarks_does_not_crash(self, blank_image_bgr: np.ndarray):
        """Verify rendering with truncated landmark list (e.g. only 5 points) does not raise IndexError."""
        few_landmarks = [Point(x=100, y=100), Point(x=120, y=100), Point(x=110, y=120)]
        out = draw_facial_mesh(blank_image_bgr, few_landmarks, copy=True)
        assert out.shape == blank_image_bgr.shape

    def test_invalid_image_raises_error(self):
        """Verify non-array or empty image raises InvalidImageError."""
        renderer = FaceMeshRenderer()
        with pytest.raises(InvalidImageError):
            renderer.render_landmarks(np.empty((0, 0, 3), dtype=np.uint8), [])
        with pytest.raises(InvalidImageError):
            renderer.render_faces(np.empty((0, 0, 3), dtype=np.uint8), [])


class TestMultipleFacesMesh:
    """Tests for multi-face mesh rendering and face crop integrity."""

    def test_render_multiple_faces_independently(self, blank_image_bgr: np.ndarray):
        """Verify multiple faces with independent landmarks are rendered on their respective coordinates."""
        # Face 1 on left side of image (x in [0.1, 0.4])
        face1_lms = [Point(x=0.1 + 0.1 * math.cos(i), y=0.5 + 0.1 * math.sin(i)) for i in range(478)]
        # Face 2 on right side of image (x in [0.6, 0.9])
        face2_lms = [Point(x=0.7 + 0.1 * math.cos(i), y=0.5 + 0.1 * math.sin(i)) for i in range(478)]

        face1 = FaceDetection(
            bbox=BoundingBox(x=40, y=160, width=80, height=80),
            confidence=0.95,
            dense_landmarks=face1_lms,
        )
        face2 = FaceDetection(
            bbox=BoundingBox(x=240, y=160, width=80, height=80),
            confidence=0.92,
            dense_landmarks=face2_lms,
        )

        out = draw_face_meshes(blank_image_bgr, [face1, face2], color="#00FF88")

        # Verify left region (face1) and right region (face2) both contain rendered pixels
        left_half = out[:, :200, :]
        right_half = out[:, 200:, :]
        assert np.count_nonzero(left_half) > 0
        assert np.count_nonzero(right_half) > 0

    def test_captured_face_crops_are_not_modified(self, blank_image_bgr: np.ndarray, synthetic_478_landmarks: List[Point]):
        """Verify that calling result.annotate(show_mesh=True) never pollutes captured face crops."""
        # Create a face with a solid color in its bounding box so crop is non-empty
        img = blank_image_bgr.copy()
        img[50:150, 50:150] = (200, 180, 160)

        face = FaceDetection(
            bbox=BoundingBox(x=50, y=50, width=100, height=100),
            confidence=0.98,
            dense_landmarks=synthetic_478_landmarks,
        )
        result = DetectionResult(faces=[face], image=img)

        # 1. Capture before annotation
        crop_before = capture(result, save=False)

        # 2. Render mesh annotation
        annotated = result.annotate(show_mesh=True, mesh_color="#FF0000")

        # 3. Capture after annotation
        crop_after = capture(result, save=False)

        # Result image must remain untouched
        assert np.array_equal(result.image, img)
        # Pre-annotation and post-annotation captures must be strictly identical
        assert np.array_equal(crop_before.image, crop_after.image)
        # Annotated image must differ from the source image
        assert not np.array_equal(annotated, img)

    def test_detection_result_annotate_and_save_mesh(
        self, tmp_path: Path, blank_image_bgr: np.ndarray, synthetic_478_landmarks: List[Point]
    ):
        """Verify DetectionResult.annotate(show_mesh=True) and save(..., show_mesh=True)."""
        img = blank_image_bgr.copy()
        img[50:150, 50:150] = (220, 200, 180)
        face = FaceDetection(
            bbox=BoundingBox(x=50, y=50, width=100, height=100),
            confidence=0.99,
            dense_landmarks=synthetic_478_landmarks,
        )
        result = DetectionResult(faces=[face], image=img)

        # Test annotate
        annotated = result.annotate(show_mesh=True, mesh_color="#00FF00", mesh_thickness=2)
        assert np.count_nonzero(annotated[:, :, 1]) > np.count_nonzero(img[:, :, 1])

        # Test save
        out_file = tmp_path / "mesh_result.jpg"
        saved_path = result.save(out_file, show_mesh=True, mesh_color="#00FF00")
        assert saved_path.exists()
        assert saved_path.stat().st_size > 0


class TestMeshStyles:
    """Milestone 2 tests: customizable facial mesh visualization styles."""

    def test_supported_mesh_styles_constant(self):
        """Verify all 5 required styles are declared in SUPPORTED_MESH_STYLES."""
        expected = {"wireframe", "neon", "minimal", "contour", "points"}
        assert set(SUPPORTED_MESH_STYLES) == expected

    @pytest.mark.parametrize("invalid_style", ["glowing", "hologram", "invalid", "", "3d_reconstruction"])
    def test_invalid_styles_raise_setting_error(self, invalid_style: str):
        """Verify invalid style strings raise actionable InvalidSettingError."""
        with pytest.raises(InvalidSettingError) as exc_info:
            validate_mesh_style(invalid_style)
        assert "Unsupported mesh style" in str(exc_info.value) or "Mesh style must be a string" in str(exc_info.value)

        with pytest.raises(InvalidSettingError):
            FaceMeshRenderer(style=invalid_style)

    @pytest.mark.parametrize("style", ["wireframe", "neon", "minimal", "contour", "points"])
    def test_each_style_renders_independently(
        self, style: str, blank_image_bgr: np.ndarray, synthetic_478_landmarks: List[Point]
    ):
        """Verify each of the 5 styles renders without errors and draws distinct visual elements."""
        out = draw_facial_mesh(
            blank_image_bgr,
            synthetic_478_landmarks,
            style=style,
            color="#00FF88",
            copy=True,
        )
        assert out.shape == blank_image_bgr.shape
        # Pixels must have been drawn
        nonzero_pixels = np.count_nonzero(out)
        assert nonzero_pixels > 0, f"Style '{style}' should draw non-zero pixels"

    def test_points_style_renders_only_dots_without_connecting_lines(
        self, blank_image_bgr: np.ndarray, synthetic_478_landmarks: List[Point]
    ):
        """Verify points style renders dots and fewer pixels than connected wireframe."""
        out_points = draw_facial_mesh(blank_image_bgr, synthetic_478_landmarks, style="points", copy=True)
        out_wireframe = draw_facial_mesh(blank_image_bgr, synthetic_478_landmarks, style="wireframe", copy=True)

        # Wireframe connects lines so it has significantly more non-zero pixels than points alone
        assert np.count_nonzero(out_points) > 0
        assert np.count_nonzero(out_wireframe) > np.count_nonzero(out_points)

    def test_neon_style_highlights_selected_landmarks(
        self, blank_image_bgr: np.ndarray, synthetic_478_landmarks: List[Point]
    ):
        """Verify neon style renders highlighted accent points in addition to contour lines."""
        out_neon = draw_facial_mesh(blank_image_bgr, synthetic_478_landmarks, style="neon", color="#00FF88", copy=True)
        out_minimal = draw_facial_mesh(blank_image_bgr, synthetic_478_landmarks, style="minimal", color="#00FF88", copy=True)

        # Neon emphasizes lines + anchor dots, modifying more pixels than minimal
        assert np.count_nonzero(out_neon) > np.count_nonzero(out_minimal)

    def test_contour_style_renders_facial_outlines(
        self, blank_image_bgr: np.ndarray, synthetic_478_landmarks: List[Point]
    ):
        """Verify contour style renders anatomical boundaries."""
        out_contour = draw_facial_mesh(blank_image_bgr, synthetic_478_landmarks, style="contour", copy=True)
        assert np.count_nonzero(out_contour) > 0

    def test_custom_color_and_thickness_across_styles(
        self, blank_image_bgr: np.ndarray, synthetic_478_landmarks: List[Point]
    ):
        """Verify custom colors and stroke thicknesses are respected."""
        for style in SUPPORTED_MESH_STYLES:
            # Custom hex color
            out_cyan = draw_facial_mesh(
                blank_image_bgr, synthetic_478_landmarks, style=style, color="#00FFFF", thickness=1, copy=True
            )
            # Blue and green channels should be activated for cyan
            assert np.count_nonzero(out_cyan[:, :, 0]) > 0
            assert np.count_nonzero(out_cyan[:, :, 1]) > 0

            # Thickness scaling
            out_thick = draw_facial_mesh(
                blank_image_bgr, synthetic_478_landmarks, style=style, thickness=3, copy=True
            )
            out_thin = draw_facial_mesh(
                blank_image_bgr, synthetic_478_landmarks, style=style, thickness=1, copy=True
            )
            if style != "points":
                assert np.count_nonzero(out_thick) > np.count_nonzero(out_thin)

    def test_invalid_thickness_and_radius_raise_error(self):
        """Verify non-positive or non-integer thickness/radius raise InvalidSettingError."""
        with pytest.raises(InvalidSettingError):
            FaceMeshRenderer(thickness=0)
        with pytest.raises(InvalidSettingError):
            FaceMeshRenderer(thickness=-2)
        with pytest.raises(InvalidSettingError):
            FaceMeshRenderer(point_radius=0)

    def test_multiple_faces_rendered_with_styles(
        self, blank_image_bgr: np.ndarray, synthetic_478_landmarks: List[Point]
    ):
        """Verify multi-face rendering works seamlessly with different styles."""
        face1 = FaceDetection(
            bbox=BoundingBox(x=30, y=30, width=100, height=100),
            confidence=0.95,
            dense_landmarks=[Point(x=p.x * 0.4, y=p.y * 0.4) for p in synthetic_478_landmarks],
        )
        face2 = FaceDetection(
            bbox=BoundingBox(x=200, y=200, width=100, height=100),
            confidence=0.93,
            dense_landmarks=[Point(x=0.5 + p.x * 0.4, y=0.5 + p.y * 0.4) for p in synthetic_478_landmarks],
        )

        for style in SUPPORTED_MESH_STYLES:
            out = draw_face_meshes(blank_image_bgr, [face1, face2], style=style, copy=True)
            # Both top-left (face1) and bottom-right (face2) quadrants must have pixels
            assert np.count_nonzero(out[:200, :200]) > 0
            assert np.count_nonzero(out[200:, 200:]) > 0

    def test_empty_landmarks_safely_handled_in_all_styles(self, blank_image_bgr: np.ndarray):
        """Verify empty or None landmarks return unmodified image across all styles."""
        snap = blank_image_bgr.copy()
        for style in SUPPORTED_MESH_STYLES:
            out1 = draw_facial_mesh(blank_image_bgr, [], style=style, copy=True)
            out2 = draw_facial_mesh(blank_image_bgr, None, style=style, copy=True)
            assert np.array_equal(out1, snap)
            assert np.array_equal(out2, snap)

    def test_compatibility_with_all_detection_overlay_shapes(
        self, blank_image_bgr: np.ndarray, synthetic_478_landmarks: List[Point]
    ):
        """Verify existing overlay shapes (rectangle, corners, circle, rounded, hud) work alongside mesh styles."""
        face = FaceDetection(
            bbox=BoundingBox(x=50, y=50, width=100, height=100),
            confidence=0.99,
            dense_landmarks=synthetic_478_landmarks,
        )
        result = DetectionResult(faces=[face], image=blank_image_bgr.copy())

        for overlay_shape in SUPPORTED_SHAPES:
            for mesh_style in SUPPORTED_MESH_STYLES:
                annotated = result.annotate(
                    shape=overlay_shape,
                    show_mesh=True,
                    mesh_style=mesh_style,
                    color="#FF0000",
                    mesh_color="#00FF00",
                )
                assert annotated.shape == blank_image_bgr.shape
                # Both red (shape overlay) and green (mesh) channels should have pixels
                assert np.count_nonzero(annotated[:, :, 2]) > 0  # Red
                assert np.count_nonzero(annotated[:, :, 1]) > 0  # Green


