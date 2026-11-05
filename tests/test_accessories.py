"""Tests for AR face accessories and virtual props rendering engine."""

from pathlib import Path
import numpy as np
import pytest

from detekface.accessories import (
    ACCESSORY_ALIASES,
    ACCESSORY_CROWN,
    ACCESSORY_FACE_MASK,
    ACCESSORY_FUTURISTIC_VISOR,
    ACCESSORY_HALO,
    ACCESSORY_SUNGLASSES,
    SUPPORTED_ACCESSORIES,
    AccessorySmoother,
    FaceAccessoryRenderer,
    compute_accessory_transform,
    load_accessory_asset,
    overlay_rgba,
    render_accessory,
)
from detekface.exceptions import ImageNotFoundError, InvalidSettingError
from detekface.types import BoundingBox, DetectionResult, FaceDetection, FaceLandmarks, HeadPose, Point


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_face() -> FaceDetection:
    """Standard synthetic frontal face for testing transforms."""
    bbox = BoundingBox(x=100, y=100, width=120, height=140)
    landmarks = FaceLandmarks(
        right_eye=Point(x=130, y=140),
        left_eye=Point(x=190, y=140),
        nose_tip=Point(x=160, y=170),
        right_mouth=Point(x=135, y=200),
        left_mouth=Point(x=185, y=200),
    )
    pose = HeadPose(yaw=0.0, pitch=0.0, roll=0.0)
    return FaceDetection(
        bbox=bbox,
        confidence=0.98,
        landmarks=landmarks,
        head_pose=pose,
        track_id=1,
    )


@pytest.fixture
def tilted_face() -> FaceDetection:
    """Synthetic face with 25-degree head tilt."""
    bbox = BoundingBox(x=150, y=150, width=100, height=120)
    landmarks = FaceLandmarks(
        right_eye=Point(x=170, y=180),
        left_eye=Point(x=225, y=205),
        nose_tip=Point(x=195, y=215),
        right_mouth=Point(x=175, y=240),
        left_mouth=Point(x=220, y=260),
    )
    pose = HeadPose(yaw=5.0, pitch=-2.0, roll=25.0)
    return FaceDetection(
        bbox=bbox,
        confidence=0.95,
        landmarks=landmarks,
        head_pose=pose,
        track_id=2,
    )


# ---------------------------------------------------------------------------
# 1. Asset Loading and Caching Tests
# ---------------------------------------------------------------------------

class TestAccessoryAssetLoading:
    """Verify built-in accessory assets and custom image loading."""

    def test_all_supported_accessories_load_valid_rgba(self) -> None:
        """All 5 supported accessories must load as non-empty 4-channel uint8 arrays."""
        for name in SUPPORTED_ACCESSORIES:
            asset = load_accessory_asset(name)
            assert isinstance(asset, np.ndarray)
            assert asset.ndim == 3
            assert asset.shape[2] == 4  # RGBA
            assert asset.dtype == np.uint8
            assert asset.shape[0] > 0 and asset.shape[1] > 0
            # Alpha channel must contain non-zero transparent regions
            assert np.any(asset[:, :, 3] > 0)

    def test_accessory_aliases_resolve_correctly(self) -> None:
        """Common aliases like 'glasses', 'shades', 'mask' resolve to canonical assets."""
        for alias, canonical in ACCESSORY_ALIASES.items():
            asset_alias = load_accessory_asset(alias)
            asset_canon = load_accessory_asset(canonical)
            np.testing.assert_array_equal(asset_alias, asset_canon)

    def test_unsupported_accessory_name_raises_invalid_setting(self) -> None:
        """Requesting an unknown accessory name raises InvalidSettingError."""
        with pytest.raises(InvalidSettingError) as exc_info:
            load_accessory_asset("pirate_hat")
        assert "Unknown accessory" in str(exc_info.value)
        assert "Supported accessories" in str(exc_info.value.hint)

    def test_missing_custom_file_path_raises_not_found(self) -> None:
        """Custom path that doesn't exist raises ImageNotFoundError."""
        fake_path = Path("non_existent_folder/accessory_file_999.png")
        with pytest.raises(ImageNotFoundError):
            load_accessory_asset(fake_path)

    def test_asset_caching_returns_same_instance(self) -> None:
        """Loading the same accessory twice uses the cache."""
        first = load_accessory_asset("sunglasses")
        second = load_accessory_asset("sunglasses")
        assert first is second


# ---------------------------------------------------------------------------
# 2. Alpha Transparency and Blending Tests (overlay_rgba)
# ---------------------------------------------------------------------------

class TestOverlayRgba:
    """Verify subpixel affine transformation and alpha compositing."""

    def test_full_opacity_blend(self) -> None:
        """Opaque pixel in overlay should completely overwrite background at opacity=1.0."""
        bg = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create small 20x20 overlay with solid green (BGR=(0, 255, 0), A=255)
        ov = np.zeros((20, 20, 4), dtype=np.uint8)
        ov[:, :, 0] = 0    # B
        ov[:, :, 1] = 255  # G
        ov[:, :, 2] = 0    # R
        ov[:, :, 3] = 255  # Alpha

        out = overlay_rgba(bg, ov, center=(50, 50), scale=1.0, angle_deg=0.0, opacity=1.0)
        # Center should be solid green
        assert np.array_equal(out[50, 50], [0, 255, 0])

    def test_zero_opacity_leaves_background_unchanged(self) -> None:
        """Opacity 0.0 leaves the background completely intact."""
        bg = np.full((100, 100, 3), 128, dtype=np.uint8)
        ov = np.full((20, 20, 4), 255, dtype=np.uint8)
        out = overlay_rgba(bg, ov, center=(50, 50), scale=1.0, angle_deg=0.0, opacity=0.0)
        assert np.array_equal(out, bg)

    def test_partial_opacity_alpha_blend(self) -> None:
        """Opacity 0.5 creates a 50/50 blend between background and overlay."""
        bg = np.full((100, 100, 3), 100, dtype=np.uint8)
        ov = np.zeros((20, 20, 4), dtype=np.uint8)
        ov[:, :, 0] = 200  # B
        ov[:, :, 1] = 200  # G
        ov[:, :, 2] = 200  # R
        ov[:, :, 3] = 255  # Solid alpha

        out = overlay_rgba(bg, ov, center=(50, 50), scale=1.0, angle_deg=0.0, opacity=0.5)
        # Expected blend: 100 * 0.5 + 200 * 0.5 = 150
        center_val = out[50, 50, 0]
        assert 145 <= center_val <= 155

    def test_transparent_overlay_pixels_do_not_overwrite_background(self) -> None:
        """Pixels with Alpha=0 must leave background untouched."""
        bg = np.full((100, 100, 3), 88, dtype=np.uint8)
        ov = np.full((20, 20, 4), 255, dtype=np.uint8)
        ov[:, :, 3] = 0  # Fully transparent

        out = overlay_rgba(bg, ov, center=(50, 50), scale=1.0, angle_deg=0.0, opacity=1.0)
        assert np.array_equal(out, bg)

    def test_boundary_clipping_no_crash(self) -> None:
        """Overlaying partially or completely outside image boundaries must clip gracefully."""
        bg = np.zeros((100, 100, 3), dtype=np.uint8)
        ov = np.full((50, 50, 4), 200, dtype=np.uint8)

        # Off top-left corner
        out1 = overlay_rgba(bg, ov, center=(-10, -10), scale=1.0, angle_deg=0.0)
        assert out1.shape == (100, 100, 3)

        # Off bottom-right corner
        out2 = overlay_rgba(bg, ov, center=(120, 120), scale=1.0, angle_deg=0.0)
        assert out2.shape == (100, 100, 3)

        # Huge scaling extending across borders
        out3 = overlay_rgba(bg, ov, center=(50, 50), scale=10.0, angle_deg=45.0)
        assert out3.shape == (100, 100, 3)


# ---------------------------------------------------------------------------
# 3. Transform Computation Tests
# ---------------------------------------------------------------------------

class TestComputeAccessoryTransform:
    """Verify landmark positioning, scale calculation, and rotation angles."""

    def test_sunglasses_centered_between_eyes(self, synthetic_face: FaceDetection) -> None:
        """Sunglasses must be centered between the two eyes."""
        transform = compute_accessory_transform(
            face=synthetic_face,
            accessory=ACCESSORY_SUNGLASSES,
            image_shape=(400, 400),
        )
        assert transform is not None
        cx, cy, scale, angle = transform
        # Eyes are at x=130 and x=190 -> center should be 160
        assert pytest.approx(cx, abs=1.0) == 160.0
        # Eyes are at y=140 -> cy should be near 140
        assert pytest.approx(cy, abs=2.0) == 140.0
        assert scale > 0.0
        assert pytest.approx(angle, abs=0.1) == 0.0

    def test_futuristic_visor_transform(self, synthetic_face: FaceDetection) -> None:
        """Futuristic visor anchors across the eye line and temples."""
        transform = compute_accessory_transform(
            face=synthetic_face,
            accessory=ACCESSORY_FUTURISTIC_VISOR,
            image_shape=(400, 400),
        )
        assert transform is not None
        cx, cy, scale, angle = transform
        assert pytest.approx(cx, abs=1.0) == 160.0
        assert pytest.approx(cy, abs=2.0) == 140.0
        assert scale > 0.0

    def test_crown_positioned_above_forehead(self, synthetic_face: FaceDetection) -> None:
        """Crown must be centered horizontally and placed above top of bbox."""
        transform = compute_accessory_transform(
            face=synthetic_face,
            accessory=ACCESSORY_CROWN,
            image_shape=(400, 400),
        )
        assert transform is not None
        cx, cy, scale, angle = transform
        # Bbox x=100, w=120 -> center 160
        assert pytest.approx(cx, abs=1.0) == 160.0
        # Top of bbox is y=100 -> crown center should be above top (cy < 100)
        assert cy < 100.0

    def test_face_mask_covers_lower_face(self, synthetic_face: FaceDetection) -> None:
        """Face mask must sit below nose and cover mouth/chin."""
        transform = compute_accessory_transform(
            face=synthetic_face,
            accessory=ACCESSORY_FACE_MASK,
            image_shape=(400, 400),
        )
        assert transform is not None
        cx, cy, scale, angle = transform
        assert pytest.approx(cx, abs=1.0) == 160.0
        # Nose is at 170, mouth at 200 -> cy should be between 170 and 220
        assert cy > 170.0

    def test_halo_positioned_above_head(self, synthetic_face: FaceDetection) -> None:
        """Halo must float above top of head."""
        transform = compute_accessory_transform(
            face=synthetic_face,
            accessory=ACCESSORY_HALO,
            image_shape=(400, 400),
        )
        assert transform is not None
        cx, cy, scale, angle = transform
        assert cy < synthetic_face.bbox.y

    def test_head_tilt_rotates_accessory(self, tilted_face: FaceDetection) -> None:
        """When face has head roll (tilt), accessory rotation angle matches image tilt."""
        transform = compute_accessory_transform(
            face=tilted_face,
            accessory=ACCESSORY_SUNGLASSES,
            image_shape=(500, 500),
        )
        assert transform is not None
        cx, cy, scale, angle = transform
        # Head roll tilt in image coordinates is approximately -24.4 to -25.0 degrees
        assert pytest.approx(angle, abs=1.5) == -24.4

    def test_scale_adjust_and_y_offset(self, synthetic_face: FaceDetection) -> None:
        """User scale_adjust and y_offset modify dimensions accordingly."""
        t1 = compute_accessory_transform(
            face=synthetic_face,
            accessory=ACCESSORY_SUNGLASSES,
            image_shape=(400, 400),
            scale_adjust=1.0,
            y_offset=0.0,
        )
        t2 = compute_accessory_transform(
            face=synthetic_face,
            accessory=ACCESSORY_SUNGLASSES,
            image_shape=(400, 400),
            scale_adjust=1.5,
            y_offset=0.1,
        )
        assert t1 is not None and t2 is not None
        assert pytest.approx(t2[2] / t1[2], abs=0.05) == 1.5
        assert t2[1] > t1[1]

    def test_extreme_profile_view_suppressed(self, synthetic_face: FaceDetection) -> None:
        """When head yaw exceeds 55 degrees, accessory transform returns None."""
        profile_face = FaceDetection(
            bbox=synthetic_face.bbox,
            confidence=synthetic_face.confidence,
            landmarks=synthetic_face.landmarks,
            head_pose=HeadPose(yaw=60.0, pitch=0.0, roll=0.0),
        )
        transform = compute_accessory_transform(
            face=profile_face,
            accessory=ACCESSORY_SUNGLASSES,
            image_shape=(400, 400),
        )
        assert transform is None


# ---------------------------------------------------------------------------
# 4. Temporal Smoothing Tests (AccessorySmoother)
# ---------------------------------------------------------------------------

class TestAccessorySmoother:
    """Verify exponential moving average tracking filter for accessories."""

    def test_smoother_reduces_jitter(self) -> None:
        """Smoother averages jittery coordinate values smoothly."""
        smoother = AccessorySmoother(alpha=0.5, deadband_px=0.0, adaptive=False)
        t1 = (100.0, 100.0, 1.0, 0.0)
        t2 = (110.0, 110.0, 1.2, 5.0)

        s1 = smoother.smooth(track_id=1, transform=t1)
        assert s1 == t1

        s2 = smoother.smooth(track_id=1, transform=t2)
        # Expected midpoint with alpha=0.5: (105, 105, 1.1, 2.5)
        assert pytest.approx(s2[0], abs=0.1) == 105.0
        assert pytest.approx(s2[1], abs=0.1) == 105.0
        assert pytest.approx(s2[2], abs=0.05) == 1.1
        assert pytest.approx(s2[3], abs=0.1) == 2.5

    def test_smoother_tracks_different_faces_independently(self) -> None:
        """Separate track_ids have independent smoothing state."""
        smoother = AccessorySmoother()
        smoother.smooth(track_id=1, transform=(50.0, 50.0, 1.0, 0.0))
        smoother.smooth(track_id=2, transform=(200.0, 200.0, 1.5, 10.0))

        s1 = smoother.smooth(track_id=1, transform=(52.0, 50.0, 1.0, 0.0))
        assert s1[0] < 60.0
        assert s1[1] < 60.0


# ---------------------------------------------------------------------------
# 5. Image & Result Rendering Integration Tests
# ---------------------------------------------------------------------------

class TestAccessoryRendering:
    """Verify render_accessory and DetectionResult.render() integration."""

    def test_render_accessory_modifies_copy_only(self, synthetic_face: FaceDetection) -> None:
        """When copy=True (default), source image is unmodified."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img_copy = img.copy()

        out = render_accessory(img, [synthetic_face], accessory="sunglasses", copy=True)
        # Source image must remain pristine black
        np.testing.assert_array_equal(img, img_copy)
        # Output image must contain rendered accessory pixels
        assert not np.array_equal(out, img)

    def test_render_all_accessories_on_image(self, synthetic_face: FaceDetection) -> None:
        """All 5 accessories render without error."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        for acc in SUPPORTED_ACCESSORIES:
            out = render_accessory(img, [synthetic_face], accessory=acc)
            assert out.shape == img.shape
            assert np.any(out > 0)

    def test_multiple_faces_rendered_simultaneously(
        self, synthetic_face: FaceDetection, tilted_face: FaceDetection
    ) -> None:
        """Multiple faces in the same frame each receive accessories."""
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        out = render_accessory(img, [synthetic_face, tilted_face], accessory="crown")
        assert out.shape == img.shape
        assert np.any(out > 0)

    def test_empty_faces_returns_unmodified_image(self) -> None:
        """No faces detected returns a clean image without error."""
        img = np.full((100, 100, 3), 50, dtype=np.uint8)
        out = render_accessory(img, [], accessory="sunglasses")
        np.testing.assert_array_equal(out, img)

    def test_detection_result_render_with_effect_flag(self, synthetic_face: FaceDetection) -> None:
        """DetectionResult.render(effect='sunglasses') draws accessory."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        res = DetectionResult(faces=[synthetic_face], image=img)

        # With default shapes + sunglasses effect
        annotated = res.render(effect="sunglasses")
        assert annotated.shape == img.shape
        assert np.any(annotated > 0)

        # With effect and shape="none" (accessory only)
        acc_only = res.render(effect="crown", shape="none")
        assert acc_only.shape == img.shape
        assert np.any(acc_only > 0)

    def test_detection_result_render_with_accessory_alias(self, synthetic_face: FaceDetection) -> None:
        """DetectionResult.render(accessory='glasses') works via alias."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        res = DetectionResult(faces=[synthetic_face], image=img)
        annotated = res.render(accessory="glasses")
        assert np.any(annotated > 0)

    def test_custom_rgba_numpy_array(self, synthetic_face: FaceDetection) -> None:
        """Providing a custom 4-channel numpy array works directly as an accessory asset."""
        custom_asset = np.zeros((64, 64, 4), dtype=np.uint8)
        custom_asset[:, :, 2] = 255  # Red
        custom_asset[:, :, 3] = 200  # Semi-transparent
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        out = render_accessory(img, [synthetic_face], accessory=custom_asset)
        assert out.shape == img.shape
        assert np.any(out > 0)

    def test_face_accessory_renderer_class(self, synthetic_face: FaceDetection) -> None:
        """FaceAccessoryRenderer directly instantiates and renders frames."""
        renderer = FaceAccessoryRenderer(accessory="sunglasses", opacity=0.9)
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        rendered = renderer.render(img, [synthetic_face])
        assert rendered.shape == img.shape
        assert np.any(rendered > 0)


# ---------------------------------------------------------------------------
# 6. Camera & Video Integration Tests
# ---------------------------------------------------------------------------

class TestCameraAndVideoIntegration:
    """Verify camera session and video pipeline integration with effects."""

    def test_camera_session_accepts_effect_param(self) -> None:
        """CameraSession initializes with effect='sunglasses'."""
        from detekface.camera import CameraSession

        session = CameraSession(
            camera_index=0,
            auto_open=False,
            effect="sunglasses",
            accessory_opacity=0.8,
            accessory_scale=1.1,
            accessory_y_offset=-0.05,
        )
        assert session.effect == "sunglasses"
        assert session.accessory_opacity == 0.8
        assert session.accessory_scale == 1.1
        assert session.accessory_y_offset == -0.05
        assert session.track_faces is True
        assert session.estimate_pose is True
        assert session._accessory_renderer is not None

    def test_camera_session_render_with_effect(self, synthetic_face: FaceDetection) -> None:
        """CameraSession.render() with an effect applies accessory to latest frame."""
        from detekface.camera import CameraSession

        session = CameraSession(
            camera_index=0,
            auto_open=False,
            effect="crown",
        )
        dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
        session._latest_frame = dummy_frame
        session._latest_result = DetectionResult(faces=[synthetic_face], image=dummy_frame)

        rendered = session.render(effect="crown")
        assert rendered.shape == dummy_frame.shape
        assert np.any(rendered > 0)

