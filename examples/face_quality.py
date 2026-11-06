"""Technical Face Quality Assessment Example (detekface 0.3.0).

Demonstrates objective capture quality assessment without subjective beauty bias:
- Measures Laplacian blur / sharpness
- Evaluates luminance / lighting exposure
- Evaluates framing, face size ratio, and edge clipping
- Checks 3D head-pose frontal suitability
- Returns actionable technical warnings
"""

from pathlib import Path
import detekface


def main() -> None:
    asset_path = Path(__file__).parent.parent / "tests" / "assets" / "sample_face.jpg"
    if not asset_path.exists():
        print(f"Sample asset not found at: {asset_path}")
        return

    print(f"Assessing technical capture quality for: {asset_path}")
    quality = detekface.check_face_quality(asset_path)

    print("\n--- Technical Quality Report ---")
    print(f"Total Faces Evaluated : {quality.face_count}")

    if quality.primary_face is not None:
        pf = quality.primary_face
        print(f"Sharpness Score       : {pf.sharpness:.1f} (Threshold >= 80.0)")
        print(f"Mean Brightness       : {pf.brightness:.1f} / 255 (Optimal: 50 - 200)")
        print(f"Face Size             : {pf.face_width}x{pf.face_height} px ({pf.area_ratio * 100:.2f}% of image)")
        print(f"Framing Status        : {'Centrally Framed' if not pf.is_clipped else 'Border Clipped'}")
        if pf.head_pose is not None:
            print(f"Head Orientation      : Yaw={pf.head_pose.yaw:.1f}°, Pitch={pf.head_pose.pitch:.1f}°, Roll={pf.head_pose.roll:.1f}°")
        print(f"Quality Warnings      : {pf.warnings if pf.warnings else 'None (Passes technical thresholds)'}")

    print("\nDisclaimer:")
    print(quality.disclaimer)


if __name__ == "__main__":
    main()

