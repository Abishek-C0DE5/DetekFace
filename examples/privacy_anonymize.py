"""Face Privacy Anonymization Example (detekface 0.3.0).

Demonstrates local privacy filtering across image files:
- Gaussian Blur
- Pixelation Mosaic
- Solid Masking (box or elliptical)
- Fail-closed mode (protects entire frame if no faces detected)
- Strict source file overwrite protection
"""

from pathlib import Path
import cv2
import detekface


def main() -> None:
    asset_path = Path(__file__).parent.parent / "tests" / "assets" / "sample_face.jpg"
    if not asset_path.exists():
        print(f"Sample asset not found at: {asset_path}")
        return

    print(f"Loading image for privacy filtering: {asset_path}")

    # 1. Blur anonymization
    blurred = detekface.anonymize(asset_path, method="blur", strength=0.6)
    print(f"Blur anonymization completed: {blurred.faces_anonymized} face(s) protected.")

    # 2. Pixelation anonymization
    pixelated = detekface.anonymize(asset_path, method="pixelate", strength=0.7)
    print(f"Pixelation anonymization completed: {pixelated.faces_anonymized} face(s) protected.")

    # 3. Solid elliptical mask
    masked = detekface.anonymize(asset_path, method="solid", color=(20, 20, 20), mask_shape="ellipse")
    print(f"Solid masking completed: {masked.faces_anonymized} face(s) protected.")

    # Save results to captures/
    out_dir = Path("captures")
    out_dir.mkdir(exist_ok=True)
    blurred.save(out_dir / "anonymized_blur.jpg")
    pixelated.save(out_dir / "anonymized_pixelate.jpg")
    masked.save(out_dir / "anonymized_solid.jpg")
    print(f"\nAll anonymized images saved to '{out_dir}/'")

    # Preview side-by-side
    combined = cv2.hconcat([blurred.image, pixelated.image, masked.image])
    cv2.imshow("Privacy Modes: Blur | Pixelate | Solid Ellipse", cv2.resize(combined, (960, 240)))
    print("Press any key to close GUI...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

