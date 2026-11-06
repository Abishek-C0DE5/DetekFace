"""Face Alignment & Straightening Example (detekface 0.3.0).

Demonstrates automatic face leveling and standardized cropping:
- Calculates rotation angle from eye landmarks
- Straightens tilted faces using affine similarity transformation
- Produces standardized canonical dimensions (e.g. 112x112 or 224x224)
- Preserves facial aspect proportions without artificial stretching
"""

from pathlib import Path
import cv2
import detekface


def main() -> None:
    asset_path = Path(__file__).parent.parent / "tests" / "assets" / "sample_face.jpg"
    if not asset_path.exists():
        print(f"Sample asset not found at: {asset_path}")
        return

    print(f"Loading image from: {asset_path}")
    result = detekface.detect(asset_path)

    if not result.has_faces:
        print("No faces detected in the image.")
        return

    print(f"Detected {result.count} face(s). Aligning primary face...")

    # Align primary face to standardized 112x112 output
    aligned = result.align(output_size=(112, 112), padding=0.25, center_on="eyes")

    print("\nAlignment Metadata:")
    print(f"  Output Dimensions: {aligned.width}x{aligned.height}")
    print(f"  Applied Rotation : {aligned.rotation_deg:.2f}°")
    print(f"  Scale Factor     : {aligned.scale_factor:.3f}")
    print(f"  Center Point     : ({aligned.center[0]:.1f}, {aligned.center[1]:.1f})")

    out_file = Path("captures") / "aligned_face.jpg"
    saved_path = aligned.save(out_file)
    print(f"\nAligned face crop saved successfully to: {saved_path}")

    # Display side-by-side comparison
    orig_annotated = result.annotate(shape="corners", color="#00FF88")
    cv2.imshow("Original Annotated", orig_annotated)
    cv2.imshow("Aligned & Straightened (112x112)", cv2.resize(aligned.image, (224, 224), interpolation=cv2.INTER_NEAREST))
    print("Press any key to close GUI windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

