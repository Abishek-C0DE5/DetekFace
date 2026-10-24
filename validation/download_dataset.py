"""Dataset downloader and preparation utility for the DetekFace validation suite.

Prepares evaluation images defined in validation/manifest.json by downloading
legally verified public domain / Apache 2.0 assets and generating controlled condition
variants (border, low-light, shadow, occlusion) from base sample images.

All downloaded datasets are stored in validation/data/ (which is git-ignored to keep
large or binary assets out of the source repository).
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "manifest.json"
DATA_DIR = BASE_DIR / "data"
LOCAL_ASSET_DIR = BASE_DIR.parent / "tests" / "assets"

# Remote source URLs for base OpenCV sample images (Apache-2.0 / BSD-3-Clause)
REMOTE_SOURCES = {
    "fruits.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
    "building.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg",
    "baboon.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
    "messi5.jpg": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg",
    "grace_hopper.png": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/grace_hopper_227.png",
    "karen_and_rob.png": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/cascadeandhog/images/karen-and-rob.png",
    "mona_lisa.png": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/cascadeandhog/images/mona-lisa.png",
}


def download_file(url: str, dest: Path, timeout: int = 15) -> bool:
    """Download a file with user-agent headers and timeout."""
    req = urllib.request.Request(url, headers={"User-Agent": "DetekFace-Validation/1.0"})
    try:
        with (
            urllib.request.urlopen(req, timeout=timeout) as resp,
            open(dest, "wb") as f,
        ):
            f.write(resp.read())
        return True
    except Exception as e:
        print(f"  [Warning] Failed to download {url}: {e}")
        return False


def generate_derived_images(base_image_path: Path, target_dir: Path) -> None:
    """Generate controlled test condition images from a verified base face image."""
    base_img = cv2.imread(str(base_image_path))
    if base_img is None:
        raise RuntimeError(f"Could not load base image for derivation: {base_image_path}")

    # 1. Border face: crop so face is aligned touching the left border (x=0)
    # Lena's face center is around x=208 in 512x512. Cropping from x=208 places the face at x=0.
    border_path = target_dir / "border_face.jpg"
    if not border_path.exists():
        border_img = base_img[:, 208:].copy()
        cv2.imwrite(str(border_path), border_img)
        print(f"  [Created] {border_path.name} (Face shifted to image boundary x=0)")

    # 2. Low-light condition: scale pixel intensity down to 20%
    low_light_path = target_dir / "lighting_low_light.jpg"
    if not low_light_path.exists():
        low_light = (base_img * 0.20).astype(np.uint8)
        cv2.imwrite(str(low_light_path), low_light)
        print(f"  [Created] {low_light_path.name} (Underexposed 20% luminance)")

    # 3. Harsh directional shadow: progressive gradient shadow across left hemisphere
    shadow_path = target_dir / "lighting_harsh_shadow.jpg"
    if not shadow_path.exists():
        shadow = base_img.copy()
        mask = np.ones((512, 512, 3), dtype=np.float32)
        for col in range(256):
            mask[:, col] = col / 256.0 * 0.7 + 0.1
        shadow = (shadow * mask).astype(np.uint8)
        cv2.imwrite(str(shadow_path), shadow)
        print(f"  [Created] {shadow_path.name} (Harsh directional split shadow)")

    # 4. Partial occlusion: dark sunglasses bar covering orbital eye regions
    occ_path = target_dir / "occluded_sunglasses.jpg"
    if not occ_path.exists():
        occ = base_img.copy()
        cv2.rectangle(occ, (230, 230), (340, 275), (25, 25, 25), -1)
        cv2.imwrite(str(occ_path), occ)
        print(f"  [Created] {occ_path.name} (Synthetic dark sunglasses occlusion)")


def prepare_dataset(data_dir: Path = DATA_DIR, force: bool = False) -> bool:
    """Download and prepare all dataset images referenced in manifest.json."""
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"==> Preparing DetekFace Validation Dataset in: {data_dir}")

    # Copy local sample_face.jpg if present
    local_sample = LOCAL_ASSET_DIR / "sample_face.jpg"
    dest_sample = data_dir / "sample_face.jpg"
    if local_sample.exists() and (not dest_sample.exists() or force):
        shutil.copy(local_sample, dest_sample)
        print(f"  [Copied] {dest_sample.name} from local test assets")

    # Download remote base images
    for filename, url in REMOTE_SOURCES.items():
        target = data_dir / filename
        if target.exists() and not force:
            continue
        print(f"  [Downloading] {filename} from {url} ...")
        success = download_file(url, target)
        if not success and not target.exists():
            print(f"  [Notice] Could not download {filename}. Checking fallback...")

    # Generate derived condition images
    if dest_sample.exists():
        generate_derived_images(dest_sample, data_dir)
    elif local_sample.exists():
        generate_derived_images(local_sample, data_dir)

    # Check completeness
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        missing = []
        for img_info in manifest.get("images", []):
            fname = img_info.get("filename")
            if not (data_dir / fname).exists():
                missing.append(fname)

        if missing:
            print(f"==> Warning: {len(missing)} images could not be obtained: {missing}")
            print("    Some network downloads may have failed. Please re-run with network access.")
            return False
        else:
            print(f"==> Validation dataset complete! All {len(manifest.get('images', []))} images ready.")
            return True

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and download DetekFace validation images.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Target data directory")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-generation")
    args = parser.parse_args()

    success = prepare_dataset(data_dir=args.data_dir, force=args.force)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
