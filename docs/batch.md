# Batch Image Detection Guide

The `detekface.detect_many()` function enables high-throughput face detection across collections of images or entire directory trees. It provides structured per-image error tracking, input order preservation, memory management, and progress reporting.

---

## 1. Quick Start Example

The simplest way to process multiple images:

```python
import detekface

# Pass a list of file paths:
results = detekface.detect_many(
    [
        "images/alice.jpg",
        "images/bob.png",
        "images/group.jpg",
    ]
)

print(f"Processed {len(results)} images:")
print(f"  Successful : {len(results.successes)}")
print(f"  Failed     : {len(results.failures)}")
print(f"  Total Faces: {results.total_faces}")

# Iterate over items in original order:
for item in results:
    if item.is_success:
        print(f"✓ {item.source}: found {item.count} face(s)")
    else:
        print(f"✗ {item.source}: error - {item.error}")
```

---

## 2. Processing Whole Directories

You can pass a directory path directly to `detect_many()`. DetekFace automatically discovers all supported image files (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`):

```python
import detekface

# Process all images inside a folder:
results = detekface.detect_many("dataset/photos/")

print(f"Found {results.total_faces} faces across {len(results.successes)} photos.")
```

---

## 3. Real-Time Progress Tracking

For large image collections, track progress with a lightweight callback without needing third-party progress bar packages:

```python
import detekface


def show_progress(completed: int, total: int) -> None:
    percent = (completed / total) * 100
    print(f"\rProgress: [{completed}/{total}] {percent:.1f}% complete", end="")


results = detekface.detect_many(
    "dataset/photos/",
    progress_callback=show_progress,
)
print("\nBatch processing finished!")
```

---

## 4. Error Handling: `fail_fast` Mode

By default (`fail_fast=False`), `detect_many()` continues processing remaining images even if some files are corrupted, missing, or unsupported:

```python
import detekface

mixed_inputs = [
    "valid_photo.jpg",
    "missing_file_123.jpg",  # Will record failure
    "another_valid.png",
]

results = detekface.detect_many(mixed_inputs, fail_fast=False)

# Check for failures:
if results.has_failures:
    print("Encountered errors on following items:")
    for failed_item in results.failures:
        print(f"  File : {failed_item.source}")
        print(f"  Error: {failed_item.error}")
```

If you prefer to abort execution immediately upon encountering an invalid image, set `fail_fast=True`:

```python
import detekface
from detekface.exceptions import ImageNotFoundError

try:
    results = detekface.detect_many(mixed_inputs, fail_fast=True)
except ImageNotFoundError as err:
    print(f"Aborted on error: {err.message}")
    print(f"Hint: {err.hint}")
```

---

## 5. Memory Management for Large Batches (`keep_images`)

When analyzing thousands of high-resolution images, keeping the decoded pixel array for every image in RAM can consume significant memory.

Set `keep_images=False` to release each image's pixel buffer immediately after detection while preserving bounding boxes and landmarks:

```python
import detekface

# Process 10,000 photos with low memory footprint:
results = detekface.detect_many(
    "huge_archive/",
    keep_images=False,  # Releases image arrays from memory
    confidence_threshold=0.6,
)

print(f"Detected {results.total_faces} faces without exhausting RAM.")
```

---

## 6. High-Throughput Downscaling (`input_size`)

To significantly accelerate batch detection on large collections, specify an `input_size` (e.g. `(320, 320)` or `(640, 640)`):

```python
import detekface

# Process images at 640x640 internal model resolution for 3x speedup.
# Coordinates are automatically rescaled to original photo dimensions!
results = detekface.detect_many(
    "photos/",
    input_size=(640, 640),
)
```

---

## 7. Saving Annotations & Face Crops in Bulk

Loop over the successful items to save annotated files or extracted face crops:

```python
from pathlib import Path
import detekface

out_dir = Path("annotated_results")
out_dir.mkdir(exist_ok=True)

results = detekface.detect_many("dataset/photos/")

for item in results.successes:
    if item.has_faces and item.result is not None:
        source_name = Path(str(item.source)).name

        # Save annotated image:
        item.result.save(out_dir / f"annotated_{source_name}", shape="hud", color="cyan")

        # Crop and save the largest face:
        item.result.capture(save_dir=out_dir / "crops", filename=f"crop_{source_name}")
```

---

## 8. Command-Line Batch Usage

You can also run batch processing straight from your terminal:

```bash
# Process a folder and save annotated images:
detekface batch ./photos --output ./annotated_photos

# Save extracted face crops as well:
detekface batch ./photos --output ./annotated --crop-dir ./avatar_crops

# Search subfolders recursively:
detekface batch ./photos --output ./annotated --recursive

# Select MediaPipe backend:
detekface batch ./photos --backend mediapipe --output ./annotated
```
