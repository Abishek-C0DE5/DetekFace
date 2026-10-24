# Quality Validation & Evaluation Methodology

This document outlines the reproducible quality validation suite for `detekface`.

> **Guiding Principle**: This milestone is about measuring underlying detector quality and failure modes, **not claiming that Python wrapper code makes the underlying model more accurate**. Evaluation numbers report empirical observations from actual model weights running in the documented test environment.

---

## 1. Evaluation Metrics

`detekface` evaluates detection outputs using standard computer vision object detection evaluation metrics matching the PASCAL VOC and COCO criteria ($IoU \ge 0.50$).

### Intersection over Union (IoU)

Given a predicted bounding box $B_{\text{pred}} = (x_1, y_1, w_1, h_1)$ and an annotated ground-truth box $B_{\text{gt}} = (x_2, y_2, w_2, h_2)$, the Intersection over Union is defined as:

$$\text{IoU}(B_{\text{pred}}, B_{\text{gt}}) = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})}$$

Where:
- $\text{Area}(B_{\text{pred}} \cap B_{\text{gt}}) = \max(0, x_{\text{inter2}} - x_{\text{inter1}}) \times \max(0, y_{\text{inter2}} - y_{\text{inter1}})$
- $\text{Area}(B_{\text{pred}} \cup B_{\text{gt}}) = \text{Area}(B_{\text{pred}}) + \text{Area}(B_{\text{gt}}) - \text{Area}(B_{\text{pred}} \cap B_{\text{gt}})$

### Matching Algorithm

Matching uses greedy bipartite matching:
1. Predictions are sorted in descending order of confidence score.
2. Each prediction is matched to the unmatched ground-truth box with the highest $\text{IoU} \ge \text{iou\_threshold}$ (default `0.50`).
3. If matched, the pair is recorded as a **True Positive (TP)**, and that ground-truth box is marked as matched (cannot be matched again).
4. Any prediction that fails to match an unmatched ground-truth box (or matches an already-claimed box) is recorded as a **False Positive (FP)**.
5. Any ground-truth boxes remaining unmatched after evaluating all predictions are recorded as **False Negatives (FN)**.

### Performance Formulas

- **Precision**: Proportion of predicted face detections that correspond to real human faces:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **Recall**: Proportion of ground-truth human faces that were successfully detected:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **F1-Score**: Harmonic mean of Precision and Recall:
  $$\text{F}_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Mean IoU**: Average overlap accuracy across all True Positive detections:
  $$\text{Mean IoU} = \frac{1}{\text{TP}} \sum_{i=1}^{\text{TP}} \text{IoU}_i$$

---

## 2. Dataset Licensing, Provenance & Ethical Use

To respect intellectual property, privacy, and repository cleanliness, **large datasets (such as 3+ GB WIDER FACE tarballs) are kept out of the Git repository**. 

Instead, `detekface` provides `validation/download_dataset.py`, which populates the git-ignored `validation/data/` directory using legally verified, permissive open-source assets and deterministic controlled transforms.

| Sample ID | File Name | Dimension | Source & Provenance | Permitted License |
| :--- | :--- | :--- | :--- | :--- |
| `val_01` | `fruits.jpg` | 512x480 | [OpenCV Samples](https://github.com/opencv/opencv/blob/master/samples/data/fruits.jpg) | Apache-2.0 / BSD-3 |
| `val_02` | `building.jpg` | 868x600 | [OpenCV Samples](https://github.com/opencv/opencv/blob/master/samples/data/building.jpg) | Apache-2.0 / BSD-3 |
| `val_03` | `baboon.jpg` | 512x512 | [OpenCV Samples](https://github.com/opencv/opencv/blob/master/samples/data/baboon.jpg) | Apache-2.0 / BSD-3 |
| `val_04` | `sample_face.jpg` | 512x512 | [OpenCV Samples](https://github.com/opencv/opencv/blob/master/samples/data/lena.jpg) | Apache-2.0 / BSD-3 |
| `val_05` | `karen_and_rob.png` | 336x484 | [OpenCV Extra Testdata](https://github.com/opencv/opencv_extra/blob/master/testdata/cv/cascadeandhog/images/karen-and-rob.png) | Apache-2.0 / BSD-3 |
| `val_06` | `messi5.jpg` | 548x342 | [OpenCV Samples](https://github.com/opencv/opencv/blob/master/samples/data/messi5.jpg) | Apache-2.0 / BSD-3 |
| `val_07` | `border_face.jpg` | 304x512 | Derived from `sample_face.jpg` (aligned to edge `x=0`) | Apache-2.0 / BSD-3 |
| `val_08` | `lighting_low_light.jpg` | 512x512 | Derived from `sample_face.jpg` (20% luminance) | Apache-2.0 / BSD-3 |
| `val_09` | `lighting_harsh_shadow.jpg`| 512x512 | Derived from `sample_face.jpg` (directional split shadow) | Apache-2.0 / BSD-3 |
| `val_10` | `mona_lisa.png` | 520x739 | [OpenCV Extra Testdata](https://github.com/opencv/opencv_extra/blob/master/testdata/cv/cascadeandhog/images/mona-lisa.png) | Public Domain / Apache-2.0 |
| `val_11` | `occluded_sunglasses.jpg` | 512x512 | Derived from `sample_face.jpg` (dark eye sunglasses bar) | Apache-2.0 / BSD-3 |
| `val_12` | `grace_hopper.png` | 227x227 | [OpenCV Extra Testdata](https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/grace_hopper_227.png) (U.S. Navy official photography) | Public Domain (17 U.S.C. § 105) |

---

## 3. Evaluation Scenarios Covered

The validation manifest (`validation/manifest.json`) tags each image across 8 challenge dimensions:

1. **No Faces (False Positive Suppression)**:
   - Evaluates whether non-human scenes (fruits, geometric architecture, animals) trigger hallucinated face detections.
   - **Crucial Metric**: Zero False Positives (`FP = 0`).
2. **One Face vs Multiple Faces**:
   - Single frontal portrait (`sample_face.jpg`) vs multi-person group scene (`karen_and_rob.png`).
3. **Small and Distant Faces**:
   - Wide-angle shot (`messi5.jpg`) where the face occupies under 6% of frame width ($< 40$ pixels).
4. **Faces Near Image Borders**:
   - Subject cropped directly against the image boundary (`x=0`). Validates that boundary coordinates are neither lost nor overflow array bounds.
5. **Different Lighting Conditions**:
   - Severely underexposed low-light image ($20\%$ luminance) and harsh split directional shadow.
6. **Different Head Angles**:
   - Three-quarters yaw head pose (`mona_lisa.png`).
7. **Partially Occluded Faces**:
   - Synthetic dark sunglasses obscuring orbital eye landmarks; military uniform cap covering forehead.
8. **Diverse Face Appearances**:
   - Demographic representation including elderly subjects (`grace_hopper.png`), female, and male adults.

---

## 4. Empirical Evaluation Results

Tests executed on host hardware:
- **OS**: Windows 11 (AMD64 12 Logical CPUs)
- **Runtime**: Python 3.15.1, OpenCV 5.0.0
- **Thresholds**: `confidence_threshold = 0.70`, `iou_threshold = 0.50`
- **Resolution**: Native (Full Resolution)

### Overall Summary Comparison

| Metric | YuNet (Default) | MediaPipe (Optional) | Architectural Note |
| :--- | :--- | :--- | :--- |
| **Total Images** | 12 | 12 | Identical test samples |
| **Ground-Truth Faces** | 10 | 10 | Hand-verified ground-truth |
| **Predicted Faces** | 10 | 6 | MediaPipe misses distant/occluded faces |
| **True Positives (TP)** | **10** | 6 | YuNet detects all 10 faces |
| **False Positives (FP)** | **0** | **0** | Both backends achieve 0 false positives |
| **False Negatives (FN)** | **0** | 4 | MediaPipe drops 4 challenging faces |
| **Precision** | **1.0000 (100%)** | **1.0000 (100%)** | Neither model hallucinated faces |
| **Recall** | **1.0000 (100%)** | 0.6000 (60.0%) | YuNet excels in unconstrained conditions |
| **F1-Score** | **1.0000** | 0.7500 | YuNet higher overall utility |
| **Mean IoU (on TPs)** | **0.8583** | 0.7387 | YuNet bounding boxes fit ground truth closer |
| **Mean Latency per Image**| 23.18 ms | **7.76 ms** | MediaPipe is ~3x faster on CPU |

### Scenario Breakdown

| Scenario Category | Samples | YuNet F1 | MediaPipe F1 | Key Finding |
| :--- | :---: | :---: | :---: | :--- |
| **`no_faces`** | 3 | **1.000** | **1.000** | Both backends correctly produce 0 detections on objects/animals. |
| **`one_face`** | 8 | **1.000** | 0.857 | MediaPipe drops sunglasses-occluded face and distant face. |
| **`multiple_faces`** | 1 | **1.000** | 0.000 | MediaPipe misses faces in wide two-person shots ($< 15\%$ frame height). |
| **`small_and_distant`** | 1 | **1.000** | 0.000 | YuNet feature pyramid detects $31 \times 40$ px face; MediaPipe short-range does not. |
| **`faces_near_borders`**| 1 | **1.000** | **1.000** | Both backends detect edge-adjacent faces ($x=0$) correctly. |
| **`different_lighting`**| 2 | **1.000** | **1.000** | Both backends detect faces under 20% luminance and split shadows. |
| **`different_angles`**  | 1 | **1.000** | **1.000** | Both backends detect three-quarters yaw head angles. |
| **`partially_occluded`**| 2 | **1.000** | 0.667 | YuNet detects face with dark sunglasses; MediaPipe requires visible eyes. |
| **`diverse_appearances`**| 3 | **1.000** | 0.667 | Both detect elderly female officer; MediaPipe misses multi-person shot. |

---

## 5. Confidence Threshold Sweep (YuNet)

Evaluating the impact of varying `confidence_threshold` from `0.30` to `0.95`:

| Confidence Threshold | Precision | Recall | F1-Score | True Positives | False Positives | False Negatives | Operational Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.30` | 0.455 | 1.000 | 0.625 | 10 | 12 | 0 | High recall, but 12 false positives on background textures. |
| `0.50` | 0.833 | 1.000 | 0.909 | 10 | 2 | 0 | Good sensitivity, 2 minor false positives. |
| **`0.70` (Default)**| **1.000** | **1.000** | **1.000** | **10** | **0** | **0** | **Optimal balance of precision and recall**. |
| `0.85` | 1.000 | 0.900 | 0.947 | 9 | 0 | 1 | Zero false positives, but drops occluded sunglasses face (conf=0.77). |
| `0.95` | 0.000 | 0.000 | 0.000 | 0 | 0 | 10 | Overly strict; drops all faces. |

---

## 6. Resolution Scaling Tradeoffs (`input_size`)

Evaluating YuNet performance when downscaling frames prior to model inference (at `confidence_threshold=0.70`):

| Resolution Setting | Mean Latency | Precision | Recall | F1-Score | Mean IoU | Operational Tradeoff |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Native (Original)** | 19.3 ms | 1.000 | 1.000 | 1.000 | **0.858** | Highest geometric box accuracy. Baseline CPU latency. |
| **`input_size=(640, 480)`** | 25.6 ms | 1.000 | 1.000 | 1.000 | 0.776 | Good for large images; negligible overhead on small samples. |
| **`input_size=(320, 240)`** | **8.5 ms** | 1.000 | 1.000 | 1.000 | 0.740 | **2.3x speedup**. Retains 100% recall with minor IoU quantization drop. |

---

## 7. How to Reproduce

### Step 1: Download & Prepare Validation Dataset
```bash
python validation/download_dataset.py
```
This fetches the verified public domain and Apache 2.0 assets into `validation/data/` (git-ignored) and creates the controlled test conditions.

### Step 2: Run Evaluation
```bash
# Evaluate YuNet
python validation/evaluate.py --backend yunet

# Evaluate MediaPipe
python validation/evaluate.py --backend mediapipe

# Compare both backends side-by-side
python validation/evaluate.py --compare

# Export detailed JSON report
python validation/evaluate.py --compare --json validation_results.json
```

---

## 8. Limitations & Scope

1. **Small Sample Scope**: This curated 12-image validation suite is a **diagnostic regression checkpoint**, not a proof of universal 100% in-the-wild accuracy across all human populations or camera sensors.
2. **Backend Domain Differences**:
   - `YuNet` was trained on WIDER FACE, making it highly effective across varied scales, distant subjects, and partial occlusions.
   - `MediaPipe` BlazeFace short-range was trained specifically for front-facing smartphone selfie cameras; it is ultra-fast for near-field faces but expectedly drops distant faces ($< 15\%$ frame height) and heavy eye occlusions.
3. **Evaluating Large Academic Benchmarks**: For formal scientific benchmarking across tens of thousands of images, researchers should download the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) validation split (3,226 images) and evaluate using the official WIDER FACE MATLAB/Python evaluation tool.
