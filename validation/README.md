# DetekFace Validation Suite

A reproducible, standalone validation and benchmarking framework for evaluating face detection quality and regression safety across `detekface` backends.

## Purpose & Scope

This validation suite is designed to measure **underlying detector quality and failure modes**, rather than making marketing claims that wrapper code increases model accuracy.

It evaluates backends against an annotated set of legally unencumbered images covering 8 essential computer vision challenge dimensions:
1. **No faces**: Verifies false positive (FP) rejection on objects, animals, and complex textures.
2. **One face & multiple faces**: Standard single portrait and multi-person group compositions.
3. **Small & distant faces**: Distant subjects where faces occupy $< 40$ pixels in width.
4. **Faces near image borders**: Validates edge clipping, boundary coordinate transforms, and clamping.
5. **Different lighting conditions**: Severe low-light (underexposed 20% lux) and harsh split side shadows.
6. **Different head angles**: Non-frontal orientations (three-quarters yaw).
7. **Partially occluded faces**: Sunglasses and eyewear obstructing orbital eye regions.
8. **Diverse appearances**: Age diversity (elderly subjects), gender diversity, and uniforms/accessories.

## Quick Start

### 1. Prepare Dataset
Downloads legally verified public domain / Apache 2.0 assets and generates controlled condition images into `validation/data/` (which is kept out of Git via `.gitignore`):

```bash
python validation/download_dataset.py
```

### 2. Run Validation Suite

Evaluate default backend (`YuNet`):
```bash
python validation/evaluate.py --backend yunet --confidence-threshold 0.70
```

Evaluate optional backend (`MediaPipe`):
```bash
python validation/evaluate.py --backend mediapipe --confidence-threshold 0.70
```

Run both backends side-by-side with comparison:
```bash
python validation/evaluate.py --compare
```

Export detailed JSON results:
```bash
python validation/evaluate.py --compare --json validation/results.json
```

## Dataset Licensing & Provenance

Every image in `manifest.json` is verified for open-source / public domain compliance:
- **OpenCV Official Sample Assets**: Apache 2.0 / 3-Clause BSD License (`opencv/opencv` and `opencv/opencv_extra`).
- **U.S. Federal Government Works**: Public domain in the United States under 17 U.S.C. § 105 (U.S. Navy official photography).
- **Derived Condition Assets**: Generated deterministically from licensed base assets with documented parameters.
- **Large Dataset Policy**: Large academic datasets (e.g., WIDER FACE) are not committed to git. Instructions for evaluating full WIDER FACE splits are provided in `docs/validation.md`.
