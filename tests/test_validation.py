"""Unit tests for the DetekFace validation suite and evaluation logic."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from detekface.types import BoundingBox, FaceDetection
from validation.evaluate import (
    calculate_metrics,
    compute_iou,
    evaluate_backend,
    match_detections_to_ground_truth,
)


# =========================================================================
# 1. IoU Calculation Tests
# =========================================================================

def test_compute_iou_identical_boxes():
    """Identical bounding boxes must yield IoU = 1.0."""
    box_a = (10, 10, 50, 50)
    box_b = (10, 10, 50, 50)
    assert compute_iou(box_a, box_b) == pytest.approx(1.0)


def test_compute_iou_disjoint_boxes():
    """Disjoint (non-overlapping) boxes must yield IoU = 0.0."""
    box_a = (0, 0, 10, 10)
    box_b = (20, 20, 10, 10)
    assert compute_iou(box_a, box_b) == pytest.approx(0.0)


def test_compute_iou_partial_overlap():
    """Verify exact mathematical IoU on partial overlap."""
    # Box A: [0, 0] to [10, 10], Area = 100
    # Box B: [5, 0] to [15, 10], Area = 100
    # Intersection: [5, 0] to [10, 10], Area = 5 * 10 = 50
    # Union: 100 + 100 - 50 = 150
    # IoU: 50 / 150 = 1/3 ~ 0.333333
    box_a = (0, 0, 10, 10)
    box_b = (5, 0, 10, 10)
    assert compute_iou(box_a, box_b) == pytest.approx(50.0 / 150.0)


def test_compute_iou_zero_area_boxes():
    """Boxes with zero or negative width/height must yield IoU = 0.0."""
    assert compute_iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0
    assert compute_iou((0, 0, 10, 10), (0, 0, 10, -5)) == 0.0


def test_compute_iou_boundingbox_object_support():
    """Verify BoundingBox dataclass instances are accepted."""
    box_a = BoundingBox(x=10, y=10, width=40, height=40)
    box_b = BoundingBox(x=10, y=10, width=40, height=40)
    assert compute_iou(box_a, box_b) == pytest.approx(1.0)


# =========================================================================
# 2. Greedy Matching Logic Tests
# =========================================================================

def test_match_detections_perfect_match():
    """All predictions match ground truth boxes with high IoU."""
    preds = [
        FaceDetection(bbox=BoundingBox(10, 10, 40, 40), confidence=0.9),
        FaceDetection(bbox=BoundingBox(100, 100, 50, 50), confidence=0.8),
    ]
    gt = [
        (10, 10, 40, 40),
        (100, 100, 50, 50),
    ]
    tp, fp, fn, ious = match_detections_to_ground_truth(preds, gt, iou_threshold=0.50)
    assert tp == 2
    assert fp == 0
    assert fn == 0
    assert len(ious) == 2
    assert ious == pytest.approx([1.0, 1.0])


def test_match_detections_no_faces_in_gt_with_predictions():
    """When GT has 0 faces, all predictions are False Positives."""
    preds = [
        FaceDetection(bbox=BoundingBox(10, 10, 40, 40), confidence=0.9),
    ]
    tp, fp, fn, ious = match_detections_to_ground_truth(preds, [], iou_threshold=0.50)
    assert tp == 0
    assert fp == 1
    assert fn == 0
    assert len(ious) == 0


def test_match_detections_missed_gt():
    """When no predictions exist, all GT boxes are False Negatives."""
    gt = [(10, 10, 40, 40), (60, 60, 30, 30)]
    tp, fp, fn, ious = match_detections_to_ground_truth([], gt, iou_threshold=0.50)
    assert tp == 0
    assert fp == 0
    assert fn == 2
    assert len(ious) == 0


def test_match_detections_duplicate_rejection():
    """Duplicate predictions matching the same GT box must be counted as False Positives."""
    preds = [
        FaceDetection(bbox=BoundingBox(10, 10, 40, 40), confidence=0.95),  # Matches GT
        FaceDetection(bbox=BoundingBox(12, 12, 38, 38), confidence=0.85),  # Duplicate on same GT!
    ]
    gt = [(10, 10, 40, 40)]
    tp, fp, fn, ious = match_detections_to_ground_truth(preds, gt, iou_threshold=0.50)
    assert tp == 1
    assert fp == 1
    assert fn == 0
    assert len(ious) == 1


# =========================================================================
# 3. Metric Calculation Tests
# =========================================================================

def test_calculate_metrics_edge_cases():
    """Test boundary conditions for calculate_metrics."""
    # Zero detections, zero GT (e.g. Clean no-face image)
    clean = calculate_metrics(tp=0, fp=0, fn=0, matched_ious=[])
    assert clean["precision"] == 1.0
    assert clean["recall"] == 1.0
    assert clean["f1"] == 1.0
    assert clean["mean_iou"] == 0.0

    # 100% Recall, 50% Precision
    partial = calculate_metrics(tp=5, fp=5, fn=0, matched_ious=[0.8, 0.8, 0.8, 0.8, 0.8])
    assert partial["precision"] == 0.50
    assert partial["recall"] == 1.0
    assert partial["f1"] == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0), abs=1e-3)
    assert partial["mean_iou"] == pytest.approx(0.8)


# =========================================================================
# 4. Manifest Integrity Tests
# =========================================================================

def test_manifest_schema_and_boundaries():
    """Verify that all manifest entries conform to schema and bounding boxes are inside image bounds."""
    manifest_path = Path(__file__).resolve().parent.parent / "validation" / "manifest.json"
    assert manifest_path.exists(), "validation/manifest.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "images" in data
    assert len(data["images"]) >= 10

    valid_scenarios = {
        "no_faces",
        "one_face",
        "multiple_faces",
        "small_and_distant_faces",
        "faces_near_borders",
        "different_lighting",
        "different_head_angles",
        "partially_occluded",
        "diverse_appearances",
    }

    seen_scenarios = set()

    for img in data["images"]:
        assert "id" in img
        assert "filename" in img
        assert "license" in img
        assert "width" in img and img["width"] > 0
        assert "height" in img and img["height"] > 0
        assert "scenarios" in img

        for sc in img["scenarios"]:
            assert sc in valid_scenarios, f"Unknown scenario tag '{sc}' in {img['id']}"
            seen_scenarios.add(sc)

        w, h = img["width"], img["height"]
        for face in img.get("faces", []):
            bx, by, bw, bh = face["bbox"]
            assert bx >= 0 and bx < w, f"BBox x={bx} exceeds image width {w} in {img['id']}"
            assert by >= 0 and by < h, f"BBox y={by} exceeds image height {h} in {img['id']}"
            assert bw > 0 and bx + bw <= w + 5  # Allow minor border clamping tolerance
            assert bh > 0 and by + bh <= h + 5

    # All required scenario categories must be covered
    for req_sc in valid_scenarios:
        assert req_sc in seen_scenarios, f"Required scenario category '{req_sc}' not covered in manifest"


# =========================================================================
# 5. End-to-End Validation Run Test
# =========================================================================

def test_evaluate_backend_yunet_smoke():
    """Verify evaluate_backend runs successfully against validation manifest."""
    manifest_path = Path(__file__).resolve().parent.parent / "validation" / "manifest.json"
    data_dir = Path(__file__).resolve().parent.parent / "validation" / "data"

    if not data_dir.exists():
        pytest.skip("validation/data not yet populated")

    results = evaluate_backend(
        backend="yunet",
        confidence_threshold=0.70,
        manifest_path=manifest_path,
        data_dir=data_dir,
    )

    assert "summary" in results
    assert "scenario_breakdown" in results
    assert results["summary"]["total_images"] >= 10
    assert results["summary"]["precision"] >= 0.90
    assert results["summary"]["recall"] >= 0.90
