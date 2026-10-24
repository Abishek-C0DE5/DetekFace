"""Reproducible quality validation suite for DetekFace.

Measures face detector precision, recall, F1-score, and Intersection over Union (IoU)
against annotated ground-truth test cases across specific real-world challenges:
- No faces (false positive suppression)
- One face & multiple faces
- Small and distant faces
- Faces near image borders
- Different lighting conditions (low-light, harsh shadow)
- Different head angles
- Partially occluded faces
- Diverse face appearances

Supports evaluating multiple detection backends (YuNet, MediaPipe) on identical samples,
reporting confidence thresholds and resolution settings without claiming wrapper code
enhances underlying neural network weights.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2

import detekface
from detekface.types import BoundingBox, FaceDetection


# Default paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = BASE_DIR / "manifest.json"
DEFAULT_DATA_DIR = BASE_DIR / "data"


def compute_iou(
    box_a: Union[BoundingBox, Tuple[int, int, int, int], Sequence[int]],
    box_b: Union[BoundingBox, Tuple[int, int, int, int], Sequence[int]],
) -> float:
    """Compute the Intersection over Union (IoU) of two bounding boxes in (x, y, w, h) format.

    Returns:
        float: IoU value between 0.0 (disjoint) and 1.0 (identical).
    """
    if isinstance(box_a, BoundingBox):
        ax, ay, aw, ah = box_a.x, box_a.y, box_a.width, box_a.height
    else:
        ax, ay, aw, ah = int(box_a[0]), int(box_a[1]), int(box_a[2]), int(box_a[3])

    if isinstance(box_b, BoundingBox):
        bx, by, bw, bh = box_b.x, box_b.y, box_b.width, box_b.height
    else:
        bx, by, bw, bh = int(box_b[0]), int(box_b[1]), int(box_b[2]), int(box_b[3])

    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    area_inter = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    area_union = area_a + area_b - area_inter

    if area_union <= 0:
        return 0.0

    return float(area_inter) / float(area_union)


@dataclass
class MatchResult:
    """Matching evaluation result for a single image."""

    image_id: str
    ground_truth_count: int
    detection_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_ious: List[float]
    latency_ms: float
    scenarios: List[str]


def match_detections_to_ground_truth(
    predictions: Sequence[FaceDetection],
    ground_truth_boxes: Sequence[Union[Tuple[int, int, int, int], Sequence[int]]],
    iou_threshold: float = 0.50,
) -> Tuple[int, int, int, List[float]]:
    """Perform greedy bipartite matching between predictions and ground-truth boxes.

    Args:
        predictions: Detected FaceDetection instances.
        ground_truth_boxes: List of ground-truth boxes in (x, y, w, h) format.
        iou_threshold: Minimum IoU required to declare a True Positive match.

    Returns:
        Tuple of (true_positives, false_positives, false_negatives, list_of_matched_ious).
    """
    if len(ground_truth_boxes) == 0:
        # Every prediction is a False Positive
        return 0, len(predictions), 0, []

    if len(predictions) == 0:
        # Every ground-truth box is a False Negative
        return 0, 0, len(ground_truth_boxes), []

    # Sort predictions by confidence score descending
    sorted_preds = sorted(predictions, key=lambda f: f.confidence, reverse=True)

    matched_gt_indices = set()
    matched_ious: List[float] = []
    tp = 0
    fp = 0

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            if gt_idx in matched_gt_indices:
                continue

            iou = compute_iou(pred.bbox, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            tp += 1
            matched_gt_indices.add(best_gt_idx)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(ground_truth_boxes) - len(matched_gt_indices)
    return tp, fp, fn, matched_ious


def calculate_metrics(tp: int, fp: int, fn: int, matched_ious: Sequence[float]) -> Dict[str, float]:
    """Calculate Precision, Recall, F1-Score, and Mean IoU from TP, FP, and FN counts."""
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)

    if (precision + recall) > 0:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    mean_iou = statistics.mean(matched_ious) if len(matched_ious) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
    }


def evaluate_backend(
    backend: str = "yunet",
    confidence_threshold: float = 0.70,
    iou_threshold: float = 0.50,
    input_size: Optional[Tuple[int, int]] = None,
    manifest_path: Path = DEFAULT_MANIFEST,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Dict[str, Any]:
    """Execute evaluation for a specified backend across the validation manifest.

    Args:
        backend: Detection backend ('yunet' or 'mediapipe').
        confidence_threshold: Minimum confidence threshold.
        iou_threshold: Minimum IoU for True Positive matching.
        input_size: Optional fixed inference resolution.
        manifest_path: Path to manifest.json.
        data_dir: Directory containing validation images.

    Returns:
        Dict containing hardware info, parameters, per-image results, summary metrics,
        and per-scenario breakdowns.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Validation manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Initialize FaceDetector with requested backend and thresholds
    detector = detekface.FaceDetector(
        backend=backend,
        confidence_threshold=confidence_threshold,
        input_size=input_size,
    )

    image_results: List[MatchResult] = []
    scenarios_data: Dict[str, Dict[str, Any]] = {}

    for item in manifest.get("images", []):
        img_id = item["id"]
        filename = item["filename"]
        scenarios = item.get("scenarios", [])
        gt_boxes = [face["bbox"] for face in item.get("faces", [])]

        img_path = data_dir / filename
        if not img_path.exists():
            # Check tests/assets fallback if filename matches
            fallback = BASE_DIR.parent / "tests" / "assets" / filename
            if fallback.exists():
                img_path = fallback
            else:
                raise FileNotFoundError(
                    f"Validation image '{filename}' not found in {data_dir}. "
                    "Run 'python validation/download_dataset.py' to prepare the validation dataset."
                )

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not read image file: {img_path}")

        # Measure detection latency
        t0 = time.perf_counter()
        result = detector.detect(img)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        tp, fp, fn, matched_ious = match_detections_to_ground_truth(
            predictions=result.faces,
            ground_truth_boxes=gt_boxes,
            iou_threshold=iou_threshold,
        )

        match_res = MatchResult(
            image_id=img_id,
            ground_truth_count=len(gt_boxes),
            detection_count=result.count,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            matched_ious=matched_ious,
            latency_ms=latency_ms,
            scenarios=scenarios,
        )
        image_results.append(match_res)

        # Aggregate per-scenario counts
        for sc in scenarios:
            if sc not in scenarios_data:
                scenarios_data[sc] = {"tp": 0, "fp": 0, "fn": 0, "ious": [], "count": 0}
            scenarios_data[sc]["tp"] += tp
            scenarios_data[sc]["fp"] += fp
            scenarios_data[sc]["fn"] += fn
            scenarios_data[sc]["ious"].extend(matched_ious)
            scenarios_data[sc]["count"] += 1

    # Compute overall aggregate metrics
    total_tp = sum(r.true_positives for r in image_results)
    total_fp = sum(r.false_positives for r in image_results)
    total_fn = sum(r.false_negatives for r in image_results)
    all_ious = [iou for r in image_results for iou in r.matched_ious]
    latencies = [r.latency_ms for r in image_results]

    summary = calculate_metrics(total_tp, total_fp, total_fn, all_ious)
    summary["mean_latency_ms"] = round(statistics.mean(latencies), 2)
    summary["median_latency_ms"] = round(statistics.median(latencies), 2)
    summary["total_images"] = len(image_results)
    summary["total_ground_truth_faces"] = sum(r.ground_truth_count for r in image_results)
    summary["total_predicted_faces"] = sum(r.detection_count for r in image_results)

    # Compute scenario breakdown
    scenario_breakdown: Dict[str, Dict[str, Any]] = {}
    for sc, data in sorted(scenarios_data.items()):
        sc_metrics = calculate_metrics(data["tp"], data["fp"], data["fn"], data["ious"])
        sc_metrics["samples_evaluated"] = data["count"]
        scenario_breakdown[sc] = sc_metrics

    # Environment information
    env_info = {
        "os": platform.platform(),
        "cpu": platform.processor() or "Unknown",
        "python_version": sys.version.split()[0],
        "opencv_version": cv2.__version__,
        "backend": backend,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "input_size": input_size,
    }

    return {
        "environment": env_info,
        "summary": summary,
        "scenario_breakdown": scenario_breakdown,
        "image_results": [asdict(r) for r in image_results],
    }


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """Print a clean, formatted ASCII terminal report of the validation run."""
    env = results["environment"]
    summary = results["summary"]
    scenarios = results["scenario_breakdown"]
    images = results["image_results"]

    print("=" * 76)
    print(" DetekFace Reproducible Quality Validation Suite Report")
    print("=" * 76)
    print(f"Host System        : {env['os']}")
    print(f"CPU Architecture   : {env['cpu']}")
    print(f"Runtime            : Python {env['python_version']} | OpenCV {env['opencv_version']}")
    print(f"Detection Backend  : {env['backend'].upper()}")
    print(f"Confidence Thresh  : {env['confidence_threshold']}")
    print(f"Matching IoU Thresh: {env['iou_threshold']}")
    print(f"Input Size Mode    : {env['input_size'] if env['input_size'] else 'Native (Full Resolution)'}")
    print("-" * 76)

    print("\n--- Per-Sample Evaluation Details ---")
    header = f"{'Sample ID':<30} | {'GT':<3} | {'Det':<3} | {'TP':<3} | {'FP':<3} | {'FN':<3} | {'Max IoU':<7} | {'Time (ms)'}"
    print(header)
    print("-" * len(header))
    for r in images:
        max_iou_str = f"{max(r['matched_ious']):.3f}" if r["matched_ious"] else "---"
        print(
            f"{r['image_id']:<30} | {r['ground_truth_count']:<3} | {r['detection_count']:<3} | "
            f"{r['true_positives']:<3} | {r['false_positives']:<3} | {r['false_negatives']:<3} | "
            f"{max_iou_str:<7} | {r['latency_ms']:>6.2f} ms"
        )

    print("\n--- Summary Performance Metrics ---")
    print(f"Total Evaluated Images      : {summary['total_images']}")
    print(f"Total Ground-Truth Faces    : {summary['total_ground_truth_faces']}")
    print(f"Total Predicted Faces       : {summary['total_predicted_faces']}")
    print(f"True Positives (TP)         : {summary['true_positives']}")
    print(f"False Positives (FP)        : {summary['false_positives']}")
    print(f"False Negatives (FN)        : {summary['false_negatives']}")
    print(f"Precision                   : {summary['precision']:.4f} ({summary['precision'] * 100:.1f}%)")
    print(f"Recall                      : {summary['recall']:.4f} ({summary['recall'] * 100:.1f}%)")
    print(f"F1-Score                    : {summary['f1']:.4f}")
    print(f"Mean IoU (on matched faces) : {summary['mean_iou']:.4f}")
    print(f"Average Latency per image   : {summary['mean_latency_ms']} ms")

    print("\n--- Scenario-Specific Metrics Breakdown ---")
    sc_header = f"{'Scenario Tag':<28} | {'N':<3} | {'TP':<3} | {'FP':<3} | {'FN':<3} | {'Precision':<9} | {'Recall':<7} | {'F1':<6} | {'Mean IoU'}"
    print(sc_header)
    print("-" * len(sc_header))
    for sc_name, sc_data in scenarios.items():
        print(
            f"{sc_name:<28} | {sc_data['samples_evaluated']:<3} | {sc_data['true_positives']:<3} | "
            f"{sc_data['false_positives']:<3} | {sc_data['false_negatives']:<3} | "
            f"{sc_data['precision']:<9.3f} | {sc_data['recall']:<7.3f} | {sc_data['f1']:<6.3f} | {sc_data['mean_iou']:.3f}"
        )
    print("=" * 76)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DetekFace validation suite.")
    parser.add_argument(
        "--backend",
        choices=["yunet", "mediapipe"],
        default="yunet",
        help="Detection backend to evaluate",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.70,
        help="Confidence threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="IoU threshold for True Positive (default 0.50)",
    )
    parser.add_argument(
        "--input-size",
        type=str,
        default=None,
        help="Optional inference resolution (e.g. 640x480)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to manifest.json")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to images directory",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both YuNet and MediaPipe backends side-by-side",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional output JSON path for structured results",
    )
    args = parser.parse_args()

    input_size_tuple = None
    if args.input_size:
        parts = args.input_size.lower().replace(",", "x").split("x")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            input_size_tuple = (int(parts[0]), int(parts[1]))
        else:
            print(
                f"Error: Invalid --input-size format '{args.input_size}'. Use format like '640x480'.",
                file=sys.stderr,
            )
            return 1

    backends_to_run = ["yunet", "mediapipe"] if args.compare else [args.backend]
    all_runs = {}

    for b in backends_to_run:
        try:
            res = evaluate_backend(
                backend=b,
                confidence_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
                input_size=input_size_tuple,
                manifest_path=args.manifest,
                data_dir=args.data_dir,
            )
            print_evaluation_report(res)
            all_runs[b] = res
        except Exception as e:
            print(f"\n[Error] Failed to evaluate backend '{b}': {e}", file=sys.stderr)
            if not args.compare:
                return 1

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(all_runs, f, indent=2)
        print(f"\n[Exported] Detailed validation results saved to: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
