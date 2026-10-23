"""Repeatable performance benchmarking and quality validation for detekface.

Measures actual hardware specs, inference latency, throughput FPS across multiple
image resolutions (320x240, 640x480, 1280x720), and confidence threshold sensitivity.
"""

import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np

import detekface


def get_hardware_info() -> Dict[str, str]:
    """Inspect and return host system hardware and runtime environment details."""
    return {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "opencv_version": cv2.__version__,
        "cpu_processor": platform.processor() or "Unknown",
        "cpu_count": str(os.cpu_count() or "Unknown"),
    }


def run_latency_benchmark(
    detector: detekface.FaceDetector,
    resolution: Tuple[int, int],
    warmup_iters: int = 5,
    benchmark_iters: int = 50,
) -> Dict[str, float]:
    """Measure inference latency and throughput for a specific image resolution."""
    w, h = resolution

    # Create synthetic test frame with realistic structure
    base_asset = Path(__file__).parent.parent / "tests" / "assets" / "sample_face.jpg"
    if base_asset.exists():
        raw_img = cv2.imread(str(base_asset))
        test_frame = cv2.resize(raw_img, (w, h))
    else:
        test_frame = np.full((h, w, 3), 128, dtype=np.uint8)

    # Warmup iterations to prime ONNX engine and CPU caches
    for _ in range(warmup_iters):
        detector.detect(test_frame)

    latencies_ms: List[float] = []
    for _ in range(benchmark_iters):
        t0 = time.perf_counter()
        detector.detect(test_frame)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    stdev_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    p95_ms = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "width": w,
        "height": h,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "p95_ms": p95_ms,
        "stdev_ms": stdev_ms,
        "fps": fps,
    }


def run_threshold_sweep(detector: detekface.FaceDetector, asset_path: Path) -> List[Dict[str, object]]:
    """Test confidence threshold sensitivity from 0.3 to 0.9 on sample asset."""
    img = cv2.imread(str(asset_path))
    thresholds = [0.30, 0.50, 0.70, 0.85, 0.90, 0.95]
    results = []

    for thresh in thresholds:
        detector_thresh = detekface.FaceDetector(confidence_threshold=thresh)
        res = detector_thresh.detect(img)
        confidences = [round(f.confidence, 4) for f in res.faces]
        results.append({
            "threshold": thresh,
            "count": res.count,
            "confidences": confidences,
        })

    return results


def main() -> None:
    print("=" * 70)
    print(" DetekFace Milestone 10: Performance & Reliability Benchmark")
    print("=" * 70)

    hw = get_hardware_info()
    print("\n--- System & Runtime Environment ---")
    print(f"OS              : {hw['os']}")
    print(f"CPU Processor   : {hw['cpu_processor']}")
    print(f"Logical CPUs    : {hw['cpu_count']}")
    print(f"Python Runtime  : {hw['python_version']}")
    print(f"OpenCV Version  : {hw['opencv_version']}")

    detector = detekface.FaceDetector()
    print(f"Resolved Model  : {detector.model_path.name}")

    resolutions = [
        (320, 240),   # QVGA
        (640, 480),   # VGA / Standard Webcam
        (1280, 720),  # 720p HD
    ]

    print("\n--- Inference Throughput Benchmarking (50 iterations per resolution) ---")
    print(f"{'Resolution':<12} | {'Mean (ms)':<10} | {'Median (ms)':<12} | {'P95 (ms)':<10} | {'FPS':<8}")
    print("-" * 62)

    for res in resolutions:
        metrics = run_latency_benchmark(detector, res, warmup_iters=5, benchmark_iters=50)
        res_str = f"{metrics['width']}x{metrics['height']}"
        print(
            f"{res_str:<12} | "
            f"{metrics['mean_ms']:<10.2f} | "
            f"{metrics['median_ms']:<12.2f} | "
            f"{metrics['p95_ms']:<10.2f} | "
            f"{metrics['fps']:<8.1f}"
        )

    # Threshold sensitivity sweep
    asset_path = Path(__file__).parent.parent / "tests" / "assets" / "sample_face.jpg"
    if asset_path.exists():
        print("\n--- Confidence Threshold Sensitivity Analysis ---")
        sweep_data = run_threshold_sweep(detector, asset_path)
        print(f"{'Threshold':<12} | {'Faces Detected':<15} | {'Detected Confidences'}")
        print("-" * 60)
        for row in sweep_data:
            print(f"{row['threshold']:<12.2f} | {row['count']:<15} | {row['confidences']}")

    print("\n" + "=" * 70)
    print(" Benchmark completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
