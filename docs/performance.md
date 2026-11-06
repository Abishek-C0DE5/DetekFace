# Performance, Latency & Asynchronous Architecture

`detekface` 0.3.0 introduces an asynchronous multi-threaded engine designed to deliver silky-smooth 60 FPS previews while isolating model inference latency.

---

## 🏎️ Asynchronous Architecture Overview

In traditional computer-vision scripts (such as DetekFace 0.2.x), the webcam preview runs synchronously in a single thread:

```
[Camera Hardware] ---> [Read Frame] ---> [Model Inference (30-60ms)] ---> [Render UI] ---> [cv2.imshow]
                         ^                                                                     |
                         +--------------------- UI Stalls Here -------------------------------+
```

If inference takes 45ms, the maximum achievable preview rate is capped at $\frac{1}{0.045} \approx 22\text{ FPS}$, causing perceptible stutter, mouse lag, and dropped frames.

### The 0.3.0 Decoupled Pipeline (`AsyncCameraEngine`)

DetekFace 0.3.0 decouples processing into 4 independent concurrent stages:

```
+-----------------------------------------------------------------------------------------+
|                                  AsyncCameraEngine                                      |
|                                                                                         |
|  [Hardware Stream]                                                                      |
|          |                                                                              |
|          v                                                                              |
|  (Thread 1: Capture) ---> [Timestamped Frame Slot] ------------------------+            |
|                                     |                                      |            |
|                                     v (Single-slot queue; drops stale)     v            |
|                           (Thread 2: Inference)                    (Thread 3: Preview)  |
|                                     |                                      |            |
|                                     v                                      v            |
|                           [Latest Detection Result] ------------> [60 FPS Render Loop]  |
|                                                                            |            |
|                                                                            v            |
|                                                                       [cv2.imshow]      |
+-----------------------------------------------------------------------------------------+
```

1. **Frame Acquisition Thread**: Continuously captures fresh hardware frames with high-resolution wall-clock timestamps.
2. **Background Inference Thread**: Pulls the newest frame from a bounded single-slot queue, executing face detection, dense landmarking, and tracking in the background.
3. **Temporal Tracking & Smoothing**: Smooths landmark coordinates, head pose angles, and tracking identities across inference latency gaps.
4. **Preview Rendering Loop**: Operates at the display monitor refresh rate (targeting 60 FPS), overlaying the newest available detection onto the freshest acquired frame without stalling.

---

## 📊 Three Key Metrics: Capture vs. Inference vs. Render FPS

DetekFace measures three distinct framerates independently:

| Metric | Property | Typical Range | Description |
| :--- | :--- | :---: | :--- |
| **Capture FPS** | `metrics.capture_fps` | 30–60 FPS | Rate at which the physical camera sensor delivers raw image frames. |
| **Inference FPS** | `metrics.inference_fps` | 25–45 FPS | Throughput of the deep learning model (YuNet / MediaPipe) evaluating frames. |
| **Render FPS** | `metrics.render_fps` | 55–60 FPS | Real-time display framerate presented to the user on screen. |

---

## ⏱️ Stale Result Timeout Suppression

If camera inference lags or stalls (e.g. during heavy system load or when turning away), repeatedly displaying old detections causes outdated bounding boxes to float over background objects.

DetekFace enforces `stale_timeout_sec` (default `0.5` seconds):
- If the newest available detection is fresher than `stale_timeout_sec`, it is rendered seamlessly.
- If inference exceeds `stale_timeout_sec`, stale detections are automatically suppressed (cleared) until fresh inference completes, preventing visual ghosting.

---

## 📹 Physical Camera Hardware Constraints

While DetekFace's rendering pipeline easily achieves 60 FPS on modern CPUs, actual video smoothness is fundamentally bounded by the physical camera hardware and OS driver backend:

1. **DirectShow (`cv2.CAP_DSHOW` on Windows)**:
   - Many consumer USB webcams negotiate 30 FPS or 15–20 FPS under standard lighting, and drop to ~8–15 FPS in low-light environments due to automatic exposure time extension.
2. **Microsoft Media Foundation (`cv2.CAP_MSMF`)**:
   - Offers higher throughput for high-resolution 1080p/4K webcams, but driver latency varies.
3. **Bandwidth Limitations**:
   - Multiple webcams sharing a single USB 2.0 root hub often drop capture rates due to uncompressed YUY2 video bandwidth bottlenecks. Prefer MJPG camera streams where supported.

---

## ⚙️ Hardware Acceleration Options

| Backend | Execution Provider | Status | Performance Profile |
| :--- | :--- | :---: | :--- |
| **OpenCV CPU** | Multi-threaded CPU BLAS | **Default** | Highly portable, ~8–15ms per frame on modern x86_64/ARM CPUs. |
| **OpenVINO** | Intel CPU / iGPU / NPU | Optional | Accelerated inference on Intel 11th Gen+ hardware via custom OpenCV builds. |
| **CUDA / cuDNN** | NVIDIA GPU | Optional | Requires custom `opencv-python` build compiled with CUDA support. |

---

## 🔬 Benchmark Methodology

DetekFace includes an automated, reproducible benchmark suite (`tools/audit_milestone9.py`):
- Measures 10 feature scenarios under identical conditions (fixed frame count, standardized 640×480 resolution, and controlled mock hardware).
- Captures independent wall-clock timestamps (`time.perf_counter()`), P50 and P95 latencies, queue drop counts, process CPU utilization, and working set memory (RSS) via Win32 `GetProcessMemoryInfo`.
- **Zero Fabrication Guarantee**: DetekFace strictly reports measured empirical timings without interpolation or simulated multipliers.

