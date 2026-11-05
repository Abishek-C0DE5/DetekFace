"""Live Browser-Based Face & Gesture Detection Visualizer for detekface.

Streams real-time webcam frames with facial tracking, 3D head pose, and gesture
events directly to your web browser (http://localhost:5050).
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
from typing import List
import cv2

import detekface

recent_events: List[dict] = []
events_lock = threading.Lock()
latest_jpeg: bytes = b""
frame_lock = threading.Lock()
is_running = True
current_accessory = "sunglasses"


def on_gesture(event: detekface.GestureEvent) -> None:
    """Record gesture events for real-time web UI display."""
    with events_lock:
        recent_events.append(
            {
                "name": event.name,
                "track_id": event.track_id,
                "duration": round(event.duration, 2),
                "timestamp": time.strftime("%H:%M:%S"),
            }
        )
        if len(recent_events) > 25:
            recent_events.pop(0)
    print(f"[LIVE GESTURE] {event.name.upper()} | Face ID: {event.track_id} | {event.duration:.2f}s")


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>detekface 0.3.0 - Live Face & Gesture Visualizer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: #0d1117; color: #c9d1d9; padding: 20px; display: flex; flex-direction: column; align-items: center; }
        header { text-align: center; margin-bottom: 16px; }
        h1 { color: #58a6ff; font-size: 24px; margin-bottom: 4px; }
        p.subtitle { color: #8b949e; font-size: 13px; }
        .main-container { display: flex; gap: 20px; max-width: 1100px; width: 100%; justify-content: center; flex-wrap: wrap; }
        .video-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,0.5); min-width: 640px; min-height: 480px; display: flex; align-items: center; justify-content: center; }
        .video-card img { display: block; max-width: 100%; border-radius: 8px; }
        .sidebar { width: 340px; display: flex; flex-direction: column; gap: 16px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px; }
        .card h2 { font-size: 15px; color: #58a6ff; margin-bottom: 10px; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
        .guide-item { margin-bottom: 8px; font-size: 13px; line-height: 1.5; }
        .guide-item strong { color: #00ff88; }
        .btn-group { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
        .acc-btn { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 500; transition: all 0.2s; }
        .acc-btn:hover { background: #30363d; color: #58a6ff; border-color: #58a6ff; }
        .acc-btn.active { background: #1f6feb; color: white; border-color: #58a6ff; }
        #event-log { max-height: 240px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; }
        .event-item { background: #21262d; border-left: 4px solid #00ff88; padding: 6px 10px; border-radius: 4px; font-size: 12px; }
        .event-item .time { color: #8b949e; font-size: 11px; }
        .event-item .name { color: #fff; font-weight: bold; }
        .status-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; background: #238636; color: white; margin-bottom: 8px; font-weight: bold; }
    </style>
</head>
<body>
    <header>
        <span class="status-badge">&#9679; LIVE CAMERA CONNECTED</span>
        <h1>detekface 0.3.0 Live Visualizer</h1>
        <p class="subtitle">Multi-Face Tracking &bull; 3D Head Pose &bull; Real-Time Gesture Detection</p>
    </header>
    <div class="main-container">
        <div class="video-card">
            <img id="cam-view" src="/frame" alt="Waiting for webcam feed..." width="640" height="480">
        </div>
        <div class="sidebar">
            <div class="card">
                <h2>AR Face Accessories (Milestone 5)</h2>
                <div class="btn-group">
                    <button class="acc-btn active" onclick="setAccessory(this, 'sunglasses')">🕶️ Sunglasses</button>
                    <button class="acc-btn" onclick="setAccessory(this, 'crown')">👑 Crown</button>
                    <button class="acc-btn" onclick="setAccessory(this, 'face_mask')">😷 Face Mask</button>
                    <button class="acc-btn" onclick="setAccessory(this, 'futuristic_visor')">🥽 Visor</button>
                    <button class="acc-btn" onclick="setAccessory(this, 'halo')">😇 Halo</button>
                    <button class="acc-btn" onclick="setAccessory(this, 'none')">❌ None</button>
                </div>
            </div>
            <div class="card">
                <h2>Live Gesture Activity</h2>
                <div id="event-log">
                    <div style="color: #8b949e; font-size: 12px;">Waiting for gesture actions...</div>
                </div>
            </div>
            <div class="card">
                <h2>Observable Actions</h2>
                <div class="guide-item"><strong>&bull; Wink / Blink:</strong> Close left, right, or both eyes.</div>
                <div class="guide-item"><strong>&bull; Mouth Opening:</strong> Open your mouth naturally.</div>
                <div class="guide-item"><strong>&bull; Head Turn:</strong> Turn head left or right.</div>
                <div class="guide-item"><strong>&bull; Head Tilt:</strong> Tilt head sideways toward shoulder.</div>
            </div>
        </div>
    </div>
    <script>
        const camImg = document.getElementById('cam-view');
        function streamFrames() {
            const next = new Image();
            next.onload = function() {
                camImg.src = next.src;
                requestAnimationFrame(streamFrames);
            };
            next.onerror = function() {
                setTimeout(streamFrames, 80);
            };
            next.src = '/frame?t=' + Date.now();
        }
        streamFrames();

        function setAccessory(btn, name) {
            document.querySelectorAll('.acc-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            fetch('/accessory?name=' + encodeURIComponent(name));
        }

        async function updateLog() {
            try {
                const res = await fetch('/events');
                const data = await res.json();
                const logEl = document.getElementById('event-log');
                if (data.length > 0) {
                    logEl.innerHTML = data.slice().reverse().map(e => `
                        <div class="event-item">
                            <span class="time">${e.timestamp} (Face ${e.track_id ?? 1})</span><br>
                            <span class="name">&bull; ${e.name.toUpperCase()}</span> (${e.duration}s)
                        </div>
                    `).join('');
                }
            } catch (err) {}
        }
        setInterval(updateLog, 300);
    </script>
</body>
</html>
"""


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            content = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/events":
            with events_lock:
                data = list(recent_events)
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/accessory"):
            global current_accessory
            if "name=" in self.path:
                name = self.path.split("name=")[1].split("&")[0].strip()
                current_accessory = None if name in ("none", "") else name
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        elif self.path.startswith("/frame"):
            with frame_lock:
                frame_bytes = latest_jpeg
            if frame_bytes:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame_bytes)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(frame_bytes)
            else:
                self.send_error(503, "Frame not ready")
        else:
            self.send_error(404)

    def log_message(self, format: str, *args) -> None:
        pass  # Suppress HTTP access noise in stdout


def webcam_worker() -> None:
    """Continuously captures frames from webcam and runs detekface pipeline."""
    global latest_jpeg, is_running
    try:
        session = detekface.open_camera(
            camera_index=0,
            mirror=True,
            track_faces=True,
            estimate_pose=True,
            target_fps=60.0,
            async_mode=True,
        )
        print("[detekface] Webcam hardware initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Could not initialize camera: {e}")
        is_running = False
        return

    with session:
        while is_running:
            try:
                _ = session.read()
                annotated = session.render(
                    shape="corners",
                    color="#00FF88",
                    show_confidence=True,
                    show_fps=True,
                    show_count=True,
                    effect=current_accessory,
                )
                success, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if success:
                    with frame_lock:
                        latest_jpeg = encoded.tobytes()
                time.sleep(0.002)
            except Exception as e:
                print(f"[WARNING] Frame error: {e}")
                time.sleep(0.05)


def main() -> None:
    port = 5050
    server = ThreadingHTTPServer(("0.0.0.0", port), StreamingHandler)
    server.daemon_threads = True

    cam_thread = threading.Thread(target=webcam_worker, daemon=True)
    cam_thread.start()

    url = f"http://localhost:{port}"
    print(f"\n{'=' * 65}")
    print(f" LIVE STREAM READY: {url}")
    print(f"{'=' * 65}")
    print(f"Opening {url} in your browser so you can see your face live...")
    os.system(f"start {url}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping live visualizer...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
