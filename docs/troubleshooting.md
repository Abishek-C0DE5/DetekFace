# Troubleshooting & FAQ

This guide provides actionable solutions to common issues encountered when installing, configuring, or running `detekface`.

---

## 1. Webcam & Hardware Issues

### `CameraNotFoundError: Cannot open camera with index 0`

#### Potential Causes & Solutions:
1. **Camera in Use by Another Application**:
   - Webcams can only be accessed by one application at a time on Windows. Close applications like Zoom, Microsoft Teams, Skype, OBS Studio, or browser tabs that might be using the camera.
2. **Windows 11 Privacy Settings**:
   - Windows 11 blocks camera access by default if permission is disabled.
   - Go to: **Settings** -> **Privacy & security** -> **Camera**.
   - Ensure **Camera access** is toggled **On**.
   - Scroll down and ensure **Let desktop apps access your camera** is toggled **On**.
3. **Wrong Camera Index**:
   - If using an external USB webcam alongside an integrated laptop webcam, the device index might be `1` or `2`:
     ```python
     import detekface
     detekface.camera(camera_index=1)
     ```
4. **Physical Shutter or Hardware Switch**:
   - Check if your laptop or external webcam has a physical privacy slider or a dedicated keyboard function key (e.g., F8 or F10) that disables camera power.

---

### `CameraReadError: Failed to read video frame from camera`

- **Cause**: OpenCV successfully connected to the device handle, but reading image data returned `None`.
- **Solution**:
  - Test plugging the webcam into a different USB port (preferably directly into motherboard or USB 3.0 port rather than an unpowered hub).
  - Update the webcam manufacturer's drivers in Windows Device Manager.

---

### Preview Window Shows a Black or Blank Screen
- **Cause**: The camera sensor may need several initial frames to calibrate exposure and white balance.
- **Solution**: Allow the preview to run for 2-3 seconds. If the issue persists, ensure sufficient room lighting.

---

## 2. OpenCV GUI & Display Errors

### `cv2.error: The function is not implemented...`

#### Error Output:
```text
cv2.error: OpenCV(4.x.x) ... /window.cpp:1250: error: (-215:Assertion failed)
The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support.
If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake...
```

#### Cause:
You have `opencv-python-headless` installed in your Python environment instead of standard `opencv-python`. The headless build intentionally omits desktop GUI functions like `cv2.imshow()`, `cv2.waitKey()`, and `cv2.destroyAllWindows()`.

#### Solution:
Uninstall both packages to clear conflicts, then install the standard GUI build:
```powershell
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-python>=4.8
```

---

### Running in Docker, WSL2, or Headless Cloud Servers

If running on a remote Linux server or container without an X11 or Wayland display server:
- Avoid calling interactive functions (`camera()`, `result.show()`, or `video(show=True)`).
- Use `detect()` with `result.save()` or `video(show=False, output="annotated.mp4")`:
  ```python
  import detekface

  result = detekface.detect("input.jpg")
  result.save("output.jpg")  # Generates annotated image on disk without opening any windows
  ```

---

## 3. Model Download & Network Issues

### `ModelDownloadError: Failed to download official YuNet model`

#### Potential Causes:
- Corporate firewall, VPN, or proxy blocking downloads from GitHub.
- SSL inspection certificate errors.
- Air-gapped or offline development machine.

#### Offline / Manual Model Setup:
You can download the model weights once on any computer and provide them locally:
1. Download `face_detection_yunet_2023mar.onnx` (336 KB) directly from OpenCV Zoo:
   `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
2. Place the downloaded `.onnx` file in your application directory or the `detekface` cache directory:
   - **Windows**: `%LOCALAPPDATA%\detekface\face_detection_yunet_2023mar.onnx`
   - **Linux**: `~/.cache/detekface/face_detection_yunet_2023mar.onnx`
   - **macOS**: `~/Library/Caches/detekface/face_detection_yunet_2023mar.onnx`
3. Alternatively, supply the explicit local file path in Python:
   ```python
   import detekface
   result = detekface.detect("photo.jpg", model="path/to/face_detection_yunet_2023mar.onnx")
   ```

---

### `ModelCorruptionError: SHA256 checksum mismatch`

- **Cause**: An interrupted download or corrupted file exists in the cache folder.
- **Solution**: Delete the cached file and let `detekface` automatically re-download it.
  - On Windows: Run `Remove-Item "$env:LOCALAPPDATA\detekface\*.onnx"` in PowerShell.

---

## 4. Detection Accuracy & Quality

### Faces Are Not Being Detected (False Negatives)

1. **Face Resolution Too Small**:
   - Faces smaller than approximately 15x15 pixels may fail to trigger high-confidence detections. Use higher resolution images or move closer to the camera.
2. **Confidence Threshold Too High**:
   - The default threshold is `0.7` (70%). For difficult lighting, partial occlusions (e.g. masks, sunglasses), or profile angles, lower the threshold to `0.5` or `0.6`:
     ```python
     result = detekface.detect("dim_room.jpg", confidence_threshold=0.55)
     ```
3. **Severe Motion Blur or Extreme Angles**:
   - YuNet performs best on faces with yaw/pitch angles within ±60°. Ensure the subject is facing generally toward the lens without extreme motion blur.

---

### Non-Faces Detected as Faces (False Positives)

- **Solution**: Increase the confidence threshold to eliminate background noise:
  ```python
  result = detekface.detect("textured_background.jpg", confidence_threshold=0.85)
  ```

---

## 5. File & Path Formatting on Windows

### `ImageNotFoundError` or `VideoNotFoundError`

On Windows, unescaped backslashes in file paths can be misinterpreted by Python as escape characters (e.g., `\t` as a tab, `\n` as a newline).

#### Recommended Solutions:
Always use forward slashes `/`, Python raw string literals `r"..."`, or `pathlib.Path`:
```python
from pathlib import Path
import detekface

# Correct: Raw string
result = detekface.detect(r"C:\Users\Username\Pictures\photo.jpg")

# Correct: Forward slashes
result = detekface.detect("C:/Users/Username/Pictures/photo.jpg")

# Correct: pathlib.Path
result = detekface.detect(Path("C:/Users/Username/Pictures/photo.jpg"))
```

---

## 6. Still Need Help?

If you encounter an issue not covered above:
1. Verify you are using the latest version of `detekface` and `opencv-python`.
2. Inspect the exception's `.message` and `.hint` attributes.
3. Open an issue on GitHub: [https://github.com/Abishek-C0DE5/DetekFace/issues](https://github.com/Abishek-C0DE5/DetekFace/issues) with a minimal code example and traceback.
