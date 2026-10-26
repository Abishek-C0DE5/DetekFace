# Licensing & Intellectual Property Guide

DetekFace is built on permissive, open-source software and open-access deep learning models. This document summarizes all software licenses, neural model copyrights, and third-party dependency terms.

---

## 1. Summary Matrix

| Component | Author / Creator | License | Commercial Use Permitted? |
| :--- | :--- | :--- | :---: |
| **DetekFace Package** | DetekFace Authors | [MIT License](#2-detekface-package-license) | Yes |
| **YuNet ONNX Model** | Shiqi Yu (`libfacedetection`) | [MIT License](#3-yunet-face-detector-model-license) | Yes |
| **MediaPipe Model** | Google MediaPipe Team | [Apache 2.0](#4-mediapipe-blazeface-model-license) | Yes |
| **OpenCV (`opencv-python`)** | OpenCV Foundation | Apache 2.0 | Yes |
| **NumPy** | NumPy Developers | BSD 3-Clause | Yes |
| **Validation Dataset** | OpenCV / Wikimedia Commons | CC BY 4.0 & Public Domain | Yes |

---

## 2. DetekFace Package License

DetekFace source code is distributed under the **MIT License**:

```text
MIT License

Copyright (c) 2026 Abishek & DetekFace Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 3. YuNet Face Detector Model License

The official YuNet face detection neural network weights used by DetekFace are distributed under the **MIT License**:

- **Model Files**: `face_detection_yunet_2026may.onnx`, `face_detection_yunet_2023mar.onnx`
- **Original Author**: Prof. Shiqi Yu (`libfacedetection`), Southern University of Science and Technology, Shenzhen, China.
- **Repository**: [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **Copyright**: Copyright (c) 2020 Shiqi Yu
- **Permitted Uses**: Free for commercial, non-commercial, and academic research purposes.

---

## 4. MediaPipe BlazeFace Model License

The optional MediaPipe face detection neural network weights (`blaze_face_short_range.tflite`) are distributed under the **Apache License 2.0**:

- **Model File**: `blaze_face_short_range.tflite`
- **Original Author**: Google MediaPipe Authors & Google LLC
- **Repository**: [Google MediaPipe Models](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite)
- **Copyright**: Copyright (c) 2023 Google LLC
- **License**: Apache License, Version 2.0

---

## 5. Third-Party Dependencies

DetekFace dependencies use business-friendly open source licenses:

1. **OpenCV (`opencv-python`)**:
   - Licensed under Apache-2.0.
   - Allows bundling, modification, and binary redistribution.
2. **NumPy (`numpy`)**:
   - Licensed under the BSD 3-Clause License.
   - Allows commercial and private use without copyleft restrictions.
3. **MediaPipe (`mediapipe`, optional)**:
   - Licensed under Apache-2.0.

---

## 6. Validation Dataset Licensing

The reproducible sample images used in `validation/manifest.json` and `tests/assets/` are legally usable assets:

- **`sample_face.jpg`**: Official OpenCV test asset, licensed under Apache-2.0 by the OpenCV Foundation.
- **Validation Dataset Photos**: Sourced from Wikimedia Commons, licensed under Creative Commons Attribution 4.0 International (CC BY 4.0) or Public Domain (CC0).
