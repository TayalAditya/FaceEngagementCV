# FaceEngagementCV

Identifies individuals in classroom CCTV footage and scores their engagement level using computer vision.

Given a folder of reference face photos and a classroom video, the script matches every detected face to a known identity, computes an energy score from three visual signals, and writes an offline HTML report and structured JSON output.

---

## How it works

**Face matching**
Each video frame is scanned for faces. Detected face encodings (128-d vectors via dlib) are compared against reference photos. Faces that don't match any reference are clustered by encoding similarity and surfaced as `UNKNOWN_001`, `UNKNOWN_002`, etc.

**Energy scoring**
Three signals are averaged across all frames where a person appears:

```
energy_score = (face_brightness × 0.35) + (eye_openness × 0.30) + (movement_activity × 0.35)
```

| Signal | Method |
|---|---|
| `face_brightness` | Mean grayscale pixel value of face crop |
| `eye_openness` | Eye Aspect Ratio from MediaPipe FaceMesh landmarks |
| `movement_activity` | Dense optical flow (Farneback) in face bounding box |

Verdict: `High Energy ≥ 75` · `Moderate 50–74` · `Low Energy < 50`

**Outputs**
- `report.html` — fully offline single-page report: photo + name + energy bar + verdict per person
- `integration_output.json` — structured output with all computed fields per person

---

## Stack

```
Python 3.9+
opencv-python      video processing, CLAHE, optical flow, demo generation
face_recognition   dlib-backed face detection and 128-d encoding
mediapipe          FaceMesh (eye openness) + FaceDetection (small-face fallback)
numpy, Pillow
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run

1. Put reference face images in `known_faces/` — filename (without extension) is used as the identity label
2. Set `VIDEO_PATH` and `SCHOOL_NAME` at the top of `solution.py`
3. Run:

```bash
python solution.py
```
