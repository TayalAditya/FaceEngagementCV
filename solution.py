"""
identity_energy.py
Named Face Identity + Energy Report

Official-template-compatible solution with stronger matching internals.
Run: python solution.py
"""

from __future__ import annotations

import base64
import html
import json
import math
import re
import sys
import time
import types
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _install_tensorflow_doc_controls_stub() -> None:
    """Keep MediaPipe importable without pulling TensorFlow doc helpers."""

    if "tensorflow.tools.docs.doc_controls" in sys.modules:
        return

    def identity_decorator(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(fn):
            return fn

        return decorator

    tensorflow_mod = sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))
    tools_mod = sys.modules.setdefault("tensorflow.tools", types.ModuleType("tensorflow.tools"))
    docs_mod = sys.modules.setdefault("tensorflow.tools.docs", types.ModuleType("tensorflow.tools.docs"))
    doc_controls_mod = types.ModuleType("tensorflow.tools.docs.doc_controls")
    doc_controls_mod.do_not_generate_docs = identity_decorator
    doc_controls_mod.do_not_doc_inheritable = identity_decorator
    doc_controls_mod.for_subclass_implementers = identity_decorator

    tensorflow_mod.tools = tools_mod
    tools_mod.docs = docs_mod
    docs_mod.doc_controls = doc_controls_mod
    sys.modules["tensorflow.tools.docs.doc_controls"] = doc_controls_mod


_install_tensorflow_doc_controls_stub()

try:
    import face_recognition
except Exception as exc:  # pragma: no cover - import failure is fatal at runtime
    face_recognition = None
    FACE_REC_IMPORT_ERROR = exc
else:
    FACE_REC_IMPORT_ERROR = None

try:
    import mediapipe as mp
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False
else:
    MEDIAPIPE_AVAILABLE = True


KNOWN_FACES_DIR = Path("known_faces")
VIDEO_PATH = Path("classroom_video.mov")       # rename your video to this, or change path here
REPORT_HTML_OUT = Path("report.html")
INTEGRATION_OUT = Path("integration_output.json")
DEMO_OUT = Path("demo.mp4")

SCHOOL_NAME = "School Name"                    # set this before running
MATCH_THRESHOLD = 0.55
MAX_KEYFRAMES = 240

DETECT_SCALE = 0.55
DETECT_UPSAMPLE = 1
KNOWN_FACE_UPSAMPLE = 2
KNOWN_FACE_JITTERS = 3
MIN_FACE_PX = 20
UNKNOWN_CLUSTER_DISTANCE = 0.48
SECOND_PASS_MATCH_THRESHOLD = 0.59
SECOND_PASS_MARGIN = 0.03
MIN_UNKNOWN_DETECTIONS = 10
MAX_UNKNOWN_PERSONS = 5

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

KNOWN_FACE_IMAGES: Dict[str, Path] = {}
KNOWN_FACE_ENCODINGS: Dict[str, List[np.ndarray]] = {}
FLAT_KNOWN_NAMES: List[str] = []
FLAT_KNOWN_ENCODINGS: List[np.ndarray] = []

_FACE_MESH = None
_FACE_DETECTOR = None
_VIDEO_FPS = 25.0
_VIDEO_FRAME_COUNT = 0
_UNKNOWN_TEMP_COUNTER = 0


@dataclass
class DetectionGroup:
    name: str
    matched: bool
    match_confidence: float = 0.0
    best_distance: float = 1.0
    detections: List[dict] = field(default_factory=list)

    def add(self, detection: dict) -> None:
        self.detections.append(detection)
        if detection.get("matched"):
            self.match_confidence = max(self.match_confidence, float(detection.get("confidence", 0.0)))
            self.best_distance = min(self.best_distance, float(detection.get("distance", 1.0)))

    def absorb(
        self,
        other: "DetectionGroup",
        confidence: Optional[float] = None,
        distance: Optional[float] = None,
    ) -> None:
        self.detections.extend(other.detections)
        if confidence is not None:
            self.match_confidence = max(self.match_confidence, float(confidence))
        if distance is not None:
            self.best_distance = min(self.best_distance, float(distance))

    def mean_encoding(self) -> Optional[np.ndarray]:
        encodings = [np.asarray(d["encoding"], dtype=np.float32) for d in self.detections if d.get("encoding") is not None]
        if not encodings:
            return None
        return np.mean(np.stack(encodings, axis=0), axis=0)


def print_progress(current: int, total: int, prefix: str, width: int = 28) -> None:
    if total <= 0:
        return
    current = max(0, min(current, total))
    ratio = current / total
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r  {prefix}: [{bar}] {current}/{total} ({ratio * 100:5.1f}%)", end="", flush=True)


def _build_face_mesh():
    if not MEDIAPIPE_AVAILABLE:
        return None
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.30,
    )


def _build_face_detector():
    if not MEDIAPIPE_AVAILABLE:
        return None
    return mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.35,
    )


def _get_face_mesh():
    global _FACE_MESH
    if _FACE_MESH is None:
        _FACE_MESH = _build_face_mesh()
    return _FACE_MESH


def _get_face_detector():
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = _build_face_detector()
    return _FACE_DETECTOR


def _apply_clahe(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _bbox_area_xywh(bbox: Tuple[int, int, int, int]) -> int:
    _, _, width, height = bbox
    return max(0, width) * max(0, height)


def _bbox_iou_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_left = max(ax, bx)
    inter_top = max(ay, by)
    inter_right = min(ax2, bx2)
    inter_bottom = min(ay2, by2)
    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = _bbox_area_xywh(a) + _bbox_area_xywh(b) - inter
    return inter / union if union > 0 else 0.0


def _clamp_xywh(frame_shape: Tuple[int, int, int], bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    x, y, width, height = bbox
    x = max(0, min(x, frame_w))
    y = max(0, min(y, frame_h))
    width = max(0, min(width, frame_w - x))
    height = max(0, min(height, frame_h - y))
    return x, y, width, height


def _profile_prefix() -> str:
    prefix = re.sub(r"[^A-Za-z0-9]+", "", SCHOOL_NAME).upper()
    return prefix or "SCHOOL"


def _confidence_from_distance(distance: float) -> float:
    return float(np.clip(1.0 - distance, 0.0, 1.0))


def _sharpness_score(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _best_crop(detections: List[dict]) -> Optional[np.ndarray]:
    best_crop = None
    best_tuple = (-1.0, -1)
    for detection in detections:
        crop = detection.get("face_crop")
        if crop is None or crop.size == 0:
            continue
        score = (_sharpness_score(crop), crop.shape[0] * crop.shape[1])
        if score > best_tuple:
            best_tuple = score
            best_crop = crop
    return best_crop.copy() if best_crop is not None else None


def _mean_metric(detections: List[dict], key: str) -> float:
    values = [float(d.get(key, 0.0)) for d in detections]
    return float(np.mean(values)) if values else 0.0


def _frame_to_hms(frame_idx: int) -> str:
    seconds = int(frame_idx / max(_VIDEO_FPS, 1.0))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def encode_b64(img: np.ndarray, size=(240, 240)) -> str:
    if img is None or img.size == 0:
        return ""
    resized = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode("utf-8") if ok else ""


def _decode_b64(image_b64: str) -> Optional[np.ndarray]:
    if not image_b64:
        return None
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _energy_color(score: int) -> Tuple[Tuple[int, int, int], str]:
    if score >= 75:
        return (45, 143, 98), "high"
    if score >= 50:
        return (31, 122, 184), "moderate"
    return (75, 89, 196), "low"


def _extract_face_detections(frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    detect_frame = frame
    if DETECT_SCALE != 1.0:
        detect_frame = cv2.resize(
            frame,
            (int(frame.shape[1] * DETECT_SCALE), int(frame.shape[0] * DETECT_SCALE)),
            interpolation=cv2.INTER_LINEAR,
        )

    rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
    raw_locs = list(face_recognition.face_locations(rgb, number_of_times_to_upsample=DETECT_UPSAMPLE))

    detector = _get_face_detector()
    if detector is not None:
        try:
            mp_result = detector.process(rgb)
        except Exception:
            mp_result = None
        if mp_result and mp_result.detections:
            det_h, det_w = detect_frame.shape[:2]
            for det in mp_result.detections:
                rel = det.location_data.relative_bounding_box
                left = int(round(rel.xmin * det_w))
                top = int(round(rel.ymin * det_h))
                width = int(round(rel.width * det_w))
                height = int(round(rel.height * det_h))
                pad_x = int(round(width * 0.12))
                pad_y = int(round(height * 0.18))
                mp_box = (
                    max(0, left - pad_x),
                    max(0, top - pad_y),
                    max(0, width + (2 * pad_x)),
                    max(0, height + (2 * pad_y)),
                )
                overlaps = False
                for existing in raw_locs:
                    fr_box = (existing[3], existing[0], existing[1] - existing[3], existing[2] - existing[0])
                    if _bbox_iou_xywh(mp_box, fr_box) >= 0.55:
                        overlaps = True
                        break
                if overlaps:
                    continue
                raw_locs.append((mp_box[1], mp_box[0] + mp_box[2], mp_box[1] + mp_box[3], mp_box[0]))

    filtered_locs: List[Tuple[int, int, int, int]] = []
    xywh_boxes: List[Tuple[int, int, int, int]] = []
    scale_back = 1.0 / DETECT_SCALE
    for top, right, bottom, left in raw_locs:
        if DETECT_SCALE != 1.0:
            top = int(round(top * scale_back))
            right = int(round(right * scale_back))
            bottom = int(round(bottom * scale_back))
            left = int(round(left * scale_back))
        width = right - left
        height = bottom - top
        if width < MIN_FACE_PX or height < MIN_FACE_PX:
            continue
        filtered_locs.append((top, right, bottom, left))
        xywh_boxes.append((left, top, width, height))

    if not filtered_locs:
        return []

    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_full, filtered_locs)
    return list(zip(xywh_boxes, encodings))


def load_known_faces(folder: Path) -> dict:
    """
    Read every image from the folder.
    The filename without extension is the person's name.
    Encode each face with face_recognition.
    Return: { "Arjun Mehta": [128-d encoding, ...], ... }
    """

    global KNOWN_FACE_IMAGES, KNOWN_FACE_ENCODINGS
    global FLAT_KNOWN_NAMES, FLAT_KNOWN_ENCODINGS

    known: Dict[str, List[np.ndarray]] = {}
    KNOWN_FACE_IMAGES = {}
    KNOWN_FACE_ENCODINGS = {}
    FLAT_KNOWN_NAMES = []
    FLAT_KNOWN_ENCODINGS = []

    if face_recognition is None:
        print(f"  ERROR: face_recognition import failed: {FACE_REC_IMPORT_ERROR}")
        return known

    if not folder.exists():
        print(f"  ERROR: known faces folder not found: {folder}")
        return known

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for image_path in sorted(folder.iterdir()):
        if image_path.suffix.lower() not in valid_exts:
            continue
        name = image_path.stem.strip()
        if not name:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  warning: could not read {image_path.name}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=KNOWN_FACE_UPSAMPLE)
        encodings = face_recognition.face_encodings(rgb, face_locs, num_jitters=KNOWN_FACE_JITTERS)
        if not encodings:
            print(f"  warning: no face found in {image_path.name}, skipped")
            continue

        known.setdefault(name, []).extend(np.asarray(enc, dtype=np.float32) for enc in encodings)
        KNOWN_FACE_IMAGES.setdefault(name, image_path)

    for name, encodings in known.items():
        KNOWN_FACE_ENCODINGS[name] = encodings
        for encoding in encodings:
            FLAT_KNOWN_NAMES.append(name)
            FLAT_KNOWN_ENCODINGS.append(encoding)

    return known


def extract_keyframes(video_path: Path, max_frames: int) -> list:
    """
    Open the video, pull up to max_frames evenly spaced frames.
    Apply CLAHE on each frame to help with CCTV lighting.
    Return: [(frame_index, numpy_array), ...]
    """

    global _VIDEO_FPS, _VIDEO_FRAME_COUNT

    frames: List[Tuple[int, np.ndarray]] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: could not open video: {video_path}")
        return frames

    _VIDEO_FPS = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    _VIDEO_FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if _VIDEO_FRAME_COUNT <= 0:
        cap.release()
        return frames

    frame_count = min(max_frames, _VIDEO_FRAME_COUNT)
    indices = np.linspace(0, _VIDEO_FRAME_COUNT - 1, num=frame_count, dtype=int).tolist()
    selected = set(indices)

    current = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if current in selected:
            frames.append((current, _apply_clahe(frame)))
            print_progress(len(frames), frame_count, prefix="Keyframes")
        current += 1
        if len(frames) >= frame_count:
            break

    cap.release()
    if frame_count:
        print()
    return frames


def detect_and_match(frame: np.ndarray, known: dict, threshold: float) -> list:
    """
    Detect all faces in frame, compare each against known encodings.
    Return a list of dicts - one per detected face.
    """

    del known
    global _UNKNOWN_TEMP_COUNTER

    detections = []
    if face_recognition is None or not FLAT_KNOWN_ENCODINGS:
        return detections

    raw_detections = _extract_face_detections(frame)
    if not raw_detections:
        return detections

    best_named: Dict[str, dict] = {}
    unknowns: List[dict] = []

    for bbox, encoding in raw_detections:
        x, y, width, height = _clamp_xywh(frame.shape, bbox)
        if width <= 0 or height <= 0:
            continue
        crop = frame[y:y + height, x:x + width]
        if crop.size == 0:
            continue

        distances = face_recognition.face_distance(FLAT_KNOWN_ENCODINGS, encoding)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])

        if best_distance <= threshold:
            name = FLAT_KNOWN_NAMES[best_index]
            item = {
                "name": name,
                "matched": True,
                "confidence": round(_confidence_from_distance(best_distance), 2),
                "bbox": (x, y, width, height),
                "face_crop": crop.copy(),
                "encoding": np.asarray(encoding, dtype=np.float32),
                "distance": best_distance,
            }
            current = best_named.get(name)
            if current is None:
                best_named[name] = item
            else:
                current_area = _bbox_area_xywh(current["bbox"])
                new_area = _bbox_area_xywh(item["bbox"])
                if best_distance < current["distance"] or (
                    math.isclose(best_distance, current["distance"], abs_tol=1e-6) and new_area > current_area
                ):
                    best_named[name] = item
            continue

        _UNKNOWN_TEMP_COUNTER += 1
        unknowns.append(
            {
                "name": f"UNKNOWN_TMP_{_UNKNOWN_TEMP_COUNTER:04d}",
                "matched": False,
                "confidence": 0.0,
                "bbox": (x, y, width, height),
                "face_crop": crop.copy(),
                "encoding": np.asarray(encoding, dtype=np.float32),
                "distance": best_distance,
            }
        )

    detections.extend(best_named.values())
    detections.extend(unknowns)
    return detections


def compute_face_brightness(face_crop: np.ndarray) -> float:
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray) / 2.55)


def compute_eye_openness(face_crop: np.ndarray) -> float:
    """
    Average of eye height / eye width, scaled 0-100.
    MediaPipe failure returns 50.0 as a neutral fallback.
    """

    face_mesh = _get_face_mesh()
    if face_mesh is None:
        return 50.0

    crop = face_crop
    height, width = crop.shape[:2]
    if min(height, width) < 80:
        scale = 80 / min(height, width)
        crop = cv2.resize(crop, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    try:
        result = face_mesh.process(rgb)
    except Exception:
        return 50.0

    if not result.multi_face_landmarks:
        return 50.0

    landmarks = result.multi_face_landmarks[0].landmark
    image_h, image_w = crop.shape[:2]

    def eye_ratio(indices: List[int]) -> float:
        pts = np.array([(landmarks[i].x * image_w, landmarks[i].y * image_h) for i in indices], dtype=np.float32)
        width_value = np.linalg.norm(pts[0] - pts[3])
        if width_value < 1e-6:
            return 0.0
        height_value = (
            np.linalg.norm(pts[1] - pts[5]) +
            np.linalg.norm(pts[2] - pts[4])
        ) / 2.0
        return float(height_value / width_value)

    avg_ratio = (eye_ratio(LEFT_EYE) + eye_ratio(RIGHT_EYE)) / 2.0
    return float(np.clip(((avg_ratio - 0.11) / 0.19) * 100.0, 0.0, 100.0))


def compute_movement(prev_frame, curr_frame: np.ndarray, bbox: tuple) -> float:
    if prev_frame is None:
        return 0.0

    x, y, width, height = _clamp_xywh(curr_frame.shape, bbox)
    if width <= 0 or height <= 0:
        return 0.0

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_crop = prev_gray[y:y + height, x:x + width]
    curr_crop = curr_gray[y:y + height, x:x + width]
    if prev_crop.size == 0 or curr_crop.size == 0:
        return 0.0

    if prev_crop.shape != curr_crop.shape:
        curr_crop = cv2.resize(curr_crop, (prev_crop.shape[1], prev_crop.shape[0]), interpolation=cv2.INTER_LINEAR)

    if prev_crop.shape[0] < 48 or prev_crop.shape[1] < 48:
        prev_crop = cv2.resize(prev_crop, (72, 72), interpolation=cv2.INTER_LINEAR)
        curr_crop = cv2.resize(curr_crop, (72, 72), interpolation=cv2.INTER_LINEAR)

    flow = cv2.calcOpticalFlowFarneback(
        prev_crop,
        curr_crop,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.clip((np.mean(magnitude) / 4.0) * 100.0, 0.0, 100.0))


def _cluster_unknown_detections(unknown_detections: List[dict]) -> List[DetectionGroup]:
    clusters: List[DetectionGroup] = []
    for detection in sorted(unknown_detections, key=lambda item: item["frame_idx"]):
        encoding = detection.get("encoding")
        if encoding is None:
            continue

        best_cluster = None
        best_distance = float("inf")
        for cluster in clusters:
            mean_encoding = cluster.mean_encoding()
            if mean_encoding is None:
                continue
            if any(existing["frame_idx"] == detection["frame_idx"] for existing in cluster.detections):
                continue
            distance = float(np.linalg.norm(mean_encoding - encoding))
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster

        if best_cluster is not None and best_distance <= UNKNOWN_CLUSTER_DISTANCE:
            best_cluster.add(detection)
        else:
            cluster = DetectionGroup(name="UNKNOWN", matched=False)
            cluster.add(detection)
            clusters.append(cluster)

    return clusters


def _build_person_payload(group: DetectionGroup) -> dict:
    detections = sorted(group.detections, key=lambda item: item["frame_idx"])
    brightness = round(_mean_metric(detections, "brightness"))
    eye_openness = round(_mean_metric(detections, "eye_openness"))
    movement_activity = round(_mean_metric(detections, "movement"))
    energy_score = round((brightness * 0.35) + (eye_openness * 0.30) + (movement_activity * 0.35))
    profile_image_b64 = encode_b64(_best_crop(detections))

    return {
        "person_id": "",
        "name": group.name,
        "matched": group.matched,
        "match_confidence": round(group.match_confidence, 2) if group.matched else 0.0,
        "profile_image_b64": profile_image_b64,
        "frames_detected": len(detections),
        "energy_score": int(np.clip(energy_score, 0, 100)),
        "energy_breakdown": {
            "face_brightness": int(np.clip(brightness, 0, 100)),
            "eye_openness": int(np.clip(eye_openness, 0, 100)),
            "movement_activity": int(np.clip(movement_activity, 0, 100)),
        },
        "verdict": verdict(energy_score),
        "first_seen_frame": int(detections[0]["frame_idx"]) if detections else 0,
        "last_seen_frame": int(detections[-1]["frame_idx"]) if detections else 0,
    }


def _select_unknown_persons(persons: List[dict]) -> List[dict]:
    stable = [
        person for person in persons
        if person["frames_detected"] >= MIN_UNKNOWN_DETECTIONS
    ]
    stable.sort(
        key=lambda person: (
            -person["frames_detected"],
            -person["energy_score"],
            person["first_seen_frame"],
        )
    )
    return stable[:MAX_UNKNOWN_PERSONS]


def aggregate_persons(all_detections: list) -> list:
    """
    Group detections by person and build entries matching the persons array schema.
    """

    groups: Dict[str, DetectionGroup] = {}
    unknown_detections: List[dict] = []

    for detection in all_detections:
        if detection.get("matched"):
            name = detection["name"]
            group = groups.setdefault(name, DetectionGroup(name=name, matched=True))
            group.add(detection)
        else:
            unknown_detections.append(detection)

    unknown_clusters = _cluster_unknown_detections(unknown_detections)

    missing_names = [name for name in KNOWN_FACE_ENCODINGS if name not in groups]
    recovery_candidates: List[Tuple[str, float, float, DetectionGroup]] = []
    for cluster in unknown_clusters:
        mean_encoding = cluster.mean_encoding()
        if mean_encoding is None or not missing_names:
            continue
        distances = []
        for name in missing_names:
            name_distances = face_recognition.face_distance(KNOWN_FACE_ENCODINGS[name], mean_encoding)
            distances.append((name, float(np.min(name_distances))))
        distances.sort(key=lambda item: item[1])
        best_name, best_distance = distances[0]
        second_distance = distances[1][1] if len(distances) > 1 else 999.0
        margin = second_distance - best_distance
        if best_distance <= SECOND_PASS_MATCH_THRESHOLD and margin >= SECOND_PASS_MARGIN:
            recovery_candidates.append((best_name, best_distance, margin, cluster))

    recovery_candidates.sort(key=lambda item: (item[1], -item[2], -len(item[3].detections)))
    consumed_cluster_ids: set[int] = set()
    recovered_names: set[str] = set()
    for best_name, best_distance, _, cluster in recovery_candidates:
        cluster_id = id(cluster)
        if best_name in recovered_names or cluster_id in consumed_cluster_ids:
            continue
        group = groups.setdefault(best_name, DetectionGroup(name=best_name, matched=True))
        group.absorb(cluster, confidence=_confidence_from_distance(best_distance), distance=best_distance)
        recovered_names.add(best_name)
        consumed_cluster_ids.add(cluster_id)

    persons: List[dict] = []
    for group in groups.values():
        persons.append(_build_person_payload(group))

    unknown_people: List[dict] = []
    for cluster in unknown_clusters:
        if id(cluster) in consumed_cluster_ids:
            continue
        cluster.matched = False
        unknown_people.append(_build_person_payload(cluster))

    for unknown_index, person in enumerate(_select_unknown_persons(unknown_people), start=1):
        person["name"] = f"UNKNOWN_{unknown_index:03d}"
        persons.append(person)

    persons.sort(
        key=lambda person: (
            not person["matched"],
            -person["energy_score"],
            person["first_seen_frame"],
            person["name"],
        )
    )

    prefix = _profile_prefix()
    for index, person in enumerate(persons, start=1):
        person["person_id"] = f"{prefix}_P{index:04d}"

    return persons


def verdict(score: float) -> str:
    return "high" if score >= 75 else "moderate" if score >= 50 else "low"


def generate_report(persons: list, output_path: Path):
    generated_at = datetime.now().strftime("%d %b %Y %H:%M")
    matched = [person for person in persons if person["matched"]]
    unknown = [person for person in persons if not person["matched"]]
    avg_energy = round(np.mean([person["energy_score"] for person in matched])) if matched else 0

    def render_card(person: dict) -> str:
        score = int(person["energy_score"])
        energy_color = "#2d8f62" if score >= 75 else "#1f7ab8" if score >= 50 else "#4b59c4"
        confidence_html = (
            f"<div class='meta'>Match confidence: {person['match_confidence']:.2f}</div>"
            if person["matched"]
            else "<div class='meta'>Unmatched face cluster</div>"
        )
        return f"""
        <article class="card">
          <img class="avatar" src="data:image/jpeg;base64,{person['profile_image_b64']}" alt="{html.escape(person['name'])}">
          <div class="card-body">
            <div class="row-top">
              <div>
                <h3>{html.escape(person['name'])}</h3>
                <div class="meta">{html.escape(person['person_id'])}</div>
                {confidence_html}
                <div class="meta">Frames detected: {person['frames_detected']}</div>
              </div>
              <span class="badge badge-{person['verdict']}">{person['verdict'].title()}</span>
            </div>
            <div class="meter">
              <div class="meter-fill" style="width:{score}%; background:{energy_color};">{score}</div>
            </div>
            <div class="breakdown">
              <span>Brightness {person['energy_breakdown']['face_brightness']}</span>
              <span>Eye openness {person['energy_breakdown']['eye_openness']}</span>
              <span>Movement {person['energy_breakdown']['movement_activity']}</span>
            </div>
            <div class="frames">
              Seen from frame {person['first_seen_frame']} ({_frame_to_hms(person['first_seen_frame'])})
              to {person['last_seen_frame']} ({_frame_to_hms(person['last_seen_frame'])})
            </div>
          </div>
        </article>
        """.strip()

    matched_html = "\n".join(render_card(person) for person in matched) or "<p class='empty'>No known persons matched.</p>"
    unknown_html = "\n".join(render_card(person) for person in unknown) or "<p class='empty'>No unknown faces were retained.</p>"

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Named Face Identity + Energy Report</title>
  <style>
    :root {{
      --panel: rgba(255, 250, 242, 0.95);
      --ink: #263238;
      --muted: #617176;
      --line: rgba(57, 50, 39, 0.12);
      --shadow: 0 18px 36px rgba(64, 52, 36, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(199, 225, 220, 0.65), transparent 32%),
        radial-gradient(circle at top right, rgba(233, 217, 192, 0.55), transparent 28%),
        linear-gradient(180deg, #f3eee6 0%, #efe5d8 100%);
    }}
    .page {{ max-width: 1160px; margin: 0 auto; padding: 36px 20px 64px; }}
    .hero {{ background: var(--panel); border: 1px solid var(--line); border-radius: 28px; padding: 28px; box-shadow: var(--shadow); }}
    h1 {{ margin: 0 0 8px; font-size: 2.4rem; line-height: 1.05; }}
    .lead {{ margin: 0; color: var(--muted); font-size: 1.02rem; line-height: 1.55; }}
    .meta-grid {{ margin-top: 22px; display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
    .meta-box {{ padding: 14px 16px; border-radius: 18px; background: rgba(255,255,255,0.72); border: 1px solid var(--line); }}
    .meta-box .label {{ color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }}
    .meta-box .value {{ margin-top: 6px; font-size: 1.35rem; font-weight: 700; }}
    section {{ margin-top: 28px; }}
    .section-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 14px; }}
    .section-head h2 {{ margin: 0; font-size: 1.5rem; }}
    .section-head p {{ margin: 0; color: var(--muted); font-size: 0.95rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .card {{ display: grid; grid-template-columns: 110px minmax(0, 1fr); gap: 16px; padding: 18px; background: var(--panel); border: 1px solid var(--line); border-radius: 24px; box-shadow: var(--shadow); }}
    .avatar {{ width: 110px; height: 110px; border-radius: 22px; object-fit: cover; border: 1px solid var(--line); background: #ebe4d8; }}
    .row-top {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    h3 {{ margin: 0; font-size: 1.18rem; }}
    .meta {{ margin-top: 4px; color: var(--muted); font-size: 0.92rem; }}
    .badge {{ display: inline-block; padding: 6px 10px; border-radius: 999px; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; border: 1px solid currentColor; white-space: nowrap; }}
    .badge-high {{ color: #2d8f62; }}
    .badge-moderate {{ color: #1f7ab8; }}
    .badge-low {{ color: #4b59c4; }}
    .meter {{ margin-top: 14px; width: 100%; height: 18px; border-radius: 999px; background: rgba(38,50,56,0.10); overflow: hidden; }}
    .meter-fill {{ height: 100%; min-width: 44px; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; color: white; font-size: 0.78rem; font-weight: 700; }}
    .breakdown {{ display: flex; flex-wrap: wrap; gap: 10px 14px; margin-top: 12px; color: var(--ink); font-size: 0.92rem; }}
    .frames {{ margin-top: 12px; color: var(--muted); font-size: 0.92rem; line-height: 1.45; }}
    .empty {{ margin: 0; padding: 18px; color: var(--muted); border: 1px dashed var(--line); border-radius: 18px; background: rgba(255,255,255,0.6); }}
    footer {{ margin-top: 28px; color: var(--muted); font-size: 0.9rem; line-height: 1.55; }}
    @media (max-width: 680px) {{
      .card {{ grid-template-columns: 1fr; }}
      .avatar {{ width: 100%; height: auto; aspect-ratio: 1 / 1; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>Named Face Identity + Energy Report</h1>
      <p class="lead">
        Named Face Identity + Energy Report for {html.escape(SCHOOL_NAME)}.
        Energy score follows the assignment formula exactly:
        brightness x 0.35 + eye openness x 0.30 + movement activity x 0.35.
      </p>
      <div class="meta-grid">
        <div class="meta-box"><div class="label">Generated</div><div class="value">{generated_at}</div></div>
        <div class="meta-box"><div class="label">Video</div><div class="value">{html.escape(VIDEO_PATH.name)}</div></div>
        <div class="meta-box"><div class="label">Matched</div><div class="value">{len(matched)}</div></div>
        <div class="meta-box"><div class="label">Unknown</div><div class="value">{len(unknown)}</div></div>
        <div class="meta-box"><div class="label">Average Energy</div><div class="value">{avg_energy}</div></div>
      </div>
    </section>
    <section>
      <div class="section-head">
        <h2>Matched Persons</h2>
        <p>Known faces matched against the reference folder.</p>
      </div>
      <div class="grid">{matched_html}</div>
    </section>
    <section>
      <div class="section-head">
        <h2>Unknown Persons</h2>
        <p>Bonus section for unmatched face clusters retained from the sampled keyframes.</p>
      </div>
      <div class="grid">{unknown_html}</div>
    </section>
    <footer>
      Processing used evenly spaced keyframes with CCTV contrast enhancement, face recognition for identity matching,
      MediaPipe Face Mesh for eye openness, and dense optical flow for movement activity. The HTML is fully offline and
      contains no external scripts or network dependencies.
    </footer>
  </main>
</body>
</html>
"""

    output_path.write_text(html_doc, encoding="utf-8")


def write_integration_json(persons: list, output_path: Path, video_name: str, processing_time: float):
    output = {
        "source": "p1_identity_energy",
        "school": SCHOOL_NAME,
        "date": str(date.today()),
        "video_file": video_name,
        "total_persons_matched": sum(1 for person in persons if person.get("matched")),
        "total_persons_unknown": sum(1 for person in persons if not person.get("matched")),
        "processing_time_sec": round(float(processing_time), 2),
        "persons": persons,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def _draw_centered_text(frame: np.ndarray, text: str, y: int, scale: float, color: Tuple[int, int, int], thickness: int = 2) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    x = max(20, (frame.shape[1] - text_size[0]) // 2)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _demo_slide_base() -> np.ndarray:
    frame = np.full((720, 1280, 3), (238, 232, 221), dtype=np.uint8)
    cv2.rectangle(frame, (36, 36), (1244, 684), (250, 247, 241), thickness=-1)
    cv2.rectangle(frame, (36, 36), (1244, 684), (214, 201, 182), thickness=2)
    return frame


def _build_demo_intro(persons: List[dict]) -> np.ndarray:
    matched = sum(1 for person in persons if person["matched"])
    unknown = len(persons) - matched
    frame = _demo_slide_base()
    _draw_centered_text(frame, "Named Face Identity + Energy Report", 150, 1.35, (41, 52, 56), 3)
    _draw_centered_text(frame, "Named Face Identity + Energy Report", 212, 0.95, (83, 97, 103), 2)
    cv2.putText(frame, f"School: {SCHOOL_NAME}", (110, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (41, 52, 56), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Video: {VIDEO_PATH.name}", (110, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (41, 52, 56), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Matched persons: {matched}", (110, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (41, 52, 56), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Unknown persons: {unknown}", (110, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (41, 52, 56), 2, cv2.LINE_AA)
    cv2.putText(frame, "Formula: brightness x 0.35 + eye openness x 0.30 + movement activity x 0.35", (110, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (83, 97, 103), 2, cv2.LINE_AA)
    return frame


def _build_demo_person_slide(person: dict) -> np.ndarray:
    frame = _demo_slide_base()
    image = _decode_b64(person["profile_image_b64"])
    if image is not None:
        image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_LANCZOS4)
        frame[180:500, 96:416] = image
        cv2.rectangle(frame, (96, 180), (416, 500), (214, 201, 182), 2)

    energy_color, _ = _energy_color(person["energy_score"])
    cv2.putText(frame, person["name"], (480, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (41, 52, 56), 3, cv2.LINE_AA)
    cv2.putText(frame, person["person_id"], (482, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (98, 109, 113), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Verdict: {person['verdict'].title()}", (482, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.9, energy_color, 2, cv2.LINE_AA)
    if person["matched"]:
        cv2.putText(frame, f"Match confidence: {person['match_confidence']:.2f}", (482, 336), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (41, 52, 56), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Unmatched cluster retained", (482, 336), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (41, 52, 56), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frames detected: {person['frames_detected']}", (482, 382), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (41, 52, 56), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Seen: frame {person['first_seen_frame']} to {person['last_seen_frame']}", (482, 428), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (41, 52, 56), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (482, 500), (1090, 540), (214, 208, 198), thickness=-1)
    fill_width = int(round((person["energy_score"] / 100.0) * 608))
    cv2.rectangle(frame, (482, 500), (482 + fill_width, 540), energy_color, thickness=-1)
    cv2.rectangle(frame, (482, 500), (1090, 540), (180, 168, 152), thickness=2)
    cv2.putText(frame, f"Energy score: {person['energy_score']}", (500, 528), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (255, 255, 255), 2, cv2.LINE_AA)
    breakdown = person["energy_breakdown"]
    cv2.putText(frame, f"Brightness: {breakdown['face_brightness']}", (482, 592), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (41, 52, 56), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Eye openness: {breakdown['eye_openness']}", (482, 634), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (41, 52, 56), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Movement: {breakdown['movement_activity']}", (760, 592), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (41, 52, 56), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Timestamp: {_frame_to_hms(person['first_seen_frame'])} to {_frame_to_hms(person['last_seen_frame'])}", (760, 634), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (41, 52, 56), 2, cv2.LINE_AA)
    return frame


def _build_demo_outro(persons: List[dict]) -> np.ndarray:
    matched = sum(1 for person in persons if person["matched"])
    frame = _demo_slide_base()
    _draw_centered_text(frame, "Run Complete", 180, 1.25, (41, 52, 56), 3)
    _draw_centered_text(frame, f"{matched} matched, {len(persons) - matched} unknown", 258, 0.95, (83, 97, 103), 2)
    _draw_centered_text(frame, "Artifacts generated:", 360, 0.9, (41, 52, 56), 2)
    _draw_centered_text(frame, "report.html  |  integration_output.json  |  demo.mp4", 430, 0.85, (41, 52, 56), 2)
    _draw_centered_text(frame, f"Generated on {datetime.now().strftime('%d %b %Y %H:%M')}", 540, 0.78, (83, 97, 103), 2)
    return frame


def _write_demo_video(persons: List[dict], out_path: Path) -> bool:
    if not persons:
        return False
    fps = 24
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    if not writer.isOpened():
        return False

    segments: List[Tuple[np.ndarray, int]] = [(_build_demo_intro(persons), 2 * fps)]
    for person in persons:
        segments.append((_build_demo_person_slide(person), int(1.4 * fps)))
    segments.append((_build_demo_outro(persons), int(1.6 * fps)))

    for slide, frame_count in segments:
        for _ in range(frame_count):
            writer.write(slide)
    writer.release()
    return True


if __name__ == "__main__":
    t0 = time.time()

    print("Step 1 - loading known faces ...")
    known = load_known_faces(KNOWN_FACES_DIR)
    if not known:
        print("  ERROR: no faces loaded. Check known_faces/ has images named like 'Arjun Mehta.jpg'")
        raise SystemExit(1)
    print(f"  {len(known)} persons: {', '.join(list(known.keys())[:6])}")

    print("Step 2 - extracting keyframes ...")
    frames = extract_keyframes(VIDEO_PATH, MAX_KEYFRAMES)
    print(f"  {len(frames)} frames extracted from {_VIDEO_FRAME_COUNT} total frames")

    print("Step 3 & 4 - detecting + scoring faces ...")
    all_detections = []
    prev_frame = None
    for index, (frame_idx, frame) in enumerate(frames, start=1):
        detections = detect_and_match(frame, known, MATCH_THRESHOLD)
        for detection in detections:
            detection["frame_idx"] = frame_idx
            detection["brightness"] = compute_face_brightness(detection["face_crop"])
            detection["eye_openness"] = compute_eye_openness(detection["face_crop"])
            detection["movement"] = compute_movement(prev_frame, frame, detection["bbox"])
        all_detections.extend(detections)
        prev_frame = frame
        print_progress(index, len(frames), prefix="Scoring")
    if frames:
        print()

    print("Step 5 - aggregating per-person ...")
    persons = aggregate_persons(all_detections)
    t1 = round(time.time() - t0, 2)

    print("Step 6 - writing report.html ...")
    generate_report(persons, REPORT_HTML_OUT)
    print("Step 7 - writing integration_output.json ...")
    write_integration_json(persons, INTEGRATION_OUT, str(VIDEO_PATH.name), t1)
    print("Step 8 - writing demo.mp4 ...")
    if _write_demo_video(persons, DEMO_OUT):
        print(f"  demo.mp4 -> {DEMO_OUT}")
    else:
        print("  warning: could not create demo.mp4")

    if _FACE_MESH is not None and hasattr(_FACE_MESH, "close"):
        _FACE_MESH.close()
    if _FACE_DETECTOR is not None and hasattr(_FACE_DETECTOR, "close"):
        _FACE_DETECTOR.close()

    print()
    print("=" * 50)
    print(f"  Finished in {t1}s")
    print(f"  Persons found: {len(persons)}")
    for person in persons:
        print(f"    {person['name']:30s}  energy {person['energy_score']:5.1f}  ({person['verdict']})")
    print(f"  report.html              -> {REPORT_HTML_OUT}")
    print(f"  integration_output.json  -> {INTEGRATION_OUT}")
    print(f"  demo.mp4                 -> {DEMO_OUT}")
    print("=" * 50)
