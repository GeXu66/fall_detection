import os
import sys
import math
import time
from typing import List, Tuple, Optional
import argparse
import logging

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Ultralytics is required. Install with: pip install ultralytics"
    ) from exc

ULTRA_ASSET_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"


# -----------------------------
# Logging
# -----------------------------
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    level_norm = (level or "INFO").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = level_map.get(level_norm, logging.INFO)

    logger = logging.getLogger("fall_detection")
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplication when re-running in notebooks/REPLs
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except Exception:
            # Fall back to console-only if file can't be opened
            logger.warning(f"Failed to open log file: {log_file}")

# -----------------------------
# Utility drawing functions
# -----------------------------
def draw_rounded_rectangle(image: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 2, radius: int = 10) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    # Draw straight edges
    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    # Draw arcs for corners
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180.0, 0, 90, color, thickness)
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270.0, 0, 90, color, thickness)
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0.0, 0, 90, color, thickness)
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90.0, 0, 90, color, thickness)


def put_label_above_box(image: np.ndarray, text: str, box: Tuple[int, int, int, int], color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    margin = 6
    label_x1 = max(0, x1)
    label_y1 = max(0, y1 - text_h - 2 * margin)
    # draw text without filled background
    cv2.putText(image, text, (label_x1 + margin, label_y1 + text_h), font, scale, color, thickness, lineType=cv2.LINE_AA)


# -----------------------------
# Pose helpers
# -----------------------------
COCO_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO keypoint skeleton edges (0-based indices)
COCO_SKELETON_EDGES = [
    (5, 6),    # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (11, 12),        # hips
    (5, 11), (6, 12), # torso sides
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
]


def get_keypoint(points: np.ndarray, idx: int) -> Optional[Tuple[float, float, float]]:
    if points is None or idx < 0 or idx >= len(points):
        return None
    x, y, conf = points[idx]
    if conf is None:
        conf = 0.0
    return float(x), float(y), float(conf)


def avg_point(a: Optional[Tuple[float, float, float]], b: Optional[Tuple[float, float, float]]) -> Optional[Tuple[float, float]]:
    if a is None and b is None:
        return None
    if a is None:
        return (b[0], b[1])
    if b is None:
        return (a[0], a[1])
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def vector(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if a is None or b is None:
        return None
    return (b[0] - a[0], b[1] - a[1])


def angle_between(v1: Optional[Tuple[float, float]], v2: Optional[Tuple[float, float]]) -> Optional[float]:
    if v1 is None or v2 is None:
        return None
    x1, y1 = v1
    x2, y2 = v2
    n1 = math.hypot(x1, y1)
    n2 = math.hypot(x2, y2)
    if n1 <= 1e-6 or n2 <= 1e-6:
        return None
    dot = x1 * x2 + y1 * y2
    cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cosang))


def compute_bbox_aspect_ratio(xyxy: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = xyxy
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return w / h


def y_extent(points: np.ndarray, indices: List[int]) -> Optional[Tuple[float, float]]:
    ys = []
    for idx in indices:
        kp = get_keypoint(points, idx)
        if kp is not None and kp[2] > 0.3:
            ys.append(kp[1])
    if not ys:
        return None
    return (min(ys), max(ys))


def compute_fall_score(points: np.ndarray, box_xyxy: Tuple[float, float, float, float], prev_orientation: Optional[float]) -> Tuple[float, dict]:
    """
    Return a fall-likelihood score in [0, 1] and debug features.

    Heuristics (higher score => more likely a fall):
    - Low standing orientation: shoulder-hip verticality angle deviates towards horizontal
    - Large bbox aspect ratio (wide vs tall)
    - Head close to hips/ankles vertically, or head near ground (top of frame) relative to ankles
    - Upper body angle relative to ground near horizontal
    - Sudden change from vertical orientation to horizontal across frames
    """

    x1, y1, x2, y2 = box_xyxy
    bbox_ar = compute_bbox_aspect_ratio(box_xyxy)

    left_shoulder = get_keypoint(points, COCO_KP_NAMES.index("left_shoulder"))
    right_shoulder = get_keypoint(points, COCO_KP_NAMES.index("right_shoulder"))
    left_hip = get_keypoint(points, COCO_KP_NAMES.index("left_hip"))
    right_hip = get_keypoint(points, COCO_KP_NAMES.index("right_hip"))
    left_knee = get_keypoint(points, COCO_KP_NAMES.index("left_knee"))
    right_knee = get_keypoint(points, COCO_KP_NAMES.index("right_knee"))
    left_ankle = get_keypoint(points, COCO_KP_NAMES.index("left_ankle"))
    right_ankle = get_keypoint(points, COCO_KP_NAMES.index("right_ankle"))
    nose = get_keypoint(points, COCO_KP_NAMES.index("nose"))

    mid_shoulder = avg_point(left_shoulder, right_shoulder)
    mid_hip = avg_point(left_hip, right_hip)
    mid_knee = avg_point(left_knee, right_knee)
    mid_ankle = avg_point(left_ankle, right_ankle)

    torso_vec = vector(mid_hip, mid_shoulder)  # up vector when standing
    leg_vec = vector(mid_knee, mid_ankle)

    # Orientation angle: angle between torso vector and vertical axis (0 deg = vertical)
    vertical_axis = (0.0, -1.0)
    orientation_angle = angle_between(torso_vec, vertical_axis)

    # Upper body horizontalness: angle vs horizontal axis (lower is more horizontal)
    horizontal_axis = (1.0, 0.0)
    upper_body_horiz_angle = angle_between(torso_vec, horizontal_axis)

    # Head-to-ankle vertical proximity
    head_y = nose[1] if nose is not None else None
    ankle_y = None
    if mid_ankle is not None:
        ankle_y = mid_ankle[1]

    head_near_ankle = 0.0
    if head_y is not None and ankle_y is not None:
        box_h = max(1.0, y2 - y1)
        delta = abs(head_y - ankle_y) / box_h  # normalized vertical separation
        # Smaller separation indicates lying posture
        head_near_ankle = max(0.0, 1.0 - min(1.0, delta))  # 1 when delta=0, 0 when delta>=1

    # Standing/lying cues
    ar_score = min(1.0, max(0.0, (bbox_ar - 0.75) / 0.75))  # ~0 when tall, towards 1 when wide

    orientation_score = 0.0
    if orientation_angle is not None:
        # 0 deg vertical -> score 0, 90 deg horizontal -> score 1
        orientation_score = min(1.0, max(0.0, orientation_angle / 90.0))

    upper_body_horizontal_score = 0.0
    if upper_body_horiz_angle is not None:
        # 0 deg means perfectly horizontal torso
        upper_body_horizontal_score = min(1.0, max(0.0, (90.0 - upper_body_horiz_angle) / 90.0))

    # Sudden orientation change score
    orientation_change_score = 0.0
    if prev_orientation is not None and orientation_angle is not None:
        orientation_change = abs(orientation_angle - prev_orientation)
        orientation_change_score = min(1.0, orientation_change / 45.0)  # 45 deg+ in one step is high

    # Additional rule: wider-than-tall bbox (w/h > 0.6) increases fall likelihood
    bbox_rule_bonus = 0.0
    if bbox_ar > 0.6:
        # Grows linearly from 0 when ar==0.6 up to a small cap
        bbox_rule_bonus = min(0.5, max(0.0, (bbox_ar - 0.6) * 3))

    # Aggregate with weights
    weights = {
        "ar": 0.2,
        "orientation": 0.35,
        "upper_horiz": 0.2,
        "head_near_ankle": 0.15,
        "orientation_change": 0.1,
    }

    score = (
        weights["ar"] * ar_score
        + weights["orientation"] * orientation_score
        + weights["upper_horiz"] * upper_body_horizontal_score
        + weights["head_near_ankle"] * head_near_ankle
        + weights["orientation_change"] * orientation_change_score
        + bbox_rule_bonus 
    )

    debug = {
        "bbox_ar": bbox_ar,
        "orientation_angle": orientation_angle,
        "upper_body_horiz_angle": upper_body_horiz_angle,
        "head_near_ankle": head_near_ankle,
        "ar_score": ar_score,
        "orientation_score": orientation_score,
        "upper_body_horizontal_score": upper_body_horizontal_score,
        "orientation_change_score": orientation_change_score,
        "bbox_rule_bonus": bbox_rule_bonus,
    }
    return max(0.0, min(1.0, score)), debug


class TemporalSmoother:
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self.prev_scores = {}
        self.prev_orientations = {}
        self.next_id = 0

    def smooth_score(self, track_id: int, score: float) -> float:
        prev = self.prev_scores.get(track_id)
        if prev is None:
            smoothed = score
        else:
            smoothed = self.alpha * prev + (1.0 - self.alpha) * score
        self.prev_scores[track_id] = smoothed
        return smoothed

    def set_orientation(self, track_id: int, orientation: Optional[float]) -> None:
        if orientation is not None:
            self.prev_orientations[track_id] = orientation

    def get_prev_orientation(self, track_id: int) -> Optional[float]:
        return self.prev_orientations.get(track_id)

    def assign_id(self) -> int:
        tid = self.next_id
        self.next_id += 1
        return tid


def ensure_dirs(weights_dir: str, results_dir: str) -> None:
    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir, exist_ok=True)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)


def locate_or_download_weights(weights_dir: str, model_size: str = "n") -> str:
    size = (model_size or "n").lower()
    if size not in {"n", "s", "m", "l"}:
        size = "n"
    model_name = f"yolo11{size}-pose.pt"
    local_path = os.path.join(weights_dir, model_name)
    if os.path.isfile(local_path):
        return local_path

    # Attempt direct download from official Ultralytics assets if missing
    asset_url = f"{ULTRA_ASSET_BASE}/{model_name}"
    try:
        import urllib.request
        print(f"Downloading weights to {local_path} ...")
        urllib.request.urlretrieve(asset_url, local_path)
        if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
            print("Download completed.")
            return local_path
    except Exception:
        pass

    # Fallback: rely on ultralytics internal cache resolution
    try:
        _ = YOLO(model_name)
        return model_name
    except Exception:
        # If even that fails, re-raise with guidance
        raise RuntimeError(
            f"Failed to obtain weights '{model_name}'. Ensure internet connectivity or place the file at 'weights/{model_name}'."
        )


def xyxy_int(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(int(round(x1)), w - 1)),
        max(0, min(int(round(y1)), h - 1)),
        max(0, min(int(round(x2)), w - 1)),
        max(0, min(int(round(y2)), h - 1)),
    )


def process_video(
    source_video_path: str,
    weights_dir: str = "weights",
    results_dir: str = "results",
    output_name: Optional[str] = None,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: Optional[str] = None,
    show: bool = True,
    model_size: str = "n",
    downsample: int = 1,
    log_every: int = 30,
) -> str:
    logger = logging.getLogger("fall_detection")
    ensure_dirs(weights_dir, results_dir)
    weights_path = locate_or_download_weights(weights_dir, model_size)

    model = YOLO(weights_path)
    logger.info(f"Loaded model weights: {weights_path}")

    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {source_video_path}")
        raise FileNotFoundError(f"Cannot open video: {source_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = width_orig
    height = height_orig
    logger.info(f"Video opened: fps={fps:.2f}, size={width_orig}x{height_orig}, downsample={downsample}")

    # Apply optional uniform downsampling (1=no downsample; 2=1/2; 3=1/3; 4=1/4)
    if downsample is None:
        downsample = 1
    if downsample not in (1, 2, 3, 4):
        downsample = 1
    if downsample > 1:
        width = max(1, int(round(width / downsample)))
        height = max(1, int(round(height / downsample)))

    # Mirror path under results: results/<input_path_without_leading_dataset_dir>/<output_name>
    input_path_norm = source_video_path.replace("\\", "/")
    # Strip leading ./ and leading dataset/ if present
    trimmed = input_path_norm
    if trimmed.startswith("./"):
        trimmed = trimmed[2:]
    if trimmed.startswith("dataset/"):
        trimmed = trimmed[len("dataset/"):]
    trimmed_dir = os.path.dirname(trimmed)
    out_folder = os.path.join(results_dir, trimmed_dir)
    os.makedirs(out_folder, exist_ok=True)
    if output_name is None:
        base = os.path.splitext(os.path.basename(source_video_path))[0]
        ext = os.path.splitext(source_video_path)[1] or ".mp4"
        output_name = f"{base}_fall_detected{ext}"
    out_path = os.path.join(out_folder, output_name)
    logger.info(f"Output will be written to: {out_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Check codec availability.")

    smoother = TemporalSmoother(alpha=0.7)

    # Basic nearest-neighbor tracker by bbox center to keep IDs consistent frame-to-frame
    active_tracks = []  # list of (track_id, center_x, center_y)
    max_match_dist = max(20, int(0.05 * max(width, height)))
    prev_falling_by_track = {}

    try:
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame if downsampling is enabled
            if downsample > 1:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False,
                device=device,
                half=False,
                stream=False,
            )

            # Ultralytics returns a list of Results even for single image
            if not results:
                writer.write(frame)
                if log_every > 0 and (frame_index % max(1, log_every) == 0):
                    logger.debug(f"frame={frame_index}: no model results")
                frame_index += 1
                continue

            res = results[0]

            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
            kps = None
            if getattr(res, "keypoints", None) is not None and res.keypoints is not None:
                # shape: (num, num_kpts, 3)
                kps = res.keypoints.data.cpu().numpy()
            else:
                kps = np.empty((0, 17, 3))

            scores = []
            updated_tracks = []

            # match detections to active tracks via center proximity
            det_centers = []
            for b in boxes:
                cx = 0.5 * (b[0] + b[2])
                cy = 0.5 * (b[1] + b[3])
                det_centers.append((float(cx), float(cy)))

            assigned = [-1] * len(det_centers)
            used_tracks = set()
            # try to match existing tracks first
            for t_idx, (tid, tx, ty) in enumerate(active_tracks):
                best_det = -1
                best_dist = 1e9
                for di, (cx, cy) in enumerate(det_centers):
                    if assigned[di] != -1:
                        continue
                    d = math.hypot(cx - tx, cy - ty)
                    if d < best_dist:
                        best_dist = d
                        best_det = di
                if best_det != -1 and best_dist <= max_match_dist:
                    assigned[best_det] = tid
                    used_tracks.add(tid)
                    updated_tracks.append((tid, det_centers[best_det][0], det_centers[best_det][1]))

            # create new tracks for unmatched detections
            for di, (cx, cy) in enumerate(det_centers):
                if assigned[di] == -1:
                    tid = smoother.assign_id()
                    assigned[di] = tid
                    updated_tracks.append((tid, cx, cy))

            active_tracks = updated_tracks

            # Render per detection
            per_frame_fall_count = 0
            for i, box in enumerate(boxes):
                tid = assigned[i]
                kp = kps[i] if i < len(kps) else None
                if kp is None:
                    continue

                prev_orientation = smoother.get_prev_orientation(tid)
                score, debug = compute_fall_score(kp, tuple(box.tolist()), prev_orientation)

                # Update prev orientation with current orientation angle for next frame smoothing
                smoother.set_orientation(tid, debug.get("orientation_angle"))
                smoothed_score = smoother.smooth_score(tid, score)

                falling = smoothed_score >= 0.6  # threshold tuned empirically
                if falling:
                    per_frame_fall_count += 1

                # Fall state transitions per track
                prev_state = prev_falling_by_track.get(tid, False)
                if falling != prev_state:
                    state_str = "FALL STARTED" if falling else "fall cleared"
                    logger.info(
                        f"frame={frame_index} track={tid} {state_str}: smoothed={smoothed_score:.2f}, "
                        f"raw={score:.2f}, ar={debug.get('bbox_ar'):.2f}, ori={debug.get('orientation_angle') if debug.get('orientation_angle') is not None else 'nan'}, "
                        f"upper_horiz={debug.get('upper_body_horiz_angle') if debug.get('upper_body_horiz_angle') is not None else 'nan'}, "
                        f"scores={{ar:{debug.get('ar_score'):.2f}, ori:{debug.get('orientation_score'):.2f}, upper:{debug.get('upper_body_horizontal_score'):.2f}, "
                        f"head:{debug.get('head_near_ankle'):.2f}, chg:{debug.get('orientation_change_score'):.2f}, bonus:{debug.get('bbox_rule_bonus'):.2f}}}"
                    )
                prev_falling_by_track[tid] = falling

                color = (0, 255, 0) if not falling else (0, 0, 255)
                x1i, y1i, x2i, y2i = xyxy_int(tuple(box.tolist()), width, height)
                draw_rounded_rectangle(frame, (x1i, y1i), (x2i, y2i), color, thickness=3, radius=12)
                status_text = "Falling" if falling else "No Falling"
                put_label_above_box(frame, status_text, (x1i, y1i, x2i, y2i), (255, 255, 255))

                # Draw keypoints and skeleton
                pts = [(int(p[0]), int(p[1]), float(p[2])) for p in kp]
                for (x, y, c) in pts:
                    if c >= 0.3:
                        cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
                for a, b in COCO_SKELETON_EDGES:
                    xa, ya, ca = pts[a]
                    xb, yb, cb = pts[b]
                    if ca >= 0.3 and cb >= 0.3:
                        cv2.line(frame, (xa, ya), (xb, yb), (0, 255, 255), 2)

                # Periodic per-detection debug logs
                if log_every > 0 and (frame_index % max(1, log_every) == 0):
                    logger.debug(
                        f"frame={frame_index} track={tid} det: box=({x1i},{y1i},{x2i},{y2i}) ar={debug.get('bbox_ar'):.2f} "
                        f"raw={score:.2f} smoothed={smoothed_score:.2f} falling={falling}"
                    )

            writer.write(frame)
            # Periodic per-frame summary
            if log_every > 0 and (frame_index % max(1, log_every) == 0):
                logger.debug(
                    f"frame={frame_index} summary: dets={len(boxes)}, falling={per_frame_fall_count}"
                )
            if show:
                cv2.imshow("Fall Detection (Pose)", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    return out_path



DEFAULT_INPUT = "./dataset/Le2i/Home_01/Home_01/Videos/video (1).avi"


def main(argv: Optional[List[str]] = None) -> None:
    raw_args = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(description="Fall detection with YOLO pose")
    parser.add_argument(
        "-ds", "--downsample",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Uniform downsampling factor: 1 (no downsample), 2 (1/2), 3 (1/3), 4 (1/4).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to write logs to a file.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Emit per-frame summary and per-detection debug logs every N frames (0=disable).",
    )
    known, unknown = parser.parse_known_args(raw_args)

    if not unknown:
        inp = DEFAULT_INPUT
    else:
        # Join remaining tokens to support paths with spaces (e.g., "video (1).avi")
        inp = " ".join(unknown).strip()

    setup_logging(level=known.log_level, log_file=known.log_file)
    logger = logging.getLogger("fall_detection")

    base_name = os.path.splitext(os.path.basename(inp))[0]
    out_name = f"{base_name}_fall_detected.avi"

    start = time.time()
    # You can change model_size to one of {"n","s","m","l"}
    out_path = process_video(
        inp,
        weights_dir="weights",
        results_dir="results",
        output_name=out_name,
        show=True,
        model_size="m",
        downsample=known.downsample,
        log_every=known.log_every,
    )
    dur = time.time() - start
    logger.info(f"Saved result video to: {out_path} (processed in {dur:.2f}s)")


if __name__ == "__main__":
    main()


