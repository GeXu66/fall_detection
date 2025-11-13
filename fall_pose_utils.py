import math
from typing import List, Tuple, Optional
import logging

import cv2
import numpy as np


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
    if bbox_ar > 0.65:
        # Grows linearly from 0 when ar==0.6 up to a small cap
        bbox_rule_bonus = bbox_ar

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


def xyxy_int(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(int(round(x1)), w - 1)),
        max(0, min(int(round(y1)), h - 1)),
        max(0, min(int(round(x2)), w - 1)),
        max(0, min(int(round(y2)), h - 1)),
    )


