import os
import math
from typing import Optional, Tuple
import logging

import cv2
import numpy as np
from seg_bed import load_seg_model, detect_bed_mask, overlay_mask

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Ultralytics is required. Install with: pip install ultralytics"
    ) from exc

from fall_pose_utils import (
    draw_rounded_rectangle,
    put_label_above_box,
    COCO_KP_NAMES,
    COCO_SKELETON_EDGES,
    compute_fall_score,
    TemporalSmoother,
    xyxy_int,
)


ULTRA_ASSET_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"


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
    seg_size: str = "n",
    seg_imgsz: int = 640,
    bed_center_offset: Tuple[int, int] = (0, 0),
    visual_center: bool = False,
    visual_mask: bool = False,
    bed_center_box_w_scale: float = 0.8,
    bed_center_box_h_scale: float = 0.7,
) -> str:
    logger = logging.getLogger("fall_detection")
    ensure_dirs(weights_dir, results_dir)
    weights_path = locate_or_download_weights(weights_dir, model_size)

    model = YOLO(weights_path)
    logger.info(f"Loaded model weights: {weights_path}")

    def scale_bbox_to_frame(
        bbox: Optional[Tuple[int, int, int, int]],
        src_size: Optional[Tuple[int, int]],
        dst_size: Tuple[int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Scale a bbox defined on src_size into dst_size coordinates."""
        if bbox is None:
            return None
        if not src_size or not dst_size:
            return bbox
        src_w, src_h = src_size
        dst_w, dst_h = dst_size
        if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
            return bbox
        if src_w == dst_w and src_h == dst_h:
            return bbox
        sx = dst_w / float(src_w)
        sy = dst_h / float(src_h)
        x1, y1, x2, y2 = bbox
        scaled = (
            int(round(x1 * sx)),
            int(round(y1 * sy)),
            int(round(x2 * sx)),
            int(round(y2 * sy)),
        )
        x1s = max(0, min(scaled[0], dst_w - 1))
        y1s = max(0, min(scaled[1], dst_h - 1))
        x2s = max(0, min(scaled[2], dst_w - 1))
        y2s = max(0, min(scaled[3], dst_h - 1))
        if x2s < x1s:
            x1s, x2s = x2s, x1s
        if y2s < y1s:
            y1s, y2s = y2s, y1s
        return (x1s, y1s, x2s, y2s)

    # Load segmentation model once (for bed detection)
    seg_model = None
    try:
        seg_model = load_seg_model(weights_dir, seg_size)
        logger.info(f"Loaded segmentation weights (bed): size={seg_size}")
    except Exception as exc:
        logger.warning(f"Segmentation model unavailable: {exc}")

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
    log_sec_interval = max(1, int(round(fps)))

    # Apply optional uniform downsampling (1=no downsample; 2=1/2; 3=1/3; 4=1/4)
    if downsample is None:
        downsample = 1
    if downsample not in (1, 2, 3, 4, 5, 6, 7, 8):
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

    # Normalize bed_center_offset to two ints
    try:
        if isinstance(bed_center_offset, (list, tuple)) and len(bed_center_offset) >= 2:
            bed_dx = int(bed_center_offset[0])
            bed_dy = int(bed_center_offset[1])
        else:
            bed_dx, bed_dy = 0, 0
    except Exception:
        bed_dx, bed_dy = 0, 0

    smoother = TemporalSmoother(alpha=0.7)

    # Basic nearest-neighbor tracker by bbox center to keep IDs consistent frame-to-frame
    active_tracks = []  # list of (track_id, center_x, center_y)
    max_match_dist = max(20, int(0.05 * max(width, height)))
    prev_falling_by_track = {}
    prev_may_by_track = {}
    may_streak_frames_by_track = {}

    try:
        frame_index = 0
        bed_mask = None
        bed_bbox = None  # (x1, y1, x2, y2) in frame coordinates
        bed_bbox_frame_size: Optional[Tuple[int, int]] = None
        paused = False
        last_display_frame = None
        while True:
            if paused:
                if show and last_display_frame is not None:
                    paused_frame = last_display_frame.copy()
                    cv2.putText(
                        paused_frame,
                        "Paused - press Q to resume",
                        (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.imshow("Fall Detection (Pose)", paused_frame)
                    key = cv2.waitKey(100) & 0xFF
                    if key == 27:
                        break
                    if key in (ord('q'), ord('Q')):
                        paused = False
                        logger.info("Resumed playback. Press 'q' to pause again or ESC to exit.")
                        continue
                else:
                    paused = False
                continue
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame if downsampling is enabled
            if downsample > 1:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            frame_h, frame_w = frame.shape[:2]

            # On the very first frame (after any resizing), run bed segmentation once
            if frame_index == 0 and seg_model is not None and bed_mask is None:
                try:
                    bed_mask = detect_bed_mask(seg_model, frame, imgsz=seg_imgsz, conf=0.3, iou=0.45)
                    if bed_mask is not None:
                        # Compute bed mask bounding box once
                        ys, xs = np.where(bed_mask > 0)
                        if xs.size > 0 and ys.size > 0:
                            bed_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                            bed_bbox_frame_size = (frame_w, frame_h)
                            logger.info(f"Bed mask detected on first frame. bbox={bed_bbox}")
                        else:
                            bed_bbox = None
                            bed_bbox_frame_size = None
                            logger.info("Bed mask detected on first frame, but empty area.")
                    else:
                        logger.info("No bed detected on first frame.")
                except Exception as exc:
                    logger.warning(f"Bed segmentation failed: {exc}")
                    bed_mask = None
                    bed_bbox = None
                    bed_bbox_frame_size = None

            bed_bbox_current = None
            if bed_bbox is not None:
                bed_bbox_current = scale_bbox_to_frame(
                    bed_bbox,
                    bed_bbox_frame_size,
                    (frame_w, frame_h),
                )

            # Overlay bed mask if requested
            if bed_mask is not None and visual_mask:
                overlay_mask(frame, bed_mask, color=(255, 0, 0), alpha=0.25)

            # Draw B point and small B-centered rectangle if requested
            if bed_bbox_current is not None and visual_center:
                bed_x1, bed_y1, bed_x2, bed_y2 = bed_bbox_current
                bx = int((bed_x1 + bed_x2) * 0.5) + bed_dx
                by = int((bed_y1 + bed_y2) * 0.5) + bed_dy
                bx = max(0, min(bx, width - 1))
                by = max(0, min(by, height - 1))
                cv2.circle(frame, (bx, by), 6, (255, 0, 255), -1)
                cv2.putText(
                    frame,
                    "B",
                    (bx + 6, max(0, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                    lineType=cv2.LINE_AA,
                )
                # Draw small bed-centered bbox (visual aid)
                bed_w = max(1, bed_x2 - bed_x1)
                bed_h = max(1, bed_y2 - bed_y1)
                half_w = int(0.5 * bed_w * float(bed_center_box_w_scale))
                half_h = int(0.5 * bed_h * float(bed_center_box_h_scale))
                rx1 = max(0, min(bx - half_w, width - 1))
                ry1 = max(0, min(by - half_h, height - 1))
                rx2 = max(0, min(bx + half_w, width - 1))
                ry2 = max(0, min(by + half_h, height - 1))
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (180, 0, 180), 2)

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
                last_display_frame = frame
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

                # If too few reliable keypoints, mark as Image Incomplete and skip fall logic
                try:
                    valid_kp_count = int(np.sum(kp[:, 2] >= 0.3))
                except Exception:
                    valid_kp_count = 0
                min_kp_required = 6
                if valid_kp_count < min_kp_required:
                    x1i, y1i, x2i, y2i = xyxy_int(tuple(box.tolist()), width, height)
                    ax = int((x1i + x2i) * 0.5)
                    ay = int((y1i + y2i) * 0.5)
                    if visual_center:
                        cv2.circle(frame, (ax, ay), 5, (0, 255, 255), -1)
                        cv2.putText(
                            frame,
                            "A",
                            (ax + 6, max(0, ay - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                            lineType=cv2.LINE_AA,
                        )
                    draw_rounded_rectangle(frame, (x1i, y1i), (x2i, y2i), (128, 128, 128), thickness=3, radius=12)
                    put_label_above_box(frame, "Image Incomplete", (x1i, y1i, x2i, y2i), (255, 255, 255))
                    if log_every > 0 and (frame_index % max(1, log_every) == 0):
                        logger.debug(
                            f"frame={frame_index} track={tid} image_incomplete: valid_kp={valid_kp_count} < {min_kp_required}"
                        )
                    continue

                prev_orientation = smoother.get_prev_orientation(tid)
                score, debug = compute_fall_score(kp, tuple(box.tolist()), prev_orientation)

                # Update prev orientation with current orientation angle for next frame smoothing
                smoother.set_orientation(tid, debug.get("orientation_angle"))
                smoothed_score = smoother.smooth_score(tid, score)

                # Base rule from score -> may fall
                may_fall_base = smoothed_score >= 0.6  # threshold tuned empirically
                
                # New NO-FALL rule (cleaning floor posture):
                # For each side, define:
                #   a1 = vector(left_shoulder -> left_hip), b1 = vector(left_hip -> left_knee), c1 = vector(left_knee -> left_ankle)
                # If angle(a1, b1) and angle(b1, c1) are both acute (< 90 deg), mark as no falling.
                # Apply similarly for the right side; if either side satisfies, treat as no fall.
                no_fall_by_cleaning_floor = False
                try:
                    # Indices
                    ls = COCO_KP_NAMES.index("left_shoulder")
                    lh = COCO_KP_NAMES.index("left_hip")
                    lk = COCO_KP_NAMES.index("left_knee")
                    la = COCO_KP_NAMES.index("left_ankle")
                    rs = COCO_KP_NAMES.index("right_shoulder")
                    rh = COCO_KP_NAMES.index("right_hip")
                    rk = COCO_KP_NAMES.index("right_knee")
                    ra = COCO_KP_NAMES.index("right_ankle")

                    def kp_ok(idx: int) -> bool:
                        return float(kp[idx][2]) >= 0.3

                    def vec(i_from: int, i_to: int):
                        return (float(kp[i_to][0]) - float(kp[i_from][0]), float(kp[i_to][1]) - float(kp[i_from][1]))

                    def angle_deg(v1, v2):
                        vx1, vy1 = v1
                        vx2, vy2 = v2
                        n1 = math.hypot(vx1, vy1)
                        n2 = math.hypot(vx2, vy2)
                        if n1 <= 1e-6 or n2 <= 1e-6:
                            return None
                        cosang = max(-1.0, min(1.0, (vx1 * vx2 + vy1 * vy2) / (n1 * n2)))
                        return math.degrees(math.acos(cosang))

                    # Initialize for debug prints even if some branches are skipped
                    ang_ab = None
                    ang_bc = None
                    ang_ab_r = None
                    ang_bc_r = None

                    left_ok = False
                    if kp_ok(ls) and kp_ok(lh) and kp_ok(lk) and kp_ok(la):
                        # Hip angle: vectors from hip to shoulder and hip to knee
                        a1 = vec(lh, ls)
                        b1 = vec(lh, lk)
                        # Knee angle: vectors from knee to hip and knee to ankle
                        c1 = vec(lk, la)
                        ang_ab = angle_deg(a1, b1)
                        ang_bc = angle_deg(vec(lk, lh), c1)
                        if ang_ab is not None and ang_bc is not None:
                            if (ang_ab < 90.0) and (ang_bc < 90.0):
                                left_ok = True

                    right_ok = False
                    if kp_ok(rs) and kp_ok(rh) and kp_ok(rk) and kp_ok(ra):
                        # Hip angle: vectors from hip to shoulder and hip to knee
                        a1r = vec(rh, rs)
                        b1r = vec(rh, rk)
                        # Knee angle: vectors from knee to hip and knee to ankle
                        c1r = vec(rk, ra)
                        ang_ab_r = angle_deg(a1r, b1r)
                        ang_bc_r = angle_deg(vec(rk, rh), c1r)
                        if ang_ab_r is not None and ang_bc_r is not None:
                            if (ang_ab_r < 90.0) and (ang_bc_r < 90.0):
                                right_ok = True


                    no_fall_by_cleaning_floor = left_ok or right_ok
                    if no_fall_by_cleaning_floor:
                        if frame_index % 25 == 0:
                            print('--------------------------------')
                            print('angle_ab', ang_ab)
                            print('angle_bc', ang_bc)
                            print('angle_ab_r', ang_ab_r)
                            print('angle_bc_r', ang_bc_r)
                            print('left_ok', left_ok)
                            print('right_ok', right_ok)
                except Exception:
                    no_fall_by_cleaning_floor = False
                if no_fall_by_cleaning_floor:
                    # Require at least 3 reliable face keypoints to trust this rule
                    face_kps = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
                    reliable_face_points = 0
                    for name in face_kps:
                        try:
                            idx = COCO_KP_NAMES.index(name)
                            if float(kp[idx][2]) >= 0.3:
                                reliable_face_points += 1
                        except Exception:
                            continue
                    if reliable_face_points < 3:
                        no_fall_by_cleaning_floor = False

                # New rule: ankle above hip AND hip above shoulder implies may fall
                ankle_over_hip_and_hip_over_shoulder = False
                try:
                    ls = COCO_KP_NAMES.index("left_shoulder")
                    lh = COCO_KP_NAMES.index("left_hip")
                    la = COCO_KP_NAMES.index("left_ankle")
                    rs = COCO_KP_NAMES.index("right_shoulder")
                    rh = COCO_KP_NAMES.index("right_hip")
                    ra = COCO_KP_NAMES.index("right_ankle")

                    def side_inverted(s, h, a):
                        if float(kp[s][2]) < 0.3 or float(kp[h][2]) < 0.3 or float(kp[a][2]) < 0.3:
                            return False
                        y_s = float(kp[s][1])
                        y_h = float(kp[h][1])
                        y_a = float(kp[a][1])
                        box_h_local = max(1.0, float(box[3] - box[1]))
                        margin_px = 0.02 * box_h_local
                        return (y_a + margin_px) < y_h and (y_h + margin_px) < y_s

                    ankle_over_hip_and_hip_over_shoulder = side_inverted(ls, lh, la) or side_inverted(rs, rh, ra)
                except Exception:
                    ankle_over_hip_and_hip_over_shoulder = False

                # New NO-FALL rule: shoulder midpoint below hip midpoint but above lowest ankle midpoint
                no_fall_by_shoulder_between = False
                try:
                    ls_idx = COCO_KP_NAMES.index("left_shoulder")
                    rs_idx = COCO_KP_NAMES.index("right_shoulder")
                    lh_idx = COCO_KP_NAMES.index("left_hip")
                    rh_idx = COCO_KP_NAMES.index("right_hip")
                    la_idx = COCO_KP_NAMES.index("left_ankle")
                    ra_idx = COCO_KP_NAMES.index("right_ankle")

                    s_ys = [float(kp[idx][1]) for idx in (ls_idx, rs_idx) if float(kp[idx][2]) >= 0.3]
                    h_ys = [float(kp[idx][1]) for idx in (lh_idx, rh_idx) if float(kp[idx][2]) >= 0.3]
                    a_ys = [float(kp[idx][1]) for idx in (la_idx, ra_idx) if float(kp[idx][2]) >= 0.3]

                    if s_ys and h_ys and a_ys:
                        shoulder_mid_y = sum(s_ys) / len(s_ys)
                        hip_mid_y = sum(h_ys) / len(h_ys)
                        ankle_lowest_y = max(a_ys)  # lowest point in image has largest y
                        if (shoulder_mid_y > hip_mid_y) and (shoulder_mid_y < ankle_lowest_y):
                            no_fall_by_shoulder_between = True
                except Exception:
                    no_fall_by_shoulder_between = False



                # New NO-FALL rule: angle(hip->shoulder, hip->knee) < 90 deg AND wrist-ankle distance is small
                no_fall_by_angle_and_wrist_ankle = False
                try:
                    # Left side
                    ls = COCO_KP_NAMES.index("left_shoulder")
                    lh = COCO_KP_NAMES.index("left_hip")
                    lk = COCO_KP_NAMES.index("left_knee")
                    lw = COCO_KP_NAMES.index("left_wrist")
                    la = COCO_KP_NAMES.index("left_ankle")
                    # Right side
                    rs = COCO_KP_NAMES.index("right_shoulder")
                    rh = COCO_KP_NAMES.index("right_hip")
                    rk = COCO_KP_NAMES.index("right_knee")
                    rw = COCO_KP_NAMES.index("right_wrist")
                    ra = COCO_KP_NAMES.index("right_ankle")

                    def side_ok(s, h, k, w_, a_):
                        if float(kp[s][2]) < 0.3 or float(kp[h][2]) < 0.3 or float(kp[k][2]) < 0.3 or float(kp[w_][2]) < 0.3 or float(kp[a_][2]) < 0.3:
                            return False
                        vx1 = float(kp[s][0]) - float(kp[h][0])
                        vy1 = float(kp[s][1]) - float(kp[h][1])
                        vx2 = float(kp[k][0]) - float(kp[h][0])
                        vy2 = float(kp[k][1]) - float(kp[h][1])
                        n1 = math.hypot(vx1, vy1)
                        n2 = math.hypot(vx2, vy2)
                        if n1 <= 1e-6 or n2 <= 1e-6:
                            return False
                        cosang = max(-1.0, min(1.0, (vx1 * vx2 + vy1 * vy2) / (n1 * n2)))
                        ang = math.degrees(math.acos(cosang))
                        # wrist-ankle distance threshold relative to person bbox height
                        dist = math.hypot(float(kp[w_][0]) - float(kp[a_][0]), float(kp[w_][1]) - float(kp[a_][1]))
                        box_h_local = max(1.0, float(box[3] - box[1]))
                        close_thr = 0.25 * box_h_local
                        return (ang < 90.0) and (dist <= close_thr)

                    no_fall_by_angle_and_wrist_ankle = side_ok(ls, lh, lk, lw, la) or side_ok(rs, rh, rk, rw, ra)
                except Exception:
                    no_fall_by_angle_and_wrist_ankle = False
                # New NO-FALL rule: both wrists close to their corresponding ankles (relative to bbox width)
                no_fall_by_wrist_ankle_width = False
                try:
                    lw = COCO_KP_NAMES.index("left_wrist")
                    la = COCO_KP_NAMES.index("left_ankle")
                    rw = COCO_KP_NAMES.index("right_wrist")
                    ra = COCO_KP_NAMES.index("right_ankle")

                    left_valid = float(kp[lw][2]) >= 0.3 and float(kp[la][2]) >= 0.3
                    right_valid = float(kp[rw][2]) >= 0.3 and float(kp[ra][2]) >= 0.3
                    if left_valid and right_valid:
                        left_dist = math.hypot(float(kp[lw][0]) - float(kp[la][0]), float(kp[lw][1]) - float(kp[la][1]))
                        right_dist = math.hypot(float(kp[rw][0]) - float(kp[ra][0]), float(kp[rw][1]) - float(kp[ra][1]))
                        box_w_local = max(1.0, float(box[2] - box[0]))
                        dist_limit = 0.3 * box_w_local
                        no_fall_by_wrist_ankle_width = (left_dist <= dist_limit) and (right_dist <= dist_limit)
                except Exception:
                    no_fall_by_wrist_ankle_width = False
                
                # If bbox is significantly wider than tall, do not allow no-fall suppressors to trigger
                box_w_current = max(1.0, float(box[2] - box[0]))
                box_h_current = max(1.0, float(box[3] - box[1]))
                if (box_w_current / box_h_current) > 1.3:
                    no_fall_by_shoulder_between = False
                    no_fall_by_angle_and_wrist_ankle = False
                    no_fall_by_wrist_ankle_width = False
                
                # New NO-FALL rule: tall bbox and its (height + width) exceeds image height
                no_fall_by_tall_large = False
                try:
                    bw = float(max(1.0, box[2] - box[0]))
                    bh = float(max(1.0, box[3] - box[1]))
                    if (bw < bh) and ((bw + bh) > float(height)):
                        no_fall_by_tall_large = True
                except Exception:
                    no_fall_by_tall_large = False

                if ankle_over_hip_and_hip_over_shoulder:
                    # Give priority: this posture implies May fall regardless of no-fall suppressors
                    may_fall = True
                else:
                    # Otherwise, apply suppressors to the base score rule
                    may_fall = (
                        may_fall_base
                        and (not no_fall_by_cleaning_floor)
                        and (not no_fall_by_shoulder_between)
                        and (not no_fall_by_angle_and_wrist_ankle)
                        and (not no_fall_by_wrist_ankle_width)
                        and (not no_fall_by_tall_large)
                    )
                # if frame_index % 25 == 0:
                #     print('may_fall_base', may_fall_base)
                #     print('no_fall_by_angle_and_wrist_ankle', no_fall_by_angle_and_wrist_ankle)
                #     print('no_fall_by_shoulder_between', no_fall_by_shoulder_between)
                # Prepare A point and an early on-bed check to suppress escalation while on bed
                x1i, y1i, x2i, y2i = xyxy_int(tuple(box.tolist()), width, height)
                ax = int((x1i + x2i) * 0.5)
                ay = int((y1i + y2i) * 0.5)

                on_bed_for_escalation = False
                if may_fall and bed_bbox_current is not None:
                    bed_x1, bed_y1, bed_x2, bed_y2 = bed_bbox_current
                    bx_tmp = int((bed_x1 + bed_x2) * 0.5) + bed_dx
                    by_tmp = int((bed_y1 + bed_y2) * 0.5) + bed_dy
                    bx_tmp = max(0, min(bx_tmp, width - 1))
                    by_tmp = max(0, min(by_tmp, height - 1))
                    bed_w_tmp = max(1, bed_x2 - bed_x1)
                    bed_h_tmp = max(1, bed_y2 - bed_y1)
                    half_w_tmp = 0.5 * bed_w_tmp * float(bed_center_box_w_scale)
                    half_h_tmp = 0.5 * bed_h_tmp * float(bed_center_box_h_scale)
                    center_ok_tmp = (abs(float(ax) - float(bx_tmp)) <= half_w_tmp) and (abs(float(ay) - float(by_tmp)) <= half_h_tmp)
                    inter_x1 = max(x1i, bed_x1)
                    inter_y1 = max(y1i, bed_y1)
                    inter_x2 = min(x2i, bed_x2)
                    inter_y2 = min(y2i, bed_y2)
                    inter_area = 0.0
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
                    person_area = float(max(1, (x2i - x1i) * (y2i - y1i)))
                    overlap_ratio_tmp = inter_area / person_area
                    overlap_ok_tmp = overlap_ratio_tmp >= 0.6
                    on_bed_for_escalation = center_ok_tmp and overlap_ok_tmp

                # May fall streak for escalation (ignore on-bed frames)
                if may_fall and not on_bed_for_escalation:
                    streak = may_streak_frames_by_track.get(tid, 0) + 1
                    may_streak_frames_by_track[tid] = streak
                else:
                    may_streak_frames_by_track[tid] = 0

                # Escalation: continuous 3s of May fall
                needed_frames = int(round(3 * fps))
                escalate_by_streak = may_streak_frames_by_track.get(tid, 0) >= needed_frames
                falling = escalate_by_streak
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
                prev_may_by_track[tid] = may_fall

                # Colors: No Falling -> green, May Fall -> orange, Falling -> red, On Bed -> blue (set later)
                color = (0, 255, 0)
                x1i, y1i, x2i, y2i = xyxy_int(tuple(box.tolist()), width, height)
                # Compute A point (center of person bbox) and draw only if visualization is enabled
                ax = int((x1i + x2i) * 0.5)
                ay = int((y1i + y2i) * 0.5)
                if visual_center:
                    cv2.circle(frame, (ax, ay), 5, (0, 255, 255), -1)
                    cv2.putText(
                        frame,
                        "A",
                        (ax + 6, max(0, ay - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                # Classify On Bed vs Falling if bed mask exists and falling posture detected
                status_text = "No Falling"
                center_ok = None
                overlap_ok = None
                overlap_ratio = None
                center_dx = None
                center_dy = None
                half_w = None
                half_h = None
                on_bed = False
                if may_fall and bed_bbox_current is not None:
                    bed_x1, bed_y1, bed_x2, bed_y2 = bed_bbox_current
                    # Rule 1: center-in-rectangle check (rectangle centered at B)
                    bx = int((bed_x1 + bed_x2) * 0.5) + bed_dx
                    by = int((bed_y1 + bed_y2) * 0.5) + bed_dy
                    bx = max(0, min(bx, width - 1))
                    by = max(0, min(by, height - 1))
                    bed_w = max(1, bed_x2 - bed_x1)
                    bed_h = max(1, bed_y2 - bed_y1)
                    half_w = 0.5 * bed_w * float(bed_center_box_w_scale)
                    half_h = 0.5 * bed_h * float(bed_center_box_h_scale)
                    center_dx = abs(float(ax) - float(bx))
                    center_dy = abs(float(ay) - float(by))
                    center_ok = (center_dx <= half_w) and (center_dy <= half_h)

                    # Rule 2: bbox overlap area covers at least 70% of person bbox area
                    inter_x1 = max(x1i, bed_x1)
                    inter_y1 = max(y1i, bed_y1)
                    inter_x2 = min(x2i, bed_x2)
                    inter_y2 = min(y2i, bed_y2)
                    inter_area = 0.0
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
                    person_area = float(max(1, (x2i - x1i) * (y2i - y1i)))
                    overlap_ratio = inter_area / person_area
                    overlap_threshold = 0.6
                    overlap_ok = overlap_ratio >= overlap_threshold

                    on_bed = center_ok and overlap_ok
                    if on_bed:
                        status_text = "On Bed"
                        color = (255, 0, 0)
                    else:
                        if falling:
                            status_text = "Falling"
                            color = (0, 0, 255)
                        else:
                            status_text = "May Fall"
                            color = (0, 165, 255)
                    if log_every > 0 and (frame_index % max(1, log_every) == 0):
                        logger.debug(
                            f"frame={frame_index} track={tid} may_fall bed_check(2-rule): "
                            f"center_rect={center_ok} (|dx|={center_dx:.1f}<={half_w:.1f}, |dy|={center_dy:.1f}<={half_h:.1f}), "
                            f"overlap={overlap_ok} (ratio={overlap_ratio:.2f} >= {overlap_threshold}) -> {status_text}"
                        )
                else:
                    # not may fall
                    status_text = "Falling" if falling else "No Falling"
                    if falling:
                        color = (0, 0, 255)

                if (frame_index % log_sec_interval == 0):
                    streak_frames = may_streak_frames_by_track.get(tid, 0)
                    rule_summary_parts = [
                        f"fall_score>=0.6:{may_fall_base} (smoothed={smoothed_score:.2f})",
                        f"ankle_over_hip_then_hip_over_shoulder:{ankle_over_hip_and_hip_over_shoulder}",
                        f"no_fall_cleaning_floor:{no_fall_by_cleaning_floor}",
                        f"no_fall_shoulder_between:{no_fall_by_shoulder_between}",
                        f"no_fall_angle_wrist:{no_fall_by_angle_and_wrist_ankle}",
                        f"no_fall_tall_large:{no_fall_by_tall_large}",
                        f"no_fall_wrist_ankle_width:{no_fall_by_wrist_ankle_width}",
                        f"streak>=3s:{escalate_by_streak} ({streak_frames}/{needed_frames})",
                    ]
                    if center_ok is None or half_w is None or half_h is None or center_dx is None or center_dy is None:
                        rule_summary_parts.append("center_rect:n/a")
                    else:
                        rule_summary_parts.append(
                            f"center_rect:{center_ok} (dx={center_dx:.1f}/{half_w:.1f}, dy={center_dy:.1f}/{half_h:.1f})"
                        )
                    if overlap_ok is None or overlap_ratio is None:
                        rule_summary_parts.append("overlap>=0.6:n/a")
                    else:
                        rule_summary_parts.append(f"overlap>=0.6:{overlap_ok} (ratio={overlap_ratio:.2f})")
                    rule_summary_parts.append(f"on_bed:{on_bed}")
                    logger.info(
                        f"frame={frame_index} track={tid} state={status_text} may_fall={may_fall} "
                        f"falling={falling} | " + "; ".join(rule_summary_parts)
                    )

                draw_rounded_rectangle(frame, (x1i, y1i), (x2i, y2i), color, thickness=3, radius=12)
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
                        f"raw={score:.2f} smoothed={smoothed_score:.2f} falling={falling} ankle_hip_shoulder_inv={ankle_over_hip_and_hip_over_shoulder}"
                    )

            writer.write(frame)
            # Periodic per-frame summary
            if log_every > 0 and (frame_index % max(1, log_every) == 0):
                logger.debug(
                    f"frame={frame_index} summary: dets={len(boxes)}, falling={per_frame_fall_count}"
                )
            last_display_frame = frame
            if show:
                cv2.imshow("Fall Detection (Pose)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC to exit
                    break
                if key in (ord('q'), ord('Q')):
                    paused = True
                    logger.info("Paused playback. Press 'q' to resume or ESC to exit.")
            frame_index += 1
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    return out_path
