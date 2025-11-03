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
    lie_threshold: float = 0.2,
    bed_center_offset: Tuple[int, int] = (0, 0),
    visual_center: bool = False,
) -> str:
    logger = logging.getLogger("fall_detection")
    ensure_dirs(weights_dir, results_dir)
    weights_path = locate_or_download_weights(weights_dir, model_size)

    model = YOLO(weights_path)
    logger.info(f"Loaded model weights: {weights_path}")

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

    # Clamp lie_threshold to [0, 1]
    try:
        lie_threshold = float(lie_threshold)
    except Exception:
        lie_threshold = 0.2
    lie_threshold = max(0.0, min(1.0, lie_threshold))

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

    try:
        frame_index = 0
        bed_mask = None
        bed_bbox = None  # (x1, y1, x2, y2) in frame coordinates
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame if downsampling is enabled
            if downsample > 1:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # On the very first frame (after any resizing), run bed segmentation once
            if frame_index == 0 and seg_model is not None and bed_mask is None:
                try:
                    bed_mask = detect_bed_mask(seg_model, frame, imgsz=seg_imgsz, conf=0.3, iou=0.45)
                    if bed_mask is not None:
                        # Compute bed mask bounding box once
                        ys, xs = np.where(bed_mask > 0)
                        if xs.size > 0 and ys.size > 0:
                            bed_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                            logger.info(f"Bed mask detected on first frame. bbox={bed_bbox}")
                        else:
                            bed_bbox = None
                            logger.info("Bed mask detected on first frame, but empty area.")
                    else:
                        logger.info("No bed detected on first frame.")
                except Exception as exc:
                    logger.warning(f"Bed segmentation failed: {exc}")

            # Overlay bed mask if available
            # Only visualize bed mask and B when visual_center is enabled
            if bed_mask is not None and visual_center:
                overlay_mask(frame, bed_mask, color=(255, 0, 0), alpha=0.25)
                # Draw B point only if visualization is enabled
                if bed_bbox is not None:
                    bed_x1, bed_y1, bed_x2, bed_y2 = bed_bbox
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

            # (Removed legacy overlap/keypoint mask heuristics; using multi-rule (center/bottom/overlap) instead)

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
                if falling and bed_bbox is not None:
                    bed_x1, bed_y1, bed_x2, bed_y2 = bed_bbox
                    # Rule 1: center distance threshold (width-scaled)
                    bx = int((bed_x1 + bed_x2) * 0.5) + bed_dx
                    by = int((bed_y1 + bed_y2) * 0.5) + bed_dy
                    bx = max(0, min(bx, width - 1))
                    by = max(0, min(by, height - 1))
                    dist_px = math.hypot(float(ax - bx), float(ay - by))
                    center_thresh_px = lie_threshold * float(width)
                    center_ok = dist_px <= center_thresh_px

                    # Rule 2: bottom-edge y distance threshold (height-scaled)
                    bottom_diff_px = float(abs(y2i - bed_y2))
                    bottom_thresh_px = lie_threshold * float(height)
                    bottom_ok = bottom_diff_px <= bottom_thresh_px

                    # Rule 3: bbox overlap area covers at least 70% of person bbox area
                    inter_x1 = max(x1i, bed_x1)
                    inter_y1 = max(y1i, bed_y1)
                    inter_x2 = min(x2i, bed_x2)
                    inter_y2 = min(y2i, bed_y2)
                    inter_area = 0.0
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
                    person_area = float(max(1, (x2i - x1i) * (y2i - y1i)))
                    overlap_ratio = inter_area / person_area
                    overlap_ok = overlap_ratio >= 0.7

                    on_bed = center_ok and bottom_ok and overlap_ok
                    if on_bed:
                        status_text = "On Bed"
                        color = (255, 0, 0)
                    else:
                        status_text = "Falling"
                    if log_every > 0 and (frame_index % max(1, log_every) == 0):
                        logger.debug(
                            f"frame={frame_index} track={tid} bed_check(3-rule): "
                            f"center={center_ok} (dist={dist_px:.1f} <= {center_thresh_px:.1f}), "
                            f"bottom={bottom_ok} (|py2-bed_y2|={bottom_diff_px:.1f} <= {bottom_thresh_px:.1f}), "
                            f"overlap={overlap_ok} (ratio={overlap_ratio:.2f} >= 0.70) -> {status_text}"
                        )
                else:
                    status_text = "Falling" if falling else "No Falling"

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


