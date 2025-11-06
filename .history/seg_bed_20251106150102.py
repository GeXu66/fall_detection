import os
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Ultralytics is required. Install with: pip install ultralytics"
    ) from exc


def locate_or_download_seg_weights(weights_dir: str, model_size: str = "n") -> str:
    size = (model_size or "n").lower()
    if size not in {"n", "s", "m", "l", "x"}:
        size = "n"
    model_name = f"yolo11{size}-seg.pt"
    local_path = os.path.join(weights_dir, model_name)
    if os.path.isfile(local_path):
        return local_path

    # Attempt direct download via ultralytics' internal resolution
    try:
        _ = YOLO(model_name)
        return model_name
    except Exception as exc:
        raise RuntimeError(
            f"Failed to obtain segmentation weights '{model_name}'. Place it under '{weights_dir}/' or ensure internet connectivity."
        ) from exc


def load_seg_model(weights_dir: str, model_size: str = "n") -> YOLO:
    weights_path = locate_or_download_seg_weights(weights_dir, model_size)
    return YOLO(weights_path)


def _fill_polys(mask: np.ndarray, polys: list) -> None:
    if polys is None:
        return
    if isinstance(polys, list):
        for poly in polys:
            if poly is None or len(poly) == 0:
                continue
            pts = np.round(np.asarray(poly)).astype(np.int32)
            if pts.ndim == 2 and pts.shape[1] == 2:
                cv2.fillPoly(mask, [pts], 255)
    else:
        pts = np.round(np.asarray(polys)).astype(np.int32)
        if pts.ndim == 2 and pts.shape[1] == 2:
            cv2.fillPoly(mask, [pts], 255)


def detect_bed_mask(
    seg_model: YOLO,
    frame_bgr: np.ndarray,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
) -> Optional[np.ndarray]:
    if seg_model is None or frame_bgr is None:
        return None

    h, w = frame_bgr.shape[:2]
    results = seg_model.predict(
        source=frame_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
        half=False,
        stream=False,
    )
    if not results:
        return None

    res = results[0]
    if getattr(res, "masks", None) is None or res.masks is None:
        return None
    if getattr(res, "boxes", None) is None or res.boxes is None:
        return None

    names = getattr(res, "names", None)
    if names is None:
        names = getattr(seg_model, "names", None)
    if names is None:
        # Fallback to COCO names, but if unavailable just skip
        return None

    cls = res.boxes.cls.cpu().numpy().astype(int)
    mask_out = np.zeros((h, w), dtype=np.uint8)

    # res.masks.xy is a list of polygons per instance in image coordinates
    polys_list = getattr(res.masks, "xy", None)
    if polys_list is None:
        # As a fallback, try binary mask tensors (N,H,W) with potential letterboxing
        data = getattr(res.masks, "data", None)
        if data is None:
            return None
        # Resize each mask to the frame size and OR them if class==bed
        masks = data.cpu().numpy()  # (N, mh, mw)
        for i, c in enumerate(cls):
            name = names.get(int(c), str(int(c))) if isinstance(names, dict) else str(int(c))
            if name and "bed" in str(name).lower():
                m = (masks[i] > 0.5).astype(np.uint8) * 255
                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_out = np.maximum(mask_out, m_resized)
        return mask_out if np.count_nonzero(mask_out) > 0 else None

    for i, c in enumerate(cls):
        name = names.get(int(c), str(int(c))) if isinstance(names, dict) else str(int(c))
        if name and "bed" in str(name).lower():
            polys = polys_list[i]
            _fill_polys(mask_out, polys)

    return mask_out if np.count_nonzero(mask_out) > 0 else None


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.3) -> None:
    if frame_bgr is None or mask is None:
        return
    h, w = frame_bgr.shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        m = mask
    m_bool = m > 0
    if not np.any(m_bool):
        return
    overlay = frame_bgr.copy()
    overlay[m_bool] = (
        (1.0 - alpha) * overlay[m_bool] + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    frame_bgr[m_bool] = overlay[m_bool]


