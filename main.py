import os
import sys
import time
from typing import Optional, List
import argparse
import logging

from fall_pose_utils import setup_logging
from fall_detector import process_video


DEFAULT_INPUT = "./dataset/Real/fall/video1.mp4"



def main(argv: Optional[List[str]] = None) -> None:
    raw_args = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(description="Fall detection with YOLO pose")
    parser.add_argument(
        "-ds", "--downsample",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        default=4,
        help="Uniform downsampling factor: 1 (no DS) ... 8 (1/8).",
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
        default=30,
        help="Emit per-frame summary and per-detection debug logs every N frames (0=disable).",
    )
    parser.add_argument(
        "--seg-size",
        type=str,
        choices=["n", "s", "m", "l", "x"],
        default="n",
        help="Segmentation model size for bed detection: n, s, m, l, x.",
    )
    parser.add_argument(
        "--seg-imgsz",
        type=int,
        default=640,
        help="Segmentation model inference size for bed detection.",
    )
    parser.add_argument(
        "--bed-center-offset",
        nargs=2,
        type=int,
        metavar=("DX", "DY"),
        default=[20, -30],
        help="Manual offset (pixels) added to bed center: DX DY. Clipped to image bounds.",
    )
    parser.add_argument(
        "--visual-center",
        action="store_true",
        help="If set, visualize A/B centers and the small B-centered rectangle.",
    )
    parser.add_argument(
        "--visual-mask",
        action="store_true",
        help="If set, overlay the detected bed mask on output frames.",
    )
    parser.add_argument(
        "--bed-center-box-w-scale",
        type=float,
        default=0.8,
        help="Width scale for small bed-centered bbox (fraction of bed bbox width).",
    )
    parser.add_argument(
        "--bed-center-box-h-scale",
        type=float,
        default=0.6,
        help="Height scale for small bed-centered bbox (fraction of bed bbox height).",
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
        seg_size=known.seg_size,
        seg_imgsz=known.seg_imgsz,
        bed_center_offset=tuple(known.bed_center_offset),
        visual_center=known.visual_center,
        visual_mask=known.visual_mask,
        bed_center_box_w_scale=known.bed_center_box_w_scale,
        bed_center_box_h_scale=known.bed_center_box_h_scale,
    )
    dur = time.time() - start
    logger.info(f"Saved result video to: {out_path} (processed in {dur:.2f}s)")


if __name__ == "__main__":
    main()


