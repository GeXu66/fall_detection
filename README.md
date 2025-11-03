# Fall Detection (YOLO Pose)

A simple fall-detection demo using Ultralytics YOLO pose estimation and lightweight temporal smoothing.

## Environment
- Python 3.8+
- Windows, macOS, or Linux

## Install
```bash
# Create venv (recommended)
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install ultralytics opencv-python numpy
```

## Weights & Dataset
- Weights are auto-downloaded to `weights/` on first run if missing.
- The repo ignores `dataset/`, `weights/`, and common weight file types via `.gitignore`.

## Run
Use the new entrypoint. If the video path contains spaces, wrap it in quotes.
```bash
python main.py "./dataset/Le2i/Home_01/Home_01/Videos/video (1).avi"
```
Or omit the argument to use the default input:
```bash
python main.py
# defaults to ./dataset/Real/fall/video1.mp4
```

## Output
- Results are saved under a mirrored folder in `results/` with suffix `_fall_detected`.
- Example: input `dataset/Real/fall/video1.mp4` -> output `results/Real/fall/video1_fall_detected.avi`.
- A window named "Fall Detection (Pose)" previews the detection; press `ESC` to exit.

## Options
CLI flags:
```bash
python main.py \
  --downsample 4 \
  --log-level INFO \
  --log-file run.log \
  --log-every 30 \
  --seg-size n \
  --seg-imgsz 640 \
  --lie-threshold 0.2 \
  --bed-center-offset 0 0 \
  --visual-center \
  "./dataset/Le2i/Home_01/Home_01/Videos/video (1).avi"
```
- **downsample**: 1 (no DS) ... 8 (1/8)
- **log-level**: DEBUG | INFO | WARNING | ERROR | CRITICAL
- **log-file**: optional log file path
- **log-every**: debug frequency in frames (0 disables)
- **seg-size**: bed segmentation model size: n | s | m | l | x
- **seg-imgsz**: segmentation inference size
- **lie-threshold**: on-bed distance阈值，范围 [0,1]，表示 `阈值像素 = 值 * 图像宽度`。值越大，越容易被判定为 on bed（默认 0.2）。
- **bed-center-offset**: 手动修正床区中心（B 点）的偏移，单位像素，格式 `DX DY`，默认 `0 0`。正值向右/向下，负值向左/向上。应用后自动裁剪在图像范围内。
- **visual-center**: 开启后在结果视频上显示 A（人体中心）与 B（床中心）两点；不加该开关则不显示。

### On Bed 判定规则（三条件合一）
- **A 点**: 人体检测框的中心点。
- **B 点**: 床区 mask 的外接矩形框中心点（仅在首帧分割时计算），再加上 `bed-center-offset (DX,DY)` 偏移，并裁剪到图像范围内。A/B 的可视化受 `--visual-center` 控制。
- 当人物处于“falling”姿态，且同时满足以下三条时，判定为 **On Bed**，否则为 **Falling**：
  1) `dist(A,B) < lie_threshold * 图像宽度`
  2) `|person_y2 - bed_y2| < lie_threshold * 图像高度`
  3) `交叠面积 / 人体bbox面积 ≥ 70%`
  结果视频可在启用 `--visual-center` 时显示 A（黄）与 B（品红）。

示例：将 B 点向右下各偏移 100 像素
```bash
python main.py "./dataset/Real/fall/video1.mp4" --lie-threshold 0.25 --bed-center-offset 100 100
```

Model size for pose (n|s|m|l) is set inside `main.py` when calling `process_video` and can be adjusted if needed.

## Integrate as a library
You can import and use it in your own code:
```python
from fall_pose_utils import setup_logging
from fall_detector import process_video

setup_logging("INFO")
out_path = process_video(
    "./dataset/Real/fall/video1.mp4",
    results_dir="results",
    show=False,    # disable GUI in headless environments
    model_size="m",
    downsample=4,
)
print("Saved to:", out_path)
```

## Project structure
- `main.py`: CLI entrypoint.
- `fall_detector.py`: detection pipeline (`process_video`), weights handling.
- `fall_pose_utils.py`: logging, drawing, pose helpers, fall scoring, temporal smoothing.
- `seg_bed.py`: optional bed segmentation utilities (mask + overlay).
- `weights/`: YOLOv8 pose/seg weights (auto-downloaded if missing).

## Notes
- If you already tracked `dataset/` or `weights/` in git history, remove them from the index:
```bash
git rm -r --cached dataset weights
```
Then commit and push.