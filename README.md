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
Run the main script. If the video path contains spaces, wrap it in quotes.
```bash
python detector_fall_yolo_pose.py "./dataset/Le2i/Home_01/Home_01/Videos/video (1).avi"
```
Or omit the argument to use the default input:
```bash
python detector_fall_yolo_pose.py
# defaults to ./dataset/Le2i/Home_01/Home_01/Videos/video (1).avi
```

## Output
- Results video is saved under `results/<mirrored-input-path>/<filename>/<filename>_fall_detected.avi`.
- A window named "Fall Detection (Pose)" will preview the detection; press `ESC` to exit.

## Options
Change defaults by editing the call in `detector_fall_yolo_pose.py`:
- `model_size`: one of `{"n","s","m","l"}` (default: `s` in the example)
- `imgsz`, `conf`, `iou`, `device`, `show`

## Notes
- If you already tracked `dataset/` or `weights/` in git history, remove them from the index:
```bash
git rm -r --cached dataset weights
```
Then commit and push.