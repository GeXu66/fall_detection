# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: CLI entrypoint. Parses flags and calls `process_video`.
- `fall_detector.py`: Core detection pipeline, model/weights handling.
- `fall_pose_utils.py`: Logging, drawing, pose helpers, temporal smoothing.
- `seg_bed.py`: Optional bed segmentation utilities.
- `dataset/`: Input videos (not tracked).
- `results/`: Outputs written here (mirrors input paths).
- `weights/`: YOLOv8 weights (auto-downloaded if missing).

Prefer adding new pipeline logic in `fall_detector.py` and shared helpers in `fall_pose_utils.py`. Keep the CLI thin in `main.py`.

## Build, Run, and Dev Commands
- Create venv (Windows): `python -m venv .venv && .\.venv\Scripts\activate`
- Install deps: `pip install --upgrade pip && pip install ultralytics opencv-python numpy`
- Run default demo: `python main.py`
- Run with input: `python main.py "./dataset/Real/fall/video1.mp4"`

Tip: Quote paths with spaces. Outputs save under `results/` with `_fall_detected` suffix.

## Coding Style & Naming
- Python 3.8+; 4‑space indentation; UTF‑8.
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Use type hints and concise docstrings for new public functions.
- Logging via `setup_logging(level)` in `fall_pose_utils.py`; default `INFO`.

## Testing Guidelines
- No formal test suite yet. When adding tests:
  - Place under `tests/` and name `test_*.py`.
  - Prefer small, deterministic samples; avoid committing large media.
- Quick smoke test: run `python main.py` and verify an output file appears in `results/` and the preview window exits with `ESC`.

## Commit & Pull Request Guidelines
- Commits: present tense, scoped, and focused (e.g., `feat: add overlap check`).
- Include rationale and any perf or accuracy impacts.
- PRs should include:
  - Clear description, sample command(s), and expected output path under `results/`.
  - Linked issue (if any) and before/after screenshots or short clips when UI/visuals change.

## Security & Data Hygiene
- Do not commit `dataset/`, `weights/`, or generated results. If previously tracked: `git rm -r --cached dataset weights results`.
- Keep downloads and large binaries out of Git.

## Agent-Specific Notes
- Keep the CLI stable; add flags only when necessary and document them in `README.md`.
- Favor minimal, well-placed changes over broad refactors; maintain current file roles.
