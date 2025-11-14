import cv2
import numpy as np
from pathlib import Path

VIDEO_PATH = Path("./dataset/Real/fall/video1.mp4")
DOWN_SAMPLE_SCALE: float = 0.25  # (0, 1]; e.g., 0.5 halves both width and height.
GROUND_POINTS = np.float32([
    [0.0, 0.0],   # P1: 左下
    [3, 0.0],   # P2: 床脚
    [0.0, 3],   # P3: 左上
    [0.8, 3],   # P4: 右上
])

def downsample_frame(frame: np.ndarray) -> np.ndarray:
    """Resize the frame according to DOWN_SAMPLE_SCALE for faster calibration."""
    if not 0 < DOWN_SAMPLE_SCALE <= 1.0:
        raise ValueError("DOWN_SAMPLE_SCALE must be within (0, 1].")
    if DOWN_SAMPLE_SCALE == 1.0:
        return frame

    h, w = frame.shape[:2]
    new_w = max(1, int(w * DOWN_SAMPLE_SCALE))
    new_h = max(1, int(h * DOWN_SAMPLE_SCALE))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def collect_pixel_points(frame):
    selected = []
    display = frame.copy()

    def on_mouse(event, x, y, _, __):
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 4:
            selected.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(display, f"P{len(selected)}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.namedWindow("Select P1→P4")
    cv2.setMouseCallback("Select P1→P4", on_mouse)

    while True:
        cv2.imshow("Select P1→P4", display)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            selected.clear()
            break
        if len(selected) == 4:
            break

    cv2.destroyWindow("Select P1→P4")
    return np.float32(selected) if selected else None

def compute_scaled_homography(pixel_points):
    H, _ = cv2.findHomography(pixel_points, GROUND_POINTS)

    width = np.ptp(GROUND_POINTS[:, 0])
    height = np.ptp(GROUND_POINTS[:, 1])
    if width == 0 or height == 0:
        raise ValueError("Ground plane points must span non-zero width/height.")

    target_w = 600
    target_h = int(target_w * (height / width))

    sx = target_w / width
    sy = target_h / height
    tx = -GROUND_POINTS[:, 0].min() * sx
    ty = -GROUND_POINTS[:, 1].min() * sy
    scale_matrix = np.array([[sx, 0, tx],
                            [0, sy, ty],
                            [0,  0,  1]], dtype=np.float32)

    return scale_matrix @ H, (target_w, target_h)

def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Unable to read first frame.")
    first_frame = downsample_frame(first_frame)

    pixel_pts = collect_pixel_points(first_frame)
    if pixel_pts is None:
        print("Selection cancelled.")
        cap.release()
        return

    H_scaled, bird_size = compute_scaled_homography(pixel_pts)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = downsample_frame(frame)

        bird_view = cv2.warpPerspective(frame, H_scaled, bird_size)
        resized_bird = cv2.resize(bird_view, (frame.shape[1], frame.shape[0]))
        stacked = np.hstack([frame, resized_bird])

        cv2.imshow("Original (left) | Top-Down (right)", stacked)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
