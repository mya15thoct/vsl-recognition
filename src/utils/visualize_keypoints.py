"""
Visualize MediaPipe Holistic keypoints as VIDEO for paper figures.

Usage:
  # From raw video — overlay keypoints on original frames:
  python src/utils/visualize_keypoints.py --source video --input path/to/video.mp4

  # From raw video — skeleton only on white background:
  python src/utils/visualize_keypoints.py --source video --input path/to/video.mp4 --mode skeleton

  # From raw video — side-by-side (original | keypoints):
  python src/utils/visualize_keypoints.py --source video --input path/to/video.mp4 --mode sidebyside

  # From .npy file — reconstruct skeleton video from saved keypoints:
  python src/utils/visualize_keypoints.py --source npy --input path/to/sequence.npy

Output:
  Saves MP4 video to the same directory as input, or to --output path.
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# ─── MediaPipe helpers ────────────────────────────────────────────────────────
mp_drawing      = mp.solutions.drawing_utils
mp_holistic_mod = mp.solutions.holistic

# ─── Landmark index slices inside the 1662-dim vector ────────────────────────
POSE_SLICE = slice(0,    132)   # 33 × 4
FACE_SLICE = slice(132, 1536)   # 468 × 3
LH_SLICE   = slice(1536, 1599)  # 21 × 3
RH_SLICE   = slice(1599, 1662)  # 21 × 3

# ─── Drawing styles ───────────────────────────────────────────────────────────
S = {
    "pose_lm":   mp_drawing.DrawingSpec(color=(0,   80, 255), thickness=2, circle_radius=3),
    "pose_cn":   mp_drawing.DrawingSpec(color=(0,   80, 255), thickness=2),
    "face_lm":   mp_drawing.DrawingSpec(color=(0,  200,  80), thickness=1, circle_radius=1),
    "face_cn":   mp_drawing.DrawingSpec(color=(0,  200,  80), thickness=1),
    "lh_lm":     mp_drawing.DrawingSpec(color=(255, 50,  50), thickness=2, circle_radius=3),
    "lh_cn":     mp_drawing.DrawingSpec(color=(255, 50,  50), thickness=2),
    "rh_lm":     mp_drawing.DrawingSpec(color=(220,  0, 220), thickness=2, circle_radius=3),
    "rh_cn":     mp_drawing.DrawingSpec(color=(220,  0, 220), thickness=2),
}

LEGEND = [
    ("Pose",       S["pose_lm"].color),
    ("Face",       S["face_lm"].color),
    ("Left hand",  S["lh_lm"].color),
    ("Right hand", S["rh_lm"].color),
]


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _draw_landmarks(canvas: np.ndarray, results) -> np.ndarray:
    """Draw all MediaPipe Holistic landmarks onto a BGR frame."""
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            canvas, results.face_landmarks,
            mp_holistic_mod.FACEMESH_CONTOURS,
            landmark_drawing_spec=S["face_lm"],
            connection_drawing_spec=S["face_cn"],
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            canvas, results.pose_landmarks,
            mp_holistic_mod.POSE_CONNECTIONS,
            landmark_drawing_spec=S["pose_lm"],
            connection_drawing_spec=S["pose_cn"],
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            canvas, results.left_hand_landmarks,
            mp_holistic_mod.HAND_CONNECTIONS,
            landmark_drawing_spec=S["lh_lm"],
            connection_drawing_spec=S["lh_cn"],
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            canvas, results.right_hand_landmarks,
            mp_holistic_mod.HAND_CONNECTIONS,
            landmark_drawing_spec=S["rh_lm"],
            connection_drawing_spec=S["rh_cn"],
        )
    return canvas


def _add_legend(frame: np.ndarray) -> np.ndarray:
    """Add a color legend in the top-left corner."""
    for i, (label, color) in enumerate(LEGEND):
        y = 20 + i * 22
        cv2.circle(frame, (14, y), 6, color, -1)
        cv2.putText(frame, label, (26, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return frame


def _make_writer(output_path: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {output_path}")
    return writer


# ══════════════════════════════════════════════════════════════════════════════
# FROM RAW VIDEO
# ══════════════════════════════════════════════════════════════════════════════

def visualize_from_video(
    video_path: str,
    mode: str = "sidebyside",       # "overlay" | "skeleton" | "sidebyside"
    output_path: str = None,
    bg_color: tuple = (245, 245, 245),
) -> str:
    """
    Run MediaPipe on every frame of a video and write a keypoint video.

    Args:
        video_path:  Path to input video.
        mode:        'overlay'    – keypoints drawn on the original frame
                     'skeleton'   – keypoints on plain background only
                     'sidebyside' – original frame | keypoint frame
        output_path: Where to save the MP4 (auto-generated if None).
        bg_color:    Background BGR color for 'skeleton' / 'sidebyside' modes.

    Returns:
        Path to the saved video.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input : {video_path.name}  ({total} frames, {W}×{H}, {fps:.1f} fps)")

    out_w = W * 2 + 4 if mode == "sidebyside" else W   # separator=4px

    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_keypoints_{mode}.mp4"

    writer = _make_writer(output_path, fps, out_w, H)

    with mp_holistic_mod.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe inference
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # Build output frame
            if mode == "overlay":
                out_frame = _draw_landmarks(frame.copy(), results)
                _add_legend(out_frame)

            elif mode == "skeleton":
                bg = np.full((H, W, 3), bg_color, dtype=np.uint8)
                out_frame = _draw_landmarks(bg, results)
                _add_legend(out_frame)

            else:  # sidebyside
                key_frame = np.full((H, W, 3), bg_color, dtype=np.uint8)
                key_frame = _draw_landmarks(key_frame, results)
                _add_legend(key_frame)
                sep = np.full((H, 4, 3), 180, dtype=np.uint8)
                out_frame = np.concatenate([frame, sep, key_frame], axis=1)

            writer.write(out_frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  Processed {frame_idx}/{total} frames...", end="\r")

    cap.release()
    writer.release()
    print(f"\n[OK] Saved → {output_path}  ({frame_idx} frames)")
    return str(output_path)


# ══════════════════════════════════════════════════════════════════════════════
# FROM .NPY FILE
# ══════════════════════════════════════════════════════════════════════════════

def _npy_to_pts(flat, n, stride, has_vis=False):
    """Convert flat keypoint array → list of (x,y,z,[v]) or None if all-zero."""
    if np.all(flat == 0):
        return None
    pts = []
    for i in range(n):
        b = i * stride
        pts.append((float(flat[b]), float(flat[b+1]), float(flat[b+2])))
    return pts


def _draw_pts_connections(canvas, pts, connections, lm_style, cn_style, W, H):
    """Draw landmarks + connections given normalized (x,y,z) tuples."""
    if pts is None:
        return
    if connections:
        for (i, j) in connections:
            if 0 <= i < len(pts) and 0 <= j < len(pts):
                xi, yi = int(pts[i][0] * W), int(pts[i][1] * H)
                xj, yj = int(pts[j][0] * W), int(pts[j][1] * H)
                cv2.line(canvas, (xi, yi), (xj, yj),
                         cn_style.color, cn_style.thickness)
    for (x, y, *_) in pts:
        cv2.circle(canvas, (int(x*W), int(y*H)),
                   lm_style.circle_radius, lm_style.color, -1)


def visualize_from_npy(
    npy_path: str,
    fps: float = 25.0,
    canvas_size: tuple = (640, 480),
    output_path: str = None,
    bg_color: tuple = (245, 245, 245),
) -> str:
    """
    Reconstruct and draw keypoints from a saved .npy sequence as a video.

    Coordinates in the .npy are shoulder-normalized (mean≈0, scale≈shoulder width).
    We apply a display-only de-normalization (cx=0.5, d=0.22) so the skeleton
    appears centered in the canvas. This does NOT affect training data.

    Args:
        npy_path:    Path to .npy file, shape (T, 1662).
        fps:         Frames per second for the output video.
        canvas_size: (width, height) of output frames.
        output_path: Where to save the MP4.
        bg_color:    BGR background color.

    Returns:
        Path to the saved video.
    """
    npy_path = Path(npy_path)
    seq = np.load(str(npy_path))
    if seq.ndim == 1:
        seq = seq[np.newaxis, :]        # single frame

    T = seq.shape[0]
    W, H = canvas_size
    print(f"Input : {npy_path.name}  ({T} frames, {seq.shape[1]} dims)")

    if output_path is None:
        output_path = npy_path.parent / f"{npy_path.stem}_keypoints.mp4"

    writer = _make_writer(output_path, fps, W, H)

    # Display-only de-normalization constants
    cx, cy, d = 0.50, 0.50, 0.22

    POSE_CONN = list(mp_holistic_mod.POSE_CONNECTIONS)
    HAND_CONN = list(mp_holistic_mod.HAND_CONNECTIONS)
    FACE_CONN = list(mp_holistic_mod.FACEMESH_CONTOURS)

    for t in range(T):
        kp = seq[t]  # (1662,)

        # Parse + de-normalize x,y (z left unchanged)
        def denorm(arr, stride):
            a = arr.copy()
            a[0::stride] = a[0::stride] * d + cx
            a[1::stride] = a[1::stride] * d + cy
            return a

        pose_d = denorm(kp[POSE_SLICE], 4)
        face_d = denorm(kp[FACE_SLICE], 3)
        lh_d   = denorm(kp[LH_SLICE],  3)
        rh_d   = denorm(kp[RH_SLICE],  3)

        pose_pts = _npy_to_pts(pose_d, 33,  4)
        face_pts = _npy_to_pts(face_d, 468, 3)
        lh_pts   = _npy_to_pts(lh_d,  21,  3)
        rh_pts   = _npy_to_pts(rh_d,  21,  3)

        canvas = np.full((H, W, 3), bg_color, dtype=np.uint8)

        _draw_pts_connections(canvas, face_pts, FACE_CONN, S["face_lm"], S["face_cn"], W, H)
        _draw_pts_connections(canvas, pose_pts, POSE_CONN, S["pose_lm"], S["pose_cn"], W, H)
        _draw_pts_connections(canvas, lh_pts,   HAND_CONN, S["lh_lm"],   S["lh_cn"],  W, H)
        _draw_pts_connections(canvas, rh_pts,   HAND_CONN, S["rh_lm"],   S["rh_cn"],  W, H)

        _add_legend(canvas)

        # Frame counter (optional)
        cv2.putText(canvas, f"Frame {t+1}/{T}", (W - 130, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

        writer.write(canvas)

    writer.release()
    print(f"[OK] Saved → {output_path}  ({T} frames)")
    return str(output_path)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Render MediaPipe keypoints as a video from raw video or .npy file."
    )
    parser.add_argument("--source", choices=["video", "npy"], required=True,
                        help="Input type: 'video' or 'npy'.")
    parser.add_argument("--input",  required=True,
                        help="Path to video (.mp4/.mov) or sequence (.npy).")
    parser.add_argument("--mode",
                        choices=["overlay", "skeleton", "sidebyside"],
                        default="sidebyside",
                        help="(video only) Output layout. Default: sidebyside.")
    parser.add_argument("--output", default=None,
                        help="Output MP4 path (default: auto-generated next to input).")
    parser.add_argument("--fps",    type=float, default=25.0,
                        help="(npy only) FPS for output video. Default: 25.")
    parser.add_argument("--width",  type=int,   default=640,
                        help="(npy only) Canvas width.  Default: 640.")
    parser.add_argument("--height", type=int,   default=480,
                        help="(npy only) Canvas height. Default: 480.")
    args = parser.parse_args()

    if args.source == "video":
        visualize_from_video(
            video_path=args.input,
            mode=args.mode,
            output_path=args.output,
        )
    else:
        visualize_from_npy(
            npy_path=args.input,
            fps=args.fps,
            canvas_size=(args.width, args.height),
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
