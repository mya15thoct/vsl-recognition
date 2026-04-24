"""
Test model đã train trước đó bằng video tự quay từ folder data/realtime.

Cách đặt tên file:
  - Đặt tên file = tên class (khớp với tên folder trong sequences/)
  - VD: XIN_CHAO.mp4  →  label = "XIN_CHAO"
  - Nếu có nhiều video/class: XIN_CHAO_1.mp4, XIN_CHAO_2.mp4, ...
    (script tự cắt phần _1, _2 ở cuối)

Usage:
    python test_realtime_video.py              # xem video + dự đoán
    python test_realtime_video.py --no_show    # chỉ in kết quả (server/no screen)
    python test_realtime_video.py --top_k 5
    python test_realtime_video.py --model_path /path/to/best_model


"""
import sys
import argparse
import re
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# ── project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
SRC_DIR      = PROJECT_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

import tensorflow as tf
from config import (
    CHECKPOINT_DIR,
    SEQUENCE_PATH,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
)
from utils.extraction import mediapipe_detection, extract_keypoints, get_holistic_model

REALTIME_DIR = PROJECT_ROOT / 'data' / 'realtime'
VIDEO_EXTS   = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

# ── colors ────────────────────────────────────────────────────────────────────
CLR_BG     = (30,  30,  30)
CLR_GREEN  = (0,  220,  90)
CLR_CYAN   = (0,  180, 230)
CLR_RED    = (60,  60, 230)
CLR_WHITE  = (255, 255, 255)
CLR_GRAY   = (160, 160, 160)
CLR_YELLOW = (0,  220, 220)


# ══════════════════════════════════════════════════════════════════════════════
# Label extraction from filename
# ══════════════════════════════════════════════════════════════════════════════

def filename_to_label(stem: str, class_names: list) -> str | None:
    """
    Map filename (without extension) to a class label.

    Rules (tried in order):
      1. Exact match:            "XIN_CHAO"     → "XIN_CHAO"
      2. Case-insensitive match: "xin_chao"     → "XIN_CHAO"
      3. Strip trailing _N:      "XIN_CHAO_1"   → "XIN_CHAO"
      4. Strip trailing _NNN:    "XIN_CHAO_001"  → "XIN_CHAO"

    Returns matched class name, or None if no match.
    """
    upper_map = {c.upper(): c for c in class_names}

    # 1 & 2: direct / case-insensitive
    if stem.upper() in upper_map:
        return upper_map[stem.upper()]

    # 3 & 4: strip trailing _number
    stripped = re.sub(r'_\d+$', '', stem)
    if stripped.upper() in upper_map:
        return upper_map[stripped.upper()]

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Core helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: Path):
    print(f"\n{'='*60}")
    print(f"Loading model: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Hãy train model trước, hoặc truyền --model_path."
        )
    model = tf.keras.models.load_model(str(model_path))
    print(f"  ✓ Model loaded  →  input shape: {model.input_shape}")
    return model


def load_class_names(model_path: Path, sequence_path: Path) -> list:
    """
    Load class names from action_mapping.json saved alongside the model.
    Falls back to sequences folder if JSON not found.
    
    IMPORTANT: Must use the JSON saved AT TRAINING TIME, not the current
    sequences folder — which may have grown with new classes added later,
    causing index mismatches.
    """
    import json
    # Try action_mapping.json in same folder as model
    json_path = model_path.parent / 'action_mapping.json'
    if json_path.exists():
        with open(json_path) as f:
            mapping = json.load(f)   # {"0": "Baby", "1": "Bed", ...}
        # Sort by integer key to preserve training order
        classes = [mapping[k] for k in sorted(mapping, key=lambda x: int(x))]
        print(f"  ✓ {len(classes)} class names from: {json_path}  ← CORRECT (training order)")
        return classes

    # Fallback: sequences folder (may be outdated if data was added later)
    print(f"  [WARNING] action_mapping.json not found at: {json_path}")
    print(f"  [WARNING] Falling back to sequences folder — class order may be WRONG!")
    if sequence_path.exists():
        classes = sorted([d.name for d in sequence_path.iterdir() if d.is_dir()])
        print(f"  ✓ {len(classes)} class names from: {sequence_path}")
        return classes
    return []


def extract_sequence_from_video(video_path: Path, holistic):
    """Returns (keypoints_array, raw_frames) or (None, None)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    frames_data, raw_frames = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, results = mediapipe_detection(frame, holistic)
        frames_data.append(extract_keypoints(results))
        raw_frames.append(frame)
    cap.release()

    if not frames_data:
        return None, None
    return np.array(frames_data, dtype=np.float32), raw_frames


def pad_or_truncate(seq: np.ndarray, target_len: int) -> np.ndarray:
    T = seq.shape[0]
    if T >= target_len:
        return seq[:target_len]
    padded = np.zeros((target_len, seq.shape[1]), dtype=np.float32)
    padded[:T] = seq
    return padded


def predict(model, sequence: np.ndarray, class_names: list, top_k: int = 3):
    seq_batch = pad_or_truncate(sequence, model.input_shape[1])[np.newaxis]
    probs     = model.predict(seq_batch, verbose=0)[0]
    top_idx   = np.argsort(probs)[::-1][:top_k]
    return [
        {
            'rank': i + 1,
            'name': class_names[idx] if class_names else str(idx),
            'conf': float(probs[idx]),
        }
        for i, idx in enumerate(top_idx)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Overlay UI
# ══════════════════════════════════════════════════════════════════════════════

def build_overlay(frame, preds, video_name, true_label,
                  frame_idx, total_frames):
    h, w   = frame.shape[:2]
    pw     = 400
    panel  = np.full((h, pw, 3), CLR_BG, dtype=np.uint8)

    def put(img, text, x, y, color=CLR_WHITE, scale=0.6, thick=1):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    # title
    put(panel, "VSL PREDICTION", 10, 36, CLR_YELLOW, 0.75, 2)
    put(panel, video_name[:44], 10, 58, CLR_GRAY, 0.42)

    # ground truth (if known)
    if true_label:
        put(panel, f"Ground truth: {true_label}", 10, 82, CLR_CYAN, 0.52, 1)

    # progress bar
    by = 100
    bw = pw - 20
    cv2.rectangle(panel, (10, by), (10 + bw, by + 6), (60,60,60), -1)
    fill = int(bw * frame_idx / max(total_frames - 1, 1))
    cv2.rectangle(panel, (10, by), (10 + fill, by + 6), CLR_YELLOW, -1)
    put(panel, f"frame {frame_idx+1}/{total_frames}", 10, by + 20, CLR_GRAY, 0.4)

    cv2.line(panel, (10, 132), (pw - 10, 132), (70,70,70), 1)

    # top-K
    y0 = 162
    for p in preds:
        is_top1   = p['rank'] == 1
        is_correct = true_label and (p['name'].upper() == true_label.upper())

        if is_correct and is_top1:
            color = CLR_GREEN
        elif is_correct:
            color = CLR_CYAN
        elif is_top1:
            color = CLR_WHITE
        else:
            color = CLR_GRAY

        put(panel, f"#{p['rank']}", 10, y0, color, 0.55, 2)
        put(panel, p['name'][:26], 48, y0, color, 0.60, 1)

        # confidence bar
        bar_y = y0 + 8
        cv2.rectangle(panel, (10, bar_y), (pw-10, bar_y+10), (55,55,55), -1)
        fill2 = int((pw-20) * p['conf'])
        cv2.rectangle(panel, (10, bar_y), (10+fill2, bar_y+10), color, -1)
        put(panel, f"{p['conf']*100:.1f}%", pw - 70, bar_y + 9, CLR_WHITE, 0.45)
        y0 += 72

    # correct / wrong badge (top-1 vs ground truth)
    if true_label:
        correct = preds[0]['name'].upper() == true_label.upper()
        badge   = "CORRECT" if correct else "WRONG"
        bclr    = CLR_GREEN if correct else CLR_RED
        put(panel, badge, 10, h - 65, bclr, 0.8, 2)

    put(panel, "SPACE / any key -> next", 10, h - 38, CLR_GRAY, 0.44)
    put(panel, "Q / ESC         -> quit", 10, h - 18, CLR_GRAY, 0.44)

    return np.hstack([frame, panel])


def play_video(video_name, raw_frames, preds, true_label, fps=25):
    """Returns 'next' or 'quit'."""
    delay  = max(1, int(1000 / fps))
    n      = len(raw_frames)
    action = 'next'

    for i, frame in enumerate(raw_frames):
        overlay = build_overlay(frame, preds, video_name, true_label, i, n)
        cv2.imshow("VSL Test", overlay)
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord('q'), 27):
            action = 'quit'
            break

    if action != 'quit':
        overlay = build_overlay(raw_frames[-1], preds, video_name,
                                true_label, n-1, n)
        cv2.putText(overlay, "[ Press any key for next ]",
                    (20, overlay.shape[0]-70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, CLR_YELLOW, 1, cv2.LINE_AA)
        cv2.imshow("VSL Test", overlay)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            action = 'quit'

    return action


# ══════════════════════════════════════════════════════════════════════════════
# Main test loop
# ══════════════════════════════════════════════════════════════════════════════

def run(model, class_names, holistic_cfg, top_k, show):
    videos = sorted(
        f for f in REALTIME_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        print(f"\n[INFO] Không có video trong: {REALTIME_DIR}")
        return

    print(f"\nTìm thấy {len(videos)} video(s)\n")

    results = []
    correct = 0
    labeled = 0

    for video_path in videos:
        stem       = video_path.stem
        true_label = filename_to_label(stem, class_names) if class_names else None

        print(f"▶  {video_path.name}")
        if true_label:
            print(f"   Ground truth : {true_label}")
        else:
            print(f"   Ground truth : (không nhận ra tên class từ tên file)")

        # Fresh holistic per video — prevents tracking state bleeding between videos
        holistic = get_holistic_model(**holistic_cfg)
        try:
            sequence, raw_frames = extract_sequence_from_video(video_path, holistic)
        finally:
            holistic.close()
        if sequence is None:
            print("   [ERROR] Không đọc được video\n")
            continue

        target_len = model.input_shape[1]
        truncated  = len(sequence) > target_len
        print(f"   Frames    : {len(sequence)} {'→ TRUNCATED to ' + str(target_len) + ' ⚠️' if truncated else '(OK, target=' + str(target_len) + ')'}")

        preds = predict(model, sequence, class_names, top_k=top_k)
        pred1 = preds[0]

        is_correct = (true_label is not None and
                      pred1['name'].upper() == true_label.upper())
        if true_label:
            labeled += 1
            correct += int(is_correct)

        status = "✓ ĐÚNG" if is_correct else ("✗ SAI " if true_label else "   ")
        print(f"   {status}  Predicted: {pred1['name']}  ({pred1['conf']*100:.1f}%)")
        for p in preds[1:]:
            print(f"          Top-{p['rank']}: {p['name']} ({p['conf']*100:.1f}%)")
        print()

        results.append({
            'video':      video_path.name,
            'true_label': true_label or '',
            'predicted':  pred1['name'],
            'confidence': pred1['conf'],
            'correct':    is_correct,
            'frames':     len(sequence),
        })

        if show:
            cap = cv2.VideoCapture(str(video_path))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
            cap.release()
            action = play_video(video_path.name, raw_frames, preds,
                                true_label, fps=fps)
            if action == 'quit':
                break

    cv2.destroyAllWindows()

    # ── per-video table ───────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("KẾT QUẢ TỪNG VIDEO")
    print("=" * 90)
    print(f"{'Video':<35}  {'Ground Truth':<22}  {'Predicted':<22}  {'Conf':>6}  {'OK?'}")
    print("-" * 95)
    for r in results:
        ok = "✓" if r['correct'] else ("✗" if r['true_label'] else "-")
        print(f"{r['video']:<35}  {r['true_label']:<22}  "
              f"{r['predicted']:<22}  {r['confidence']*100:5.1f}%  {ok}")
    print("-" * 95)

    # ── metrics ───────────────────────────────────────────────────────────────
    labeled_results = [r for r in results if r['true_label']]
    if len(labeled_results) == 0:
        print("\n[INFO] Không có file nào khớp tên class → không tính metrics.")
        print("       Đổi tên file = tên class (VD: XIN_CHAO.mp4)")
        return

    y_true = [r['true_label'] for r in labeled_results]
    y_pred = [r['predicted']  for r in labeled_results]
    n      = len(labeled_results)
    labels = sorted(set(y_true))   # only classes that actually appear in test set
    n_cls  = len(labels)

    acc      = accuracy_score(y_true, y_pred)
    # Pass labels= so macro average is over y_true classes only,
    # not polluted by extra classes the model happened to predict.
    macro_f1 = f1_score(y_true, y_pred, average='macro',     labels=labels, zero_division=0)
    macro_pr = precision_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    macro_re = recall_score(y_true, y_pred, average='macro',  labels=labels, zero_division=0)

    print("\n" + "=" * 60)
    print("METRICS SUMMARY  (realtime / unseen signer)")
    print("=" * 60)
    print(f"  Videos tested : {n}  ({n_cls} classes)")
    print(f"  Acc (%)       : {acc*100:.2f}%")
    print(f"  Macro F1      : {macro_f1*100:.2f}%")
    print(f"  Macro Precision: {macro_pr*100:.2f}%")
    print(f"  Macro Recall  : {macro_re*100:.2f}%")
    print("=" * 60)

    # ── per-class report ──────────────────────────────────────────────────────
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    # ── paper-ready sentence ──────────────────────────────────────────────────
    print("─" * 60)
    print("→  Dùng các con số này trong paper:")
    print(f'   "On a set of {n} self-recorded videos from an unseen signer')
    print(f'    covering {n_cls} sign classes, the proposed model achieves')
    print(f'    {acc*100:.1f}% accuracy, {macro_f1*100:.1f}% macro F1,')
    print(f'    {macro_pr*100:.1f}% macro precision, and {macro_re*100:.1f}% macro recall."')
    print("─" * 60)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--no_show',    action='store_true',
                   help='Không hiện cửa sổ video (chỉ in kết quả)')
    p.add_argument('--top_k',      type=int, default=3)
    p.add_argument('--model_path', type=str, default=None)
    p.add_argument('--seq_dir',    type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path) if args.model_path else CHECKPOINT_DIR / 'best_model'
    seq_dir    = Path(args.seq_dir)    if args.seq_dir    else SEQUENCE_PATH
    show       = not args.no_show

    if not REALTIME_DIR.exists():
        REALTIME_DIR.mkdir(parents=True)
        print(f"[INFO] Đã tạo folder: {REALTIME_DIR}")
        print("       Copy video vào đó rồi chạy lại.")
        sys.exit(0)

    model       = load_model(model_path)
    class_names = load_class_names(model_path, seq_dir)

    holistic_cfg = dict(
        min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
    )

    print(f"\nRealtime dir : {REALTIME_DIR}")
    print(f"Show video   : {'YES' if show else 'NO'}")
    print(f"Top-K        : {args.top_k}")
    print(f"Label source : tên file (VD: XIN_CHAO.mp4 → label XIN_CHAO)")

    run(model, class_names, holistic_cfg, top_k=args.top_k, show=show)
    print("Done.")


if __name__ == '__main__':
    main()
