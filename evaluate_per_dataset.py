"""
Evaluate trained model separately on each of 3 ISL benchmark datasets.

Pipeline per dataset:
  1. Extract MediaPipe keypoints from raw videos
  2. Load the trained model (best_model_combined or best_model)
  3. Predict and report Accuracy / F1 / Precision / Recall
  4. Save confusion matrix PNG

Usage:
  python evaluate_per_dataset.py \
    --model_path  /home/islabworker2/mya/recognition/checkpoints/mlp/best_model_combined \
    --action_mapping /home/islabworker2/mya/recognition/checkpoints/mlp/action_mapping_combined.json \
    --data_dir    /home/islabworker2/mya/vsl-recognition/data \
    --out_dir     /home/islabworker2/mya/vsl-recognition/eval_results

  # Skip re-extraction if keypoints already exist:
  python evaluate_per_dataset.py ... --skip_extraction

  # Evaluate only one dataset:
  python evaluate_per_dataset.py ... --dataset include
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ── MediaPipe ────────────────────────────────────────────────────────────────
import mediapipe as mp

mp_holistic = mp.solutions.holistic

def get_holistic():
    return mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

def extract_keypoints(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])  # (1662,)


# ── Dataset registry ─────────────────────────────────────────────────────────
def get_datasets(data_dir: Path) -> dict:
    return {
        "include": {
            "label":    "INCLUDE (263 classes)",
            "video_dir": data_dir / "INCLUDE",
            "seq_dir":   data_dir / "_sequences" / "INCLUDE",
        },
        "isl": {
            "label":    "ISL Video Dataset (60 classes)",
            "video_dir": data_dir / "ISL_Video",
            "seq_dir":   data_dir / "_sequences" / "ISL_Video",
        },
        "medical": {
            "label":    "INCLUDE-24 Medical (24 classes)",
            "video_dir": data_dir / "INCLUDE_24_Medical",
            "seq_dir":   data_dir / "_sequences" / "INCLUDE_24_Medical",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — EXTRACT KEYPOINTS
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI"}

def extract_dataset(video_dir: Path, seq_dir: Path, dataset_label: str):
    """Extract MediaPipe keypoints from all videos in video_dir → seq_dir."""
    print(f"\n[EXTRACT] {dataset_label}")
    print(f"  Source : {video_dir}")
    print(f"  Output : {seq_dir}")

    if not video_dir.exists():
        print(f"  [ERROR] video_dir not found: {video_dir}")
        return False

    class_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        # Some datasets put videos directly without class subfolders
        # Try to detect flat structure
        all_videos = [f for f in video_dir.rglob("*") if f.suffix in VIDEO_EXTS]
        if not all_videos:
            print(f"  [ERROR] No class folders or videos found in {video_dir}")
            return False
        print(f"  [WARN] No class subfolders found — found {len(all_videos)} videos at root")
        print("  Please organise videos into class subfolders:")
        print("    video_dir/CLASS_NAME/video1.mp4")
        return False

    seq_dir.mkdir(parents=True, exist_ok=True)
    holistic = get_holistic()
    total, skipped = 0, 0

    for cls_idx, cls_dir in enumerate(class_dirs):
        videos = [f for f in cls_dir.iterdir() if f.suffix in VIDEO_EXTS]
        if not videos:
            continue

        out_cls = seq_dir / cls_dir.name
        out_cls.mkdir(parents=True, exist_ok=True)
        print(f"  [{cls_idx+1}/{len(class_dirs)}] {cls_dir.name} — {len(videos)} videos", end="", flush=True)

        done = 0
        for vpath in videos:
            out_npy = out_cls / f"{vpath.stem}.npy"
            if out_npy.exists():          # resume-friendly
                done += 1
                continue

            cap = cv2.VideoCapture(str(vpath))
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = holistic.process(frame_rgb)
                frames.append(extract_keypoints(results))
            cap.release()

            if not frames:
                skipped += 1
                continue

            np.save(out_npy, np.array(frames, dtype=np.float32))
            done += 1
            total += 1

        print(f" → {done} saved")

    holistic.close()
    print(f"\n  [OK] Extracted {total} sequences (skipped {skipped})")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD SEQUENCES & MAP LABELS
# ─────────────────────────────────────────────────────────────────────────────

def load_sequences_for_eval(seq_dir: Path, action_names: list, seq_length: int):
    """
    Load .npy sequences from seq_dir.
    Only keeps classes that exist in the model's action_names.
    Returns X, y_true (integer), class_names_present.
    """
    model_name_upper = {n.upper(): i for i, n in enumerate(action_names)}

    X, y, found_classes = [], [], []
    class_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])

    matched, unmatched = 0, 0
    for cls_dir in class_dirs:
        label_idx = model_name_upper.get(cls_dir.name.upper())
        if label_idx is None:
            unmatched += 1
            continue
        matched += 1
        found_classes.append(cls_dir.name.upper())

        for npy in sorted(cls_dir.glob("*.npy")):
            seq = np.load(npy).astype(np.float32)
            padded = np.zeros((seq_length, 1662), dtype=np.float32)
            L = min(len(seq), seq_length)
            padded[:L] = seq[:L]
            X.append(padded)
            y.append(label_idx)

    print(f"  Matched classes   : {matched}")
    print(f"  Unmatched (skipped): {unmatched}")
    print(f"  Total samples     : {len(X)}")

    if not X:
        return None, None, []

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), found_classes


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — EVALUATE & PLOT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_dataset(model, action_names, seq_dir, dataset_label, out_dir):
    seq_length = model.input_shape[1]

    print(f"\n{'='*65}")
    print(f"  EVALUATING: {dataset_label}")
    print(f"{'='*65}")

    X, y_true, found_classes = load_sequences_for_eval(seq_dir, action_names, seq_length)
    if X is None:
        print(f"  [SKIP] No matching samples found for {dataset_label}")
        return None

    print(f"\n  Running inference on {len(X)} samples...")
    y_pred_prob = model.predict(X, batch_size=32, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    unique_labels = np.unique(y_true)
    class_names   = [action_names[i] for i in unique_labels]

    report_dict = classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=class_names,
        zero_division=0,
    )

    acc     = report_dict.get("accuracy", 0.0)
    f1      = report_dict["macro avg"]["f1-score"]
    prec    = report_dict["macro avg"]["precision"]
    rec     = report_dict["macro avg"]["recall"]

    print(f"\n  {report_str}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  F1 Score : {f1*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")

    # Save text report
    safe_label = dataset_label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"report_{safe_label}.txt"
    with open(report_path, "w") as f:
        f.write(f"Dataset : {dataset_label}\n")
        f.write(f"Accuracy : {acc*100:.2f}%\n")
        f.write(f"F1       : {f1*100:.2f}%\n")
        f.write(f"Precision: {prec*100:.2f}%\n")
        f.write(f"Recall   : {rec*100:.2f}%\n\n")
        f.write(report_str)
    print(f"\n  Report saved: {report_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    n = len(class_names)
    fig_size = max(12, n * 0.25)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(cm_norm, ax=ax, cmap="Blues", vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0, annot=(n <= 30))
    ax.set_title(f"Confusion Matrix — {dataset_label} ({n} classes, acc={acc*100:.1f}%)")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    lsize = max(4, 10 - n // 20)
    ax.tick_params(axis="x", labelsize=lsize, rotation=90)
    ax.tick_params(axis="y", labelsize=lsize, rotation=0)
    plt.tight_layout()
    cm_path = out_dir / f"confusion_{safe_label}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix: {cm_path}")

    return {"dataset": dataset_label, "accuracy": acc, "f1": f1,
            "precision": prec, "recall": rec, "n_classes": n,
            "n_samples": len(X)}


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: list, out_dir: Path):
    print(f"\n{'='*65}")
    print("  FINAL SUMMARY — MODEL PERFORMANCE PER DATASET")
    print(f"{'='*65}")
    header = f"{'Dataset':<35} {'Classes':>7} {'Samples':>8} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}"
    print(header)
    print("-" * 65)

    lines = [header, "-" * 65]
    for r in results:
        if r is None:
            continue
        row = (f"{r['dataset']:<35} {r['n_classes']:>7} {r['n_samples']:>8} "
               f"{r['accuracy']*100:>6.1f}% {r['f1']*100:>6.1f}% "
               f"{r['precision']*100:>6.1f}% {r['recall']*100:>6.1f}%")
        print(row)
        lines.append(row)

    print(f"{'='*65}\n")

    summary_path = out_dir / "summary_all_datasets.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary saved: {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model separately on 3 ISL datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to trained Keras model (best_model or best_model_combined)")
    parser.add_argument("--action_mapping", required=True,
                        help="Path to action_mapping.json (or action_mapping_combined.json)")
    parser.add_argument("--data_dir",
                        default="/home/islabworker2/mya/vsl-recognition/data",
                        help="Root dir containing INCLUDE/, ISL_Video/, INCLUDE_24_Medical/")
    parser.add_argument("--out_dir",
                        default="/home/islabworker2/mya/vsl-recognition/eval_results",
                        help="Where to save reports and confusion matrices")
    parser.add_argument("--dataset", choices=["include", "isl", "medical", "all"],
                        default="all", help="Which dataset to evaluate")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip keypoint extraction (use existing .npy sequences)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    datasets = get_datasets(data_dir)

    # Load model
    print(f"\n[MODEL] Loading from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path)
    seq_len = model.input_shape[1]
    print(f"  Input shape : {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Seq length  : {seq_len}")

    # Load action mapping
    with open(args.action_mapping) as f:
        mapping = json.load(f)
    action_names = [mapping[str(i)] for i in range(len(mapping))]
    print(f"  Model classes: {len(action_names)}")

    # Select datasets
    if args.dataset == "all":
        selected = list(datasets.keys())
    else:
        selected = [args.dataset]

    t0 = time.time()
    results = []

    for key in selected:
        ds = datasets[key]
        label    = ds["label"]
        video_dir = ds["video_dir"]
        seq_dir   = ds["seq_dir"]

        # Step 1: Extract keypoints
        if not args.skip_extraction:
            ok = extract_dataset(video_dir, seq_dir, label)
            if not ok:
                results.append(None)
                continue
        else:
            if not seq_dir.exists():
                print(f"\n[SKIP] {label} — seq_dir not found: {seq_dir}")
                print("  Run without --skip_extraction to extract keypoints first.")
                results.append(None)
                continue
            print(f"\n[SKIP EXTRACTION] Using existing sequences: {seq_dir}")

        # Step 2: Evaluate
        result = evaluate_on_dataset(model, action_names, seq_dir, label, out_dir)
        results.append(result)

    # Final table
    valid = [r for r in results if r is not None]
    if valid:
        print_summary_table(valid, out_dir)

    elapsed = time.time() - t0
    print(f"Total time: {elapsed/60:.1f} minutes\n")


if __name__ == "__main__":
    main()
