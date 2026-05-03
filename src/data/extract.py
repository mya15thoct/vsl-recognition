"""
Extract keypoints from sign language videos and static images.

Two separate functions:
  extract_keypoints_from_videos() → recognition/sequences/   (from vsl_data/)
  extract_keypoints_from_images() → ISL-Sequences/           (from ISL-Frames-Data/)
"""
import cv2
import numpy as np
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, SEQUENCE_PATH, SEQUENCE_LENGTH
from utils.extraction import mediapipe_detection, extract_keypoints, get_holistic_model


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_keypoints_from_videos():
    """Extract keypoints from sign language videos (vsl_data/ → recognition/sequences/)."""
    print("=" * 60)
    print("EXTRACTING KEYPOINTS FROM VIDEOS")
    print(f"  Source: {DATA_DIR}")
    print(f"  Output: {SEQUENCE_PATH}")
    print("=" * 60)

    classes = sorted([d.name for d in DATA_DIR.iterdir()
                      if d.is_dir() and d.name != 'sequences'])

    print(f"\nFound {len(classes)} classes\n")

    print("Initializing MediaPipe Holistic...")
    holistic = get_holistic_model()

    total_processed = 0

    for class_idx, class_name in enumerate(classes):
        class_path = DATA_DIR / class_name

        videos = (list(class_path.glob('*.MOV')) +
                  list(class_path.glob('*.mov')) +
                  list(class_path.glob('*.MP4')) +
                  list(class_path.glob('*.mp4')))

        if not videos:
            print(f"[WARNING] No videos in {class_name}")
            continue

        print(f"[{class_idx+1}/{len(classes)}] '{class_name}' ({len(videos)} videos)")

        seq_class_path = SEQUENCE_PATH / class_name
        seq_class_path.mkdir(parents=True, exist_ok=True)

        for video_path in videos:
            cap = cv2.VideoCapture(str(video_path))
            frames_data = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                frames_data.append(keypoints)

            cap.release()

            if not frames_data:
                print(f"  [WARNING] No frames from {video_path.name}")
                continue

            sequence_data = np.array(frames_data)
            np.save(seq_class_path / f"{video_path.stem}.npy", sequence_data)
            total_processed += 1

        print(f"  → {len(videos)} sequences saved")

    holistic.close()

    print(f"\n{'=' * 60}")
    print(f"[OK] VIDEO EXTRACTION COMPLETE")
    print(f"  Total: {total_processed}  |  Saved to: {SEQUENCE_PATH}")
    print(f"{'=' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_keypoints_from_images(image_dir=None, sequence_path=None):
    """
    Extract keypoints from static images (ISL-Frames-Data/ → ISL-Sequences/).

    Each image saved as (1, 1662) .npy:
        ISL-Frames-Data/Come/img1.jpg  →  ISL-Sequences/Come/img1_static.npy
    """
    from config import IMAGE_DIR, ISL_SEQUENCE_PATH

    image_dir     = Path(image_dir     or IMAGE_DIR)
    sequence_path = Path(sequence_path or ISL_SEQUENCE_PATH)

    print("=" * 60)
    print("EXTRACTING KEYPOINTS FROM STATIC IMAGES")
    print(f"  Source: {image_dir}")
    print(f"  Output: {sequence_path}")
    print("=" * 60)

    if not image_dir.exists():
        print(f"[ERROR] image_dir not found: {image_dir}")
        return

    classes = sorted([d.name for d in image_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(classes)} classes\n")

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    print("Initializing MediaPipe Holistic...")
    holistic = get_holistic_model()

    total_processed = 0
    total_skipped   = 0

    for class_idx, class_name in enumerate(classes):
        class_path = image_dir / class_name
        images = sorted([p for p in class_path.iterdir()
                         if p.suffix.lower() in IMAGE_EXTENSIONS])

        if not images:
            print(f"  [WARNING] No images in {class_name}")
            continue

        print(f"  [{class_idx+1}/{len(classes)}] '{class_name}' ({len(images)} images)")

        seq_class_path = sequence_path / class_name
        seq_class_path.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"    [WARNING] Cannot read {img_path.name}")
                total_skipped += 1
                continue

            image, results = mediapipe_detection(frame, holistic)
            keypoints     = extract_keypoints(results)       # (1662,)
            sequence_data = keypoints[np.newaxis, :]         # (1, 1662)
            save_name     = f"{img_path.stem}_static"
            np.save(seq_class_path / f"{save_name}.npy", sequence_data)
            total_processed += 1

    holistic.close()

    print(f"\n{'=' * 60}")
    print(f"[OK] IMAGE EXTRACTION COMPLETE")
    print(f"  Processed: {total_processed}  |  Skipped: {total_skipped}")
    print(f"  Saved to:  {sequence_path}")
    print(f"{'=' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['video', 'image', 'both'], default='video')
    parser.add_argument('--image_dir',     type=str, default=None,
                        help='Source images folder (default: IMAGE_DIR from config)')
    parser.add_argument('--sequence_path', type=str, default=None,
                        help='Output folder for images (default: ISL_SEQUENCE_PATH)')
    args = parser.parse_args()

    if args.mode in ('video', 'both'):
        extract_keypoints_from_videos()
    if args.mode in ('image', 'both'):
        extract_keypoints_from_images(image_dir=args.image_dir,
                                      sequence_path=args.sequence_path)
