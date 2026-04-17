"""
Extract keypoints from sign language videos
"""
import cv2
import numpy as np
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, SEQUENCE_PATH, SEQUENCE_LENGTH
from utils.extraction import mediapipe_detection, extract_keypoints, get_holistic_model

def extract_keypoints_from_videos():
    """
    Extract keypoints from sign language videos
    """
    print("="*60)
    print("EXTRACTING KEYPOINTS FROM DATASET")
    print("="*60)
    
    # Get all class folders
    classes = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != 'sequences']
    classes = sorted(classes)
    
    print(f"\nFound {len(classes)} classes")
    print(f"Sequence length: {SEQUENCE_LENGTH} frames")
    print(f"Output: {SEQUENCE_PATH}\n")
    
    # Create sequences directory
    SEQUENCE_PATH.mkdir(parents=True, exist_ok=True)
    
    # Initialize MediaPipe
    print("Initializing MediaPipe Holistic...")
    holistic = get_holistic_model()
    
    total_processed = 0
    
    for class_idx, class_name in enumerate(classes):
        class_path = DATA_DIR / class_name
        
        # Get all video files (.MOV and .MP4, case insensitive)
        videos = (list(class_path.glob('*.MOV')) + 
                 list(class_path.glob('*.mov')) +
                 list(class_path.glob('*.MP4')) + 
                 list(class_path.glob('*.mp4')))
        
        if not videos:
            print(f"[WARNING] No videos found in {class_name}")
            continue
            
        print(f"[{class_idx+1}/{len(classes)}] Processing '{class_name}' ({len(videos)} videos)")
        
        # Create class folder in sequences
        seq_class_path = SEQUENCE_PATH / class_name
        seq_class_path.mkdir(parents=True, exist_ok=True)
        
        # Process each video
        for video_idx, video_path in enumerate(videos):
            cap = cv2.VideoCapture(str(video_path))
            frames_data = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # MediaPipe detection
                image, results = mediapipe_detection(frame, holistic)
                
                # Extract keypoints
                keypoints = extract_keypoints(results)
                frames_data.append(keypoints)
            
            cap.release()
            
            # Skip if no frames extracted
            if len(frames_data) == 0:
                print(f"    [WARNING] No frames extracted from {video_path.name}")
                continue
            
            # Save sequence with all frames (no downsampling)
            sequence_data = np.array(frames_data)
            video_name = video_path.stem  # Filename without extension
            np.save(seq_class_path / f"{video_name}.npy", sequence_data)
            
            total_processed += 1
        
        print(f"  [OK] Saved {len(videos)} sequences")
    
    holistic.close()
    
    print(f"\n{'='*60}")
    print(f"[OK] EXTRACTION COMPLETE")
    print(f"  Total sequences: {total_processed}")
    print(f"  Classes: {len(classes)}")
    print(f"  Saved to: {SEQUENCE_PATH}")
    print(f"{'='*60}")

def extract_keypoints_from_images(image_dir=None, sequence_path=None):
    """
    Extract keypoints from static images (1 frame per image).

    Saves each image as a (1, 1662) .npy file in the same sequences folder
    structure as video sequences → data_loader.py loads them automatically.

    Expected image_dir structure (same as video DATA_DIR):
        image_dir/
            BRING/
                img1.jpg
                img2.png
            WATER/
                img1.jpg
            ...

    Args:
        image_dir:     Root folder containing per-class image subfolders.
                       Defaults to DATA_DIR / 'images'.
        sequence_path: Where to save .npy files.
                       Defaults to SEQUENCE_PATH (same as video sequences).
    """
    from config import DATA_DIR, SEQUENCE_PATH

    image_dir     = image_dir     or DATA_DIR / 'images'
    sequence_path = sequence_path or SEQUENCE_PATH

    print("=" * 60)
    print("EXTRACTING KEYPOINTS FROM STATIC IMAGES")
    print("=" * 60)

    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"[ERROR] image_dir not found: {image_dir}")
        return

    classes = sorted([d.name for d in image_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(classes)} image classes in {image_dir}")
    print(f"Output: {sequence_path}\n")

    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # Build case-insensitive lookup from EXISTING video sequence folders
    seq_path = Path(sequence_path)
    existing_folders = {d.name.upper(): d.name for d in seq_path.iterdir() if d.is_dir()} \
                       if seq_path.exists() else {}
    print(f"Found {len(existing_folders)} existing video sequence folders.")
    print("Matching image folder names to existing sequence folders (case-insensitive)...\n")

    print("Initializing MediaPipe Holistic...")
    holistic = get_holistic_model()

    total_processed = 0
    total_skipped   = 0
    total_unmatched = 0

    for class_idx, class_name in enumerate(classes):
        # Case-insensitive match to existing sequence folder
        matched_name = existing_folders.get(class_name.upper())
        if matched_name is None:
            print(f"  [SKIP] '{class_name}' — no matching video sequence folder found")
            total_unmatched += 1
            continue

        if matched_name != class_name:
            print(f"  [MATCH] '{class_name}' → '{matched_name}'")

        class_path = image_dir / class_name
        images = [p for p in class_path.iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS]

        if not images:
            print(f"  [WARNING] No images found in {class_name}")
            continue

        print(f"  [{class_idx+1}/{len(classes)}] '{matched_name}' ({len(images)} images)")

        # Save using the MATCHED folder name (matches video sequences exactly)
        seq_class_path = seq_path / matched_name
        seq_class_path.mkdir(parents=True, exist_ok=True)


        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"    [WARNING] Cannot read {img_path.name}, skipping")
                total_skipped += 1
                continue

            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)          # (1662,)

            # Save as (1, 1662) — same shape convention as 1-frame video
            sequence_data = keypoints[np.newaxis, :]        # (1, 1662)
            save_name = f"{img_path.stem}_static"           # avoid name clash with videos
            np.save(seq_class_path / f"{save_name}.npy", sequence_data)
            total_processed += 1

    holistic.close()

    print(f"\n{'=' * 60}")
    print(f"[OK] STATIC IMAGE EXTRACTION COMPLETE")
    print(f"  Processed:  {total_processed}")
    print(f"  Skipped:    {total_skipped}   (unreadable images)")
    print(f"  Unmatched:  {total_unmatched}  (no video folder match — skipped)")
    print(f"  Saved to:   {sequence_path}")
    print(f"  → data_loader.py will load these alongside video sequences")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['video', 'image', 'both'], default='video',
                        help='What to extract: video sequences, static images, or both')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Root folder of static images (default: DATA_DIR/images)')
    args = parser.parse_args()

    if args.mode in ('video', 'both'):
        extract_keypoints_from_videos()
    if args.mode in ('image', 'both'):
        extract_keypoints_from_images(image_dir=args.image_dir)
