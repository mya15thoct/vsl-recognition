"""
Explore INCLUDE dataset structure and prepare for training
"""
import os
from pathlib import Path
import numpy as np

# Dataset path
data_dir = Path('data/INCLUDE/ProcessedData_vivit')

# List all classes
classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
classes = sorted(classes)

print("="*60)
print("INCLUDE DATASET STRUCTURE")
print("="*60)
print(f"\nTotal classes: {len(classes)}")
print(f"\nFirst 10 classes: {classes[:10]}")

# Count videos per class
print(f"\nVideos per class:")
for cls in classes[:10]:
    cls_path = data_dir / cls
    videos = list(cls_path.glob('*.MOV'))
    print(f"  {cls:15s}: {len(videos)} videos")

# Total videos
total_videos = sum(len(list((data_dir / cls).glob('*.MOV'))) for cls in classes)
print(f"\n{'='*60}")
print(f"TOTAL VIDEOS: {total_videos}")
print(f"{'='*60}")

# Sample a video to check
sample_class = classes[0]
sample_videos = list((data_dir / sample_class).glob('*.MOV'))
if sample_videos:
    sample_video = sample_videos[0]
    print(f"\nSample video: {sample_video}")
    print(f"File size: {sample_video.stat().st_size / 1024:.1f} KB")

print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("1. Extract keypoints from .MOV files")
print("2. Or check if landmarks already exist")
print("3. Update config.py with new paths")
print("4. Adapt data_loader.py")
print("="*60)
