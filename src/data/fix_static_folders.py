"""
Cleanup script: Fix case-mismatch folders in sequences directory.

Problem: Image extraction (old code) created folders with wrong case
  e.g. 'Afternoon' instead of 'AFTERNOON', 'Adult' instead of 'ADULT'

This script:
  1. Finds all *_static.npy files in wrong-case folders
  2. Moves them to the correct (video) folder
  3. Deletes the now-empty wrong-case folders

Usage:
  python src/data/fix_static_folders.py
  python src/data/fix_static_folders.py --dry_run   ← preview only, no changes
"""

import shutil
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import SEQUENCE_PATH


def fix_static_folders(sequence_path=None, dry_run=False):
    seq_path = Path(sequence_path or SEQUENCE_PATH)

    print("=" * 60)
    print("FIXING STATIC IMAGE FOLDER CASE MISMATCH")
    if dry_run:
        print("  [DRY RUN] No changes will be made")
    print("=" * 60)

    all_folders  = [d for d in seq_path.iterdir() if d.is_dir()]

    # Build lookup: UPPERCASE → actual folder name
    # Priority: folders with VIDEO files (no _static suffix) = "correct" folders
    video_folders  = {}  # uppercase → Path
    static_folders = {}  # uppercase → Path

    for folder in all_folders:
        has_video  = any(f for f in folder.glob('*.npy') if '_static' not in f.stem)
        has_static = any(folder.glob('*_static.npy'))
        key = folder.name.upper()

        if has_video:
            video_folders[key] = folder
        elif has_static:
            static_folders[key] = folder

    print(f"\nFolders with videos  : {len(video_folders)}")
    print(f"Folders with only images (potential wrong-case): {len(static_folders)}")
    print()

    moved   = 0
    deleted = 0
    skipped = 0

    for key, static_folder in sorted(static_folders.items()):
        # Check if there's a matching video folder
        if key in video_folders:
            target_folder = video_folders[key]

            if static_folder.name == target_folder.name:
                # Same name → already correct, skip
                skipped += 1
                continue

            # Move all *_static.npy files to target folder
            static_files = list(static_folder.glob('*_static.npy'))
            print(f"  MERGE: '{static_folder.name}' → '{target_folder.name}'"
                  f" ({len(static_files)} files)")

            for f in static_files:
                dest = target_folder / f.name
                if not dry_run:
                    shutil.move(str(f), str(dest))
                moved += 1

            # Delete now-empty wrong-case folder
            if not dry_run:
                try:
                    static_folder.rmdir()
                    deleted += 1
                    print(f"    Deleted empty folder: {static_folder.name}")
                except OSError:
                    print(f"    [WARNING] Could not delete {static_folder.name} (not empty?)")
            else:
                print(f"    [DRY RUN] Would delete: {static_folder.name}")
                deleted += 1
        else:
            print(f"  [NO MATCH] '{static_folder.name}' — no video folder match, left as-is")

    print(f"\n{'=' * 60}")
    if dry_run:
        print(f"[DRY RUN] Would move:   {moved} files")
        print(f"[DRY RUN] Would delete: {deleted} folders")
    else:
        print(f"[OK] Moved:   {moved} static .npy files")
        print(f"[OK] Deleted: {deleted} wrong-case folders")
    print(f"     Skipped: {skipped} (already correct)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence_path', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true',
                        help='Preview changes without modifying files')
    args = parser.parse_args()

    fix_static_folders(sequence_path=args.sequence_path, dry_run=args.dry_run)
