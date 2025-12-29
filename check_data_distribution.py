"""
Check data distribution across classes
"""
from pathlib import Path
import sys

# Path to sequences
SEQUENCE_PATH = Path(__file__).parent / "data" / "INCLUDE" / "sequences"

if not SEQUENCE_PATH.exists():
    print(f"Error: {SEQUENCE_PATH} does not exist")
    sys.exit(1)

# Get all class folders
class_folders = sorted([d for d in SEQUENCE_PATH.iterdir() if d.is_dir()])

print("=" * 70)
print("DATA DISTRIBUTION ANALYSIS")
print("=" * 70)
print(f"\nTotal classes: {len(class_folders)}\n")

# Count samples per class
distribution = []
for class_folder in class_folders:
    npy_files = list(class_folder.glob("*.npy"))
    count = len(npy_files)
    distribution.append((class_folder.name, count))

# Sort by count
distribution.sort(key=lambda x: x[1])

# Display
print(f"{'Class':<50} {'Samples':>10}")
print("-" * 70)

for class_name, count in distribution:
    marker = " ⚠️ TOO FEW!" if count < 2 else ""
    print(f"{class_name:<50} {count:>10}{marker}")

print("-" * 70)

# Summary
total_samples = sum(count for _, count in distribution)
min_samples = min(count for _, count in distribution)
max_samples = max(count for _, count in distribution)
avg_samples = total_samples / len(distribution) if distribution else 0

print(f"\nTotal samples: {total_samples}")
print(f"Min samples per class: {min_samples}")
print(f"Max samples per class: {max_samples}")
print(f"Average samples per class: {avg_samples:.2f}")

# Classes with < 2 samples
problematic = [(name, count) for name, count in distribution if count < 2]
if problematic:
    print(f"\n⚠️  PROBLEMATIC CLASSES (< 2 samples): {len(problematic)}")
    for name, count in problematic:
        print(f"   - {name}: {count} sample(s)")
else:
    print("\n✓ All classes have at least 2 samples")

print("=" * 70)
