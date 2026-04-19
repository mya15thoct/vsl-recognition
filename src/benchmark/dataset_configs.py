"""
Configurations for the 3 public benchmark datasets.

HOW TO VERIFY KAGGLE SLUGS:
  Go to each dataset page and copy the slug from the URL:
    https://www.kaggle.com/datasets/<owner>/<dataset-name>

  - INCLUDE (full, 263 classes):
      https://www.kaggle.com/datasets/vaishnaviasonawane/include-dataset
  - ISL Video Dataset:
      Search "ISL video dataset" on Kaggle; reference: Madhav et al. (LNNS 2025)
  - INCLUDE-24 Medical Modified:
      https://www.kaggle.com/datasets/linardur/include-24-medical-modified
"""

from pathlib import Path

BASE_DIR = Path("/mnt/ngan/benchmark")

DATASETS = {
    "include": {
        "name": "INCLUDE",
        "description": "Indian Sign Language, 263 word-level signs (videos)",
        # Verify at: https://www.kaggle.com/datasets/vaishnaviasonawane/include-dataset
        "kaggle_slug": "vaishnaviasonawane/include-dataset",
        "expected_classes": 263,
        "raw_dir":        BASE_DIR / "include" / "raw",
        "seq_dir":        BASE_DIR / "include" / "sequences",
        "checkpoint_dir": BASE_DIR / "include" / "checkpoints",
        "logs_dir":       BASE_DIR / "include" / "logs",
        "video_extensions": [".mp4", ".avi", ".mov"],
        "paper": "Madhav et al. (2025), LNNS vol. 1117, Springer",
    },
    "isl": {
        "name": "ISL Video Dataset",
        "description": "Indian Sign Language video dataset (Madhav et al.)",
        # UPDATE THIS SLUG after searching Kaggle for the dataset used in:
        # "An efficient real-time word-level recognition of Indian sign language"
        "kaggle_slug": "UPDATE_ME/isl-video-dataset",
        "expected_classes": None,
        "raw_dir":        BASE_DIR / "isl" / "raw",
        "seq_dir":        BASE_DIR / "isl" / "sequences",
        "checkpoint_dir": BASE_DIR / "isl" / "checkpoints",
        "logs_dir":       BASE_DIR / "isl" / "logs",
        "video_extensions": [".mp4", ".avi", ".mov"],
        "paper": "Madhav et al. (2025), LNNS vol. 1117, Springer",
    },
    "include24": {
        "name": "INCLUDE-24 Medical Modified",
        "description": "24 medical sign classes, modified from INCLUDE",
        # Confirmed at: https://www.kaggle.com/datasets/linardur/include-24-medical-modified
        "kaggle_slug": "linardur/include-24-medical-modified",
        "expected_classes": 24,
        "raw_dir":        BASE_DIR / "include24" / "raw",
        "seq_dir":        BASE_DIR / "include24" / "sequences",
        "checkpoint_dir": BASE_DIR / "include24" / "checkpoints",
        "logs_dir":       BASE_DIR / "include24" / "logs",
        "video_extensions": [".mp4", ".avi", ".mov"],
        "paper": "linardur (2023), Kaggle",
    },
}
