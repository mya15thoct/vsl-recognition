from pathlib import Path

BASE_DIR = Path("/home/islabworker2/mya/vsl-recognition/benchmark")

DATASETS = {
    "include": {
        "name": "INCLUDE",
        "description": "Indian Sign Language, 263 word-level signs (videos)",
        # https://www.kaggle.com/datasets/daskoushik/include  (~56 GB)
        "kaggle_slug": "daskoushik/include",
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
        "description": "Indian Sign Language video dataset",
        # https://www.kaggle.com/datasets/prasadshet/indian-sign-language-video-dataset  (~3.4 GB)
        "kaggle_slug": "prasadshet/indian-sign-language-video-dataset",
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
        # https://www.kaggle.com/datasets/linardur/include-24-medical-modified  (~7 GB)
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
