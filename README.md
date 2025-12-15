# Vietnamese Sign Language Detection

Sign language action detection for Vietnamese Sign Language (VSL) using MediaPipe and LSTM.

## Project Structure

```
extract_point/
├── sign_language_detection/     # Main source code
│   ├── train.py                 # Model training script
│   ├── inference.py             # Inference script (webcam/video)
│   ├── config.py                # Configuration and parameters
│   ├── data/                    # Data processing module
│   ├── model/                   # LSTM model architecture
│   └── utils/                   # Utility functions (MediaPipe, visualization)
│
└── data/                        # Data and dataset
    ├── prepare_dataset.py       # Dataset preparation script from QIPEDC
    ├── VSL_Isolated/            # Dataset videos and frames
    └── qipedc_raw/              # Raw crawled data
```

## Installation

### Option 1: Conda (Recommended for Linux)

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate sign_language_detection
```

Or use the setup script:
```bash
chmod +x setup_linux.sh
./setup_linux.sh
```

### Option 2: Pip

```bash
cd sign_language_detection
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset (Optional)

If you don't have a dataset yet, crawl from QIPEDC:

```bash
cd data
python prepare_dataset.py
```

This will download Vietnamese Sign Language videos from the QIPEDC website.

### 2. Collect Keypoints

Extract keypoints from videos:

```bash
cd sign_language_detection
python -m data.collect_data
```

This processes all videos in `../data/VSL_Isolated/` and saves keypoint sequences.

### 3. Train Model

```bash
python train.py
```

Models will be saved to the `models/` folder.

### 4. Run Inference

**Using webcam:**
```bash
python inference.py
```

**Using video file:**
```bash
python inference.py --video path/to/video.mp4
```

Press `q` to quit.

## Technology Stack

- **MediaPipe Holistic**: Extract keypoints (pose, face, hands)
- **LSTM**: Sequential action classification model
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Video and image processing

## Dataset

Dataset is crawled from [QIPEDC](https://qipedc.moet.gov.vn) - Vietnamese Sign Language Dictionary.

## Additional Information

- For detailed module documentation: [sign_language_detection/README.md](sign_language_detection/README.md)
- Dataset preparation script: [data/prepare_dataset.py](data/prepare_dataset.py)

## License

Research and educational purposes.
