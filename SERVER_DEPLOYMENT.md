# Server Deployment Guide - Linux

Complete guide for deploying the Sign Language Detection project on a Linux server.

---

## Prerequisites on Server

- Linux OS (Ubuntu 18.04+ recommended)
- Miniconda or Anaconda installed
- Git installed
- GPU (optional, but recommended for faster training)

---

## Step-by-Step Deployment

### 1. Clone Repository from GitHub

```bash
# Navigate to your project directory
cd ~

# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Navigate into project directory
cd YOUR_REPO_NAME
```

---

### 2. Setup Conda Environment

```bash
# Make setup script executable (if not already)
chmod +x setup_linux.sh

# Run setup script (recommended)
./setup_linux.sh

# OR manually create environment
conda env create -f environment.yml
```

---

### 3. Activate Environment

```bash
conda activate sign_language_detection
```

---

### 4. Verify Installation

```bash
# Test Python and dependencies
python -c "import tensorflow as tf; import cv2; import mediapipe as mp; print('All imports successful')"

# Check TensorFlow GPU support (if you have GPU)
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

---

### 5. Prepare Dataset (Optional)

If you need to download dataset on server:

```bash
cd data
python prepare_dataset.py
```

**Note:** This requires Chrome/Chromium for web scraping. If not available:
```bash
# Install Chrome on Ubuntu
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
sudo apt-get install -f
```

**Alternative:** Upload your existing dataset from local machine:
```bash
# From your local machine
scp -r data/VSL_Isolated username@server_ip:/path/to/project/data/
```

---

### 6. Extract Keypoints

```bash
cd sign_language_detection
python -m data.collect_data
```

**For headless server (no display):**
```bash
# Set environment variable to avoid display issues
export DISPLAY=:0
python -m data.collect_data
```

---

### 7. Train Model

```bash
# Make sure you're in sign_language_detection directory
cd sign_language_detection

# Start training
python train.py
```

**Run in background (recommended for long training):**
```bash
# Using nohup
nohup python train.py > training.log 2>&1 &

# Check progress
tail -f training.log

# Or using screen
screen -S training
python train.py
# Press Ctrl+A, then D to detach
# Reattach with: screen -r training
```

---

### 8. Monitor Training Progress

```bash
# Start TensorBoard
tensorboard --logdir=logs/training --host=0.0.0.0 --port=6006
```

**Access from your browser:**
```
http://SERVER_IP:6006
```

**Note:** Make sure port 6006 is open in firewall:
```bash
sudo ufw allow 6006
```

---

### 9. Evaluate Model

```bash
cd sign_language_detection
python evaluate.py
```

**View confusion matrix:**
```bash
# Copy to local machine to view
scp username@server_ip:/path/to/project/sign_language_detection/models/confusion_matrix.png ~/Downloads/
```

---

### 10. Run Inference (if server has camera)

```bash
cd sign_language_detection
python inference.py
```

**For video file:**
```bash
python inference.py --video /path/to/video.mp4
```

---

## Complete Command Sequence

Here's the complete sequence to run on a fresh server:

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Setup environment
chmod +x setup_linux.sh
./setup_linux.sh
conda activate sign_language_detection

# 3. Verify installation
python -c "import tensorflow as tf; import cv2; import mediapipe as mp; print('OK')"

# 4. Upload dataset (if you have it locally)
# From local: scp -r data/VSL_Isolated username@server_ip:/path/to/project/data/

# 5. Collect keypoints
cd sign_language_detection
python -m data.collect_data

# 6. Train model (in background)
nohup python train.py > training.log 2>&1 &

# 7. Monitor training
tail -f training.log

# 8. Start TensorBoard (in another terminal)
tensorboard --logdir=logs/training --host=0.0.0.0 --port=6006 &

# 9. After training completes, evaluate
python evaluate.py
```

---

## Useful Server Commands

### Check GPU Usage (if GPU available)

```bash
# Install nvidia-smi if not available
nvidia-smi

# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

### Check Training Process

```bash
# List Python processes
ps aux | grep python

# Kill training if needed
pkill -f train.py
```

### Check Disk Space

```bash
df -h
du -sh data/
```

### Monitor System Resources

```bash
# CPU and Memory
htop

# Or
top
```

---

## Troubleshooting on Server

### Issue: Out of Memory

```bash
# Reduce batch size in config.py
nano sign_language_detection/config.py
# Change BATCH_SIZE = 32 to BATCH_SIZE = 16 or 8
```

### Issue: No Display for OpenCV

```bash
# Set headless mode
export DISPLAY=:0

# Or modify code to save results instead of showing
```

### Issue: Permission Denied

```bash
# Make sure files are executable
chmod +x setup_linux.sh
chmod -R 755 sign_language_detection/
```

### Issue: Conda Command Not Found

```bash
# Add conda to PATH
export PATH="$HOME/miniconda3/bin:$PATH"

# Or permanently
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## Files to .gitignore Before Push

Make sure these are in `.gitignore`:

```
# Data
data/VSL_Isolated/
data/qipedc_raw/

# Models
sign_language_detection/models/
sign_language_detection/logs/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/

# Jupyter
*.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```

---

## Recommended Workflow

1. **Local Development:**
   - Develop and test code locally
   - Use small dataset for testing
   - Push code to GitHub

2. **Server Training:**
   - Clone from GitHub
   - Upload full dataset (if large)
   - Train on server GPU
   - Download trained models

3. **Local Inference:**
   - Download trained models
   - Run inference locally for testing

---

## Performance Tips

1. **Use GPU:** Ensure TensorFlow can access GPU
2. **Batch Size:** Increase if you have enough memory
3. **Mixed Precision:** For faster training on newer GPUs
   ```python
   # Add to train.py
   from tensorflow.keras import mixed_precision
   policy = mixed_precision.Policy('mixed_float16')
   mixed_precision.set_global_policy(policy)
   ```

4. **Data Pipeline:** Use tf.data API for faster data loading
5. **Model Checkpoint:** Only save best models to save disk space
