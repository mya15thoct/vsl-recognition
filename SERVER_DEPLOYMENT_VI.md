# Hướng Dẫn Deploy Lên Server

Hướng dẫn chi tiết các lệnh cần chạy trên server Linux sau khi push GitHub.

---

## Chuẩn Bị Server

- Linux OS (Ubuntu 18.04+)
- Miniconda hoặc Anaconda
- Git
- GPU (tùy chọn, nhưng khuyến khích cho training nhanh)

---

## Các Bước Deploy

### 1. Clone Repository

```bash
# Di chuyển đến thư mục project
cd ~

# Clone từ GitHub
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Vào thư mục project
cd YOUR_REPO_NAME
```

---

### 2. Setup Môi Trường Conda

```bash
# Cho phép chạy script setup
chmod +x setup_linux.sh

# Chạy script setup (khuyến khích)
./setup_linux.sh

# HOẶC tạo environment thủ công
conda env create -f environment.yml
```

---

### 3. Kích Hoạt Environment

```bash
conda activate sign_language_detection
```

---

### 4. Kiểm Tra Cài Đặt

```bash
# Test Python và dependencies
python -c "import tensorflow as tf; import cv2; import mediapipe as mp; print('Cài đặt thành công!')"

# Kiểm tra GPU (nếu có)
python -c "import tensorflow as tf; print('GPU có sẵn:', tf.config.list_physical_devices('GPU'))"
```

---

### 5. Upload Dataset

**Nếu bạn đã có dataset ở máy local:**

```bash
# Chạy từ máy local của bạn
scp -r data/VSL_Isolated username@server_ip:/path/to/project/data/
```

**Hoặc tải dataset trực tiếp trên server:**

```bash
cd data
python prepare_dataset.py
```

---

### 6. Trích Xuất Keypoints

```bash
cd sign_language_detection
python -m data.collect_data
```

**Nếu server không có display:**
```bash
export DISPLAY=:0
python -m data.collect_data
```

---

### 7. Train Model

```bash
cd sign_language_detection

# Train thường
python train.py

# HOẶC chạy nền (khuyến khích cho training lâu)
nohup python train.py > training.log 2>&1 &

# Xem tiến trình
tail -f training.log
```

**Sử dụng screen (khuyến khích):**
```bash
screen -S training
python train.py
# Nhấn Ctrl+A, sau đó nhấn D để thoát
# Quay lại với: screen -r training
```

---

### 8. Theo Dõi Training

```bash
# Mở TensorBoard
tensorboard --logdir=logs/training --host=0.0.0.0 --port=6006 &

# Mở firewall cho port 6006
sudo ufw allow 6006
```

**Truy cập từ trình duyệt:**
```
http://SERVER_IP:6006
```

---

### 9. Đánh Giá Model

```bash
cd sign_language_detection
python evaluate.py
```

**Tải confusion matrix về máy local:**
```bash
# Chạy từ máy local
scp username@server_ip:/path/to/project/sign_language_detection/models/confusion_matrix.png ~/Downloads/
```

---

### 10. Chạy Inference (nếu server có camera)

```bash
cd sign_language_detection
python inference.py

# Hoặc với video file
python inference.py --video /path/to/video.mp4
```

---

## Chuỗi Lệnh Hoàn Chỉnh

Đây là chuỗi lệnh hoàn chỉnh để chạy trên server mới:

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Setup environment
chmod +x setup_linux.sh
./setup_linux.sh
conda activate sign_language_detection

# 3. Kiểm tra cài đặt
python -c "import tensorflow as tf; import cv2; import mediapipe as mp; print('OK')"

# 4. Upload dataset từ máy local (nếu có)
# Từ máy local: scp -r data/VSL_Isolated username@server_ip:/path/to/project/data/

# 5. Trích xuất keypoints
cd sign_language_detection
python -m data.collect_data

# 6. Train model (chạy nền)
nohup python train.py > training.log 2>&1 &

# 7. Theo dõi training
tail -f training.log

# 8. Mở TensorBoard (terminal khác)
tensorboard --logdir=logs/training --host=0.0.0.0 --port=6006 &

# 9. Sau khi training xong, đánh giá
python evaluate.py
```

---

## Lệnh Hữu Ích Trên Server

### Kiểm Tra GPU

```bash
# Xem thông tin GPU
nvidia-smi

# Theo dõi GPU theo thời gian thực
watch -n 1 nvidia-smi
```

### Kiểm Tra Process Training

```bash
# Liệt kê processes Python
ps aux | grep python

# Kill training nếu cần
pkill -f train.py
```

### Kiểm Tra Dung Lượng

```bash
df -h
du -sh data/
```

### Theo Dõi Tài Nguyên

```bash
htop
# hoặc
top
```

---

## Xử Lý Sự Cố

### Lỗi: Hết Memory

```bash
# Giảm batch size trong config.py
nano sign_language_detection/config.py
# Đổi BATCH_SIZE = 32 thành 16 hoặc 8
```

### Lỗi: Không Có Display

```bash
export DISPLAY=:0
```

### Lỗi: Permission Denied

```bash
chmod +x setup_linux.sh
chmod -R 755 sign_language_detection/
```

### Lỗi: Không Tìm Thấy Conda

```bash
# Thêm conda vào PATH
export PATH="$HOME/miniconda3/bin:$PATH"

# Hoặc vĩnh viễn
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## Trước Khi Push Lên GitHub

Tạo file `.gitignore`:

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

# OS
.DS_Store

# IDE
.vscode/
```

---

## Quy Trình Khuyến Nghị

1. **Máy Local:**
   - Phát triển code
   - Test với dataset nhỏ
   - Push lên GitHub

2. **Server:**
   - Clone từ GitHub
   - Upload dataset đầy đủ
   - Train với GPU
   - Download models đã train

3. **Máy Local:**
   - Download models
   - Chạy inference để test
