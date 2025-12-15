# Quick Server Commands - Cheat Sheet

## Chuỗi Lệnh Nhanh Trên Server

```bash
# 1. Clone & Setup (chỉ chạy 1 lần)
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
chmod +x setup_linux.sh
./setup_linux.sh
conda activate sign_language_detection

# 2. Upload dataset (từ máy local)
scp -r data/VSL_Isolated username@server_ip:/path/to/project/data/

# 3. Chạy trên server
cd sign_language_detection
python -m data.collect_data              # Trích xuất keypoints
nohup python train.py > train.log 2>&1 & # Train (chạy nền)
tail -f train.log                        # Xem tiến trình
python evaluate.py                       # Đánh giá accuracy
```

## Theo Dõi Training

```bash
# TensorBoard
tensorboard --logdir=logs/training --host=0.0.0.0 --port=6006 &

# Truy cập: http://SERVER_IP:6006
```

## Các Lệnh Hữu Ích

```bash
# Xem processes
ps aux | grep python

# Kill training
pkill -f train.py

# Xem GPU
nvidia-smi
watch -n 1 nvidia-smi

# Screen (khuyến khích)
screen -S training
python train.py
# Ctrl+A, D để thoát
screen -r training  # Quay lại
```

---

📖 **Xem chi tiết:** [SERVER_DEPLOYMENT_VI.md](SERVER_DEPLOYMENT_VI.md)
