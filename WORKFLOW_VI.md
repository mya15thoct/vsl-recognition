# Hướng Dẫn Sử Dụng Project - Vietnamese Guide

## Tổng Quan Luồng Hoạt Động

```
Bước 1: Chuẩn bị dữ liệu
    ↓
Bước 2: Trích xuất keypoints
    ↓
Bước 3: Huấn luyện model
    ↓
Bước 4: Chạy inference (test)
```

---

## Cách Chạy Project (Chi Tiết)

### Bước 1: Chuẩn Bị Dataset (Tùy chọn)

**Nếu chưa có dataset:**

```bash
cd data
python prepare_dataset.py
```

**Làm gì:**
- Tải video ngôn ngữ ký hiệu từ website QIPEDC
- Lưu video và trích xuất frames
- Tạo cấu trúc thư mục `VSL_Isolated/`

**Kết quả:**
- `data/VSL_Isolated/videos/` - Các video đã tải
- `data/VSL_Isolated/frames/` - Frames từ video
- `data/VSL_Isolated/dictionary.txt` - Ánh xạ class

---

### Bước 2: Trích Xuất Keypoints

```bash
cd sign_language_detection
python -m data.collect_data
```

**Làm gì:**
- Dùng MediaPipe để phát hiện pose, mặt, tay
- Xử lý từng frame của video
- Trích xuất 1662 keypoints/frame
- Lưu chuỗi 30 frames/video

**Kết quả:**
- `data/VSL_Isolated/sequences/[hành động]/[số thứ tự]/[số thứ tự].npy`

---

### Bước 3: Huấn Luyện Model

```bash
cd sign_language_detection
python train.py
```

**Làm gì:**
- Load tất cả keypoint sequences
- Chia dữ liệu train/test (80/20)
- Huấn luyện LSTM model
- Lưu model tốt nhất

**Kiến trúc model:**
```
Input (30 frames × 1662 keypoints)
    ↓
3 LSTM Layers (64, 128, 64 units)
    ↓
2 Dense Layers (64, 32 units)
    ↓
Output (số hành động)
```

**Kết quả:**
- `models/action_model_best.h5` - Model tốt nhất
- `models/action_model_final.h5` - Model cuối cùng
- `models/actions.npy` - Danh sách hành động
- `logs/training/` - Logs cho TensorBoard

**Trong quá trình train, bạn sẽ thấy:**
```
Epoch 1/2000
loss: 2.5431 - accuracy: 0.2341 - val_accuracy: 0.3211
Epoch 2/2000
loss: 1.8234 - accuracy: 0.4512 - val_accuracy: 0.5123
...
```

---

### Bước 4: Chạy Inference (Test)

**Webcam (thời gian thực):**
```bash
cd sign_language_detection
python inference.py
```

**File video:**
```bash
python inference.py --video duong/dan/video.mp4
```

**Nhấn 'q' để thoát**

---

## Đánh Giá Accuracy (Độ Chính Xác)

### Cách 1: Xem Metrics Khi Train

Trong quá trình train, bạn đã thấy:
- `accuracy` - Độ chính xác trên tập train
- `val_accuracy` - Độ chính xác trên tập test

**Xem chi tiết trên TensorBoard:**
```bash
cd sign_language_detection
tensorboard --logdir=logs/training
```

Mở trình duyệt: http://localhost:6006

### Cách 2: Chạy Script Đánh Giá

Tôi đã tạo file `evaluate.py` để đánh giá chi tiết:

```bash
cd sign_language_detection
python evaluate.py
```

**Kết quả:**
```
Test Accuracy: 0.8523 (85.23%)

Classification Report:
              precision    recall  f1-score   support
    000001       0.85      0.90      0.87        10
    000002       0.88      0.82      0.85         8
    ...

Per-Class Accuracy:
000001: 90.00% (10 samples)
000002: 82.50% (8 samples)
...

Confusion matrix saved to: models/confusion_matrix.png
```

**File `evaluate.py` sẽ tạo:**
- Report đầy đủ về accuracy, precision, recall, f1-score
- Accuracy từng class
- Confusion matrix (ma trận nhầm lẫn)
- Biểu đồ confusion matrix (PNG)

---

## Các Lệnh Nhanh

```bash
# Luồng hoàn chỉnh từ đầu
cd data
python prepare_dataset.py          # 1. Chuẩn bị dataset

cd ../sign_language_detection
python -m data.collect_data         # 2. Trích xuất keypoints
python train.py                     # 3. Train model
python evaluate.py                  # 4. Đánh giá accuracy
python inference.py                 # 5. Test với webcam

# Xem quá trình training
tensorboard --logdir=logs/training
```

---

## Cấu Hình

Chỉnh sửa `sign_language_detection/config.py`:

```python
# Tham số training
EPOCHS = 2000              # Số epoch
BATCH_SIZE = 32           # Batch size
LEARNING_RATE = 0.001     # Learning rate

# Tham số dữ liệu
SEQUENCE_LENGTH = 30      # Số frames/sequence
NO_SEQUENCES = 30         # Số sequences/hành động

# Kiến trúc model
LSTM_UNITS = [64, 128, 64]
DENSE_UNITS = [64, 32]
```

---

## Giải Thích Chi Tiết

### MediaPipe Keypoints

MediaPipe trích xuất 1662 keypoints/frame:
- **Pose**: 33 điểm × 4 (x, y, z, visibility) = 132
- **Face**: 468 điểm × 3 (x, y, z) = 1404
- **Hands**: 21 điểm × 3 × 2 tay = 126

**Tổng: 132 + 1404 + 126 = 1662 keypoints**

### LSTM Model

- **Input**: Chuỗi 30 frames, mỗi frame 1662 keypoints
- **LSTM Layers**: Học patterns theo thời gian
- **Dense Layers**: Phân loại hành động
- **Output**: Xác suất cho mỗi hành động

### Train/Test Split

- **80% train**: Dùng để học
- **20% test**: Dùng để đánh giá
- **random_state=42**: Đảm bảo chia giống nhau mỗi lần

---

## Khắc Phục Sự Cố

**Vấn đề: Accuracy thấp**
- Tăng `EPOCHS` trong config.py
- Thu thập nhiều sequences hơn
- Tăng `SEQUENCE_LENGTH` cho video dài hơn
- Kiểm tra chất lượng dữ liệu

**Vấn đề: Overfitting**
- Thêm dropout layers vào model
- Giảm độ phức tạp model
- Tăng tỷ lệ train/test split

**Vấn đề: Train chậm**
- Giảm `EPOCHS`
- Giảm `SEQUENCE_LENGTH`
- Dùng GPU nếu có
- Giảm số lượng hành động

---

## Tài Liệu Tham Khảo

1. [WORKFLOW.md](WORKFLOW.md) - Hướng dẫn bằng tiếng Anh
2. [README.md](README.md) - Tổng quan project
3. [sign_language_detection/README.md](sign_language_detection/README.md) - Chi tiết module
