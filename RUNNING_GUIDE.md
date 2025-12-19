# Hướng Dẫn Chạy Project - Chi Tiết

## 📋 QUY TRÌNH HOÀN CHỈNH

```
Bước 1: Validate MediaPipe Quality
    ↓
Bước 2: Extract Keypoints (nếu quality OK)
    ↓
Xong! (Chưa có LSTM, chỉ focus extraction)
```

---

## 🚀 BƯỚC 1: VALIDATE MEDIAPIPE QUALITY

### Lệnh:
```bash
cd sign_language_detection
python -m scripts.validate_extraction
```

### Quá trình diễn ra:

#### 1.1. **Khởi tạo (1-2 giây)**
```
- Load MediaPipe Holistic model
- Cấu hình: confidence = 0.5 (từ config.py)
- Tìm tất cả videos trong data/VSL_Isolated/videos/
```

#### 1.2. **Random chọn videos (1 giây)**
```
- Tổng số videos trong dataset: X videos
- Random chọn 10 videos để validate
```

#### 1.3. **Validate video đầu tiên (VỚI VISUALIZATION)** 
```
Video 1: 000001.mp4
    ↓
Mở cửa sổ hiển thị video với keypoints overlay
    ↓
Với mỗi frame (30-60 frames/video):
    1. Đọc frame từ video
    2. MediaPipe detect: Pose, Face, Hands
    3. Vẽ keypoints lên frame
    4. Hiển thị lên màn hình
    5. Lưu metrics:
       - Detection: Có detect được không?
       - Confidence: Độ tin cậy (0-1)
       - Position: Vị trí keypoints
    ↓
Tính toán metrics cho video:
    - Detection rate: 98.5% (detect được 98.5% frames)
    - Confidence: 0.856 (độ tin cậy trung bình)
    - Consistency: 0.782 (chuyển động mượt)
    ↓
Kết luận: ✅ Good hoặc ❌ Poor
```

**Nhấn 'q' để skip visualization và qua video tiếp theo**

#### 1.4. **Validate 9 videos còn lại (KHÔNG hiển thị)**
```
Video 2, 3, 4, ..., 10
    ↓
Mỗi video (không hiển thị, chỉ tính toán):
    1. Đọc tất cả frames
    2. MediaPipe detect
    3. Tính metrics
    4. In kết quả
```

#### 1.5. **Tổng kết (cuối cùng)**
```
OVERALL SUMMARY:
- Good quality: 8/10 (80%)
- Average pose detection: 95.2%
- Average confidence: 0.823
- Average consistency: 0.756
    ↓
Kết luận:
✅ >= 80% → EXTRACTION QUALITY IS GOOD - Ready!
⚠️ 60-80% → ACCEPTABLE - Can improve
❌ < 60% → NEEDS IMPROVEMENT
```

### Output mẫu:
```
MEDIAPIPE EXTRACTION QUALITY VALIDATION
========================================

Total videos in dataset: 81
Validating 10 sample videos...

[Cửa sổ hiển thị video với keypoints...]

✅ Video 1: 000001.mp4
   Total frames: 30
   Pose detection: 100.0%
   Face detection: 100.0%
   Left hand: 96.7%
   Right hand: 93.3%
   Avg confidence: 0.892
   Consistency: 0.845

✅ Video 2: 000002.mp4
   Total frames: 28
   Pose detection: 96.4%
   ...

OVERALL SUMMARY
===============
Good quality: 9/10 (90%)
Average pose detection: 97.8%
Average confidence: 0.856

✅ EXTRACTION QUALITY IS GOOD
   Ready for extraction!
```

### Thời gian: **2-5 phút** (tùy số frames)

---

## 🎯 BƯỚC 2: EXTRACT KEYPOINTS

### Chỉ chạy NẾU bước 1 cho kết quả ≥80% good!

### Lệnh:
```bash
cd sign_language_detection
python -m data.collect_data
```

### Quá trình diễn ra:

#### 2.1. **Khởi tạo (2-3 giây)**
```
- Load MediaPipe Holistic
- Scan data/VSL_Isolated/ để tìm tất cả folders
- Tìm thấy: 81 actions (81 từ ngôn ngữ ký hiệu)
```

#### 2.2. **Xử lý từng action (81 lần)**
```
For mỗi action (000001, 000002, ..., 000081):
    ↓
    1. Kiểm tra folder:
       - Có tồn tại không?
       - Có video không?
       
    2. Load videos (.mp4, .avi)
    
    3. Xử lý mỗi video:
       
       Video 1 của action 000001:
           ↓
           a. Mở video
           b. Đọc 30 frames đầu tiên
           c. Mỗi frame:
              - MediaPipe detect
              - Extract 1662 keypoints
              - Lưu vào array
           d. Gom 30 frames thành 1 sequence
           e. Save: data/VSL_Isolated/sequences/000001/0/0.npy
              
       Video 2:
           → sequences/000001/1/1.npy
           
       ... (lặp lại cho 30 videos)
    
    4. In kết quả:
       [OK] Saved 30 sequences
```

#### 2.3. **Output cuối cùng**
```
COLLECTION COMPLETE
- Total sequences: 2,430
  (81 actions × 30 sequences)
- Saved to: data/VSL_Isolated/sequences/
```

### Cấu trúc output:
```
data/VSL_Isolated/sequences/
├── 000001/
│   ├── 0/
│   │   └── 0.npy          # Shape: (30, 1662)
│   ├── 1/
│   │   └── 1.npy          # Shape: (30, 1662)
│   └── ...
│       └── 29.npy
├── 000002/
│   └── ...
└── 000081/
    └── ...
```

**Mỗi file .npy chứa:**
- 30 frames
- Mỗi frame: 1662 keypoints
- Shape: (30, 1662)
- Dung lượng: ~400KB/file

### Thời gian: **10-30 phút** (tùy số videos)

---

## 📊 METRICS & GIẢI THÍCH

### Detection Rate (Tỷ lệ detect)
```
= (Số frames detect được / Tổng số frames) × 100%

Tốt: > 95%
Trung bình: 80-95%
Kém: < 80%
```

**Ý nghĩa:** MediaPipe có thể detect được bao nhiêu % frames trong video

### Confidence Score (Độ tin cậy)
```
= Trung bình visibility của tất cả keypoints

Tốt: > 0.8
Trung bình: 0.5-0.8
Kém: < 0.5
```

**Ý nghĩa:** MediaPipe chắc chắn bao nhiêu % về vị trí keypoints

### Consistency (Nhất quán)
```
= Đo độ mượt của chuyển động giữa các frames

Tốt: > 0.7
Trung bình: 0.4-0.7
Kém: < 0.4
```

**Ý nghĩa:** Keypoints có nhảy lung tung không, chuyển động có mượt không

---

## 🔧 TROUBLESHOOTING

### Vấn đề 1: Validation cho kết quả < 80%

**Nguyên nhân:**
- Video quality kém
- Lighting xấu
- Người ở xa camera
- Camera angle không tốt

**Giải pháp:**
1. Xem video nào kém trong kết quả
2. Kiểm tra video đó
3. Thay thế hoặc re-record video kém

### Vấn đề 2: MediaPipe không detect được

**Nguyên nhân:**
- Video quá tối
- Người quay lưng
- Bị che khuất

**Giải pháp:**
- Xóa video đó
- Hoặc re-record

### Vấn đề 3: Lỗi import

**Lỗi:**
```
ModuleNotFoundError: No module named 'mediapipe'
```

**Giải pháp:**
```bash
pip install mediapipe opencv-python numpy
```

---

## ✅ CHECKLIST

Trước khi chạy:
- [ ] Đã cài dependencies: `pip install -r requirements.txt`
- [ ] Có data trong: `data/VSL_Isolated/videos/`
- [ ] Đã vào đúng folder: `cd sign_language_detection`

Sau khi validate (Bước 1):
- [ ] Quality ≥ 80% → Proceed to Bước 2
- [ ] Quality < 80% → Fix data trước

Sau khi extract (Bước 2):
- [ ] Kiểm tra: `data/VSL_Isolated/sequences/` có files không
- [ ] Kiểm tra số lượng: 81 folders × 30 sequences = 2,430 files

---

## 🎯 LỆNH NHANH

```bash
# Bước 1: Validate
cd "c:\IT\sign language\extract_point\sign_language_detection"
python -m scripts.validate_extraction

# Nếu kết quả >= 80% good:

# Bước 2: Extract
python -m data.collect_data

# Xong! Kiểm tra output
ls ../data/VSL_Isolated/sequences/
```
