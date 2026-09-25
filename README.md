# HỆ THỐNG ĐÁNH GIÁ CHUẨN COCO (COCOEVAL BENCHMARK) CHO 8 MÔ HÌNH OBJECT DETECTION

## 1. Mục tiêu và Ý nghĩa Khoa học
Trong nghiên cứu khoa học thị giác máy tính, mỗi framework (Ultralytics, Torchvision, MMDetection, PaddleDetection) thường có cách tính toán độ chính xác nội bộ khác nhau (khác nhau về thuật toán nội suy điểm PR curve, confidence threshold khi NMS, IoU threshold). Điều này có thể dẫn đến việc so sánh giữa các mô hình thiếu tính công bằng tuyệt đối.

Bộ công cụ này chuẩn hóa việc đánh giá toàn bộ **8 mô hình** về **duy nhất một thước đo chuẩn mực quốc tế**: **COCO Benchmark API (`pycocotools.cocoeval.COCOeval`)**:
- Tất cả các mô hình cùng nhận diện trên **1.481 ảnh** của tập Test chuẩn.
- Bounding boxes được trích xuất về định dạng chuẩn COCO: `{"image_id", "category_id", "bbox": [xmin, ymin, w, h], "score"}`.
- Đánh giá cùng một lúc bằng một hàm `COCOeval` duy nhất, đảm bảo tính khách quan và thuyết phục 100% trước Hội đồng Khoa học.

---

## 2. Phân chia trách nhiệm trong nhóm nghiên cứu
* **Trần Anh Vũ (Chủ nhiệm/Lead)**:
  * Phụ trách huấn luyện, tối ưu và đánh giá 2 mô hình cốt lõi:
    1. **YOLOv26X** (One-stage SOTA thế hệ mới nhất của Ultralytics).
    2. **Faster R-CNN + ResNet50** (Two-stage Baseline kinh điển của Torchvision).
  * Thiết kế, tối ưu khung đánh giá chuẩn COCO dùng chung (`Common_Evaluate`) và phụ trách việc hợp nhất (Master Merge) báo cáo 8 mô hình toàn đội.
* **Các đồng nghiệp trong nhóm**:
  * Phụ trách huấn luyện 6 mô hình còn lại:
    1. **FasterRCNN_MobileNetv3Large**
    2. **FCOS + ResNet50-FPN**
    3. **YOLOv11**
    4. **RetinaNet**
    5. **RFDETF-Medium** (RF-DETR / RT-DETR Medium)
    6. **RTDETR-L**
  * Các tham số cấu hình của 6 mô hình này đã được **để trống hoàn toàn** trong [`models_config.json`](models_config.json). Đồng nghiệp chỉ cần điền đường dẫn trọng số `.pt` / `.pth`, siêu tham số `patience` và bật `"enabled": true` cho mô hình mình phụ trách.

---

## 3. Tập dữ liệu Roboflow COCO (Version 5)
Hệ thống kết nối và đồng bộ tự động với tài khoản Roboflow:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="<ROBOFLOW_API_KEY>")  # Đọc tự động từ secrets.json hoặc biến môi trường
project = rf.workspace("nckhcict2025").project("completed-project")
version = project.version(5)
dataset = version.download("coco")
```
* **Số lớp nguyên liệu hợp lệ**: **32 lớp** (Hệ thống tự động loại bỏ lớp rác `19-8-BoSungAnhChoCacLopBiTh-v3zo` do Roboflow sinh ra với 0 annotations).
* **Số ảnh kiểm thử (Test Split)**: **1.481 ảnh** kích thước $640 \times 640$.

---

## 4. Bảng 10 thông số chuẩn được đo lường
| STT | Tên thông số | Định dạng | Phương pháp đo lường |
| :---: | :--- | :---: | :--- |
| 1 | **Mô hình** | Text | Tên định danh của mô hình |
| 2 | **Precision** | Float (`.4f`) | Độ chính xác tại $\text{IoU} = 0.50$ (TP / (TP + FP)) |
| 3 | **Recall** | Float (`.4f`) | Độ nhạy bao phủ đối tượng theo chuẩn COCO (AR@100 - `stats[8]`) |
| 4 | **mAP@50** | Float (`.4f`) | Mean Average Precision tại $\text{IoU} = 0.50$ (`stats[1]`) |
| 5 | **mAP@50-95** | Float (`.4f`) | COCO Primary Metric (trung bình AP từ IoU 0.50 đến 0.95, bước 0.05) |
| 6 | **Patience** | Integer | Số epoch chờ khi Early Stopping trong quá trình huấn luyện |
| 7 | **GFLOPs** | Float (`.2f`) | Độ phức tạp tính toán (tỷ phép tính) đo bằng `thop` trên Dummy Input $1 \times 3 \times 640 \times 640$ |
| 8 | **Parameters** | Float (`.2f` M) | Tổng số tham số của mô hình (triệu tham số) |
| 9 | **Latency** | Float (`.2f` ms) | Thời gian xử lý 1 ảnh trên GPU A100 (đo trên toàn bộ 1.481 ảnh với batch size = 1) |
| 10 | **FPS** | Float (`.2f`) | Tốc độ khung hình trên giây ($\text{FPS} = 1000 / \text{Latency}$) |

---

## 5. Cấu trúc thư mục tối giản & Bảo mật (Clean & Secure)
```text
Common_Evaluate/
├── common_evaluate.py       # Script thực thi duy nhất (Dataset, Adapters, Benchmark, COCOeval, Merge Engine)
├── models_config.json       # Cấu hình tập trung (Đã xóa API Key để an toàn chia sẻ/commit git)
├── .env                     # File biến môi trường chứa Roboflow API Key của bạn (Được .gitignore bảo vệ tuyệt đối)
├── .env.example             # File mẫu để đồng nghiệp sao chép và tự tạo .env
├── .gitignore               # Tự động loại trừ .env và kết quả tạm
└── README.md                # Tài liệu hướng dẫn chi tiết quy trình hợp tác nhóm
```

### Cơ chế bảo mật API Key riêng tư:
Hệ thống tự động tìm kiếm Roboflow API Key theo thứ tự ưu tiên:
1. **Biến môi trường hệ thống**: `ROBOFLOW_API_KEY`
2. **File môi trường riêng tư**: `Common_Evaluate/.env` (`ROBOFLOW_API_KEY=...`)
3. **File config**: `models_config.json` (dự phòng, để trống mặc định)

---

## 6. Quy trình làm việc nhóm phân tán (Decentralized Multi-User Workflow)

### Bước 1: Đồng nghiệp cấu hình mô hình của mình trong `models_config.json`
Đồng nghiệp chỉ cần mở file [`models_config.json`](models_config.json):
1. Đổi `"volume_name"` thành tên Modal Volume trên tài khoản Modal của mình (nếu dùng tên khác):
   ```json
   "modal_settings": {
     "volume_name": "volume-cua-dong-nghiep",
     "app_name": "app-common-coco-evaluate"
   }
   ```
2. Với 2 mô hình của Vũ (`YOLOv26X`, `FasterRCNN_ResNet50`): Đổi `"enabled": false`.
3. Với các mô hình mình phụ trách (ví dụ `YOLOv11` và `RetinaNet`):
   - Điền đường dẫn file trọng số `weights` trên Modal: `"/data/runs/TenThuMucTrain/best.pt"`
   - Điền `patience`: ví dụ `10`
   - Đặt `"enabled": true`

### Bước 2: Chạy đánh giá độc lập
* **Cách A: Chạy trên Modal Cloud GPU A100 (Khuyến nghị)**:
  ```bash
  # Tự động đọc config máy bạn, truyền lên Modal và chỉ đánh giá các mô hình có enabled: true
  modal run "Common_Evaluate/common_evaluate.py"::evaluate_my_models
  ```
  *Hoặc đánh giá đích danh 1 mô hình:*
  ```bash
  modal run "Common_Evaluate/common_evaluate.py"::evaluate_single --model-name YOLOv11
  ```
* **Cách B: Chạy trên máy cục bộ (Local / Colab / Kaggle)**:
  ```bash
  # Chạy các mô hình có enabled: true
  python "Common_Evaluate/common_evaluate.py"

  # Hoặc chỉ định rõ danh sách mô hình
  python "Common_Evaluate/common_evaluate.py" --models YOLOv11,RetinaNet
  ```

### Bước 3: Xuất kết quả độc lập & Nộp file về cho nhóm
Sau khi chạy xong, trong thư mục `Common_Evaluate/Results/` sẽ xuất hiện các tệp:
- `result_<TenMoHinh>.json` (Ví dụ `result_YOLOv11.json`, `result_RetinaNet.json`): Tệp JSON chứa đầy đủ 10 chỉ số chuẩn hóa và mAP của từng lớp trong 32 lớp.
- `{TenMoHinh}_confusion_matrix_raw.png` & `{TenMoHinh}_confusion_matrix_normalized.png`: Ma trận nhầm lẫn 300 DPI.
- `{TenMoHinh}_test_per_class_metrics.csv`: Bảng chi tiết 32 lớp.

> **LƯU Ý QUAN TRỌNG**: Đồng nghiệp **chỉ cần gửi file `result_<TenMoHinh>.json`** (kích thước vài KB) qua Zalo/Drive/Git cho Vũ, **không cần gửi file weights nặng hàng trăm MB!**

---

## 7. Hợp nhất kết quả toàn đội (Master Merge dành cho Vũ)
Khi bạn (Vũ) nhận được các file `result_*.json` từ các đồng nghiệp:
1. Gom tất cả các file `result_*.json` của toàn bộ 8 mô hình vào một thư mục (ví dụ `team_results/`).
2. Chạy lệnh hợp nhất duy nhất:
   ```bash
   python "Common_Evaluate/common_evaluate.py" --merge-dir ./team_results
   ```
3. Hệ thống sẽ tự động tổng hợp:
   - `full_8_models_scientific_comparison.md`: Bảng 10 cột chuẩn của cả 8 mô hình.
   - `full_8_models_comparison_table.csv`: File CSV sẵn sàng chèn vào bài báo NCKH.
   - `full_8_models_coco_metrics_comparison.png` (300 DPI): Biểu đồ so sánh trực quan 4 panel (mAP50, mAP50-95, GFLOPs, FPS).
   - `all_classes_cross_model_map50_matrix.csv`: Ma trận đối đầu mAP@50 chi tiết 32 lớp giữa cả 8 mô hình.
