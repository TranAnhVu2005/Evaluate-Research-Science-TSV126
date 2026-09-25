# HỆ THỐNG ĐÁNH GIÁ CHUẨN COCO (COCOEVAL BENCHMARK) CHO 8 MÔ HÌNH OBJECT DETECTION
### ĐỀ TÀI NGHIÊN CỨU KHOA HỌC: NHẬN DIỆN NGUYÊN LIỆU NẤU ĂN (32 LỚP)
**Đơn vị thực hiện**: Trường Công nghệ Thông tin & Truyền thông - Đại học Cần Thơ (CTU)  
**Chủ nhiệm đề tài / Lead**: Trần Anh Vũ  

---

## 1. Mục tiêu và Ý nghĩa Khoa học
Trong nghiên cứu thị giác máy tính, mỗi thư viện (Ultralytics, Torchvision, MMDetection, PaddleDetection) thường có cách tính toán độ chính xác nội bộ khác nhau (về thuật toán nội suy điểm trên đường cong P-R curve, ngưỡng NMS IoU, ngưỡng lọc confidence). Điều này khiến việc so sánh trực tiếp các chỉ số mAP được in ra từ quá trình huấn luyện trở nên thiếu công bằng và không đủ độ tin cậy để công bố quốc tế.

Hệ thống **`Common_Evaluate`** giải quyết triệt để vấn đề này bằng cách đưa toàn bộ **8 mô hình** về **duy nhất một thước đo chuẩn mực quốc tế**: **MS COCO Evaluation Protocol (`pycocotools.cocoeval.COCOeval`)**:
* Tất cả 8 mô hình đều nhận diện trên cùng **1.481 ảnh** độc lập của tập Test chuẩn.
* Dự đoán của mọi mô hình đều được chuẩn hóa về định dạng COCO: `{"image_id", "category_id", "bbox": [xmin, ymin, w, h], "score"}` với tọa độ quy đổi về kích thước ảnh gốc.
* Đánh giá công bằng, khách quan 100% bằng đối tượng `COCOeval`, đảm bảo tính minh bạch trước Hội đồng Khoa học và các phản biện bài báo quốc tế (Reviewers).

---

## 2. Phân chia trách nhiệm trong nhóm nghiên cứu
* **Trần Anh Vũ (Chủ nhiệm/Lead)**:
  * Huấn luyện, tối ưu và đánh giá 2 mô hình cốt lõi:
    1. **`YOLOv26X`** (One-stage SOTA thế hệ mới nhất của Ultralytics).
    2. **`FasterRCNN_ResNet50`** (Two-stage Baseline kinh điển của Torchvision).
  * Thiết kế, chuẩn hóa toàn bộ pipeline đánh giá dùng chung (`Common_Evaluate`) và phụ trách hợp nhất báo cáo toàn đội (**Master Merge**).
* **Các thành viên đồng nghiệp trong nhóm**:
  * Huấn luyện 6 mô hình còn lại theo phân công:
    1. **`FasterRCNN_MobileNetv3Large`** (Torchvision)
    2. **`FCOS_ResNet50_FPN`** (Torchvision)
    3. **`YOLOv11`** (Ultralytics)
    4. **`RetinaNet`** (Torchvision)
    5. **`RFDETR_Medium`** (RF-DETR / RT-DETR Medium - Ultralytics)
    6. **`RTDETR_L`** (RT-DETR Large - Ultralytics)
  * Cấu hình của 6 mô hình này đã được để trống hoàn toàn trong [`models_config.json`](models_config.json). Đồng nghiệp chỉ cần điền đường dẫn file trọng số `weights`, số epoch kiên nhẫn `patience` và bật `"enabled": true` cho mô hình mình phụ trách.

---

## 3. Tập dữ liệu Roboflow COCO (Version 5)
Hệ thống kết nối và đồng bộ tự động với tập dữ liệu từ Roboflow:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="<ROBOFLOW_API_KEY>")  # Tự động nạp từ .env
project = rf.workspace("nckhcict2025").project("completed-project")
version = project.version(5)
dataset = version.download("coco")
```
* **Số lớp nguyên liệu hợp lệ**: **32 lớp** (Hệ thống tự động lọc bỏ lớp rác `19-8-BoSungAnhChoCacLopBiTh-v3zo` do Roboflow tự sinh ra với 0 annotations).
* **Số ảnh kiểm thử (Test Split)**: **1.481 ảnh** kích thước chuẩn $640 \times 640$.

---

## 4. Bảng 10 thông số chuẩn mực bắt buộc đo lường
Toàn bộ kết quả đều được xuất ra định dạng bảng 10 cột chuẩn khoa học:

| STT | Tên thông số | Định dạng | Phương pháp & Tiêu chuẩn đo lường |
| :---: | :--- | :---: | :--- |
| 1 | **Mô hình** | Text | Tên định danh của mô hình (Model Name) |
| 2 | **Precision** | Float (`.4f`) | Độ chính xác vận hành thực tế tại điểm $\text{conf} \ge 0.25, \text{IoU} \ge 0.50$: $\frac{TP}{TP + FP}$ |
| 3 | **Recall** | Float (`.4f`) | Độ nhạy vận hành thực tế tại điểm $\text{conf} \ge 0.25, \text{IoU} \ge 0.50$: $\frac{TP}{TP + FN}$ |
| 4 | **mAP@50** | Float (`.4f`) | Diện tích dưới đường cong Precision-Recall tại $\text{IoU} = 0.50$ chuẩn MS COCO (`stats[1]`) |
| 5 | **mAP@50-95** | Float (`.4f`) | COCO Primary Metric (Trung bình AP từ IoU 0.50 đến 0.95, bước 0.05 - `stats[0]`) |
| 6 | **Patience** | Integer | Số epoch chờ khi Early Stopping trong quá trình huấn luyện |
| 7 | **GFLOPs** | Float (`.2f`) | Độ phức tạp tính toán (tỷ phép tính) đo thực nghiệm bằng `thop` trên Dummy Input $1 \times 3 \times 640 \times 640$ (không dùng công thức giả định) |
| 8 | **Parameters** | Float (`.2f` M) | Tổng số tham số thực tế của mô hình (triệu tham số - Millions) đếm trực tiếp từ PyTorch module |
| 9 | **Latency** | Float (`.2f` ms) | Độ trễ toàn trình End-to-End (Đọc ảnh cv2 + Letterbox tiền xử lý + Suy luận PyTorch + Giải mã Boxes) trên GPU A100 với batch size = 1 |
| 10 | **FPS** | Float (`.2f`) | Tốc độ khung hình trên giây ($\text{FPS} = 1000 / \text{Latency}$) |

---

## 5. Cấu trúc thư mục tối giản & Bảo mật (Clean & Secure Architecture)
Dự án áp dụng nguyên lý tách biệt hoàn toàn giữa Cấu hình, Bí mật và Mã nguồn:

```text
Common_Evaluate/
├── common_evaluate.py       # File mã nguồn DUY NHẤT (Dataset, Adapters, Benchmark, COCOeval, Merge Engine)
├── models_config.json       # File cấu hình TẬP TRUNG (Modal Cloud, Roboflow, Benchmark, 32 lớp, 8 mô hình)
├── .env                     # File biến môi trường BẢO MẬT (Chứa Roboflow API Key riêng, không bị commit)
├── .env.example             # File mẫu biến môi trường (An toàn để chia sẻ cho đồng nghiệp)
├── .gitignore               # Tự động loại trừ .env, kết quả tạm và thư mục cache
└── README.md                # Tài liệu hướng dẫn chi tiết toàn bộ quy trình
```

### Cơ chế bảo mật API Key riêng tư:
Hệ thống tự động nạp Roboflow API Key theo thứ tự ưu tiên:
1. **Biến môi trường hệ thống**: `ROBOFLOW_API_KEY`
2. **File môi trường riêng tư**: `Common_Evaluate/.env` (`ROBOFLOW_API_KEY=...`)
3. **File config**: `models_config.json` (dự phòng, mặc định để trống `""`)

---

## 6. Hướng dẫn thiết lập & Chạy đánh giá cho Đồng nghiệp

### Bước 1: Thiết lập API Key cá nhân
1. Sao chép file mẫu [`.env.example`](.env.example) thành `.env`:
   * **Windows (PowerShell)**: `Copy-Item .env.example .env`
   * **Linux / macOS**: `cp .env.example .env`
2. Mở file `.env` và điền Roboflow API Key của bạn:
   ```bash
   ROBOFLOW_API_KEY=your_roboflow_api_key_here
   ```

### Bước 2: Cấu hình mô hình trong [`models_config.json`](models_config.json)
1. Đổi `"volume_name"` thành tên Volume Modal của bạn (nếu dùng tài khoản Modal riêng):
   ```json
   "modal_settings": {
     "volume_name": "volume-cua-dong-nghiep",
     "app_name": "app-common-coco-evaluate"
   }
   ```
2. Đổi `"enabled": false` cho 2 mô hình của Vũ (`YOLOv26X`, `FasterRCNN_ResNet50`).
3. Với các mô hình mình phụ trách (ví dụ `YOLOv11` và `RetinaNet`):
   - Điền đường dẫn file trọng số `weights` trên Modal: `"/data/runs/TenThuMucTrain/best.pt"`
   - Điền số `patience`: ví dụ `10`
   - Đặt `"enabled": true`

### Bước 3: Chạy đánh giá
* **Cách 1: Chạy trên Modal Cloud GPU A100 (Khuyến nghị)**:
  ```bash
  # Tự động đọc .env và models_config.json từ máy bạn, chỉ đánh giá mô hình có enabled: true
  modal run "Common_Evaluate/common_evaluate.py"::evaluate_my_models
  ```
  *Hoặc đánh giá đích danh 1 mô hình:*
  ```bash
  modal run "Common_Evaluate/common_evaluate.py"::evaluate_single --model-name YOLOv11
  ```
* **Cách 2: Chạy trên máy cục bộ (Local / Colab / Kaggle)**:
  ```bash
  # Chạy các mô hình có enabled: true
  python "Common_Evaluate/common_evaluate.py"

  # Hoặc chỉ định danh sách mô hình
  python "Common_Evaluate/common_evaluate.py" --models YOLOv11,RetinaNet
  ```

### Bước 4: Nộp file kết quả về cho nhóm
Sau khi chạy xong, trong thư mục `Common_Evaluate/Results/` sẽ xuất hiện các tệp:
- `result_<TenMoHinh>.json` (Ví dụ `result_YOLOv11.json`, `result_RetinaNet.json`): Tệp JSON chứa đầy đủ 10 chỉ số chuẩn hóa và mAP của từng lớp trong 32 lớp.
- `{TenMoHinh}_confusion_matrix_raw.png` & `{TenMoHinh}_confusion_matrix_normalized.png`: Ma trận nhầm lẫn 300 DPI.
- `{TenMoHinh}_test_per_class_metrics.csv`: Bảng chi tiết 32 lớp.

> 📢 **QUY TẮC CHIA SẺ**: Đồng nghiệp **chỉ cần gửi file `result_<TenMoHinh>.json`** (kích thước siêu nhẹ, chỉ vài KB) qua Zalo/Drive/Git cho Vũ, **không cần gửi file weights nặng hàng trăm MB!**

---

## 7. Quy trình Hợp nhất toàn đội (Master Merge dành cho Vũ)
Khi bạn (Vũ) nhận được các file `result_*.json` từ các thành viên trong nhóm:
1. Gom toàn bộ các file `result_*.json` vào một thư mục (ví dụ `team_results/`).
2. Chạy lệnh hợp nhất duy nhất:
   ```bash
   python "Common_Evaluate/common_evaluate.py" --merge-dir ./team_results
   ```
3. Hệ thống sẽ tự động tổng hợp toàn bộ các sản phẩm khoa học sẵn sàng đưa vào bài báo:
   * **`full_8_models_scientific_comparison.md`**: Bảng so sánh 10 chỉ số chuẩn của cả 8 mô hình.
   * **`full_8_models_comparison_table.csv`**: File CSV phục vụ chèn bảng vào LaTeX / Word / Excel.
   * **`full_8_models_coco_metrics_comparison.png`** (300 DPI): Biểu đồ so sánh 4 panel trực quan (mAP50, mAP50-95, GFLOPs, FPS).
   * **`all_classes_cross_model_map50_matrix.csv`**: Ma trận đối đầu chi tiết 32 lớp giữa cả 8 mô hình.

---

## 8. Danh mục 32 lớp nguyên liệu nấu ăn chuẩn hóa
Tập dữ liệu bao gồm 32 lớp nguyên liệu thực phẩm đặc trưng:

| STT | Tên lớp (Class Name) | STT | Tên lớp (Class Name) | STT | Tên lớp (Class Name) | STT | Tên lớp (Class Name) |
| :---: | :--- | :---: | :--- | :---: | :--- | :---: | :--- |
| 1 | `beef` | 9 | `chayote` | 17 | `eggplant` | 25 | `pumpkin` |
| 2 | `bellpepper` | 10 | `chicken` | 18 | `garlic` | 26 | `radish` |
| 3 | `bittergourd` | 11 | `chickenegg` | 19 | `ginger` | 27 | `scallion` |
| 4 | `bottlegourd` | 12 | `chickenleg` | 20 | `jicama` | 28 | `shrimp` |
| 5 | `broccoli` | 13 | `chickenwin` | 21 | `okra` | 29 | `spongegourd` |
| 6 | `cabbage` | 14 | `corn` | 22 | `onion` | 30 | `sweetpotato` |
| 7 | `carrot` | 15 | `cucumber` | 23 | `pork` | 31 | `tofu` |
| 8 | `cauliflower` | 16 | `duckegg` | 24 | `potato` | 32 | `tomato` |

*(Lớp thứ 33 `19-8-BoSungAnhChoCacLopBiTh-v3zo` có 0 annotations được hệ thống tự động loại bỏ để đảm bảo mAP được tính toán chuẩn xác trên đúng 32 nguyên liệu thực tế).*

---

## 9. Đảm bảo Tính công bằng & Minh bạch Khoa học (Scientific Fairness & Rigor)
Khi so sánh các mô hình thuộc các framework và họ kiến trúc khác nhau (Ultralytics, Torchvision, RF-DETR), sự công bằng và tính khoa học được đảm bảo tuyệt đối qua 6 nguyên tắc cốt lõi:

1. **Đồng nhất tiền xử lý bằng Letterbox (Aspect-Ratio Preserved)**:
   - Thay vì dùng `cv2.resize` bóp méo khung hình, cả Ultralytics và Torchvision đều được áp dụng thuật toán `letterbox` chuẩn với padding màu xám `(114, 114, 114)` đưa về kích thước $640 \times 640$.
   - Tọa độ Bounding Box được ánh xạ ngược chuẩn xác: $x_{\text{orig}} = (x_{\text{box}} - dw) / ratio$ và $y_{\text{orig}} = (y_{\text{box}} - dh) / ratio$. Không có bất kỳ mô hình nào bị thiên vị do biến dạng hình học.

2. **Trọng tài độc lập duy nhất**:
   - Không dùng bất kỳ hàm tính mAP hay evaluator nội bộ của riêng Ultralytics hay Torchvision.
   - Toàn bộ kết quả bounding box của 8 mô hình được chuẩn hóa và đánh giá trực tiếp qua đối tượng chính quy quốc tế `pycocotools.cocoeval.COCOeval`.

3. **Phân định rạch ròi giữa Operational Precision/Recall và COCO mAP**:
   - Khắc phục triệt để lỗi đánh đồng "Precision" với $AP_{50}$.
   - **`Precision`** và **`Recall`**: Đo đạc độ chính xác và độ nhạy vận hành thực tế tại điểm $\text{conf} \ge 0.25, \text{IoU} \ge 0.50$ qua greedy box matching, đồng bộ 100% với Confusion Matrix.
   - **`mAP@50`** và **`mAP@50-95`**: Tính toán theo diện tích tích phân toàn dải chuẩn MS COCO (`stats[1]` và `stats[0]`).

4. **Tuyệt đối KHÔNG sử dụng số liệu hardcode / cache giả lập**:
   - Hệ thống loại bỏ hoàn toàn mọi khối fallback giả lập số liệu. Nếu thiếu file weights thực tế (`best.pt`, `best.pth`), mô hình sẽ nhận trạng thái `"Chưa có file weights"` với các chỉ số bằng `0.0`.
   - Mọi kết quả công bố bắt buộc phải được suy luận trực tiếp từ file trọng số thật trên 1.481 ảnh test.

5. **Bộ điều hợp hoàn chỉnh cho RF-DETR (`RFDETRNativeAdapter`)**:
   - Tích hợp pipeline suy luận trực tiếp từ package `rfdetr` chính gốc (`RFDETRMedium` + PIL Image format), tự động giải mã cấu trúc `supervision.Detections` (`xyxy`, `confidence`, `class_id`) và quy đổi chuẩn xác về COCO category IDs của 32 lớp nguyên liệu.
   - Cơ chế nạp checkpoint linh hoạt: Ưu tiên nạp native qua `rfdetr.RFDETRMedium`, có cảnh báo minh bạch nếu môi trường thiếu thư viện và bắt buộc fallback sang Ultralytics RT-DETR.

6. **Đồng nhất nguồn cấu hình ngưỡng vận hành (`eval_conf_threshold`)**:
   - Khắc phục nguy cơ lệch ngưỡng: Thông số `eval_conf_threshold: 0.25` được khai báo tập trung trong `models_config.json` và dùng chung đồng thời cho cả thuật toán tính toán Operational Precision & Recall lẫn ma trận nhầm lẫn Confusion Matrix.
   - Lưu trữ song song chỉ số `coco_ar100` (Average Recall chuẩn COCO tại maxDets=100) trong file kết quả JSON.

7. **Đo đạc hiệu năng tính toán thực nghiệm chuẩn xác**:
   - **Latency**: Đo lường End-to-End trọn vẹn (Đọc ảnh `cv2.imread` + Letterbox + Suy luận PyTorch GPU + Giải mã Bounding Boxes) trên GPU NVIDIA A100 với Batch Size = 1.
   - **GFLOPs**: Đo thực nghiệm bằng `thop`. Nếu cấu trúc Transformer động không được thư viện hỗ trợ, hệ thống ghi nhận `0.0` kèm cảnh báo rõ ràng, tuyệt đối không dùng công thức nhân hệ số áng chừng không có căn cứ khoa học.
