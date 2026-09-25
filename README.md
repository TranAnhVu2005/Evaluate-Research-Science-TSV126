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

Ba adapter Ultralytics, Torchvision và RF-DETR vẫn cần thiết, nhưng chúng không phải evaluator. Chúng chỉ tải đúng kiến trúc checkpoint, chạy inference và chuẩn hóa đầu ra khác nhau của từng thư viện thành bốn trường COCO chung: `bbox`, `score`, `category_id`, `image_id`. Sau bước này, toàn bộ metric chỉ được tính bởi `COCOeval`.

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
    5. **`RFDETR_Medium`** (RF-DETR chính thức của Roboflow; không phải Ultralytics RT-DETR)
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

## 4. Các thông số chuẩn mực bắt buộc đo lường
Toàn bộ kết quả được xuất theo cùng một schema; bảng dưới đây là các trường cốt lõi:

| STT | Tên thông số | Định dạng | Phương pháp & Tiêu chuẩn đo lường |
| :---: | :--- | :---: | :--- |
| 1 | **Mô hình** | Text | Tên định danh của mô hình (Model Name) |
| 2 | **Precision** | Float (`.4f`) | Micro precision từ matching của `COCOeval` tại `conf=0.25`, `IoU=0.50` (TP / (TP + FP)) |
| 3 | **Recall** | Float (`.4f`) | Micro recall từ cùng matching của `COCOeval` (TP / (TP + FN)); COCO AR@100 được xuất ở cột riêng |
| 4 | **mAP@50** | Float (`.4f`) | Mean Average Precision tại $\text{IoU} = 0.50$ (`stats[1]`) |
| 5 | **mAP@50-95** | Float (`.4f`) | COCO Primary Challenge Metric (Trung bình AP từ IoU 0.50 đến 0.95, bước 0.05) |
| 6 | **Patience** | Integer | Số epoch chờ khi Early Stopping trong quá trình huấn luyện |
| 7 | **GFLOPs** | Float (`.2f`) | Độ phức tạp tính toán (tỷ phép tính) đo bằng `thop` trên Dummy Input $1 \times 3 \times 640 \times 640$ |
| 8 | **Parameters** | Float (`.2f` M) | Tổng số tham số của mô hình (triệu tham số - Millions) |
| 9 | **Latency** | Float (`.2f` ms) | Thời gian xử lý trung bình 1 ảnh trên GPU A100 (đo trên 1.481 ảnh với batch size = 1) |
| 10 | **FPS** | Float (`.2f`) | Tốc độ khung hình trên giây ($\text{FPS} = 1000 / \text{Latency}$) |

File kết quả còn xuất `F1`, `mAP@75`, `AP_small/medium/large`, `AR@1/10/100` và metric từng lớp.

Nếu `thop` không profile được một kiến trúc detector, `GFLOPs` được ghi là `null`/`-`; hệ thống không nội suy GFLOPs từ số parameters. Latency không bao gồm thời gian đọc ảnh từ ổ đĩa.

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

Các trường cấu hình quan trọng cho từng framework:

- `family`: một trong `ultralytics`, `torchvision`, `rfdetr`.
- `model_type`: `yolo`, `rtdetr`, hoặc đúng tên constructor Torchvision đã huấn luyện.
- `model_class`: class RF-DETR chính thức, ví dụ `RFDETRMedium`.
- `imgsz`: kích thước inference đúng với checkpoint; RF-DETR Medium mặc định là 576.
- `label_offset`: thường là `1` cho Torchvision và `0` cho RF-DETR/Ultralytics.
- `checkpoint_classes`: chỉ cần khai báo khi thứ tự/số lớp trong checkpoint khác danh sách `classes` chung.
- `class_name_map` và `ignored_checkpoint_classes`: ánh xạ bí danh hoặc bỏ lớp rác một cách tường minh. Pipeline không tự đoán class ID.

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
- `{TenMoHinh}_confusion_matrix.csv` và `{TenMoHinh}_confusion_matrix_normalized.png`: Ma trận nhầm lẫn có thêm hàng/cột `background` để tính cả FP và FN.
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
   * **`summary_coco_comparison.md`** và **`summary_coco_comparison.csv`**: Bảng tổng hợp các metric của cả 8 mô hình.
   * **`comparison_map_chart.png`**: Biểu đồ mAP50 và mAP50-95.
   * **`comparison_tradeoff_chart.png`**: Biểu đồ trade-off accuracy/speed.
   * **`all_models_per_class_comparison.csv`**: Bảng mAP theo 32 lớp giữa các mô hình.

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

## 9. Đảm bảo Tính công bằng Khoa học (Scientific Fairness)
Khi so sánh các mô hình thuộc framework Ultralytics (YOLOv26X, YOLOv11, RT-DETR) với Torchvision (Faster R-CNN, FCOS, RetinaNet), sự công bằng tuyệt đối được đảm bảo nhờ:
1. **Trọng tài độc lập duy nhất**: Không sử dụng hàm tính mAP nội bộ của từng thư viện mà toàn bộ dự đoán được đưa vào đối tượng chuẩn quốc tế `pycocotools.cocoeval.COCOeval`.
2. **Quy đổi kích thước chuẩn xác**: Tọa độ bounding box dự đoán của mọi mô hình đều được ánh xạ ngược về kích thước pixel gốc của ảnh test $(orig\_w, orig\_h)$.
3. **Đồng nhất ngưỡng Confidence**: Cả 3 họ adapter đều thiết lập `conf_threshold = 0.001` để xuất dải điểm số cho `COCOeval`; Precision/Recall/F1 được suy ra trực tiếp từ `COCOeval.evalImgs` tại `operating_conf_threshold`, có áp dụng cùng quy tắc match, ignore và crowd của COCO.
4. **Môi trường phần cứng chuẩn hóa**: Đo đạc độ trễ trên cùng GPU NVIDIA A100 (Modal Cloud) với `batch_size = 1`, cơ chế đồng bộ `torch.cuda.synchronize()` và loại bỏ độ trễ khởi động kernel bằng bước warm-up.

Mỗi `result_<model>.json` chứa protocol `common-coco-v3`, `protocol_id`, SHA-256 của annotation test và checkpoint, phiên bản thư viện, cấu hình ngưỡng và danh sách lớp. Lệnh merge sẽ từ chối file thiếu protocol hoặc file được tạo từ dataset/evaluator khác; vì vậy không được chép metric từ log train hay cache evaluator cũ vào bảng chung. Confusion matrix chỉ dùng phân tích trực quan, không phải nguồn của Precision/Recall/F1 trong bảng kết quả.
