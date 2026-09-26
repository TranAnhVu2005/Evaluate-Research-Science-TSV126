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
* Đánh giá thống nhất bằng đối tượng `COCOeval`, với protocol và provenance có thể kiểm chứng khi tổng hợp kết quả.

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
  * Pipeline hỗ trợ đủ 8 kiến trúc. Mỗi lượt chạy mặc định chỉ đánh giá các mô hình có `"enabled": true`; người dùng tự chọn model cần chạy và chỉ phải cung cấp checkpoint cho các model đã bật.

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
| 2 | **Precision** | Float (`.4f`) | Micro precision từ matching của `COCOeval` tại `conf=0.55`, `IoU=0.50` (TP / (TP + FP)) |
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
project-root/
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
2. **File môi trường riêng tư**: `.env` (`ROBOFLOW_API_KEY=...`)
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
2. Chọn một hoặc nhiều mô hình cần đánh giá bằng trường `enabled`.
3. Với từng mô hình đã bật:
   - Điền đường dẫn checkpoint trên Modal vào `weights`, ví dụ `"/data/runs/TenThuMucTrain/best.pt"`.
   - Có thể điền đường dẫn dự phòng trên máy local vào `local_weights`.
   - Điền đúng `patience` đã dùng khi train. Không dùng pretrained checkpoint hoặc random weights thay cho checkpoint sau train.
   - Đặt `"enabled": true`; các model chưa muốn chạy giữ `"enabled": false`.

Khi chạy local, cũng có thể ghi đè đường dẫn bằng biến môi trường `MODEL_WEIGHTS_<MODEL_KEY>`, ví dụ `MODEL_WEIGHTS_YOLOV11`. Pipeline chỉ kiểm tra checkpoint của các model được chọn trong lượt chạy hiện tại.

Các trường cấu hình quan trọng cho từng framework:

- `family`: một trong `ultralytics`, `torchvision`, `rfdetr`.
- `model_type`: `yolo`, `rtdetr`, hoặc đúng tên constructor Torchvision đã huấn luyện.
- `model_class`: class RF-DETR chính thức, ví dụ `RFDETRMedium`.
- `imgsz`: bắt buộc là `640` cho cả 8 mô hình. RF-DETR Medium được nạp với resolution override 640; checkpoint phải tương thích với API RF-DETR đang được pin.
- `label_offset`: là `1` cho các checkpoint Torchvision hiện tại và `0` cho RF-DETR/Ultralytics. Với FCOS/RetinaNet được train bằng nhãn một-based, output nhỏ hơn `label_offset` là kênh dự phòng và bị loại trước giới hạn 100 detection.
- `checkpoint_classes`: khai báo khi thứ tự/số lớp khác danh sách `classes` chung; với checkpoint Torchvision không chứa metadata `class_names`, trường này là bắt buộc để hệ thống không phải đoán thứ tự lớp.
- `class_name_map` và `ignored_checkpoint_classes`: ánh xạ bí danh hoặc bỏ lớp rác một cách tường minh. Pipeline không tự đoán class ID.

### Bước 3: Chạy đánh giá
* **Cách 1: Chạy trên Modal Cloud GPU A100 (Khuyến nghị)**:
  ```bash
  # Tự động đọc .env và models_config.json, sau đó chạy model có enabled=true
  modal run common_evaluate.py::evaluate_my_models
  ```
  *Hoặc đánh giá đích danh 1 mô hình:*
  ```bash
  modal run common_evaluate.py::evaluate_single --model-name YOLOv11
  ```
* **Cách 2: Chạy trên máy cục bộ (Local / Colab / Kaggle)**:
  ```bash
  # Chạy các model có enabled=true
  python common_evaluate.py

  # Hoặc chỉ định danh sách mô hình
  python common_evaluate.py --models YOLOv11,RetinaNet
  ```

### Bước 4: Nộp file kết quả về cho nhóm
Sau khi chạy xong, trong thư mục `Results/` sẽ xuất hiện các tệp:
- `result_<TenMoHinh>.json` (Ví dụ `result_YOLOv11.json`, `result_RetinaNet.json`): Tệp JSON chứa đầy đủ 10 chỉ số chuẩn hóa và mAP của từng lớp trong 32 lớp.
- `{TenMoHinh}_test_per_class_metrics.csv`: Bảng chi tiết 32 lớp.

> 📢 **QUY TẮC CHIA SẺ**: Đồng nghiệp **chỉ cần gửi file `result_<TenMoHinh>.json`** (kích thước siêu nhẹ, chỉ vài KB) qua Zalo/Drive/Git cho Vũ, **không cần gửi file weights nặng hàng trăm MB!**

---

## 7. Quy trình Hợp nhất toàn đội (Master Merge dành cho Vũ)
Khi bạn (Vũ) nhận được các file `result_*.json` từ các thành viên trong nhóm:
1. Gom toàn bộ các file `result_*.json` vào một thư mục (ví dụ `team_results/`).
2. Chạy lệnh hợp nhất duy nhất:
   ```bash
   python common_evaluate.py --merge-dir ./team_results
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
Khi so sánh các mô hình thuộc framework Ultralytics (YOLOv26X, YOLOv11, RT-DETR) với Torchvision (Faster R-CNN, FCOS, RetinaNet), tính nhất quán của benchmark được đảm bảo nhờ:
1. **Trọng tài độc lập duy nhất**: Không sử dụng hàm tính mAP nội bộ của từng thư viện mà toàn bộ dự đoán được đưa vào đối tượng chuẩn quốc tế `pycocotools.cocoeval.COCOeval`.
2. **Quy đổi kích thước chuẩn xác**: Tọa độ bounding box dự đoán của mọi mô hình đều được ánh xạ ngược về kích thước pixel gốc của ảnh test $(orig\_w, orig\_h)$.
3. **Không cắt dự đoán theo confidence trước COCOeval**: Cả 3 họ adapter đều thiết lập `conf_threshold = 0.0`; mỗi model xuất tối đa 100 dự đoán theo điểm số và `COCOeval` thực hiện đánh giá/xếp hạng. Cách này tránh thiên lệch do các model hiệu chỉnh confidence khác nhau. Precision/Recall/F1 được suy ra trực tiếp từ `COCOeval.evalImgs` tại `operating_conf_threshold`, có áp dụng cùng quy tắc match, ignore và crowd của COCO.
4. **Môi trường phần cứng chuẩn hóa**: Đo đạc độ trễ trên cùng GPU NVIDIA A100 (Modal Cloud) với `batch_size = 1`, cơ chế đồng bộ `torch.cuda.synchronize()` và loại bỏ độ trễ khởi động kernel bằng bước warm-up.
5. **Khóa giao thức đầu vào**: Cả 8 model bắt buộc inference ở `imgsz = 640`, `batch_size = 1`; ảnh test phải đúng 640 × 640 và merge kiểm tra lại các giá trị này trong cả protocol lẫn provenance.

NMS và hậu xử lý tạo prediction vẫn thuộc implementation chuẩn của từng kiến trúc; `COCOeval` thống nhất cách chấm các prediction đó, không thay thế hậu xử lý nội tại của model.

Mỗi `result_<model>.json` chứa protocol `common-coco-v7-640-b1`, `protocol_id`, SHA-256 của annotation test và checkpoint, phiên bản thư viện, kích thước inference 640, batch size 1, cấu hình ngưỡng, hậu xử lý, NMS và ánh xạ lớp. Lệnh merge chấp nhận kết quả của các model được đánh giá ở những lượt khác nhau nhưng từ chối file sai protocol, sai kích thước, sai batch, sai cấu hình inference hoặc được tạo từ dataset/evaluator khác; model chưa có kết quả được ghi rõ là đang chờ và không tham gia biểu đồ so sánh. Không được chép metric từ log train, cache evaluator cũ hoặc bộ so khớp riêng vào bảng chung; toàn bộ metric chất lượng đều xuất phát từ `pycocotools.COCOeval`.
