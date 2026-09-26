"""Config-driven COCOeval benchmark for eight object-detection models."""

import os
import time
import json
import argparse
import gzip
import gc
import hashlib
import importlib.metadata
import platform
import uuid
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import modal

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(CURRENT_DIR, "models_config.json")
PROTOCOL_NAME = "common-coco-v5-640-b1"
REQUIRED_MODEL_KEYS = {
    "YOLOv26X",
    "FasterRCNN_ResNet50",
    "FasterRCNN_MobileNetv3Large",
    "FCOS_ResNet50_FPN",
    "YOLOv11",
    "RetinaNet",
    "RFDETR_Medium",
    "RTDETR_L",
}
REQUIRED_MODEL_ROUTES = {
    "YOLOv26X": ("ultralytics", {"yolo"}),
    "FasterRCNN_ResNet50": (
        "torchvision",
        {"fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2"},
    ),
    "FasterRCNN_MobileNetv3Large": (
        "torchvision",
        {"fasterrcnn_mobilenet_v3_large_fpn", "fasterrcnn_mobilenet_v3_large_320_fpn"},
    ),
    "FCOS_ResNet50_FPN": ("torchvision", {"fcos_resnet50_fpn"}),
    "YOLOv11": ("ultralytics", {"yolo"}),
    "RetinaNet": (
        "torchvision",
        {"retinanet_resnet50_fpn", "retinanet_resnet50_fpn_v2"},
    ),
    "RFDETR_Medium": ("rfdetr", {"rfdetr_medium"}),
    "RTDETR_L": ("ultralytics", {"rtdetr"}),
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Nạp toàn bộ cấu hình từ file models_config.json.
    Cho phép tách biệt 100% giữa code thực thi và các thông số cài đặt.
    """
    path = config_path or CONFIG_FILE
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Không thể đọc file cấu hình tại {path}: {e}")
    return {}


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Create a content fingerprint for dataset and checkpoint provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source_file(path: str) -> str:
    """Hash source consistently across Git checkouts using LF or CRLF line endings."""
    with open(path, "rb") as source:
        normalized = source.read().replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def installed_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def describe_runtime_device(device: str) -> str:
    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            index = torch.device(device).index or 0
            return torch.cuda.get_device_name(index)
    except Exception:
        pass
    return platform.processor() or "cpu"


def default_compute_device() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def compute_protocol_id(protocol: Dict[str, Any]) -> str:
    """Compute the signed protocol payload, excluding its own identifier field."""
    payload = {key: value for key, value in protocol.items() if key != "protocol_id"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_payload_sha256(payload: Dict[str, Any]) -> str:
    """Hash a JSON-compatible provenance payload deterministically."""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Render a simple GFM table without pandas' optional tabulate dependency."""
    def escape_cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    headers = [escape_cell(column) for column in dataframe.columns]
    rows = [
        [escape_cell(value) for value in row]
        for row in dataframe.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_requested_inference_config(
    model_cfg: Dict[str, Any],
    bench_cfg: Dict[str, Any],
    evaluated_classes: List[str],
) -> Dict[str, Any]:
    """Return the complete model-specific inference request recorded in every result."""
    return {
        "family": model_cfg.get("family"),
        "model_type": model_cfg.get("model_type"),
        "model_class": model_cfg.get("model_class"),
        "imgsz": int(model_cfg.get("imgsz", bench_cfg["imgsz"])),
        "batch_size": int(bench_cfg["batch_size"]),
        "prediction_conf_threshold": float(bench_cfg["conf_threshold"]),
        "max_detections": int(bench_cfg["max_detections"]),
        "label_offset": (
            int(model_cfg["label_offset"]) if "label_offset" in model_cfg else None
        ),
        "nms_iou_threshold": model_cfg.get("nms_iou_threshold"),
        "constructor_kwargs": dict(model_cfg.get("constructor_kwargs", {})),
        "checkpoint_classes": list(
            model_cfg.get("checkpoint_classes") or evaluated_classes
        ),
        "class_name_map": dict(model_cfg.get("class_name_map", {})),
        "ignored_checkpoint_classes": list(
            model_cfg.get("ignored_checkpoint_classes", [])
        ),
    }


def build_resolved_inference_config(
    adapter: Any,
    requested_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Record the adapter's effective settings and resolved class/category mapping."""
    checkpoint_classes = getattr(adapter, "checkpoint_classes", {})
    if isinstance(checkpoint_classes, dict):
        indexed_classes = checkpoint_classes
    else:
        indexed_classes = {
            index: name for index, name in enumerate(checkpoint_classes or [])
        }
    resolved = {
        "requested": requested_config,
        "effective": {
            "imgsz": int(getattr(adapter, "imgsz")),
            "prediction_conf_threshold": float(getattr(adapter, "conf_thresh")),
            "max_detections": int(getattr(adapter, "max_detections")),
            "backend_max_detections": int(
                getattr(adapter, "backend_max_detections", getattr(adapter, "max_detections"))
            ),
            "nms_iou_threshold": getattr(adapter, "nms_iou_threshold", None),
            "constructor_kwargs": dict(
                getattr(adapter, "effective_constructor_kwargs", {})
            ),
            "background_class_id": getattr(adapter, "background_class_id", None),
        },
        "checkpoint_classes": [
            {"index": int(index), "name": str(name)}
            for index, name in sorted(indexed_classes.items())
        ],
        "class_idx_to_category_id": {
            str(int(index)): int(category_id)
            for index, category_id in sorted(
                getattr(adapter, "class_idx_to_cat_id", {}).items()
            )
        },
        "ignored_class_indices": sorted(
            int(index) for index in getattr(adapter, "ignored_class_indices", set())
        ),
    }
    return resolved


def build_protocol_metadata(
    bench_cfg: Dict[str, Any],
    annotation_sha256: str,
    image_ids: List[int],
    valid_categories: List[Dict[str, Any]],
    runtime_device: str,
) -> Dict[str, Any]:
    """Describe the evaluator completely; only equal protocol IDs may be merged."""
    protocol = {
        "name": bench_cfg.get("protocol_name", PROTOCOL_NAME),
        "evaluator": "pycocotools.cocoeval.COCOeval",
        "evaluator_source_sha256": sha256_source_file(os.path.abspath(__file__)),
        "iou_type": "bbox",
        "annotation_sha256": annotation_sha256,
        "image_ids_sha256": hashlib.sha256(
            json.dumps(sorted(int(i) for i in image_ids), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "num_test_images": len(image_ids),
        "categories": [{"id": int(c["id"]), "name": c["name"]} for c in valid_categories],
        "settings": {
            "inference_imgsz": int(bench_cfg.get("imgsz", 640)),
            "inference_batch_size": int(bench_cfg.get("batch_size", 1)),
            "prediction_conf_threshold": float(bench_cfg.get("conf_threshold", 0.0)),
            "operating_conf_threshold": float(bench_cfg.get("operating_conf_threshold", 0.25)),
            "operating_iou_threshold": float(bench_cfg.get("iou_threshold", 0.50)),
            "operating_metric_source": "pycocotools.COCOeval.evalImgs",
            "operating_averaging": "micro_over_all_categories",
            "max_detections": int(bench_cfg.get("max_detections", 100)),
            "latency_batch_size": int(bench_cfg.get("batch_size", 1)),
            "latency_warmup_runs": int(bench_cfg.get("warmup_runs", 10)),
            "runtime_device": runtime_device,
            "coco_iou_thresholds": [round(float(v), 2) for v in np.arange(0.50, 0.96, 0.05)],
            "coco_area_ranges": ["all", "small", "medium", "large"],
        },
        "software": {
            "python": platform.python_version(),
            "pycocotools": installed_version("pycocotools"),
            "torch": installed_version("torch"),
            "torchvision": installed_version("torchvision"),
            "ultralytics": installed_version("ultralytics"),
            "rfdetr": installed_version("rfdetr"),
        },
    }
    protocol["protocol_id"] = compute_protocol_id(protocol)
    return protocol


def validate_configuration(config_data: Dict[str, Any]) -> None:
    classes = config_data.get("classes", [])
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("config.classes must be a non-empty list of unique class names")
    if len(classes) != 32:
        raise ValueError(f"The common COCO protocol requires exactly 32 classes, found {len(classes)}")

    bench_cfg = config_data.get("benchmark_settings", {})
    required_imgsz = int(bench_cfg.get("imgsz", 0))
    required_batch_size = int(bench_cfg.get("batch_size", 0))
    expected_models = int(bench_cfg.get("expected_num_models", 8))
    if bench_cfg.get("protocol_name") != PROTOCOL_NAME:
        raise ValueError(f"benchmark_settings.protocol_name must be {PROTOCOL_NAME}")
    if required_imgsz != 640:
        raise ValueError("The common COCO protocol requires benchmark_settings.imgsz = 640")
    if required_batch_size != 1:
        raise ValueError("The common COCO protocol requires benchmark_settings.batch_size = 1")
    if expected_models != len(REQUIRED_MODEL_KEYS):
        raise ValueError("benchmark_settings.expected_num_models must be 8")
    locked_settings = {
        "conf_threshold": 0.0,
        "operating_conf_threshold": 0.25,
        "iou_threshold": 0.50,
        "max_detections": 100,
        "expected_num_classes": 32,
        "expected_test_images": 1481,
    }
    for setting_name, expected_value in locked_settings.items():
        actual_value = bench_cfg.get(setting_name)
        if actual_value is None or not np.isclose(float(actual_value), float(expected_value)):
            raise ValueError(
                f"benchmark_settings.{setting_name} must be {expected_value}, got {actual_value}"
            )

    supported = {
        "ultralytics": {"yolo", "rtdetr"},
        "torchvision": {
            "fasterrcnn_resnet50_fpn",
            "fasterrcnn_resnet50_fpn_v2",
            "fasterrcnn_mobilenet_v3_large_fpn",
            "fasterrcnn_mobilenet_v3_large_320_fpn",
            "fcos_resnet50_fpn",
            "retinanet_resnet50_fpn",
            "retinanet_resnet50_fpn_v2",
        },
        "rfdetr": {"rfdetr_medium"},
    }
    models = config_data.get("models", {})
    if not models:
        raise ValueError("config.models is empty")
    if set(models) != REQUIRED_MODEL_KEYS:
        missing = sorted(REQUIRED_MODEL_KEYS - set(models))
        unexpected = sorted(set(models) - REQUIRED_MODEL_KEYS)
        raise ValueError(
            f"The protocol requires the canonical 8 model keys; missing={missing}, "
            f"unexpected={unexpected}"
        )
    for model_key, model_cfg in models.items():
        family = model_cfg.get("family")
        model_type = model_cfg.get("model_type")
        if family not in supported or model_type not in supported[family]:
            raise ValueError(
                f"Unsupported configuration for {model_key}: family={family}, model_type={model_type}"
            )
        required_family, allowed_model_types = REQUIRED_MODEL_ROUTES[model_key]
        if family != required_family or model_type not in allowed_model_types:
            raise ValueError(
                f"{model_key} must use family={required_family} and model_type in "
                f"{sorted(allowed_model_types)}, got family={family}, model_type={model_type}"
            )
        checkpoint_classes = model_cfg.get("checkpoint_classes")
        if checkpoint_classes and len(checkpoint_classes) != len(set(checkpoint_classes)):
            raise ValueError(f"{model_key}.checkpoint_classes contains duplicates")
        if family == "rfdetr" and not model_cfg.get("model_class"):
            raise ValueError(f"{model_key}.model_class is required for RF-DETR")
        if family == "torchvision" and int(model_cfg.get("label_offset", -1)) != 1:
            raise ValueError(f"{model_key}.label_offset must be 1 for Torchvision detectors")
        if family == "rfdetr" and int(model_cfg.get("label_offset", -1)) != 0:
            raise ValueError(f"{model_key}.label_offset must be 0 for RF-DETR")
        constructor_kwargs = set(model_cfg.get("constructor_kwargs", {}))
        protocol_controlled_kwargs = {
            "min_size",
            "max_size",
            "box_score_thresh",
            "score_thresh",
            "box_detections_per_img",
            "detections_per_img",
        }
        forbidden_kwargs = sorted(constructor_kwargs & protocol_controlled_kwargs)
        if forbidden_kwargs:
            raise ValueError(
                f"{model_key}.constructor_kwargs cannot override protocol-controlled settings: "
                f"{forbidden_kwargs}"
            )
        if not isinstance(model_cfg.get("enabled"), bool):
            raise ValueError(f"{model_key}.enabled must be true or false")
        model_imgsz = int(model_cfg.get("imgsz", required_imgsz))
        if model_imgsz != required_imgsz:
            raise ValueError(
                f"{model_key}.imgsz={model_imgsz} violates the common inference size "
                f"of {required_imgsz}"
            )


def resolve_checkpoint_path(model_key: str, model_cfg: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    """Resolve a real checkpoint without ever falling back to pretrained/random weights."""
    env_key = "MODEL_WEIGHTS_" + "".join(
        char if char.isalnum() else "_" for char in model_key.upper()
    )
    candidates = []
    env_path = os.environ.get(env_key, "").strip()
    if env_path:
        candidates.append(os.path.abspath(os.path.expanduser(env_path)))

    cloud_path = str(model_cfg.get("weights", "")).strip()
    if cloud_path:
        candidates.append(os.path.abspath(os.path.expanduser(cloud_path)))

    local_spec = str(model_cfg.get("local_weights", "")).strip()
    if local_spec:
        local_path = (
            local_spec if os.path.isabs(local_spec) else os.path.join(CURRENT_DIR, local_spec)
        )
        candidates.append(os.path.abspath(os.path.expanduser(local_path)))

    unique_candidates = list(dict.fromkeys(candidates))
    for candidate in unique_candidates:
        if os.path.isfile(candidate):
            return candidate, unique_candidates
    return None, unique_candidates


def get_roboflow_api_key(config_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Trích xuất Roboflow API Key theo thứ tự ưu tiên bảo mật:
    1. Biến môi trường hệ thống: ROBOFLOW_API_KEY
    2. Tệp môi trường bảo mật: .env
    3. Tệp cấu hình: models_config.json (nếu có điền)
    """
    # 1. Kiểm tra biến môi trường hệ thống
    env_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if env_key:
        return env_key

    # 2. Kiểm tra file .env
    candidates = [
        os.path.join(CURRENT_DIR, ".env"),
        os.path.join(os.path.dirname(CURRENT_DIR), ".env"),
        ".env"
    ]
    for env_file in candidates:
        if os.path.exists(env_file):
            try:
                with open(env_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() == "ROBOFLOW_API_KEY":
                                val = v.strip().strip('"').strip("'")
                                if val and val != "your_roboflow_api_key_here":
                                    return val
            except Exception:
                pass

    # 3. Fallback từ config_data (nếu có)
    if config_data:
        cfg_key = config_data.get("roboflow", {}).get("api_key", "").strip()
        if cfg_key:
            return cfg_key

    return ""


# Nạp cấu hình toàn cục
GLOBAL_CONFIG = load_config()
MODAL_SETTINGS = GLOBAL_CONFIG.get("modal_settings", {})
MODAL_VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", MODAL_SETTINGS.get("volume_name", "volume-10-09-2026-yolo26x-640"))
APP_NAME = MODAL_SETTINGS.get("app_name", "app-common-coco-evaluate")
MODAL_GPU = MODAL_SETTINGS.get("gpu", "A100")
MODAL_TIMEOUT = int(MODAL_SETTINGS.get("timeout_seconds", 7200))

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "ultralytics==8.4.161",
        "rfdetr==1.11.0",
        "roboflow>=1.1.33",
        "opencv-python-headless>=4.10.0.84",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pyyaml>=6.0",
        "pillow>=10.3.0",
        "thop>=0.1.1",
        "tabulate>=0.9.0",
        "pycocotools==2.0.8",
        "tqdm>=4.66.0"
    )
)

# Chỉ đóng gói checkpoint cục bộ khi thư mục tồn tại. Nhờ vậy cùng source có thể
# chạy với Modal Volume/env path hoặc với ./models mà không làm bước build bị lỗi.
LOCAL_MODELS_DIR = os.path.join(CURRENT_DIR, "models")
if os.path.isdir(LOCAL_MODELS_DIR):
    image = image.add_local_dir(LOCAL_MODELS_DIR, remote_path="/models")


# ==============================================================================
# PHẦN 2: QUẢN LÝ DỮ LIỆU CHUẨN COCO TỪ ROBOFLOW
# ==============================================================================
class RoboflowCOCODatasetManager:
    """Quản lý việc tải, lưu trữ đệm (cache), làm sạch 32 lớp và nạp tập test."""

    def __init__(
        self,
        api_key: str = "",
        dataset_root: Optional[str] = None,
        workspace: str = "nckhcict2025",
        project: str = "completed-project",
        version: int = 5,
        dummy_cat_name: str = "19-8-BoSungAnhChoCacLopBiTh-v3zo",
    ):
        self.api_key = api_key if api_key else get_roboflow_api_key()
        self.dataset_root = dataset_root
        self.workspace = workspace
        self.project = project
        self.version = version
        self.dummy_cat_name = dummy_cat_name
        self.coco_gt = None
        self.valid_categories = []
        self.valid_cat_ids = []
        self.class_names = []
        self.name_to_cat_id = {}
        self.test_images = {}
        self.test_image_paths = []
        self.test_annotation_sha256 = ""

    def ensure_dataset(self, target_dir: Optional[str] = None) -> str:
        """Đảm bảo tập dữ liệu có sẵn (tái sử dụng từ cache hoặc tải từ Roboflow)."""
        candidate_paths = [
            target_dir if target_dir else None,
            self.dataset_root if self.dataset_root else None,
            "/data/dataset/Completed-Project-5",
            "./dataset/Completed-Project-5",
            os.path.abspath(os.path.join(CURRENT_DIR, "..", "Completed-Project-5"))
        ]
        candidate_paths = [p for p in candidate_paths if p]

        for p in candidate_paths:
            if os.path.exists(p):
                ann = os.path.join(p, "test", "_annotations.coco.json")
                if os.path.exists(ann):
                    print(f"[DATASET] Tìm thấy tập dữ liệu COCO có sẵn tại: {os.path.abspath(p)}")
                    self.dataset_root = p
                    return p

        download_dir = target_dir or self.dataset_root or os.path.join(CURRENT_DIR, "dataset")
        os.makedirs(download_dir, exist_ok=True)
        if not self.api_key:
            raise ValueError(
                "[ERROR] Thiếu Roboflow API Key! Vui lòng điền API Key vào tệp '.env' "
                "hoặc thiết lập biến môi trường ROBOFLOW_API_KEY."
            )

        from roboflow import Roboflow
        rf = Roboflow(api_key=self.api_key)
        project = rf.workspace(self.workspace).project(self.project)
        version = project.version(self.version)

        curr_dir = os.getcwd()
        try:
            os.chdir(download_dir)
            dataset = version.download("coco")
            actual_location = getattr(dataset, "location", download_dir)
        finally:
            os.chdir(curr_dir)

        actual_path = actual_location
        if not os.path.exists(os.path.join(actual_path, "test", "_annotations.coco.json")):
            for root, _, files in os.walk(download_dir):
                if "_annotations.coco.json" in files and os.path.basename(root) == "test":
                    actual_path = os.path.dirname(root)
                    break

        print(f"[DATASET] Tải thành công tập dữ liệu COCO về: {actual_path}")
        self.dataset_root = actual_path
        return actual_path

    def load_test_split(self):
        """Đọc và lọc bỏ lớp rác từ _annotations.coco.json, lập chỉ mục 1.481 ảnh test."""
        from pycocotools.coco import COCO

        test_ann = os.path.join(self.dataset_root, "test", "_annotations.coco.json")
        if not os.path.exists(test_ann):
            for root, _, files in os.walk(self.dataset_root):
                if "_annotations.coco.json" in files and os.path.basename(root) == "test":
                    test_ann = os.path.join(root, "_annotations.coco.json")
                    break

        if not os.path.exists(test_ann):
            raise FileNotFoundError(f"[ERROR] Không tìm thấy test/_annotations.coco.json trong {self.dataset_root}")

        print(f"[DATASET] Nạp COCO Ground Truth từ: {test_ann}")
        self.test_annotation_sha256 = sha256_file(test_ann)
        self.coco_gt = COCO(test_ann)

        raw_cats = sorted(self.coco_gt.dataset.get("categories", []), key=lambda x: x["id"])
        self.valid_categories = [c for c in raw_cats if c["name"].strip() != self.dummy_cat_name]
        self.valid_cat_ids = [c["id"] for c in self.valid_categories]
        self.class_names = [c["name"] for c in self.valid_categories]
        self.name_to_cat_id = {c["name"]: c["id"] for c in self.valid_categories}

        test_dir = os.path.dirname(test_ann)
        self.test_images = {}
        self.test_image_paths = []

        for img_id in self.coco_gt.getImgIds():
            info = self.coco_gt.loadImgs(img_id)[0]
            fn = info["file_name"]
            candidates = [
                os.path.join(test_dir, fn),
                os.path.join(test_dir, "images", fn),
                os.path.join(self.dataset_root, "test", fn),
                os.path.join(self.dataset_root, "test", "images", fn)
            ]
            actual_f = next((p for p in candidates if os.path.exists(p)), None)
            if actual_f:
                self.test_images[img_id] = {
                    "path": actual_f,
                    "width": info["width"],
                    "height": info["height"]
                }
                self.test_image_paths.append(actual_f)

        missing_image_ids = sorted(set(self.coco_gt.getImgIds()) - set(self.test_images))
        if missing_image_ids:
            raise FileNotFoundError(
                f"Missing {len(missing_image_ids)} test images declared in the COCO JSON; "
                f"first image IDs: {missing_image_ids[:10]}. Refusing a partial evaluation."
            )

        print(f"[DATASET] Sẵn sàng đánh giá trên {len(self.valid_categories)} lớp nguyên liệu và {len(self.test_image_paths)} ảnh test.\n")
        return self.coco_gt, self.valid_categories, self.test_images


# ==============================================================================
# PHẦN 3: CHUẨN HÓA ĐẦU RA MÔ HÌNH VỀ COCO
# ==============================================================================
def build_class_mapping(
    indexed_names: Dict[int, str],
    name_to_cat_id: Dict[str, int],
    model_cfg: Dict[str, Any],
    backend_name: str,
) -> Tuple[Dict[int, int], set[int]]:
    """Build an explicit checkpoint-index -> COCO-category mapping."""
    aliases = model_cfg.get("class_name_map", {})
    ignored_names = set(model_cfg.get("ignored_checkpoint_classes", []))
    index_to_cat_id = {}
    ignored_indices = set()
    unknown = []

    for index, raw_name in indexed_names.items():
        if raw_name in ignored_names:
            ignored_indices.add(int(index))
            continue
        canonical_name = aliases.get(raw_name, raw_name)
        if canonical_name not in name_to_cat_id:
            unknown.append(raw_name)
            continue
        index_to_cat_id[int(index)] = int(name_to_cat_id[canonical_name])

    if (
        unknown
        or len(index_to_cat_id) != len(name_to_cat_id)
        or set(index_to_cat_id.values()) != set(name_to_cat_id.values())
    ):
        raise ValueError(
            f"{backend_name} checkpoint classes do not map one-to-one onto COCO ground truth. "
            f"Unknown={unknown}; configure checkpoint_classes/class_name_map/"
            "ignored_checkpoint_classes."
        )
    return index_to_cat_id, ignored_indices


def normalize_coco_detection(
    box_xyxy: Any,
    score: float,
    category_id: int,
    image_width: int,
    image_height: int,
) -> Optional[Dict[str, Any]]:
    """Clip one xyxy prediction to the source image and convert it to COCO xywh."""
    xmin = max(0.0, min(float(image_width), float(box_xyxy[0])))
    ymin = max(0.0, min(float(image_height), float(box_xyxy[1])))
    xmax = max(0.0, min(float(image_width), float(box_xyxy[2])))
    ymax = max(0.0, min(float(image_height), float(box_xyxy[3])))
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0.0 or height <= 0.0:
        return None
    return {
        "bbox": [xmin, ymin, width, height],
        "score": float(score),
        "category_id": int(category_id),
    }


class UltralyticsAdapter:
    """Bộ điều hợp cho YOLO và RT-DETR thuộc Ultralytics."""
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        weights_path: str,
        name_to_cat_id: Dict[str, int],
        device: str = "cuda:0",
        conf_thresh: float = 0.0,
        imgsz: int = 640,
        max_detections: int = 100,
    ):
        self.device = device
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.max_detections = max_detections
        self.nms_iou_threshold = model_cfg.get("nms_iou_threshold")

        model_type = str(model_cfg.get("model_type", "yolo")).lower()
        if model_type == "rtdetr":
            from ultralytics import RTDETR
            self.model = RTDETR(weights_path)
        elif model_type == "yolo":
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
        else:
            raise ValueError(f"model_type Ultralytics không được hỗ trợ: {model_type}")

        checkpoint_names = getattr(self.model, "names", {})
        if isinstance(checkpoint_names, list):
            checkpoint_names = {i: str(name) for i, name in enumerate(checkpoint_names)}
        else:
            checkpoint_names = {int(index): str(name) for index, name in checkpoint_names.items()}
        self.checkpoint_classes = checkpoint_names
        self.class_idx_to_cat_id, self.ignored_class_indices = build_class_mapping(
            checkpoint_names, name_to_cat_id, model_cfg, "Ultralytics"
        )
        self.backend_max_detections = (
            max(self.max_detections, 300)
            if self.ignored_class_indices
            else self.max_detections
        )

    def predict_image(self, image_bgr: np.ndarray, orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
        predict_kwargs = {
            "source": image_bgr,
            "conf": self.conf_thresh,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False,
            "max_det": self.backend_max_detections,
        }
        if self.nms_iou_threshold is not None:
            predict_kwargs["iou"] = float(self.nms_iou_threshold)
        res = self.model.predict(**predict_kwargs)[0]
        detections = []

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls_idx in zip(boxes, scores, classes):
                if cls_idx in self.ignored_class_indices:
                    continue
                if cls_idx not in self.class_idx_to_cat_id:
                    raise ValueError(f"Không ánh xạ được class index {cls_idx} của checkpoint Ultralytics")
                detection = normalize_coco_detection(
                    box, score, self.class_idx_to_cat_id[cls_idx], orig_w, orig_h
                )
                if detection is not None:
                    detections.append(detection)
        detections.sort(key=lambda detection: detection["score"], reverse=True)
        return detections[: self.max_detections]

    def get_torch_module(self):
        return self.model.model


class TorchvisionAdapter:
    """Bộ điều hợp cho họ mô hình Torchvision: Faster R-CNN, FCOS, RetinaNet."""
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        weights_path: str,
        class_names: List[str],
        name_to_cat_id: Dict[str, int],
        device: str = "cuda:0",
        conf_thresh: float = 0.0,
        imgsz: int = 640,
        max_detections: int = 100,
    ):
        import torch
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn,
            fasterrcnn_resnet50_fpn_v2,
            fasterrcnn_mobilenet_v3_large_320_fpn,
            fasterrcnn_mobilenet_v3_large_fpn,
            fcos_resnet50_fpn,
            retinanet_resnet50_fpn,
            retinanet_resnet50_fpn_v2,
        )

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.max_detections = max_detections
        self.label_offset = int(model_cfg.get("label_offset", 1))
        self.checkpoint_classes = model_cfg.get("checkpoint_classes") or class_names
        indexed_names = {index: name for index, name in enumerate(self.checkpoint_classes)}
        self.class_idx_to_cat_id, self.ignored_class_indices = build_class_mapping(
            indexed_names, name_to_cat_id, model_cfg, "Torchvision"
        )
        self.backend_max_detections = (
            max(self.max_detections, 300)
            if self.ignored_class_indices
            else self.max_detections
        )
        num_model_classes = len(self.checkpoint_classes) + self.label_offset
        model_type = str(model_cfg["model_type"])
        kwargs = dict(model_cfg.get("constructor_kwargs", {}))
        kwargs["min_size"] = int(imgsz)
        kwargs["max_size"] = int(imgsz)

        ckpt = torch.load(weights_path, map_location=self.device)
        embedded_classes = ckpt.get("class_names") if isinstance(ckpt, dict) else None
        if embedded_classes is not None:
            embedded_classes = [str(name) for name in embedded_classes]
            if embedded_classes != list(self.checkpoint_classes):
                raise ValueError(
                    "Torchvision checkpoint class order does not match checkpoint_classes: "
                    f"checkpoint={embedded_classes}, configured={list(self.checkpoint_classes)}"
                )
        elif not model_cfg.get("checkpoint_classes"):
            raise ValueError(
                "Torchvision checkpoint does not contain class_names. Configure checkpoint_classes "
                "explicitly so class order can be verified before COCO evaluation."
            )
        if hasattr(ckpt, "state_dict"):
            state_dict = ckpt.state_dict()
        elif isinstance(ckpt, dict):
            state_dict = ckpt.get(
                "model_state_dict",
                ckpt.get("state_dict", ckpt.get("model", ckpt)),
            )
            if hasattr(state_dict, "state_dict"):
                state_dict = state_dict.state_dict()
        else:
            raise TypeError(
                f"Unsupported Torchvision checkpoint payload: {type(ckpt).__name__}"
            )
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("Torchvision checkpoint does not contain a non-empty state_dict")
        for prefix in ("module.", "model."):
            if state_dict and all(str(key).startswith(prefix) for key in state_dict):
                state_dict = {str(key)[len(prefix):]: value for key, value in state_dict.items()}

        if model_type in {"fasterrcnn_resnet50_fpn", "fasterrcnn_resnet50_fpn_v2"}:
            kwargs["box_score_thresh"] = float(conf_thresh)
            kwargs["box_detections_per_img"] = int(self.backend_max_detections)
            constructor = (
                fasterrcnn_resnet50_fpn_v2
                if model_type.endswith("_v2")
                else fasterrcnn_resnet50_fpn
            )
            self.model = constructor(
                weights=None, weights_backbone=None, num_classes=num_model_classes, **kwargs
            )
        elif model_type in {
            "fasterrcnn_mobilenet_v3_large_fpn",
            "fasterrcnn_mobilenet_v3_large_320_fpn",
        }:
            constructor = (
                fasterrcnn_mobilenet_v3_large_320_fpn
                if model_type.endswith("_320_fpn")
                else fasterrcnn_mobilenet_v3_large_fpn
            )
            kwargs["box_score_thresh"] = float(conf_thresh)
            kwargs["box_detections_per_img"] = int(self.backend_max_detections)
            self.model = constructor(weights=None, weights_backbone=None, num_classes=num_model_classes, **kwargs)
        elif model_type == "fcos_resnet50_fpn":
            kwargs["score_thresh"] = float(conf_thresh)
            kwargs["detections_per_img"] = int(self.backend_max_detections)
            self.model = fcos_resnet50_fpn(
                weights=None, weights_backbone=None, num_classes=num_model_classes, **kwargs
            )
        elif model_type in {"retinanet_resnet50_fpn", "retinanet_resnet50_fpn_v2"}:
            kwargs["score_thresh"] = float(conf_thresh)
            kwargs["detections_per_img"] = int(self.backend_max_detections)
            constructor = (
                retinanet_resnet50_fpn_v2
                if model_type.endswith("_v2")
                else retinanet_resnet50_fpn
            )
            self.model = constructor(
                weights=None, weights_backbone=None, num_classes=num_model_classes, **kwargs
            )
        else:
            raise ValueError(f"Không hỗ trợ kiến trúc Torchvision: {model_type}")

        self.effective_constructor_kwargs = dict(kwargs)

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint không tương thích kiến trúc {model_type} với label_offset={self.label_offset}. "
                "Không được fallback sang kiến trúc/số lớp khác vì sẽ làm sai benchmark."
            ) from exc

        self.model.to(self.device)
        self.model.eval()

    def predict_image(self, image_bgr: np.ndarray, orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
        import torch

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Không resize thủ công: GeneralizedRCNNTransform của model thực hiện đúng
        # min_size/max_size đã khai báo trong constructor_kwargs.
        img_tensor = torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0

        with torch.no_grad():
            output = self.model([img_tensor.to(self.device)])[0]

        detections = []
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        order = np.argsort(-scores, kind="stable")
        for box, score, label in zip(boxes[order], scores[order], labels[order]):
            lbl = int(label)
            cls_idx = lbl - self.label_offset
            if score < self.conf_thresh:
                continue
            if not 0 <= cls_idx < len(self.checkpoint_classes):
                raise ValueError(
                    f"Label {lbl} is invalid for label_offset={self.label_offset} "
                    f"and {len(self.checkpoint_classes)} checkpoint classes"
                )
            if cls_idx in self.ignored_class_indices:
                continue
            if cls_idx not in self.class_idx_to_cat_id:
                raise ValueError(f"Checkpoint class index {cls_idx} has no COCO mapping")

            detection = normalize_coco_detection(
                box, score, self.class_idx_to_cat_id[cls_idx], orig_w, orig_h
            )
            if detection is not None:
                detections.append(detection)
                if len(detections) == self.max_detections:
                    break
        return detections

    def get_torch_module(self):
        return self.model


class RFDETRAdapter:
    """Bộ điều hợp riêng cho package RF-DETR của Roboflow (không phải Ultralytics RT-DETR)."""
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        weights_path: str,
        class_names: List[str],
        name_to_cat_id: Dict[str, int],
        device: str = "cuda:0",
        conf_thresh: float = 0.0,
        imgsz: int = 640,
        max_detections: int = 100,
    ):
        import rfdetr

        self.conf_thresh = conf_thresh
        self.max_detections = max_detections
        self.label_offset = int(model_cfg.get("label_offset", 0))
        self.checkpoint_classes = model_cfg.get("checkpoint_classes") or class_names
        indexed_names = {index: name for index, name in enumerate(self.checkpoint_classes)}
        self.class_idx_to_cat_id, self.ignored_class_indices = build_class_mapping(
            indexed_names, name_to_cat_id, model_cfg, "RF-DETR"
        )
        model_class_name = model_cfg.get("model_class", "RFDETRMedium")
        model_class = getattr(rfdetr, model_class_name, None)
        if model_class is None:
            raise ValueError(f"Không tìm thấy class {model_class_name} trong package rfdetr")

        # RF-DETR Medium defaults to 576, but the shared benchmark is fixed at 640.
        # RF-DETR supports a square resolution override when it is divisible by
        # patch_size * num_windows (640 is valid for the Medium variant).
        self.model = model_class.from_checkpoint(
            weights_path,
            device=device,
            resolution=int(imgsz),
        )
        model_context = getattr(self.model, "model", None)
        embedded_classes = getattr(model_context, "class_names", None)
        if embedded_classes is not None and list(embedded_classes) != list(self.checkpoint_classes):
            raise ValueError(
                "RF-DETR checkpoint class order does not match checkpoint_classes: "
                f"checkpoint={list(embedded_classes)}, configured={list(self.checkpoint_classes)}"
            )
        model_args = getattr(model_context, "args", None)
        self.background_class_id = int(
            getattr(model_args, "num_classes", len(self.checkpoint_classes))
        )
        self.backend_max_detections = int(
            getattr(model_args, "num_select", max(self.max_detections, 300))
        )
        if self.background_class_id != len(self.checkpoint_classes):
            raise ValueError(
                "RF-DETR checkpoint class count does not match checkpoint_classes: "
                f"checkpoint={self.background_class_id}, configured={len(self.checkpoint_classes)}"
            )
        self.imgsz = int(getattr(getattr(self.model, "model_config", None), "resolution", imgsz))
        if self.imgsz != int(imgsz):
            raise RuntimeError(
                f"RF-DETR did not apply the required resolution={imgsz}; "
                f"effective resolution is {self.imgsz}"
            )

    def predict_image(self, image_bgr: np.ndarray, orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
        from PIL import Image

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        output = self.model.predict(Image.fromarray(image_rgb), threshold=self.conf_thresh)
        if not hasattr(output, "xyxy"):
            raise TypeError("RF-DETR predict() không trả về supervision.Detections như mong đợi")
        boxes = np.asarray(output.xyxy)
        scores = np.asarray(output.confidence)
        labels = np.asarray(output.class_id).astype(int)
        # RF-DETR exposes its final no-object logit as class_id == num_classes.
        # Remove it before max_detections so a background row never consumes one
        # of the 100 COCO detection slots.
        foreground = labels != self.background_class_id
        class_indices = labels - self.label_offset
        invalid_foreground = foreground & (
            (class_indices < 0) | (class_indices >= len(self.checkpoint_classes))
        )
        if np.any(invalid_foreground):
            invalid_labels = sorted(int(label) for label in np.unique(labels[invalid_foreground]))
            raise ValueError(
                f"RF-DETR labels {invalid_labels} are outside configured checkpoint classes"
            )
        ignored = np.isin(class_indices, list(self.ignored_class_indices))
        candidates = np.flatnonzero(
            foreground & ~ignored & (scores >= self.conf_thresh)
        )
        order = candidates[np.argsort(-scores[candidates], kind="stable")][
            : self.max_detections
        ]
        detections = []
        for box, score, raw_label in zip(boxes[order], scores[order], labels[order]):
            cls_idx = int(raw_label) - self.label_offset
            if not 0 <= cls_idx < len(self.checkpoint_classes):
                raise ValueError(f"RF-DETR label {raw_label} is outside configured checkpoint classes")
            if cls_idx in self.ignored_class_indices:
                continue
            if cls_idx not in self.class_idx_to_cat_id:
                raise ValueError(f"RF-DETR class index {cls_idx} has no COCO mapping")
            detection = normalize_coco_detection(
                box, score, self.class_idx_to_cat_id[cls_idx], orig_w, orig_h
            )
            if detection is not None:
                detections.append(detection)
        return detections

    def get_torch_module(self):
        import torch

        # RF-DETR wraps the actual nn.Module twice:
        # RFDETR -> ModelContext -> LWDETR.  Returning ModelContext here breaks
        # the shared complexity benchmark because it does not implement eval(),
        # to(), parameters(), or modules().
        candidate = self.model
        for _ in range(3):
            if isinstance(candidate, torch.nn.Module):
                return candidate
            candidate = getattr(candidate, "model", None)
            if candidate is None:
                break
        raise TypeError("RF-DETR did not expose an underlying torch.nn.Module")


# ==============================================================================
# PHẦN 4: ĐO ĐẠC HIỆU NĂNG TÍNH TOÁN (BENCHMARK ENGINE)
# ==============================================================================
class BenchmarkEngine:
    """Đo đạc 4 thông số: GFLOPs, Parameters, Latency và FPS."""

    @staticmethod
    def measure_complexity(
        adapter: Any, device: str = "cuda:0", imgsz: int = 640
    ) -> Tuple[Optional[float], float]:
        import torch
        torch_model = adapter.get_torch_module()
        effective_device = torch.device(
            device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu"
        )
        torch_model.eval().to(effective_device)

        gflops = None
        params_m = round(sum(p.numel() for p in torch_model.parameters()) / 1e6, 2)
        modules = list(torch_model.modules())
        profile_state = [
            (
                module,
                set(getattr(module, "_forward_hooks", {})),
                set(getattr(module, "_forward_pre_hooks", {})),
                set(getattr(module, "_buffers", {})),
            )
            for module in modules
        ]

        def cleanup_thop_state() -> None:
            """Remove only hooks/buffers added by THOP, preserving model-owned hooks."""
            for module, forward_hooks, pre_hooks, buffers in profile_state:
                for hook_id in set(getattr(module, "_forward_hooks", {})) - forward_hooks:
                    module._forward_hooks.pop(hook_id, None)
                    getattr(module, "_forward_hooks_with_kwargs", {}).pop(hook_id, None)
                    getattr(module, "_forward_hooks_always_called", {}).pop(hook_id, None)
                for hook_id in set(getattr(module, "_forward_pre_hooks", {})) - pre_hooks:
                    module._forward_pre_hooks.pop(hook_id, None)
                    getattr(module, "_forward_pre_hooks_with_kwargs", {}).pop(hook_id, None)
                for buffer_name in set(getattr(module, "_buffers", {})) - buffers:
                    if buffer_name in {"total_ops", "total_params"}:
                        module._buffers.pop(buffer_name, None)

        try:
            from thop import profile
            dummy = torch.randn(1, 3, imgsz, imgsz).to(effective_device)
            try:
                flops, _ = profile(torch_model, inputs=(dummy,), verbose=False)
            except Exception:
                cleanup_thop_state()
                flops, _ = profile(torch_model, inputs=([dummy[0]],), verbose=False)
            gflops = round(flops / 1e9, 2)
        except Exception as exc:
            print(f"  [WARNING] GFLOPs unavailable (THOP profiling failed): {exc}")
        finally:
            cleanup_thop_state()

        return gflops, params_m

    @staticmethod
    def measure_latency_and_fps(adapter: Any, test_image_paths: List[str], device: str = "cuda:0", warmup: int = 10) -> Tuple[float, float]:
        import torch
        total_images = len(test_image_paths)
        if total_images == 0:
            return 0.0, 0.0
        use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

        for i in range(min(warmup, total_images)):
            img = cv2.imread(test_image_paths[i])
            if img is None:
                raise RuntimeError(f"Cannot decode benchmark image: {test_image_paths[i]}")
            adapter.predict_image(img, img.shape[1], img.shape[0])

        if use_cuda:
            torch.cuda.synchronize()

        elapsed = 0.0
        measured_images = 0
        for p in test_image_paths:
            img = cv2.imread(p)
            if img is None:
                raise RuntimeError(f"Cannot decode benchmark image: {p}")
            if use_cuda:
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            adapter.predict_image(img, img.shape[1], img.shape[0])
            if use_cuda:
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start_time
            measured_images += 1

        latency_ms = round((elapsed / measured_images) * 1000, 2)
        fps = round(measured_images / elapsed, 2) if elapsed > 0 else 0.0
        return latency_ms, fps


# ==============================================================================
# PHẦN 5: BỘ ĐÁNH GIÁ CHUẨN COCO
# ==============================================================================
class COCOBenchmarkEvaluator:
    """Đánh giá toàn diện bằng pycocotools COCOeval trên đúng 32 lớp nguyên liệu."""

    @staticmethod
    def _operating_point_from_cocoeval(
        coco_eval,
        coco_gt,
        score_threshold: float,
        iou_threshold: float,
    ) -> Dict[str, Any]:
        """Derive raw P/R/F1 from COCOeval matches, including COCO ignore/crowd rules."""
        iou_indexes = np.flatnonzero(np.isclose(coco_eval.params.iouThrs, iou_threshold))
        if len(iou_indexes) != 1:
            raise ValueError(
                f"Operating IoU {iou_threshold} is not in COCOeval iouThrs="
                f"{coco_eval.params.iouThrs.tolist()}"
            )
        iou_index = int(iou_indexes[0])
        area_all = list(coco_eval.params.areaRng[0])
        max_det = int(coco_eval.params.maxDets[-1])
        counts = {
            int(cat_id): {"TP": 0, "FP": 0, "FN": 0}
            for cat_id in coco_eval.params.catIds
        }

        for eval_image in coco_eval.evalImgs:
            if eval_image is None:
                continue
            if list(eval_image["aRng"]) != area_all or int(eval_image["maxDet"]) != max_det:
                continue

            cat_id = int(eval_image["category_id"])
            if cat_id not in counts:
                continue
            scores = np.asarray(eval_image["dtScores"], dtype=np.float64)
            selected = scores >= score_threshold
            matches = np.asarray(eval_image["dtMatches"])[iou_index, selected]
            ignored_detections = np.asarray(eval_image["dtIgnore"])[iou_index, selected].astype(bool)
            valid_detections = ~ignored_detections
            true_positive_mask = (matches > 0) & valid_detections
            false_positive_mask = (matches == 0) & valid_detections

            valid_ground_truth = int(
                np.count_nonzero(~np.asarray(eval_image["gtIgnore"], dtype=bool))
            )
            matched_ground_truth_ids = {
                int(gt_id) for gt_id in matches[true_positive_mask] if int(gt_id) > 0
            }
            counts[cat_id]["TP"] += len(matched_ground_truth_ids)
            counts[cat_id]["FP"] += int(np.count_nonzero(false_positive_mask))
            counts[cat_id]["FN"] += max(0, valid_ground_truth - len(matched_ground_truth_ids))

        per_class = {}
        for cat_id, cls_counts in counts.items():
            tp, fp, fn = cls_counts["TP"], cls_counts["FP"], cls_counts["FN"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            cat_name = coco_gt.loadCats(cat_id)[0]["name"]
            per_class[cat_name] = {
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1, 4),
                "TP": tp,
                "FP": fp,
                "FN": fn,
            }

        tp = sum(item["TP"] for item in counts.values())
        fp = sum(item["FP"] for item in counts.values())
        fn = sum(item["FN"] for item in counts.values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "score_threshold": score_threshold,
            "iou_threshold": iou_threshold,
            "source": "pycocotools.COCOeval.evalImgs",
            "per_class": per_class,
        }

    @staticmethod
    def evaluate(
        coco_gt,
        predictions_list: List[Dict[str, Any]],
        valid_cat_ids: List[int],
        image_ids: List[int],
        max_detections: int = 100,
        operating_score_threshold: float = 0.25,
        operating_iou_threshold: float = 0.50,
    ) -> Dict[str, Any]:
        from pycocotools.cocoeval import COCOeval

        if predictions_list:
            coco_dt = coco_gt.loadRes(predictions_list)
        else:
            # COCO.loadRes([]) is not supported consistently across pycocotools
            # versions. Build a valid empty result dataset so even the zero-
            # detection case is evaluated by COCOeval rather than manual metrics.
            from pycocotools.coco import COCO

            selected_image_ids = set(int(image_id) for image_id in image_ids)
            coco_dt = COCO()
            coco_dt.dataset = {
                "images": [
                    image
                    for image in coco_gt.dataset.get("images", [])
                    if int(image["id"]) in selected_image_ids
                ],
                "categories": [
                    category
                    for category in coco_gt.dataset.get("categories", [])
                    if int(category["id"]) in set(int(cat_id) for cat_id in valid_cat_ids)
                ],
                "annotations": [],
            }
            coco_dt.createIndex()
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.catIds = valid_cat_ids
        coco_eval.params.imgIds = sorted(image_ids)
        # Giữ giao thức COCO chuẩn; maxDets cuối là giá trị dùng cho AP/AR chính.
        coco_eval.params.maxDets = [1, 10, int(max_detections)]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        operating_metrics = COCOBenchmarkEvaluator._operating_point_from_cocoeval(
            coco_eval,
            coco_gt,
            score_threshold=operating_score_threshold,
            iou_threshold=operating_iou_threshold,
        )

        stats = coco_eval.stats
        map50_95 = float(stats[0])
        map50 = float(stats[1])
        map75 = float(stats[2])

        per_class_metrics = {}
        for idx_k, cat_id in enumerate(coco_eval.params.catIds):
            # COCOeval may normalize catIds to NumPy integer scalars.  COCO.loadCats
            # only handles Python int (or an iterable), so normalize explicitly.
            cat_name = coco_gt.loadCats(int(cat_id))[0]["name"]
            p_all = coco_eval.eval["precision"][:, :, idx_k, 0, 2]
            v_all = p_all[p_all > -1]
            cls_map50_95 = float(np.mean(v_all)) if len(v_all) > 0 else 0.0

            p_50 = coco_eval.eval["precision"][0, :, idx_k, 0, 2]
            v_50 = p_50[p_50 > -1]
            cls_map50 = float(np.mean(v_50)) if len(v_50) > 0 else 0.0

            r_cls = coco_eval.eval["recall"][:, idx_k, 0, 2]
            v_r = r_cls[r_cls > -1]
            cls_ar100 = float(np.mean(v_r)) if len(v_r) > 0 else 0.0

            per_class_metrics[cat_name] = {
                "Precision": operating_metrics["per_class"][cat_name]["Precision"],
                "Recall": operating_metrics["per_class"][cat_name]["Recall"],
                "F1": operating_metrics["per_class"][cat_name]["F1"],
                "TP": operating_metrics["per_class"][cat_name]["TP"],
                "FP": operating_metrics["per_class"][cat_name]["FP"],
                "FN": operating_metrics["per_class"][cat_name]["FN"],
                "mAP50": round(cls_map50, 4),
                "mAP50-95": round(cls_map50_95, 4),
                "AR100": round(cls_ar100, 4),
            }

        return {
            "Precision": operating_metrics["Precision"],
            "Recall": operating_metrics["Recall"],
            "F1": operating_metrics["F1"],
            "mAP50_95": round(map50_95, 4),
            "mAP50": round(map50, 4),
            "mAP75": round(map75, 4),
            "AP_small": round(float(stats[3]), 4),
            "AP_medium": round(float(stats[4]), 4),
            "AP_large": round(float(stats[5]), 4),
            "AR1": round(float(stats[6]), 4),
            "AR10": round(float(stats[7]), 4),
            "AR100": round(float(stats[8]), 4),
            "operating_point": {
                key: operating_metrics[key]
                for key in ("TP", "FP", "FN", "score_threshold", "iou_threshold", "source")
            },
            "per_class": per_class_metrics
        }


# ==============================================================================
# PHẦN 6: XUẤT BÁO CÁO KHOA HỌC & TÍNH NĂNG GỘP KẾT QUẢ TOÀN ĐỘI (MASTER MERGE)
# ==============================================================================
class ScientificReportVisualizer:
    """Xuất bảng metric, biểu đồ cấu hình DPI và hỗ trợ gộp kết quả toàn đội."""

    @staticmethod
    def generate_all_reports(
        results_list: List[Dict[str, Any]],
        valid_categories: List[Dict[str, Any]],
        output_dir: str,
        chart_dpi: int = 300,
    ):
        os.makedirs(output_dir, exist_ok=True)
        class_names = [c["name"] for c in valid_categories]

        # 1. BẢNG 10 THÔNG SỐ CHUẨN BẮT BUỘC
        summary_rows = []
        for r in results_list:
            is_pending = "Chờ" in r.get("status", "")
            patience_val = r.get("Patience")
            patience_str = "-" if (patience_val is None or str(patience_val).strip() in ["", "null", "None"]) else str(patience_val)
            gflops_value = r.get("GFLOPs")

            summary_rows.append({
                "Mô hình": r["model_name"],
                "Precision": "-" if is_pending else f"{float(r['Precision']):.4f}",
                "Recall": "-" if is_pending else f"{float(r['Recall']):.4f}",
                "F1": "-" if is_pending else f"{float(r.get('F1', 0.0)):.4f}",
                "mAP@50": "-" if is_pending else f"{float(r['mAP50']):.4f}",
                "mAP@50-95": "-" if is_pending else f"{float(r['mAP50_95']):.4f}",
                "mAP@75": "-" if is_pending else f"{float(r.get('mAP75', 0.0)):.4f}",
                "AR@100": "-" if is_pending else f"{float(r.get('AR100', 0.0)):.4f}",
                "Patience": patience_str,
                "GFLOPs": "-" if is_pending or gflops_value is None else f"{float(gflops_value):.2f}",
                "Parameters": "-" if is_pending else f"{float(r['Parameters']):.2f}",
                "Latency": "-" if is_pending else f"{float(r['Latency']):.2f}",
                "FPS": "-" if is_pending else f"{float(r['FPS']):.2f}",
                "Trạng thái": r.get("status", "Hoàn tất")
            })

        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, "summary_coco_comparison.csv")
        summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 2. XUẤT BẢNG MARKDOWN TRỰC QUAN
        md_path = os.path.join(output_dir, "summary_coco_comparison.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# BẢNG TỔNG HỢP SO SÁNH CÁC MÔ HÌNH OBJECT DETECTION (COCO BENCHMARK)\n\n")
            f.write(f"- **Đề tài**: Nghiên cứu khoa học - Nhận diện nguyên liệu nấu ăn (32 lớp)\n")
            f.write(f"- **Thời gian**: {time.strftime('%d/%m/%Y %H:%M:%S')}\n")
            first_protocol = next((r.get("protocol") for r in results_list if r.get("protocol")), None)
            if first_protocol:
                settings = first_protocol["settings"]
                f.write(
                    f"- **Chuẩn đánh giá**: pycocotools COCOeval, "
                    f"{first_protocol['num_test_images']} ảnh, protocol `{first_protocol['protocol_id']}`\n"
                )
                f.write(
                    f"- **P/R/F1**: suy ra từ COCOeval matches tại "
                    f"conf={settings['operating_conf_threshold']}, "
                    f"IoU={settings['operating_iou_threshold']}; các cột mAP/AR là COCO metrics.\n\n"
                )
            else:
                f.write("- **Chuẩn đánh giá**: chưa có kết quả hoàn tất để xác định protocol.\n\n")
            f.write(dataframe_to_markdown(summary_df))
            f.write("\n\n*Ghi chú: Không đồng nhất Precision/Recall tại một ngưỡng cố định với COCO AP/AR tích phân.*\n")

        # 3. BẢNG ĐỐI CHIẾU CHÉO 32 LỚP NGUYÊN LIỆU
        cross_rows = []
        for cname in class_names:
            row_dict = {"Class Name": cname}
            for res in results_list:
                mname = res["model_name"]
                cdata = res.get("per_class", {}).get(cname, {})
                row_dict[f"{mname} (mAP50)"] = cdata.get("mAP50", "-")
                row_dict[f"{mname} (mAP50-95)"] = cdata.get("mAP50-95", "-")
            cross_rows.append(row_dict)

        cross_df = pd.DataFrame(cross_rows)
        cross_csv = os.path.join(output_dir, "all_models_per_class_comparison.csv")
        cross_df.to_csv(cross_csv, index=False, encoding="utf-8-sig")

        # 4. BIỂU ĐỒ SO SÁNH ĐỒ HỌA 300 DPI
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            evaluated = [r for r in results_list if "Chờ" not in r.get("status", "")]
            if len(evaluated) > 0:
                m_names = [r["model_name"] for r in evaluated]
                m_50 = [float(r["mAP50"]) * 100 for r in evaluated]
                m_50_95 = [float(r["mAP50_95"]) * 100 for r in evaluated]
                fps_list = [float(r["FPS"]) for r in evaluated]
                gflops_list = [
                    float(r["GFLOPs"]) if r.get("GFLOPs") is not None else np.nan
                    for r in evaluated
                ]
                params_list = [float(r["Parameters"]) for r in evaluated]

                x = np.arange(len(m_names))
                width = 0.35
                plt.figure(figsize=(max(10, len(m_names) * 2), 7), dpi=chart_dpi)
                sns.set_theme(style="whitegrid")
                plt.bar(x - width/2, m_50, width, label="mAP@50 (%)", color="#1f77b4")
                plt.bar(x + width/2, m_50_95, width, label="mAP@50-95 (%)", color="#ff7f0e")
                plt.ylabel("Độ chính xác (%)", fontsize=14, weight="bold")
                plt.title("Độ chính xác mAP@50 và mAP@50-95 chuẩn COCO Benchmark", fontsize=16, weight="bold", pad=15)
                plt.xticks(x, m_names, rotation=25, ha="right", fontsize=11, weight="bold")
                plt.legend(fontsize=12)
                plt.ylim(0, 105)

                for i in range(len(m_names)):
                    plt.text(x[i] - width/2, m_50[i] + 1.2, f"{m_50[i]:.1f}%", ha="center", fontsize=9, weight="bold")
                    plt.text(x[i] + width/2, m_50_95[i] + 1.2, f"{m_50_95[i]:.1f}%", ha="center", fontsize=9, weight="bold")

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "comparison_map_chart.png"), dpi=chart_dpi)
                plt.close()

                # Biểu đồ Trade-off Accuracy vs Speed
                plt.figure(figsize=(12, 7), dpi=chart_dpi)
                bubble_sizes = [max(50, p * 15) for p in params_list]
                if not all(np.isnan(v) for v in gflops_list):
                    scatter = plt.scatter(fps_list, m_50_95, s=bubble_sizes, c=gflops_list, cmap="plasma", alpha=0.85, edgecolors="black", linewidth=1.5)
                    cbar = plt.colorbar(scatter)
                    cbar.set_label("GFLOPs", fontsize=12, weight="bold")
                else:
                    plt.scatter(fps_list, m_50_95, s=bubble_sizes, color="#1f77b4", alpha=0.85, edgecolors="black", linewidth=1.5)

                for i, txt in enumerate(m_names):
                    plt.annotate(txt, (fps_list[i] + 0.5, m_50_95[i] + 0.3), fontsize=10, weight="bold")

                plt.xlabel("Tốc độ khung hình (FPS) - Càng cao càng tốt", fontsize=13, weight="bold")
                plt.ylabel("Độ chính xác mAP@50-95 (%) - Càng cao càng tốt", fontsize=13, weight="bold")
                plt.title("Biểu đồ Trade-off: Accuracy vs Speed (Bóng biểu thị số tham số Params)", fontsize=15, weight="bold", pad=15)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "comparison_tradeoff_chart.png"), dpi=chart_dpi)
                plt.close()

        except Exception as e:
            print(f"  [VISUALIZER] Không thể xuất biểu đồ: {e}")

        print(f"\n[DONE] Đã lưu báo cáo khoa học tại: {os.path.abspath(output_dir)}")
        print(f"  - {csv_path}")
        print(f"  - {md_path}")
        print(f"  - {cross_csv}")


# ==============================================================================
# PHẦN 7: ĐIỀU PHỐI ĐÁNH GIÁ MÔ HÌNH RIÊNG & HỢP NHẤT TOÀN ĐỘI
# ==============================================================================
def execute_common_coco_evaluation(
    dataset_dir: Optional[str] = None,
    output_dir: str = "Common_Evaluate_Results",
    target_models: Optional[List[str]] = None,
    config_data: Optional[Dict[str, Any]] = None,
    device: str = "cuda:0"
) -> List[Dict[str, Any]]:
    """
    Quy trình đánh giá có chọn lọc:
    - Nếu target_models được cung cấp: CHỈ đánh giá các mô hình trong danh sách đó.
    - Nếu không: chỉ đánh giá các mô hình có `enabled: true`.
    - Mỗi mô hình sau khi đánh giá sẽ xuất một file result_<model>.json để đồng nghiệp gửi nộp.
    """
    os.makedirs(output_dir, exist_ok=True)

    if config_data is None:
        if not os.path.exists(CONFIG_FILE):
            raise FileNotFoundError(f"[ERROR] Không tìm thấy tệp cấu hình tại {CONFIG_FILE}")
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config_data = json.load(f)

    validate_configuration(config_data)

    rf_info = config_data.get("roboflow", {})
    api_key = rf_info.get("api_key", "") or get_roboflow_api_key(config_data)
    workspace_name = rf_info.get("workspace", "nckhcict2025")
    project_name = rf_info.get("project", "completed-project")
    version_num = rf_info.get("version", 5)
    dummy_cat = rf_info.get("dummy_category_name", "19-8-BoSungAnhChoCacLopBiTh-v3zo")

    bench_cfg = config_data.get("benchmark_settings", {})
    img_size = int(bench_cfg.get("imgsz", 640))
    conf_thresh = float(bench_cfg.get("conf_threshold", 0.0))
    operating_conf_thresh = float(bench_cfg.get("operating_conf_threshold", 0.25))
    operating_iou_thresh = float(bench_cfg.get("iou_threshold", 0.50))
    max_detections = int(bench_cfg.get("max_detections", 100))
    warmup_n = int(bench_cfg.get("warmup_runs", 10))
    chart_dpi = int(bench_cfg.get("chart_dpi", 300))
    batch_size = int(bench_cfg.get("batch_size", 1))
    configured_classes = config_data.get("classes", [])

    if batch_size != 1:
        raise ValueError("The current latency protocol requires benchmark_settings.batch_size = 1")
    if not 0.0 <= conf_thresh <= operating_conf_thresh <= 1.0:
        raise ValueError("Require 0 <= conf_threshold <= operating_conf_threshold <= 1")
    if not 0.0 < operating_iou_thresh <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if max_detections != 100:
        raise ValueError(
            "The common COCO protocol requires max_detections = 100 because COCOeval.summarize() "
            "defines the primary AP/AR statistics at maxDets=100"
        )

    all_models_dict = config_data.get("models", {})

    # Lọc danh sách mô hình cần chạy trong phiên này
    if target_models:
        unknown_models = sorted(set(target_models) - set(all_models_dict))
        if unknown_models:
            raise ValueError(f"Unknown model keys in --models: {unknown_models}")
        eval_queue = {k: v for k, v in all_models_dict.items() if k in target_models}
    else:
        # Cả 8 kiến trúc đều được hỗ trợ, nhưng người dùng quyết định mô hình
        # nào tham gia lượt chạy hiện tại bằng cờ enabled.
        eval_queue = {
            model_key: model_cfg
            for model_key, model_cfg in all_models_dict.items()
            if model_cfg.get("enabled", False)
        }

    if not eval_queue:
        print(
            "[NOTICE] Không có mô hình nào được bật. "
            "Đặt enabled=true cho ít nhất một mô hình trong models_config.json."
        )
        return []

    resolved_checkpoint_paths = {}
    missing_checkpoints = []
    for model_key, model_cfg in eval_queue.items():
        resolved_path, attempted_paths = resolve_checkpoint_path(model_key, model_cfg)
        if resolved_path:
            resolved_checkpoint_paths[model_key] = resolved_path
        else:
            env_key = "MODEL_WEIGHTS_" + "".join(
                char if char.isalnum() else "_" for char in model_key.upper()
            )
            attempted = ", ".join(attempted_paths) if attempted_paths else "no path configured"
            missing_checkpoints.append(
                f"  - {model_key}: {attempted}; or set {env_key}"
            )
    if missing_checkpoints:
        raise FileNotFoundError(
            "The evaluation requires a trained checkpoint for every selected model.\n"
            + "\n".join(missing_checkpoints)
        )

    print("\n" + "="*85)
    print(f"BẮT ĐẦU ĐÁNH GIÁ CHỌN LỌC {len(eval_queue)} MÔ HÌNH ĐƯỢC CHỈ ĐỊNH:")
    for m in eval_queue.keys():
        print(f"  - [{m}] ({all_models_dict[m].get('display_name')})")
    print(f"THƯ MỤC XUẤT KẾT QUẢ: {os.path.abspath(output_dir)}")
    print("="*85 + "\n")

    # Quản lý tập dữ liệu COCO
    ds_manager = RoboflowCOCODatasetManager(
        api_key=api_key,
        dataset_root=dataset_dir,
        workspace=workspace_name,
        project=project_name,
        version=version_num,
        dummy_cat_name=dummy_cat,
    )
    ds_manager.ensure_dataset(target_dir=dataset_dir)
    coco_gt, valid_categories, test_images = ds_manager.load_test_split()

    valid_cat_ids = ds_manager.valid_cat_ids
    class_names = ds_manager.class_names
    name_to_cat_id = ds_manager.name_to_cat_id
    test_image_paths = ds_manager.test_image_paths

    expected_classes = int(bench_cfg.get("expected_num_classes", len(configured_classes) or 32))
    expected_images = bench_cfg.get("expected_test_images")
    if len(valid_categories) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} evaluated classes, found {len(valid_categories)} in test annotations"
        )
    if expected_images is not None and len(test_images) != int(expected_images):
        raise ValueError(
            f"Expected {int(expected_images)} test images, found {len(test_images)}. "
            "Check the dataset version before comparing models."
        )
    if configured_classes and class_names != configured_classes:
        raise ValueError(
            "Class names/order in models_config.json do not exactly match the COCO test annotations"
        )
    invalid_image_sizes = [
        (int(image_id), int(info["width"]), int(info["height"]))
        for image_id, info in test_images.items()
        if (int(info["width"]), int(info["height"])) != (img_size, img_size)
    ]
    if invalid_image_sizes:
        raise ValueError(
            f"The common protocol requires every test image to be {img_size}x{img_size}; "
            f"found {len(invalid_image_sizes)} mismatches, first={invalid_image_sizes[:5]}"
        )

    runtime_device = describe_runtime_device(device)
    protocol = build_protocol_metadata(
        bench_cfg,
        ds_manager.test_annotation_sha256,
        list(test_images.keys()),
        valid_categories,
        runtime_device,
    )
    print(f"[PROTOCOL] {protocol['name']} / {protocol['protocol_id'][:12]}")

    results_list = []

    for m_key, raw_model_cfg in eval_queue.items():
        m_cfg = dict(raw_model_cfg)
        m_cfg.setdefault(
            "ignored_checkpoint_classes",
            config_data.get("ignored_checkpoint_classes", []),
        )
        disp_name = m_cfg.get("display_name", m_key)
        print("\n" + "-"*85)
        print(f"TIẾN HÀNH ĐÁNH GIÁ MÔ HÌNH: [{disp_name}]")
        print("-"*85)

        # Đã kiểm tra đồng loạt trước khi tải dataset để không bao giờ xuất báo
        # cáo một phần khi thiếu checkpoint của một model được chọn.
        actual_weights = resolved_checkpoint_paths[m_key]

        print(f"  [MODEL] Khởi tạo mô hình từ: {actual_weights}")
        model_img_size = int(m_cfg.get("imgsz", img_size))

        # Khởi tạo Adapter
        if m_cfg["family"] == "ultralytics":
            adapter = UltralyticsAdapter(
                m_cfg, actual_weights, name_to_cat_id,
                device=device, conf_thresh=conf_thresh, imgsz=model_img_size,
                max_detections=max_detections,
            )
        elif m_cfg["family"] == "torchvision":
            adapter = TorchvisionAdapter(
                m_cfg, actual_weights, class_names, name_to_cat_id,
                device=device, conf_thresh=conf_thresh, imgsz=model_img_size,
                max_detections=max_detections,
            )
        elif m_cfg["family"] == "rfdetr":
            adapter = RFDETRAdapter(
                m_cfg, actual_weights, class_names, name_to_cat_id,
                device=device, conf_thresh=conf_thresh, imgsz=model_img_size,
                max_detections=max_detections,
            )
        else:
            raise ValueError(f"Unsupported model family: {m_cfg['family']}")
        effective_img_size = int(getattr(adapter, "imgsz", model_img_size))
        if effective_img_size != img_size:
            raise ValueError(
                f"{m_key} must run at the common imgsz={img_size}, but its adapter reports "
                f"effective resolution={effective_img_size}."
            )
        requested_inference_config = build_requested_inference_config(
            m_cfg, bench_cfg, class_names
        )
        resolved_inference_config = build_resolved_inference_config(
            adapter, requested_inference_config
        )

        # 1. Đo GFLOPs và Parameters
        print("  [BENCHMARK] Đo GFLOPs và Parameters...")
        gflops, params_m = BenchmarkEngine.measure_complexity(
            adapter, device=device, imgsz=effective_img_size
        )

        # 2. Đo Latency và FPS
        print("  [BENCHMARK] Đo Latency & FPS...")
        latency_ms, fps = BenchmarkEngine.measure_latency_and_fps(adapter, test_image_paths, device=device, warmup=warmup_n)

        # 3. Thu thập dự đoán COCO
        print(f"  [INFERENCE] Đang suy luận trên {len(test_images)} ảnh test...")
        coco_predictions = []
        for img_id, info in test_images.items():
            img_mat = cv2.imread(info["path"])
            if img_mat is None:
                raise RuntimeError(f"Cannot decode test image: {info['path']}")
            actual_h, actual_w = img_mat.shape[:2]
            if (actual_w, actual_h) != (int(info["width"]), int(info["height"])):
                raise ValueError(
                    f"Image dimensions disagree with COCO metadata for image_id={img_id}: "
                    f"file={actual_w}x{actual_h}, annotation={info['width']}x{info['height']}"
                )
            preds = adapter.predict_image(img_mat, actual_w, actual_h)
            for p in preds:
                coco_predictions.append({
                    "image_id": int(img_id),
                    "category_id": int(p["category_id"]),
                    "bbox": p["bbox"],
                    "score": float(p["score"])
                })

        prefix = os.path.join(output_dir, m_key)
        with gzip.open(f"{prefix}_coco_predictions.json.gz", "wt", encoding="utf-8") as pred_file:
            json.dump(coco_predictions, pred_file, ensure_ascii=False)

        # 4. Đánh giá COCOeval
        print("  [COCOEVAL] Tính toán metric COCO...")
        eval_metrics = COCOBenchmarkEvaluator.evaluate(
            coco_gt,
            coco_predictions,
            valid_cat_ids,
            list(test_images.keys()),
            max_detections=max_detections,
            operating_score_threshold=operating_conf_thresh,
            operating_iou_threshold=operating_iou_thresh,
        )

        # 5. Lưu CSV chi tiết 32 lớp. Tất cả metric chất lượng đều đến từ COCOeval.
        per_class_combined = {}
        per_class_rows = []
        for cname, cdata in eval_metrics["per_class"].items():
            combined = {
                "Class Name": cname,
                "Precision": cdata["Precision"],
                "Recall": cdata["Recall"],
                "F1": cdata["F1"],
                "TP": cdata["TP"],
                "FP": cdata["FP"],
                "FN": cdata["FN"],
                "mAP50": cdata["mAP50"],
                "mAP50-95": cdata["mAP50-95"],
                "AR100": cdata["AR100"],
            }
            per_class_rows.append(combined)
            per_class_combined[cname] = {k: v for k, v in combined.items() if k != "Class Name"}
        if per_class_rows:
            pd.DataFrame(per_class_rows).to_csv(
                f"{prefix}_test_per_class_metrics.csv", index=False, encoding="utf-8-sig"
            )

        res_item = {
            "model_key": m_key,
            "model_name": disp_name,
            "Precision": eval_metrics["Precision"],
            "Recall": eval_metrics["Recall"],
            "F1": eval_metrics["F1"],
            "mAP50": eval_metrics["mAP50"],
            "mAP50_95": eval_metrics["mAP50_95"],
            "mAP75": eval_metrics["mAP75"],
            "AP_small": eval_metrics["AP_small"],
            "AP_medium": eval_metrics["AP_medium"],
            "AP_large": eval_metrics["AP_large"],
            "AR1": eval_metrics["AR1"],
            "AR10": eval_metrics["AR10"],
            "AR100": eval_metrics["AR100"],
            "Patience": m_cfg.get("patience", 10),
            "GFLOPs": gflops,
            "Parameters": params_m,
            "Latency": latency_ms,
            "FPS": fps,
            "status": "Hoàn tất đánh giá",
            "protocol": protocol,
            "model_provenance": {
                "family": m_cfg["family"],
                "model_type": m_cfg.get("model_type"),
                "model_class": m_cfg.get("model_class"),
                "checkpoint_sha256": sha256_file(actual_weights),
                "inference_imgsz": effective_img_size,
                "inference_batch_size": batch_size,
                "device": device,
                "inference_config": resolved_inference_config,
                "inference_config_sha256": compute_payload_sha256(
                    resolved_inference_config
                ),
            },
            "operating_point": eval_metrics["operating_point"],
            "per_class": per_class_combined,
        }
        results_list.append(res_item)

        # ĐÓNG GÓI KẾT QUẢ ĐỘC LẬP: Xuất tệp JSON riêng của mô hình này để gửi nộp cho nhóm
        res_json_file = os.path.join(output_dir, f"result_{m_key}.json")
        with open(res_json_file, "w", encoding="utf-8") as rf:
            json.dump(res_item, rf, ensure_ascii=False, indent=2)
        print(f"  --> [ĐÓNG GÓI] Đã lưu file kết quả chuẩn hóa: {res_json_file}")

        # Giải phóng model trước khi khởi tạo model kế tiếp. Nếu không, Python
        # vẫn giữ adapter cũ trong lúc dựng adapter mới và có thể làm GPU OOM.
        del adapter
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Xuất báo cáo tổng hợp cho các mô hình vừa chạy
    ScientificReportVisualizer.generate_all_reports(
        results_list, valid_categories, output_dir, chart_dpi=chart_dpi
    )
    return results_list


def merge_team_results(results_dir: str, output_dir: Optional[str] = None):
    """
    TÍNH NĂNG HỢP NHẤT TOÀN ĐỘI (MASTER MERGE):
    Quét toàn bộ các tệp `result_<model>.json` do đồng nghiệp gửi về,
    tự động gộp thành bảng tổng hợp 8 mô hình và các biểu đồ so sánh.
    """
    if not output_dir:
        output_dir = results_dir

    print("\n" + "="*85)
    print("HỢP NHẤT KẾT QUẢ ĐÁNH GIÁ CỦA CÁC THÀNH VIÊN TRONG NHÓM (MASTER MERGE)")
    print(f"THƯ MỤC CHỨA KẾT QUẢ: {os.path.abspath(results_dir)}")
    print("="*85)

    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"[ERROR] Không tìm thấy {CONFIG_FILE}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    validate_configuration(config_data)
    all_models = config_data.get("models", {})
    bench_cfg = config_data.get("benchmark_settings", {})
    required_protocol_name = bench_cfg.get("protocol_name", PROTOCOL_NAME)
    required_imgsz = int(bench_cfg.get("imgsz", 640))
    required_batch_size = int(bench_cfg.get("batch_size", 1))

    # Đọc tất cả các file result_*.json
    collected_results = {}
    protocol_ids = set()
    for fn in os.listdir(results_dir):
        if fn.startswith("result_") and fn.endswith(".json"):
            fp = os.path.join(results_dir, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    m_key = data.get("model_key", fn.replace("result_", "").replace(".json", ""))
                    if m_key not in all_models:
                        raise ValueError(f"Unknown model_key '{m_key}'")
                    protocol_id = data.get("protocol", {}).get("protocol_id")
                    if not protocol_id:
                        raise ValueError(
                            f"Missing protocol.protocol_id; re-run with {required_protocol_name}"
                        )
                    protocol = data.get("protocol", {})
                    protocol_settings = protocol.get("settings", {})
                    if protocol_id != compute_protocol_id(protocol):
                        raise ValueError("protocol_id does not match the protocol payload")
                    if protocol.get("name") != required_protocol_name:
                        raise ValueError(
                            f"Protocol name must be {required_protocol_name}, got {protocol.get('name')}"
                        )
                    if protocol.get("evaluator") != "pycocotools.cocoeval.COCOeval":
                        raise ValueError("Result was not produced by pycocotools COCOeval")
                    if int(protocol_settings.get("inference_imgsz", -1)) != required_imgsz:
                        raise ValueError(f"Result was not evaluated at imgsz={required_imgsz}")
                    if int(protocol_settings.get("inference_batch_size", -1)) != required_batch_size:
                        raise ValueError(f"Result was not evaluated at batch_size={required_batch_size}")
                    expected_protocol_settings = {
                        "prediction_conf_threshold": bench_cfg["conf_threshold"],
                        "operating_conf_threshold": bench_cfg["operating_conf_threshold"],
                        "operating_iou_threshold": bench_cfg["iou_threshold"],
                        "max_detections": bench_cfg["max_detections"],
                    }
                    for setting_name, expected_value in expected_protocol_settings.items():
                        actual_value = protocol_settings.get(setting_name)
                        if actual_value is None or not np.isclose(
                            float(actual_value), float(expected_value)
                        ):
                            raise ValueError(
                                f"protocol.settings.{setting_name} must be {expected_value}"
                            )
                    if int(protocol.get("num_test_images", -1)) != int(
                        bench_cfg["expected_test_images"]
                    ):
                        raise ValueError("Result has the wrong number of test images")
                    protocol_class_names = [
                        category.get("name") for category in protocol.get("categories", [])
                    ]
                    if protocol_class_names != config_data["classes"]:
                        raise ValueError("Result categories do not match config.classes")

                    provenance = data.get("model_provenance", {})
                    if int(provenance.get("inference_imgsz", -1)) != required_imgsz:
                        raise ValueError(
                            f"model_provenance.inference_imgsz must be {required_imgsz}"
                        )
                    if int(provenance.get("inference_batch_size", -1)) != required_batch_size:
                        raise ValueError(
                            f"model_provenance.inference_batch_size must be {required_batch_size}"
                        )
                    expected_model_cfg = all_models[m_key]
                    for provenance_key in ("family", "model_type"):
                        if provenance.get(provenance_key) != expected_model_cfg.get(provenance_key):
                            raise ValueError(
                                f"model_provenance.{provenance_key} does not match {m_key} config"
                            )
                    if expected_model_cfg.get("model_class") and provenance.get(
                        "model_class"
                    ) != expected_model_cfg.get("model_class"):
                        raise ValueError(f"model_provenance.model_class does not match {m_key} config")
                    checkpoint_hash = str(provenance.get("checkpoint_sha256", ""))
                    if len(checkpoint_hash) != 64 or any(
                        char not in "0123456789abcdef" for char in checkpoint_hash.lower()
                    ):
                        raise ValueError("model_provenance.checkpoint_sha256 is invalid")
                    inference_config = provenance.get("inference_config")
                    if not isinstance(inference_config, dict):
                        raise ValueError("model_provenance.inference_config is missing")
                    inference_config_hash = str(
                        provenance.get("inference_config_sha256", "")
                    )
                    if inference_config_hash != compute_payload_sha256(inference_config):
                        raise ValueError(
                            "model_provenance.inference_config_sha256 does not match its payload"
                        )
                    expected_model_inference_cfg = dict(expected_model_cfg)
                    expected_model_inference_cfg.setdefault(
                        "ignored_checkpoint_classes",
                        config_data.get("ignored_checkpoint_classes", []),
                    )
                    expected_requested_config = build_requested_inference_config(
                        expected_model_inference_cfg,
                        bench_cfg,
                        config_data["classes"],
                    )
                    if inference_config.get("requested") != expected_requested_config:
                        raise ValueError(
                            "model_provenance.inference_config.requested does not match "
                            f"the current {m_key} configuration"
                        )
                    if m_key in collected_results:
                        raise ValueError(f"Duplicate result for model_key '{m_key}'")
                    protocol_ids.add(protocol_id)
                    collected_results[m_key] = data
                    print(f"  [NẠP] Đã nạp kết quả mô hình: {data.get('model_name', m_key)}")
            except Exception as e:
                raise ValueError(f"Invalid result file {fn}: {e}") from e

    if len(protocol_ids) > 1:
        raise ValueError(
            "Refusing to merge results created by different evaluator/dataset protocols: "
            + ", ".join(sorted(protocol_ids))
        )

    # Hiển thị đủ danh mục 8 kiến trúc; model chưa được bật/chưa có kết quả được
    # đánh dấu là đang chờ, tuyệt đối không giả lập metric cho model đó.
    master_results_list = []
    for model_key, model_cfg in all_models.items():
        if model_key in collected_results:
            master_results_list.append(collected_results[model_key])
        else:
            master_results_list.append({
                "model_key": model_key,
                "model_name": model_cfg.get("display_name", model_key),
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0,
                "mAP50": 0.0,
                "mAP50_95": 0.0,
                "mAP75": 0.0,
                "AR100": 0.0,
                "Patience": model_cfg.get("patience"),
                "GFLOPs": 0.0,
                "Parameters": 0.0,
                "Latency": 0.0,
                "FPS": 0.0,
                "status": "Chờ kết quả đánh giá",
                "per_class": {},
            })

    # Lấy danh mục 32 lớp từ cấu hình
    class_names = config_data.get("classes", [
        "beef", "bellpepper", "bittergourd", "bottlegourd", "broccoli", "cabbage",
        "carrot", "cauliflower", "chayote", "chicken", "chickenegg", "chickenleg",
        "chickenwin", "corn", "cucumber", "duckegg", "eggplant", "garlic", "ginger",
        "jicama", "okra", "onion", "pork", "potato", "pumpkin", "radish", "scallion",
        "shrimp", "spongegourd", "sweetpotato", "tofu", "tomato"
    ])
    dummy_categories = [{"id": i+1, "name": n} for i, n in enumerate(class_names)]

    chart_dpi = int(config_data.get("benchmark_settings", {}).get("chart_dpi", 300))
    ScientificReportVisualizer.generate_all_reports(
        master_results_list, dummy_categories, output_dir, chart_dpi=chart_dpi
    )
    print(f"\n[THÀNH CÔNG] Đã hợp nhất xong kết quả toàn đội vào: {os.path.abspath(output_dir)}")


# ==============================================================================
# PHẦN 8: MODAL CLOUD ENTRYPOINTS & LOCAL CLI
# ==============================================================================
@app.function(
    image=image,
    gpu=MODAL_GPU,
    volumes={"/data": volume},
    timeout=MODAL_TIMEOUT
)
def evaluate_models_cloud(model_names: Optional[List[str]] = None, config_dict: Optional[dict] = None):
    """Hàm chạy trên Modal Cloud cho các mô hình được chọn."""
    cfg = config_dict or load_config()
    m_cfg = cfg.get("modal_settings", {})
    cloud_dataset = m_cfg.get("cloud_dataset_dir", "/data/dataset/Completed-Project-5")
    cloud_output_root = m_cfg.get(
        "cloud_output_dir", "/data/runs/Common_COCO_Evaluation_Results"
    )
    run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    cloud_output = os.path.join(cloud_output_root, run_id)

    execute_common_coco_evaluation(
        dataset_dir=cloud_dataset,
        output_dir=cloud_output,
        target_models=model_names,
        config_data=cfg,
        device="cuda:0"
    )

    files_data = {}
    if os.path.exists(cloud_output):
        for fname in os.listdir(cloud_output):
            fp = os.path.join(cloud_output, fname)
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    files_data[fname] = f.read()

    volume.commit()
    return files_data


@app.local_entrypoint()
def evaluate_my_models():
    """
    Lệnh CLI chạy các mô hình có enabled=true trên Modal Cloud:
    modal run common_evaluate.py::evaluate_my_models
    """
    local_cfg = load_config()
    api_key = get_roboflow_api_key(local_cfg)
    if "roboflow" not in local_cfg:
        local_cfg["roboflow"] = {}
    local_cfg["roboflow"]["api_key"] = api_key

    bench_cfg = local_cfg.get("benchmark_settings", {})
    sub_dir = bench_cfg.get("local_output_dir", "Results").split("/")[-1]
    local_out = os.path.join(CURRENT_DIR, sub_dir)

    print(f"[CLOUD] Bắt đầu đánh giá các mô hình đã bật trên Modal Cloud ({MODAL_GPU})...")
    files = evaluate_models_cloud.remote(config_dict=local_cfg)

    os.makedirs(local_out, exist_ok=True)
    for fname, data in files.items():
        out_p = os.path.join(local_out, fname)
        with open(out_p, "wb") as f:
            f.write(data)
        print(f"  [SAVED] {out_p}")

    print(f"\n[SUCCESS] Hoàn tất! Đã lưu kết quả tại: {os.path.abspath(local_out)}")


@app.local_entrypoint()
def evaluate_single(model_name: str):
    """
    Lệnh CLI đánh giá 1 mô hình cụ thể:
    modal run common_evaluate.py::evaluate_single --model-name YOLOv26X
    """
    local_cfg = load_config()
    api_key = get_roboflow_api_key(local_cfg)
    if "roboflow" not in local_cfg:
        local_cfg["roboflow"] = {}
    local_cfg["roboflow"]["api_key"] = api_key

    bench_cfg = local_cfg.get("benchmark_settings", {})
    sub_dir = bench_cfg.get("local_output_dir", "Results").split("/")[-1]
    local_out = os.path.join(CURRENT_DIR, sub_dir)

    print(f"[CLOUD] Đang đánh giá mô hình '{model_name}' trên GPU {MODAL_GPU}...")
    files = evaluate_models_cloud.remote(model_names=[model_name], config_dict=local_cfg)

    os.makedirs(local_out, exist_ok=True)
    for fname, data in files.items():
        out_p = os.path.join(local_out, fname)
        with open(out_p, "wb") as f:
            f.write(data)
        print(f"  [SAVED] {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hệ thống đánh giá chuẩn COCO Benchmark cho các mô hình Object Detection.")
    parser.add_argument("--data-dir", type=str, default="", help="Đường dẫn thư mục dataset COCO.")
    parser.add_argument("--output-dir", type=str, default=os.path.join(CURRENT_DIR, "Results"), help="Thư mục lưu báo cáo.")
    parser.add_argument("--models", type=str, default="", help="Danh sách mô hình cần đánh giá, cách nhau bởi dấu phẩy (vd: YOLOv26X,FasterRCNN_ResNet50).")
    parser.add_argument("--merge-dir", type=str, default="", help="Đường dẫn thư mục chứa các file result_*.json để hợp nhất toàn đội.")
    parser.add_argument("--device", type=str, default=default_compute_device(), help="Thiết bị GPU/CPU.")
    args = parser.parse_args()

    # Nếu gọi lệnh hợp nhất toàn đội
    if args.merge_dir:
        merge_team_results(results_dir=args.merge_dir, output_dir=args.output_dir)
    else:
        target_list = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
        execute_common_coco_evaluation(
            dataset_dir=args.data_dir if args.data_dir else None,
            output_dir=args.output_dir,
            target_models=target_list,
            device=args.device
        )
