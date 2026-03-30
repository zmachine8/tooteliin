import os
from unittest import result
import cv2
import csv
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from dynamsoft_barcode_reader_bundle import *

# =========================
# KONFIGURATSIOON
# =========================
MODE = "rtsp"  # "rtsp" või "images"
TOOTE_NIMI = "kalkun"
STREAM_URL = "rtsp://172.17.37.81:8554/" + TOOTE_NIMI
SHARED_CAPTURE_FOLDER = "captures"  # kasutatakse ainult MODE="images" korral
TEST_IMAGES_FOLDER = SHARED_CAPTURE_FOLDER
TEST_IMAGE_PREFIX = "capture_"

# Dynamsoft
DYNAMSOFT_LICENSE = "t0088YQEAACNxJmkf8GttAqbAp6SwlzBDDmGyqS+wr7cKNFZA60wxkoMTAEVucd4B5oz5RrBs9qmv9rznWBwM6hEuifMcw0O0H0z/aOJuGt6bVY2tltgAycZJgQ=="
TEMPLATE_PATH = None  # määratakse BASE_DIR põhjal allpool
PRODUCT_DB_PATH = None
EXPECTED_DATES_PATH = None  # määratakse BASE_DIR põhjal allpool

# Kuupäev laaditakse hiljem expected_dates.json failist
CAPTURE_DATE_STR = None

EXPECTED_DATES = {}

# Liikumistuvastus
MOTION_THRESHOLD = 18.0
STABLE_THRESHOLD = 3.5
CAPTURE_DELAY = 2.5
STABLE_TIME_REQUIRED = 0.7
FRAME_SLEEP = 0.02

# Rohelise ekraani tuvastus
GREEN_G_MIN = 200
GREEN_BR_MAX = 50

# Seminar 5 debug-salvestus
DEBUG_MODE = True
BARCODE_SLOT_MARGIN_PX = 80
BARCODE_ASSIGN_NEAREST_FALLBACK = True

# OCR
OCR_VARIANT_NAMES = ["orig", "clahe", "sharp", "otsu", "adaptive"]
DATE_UPSCALE_TARGET_WIDTH = 240
DATE_UPSCALE_FACTOR = 2.0
PADDLE_BATCH_SIZE = 16

# DINOv2 tootetuvastus / sildi-sarnasuse analüüs
ENABLE_DINOV2_PRODUCT_ANALYSIS = True
DINOV2_MODEL_NAME = "facebook/dinov2-small"
DINOV2_REFERENCE_ROOT = None  # määratakse BASE_DIR põhjal
DINOV2_DEVICE = "auto"
DINOV2_BATCH_SIZE = 8
DINOV2_POSITIVE_AREAS = ["label1", "label2", "product_area"]
DINOV2_EMPTY_CLASS_BY_AREA = {
    "label1": "empty_label1",
    "label2": "empty_label2",
    "product_area": "empty_product_area",
}
DINOV2_SAVE_QUERY_PRODUCT_AREA = True

BASE_DIR = Path(__file__).resolve().parent
ROOT_HELPERS_PATH = BASE_DIR / "helpers.py"
TEMPLATE_PATH = str(BASE_DIR / "minimal_template_fullframe.json")
PRODUCT_DB_PATH = str(BASE_DIR / "barcode_data.json")
EXPECTED_DATES_PATH = str(BASE_DIR / "expected_dates.json")
DINOV2_REFERENCE_ROOT = str(BASE_DIR / "sildid_ja_toode")

# Laadi kuupäevad ja määra CAPTURE_DATE_STR
if os.path.exists(EXPECTED_DATES_PATH):
    with open(EXPECTED_DATES_PATH, "r", encoding="utf-8") as f:
        EXPECTED_DATES = json.load(f)
    # Otsi kuupäeva TOOTE_NIMI järgi
    CAPTURE_DATE_STR = EXPECTED_DATES.get(TOOTE_NIMI, "01.01.2026")  # vaikimisi kuupäev fallback-iks
else:
    EXPECTED_DATES = {}
    CAPTURE_DATE_STR = "01.01.2026"


# =========================
# ABI
# =========================
def load_root_helpers_module():
    spec = importlib.util.spec_from_file_location("project_helpers", ROOT_HELPERS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


helpers = load_root_helpers_module()
get_formatted_date = helpers.get_formatted_date


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_takt_image_name(takt_index: int) -> str:
    return f"takt_{takt_index:03d}.png"


def is_image_file(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")) and not os.path.basename(name).startswith("._")


def normalize_class_name_for_stream(name: str) -> str:
    value = normalize_product_key(name)
    if value.endswith("_product_area"):
        value = value[: -len("_product_area")]
    return value


def summarize_ms(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    arr = np.array(values, dtype=float)
    return {
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def is_green_screen(frame) -> bool:
    if frame is None:
        return False
    small = cv2.resize(frame, (64, 64))
    avg_color = np.mean(small, axis=(0, 1))
    return avg_color[1] > GREEN_G_MIN and avg_color[0] < GREEN_BR_MAX and avg_color[2] < GREEN_BR_MAX


def measure_change(f1, f2):
    t0 = time.perf_counter()
    if f1 is None or f2 is None:
        return 0.0, 0.0
    small1 = cv2.resize(f1, (160, 120))
    small2 = cv2.resize(f2, (160, 120))
    gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
    score = float(np.mean(cv2.absdiff(gray1, gray2)))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return score, elapsed_ms


def clamp_roi(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def safe_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clamp_roi(x1, y1, x2, y2, w, h)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return img[y1:y2, x1:x2].copy()


def normalize_product_key(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def infer_stream_product_key(stream_url: str) -> str:
    return normalize_product_key(stream_url.rstrip("/").split("/")[-1])


def _stream_key_aliases(stream_key: str) -> List[str]:
    stream_key = normalize_product_key(stream_key)
    alias_map = {
        "salami": ["salami", "salaami", "keedusalaami"],
        "veis": ["veis", "veise", "veiseliha"],
        "kalkun": ["kalkun", "kalkuni"],
        "rulaad": ["rulaad", "rulaadi"],
    }
    aliases = alias_map.get(stream_key, [stream_key])
    return [a for a in aliases if a]


def find_default_product_entry(product_db: Dict[str, Any], stream_key: str) -> Optional[Dict[str, Any]]:
    stream_key = normalize_product_key(stream_key)
    if not stream_key:
        return None

    aliases = _stream_key_aliases(stream_key)

    for ean, product in product_db.items():
        name = normalize_product_key(product.get("ITEMNAME") or product.get("name"))
        if any(alias in name for alias in aliases):
            context = product.copy()
            context["_ean"] = ean
            return context
    return None


def build_center_barcode_crops(frame: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    if frame is None or frame.size == 0:
        return []

    h, w = frame.shape[:2]
    regions = [
        ("center_wide", (0.18, 0.16, 0.82, 0.90)),
        ("center_tight", (0.28, 0.16, 0.72, 0.90)),
        ("middle_band", (0.08, 0.26, 0.92, 0.74)),
    ]

    out: List[Tuple[str, np.ndarray]] = []
    for name, (rx1, ry1, rx2, ry2) in regions:
        crop = safe_crop(
            frame,
            int(w * rx1),
            int(h * ry1),
            int(w * rx2),
            int(h * ry2),
        )
        if crop is not None and crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
            out.append((name, crop))
    return out


def build_roi_barcode_crops(frame: np.ndarray, product_cfg: Optional[Dict[str, Any]]) -> List[Tuple[str, np.ndarray]]:
    if frame is None or frame.size == 0 or not product_cfg or "rois" not in product_cfg:
        return []

    out: List[Tuple[str, np.ndarray]] = []
    rois = product_cfg.get("rois") or {}
    for package_name in sorted(rois.keys()):
        ((x1, y1), (x2, y2)) = rois[package_name]
        crop = safe_crop(frame, int(x1), int(y1), int(x2), int(y2))
        if crop is not None and crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
            out.append((f"roi_{package_name}", crop))
    return out


def build_product_barcode_candidates(package_img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    if package_img is None or package_img.size == 0:
        return []

    h, w = package_img.shape[:2]
    candidates: List[Tuple[str, np.ndarray]] = []

    def add(name: str, img: np.ndarray) -> None:
        if img is None or img.size == 0:
            return
        ih, iw = img.shape[:2]
        if ih < 20 or iw < 20:
            return
        candidates.append((name, img))

    # full package in multiple orientations
    add("product_full", package_img)
    add("product_rot90_cw", cv2.rotate(package_img, cv2.ROTATE_90_CLOCKWISE))
    add("product_rot90_ccw", cv2.rotate(package_img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    add("product_rot180", cv2.rotate(package_img, cv2.ROTATE_180))

    # tighter likely barcode bands from the full product image
    regions = [
        ("bottom_band", (0.08, 0.68, 0.92, 0.98)),
        ("top_band", (0.08, 0.02, 0.92, 0.32)),
        ("middle_band", (0.06, 0.34, 0.94, 0.66)),
        ("center_tight", (0.18, 0.18, 0.82, 0.82)),
    ]
    for base_name, (rx1, ry1, rx2, ry2) in regions:
        crop = safe_crop(package_img, int(w * rx1), int(h * ry1), int(w * rx2), int(h * ry2))
        add(base_name, crop)
        add(f"{base_name}_rot90_cw", cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE))
        add(f"{base_name}_rot90_ccw", cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE))

    return candidates


def resolve_expected_date(product_name: str) -> str:
    product_name = normalize_product_key(product_name)
    if product_name in EXPECTED_DATES:
        return EXPECTED_DATES[product_name]

    for key, value in EXPECTED_DATES.items():
        if key in product_name:
            return value

    return CAPTURE_DATE_STR


class RTSPStreamReader:
    def __init__(self, url: str):
        self.cap = cv2.VideoCapture(url)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not self.running:
                break
            with self.lock:
                self.ret = ret
                self.frame = frame
            if not ret:
                break

    def read(self):
        with self.lock:
            if self.frame is None:
                return self.ret, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class BarcodeProductReader:
    def __init__(self, license_key: str, template_path: str, product_db_path: str, capture_date_str: str, stream_product_key: str = ""):
        LicenseManager.init_license(license_key)
        self.router = CaptureVisionRouter()

        self.template_name = "ReadBarcodes_Default"
        self.template_loaded = False
        if os.path.exists(template_path):
            err_code, err_msg = self.router.init_settings_from_file(template_path)
            if err_code == EnumErrorCode.EC_OK:
                self.template_loaded = True
            else:
                print(f"Hoiatus: JSON seadeid ei õnnestunud laadida ({err_msg}). Kasutan default seadeid.")
        else:
            print(f"Hoiatus: {template_path} puudub. Kasutan default seadeid.")

        if not os.path.exists(product_db_path):
            raise FileNotFoundError(f"Tooteandmebaasi ei leitud: {product_db_path}")

        with open(product_db_path, "r", encoding="utf-8") as f:
            self.product_db = json.load(f)

        self.capture_date = datetime.strptime(capture_date_str, "%d.%m.%Y")
        self.stream_product_key = normalize_product_key(stream_product_key)
        self.default_product = find_default_product_entry(self.product_db, self.stream_product_key)

    def lookup_product(self, ean: str):
        product = self.product_db.get(ean)
        if not product:
            return None

        name = product.get("name") or product.get("ITEMNAME") or "Tundmatu toode"
        expiry_duration = product.get("expiry_duration")
        if expiry_duration is None:
            expiry_duration = product.get("BESTBEFOREDAYS")

        if expiry_duration is None:
            expiry_date_str = "Puudub"
        else:
            expiry_date = self.capture_date + timedelta(days=int(expiry_duration))
            expiry_date_str = expiry_date.strftime("%d.%m.%Y")

        return {
            "name": name,
            "expiry_duration": expiry_duration,
            "expiry_date_str": expiry_date_str,
        }

    def _parse_capture_result(self, result, source_name: str, elapsed_ms: float, frame_width: Optional[int] = None) -> Dict[str, Any]:
        items = result.get_items() if result is not None else None
        barcodes = []

        if items:
            for item in items:
                if item.get_type() == EnumCapturedResultItemType.CRIT_BARCODE:
                    ean = item.get_text()
                    product_info = self.lookup_product(ean)
                    points = _extract_points_from_item(item)

                    barcodes.append({
                        "ean": ean,
                        "product": product_info,
                        "points": points,
                    })

        if frame_width is not None and barcodes:
            barcodes = sort_barcodes_center_to_right(barcodes, frame_width)

        return {
            "image_path": source_name,
            "elapsed_ms": elapsed_ms,
            "barcodes": barcodes,
        }

    def _capture_crop(self, image_or_path, source_name: str, crop_name: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        result = self.router.capture(image_or_path, self.template_name)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        parsed = self._parse_capture_result(result, source_name, elapsed_ms)
        parsed["source_crop"] = crop_name
        for bc in parsed["barcodes"]:
            bc["source_crop"] = crop_name
        return parsed

    def _choose_best_barcodes(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        def score_bc(bc: Dict[str, Any]) -> Tuple[int, int]:
            product = bc.get("product")
            ean = bc.get("ean", "")
            known = 1 if product else 0
            stream_match = 0
            if product:
                name = normalize_product_key(product.get("name"))
                if self.stream_product_key and self.stream_product_key in name:
                    stream_match = 1
            return (stream_match, known)

        dedup: Dict[str, Dict[str, Any]] = {}
        ordered: List[Dict[str, Any]] = []
        for bc in candidates:
            ean = bc.get("ean")
            if not ean:
                continue
            if ean not in dedup:
                dedup[ean] = bc
                ordered.append(bc)
                continue
            if score_bc(bc) > score_bc(dedup[ean]):
                dedup[ean] = bc

        ordered = [dedup[bc["ean"]] for bc in ordered if bc.get("ean") in dedup]
        ordered.sort(key=lambda bc: score_bc(bc), reverse=True)
        return ordered

    def read_image(self, image_path: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        result = self.router.capture(image_path, self.template_name)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._parse_capture_result(result, image_path, elapsed_ms)

    def read_frame(self, frame: np.ndarray, source_name: str = "<frame>") -> Dict[str, Any]:
        t0 = time.perf_counter()
        result = self.router.capture(frame, self.template_name)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        frame_width = frame.shape[1] if frame is not None else None
        return self._parse_capture_result(result, source_name, elapsed_ms, frame_width=frame_width)

    def read_barcodes_from_crops(self, crops: List[np.ndarray], source_name: str, crop_prefix: str = "label1") -> Dict[str, Any]:
        slot_barcodes: List[Dict[str, Any]] = []
        elapsed_values: List[float] = []
        slot_elapsed_ms: Dict[int, float] = {}
        for idx, crop in enumerate(crops, start=1):
            if crop is None or crop.size == 0:
                continue

            crop_name = f"{crop_prefix}_s{idx}"
            slot_candidates: List[Dict[str, Any]] = []
            candidate_images = build_product_barcode_candidates(crop) if crop_prefix == "product" else [(crop_name, crop)]

            for candidate_name, candidate_img in candidate_images:
                parsed = self._capture_crop(candidate_img, source_name, candidate_name)
                elapsed_values.append(parsed["elapsed_ms"])
                slot_elapsed_ms[idx] = slot_elapsed_ms.get(idx, 0.0) + parsed["elapsed_ms"]

                for bc in parsed["barcodes"]:
                    bc["source_crop"] = candidate_name
                    bc["slot_index"] = idx
                    slot_candidates.append(bc)

            best_for_slot = self._choose_best_barcodes(slot_candidates)
            if best_for_slot:
                chosen = best_for_slot[0]
                chosen["slot_index"] = idx
                chosen["slot_elapsed_ms"] = slot_elapsed_ms.get(idx, 0.0)
                slot_barcodes.append(chosen)

        total_elapsed_ms = float(sum(elapsed_values)) if elapsed_values else 0.0
        return {
            "image_path": source_name,
            "elapsed_ms": total_elapsed_ms,
            "barcodes": slot_barcodes,
            "source_crop": crop_prefix,
            "slot_elapsed_ms": slot_elapsed_ms,
        }


@dataclass
class GallerySample:
    class_name: str
    image_path: str
    embedding: np.ndarray


class Dinov2ProductAnalyzer:
    def __init__(self, reference_root: str, model_name: str, batch_size: int = 8, device: str = "auto"):
        self.reference_root = reference_root
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.device_mode = device
        self.enabled = False
        self.error_message = ""
        self.device = "cpu"
        self.samples_by_area: Dict[str, List[GallerySample]] = defaultdict(list)
        self.class_centroids_by_area: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
        self.calibration_by_area: Dict[str, Dict[str, float]] = {}
        self.reference_counts_by_area: Dict[str, Dict[str, int]] = {}

        try:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModel

            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

            self.torch = torch
            self.Image = Image
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            if device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = device
            self.model.to(self.device)
            self.model.eval()

            self._load_reference_gallery()
            self.enabled = any(self.samples_by_area.values())
            if not self.enabled:
                self.error_message = f"Viitegaleriid ei leitud asukohast: {reference_root}"
        except Exception as e:
            self.error_message = f"DINOv2 initsialiseerimine ebaõnnestus: {e}"

    def _pil_from_bgr(self, image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.Image.fromarray(rgb)

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        embedding = embedding.astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-12:
            return embedding
        return embedding / norm

    def embed_images(self, images_bgr: List[np.ndarray]) -> List[np.ndarray]:
        if not images_bgr:
            return []
        outputs: List[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, len(images_bgr), self.batch_size):
                batch = images_bgr[start:start + self.batch_size]
                pil_images = [self._pil_from_bgr(img) for img in batch]
                inputs = self.processor(images=pil_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                model_out = self.model(**inputs)
                if hasattr(model_out, "pooler_output") and model_out.pooler_output is not None:
                    batch_emb = model_out.pooler_output
                else:
                    batch_emb = model_out.last_hidden_state[:, 0]
                batch_emb = batch_emb.detach().cpu().numpy()
                for emb in batch_emb:
                    outputs.append(self._normalize_embedding(emb))
        return outputs

    def _iter_reference_images(self) -> List[Tuple[str, str, str]]:
        rows = []
        if not os.path.isdir(self.reference_root):
            return rows

        for entry in sorted(os.listdir(self.reference_root)):
            if entry.startswith('.'):
                continue
            entry_path = os.path.join(self.reference_root, entry)
            if not os.path.isdir(entry_path):
                continue

            matched_empty_area = None
            for area_name, empty_class_name in DINOV2_EMPTY_CLASS_BY_AREA.items():
                if entry == empty_class_name:
                    matched_empty_area = area_name
                    break

            if matched_empty_area is not None:
                for filename in sorted(os.listdir(entry_path)):
                    if is_image_file(filename):
                        rows.append((matched_empty_area, entry, os.path.join(entry_path, filename)))
                continue

            for area in DINOV2_POSITIVE_AREAS:
                area_path = os.path.join(entry_path, area)
                if not os.path.isdir(area_path):
                    continue
                for filename in sorted(os.listdir(area_path)):
                    if is_image_file(filename):
                        rows.append((area, entry, os.path.join(area_path, filename)))
        return rows

    def _load_reference_gallery(self) -> None:
        rows = self._iter_reference_images()
        if not rows:
            return

        decoded_rows = []
        loaded_images = []
        for area, class_name, img_path in rows:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue
            decoded_rows.append((area, class_name, img_path))
            loaded_images.append(img)

        embeddings = self.embed_images(loaded_images)
        for (area, class_name, img_path), emb in zip(decoded_rows, embeddings):
            self.samples_by_area[area].append(GallerySample(class_name=class_name, image_path=img_path, embedding=emb))

        for area, samples in self.samples_by_area.items():
            counts = Counter(s.class_name for s in samples)
            self.reference_counts_by_area[area] = dict(counts)
            for class_name in counts:
                class_embs = np.stack([s.embedding for s in samples if s.class_name == class_name], axis=0)
                centroid = np.mean(class_embs, axis=0)
                self.class_centroids_by_area[area][class_name] = self._normalize_embedding(centroid)

        self._calibrate()

    def _cosine_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(1.0 - float(np.dot(emb1, emb2)))

    def _distances_to_class(self, embedding: np.ndarray, area: str, class_name: str) -> List[float]:
        samples = [s for s in self.samples_by_area.get(area, []) if s.class_name == class_name]
        return [self._cosine_distance(embedding, s.embedding) for s in samples]

    def _calibrate(self) -> None:
        for area, samples in self.samples_by_area.items():
            empty_class_name = DINOV2_EMPTY_CLASS_BY_AREA.get(area, "__missing_empty__")
            positive_classes = [c for c in self.reference_counts_by_area.get(area, {}) if c != empty_class_name]
            intra_positive = []
            positive_vs_empty = []
            class_nn_correct = 0
            class_nn_total = 0

            for idx, sample in enumerate(samples):
                other_samples = [s for j, s in enumerate(samples) if j != idx]
                if not other_samples:
                    continue
                distances = sorted((self._cosine_distance(sample.embedding, s.embedding), s.class_name) for s in other_samples)
                nearest_dist, nearest_class = distances[0]
                if sample.class_name != empty_class_name:
                    class_nn_total += 1
                    if nearest_class == sample.class_name:
                        class_nn_correct += 1
                for dist, other_class in distances:
                    if other_class == sample.class_name and sample.class_name != empty_class_name:
                        intra_positive.append(dist)
                        break
                if sample.class_name != empty_class_name and empty_class_name in self.reference_counts_by_area.get(area, {}):
                    empty_dists = [self._cosine_distance(sample.embedding, s.embedding) for s in samples if s.class_name == empty_class_name]
                    if empty_dists:
                        positive_vs_empty.append(min(empty_dists))

            threshold = None
            if intra_positive and positive_vs_empty:
                threshold = (float(np.median(intra_positive)) + float(np.median(positive_vs_empty))) / 2.0

            self.calibration_by_area[area] = {
                "intra_positive_median": float(np.median(intra_positive)) if intra_positive else np.nan,
                "positive_vs_empty_median": float(np.median(positive_vs_empty)) if positive_vs_empty else np.nan,
                "empty_threshold": threshold if threshold is not None else np.nan,
                "leave_one_out_accuracy": (class_nn_correct / class_nn_total) if class_nn_total else np.nan,
                "positive_ref_count": float(sum(self.reference_counts_by_area.get(area, {}).get(c, 0) for c in positive_classes)),
                "empty_ref_count": float(self.reference_counts_by_area.get(area, {}).get(empty_class_name, 0)),
            }

    def classify_crop(self, crop_bgr: np.ndarray, area: str = "product_area") -> Dict[str, Any]:
        if not self.enabled or crop_bgr is None or crop_bgr.size == 0:
            return {
                "predicted_product": "",
                "predicted_product_nearest": "",
                "predicted_product_centroid": "",
                "decision_method": "",
                "nearest_distance": None,
                "nearest_image": "",
                "nearest_class": "",
                "empty_distance": None,
                "is_empty_like": None,
                "area": area,
            }

        emb = self.embed_images([crop_bgr])[0]
        area_samples = self.samples_by_area.get(area, [])
        if not area_samples:
            return {
                "predicted_product": "",
                "predicted_product_nearest": "",
                "predicted_product_centroid": "",
                "decision_method": "",
                "nearest_distance": None,
                "nearest_image": "",
                "nearest_class": "",
                "empty_distance": None,
                "is_empty_like": None,
                "area": area,
            }

        distances = sorted((self._cosine_distance(emb, s.embedding), s) for s in area_samples)
        nearest_distance, nearest_sample = distances[0]

        empty_class_name = DINOV2_EMPTY_CLASS_BY_AREA.get(area, "__missing_empty__")
        positive_distances = [(dist, s) for dist, s in distances if s.class_name != empty_class_name]
        empty_distances = [(dist, s) for dist, s in distances if s.class_name == empty_class_name]

        nearest_positive_distance = positive_distances[0][0] if positive_distances else None
        nearest_positive_sample = positive_distances[0][1] if positive_distances else None
        nearest_empty_distance = empty_distances[0][0] if empty_distances else None

        nearest_prediction = nearest_positive_sample.class_name if nearest_positive_sample is not None else nearest_sample.class_name

        threshold = self.calibration_by_area.get(area, {}).get("empty_threshold")
        is_empty_like = None
        if nearest_empty_distance is not None and nearest_positive_distance is not None:
            if threshold == threshold:  # not NaN
                is_empty_like = bool(nearest_empty_distance <= threshold and nearest_empty_distance < nearest_positive_distance)
            else:
                is_empty_like = bool(nearest_empty_distance < nearest_positive_distance)

        centroid_distances = {}
        for class_name, centroid in self.class_centroids_by_area.get(area, {}).items():
            if class_name == empty_class_name:
                continue
            centroid_distances[class_name] = self._cosine_distance(emb, centroid)

        centroid_prediction = min(centroid_distances.items(), key=lambda kv: kv[1])[0] if centroid_distances else nearest_prediction

        # label1 puhul on 1-NN liiga tundlik üksikutele kehvadele näidetele,
        # seega kasutame lõppotsusena centroidi-põhist klassi.
        final_prediction = centroid_prediction if area == "label1" else nearest_prediction
        decision_method = "centroid" if area == "label1" else "nearest"

        return {
            "predicted_product": final_prediction,
            "predicted_product_nearest": nearest_prediction,
            "predicted_product_centroid": centroid_prediction,
            "decision_method": decision_method,
            "nearest_distance": nearest_positive_distance if nearest_positive_distance is not None else nearest_distance,
            "nearest_image": nearest_positive_sample.image_path if nearest_positive_sample is not None else nearest_sample.image_path,
            "nearest_class": nearest_positive_sample.class_name if nearest_positive_sample is not None else nearest_sample.class_name,
            "empty_distance": nearest_empty_distance,
            "is_empty_like": is_empty_like,
            "area": area,
        }

    def save_reference_histograms(self, out_dir: str) -> List[str]:
        saved = []
        os.makedirs(out_dir, exist_ok=True)
        for area, samples in self.samples_by_area.items():
            if area not in DINOV2_POSITIVE_AREAS:
                continue
            empty_class_name = DINOV2_EMPTY_CLASS_BY_AREA.get(area, "__missing_empty__")
            positives = [s for s in samples if s.class_name != empty_class_name]
            empties = [s for s in samples if s.class_name == empty_class_name]
            intra_positive = []
            positive_vs_empty = []
            for idx, sample in enumerate(positives):
                same_class = [s for j, s in enumerate(positives) if j != idx and s.class_name == sample.class_name]
                if same_class:
                    intra_positive.append(min(self._cosine_distance(sample.embedding, s.embedding) for s in same_class))
                if empties:
                    positive_vs_empty.append(min(self._cosine_distance(sample.embedding, s.embedding) for s in empties))
            if not intra_positive and not positive_vs_empty:
                continue
            plt.figure(figsize=(8, 5))
            if intra_positive:
                plt.hist(intra_positive, bins=20, alpha=0.7, label="same-class nearest")
            if positive_vs_empty:
                plt.hist(positive_vs_empty, bins=20, alpha=0.7, label=f"{area} vs empty nearest")
            thr = self.calibration_by_area.get(area, {}).get("empty_threshold")
            if thr == thr:
                plt.axvline(thr, linestyle="--", linewidth=2, label=f"threshold={thr:.4f}")
            plt.xlabel("Cosine distance")
            plt.ylabel("Count")
            plt.title(f"DINOv2 reference distances: {area}")
            plt.legend()
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"dinov2_hist_{area}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            saved.append(out_path)
        return saved

    def write_reference_report(self, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("DINOv2 reference gallery report\n")
            f.write("=============================\n\n")
            f.write(f"Reference root: {self.reference_root}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Device: {self.device}\n\n")
            for area in sorted(self.samples_by_area.keys()):
                f.write(f"Area: {area}\n")
                counts = self.reference_counts_by_area.get(area, {})
                for class_name in sorted(counts.keys()):
                    f.write(f"  {class_name}: {counts[class_name]}\n")
                calib = self.calibration_by_area.get(area, {})
                f.write(f"  leave_one_out_accuracy: {calib.get('leave_one_out_accuracy')}\n")
                f.write(f"  intra_positive_median: {calib.get('intra_positive_median')}\n")
                f.write(f"  positive_vs_empty_median: {calib.get('positive_vs_empty_median')}\n")
                f.write(f"  empty_threshold: {calib.get('empty_threshold')}\n\n")


dinov2_product_analyzer: Optional[Dinov2ProductAnalyzer] = None


def init_dinov2_product_analyzer() -> Optional[Dinov2ProductAnalyzer]:
    global dinov2_product_analyzer
    if not ENABLE_DINOV2_PRODUCT_ANALYSIS:
        return None
    if dinov2_product_analyzer is None:
        dinov2_product_analyzer = Dinov2ProductAnalyzer(
            reference_root=DINOV2_REFERENCE_ROOT,
            model_name=DINOV2_MODEL_NAME,
            batch_size=DINOV2_BATCH_SIZE,
            device=DINOV2_DEVICE,
        )
    return dinov2_product_analyzer


class PaddleDateOCR:
    def __init__(self):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        import paddlex as pdx
        self.model = pdx.create_model("PP-OCRv5_server_rec")
        self.model_name = "PaddleOCR"

    def predict_raw(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        if not images:
            return []
        outputs = []
        for pred in self.model.predict(images):
            text = ""
            score = None
            if isinstance(pred, dict):
                text = str(pred.get("rec_text", "") or "")
                rec_score = pred.get("rec_score")
                if rec_score is not None:
                    try:
                        score = float(rec_score)
                    except Exception:
                        score = None
            else:
                text = str(pred) if pred is not None else ""
            outputs.append({"text": text, "score": score})
        return outputs


paddle_date_ocr: Optional[PaddleDateOCR] = None


def init_paddle_ocr() -> PaddleDateOCR:
    global paddle_date_ocr
    if paddle_date_ocr is None:
        paddle_date_ocr = PaddleDateOCR()
    return paddle_date_ocr


# =========================
# OCR eeltöötlus
# =========================
def upscale_if_needed(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((4, 4, 3), dtype=np.uint8)
    h, w = img_bgr.shape[:2]
    scale = DATE_UPSCALE_FACTOR if w < DATE_UPSCALE_TARGET_WIDTH else 1.0
    if scale == 1.0:
        return img_bgr.copy()
    return cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def reduce_glare(gray: np.ndarray) -> np.ndarray:
    if gray is None or gray.size == 0:
        return gray
    bright_mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)[1]
    if int(np.count_nonzero(bright_mask)) == 0:
        return gray
    kernel = np.ones((3, 3), np.uint8)
    bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)
    return cv2.inpaint(gray, bright_mask, 3, cv2.INPAINT_TELEA)


def build_date_variants(date_crop_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    base = upscale_if_needed(date_crop_bgr)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    gray = reduce_glare(gray)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.2)
    sharp = cv2.addWeighted(clahe, 1.55, blur, -0.55, 0)
    otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    variants = [
        ("orig", cv2.cvtColor(base, cv2.COLOR_BGR2RGB)),
        ("clahe", cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)),
        ("sharp", cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)),
        ("otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)),
        ("adaptive", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)),
    ]
    return variants


def score_candidate(normalized: str, raw_text: str, variant_name: str, expected_date: Optional[str], rec_score: Optional[float], votes: int) -> float:
    score = 0.0
    if normalized and normalized != "NA":
        score += 100.0
    else:
        return score

    if expected_date and normalized == expected_date:
        score += 300.0

    if votes > 1:
        score += votes * 40.0

    digits = sum(ch.isdigit() for ch in (raw_text or ""))
    score += min(digits, 8) * 3.0

    if rec_score is not None:
        score += max(0.0, min(rec_score, 1.0)) * 30.0

    if variant_name == "orig":
        score += 8.0
    elif variant_name == "clahe":
        score += 6.0
    elif variant_name == "sharp":
        score += 5.0
    elif variant_name == "otsu":
        score += 4.0
    elif variant_name == "adaptive":
        score += 3.0

    return score


def choose_best_ocr_candidate(candidates: List[Dict[str, Any]], expected_date: Optional[str]) -> Dict[str, Any]:
    valid_norms = [c["normalized"] for c in candidates if c["normalized"] and c["normalized"] != "NA"]
    vote_counts = Counter(valid_norms)

    best = {
        "date_exists": False,
        "predicted_date": "NA",
        "raw_text": "",
        "variant": "",
        "score": None,
        "all_candidates": candidates,
    }
    best_score = -1.0

    for cand in candidates:
        cand["votes"] = vote_counts.get(cand["normalized"], 0)
        cand_score = score_candidate(
            normalized=cand["normalized"],
            raw_text=cand["raw_text"],
            variant_name=cand["variant"],
            expected_date=expected_date,
            rec_score=cand.get("ocr_score"),
            votes=cand["votes"],
        )
        cand["candidate_score"] = cand_score
        if cand_score > best_score:
            best_score = cand_score
            best = {
                "date_exists": cand["normalized"] not in (None, "", "NA"),
                "predicted_date": cand["normalized"] if cand["normalized"] else "NA",
                "raw_text": cand["raw_text"],
                "variant": cand["variant"],
                "score": cand.get("ocr_score"),
                "all_candidates": candidates,
            }

    return best


def run_date_ocr_for_crops(date_crops: List[np.ndarray], expected_date: Optional[str]) -> Tuple[List[Dict[str, Any]], float]:
    ocr = init_paddle_ocr()
    inputs: List[np.ndarray] = []
    meta: List[Tuple[int, str]] = []

    for crop_index, crop in enumerate(date_crops):
        for variant_name, variant_img in build_date_variants(crop):
            inputs.append(variant_img)
            meta.append((crop_index, variant_name))

    t0 = time.perf_counter()
    predictions = ocr.predict_raw(inputs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    grouped: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(date_crops))}
    for (crop_index, variant_name), pred in zip(meta, predictions):
        raw_text = str(pred.get("text", "") or "").strip()
        normalized = get_formatted_date(raw_text)
        grouped[crop_index].append({
            "variant": variant_name,
            "raw_text": raw_text,
            "normalized": normalized,
            "ocr_score": pred.get("score"),
        })

    best_results = []
    for crop_index in range(len(date_crops)):
        best_results.append(choose_best_ocr_candidate(grouped.get(crop_index, []), expected_date))

    return best_results, elapsed_ms


# =========================
# Raportid
# =========================
def write_results_report(results: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Triipkoodi ja kuupäeva tulemused\n")
        f.write("===========================\n\n")
        for entry in results:
            f.write(f"Pilt: {os.path.basename(entry['image_path'])}\n")
            if entry.get("video_time_s") is not None:
                f.write(f"Aeg videos: {entry['video_time_s']:.3f} s\n")
            f.write(f"Triipkoodi lugemise aeg: {entry['elapsed_ms']:.1f} ms\n")
            if entry.get("ocr_processing_ms") is not None:
                f.write(f"OCR aeg (4 kuupäeva ala): {entry['ocr_processing_ms']:.1f} ms\n")

            barcodes = entry["barcodes"]
            barcode_by_slot = {bc.get("slot_index"): bc for bc in barcodes if bc.get("slot_index") is not None}
            if not barcodes:
                f.write("Triipkood: ei leitud\n")
            else:
                for bc in barcodes:
                    product = bc.get("product")
                    slot_txt = f"S{bc.get('slot_index')}" if bc.get("slot_index") is not None else "S?"
                    crop_txt = bc.get("source_crop", "-")
                    if product:
                        f.write(
                            f"{slot_txt}: EAN13: {bc['ean']} | Toode: {product['name']} | Säilivusaeg: {product['expiry_date_str']} | crop={crop_txt}\n"
                        )
                    else:
                        f.write(f"{slot_txt}: EAN13: {bc['ean']} | Toode: andmebaasist puudub | Säilivusaeg: puudub | crop={crop_txt}\n")

            date_results = entry.get("date_results", [])
            if not date_results:
                f.write("Kuupäevad: ei töödeldud\n\n")
                continue

            for idx, dr in enumerate(date_results, start=1):
                status = "olemas" if dr.get("date_exists") else "puudub"
                predicted = dr.get("predicted_date", "NA")
                expected = dr.get("expected_date", "") or "-"
                variant = dr.get("winning_variant", "") or "-"
                raw_text = dr.get("raw_text", "") or ""
                is_correct = dr.get("is_correct")
                correctness = "oige" if is_correct else ("vale" if dr.get("date_exists") else "-")
                f.write(
                    f"  S{idx}: kuupäev {status} | tulemus={predicted} | oodatav={expected} | staatus={correctness} | variant={variant} | raw={raw_text}\n"
                )

            for result_key, prefix, label_name in [
                ("label1_results", "L1", "label1"),
                ("label2_results", "L2", "label2"),
                ("product_area_results", "P", "product_area"),
            ]:
                area_results = entry.get(result_key, [])
                if area_results:
                    for idx, pr in enumerate(area_results, start=1):
                        pred = pr.get("predicted_product", "") or "-"
                        pred_nearest = pr.get("predicted_product_nearest", "") or "-"
                        centroid_pred = pr.get("predicted_product_centroid", "") or "-"
                        method = pr.get("decision_method", "") or "-"
                        nd = pr.get("nearest_distance")
                        ed = pr.get("empty_distance")
                        nd_txt = f"{nd:.4f}" if nd is not None else "-"
                        ed_txt = f"{ed:.4f}" if ed is not None else "-"
                        empty_like = pr.get("is_empty_like")
                        empty_txt = "jah" if empty_like is True else ("ei" if empty_like is False else "-")
                        nearest_class = pr.get("nearest_class", "") or "-"
                        f.write(
                            f"  {prefix}{idx}: area={label_name} | nn={pred} | centroid={centroid_pred} | nearest_class={nearest_class} | d_pos={nd_txt} | d_empty={ed_txt} | empty_like={empty_txt}\n"
                        )
            f.write("\n")

        stats = build_statistics(results)
        f.write("Statistika\n")
        f.write("==========\n")
        f.write(f"Takte kokku: {stats['total_images']}\n")
        f.write(f"Tuvastatud taktide arv: {stats['detected_images']}\n")
        f.write(f"Triipkoodi tuvastusmäär: {stats['detection_rate_percent']:.2f}%\n")
        f.write(f"Keskmine triipkoode pildi kohta: {stats['avg_barcodes_per_image']:.3f}\n")
        f.write(f"Pakke kokku OCR-is: {stats['date_slots_total']}\n")
        f.write(f"Kuupäev olemas: {stats['date_present_count']}\n")
        f.write(f"Kuupäev puudub: {stats['date_missing_count']}\n")
        f.write(f"Kuupäeva saagis: {stats['date_presence_rate_percent']:.2f}%\n")
        f.write(f"Õige kuupäev: {stats['date_correct_count']}\n")
        f.write(f"Vale kuupäev: {stats['date_wrong_count']}\n")
        f.write(f"Kuupäeva täpsus: {stats['date_accuracy_percent']:.2f}%\n")
        for area in DINOV2_POSITIVE_AREAS:
            f.write(f"DINOv2 {area} slots kokku: {stats[area]['slots_total']}\n")
            f.write(f"DINOv2 {area} empty_like: {stats[area]['empty_like_count']}\n")
            f.write(f"DINOv2 {area} non-empty: {stats[area]['non_empty_count']}\n")
            f.write(f"DINOv2 {area} õige toode: {stats[area]['correct_count']}\n")
            f.write(f"DINOv2 {area} toote täpsus: {stats[area]['accuracy_percent']:.2f}%\n")
        if stats['barcode_reading_time']:
            f.write(f"Triipkoodi keskmine lugemisaeg: {stats['barcode_reading_time']['mean_ms']:.2f} ms\n")
            f.write(f"Triipkoodi maksimaalne lugemisaeg: {stats['barcode_reading_time']['max_ms']:.2f} ms\n")
        if stats['ocr_processing_time']:
            f.write(f"OCR keskmine aeg takti kohta: {stats['ocr_processing_time']['mean_ms']:.2f} ms\n")
            f.write(f"OCR maksimaalne aeg takti kohta: {stats['ocr_processing_time']['max_ms']:.2f} ms\n")


def _csv_area_fields(prefix: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        f"{prefix}_predicted_product": result.get("predicted_product", ""),
        f"{prefix}_predicted_product_nearest": result.get("predicted_product_nearest", ""),
        f"{prefix}_predicted_product_centroid": result.get("predicted_product_centroid", ""),
        f"{prefix}_decision_method": result.get("decision_method", ""),
        f"{prefix}_nearest_class": result.get("nearest_class", ""),
        f"{prefix}_nearest_distance": f"{result.get('nearest_distance'):.4f}" if result.get("nearest_distance") is not None else "",
        f"{prefix}_empty_distance": f"{result.get('empty_distance'):.4f}" if result.get("empty_distance") is not None else "",
        f"{prefix}_is_empty_like": result.get("is_empty_like", ""),
    }



def write_ocr_csv_report(results: List[Dict[str, Any]], path: str) -> None:
    headers = [
        "capture_file", "video_time_s", "slot", "slot_product_image", "barcode_found", "ean", "all_eans", "product_name", "source_crop",
        "expected_date", "date_exists", "predicted_date", "is_correct", "winning_variant",
        "winning_raw_text", "barcode_elapsed_ms", "ocr_processing_ms",
        "label1_predicted_product", "label1_predicted_product_nearest", "label1_predicted_product_centroid", "label1_decision_method", "label1_nearest_class", "label1_nearest_distance", "label1_empty_distance", "label1_is_empty_like",
        "label2_predicted_product", "label2_predicted_product_nearest", "label2_predicted_product_centroid", "label2_decision_method", "label2_nearest_class", "label2_nearest_distance", "label2_empty_distance", "label2_is_empty_like",
        "product_predicted_product", "product_predicted_product_nearest", "product_predicted_product_centroid", "product_decision_method", "product_nearest_class", "product_nearest_distance", "product_empty_distance", "product_is_empty_like"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for entry in results:
            barcode_list = entry.get("barcodes", [])
            barcode_eans = [bc.get("ean", "") for bc in barcode_list if bc.get("ean")]
            barcode_eans_str = ",".join(barcode_eans)

            first_barcode = barcode_list[0] if barcode_list else {}
            product = first_barcode.get("product") or {}
            date_results = entry.get("date_results") or []
            if not date_results:
                writer.writerow({
                    "capture_file": os.path.basename(entry["image_path"]),
                    "video_time_s": f"{entry.get('video_time_s', 0.0):.3f}" if entry.get("video_time_s") is not None else "",
                    "slot": "-",
                    "slot_product_image": "",
                    "barcode_found": bool(entry.get("barcodes")),
                    "ean": first_barcode.get("ean", ""),
                    "all_eans": barcode_eans_str,
                    "product_name": product.get("name", ""),
                    "source_crop": first_barcode.get("source_crop", entry.get("source_crop", "")),
                    "expected_date": "",
                    "date_exists": False,
                    "predicted_date": "NA",
                    "is_correct": "",
                    "winning_variant": "",
                    "winning_raw_text": "",
                    "barcode_elapsed_ms": f"{entry.get('elapsed_ms', 0.0):.2f}",
                    "ocr_processing_ms": f"{entry.get('ocr_processing_ms', 0.0):.2f}" if entry.get("ocr_processing_ms") is not None else "",
                    **_csv_area_fields("label1", {}),
                    **_csv_area_fields("label2", {}),
                    **_csv_area_fields("product", {}),
                })
                continue

            label1_results = entry.get("label1_results") or []
            label2_results = entry.get("label2_results") or []
            product_results = entry.get("product_area_results") or []
            barcode_by_slot = {bc.get("slot_index"): bc for bc in barcode_list if bc.get("slot_index") is not None}
            for idx, dr in enumerate(date_results, start=1):
                l1 = label1_results[idx - 1] if idx - 1 < len(label1_results) else {}
                l2 = label2_results[idx - 1] if idx - 1 < len(label2_results) else {}
                pr = product_results[idx - 1] if idx - 1 < len(product_results) else {}
                slot_bc = barcode_by_slot.get(idx, {})
                slot_product = slot_bc.get("product") or {}
                writer.writerow({
                    "capture_file": os.path.basename(entry["image_path"]),
                    "video_time_s": f"{entry.get('video_time_s', 0.0):.3f}" if entry.get("video_time_s") is not None else "",
                    "slot": idx,
                    "slot_product_image": f"takt_{entry.get('takt_index', 0):03d}_s{idx}_product.png",
                    "barcode_found": bool(slot_bc),
                    "ean": slot_bc.get("ean", ""),
                    "all_eans": barcode_eans_str,
                    "product_name": slot_product.get("name", ""),
                    "source_crop": slot_bc.get("source_crop", entry.get("source_crop", "")),
                    "expected_date": dr.get("expected_date", ""),
                    "date_exists": dr.get("date_exists", False),
                    "predicted_date": dr.get("predicted_date", "NA"),
                    "is_correct": dr.get("is_correct", ""),
                    "winning_variant": dr.get("winning_variant", ""),
                    "winning_raw_text": dr.get("raw_text", ""),
                    "barcode_elapsed_ms": f"{slot_bc.get('slot_elapsed_ms', entry.get('elapsed_ms', 0.0)):.2f}",
                    "ocr_processing_ms": f"{entry.get('ocr_processing_ms', 0.0):.2f}" if entry.get("ocr_processing_ms") is not None else "",
                    **_csv_area_fields("label1", l1),
                    **_csv_area_fields("label2", l2),
                    **_csv_area_fields("product", pr),
                })



def _build_area_stats(results: List[Dict[str, Any]], result_key: str, expected_stream_key: str) -> Dict[str, Any]:
    area_slots = [pr for r in results for pr in r.get(result_key, [])]
    slots_total = len(area_slots)
    empty_like_count = sum(1 for p in area_slots if p.get("is_empty_like") is True)
    non_empty_slots = [p for p in area_slots if p.get("is_empty_like") is not True]
    predicted_count = sum(1 for p in area_slots if p.get("predicted_product"))
    correct_count = sum(
        1 for p in non_empty_slots
        if normalize_class_name_for_stream(p.get("predicted_product", "")) == expected_stream_key
    )
    return {
        "slots_total": slots_total,
        "predicted_count": predicted_count,
        "empty_like_count": empty_like_count,
        "non_empty_count": len(non_empty_slots),
        "correct_count": correct_count,
        "accuracy_percent": (correct_count / len(non_empty_slots) * 100.0) if non_empty_slots else 0.0,
    }


def build_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_images = len(results)
    detected_images = sum(1 for r in results if len(r["barcodes"]) > 0)
    total_barcodes = sum(len(r["barcodes"]) for r in results)
    reading_stats = summarize_ms([r["elapsed_ms"] for r in results])
    ocr_stats = summarize_ms([r["ocr_processing_ms"] for r in results if r.get("ocr_processing_ms") is not None])

    date_slots = [dr for r in results for dr in r.get("date_results", [])]
    date_slots_total = len(date_slots)
    date_present_count = sum(1 for d in date_slots if d.get("date_exists"))
    date_missing_count = date_slots_total - date_present_count
    date_correct_count = sum(1 for d in date_slots if d.get("is_correct") is True)
    date_wrong_count = sum(1 for d in date_slots if d.get("date_exists") and d.get("is_correct") is False)

    expected_stream_key = infer_stream_product_key(STREAM_URL)
    dinov2_stats = {
        "label1": _build_area_stats(results, "label1_results", expected_stream_key),
        "label2": _build_area_stats(results, "label2_results", expected_stream_key),
        "product_area": _build_area_stats(results, "product_area_results", expected_stream_key),
    }

    return {
        "total_images": total_images,
        "detected_images": detected_images,
        "detection_rate_percent": (detected_images / total_images * 100.0) if total_images else 0.0,
        "avg_barcodes_per_image": (total_barcodes / total_images) if total_images else 0.0,
        "barcode_reading_time": reading_stats,
        "ocr_processing_time": ocr_stats,
        "date_slots_total": date_slots_total,
        "date_present_count": date_present_count,
        "date_missing_count": date_missing_count,
        "date_presence_rate_percent": (date_present_count / date_slots_total * 100.0) if date_slots_total else 0.0,
        "date_correct_count": date_correct_count,
        "date_wrong_count": date_wrong_count,
        "date_accuracy_percent": (date_correct_count / date_present_count * 100.0) if date_present_count else 0.0,
        **dinov2_stats,
    }


# =========================
# Töötlemine
# =========================
def save_graph(timestamps, change_scores, capture_times, report_folder: str):
    if not timestamps or not change_scores:
        return
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, change_scores, label="Change score (MAE)")
    plt.axhline(y=MOTION_THRESHOLD, linestyle="--", label=f"Motion threshold = {MOTION_THRESHOLD}")
    plt.axhline(y=STABLE_THRESHOLD, linestyle=":", label=f"Stable threshold = {STABLE_THRESHOLD}")
    for i, t in enumerate(capture_times):
        label = "Captured image" if i == 0 else None
        plt.axvline(x=t, linestyle="-.", alpha=0.8, label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("MAE change score")
    plt.title("Motion analysis")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(report_folder, "liikumise_graafik.png")
    plt.savefig(out, dpi=150)
    plt.close()

def _extract_points_from_item(item) -> List[Tuple[float, float]]:
    try:
        loc = item.get_location()
        if loc is None:
            return []

        if hasattr(loc, "points") and loc.points:
            pts = loc.points
            out = []
            for p in pts:
                x = getattr(p, "x", None)
                y = getattr(p, "y", None)
                if x is not None and y is not None:
                    out.append((float(x), float(y)))
            if out:
                return out

        if hasattr(loc, "get_points"):
            pts = loc.get_points()
            out = []
            for p in pts:
                x = getattr(p, "x", None)
                y = getattr(p, "y", None)
                if x is not None and y is not None:
                    out.append((float(x), float(y)))
            if out:
                return out
    except Exception:
        pass

    return []


def barcode_center_x(bc: Dict[str, Any]) -> float:
    points = bc.get("points") or []
    if not points:
        return float("inf")
    xs = [p[0] for p in points]
    return sum(xs) / len(xs)


def sort_barcodes_center_to_right(barcodes: List[Dict[str, Any]], frame_width: int) -> List[Dict[str, Any]]:
    if not barcodes:
        return []

    frame_center_x = frame_width / 2.0

    right_side = [bc for bc in barcodes if barcode_center_x(bc) >= frame_center_x]
    left_side = [bc for bc in barcodes if barcode_center_x(bc) < frame_center_x]

    right_side_sorted = sorted(right_side, key=lambda bc: barcode_center_x(bc) - frame_center_x)
    left_side_sorted = sorted(left_side, key=lambda bc: frame_center_x - barcode_center_x(bc))

    return right_side_sorted + left_side_sorted

def barcode_center_point_from_points(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def assign_barcodes_to_product_slots(
    barcodes: List[Dict[str, Any]],
    rois: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not barcodes or not rois:
        return barcodes

    roi_items: List[Tuple[int, str, Tuple[int, int, int, int], Tuple[float, float]]] = []
    for slot_index, package_name in enumerate(sorted(rois.keys()), start=1):
        ((x1, y1), (x2, y2)) = rois[package_name]
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        roi_center = ((x1i + x2i) / 2.0, (y1i + y2i) / 2.0)
        roi_items.append((slot_index, package_name, (x1i, y1i, x2i, y2i), roi_center))

    assigned: List[Dict[str, Any]] = []

    for bc in barcodes:
        center = barcode_center_point_from_points(bc.get("points") or [])
        item = dict(bc)
        item["source_crop"] = item.get("source_crop", "full_frame")

        if DEBUG_MODE:
            print(f"DEBUG barcode ean={bc.get('ean')} center={center} points={bc.get('points')}")
            for slot_index, package_name, (x1, y1, x2, y2), _ in roi_items:
                print(f"DEBUG roi S{slot_index} {package_name} = {(x1, y1, x2, y2)}")

        if center is None:
            item["slot_index"] = None
            item["slot_name"] = None
            assigned.append(item)
            continue

        cx, cy = center
        matched = False
        for slot_index, package_name, (x1, y1, x2, y2), _ in roi_items:
            if (x1 - BARCODE_SLOT_MARGIN_PX) <= cx <= (x2 + BARCODE_SLOT_MARGIN_PX) and (y1 - BARCODE_SLOT_MARGIN_PX) <= cy <= (y2 + BARCODE_SLOT_MARGIN_PX):
                item["slot_index"] = slot_index
                item["slot_name"] = package_name
                matched = True
                if DEBUG_MODE:
                    print(f"DEBUG assigned by margin: ean={bc.get('ean')} -> S{slot_index} ({package_name})")
                break

        if not matched and BARCODE_ASSIGN_NEAREST_FALLBACK:
            best_slot = None
            best_dist2 = None
            for slot_index, package_name, _, (rcx, rcy) in roi_items:
                dist2 = (cx - rcx) ** 2 + (cy - rcy) ** 2
                if best_dist2 is None or dist2 < best_dist2:
                    best_dist2 = dist2
                    best_slot = (slot_index, package_name)
            if best_slot is not None:
                item["slot_index"] = best_slot[0]
                item["slot_name"] = best_slot[1]
                matched = True
                if DEBUG_MODE:
                    print(f"DEBUG assigned by nearest fallback: ean={bc.get('ean')} -> S{best_slot[0]} ({best_slot[1]}) dist2={best_dist2}")

        if not matched:
            item["slot_index"] = None
            item["slot_name"] = None
            if DEBUG_MODE:
                print(f"DEBUG unassigned barcode ean={bc.get('ean')} center={center}")

        assigned.append(item)

    assigned.sort(key=lambda bc: (
        bc.get("slot_index") if bc.get("slot_index") is not None else 9999,
        bc.get("ean", ""),
    ))
    return assigned

def get_context_product(reader: BarcodeProductReader, barcode_result: Dict[str, Any], current_product: Optional[Dict[str, Any]]):
    barcodes = barcode_result.get("barcodes", [])
    if barcodes:
        for bc in barcodes:
            ean = bc.get("ean")
            if not ean:
                continue

            product = reader.product_db.get(ean)
            if product is None:
                continue

            context = product.copy()
            context["_ean"] = ean
            return context

    if current_product is not None:
        return current_product

    if reader.default_product is not None:
        return reader.default_product.copy()

    return None



def print_barcode_result(barcode_result: Dict[str, Any], rel_time: float) -> None:
    barcodes = barcode_result.get("barcodes", [])
    source_name = os.path.splitext(os.path.basename(barcode_result.get("image_path", "")))[0]

    if not barcodes:
        print(
            f"Aeg videos: {rel_time:.3f} s | {source_name} | full_frame | Triipkoodi ei leitud | "
            f"Lugemisaeg: {barcode_result.get('elapsed_ms', 0.0):.1f} ms"
        )
        return

    for bc in barcodes:
        product = bc.get("product")
        slot_index = bc.get("slot_index")
        slot_txt = f"S{slot_index}" if slot_index is not None else "S?"
        crop_txt = bc.get("source_crop", barcode_result.get("source_crop", "-"))
        read_ms = bc.get("slot_elapsed_ms", barcode_result.get("elapsed_ms", 0.0))
        if product:
            print(
                f"Aeg videos: {rel_time:.3f} s | {source_name} | {slot_txt} | EAN13: {bc['ean']} | "
                f"Toode: {product['name']} | Säilivusaeg: {product['expiry_date_str']} | "
                f"Lugemisaeg: {read_ms:.1f} ms | crop={crop_txt}"
            )
        else:
            print(
                f"Aeg videos: {rel_time:.3f} s | {source_name} | {slot_txt} | EAN13: {bc['ean']} | "
                f"Toode: andmebaasist puudub | Säilivusaeg: puudub | "
                f"Lugemisaeg: {read_ms:.1f} ms | crop={crop_txt}"
            )


def extract_and_normalize_packages(frame: np.ndarray, rois: Dict[str, Any]) -> List[np.ndarray]:
    package_names = sorted(rois.keys())
    packages = []

    for package_name in package_names:
        ((x1, y1), (x2, y2)) = rois[package_name]
        crop = safe_crop(frame, int(x1), int(y1), int(x2), int(y2))
        rotated = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        packages.append(rotated)

    valid_packages = [pkg for pkg in packages if pkg is not None and pkg.size > 0]
    if not valid_packages:
        return []

    target_h = max(pkg.shape[0] for pkg in valid_packages)
    target_w = max(pkg.shape[1] for pkg in valid_packages)

    normalized = [cv2.resize(pkg, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for pkg in valid_packages]
    return normalized



def extract_detail_areas(package_img: np.ndarray, product_cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    h, w = package_img.shape[:2]

    date_area = product_cfg["date_area"]
    (dx1, dy1), (dx2, dy2) = date_area
    date_crop = safe_crop(package_img, int(dx1), int(dy1), int(dx2), int(dy2))

    label1_below = int(product_cfg["label1_below"])
    label1_crop = safe_crop(package_img, 0, label1_below, w, h)

    label2_above = int(product_cfg["label2_above"])
    label2_crop = safe_crop(package_img, 0, 0, w, label2_above)

    product_area_between = product_cfg["product_area_between"]
    py1, py2 = int(product_area_between[0]), int(product_area_between[1])
    product_crop = safe_crop(package_img, 0, py1, w, py2)

    return {
        "date": date_crop,
        "label1": label1_crop,
        "label2": label2_crop,
        "product_area": product_crop,
    }



def validate_product_context(product_cfg: Optional[Dict[str, Any]]) -> bool:
    required_keys = ["rois", "date_area", "label1_below", "label2_above", "product_area_between"]
    return bool(product_cfg) and all(k in product_cfg for k in required_keys)


def detect_barcodes_from_full_frame(reader: BarcodeProductReader, frame: np.ndarray, image_path: str, product_cfg: Dict[str, Any]) -> Dict[str, Any]:
    barcode_result = reader.read_frame(frame, image_path)
    barcode_result["source_crop"] = "full_frame"
    for bc in barcode_result.get("barcodes", []):
        bc["source_crop"] = "full_frame"

    if product_cfg and "rois" in product_cfg:
        barcode_result["barcodes"] = assign_barcodes_to_product_slots(
            barcode_result.get("barcodes", []),
            product_cfg.get("rois") or {},
        )
    return barcode_result


def save_slicing_debug(frame: np.ndarray, takt_index: int, packages: List[np.ndarray], package_details: List[Dict[str, np.ndarray]], date_results: Optional[List[Dict[str, Any]]] = None) -> None:
    base_folder = STREAM_URL.split('/')[-1]
    debug_root = os.path.join(base_folder)
    subdirs = ["full_frames", "products", "date", "label1", "label2", "product_area", "date_best"]
    for subdir in subdirs:
        ensure_dir(os.path.join(debug_root, subdir))

    save_t0 = time.perf_counter()
    cv2.imwrite(os.path.join(debug_root, "full_frames", f"takt_{takt_index}_full.png"), frame)

    for slice_number, (package_img, details) in enumerate(zip(packages, package_details), start=1):
        cv2.imwrite(os.path.join(debug_root, "products", f"takt_{takt_index}_s{slice_number}_product.png"), package_img)
        cv2.imwrite(os.path.join(debug_root, "date", f"takt_{takt_index}_s{slice_number}_date.png"), details["date"])
        cv2.imwrite(os.path.join(debug_root, "label1", f"takt_{takt_index}_s{slice_number}_label1.png"), details["label1"])
        cv2.imwrite(os.path.join(debug_root, "label2", f"takt_{takt_index}_s{slice_number}_label2.png"), details["label2"])
        cv2.imwrite(os.path.join(debug_root, "product_area", f"takt_{takt_index}_s{slice_number}_product_area.png"), details["product_area"])

        if date_results and slice_number - 1 < len(date_results):
            dr = date_results[slice_number - 1]
            best_variant = dr.get("best_variant_image_bgr")
            if best_variant is not None:
                cv2.imwrite(os.path.join(debug_root, "date_best", f"takt_{takt_index}_s{slice_number}_best.png"), best_variant)

    save_ms = (time.perf_counter() - save_t0) * 1000.0
    print(f"Salvestamise aeg: {save_ms:.2f} ms")



def build_date_slot_results(date_crops: List[np.ndarray], expected_date: str) -> Tuple[List[Dict[str, Any]], float]:
    raw_results, elapsed_ms = run_date_ocr_for_crops(date_crops, expected_date=expected_date)
    out = []
    for crop, item in zip(date_crops, raw_results):
        is_correct = item["predicted_date"] == expected_date if item["date_exists"] else False
        best_variant_image_bgr = None
        for variant_name, variant_img_rgb in build_date_variants(crop):
            if variant_name == item.get("variant"):
                best_variant_image_bgr = cv2.cvtColor(variant_img_rgb, cv2.COLOR_RGB2BGR)
                break
        out.append({
            "expected_date": expected_date,
            "date_exists": bool(item["date_exists"]),
            "predicted_date": item["predicted_date"],
            "is_correct": is_correct,
            "winning_variant": item.get("variant", ""),
            "raw_text": item.get("raw_text", ""),
            "best_variant_image_bgr": best_variant_image_bgr,
            "candidate_count": len(item.get("all_candidates", [])),
        })
    return out, elapsed_ms


def run_dinov2_for_area(crops: List[np.ndarray], area: str) -> List[Dict[str, Any]]:
    analyzer = init_dinov2_product_analyzer()
    if analyzer is None or not analyzer.enabled:
        return []
    return [analyzer.classify_crop(crop, area=area) for crop in crops]


def save_area_query_debug(area: str, crops: List[np.ndarray], area_results: List[Dict[str, Any]], takt_index: int) -> None:
    if area == "product_area" and not DINOV2_SAVE_QUERY_PRODUCT_AREA:
        return
    base_folder = STREAM_URL.split('/')[-1]
    debug_root = os.path.join(base_folder, f"{area}_query")
    ensure_dir(debug_root)
    for idx, (crop, result) in enumerate(zip(crops, area_results), start=1):
        pred = result.get("predicted_product") or "unknown"
        pred = pred.replace("/", "_")
        out_name = f"takt_{takt_index}_s{idx}_{pred}.png"
        cv2.imwrite(os.path.join(debug_root, out_name), crop)


def process_captured_frame(
    reader: BarcodeProductReader,
    frame: np.ndarray,
    image_path: str,
    rel_time: float,
    results: List[Dict[str, Any]],
    current_product: Optional[Dict[str, Any]],
    takt_index: int,
) -> Optional[Dict[str, Any]]:
    active_product = current_product.copy() if current_product is not None else None
    if active_product is None and reader.default_product is not None:
        active_product = reader.default_product.copy()

    if not validate_product_context(active_product):
        print(f"Aeg videos: {rel_time:.3f} s | {os.path.basename(image_path)} | Toote kontekst puudub, tükeldamine jäi tegemata.")
        barcode_result = {
            "image_path": image_path,
            "elapsed_ms": 0.0,
            "barcodes": [],
            "video_time_s": rel_time,
            "date_results": [],
            "ocr_processing_ms": None,
            "label1_results": [],
            "label2_results": [],
            "product_area_results": [],
        }
        results.append(barcode_result)
        return active_product

    loop_start_t = time.perf_counter()
    packages = extract_and_normalize_packages(frame, active_product["rois"])
    if len(packages) != 4:
        print(f"Hoiatus: oodati 4 pakendit, saadi {len(packages)}")
    package_details = [extract_detail_areas(pkg, active_product) for pkg in packages]
    slicing_ms = (time.perf_counter() - loop_start_t) * 1000.0

    barcode_result = detect_barcodes_from_full_frame(reader, frame, image_path, active_product)
    barcode_result["video_time_s"] = rel_time
    barcode_result["takt_index"] = takt_index
    barcode_result["date_results"] = []
    barcode_result["ocr_processing_ms"] = None
    barcode_result["label1_results"] = []
    barcode_result["label2_results"] = []
    barcode_result["product_area_results"] = []
    results.append(barcode_result)
    print_barcode_result(barcode_result, rel_time)

    active_product = get_context_product(reader, barcode_result, active_product)
    if not validate_product_context(active_product):
        print("Meil pole tooteinfot. Jätkame...")
        return active_product

    product_name = active_product.get("ITEMNAME") or active_product.get("name") or "Tundmatu toode"
    expiry_date_str = resolve_expected_date(product_name)
    ean_str = active_product.get("_ean", "Tundmatu")

    date_crops = [d["date"] for d in package_details]
    ocr_results, ocr_elapsed_ms = build_date_slot_results(date_crops, expiry_date_str)
    barcode_result["date_results"] = ocr_results
    barcode_result["ocr_processing_ms"] = ocr_elapsed_ms

    label1_crops = [d["label1"] for d in package_details]
    label2_crops = [d["label2"] for d in package_details]
    product_crops = [d["product_area"] for d in package_details]
    label1_results = run_dinov2_for_area(label1_crops, "label1")
    label2_results = run_dinov2_for_area(label2_crops, "label2")
    product_area_results = run_dinov2_for_area(product_crops, "product_area")
    barcode_result["label1_results"] = label1_results
    barcode_result["label2_results"] = label2_results
    barcode_result["product_area_results"] = product_area_results

    print("\n -------------------------------------------- \n")
    print(f"Takt {takt_index}. Liikumine oli tuvastatud {rel_time:.2f}s juures!")
    print(f"Kontekst: EAN {ean_str} | {product_name} | Aegub {expiry_date_str}")
    print(f"Tükeldamise eeltöötluse aeg: {slicing_ms:.2f} ms")
    print(f"PaddleOCR aeg (4 kuupäeva ala + variandid): {ocr_elapsed_ms:.2f} ms")

    for slot_index, dr in enumerate(ocr_results, start=1):
        status = "olemas" if dr["date_exists"] else "puudub"
        corr = "oige" if dr["is_correct"] else ("vale" if dr["date_exists"] else "-")
        print(
            f"  S{slot_index}: kuupäev {status} | tulemus={dr['predicted_date']} | "
            f"oodatav={expiry_date_str} | staatus={corr} | variant={dr['winning_variant']} | raw={dr['raw_text']}"
        )

    for area_name, area_results, prefix in [
        ("label1", label1_results, "L1"),
        ("label2", label2_results, "L2"),
        ("product_area", product_area_results, "P"),
    ]:
        if area_results:
            print(f"  -- DINOv2 {area_name} --")
            for slot_index, pr in enumerate(area_results, start=1):
                nd = pr.get("nearest_distance")
                ed = pr.get("empty_distance")
                nd_txt = f"{nd:.4f}" if nd is not None else "-"
                ed_txt = f"{ed:.4f}" if ed is not None else "-"
                empty_txt = "jah" if pr.get("is_empty_like") is True else ("ei" if pr.get("is_empty_like") is False else "-")
                print(
                    f"  {prefix}{slot_index}: final={pr.get('predicted_product', '-') } | nn={pr.get('predicted_product_nearest', '-') } | "
                    f"centroid={pr.get('predicted_product_centroid', '-') } | method={pr.get('decision_method', '-') } | "
                    f"nearest_class={pr.get('nearest_class', '-') } | d_pos={nd_txt} | d_empty={ed_txt} | empty_like={empty_txt}"
                )

    if DEBUG_MODE:
        save_slicing_debug(frame, takt_index, packages, package_details, ocr_results)
        for area_name, crops, area_results in [
            ("label1", label1_crops, label1_results),
            ("label2", label2_crops, label2_results),
            ("product_area", product_crops, product_area_results),
        ]:
            if area_results:
                save_area_query_debug(area_name, crops, area_results, takt_index)

    return active_product


def write_dinov2_summary_reports(results: List[Dict[str, Any]], analyzer: Dinov2ProductAnalyzer, report_folder: str) -> None:
    stats = build_statistics(results)
    for area, result_key in [("label1", "label1_results"), ("label2", "label2_results"), ("product_area", "product_area_results")]:
        out_path = os.path.join(report_folder, f"dinov2_{area}_report.txt")
        area_stats = stats[area]
        calib = analyzer.calibration_by_area.get(area, {})
        counts = analyzer.reference_counts_by_area.get(area, {})
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"DINOv2 kokkuvõte: {area}\n")
            f.write("===========================\n\n")
            f.write(f"Model: {analyzer.model_name}\n")
            f.write(f"Device: {analyzer.device}\n")
            f.write(f"Reference root: {analyzer.reference_root}\n\n")
            f.write("Reference klassid\n")
            f.write("----------------\n")
            for class_name in sorted(counts.keys()):
                f.write(f"{class_name}: {counts[class_name]}\n")
            f.write("\nKalibratsioon\n")
            f.write("-----------\n")
            f.write(f"leave_one_out_accuracy: {calib.get('leave_one_out_accuracy')}\n")
            f.write(f"intra_positive_median: {calib.get('intra_positive_median')}\n")
            f.write(f"positive_vs_empty_median: {calib.get('positive_vs_empty_median')}\n")
            f.write(f"empty_threshold: {calib.get('empty_threshold')}\n\n")
            f.write("Query statistika\n")
            f.write("--------------\n")
            f.write(f"slots kokku: {area_stats['slots_total']}\n")
            f.write(f"empty_like: {area_stats['empty_like_count']}\n")
            f.write(f"non-empty: {area_stats['non_empty_count']}\n")
            f.write(f"õige toode: {area_stats['correct_count']}\n")
            f.write(f"toote täpsus: {area_stats['accuracy_percent']:.2f}%\n")
            f.write(f"histogramm: dinov2_hist_{area}.png\n")


def finalize_reports(results: List[Dict[str, Any]], report_folder: str) -> None:
    txt_path = os.path.join(report_folder, "triipkoodi_ja_kuupaeva_tulemused.txt")
    csv_path = os.path.join(report_folder, "ocr_raport.csv")
    write_results_report(results, txt_path)
    write_ocr_csv_report(results, csv_path)
    print(f"Tulemused salvestatud: {txt_path}")
    print(f"OCR raport salvestatud: {csv_path}")

    stats = build_statistics(results)
    print("\nStatistika")
    print("==========")
    print(f"Takte kokku: {stats['total_images']}")
    print(f"Tuvastatud taktide arv: {stats['detected_images']}")
    print(f"Triipkoodi tuvastusmäär: {stats['detection_rate_percent']:.2f}%")
    print(f"Pakke kokku OCR-is: {stats['date_slots_total']}")
    print(f"Kuupäev olemas: {stats['date_present_count']}")
    print(f"Kuupäev puudub: {stats['date_missing_count']}")
    print(f"Kuupäeva saagis: {stats['date_presence_rate_percent']:.2f}%")
    print(f"Õige kuupäev: {stats['date_correct_count']}")
    print(f"Vale kuupäev: {stats['date_wrong_count']}")
    print(f"Kuupäeva täpsus: {stats['date_accuracy_percent']:.2f}%")
    if stats['barcode_reading_time']:
        print(f"Triipkoodi keskmine lugemisaeg: {stats['barcode_reading_time']['mean_ms']:.2f} ms")
        print(f"Triipkoodi maksimaalne lugemisaeg: {stats['barcode_reading_time']['max_ms']:.2f} ms")
    if stats['ocr_processing_time']:
        print(f"OCR keskmine aeg takti kohta: {stats['ocr_processing_time']['mean_ms']:.2f} ms")
        print(f"OCR maksimaalne aeg takti kohta: {stats['ocr_processing_time']['max_ms']:.2f} ms")
    for area in DINOV2_POSITIVE_AREAS:
        area_stats = stats[area]
        if area_stats['slots_total']:
            print(f"DINOv2 {area} slots kokku: {area_stats['slots_total']}")
            print(f"DINOv2 {area} empty_like: {area_stats['empty_like_count']}")
            print(f"DINOv2 {area} non-empty: {area_stats['non_empty_count']}")
            print(f"DINOv2 {area} õige toode: {area_stats['correct_count']}")
            print(f"DINOv2 {area} toote täpsus: {area_stats['accuracy_percent']:.2f}%")

    analyzer = init_dinov2_product_analyzer()
    if analyzer is not None and analyzer.enabled:
        analyzer.write_reference_report(os.path.join(report_folder, "dinov2_reference_report.txt"))
        analyzer.save_reference_histograms(report_folder)
        write_dinov2_summary_reports(results, analyzer, report_folder)


def run_images_mode(reader: BarcodeProductReader, report_folder: str):
    ensure_dir(TEST_IMAGES_FOLDER)
    results = []
    files = sorted(
        f for f in os.listdir(TEST_IMAGES_FOLDER)
        if f.startswith(TEST_IMAGE_PREFIX) and f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    current_product = None
    for idx, filename in enumerate(files, start=1):
        full_path = os.path.join(TEST_IMAGES_FOLDER, filename)
        frame = cv2.imread(full_path)
        if frame is None:
            continue
        current_product = process_captured_frame(
            reader=reader,
            frame=frame,
            image_path=full_path,
            rel_time=0.0,
            results=results,
            current_product=current_product,
            takt_index=idx,
        )

    finalize_reports(results, report_folder)



def run_rtsp_mode(reader: BarcodeProductReader, report_folder: str):
    stream = RTSPStreamReader(STREAM_URL)
    time.sleep(2)

    if not stream.ret:
        print(f"Viga ühendusega: {STREAM_URL}")
        raise SystemExit(1)

    timestamps = []
    change_scores = []
    capture_times = []
    results = []

    started = False
    green_cooldown = False
    cycle_start_time = 0.0
    total_triggers = 0
    current_product = reader.default_product.copy() if reader.default_product is not None else None

    in_motion_event = False
    capture_armed = False
    capture_ready_time = None
    stable_since = None

    try:
        while True:
            ret1, frame1 = stream.read()
            time.sleep(FRAME_SLEEP)
            ret2, frame2 = stream.read()
            if not ret1 or not ret2:
                break

            now = time.time()
            current_is_green = is_green_screen(frame2)

            if not started:
                if current_is_green:
                    print(">>> Start green detected. Starting cycle.")
                    started = True
                    green_cooldown = True
                    cycle_start_time = now
                continue

            if not current_is_green:
                green_cooldown = False

            if current_is_green and not green_cooldown:
                print(">>> End green detected. Stopping cycle.")
                break

            change, _ = measure_change(frame1, frame2)
            rel_time = now - cycle_start_time if cycle_start_time else 0.0
            timestamps.append(rel_time)
            change_scores.append(change)

            if change > MOTION_THRESHOLD:
                if not in_motion_event:
                    in_motion_event = True
                    capture_armed = True
                    capture_ready_time = now + CAPTURE_DELAY
                    stable_since = None
                    print(f">>> Motion event START at {rel_time:.3f}s (score={change:.3f})")
            else:
                if in_motion_event:
                    in_motion_event = False

            if capture_armed and now >= capture_ready_time:
                if change < STABLE_THRESHOLD:
                    if stable_since is None:
                        stable_since = now
                    elif (now - stable_since) >= STABLE_TIME_REQUIRED:
                        ret_cap, capture_frame = stream.read()
                        if ret_cap and capture_frame is not None:
                            total_triggers += 1
                            takt_name = build_takt_image_name(total_triggers)
                            capture_times.append(rel_time)
                            print(f">>> Captured stable frame: {takt_name} at {rel_time:.3f}s")
                            current_product = process_captured_frame(
                                reader=reader,
                                frame=capture_frame,
                                image_path=takt_name,
                                rel_time=rel_time,
                                results=results,
                                current_product=current_product,
                                takt_index=total_triggers,
                            )
                        capture_armed = False
                        capture_ready_time = None
                        stable_since = None
                else:
                    stable_since = None
    finally:
        stream.stop()

    save_graph(timestamps, change_scores, capture_times, report_folder)
    finalize_reports(results, report_folder)



def main():
    base_folder = STREAM_URL.split('/')[-1]
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_folder = os.path.join("reports", f"{base_folder}_{date_str}")
    ensure_dir(report_folder)

    init_paddle_ocr()
    analyzer = init_dinov2_product_analyzer()
    if analyzer is not None and not analyzer.enabled and analyzer.error_message:
        print(analyzer.error_message)

    reader = BarcodeProductReader(
        license_key=DYNAMSOFT_LICENSE,
        template_path=TEMPLATE_PATH,
        product_db_path=PRODUCT_DB_PATH,
        capture_date_str=CAPTURE_DATE_STR,
        stream_product_key=infer_stream_product_key(STREAM_URL),
    )

    if reader.default_product is not None:
        default_name = reader.default_product.get("ITEMNAME") or reader.default_product.get("name") or "?"
        print(f"Default product from stream: {default_name}")
    else:
        print(f"Default product from stream: puudub ({infer_stream_product_key(STREAM_URL)})")

    print(f"Mode: {MODE}")
    print(f"Video kuupäev: {CAPTURE_DATE_STR}")
    print(f"DEBUG_MODE: {DEBUG_MODE}")
    print(f"OCR model: {paddle_date_ocr.model_name if paddle_date_ocr else 'PaddleOCR'}")
    print(f"Default product from stream: {(reader.default_product or {}).get('ITEMNAME', (reader.default_product or {}).get('name', 'PUUDUB'))}")
    if analyzer is not None:
        print(f"DINOv2 enabled: {analyzer.enabled}")
        print(f"DINOv2 reference root: {DINOV2_REFERENCE_ROOT}")
        if analyzer.enabled:
            print(f"DINOv2 model: {DINOV2_MODEL_NAME} | device={analyzer.device} | areas={','.join(DINOV2_POSITIVE_AREAS)}")
        elif analyzer.error_message:
            print(f"DINOv2 info: {analyzer.error_message}")

    if MODE == "images":
        run_images_mode(reader, report_folder)
    elif MODE == "rtsp":
        run_rtsp_mode(reader, report_folder)
    else:
        raise ValueError("MODE peab olema 'rtsp' või 'images'")


if __name__ == "__main__":
    main()
