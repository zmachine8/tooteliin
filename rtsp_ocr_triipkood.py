import os
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
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

from dynamsoft_barcode_reader_bundle import *

# =========================
# KONFIGURATSIOON
# =========================
MODE = "rtsp"  # "rtsp" või "images"

STREAM_URL = "rtsp://172.17.37.81:8554/rulaad"
SHARED_CAPTURE_FOLDER = "captures"
TEST_IMAGES_FOLDER = SHARED_CAPTURE_FOLDER
TEST_IMAGE_PREFIX = "capture_"

# Dynamsoft
DYNAMSOFT_LICENSE = "t0088YQEAACNxJmkf8GttAqbAp6SwlzBDDmGyqS+wr7cKNFZA60wxkoMTAEVucd4B5oz5RrBs9qmv9rznWBwM6hEuifMcw0O0H0z/aOJuGt6bVY2tltgAycZJgQ=="
TEMPLATE_PATH = "minimal_template.json"  # valikuline: kui puudub, kasutatakse default seadeid
PRODUCT_DB_PATH = "barcode_data.json"

# Ülesandes antud kuupäev
CAPTURE_DATE_STR = "22.03.2026"

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

# OCR
OCR_VARIANT_NAMES = ["orig", "clahe", "sharp", "otsu", "adaptive"]
DATE_UPSCALE_TARGET_WIDTH = 240
DATE_UPSCALE_FACTOR = 2.0
PADDLE_BATCH_SIZE = 16

BASE_DIR = Path(__file__).resolve().parent
SEMINAR5_DIR = BASE_DIR / "seminar5"
ROOT_HELPERS_PATH = BASE_DIR / "helpers.py"


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
    def __init__(self, license_key: str, template_path: str, product_db_path: str, capture_date_str: str):
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

    def _parse_capture_result(self, result, source_name: str, elapsed_ms: float) -> Dict[str, Any]:
        items = result.get_items() if result is not None else None
        barcodes = []
        if items:
            for item in items:
                if item.get_type() == EnumCapturedResultItemType.CRIT_BARCODE:
                    ean = item.get_text()
                    product_info = self.lookup_product(ean)
                    barcodes.append({
                        "ean": ean,
                        "product": product_info,
                    })

        return {
            "image_path": source_name,
            "elapsed_ms": elapsed_ms,
            "barcodes": barcodes,
        }

    def read_image(self, image_path: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        result = self.router.capture(image_path, self.template_name)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._parse_capture_result(result, image_path, elapsed_ms)

    def read_frame(self, frame: np.ndarray, source_name: str = "<frame>") -> Dict[str, Any]:
        t0 = time.perf_counter()
        result = self.router.capture(frame, self.template_name)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._parse_capture_result(result, source_name, elapsed_ms)


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
            if not barcodes:
                f.write("Triipkood: ei leitud\n")
            else:
                for bc in barcodes:
                    product = bc.get("product")
                    if product:
                        f.write(
                            f"EAN13: {bc['ean']} | Toode: {product['name']} | Säilivusaeg: {product['expiry_date_str']}\n"
                        )
                    else:
                        f.write(f"EAN13: {bc['ean']} | Toode: andmebaasist puudub | Säilivusaeg: puudub\n")

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
        if stats['barcode_reading_time']:
            f.write(f"Triipkoodi keskmine lugemisaeg: {stats['barcode_reading_time']['mean_ms']:.2f} ms\n")
            f.write(f"Triipkoodi maksimaalne lugemisaeg: {stats['barcode_reading_time']['max_ms']:.2f} ms\n")
        if stats['ocr_processing_time']:
            f.write(f"OCR keskmine aeg takti kohta: {stats['ocr_processing_time']['mean_ms']:.2f} ms\n")
            f.write(f"OCR maksimaalne aeg takti kohta: {stats['ocr_processing_time']['max_ms']:.2f} ms\n")



def write_ocr_csv_report(results: List[Dict[str, Any]], path: str) -> None:
    headers = [
        "capture_file", "video_time_s", "slot", "barcode_found", "ean", "product_name",
        "expected_date", "date_exists", "predicted_date", "is_correct", "winning_variant",
        "winning_raw_text", "barcode_elapsed_ms", "ocr_processing_ms"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for entry in results:
            first_barcode = entry.get("barcodes", [{}])[0] if entry.get("barcodes") else {}
            product = first_barcode.get("product") or {}
            date_results = entry.get("date_results") or []
            if not date_results:
                writer.writerow({
                    "capture_file": os.path.basename(entry["image_path"]),
                    "video_time_s": f"{entry.get('video_time_s', 0.0):.3f}" if entry.get("video_time_s") is not None else "",
                    "slot": "-",
                    "barcode_found": bool(entry.get("barcodes")),
                    "ean": first_barcode.get("ean", ""),
                    "product_name": product.get("name", ""),
                    "expected_date": "",
                    "date_exists": False,
                    "predicted_date": "NA",
                    "is_correct": "",
                    "winning_variant": "",
                    "winning_raw_text": "",
                    "barcode_elapsed_ms": f"{entry.get('elapsed_ms', 0.0):.2f}",
                    "ocr_processing_ms": f"{entry.get('ocr_processing_ms', 0.0):.2f}" if entry.get("ocr_processing_ms") is not None else "",
                })
                continue

            for idx, dr in enumerate(date_results, start=1):
                writer.writerow({
                    "capture_file": os.path.basename(entry["image_path"]),
                    "video_time_s": f"{entry.get('video_time_s', 0.0):.3f}" if entry.get("video_time_s") is not None else "",
                    "slot": idx,
                    "barcode_found": bool(entry.get("barcodes")),
                    "ean": first_barcode.get("ean", ""),
                    "product_name": product.get("name", ""),
                    "expected_date": dr.get("expected_date", ""),
                    "date_exists": dr.get("date_exists", False),
                    "predicted_date": dr.get("predicted_date", "NA"),
                    "is_correct": dr.get("is_correct", ""),
                    "winning_variant": dr.get("winning_variant", ""),
                    "winning_raw_text": dr.get("raw_text", ""),
                    "barcode_elapsed_ms": f"{entry.get('elapsed_ms', 0.0):.2f}",
                    "ocr_processing_ms": f"{entry.get('ocr_processing_ms', 0.0):.2f}" if entry.get("ocr_processing_ms") is not None else "",
                })



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



def get_context_product(reader: BarcodeProductReader, barcode_result: Dict[str, Any], current_product: Optional[Dict[str, Any]]):
    barcodes = barcode_result.get("barcodes", [])
    if not barcodes:
        return current_product

    ean = barcodes[0].get("ean")
    if not ean:
        return current_product

    product = reader.product_db.get(ean)
    if product is None:
        return current_product

    context = product.copy()
    context["_ean"] = ean
    return context



def print_barcode_result(barcode_result: Dict[str, Any], rel_time: float) -> None:
    if not barcode_result["barcodes"]:
        print(
            f"Aeg videos: {rel_time:.3f} s | {os.path.basename(barcode_result['image_path'])} | Triipkoodi ei leitud | "
            f"Lugemisaeg: {barcode_result['elapsed_ms']:.1f} ms"
        )
        return

    for bc in barcode_result["barcodes"]:
        product = bc.get("product")
        if product:
            print(
                f"Aeg videos: {rel_time:.3f} s | EAN13: {bc['ean']} | Toode: {product['name']} | "
                f"Säilivusaeg: {product['expiry_date_str']} | Lugemisaeg: {barcode_result['elapsed_ms']:.1f} ms"
            )
        else:
            print(
                f"Aeg videos: {rel_time:.3f} s | EAN13: {bc['ean']} | Toode: andmebaasist puudub | "
                f"Säilivusaeg: puudub | Lugemisaeg: {barcode_result['elapsed_ms']:.1f} ms"
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



def save_slicing_debug(frame: np.ndarray, takt_index: int, packages: List[np.ndarray], package_details: List[Dict[str, np.ndarray]], date_results: Optional[List[Dict[str, Any]]] = None) -> None:
    base_folder = STREAM_URL.split('/')[-1]
    debug_root = os.path.join(base_folder)
    subdirs = ["full_frames", "date", "label1", "label2", "product_area", "date_best"]
    for subdir in subdirs:
        ensure_dir(os.path.join(debug_root, subdir))

    save_t0 = time.perf_counter()
    cv2.imwrite(os.path.join(debug_root, "full_frames", f"takt_{takt_index}_full.png"), frame)

    for slice_number, (package_img, details) in enumerate(zip(packages, package_details), start=1):
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



def process_captured_frame(
    reader: BarcodeProductReader,
    frame: np.ndarray,
    image_path: str,
    rel_time: float,
    results: List[Dict[str, Any]],
    current_product: Optional[Dict[str, Any]],
    takt_index: int,
) -> Optional[Dict[str, Any]]:
    barcode_result = reader.read_frame(frame, image_path)
    barcode_result["video_time_s"] = rel_time
    barcode_result["date_results"] = []
    barcode_result["ocr_processing_ms"] = None
    results.append(barcode_result)
    print_barcode_result(barcode_result, rel_time)

    current_product = get_context_product(reader, barcode_result, current_product)
    if current_product is None:
        print("Meil pole tooteinfot. Jätkame...")
        return current_product

    required_keys = ["rois", "date_area", "label1_below", "label2_above", "product_area_between"]
    if not all(k in current_product for k in required_keys):
        print(f"VIGA: Tooteinfot EAN {current_product.get('_ean')} on puudulik!")
        return current_product

    product_name = current_product.get("ITEMNAME") or current_product.get("name") or "Tundmatu toode"
    expiry_duration = current_product.get("BESTBEFOREDAYS", current_product.get("expiry_duration", 0))
    expiry_date = reader.capture_date + timedelta(days=int(expiry_duration)) if expiry_duration is not None else reader.capture_date
    expiry_date_str = expiry_date.strftime('%d.%m.%Y')
    ean_str = current_product.get("_ean", "Tundmatu")

    loop_start_t = time.perf_counter()
    packages = extract_and_normalize_packages(frame, current_product["rois"])
    if len(packages) != 4:
        print(f"Hoiatus: oodati 4 pakendit, saadi {len(packages)}")

    package_details = [extract_detail_areas(pkg, current_product) for pkg in packages]
    slicing_ms = (time.perf_counter() - loop_start_t) * 1000.0

    date_crops = [d["date"] for d in package_details]
    ocr_results, ocr_elapsed_ms = build_date_slot_results(date_crops, expiry_date_str)
    barcode_result["date_results"] = ocr_results
    barcode_result["ocr_processing_ms"] = ocr_elapsed_ms

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

    if DEBUG_MODE:
        save_slicing_debug(frame, takt_index, packages, package_details, ocr_results)

    return current_product



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
    ensure_dir(SHARED_CAPTURE_FOLDER)
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
    capture_index = 0
    total_triggers = 0
    current_product = None

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
                            filename = os.path.join(SHARED_CAPTURE_FOLDER, f"capture_{capture_index:03d}.jpg")
                            cv2.imwrite(filename, capture_frame)
                            capture_times.append(rel_time)
                            total_triggers += 1
                            print(f">>> Captured stable frame: {filename} at {rel_time:.3f}s")
                            current_product = process_captured_frame(
                                reader=reader,
                                frame=capture_frame,
                                image_path=filename,
                                rel_time=rel_time,
                                results=results,
                                current_product=current_product,
                                takt_index=total_triggers,
                            )
                            capture_index += 1
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

    reader = BarcodeProductReader(
        license_key=DYNAMSOFT_LICENSE,
        template_path=TEMPLATE_PATH,
        product_db_path=PRODUCT_DB_PATH,
        capture_date_str=CAPTURE_DATE_STR,
    )

    print(f"Mode: {MODE}")
    print(f"Video kuupäev: {CAPTURE_DATE_STR}")
    print(f"DEBUG_MODE: {DEBUG_MODE}")
    print(f"OCR model: {paddle_date_ocr.model_name if paddle_date_ocr else 'PaddleOCR'}")

    if MODE == "images":
        run_images_mode(reader, report_folder)
    elif MODE == "rtsp":
        run_rtsp_mode(reader, report_folder)
    else:
        raise ValueError("MODE peab olema 'rtsp' või 'images'")


if __name__ == "__main__":
    main()
