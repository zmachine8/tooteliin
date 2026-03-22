import os
import cv2
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
from datetime import datetime, timedelta
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

# =========================
# ABI
# =========================
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
    # BGR
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



def write_results_report(results: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Triipkoodi tulemused\n")
        f.write("===================\n\n")
        for entry in results:
            f.write(f"Pilt: {os.path.basename(entry['image_path'])}\n")
            if entry.get("video_time_s") is not None:
                f.write(f"Aeg videos: {entry['video_time_s']:.3f} s\n")
            f.write(f"Lugemise aeg: {entry['elapsed_ms']:.1f} ms\n")
            barcodes = entry["barcodes"]
            if not barcodes:
                f.write("Tulemus: Triipkoodi ei leitud\n\n")
                continue

            for bc in barcodes:
                product = bc.get("product")
                if product:
                    f.write(
                        f"EAN13: {bc['ean']} | Toode: {product['name']} | Säilivusaeg: {product['expiry_date_str']}\n"
                    )
                else:
                    f.write(f"EAN13: {bc['ean']} | Toode: andmebaasist puudub | Säilivusaeg: puudub\n")
            f.write("\n")

        stats = build_statistics(results)
        f.write("Statistika\n")
        f.write("==========\n")
        f.write(f"Takte kokku: {stats['total_images']}\n")
        f.write(f"Tuvastatud taktide arv: {stats['detected_images']}\n")
        f.write(f"Tuvastusmäär: {stats['detection_rate_percent']:.2f}%\n")
        f.write(f"Keskmine triipkoode pildi kohta: {stats['avg_barcodes_per_image']:.3f}\n")
        if stats['barcode_reading_time']:
            f.write(f"Keskmine lugemisaeg: {stats['barcode_reading_time']['mean_ms']:.2f} ms\n")
            f.write(f"Maksimaalne lugemisaeg: {stats['barcode_reading_time']['max_ms']:.2f} ms\n")



def build_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_images = len(results)
    detected_images = sum(1 for r in results if len(r["barcodes"]) > 0)
    total_barcodes = sum(len(r["barcodes"]) for r in results)
    reading_stats = summarize_ms([r["elapsed_ms"] for r in results])

    return {
        "total_images": total_images,
        "detected_images": detected_images,
        "detection_rate_percent": (detected_images / total_images * 100.0) if total_images else 0.0,
        "avg_barcodes_per_image": (total_barcodes / total_images) if total_images else 0.0,
        "barcode_reading_time": reading_stats,
    }



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



def save_slicing_debug(frame: np.ndarray, takt_index: int, packages: List[np.ndarray], package_details: List[Dict[str, np.ndarray]]) -> None:
    base_folder = STREAM_URL.split('/')[-1]
    debug_root = os.path.join(base_folder)
    subdirs = ["full_frames", "date", "label1", "label2", "product_area"]
    for subdir in subdirs:
        ensure_dir(os.path.join(debug_root, subdir))

    save_t0 = time.perf_counter()
    cv2.imwrite(os.path.join(debug_root, "full_frames", f"takt_{takt_index}_full.png"), frame)

    for slice_number, (package_img, details) in enumerate(zip(packages, package_details), start=1):
        cv2.imwrite(os.path.join(debug_root, "date", f"takt_{takt_index}_s{slice_number}_date.png"), details["date"])
        cv2.imwrite(os.path.join(debug_root, "label1", f"takt_{takt_index}_s{slice_number}_label1.png"), details["label1"])
        cv2.imwrite(os.path.join(debug_root, "label2", f"takt_{takt_index}_s{slice_number}_label2.png"), details["label2"])
        cv2.imwrite(os.path.join(debug_root, "product_area", f"takt_{takt_index}_s{slice_number}_product_area.png"), details["product_area"])

    save_ms = (time.perf_counter() - save_t0) * 1000.0
    print(f"Salvestamise aeg: {save_ms:.2f} ms")



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
    ean_str = current_product.get("_ean", "Tundmatu")

    loop_start_t = time.perf_counter()
    packages = extract_and_normalize_packages(frame, current_product["rois"])
    if len(packages) != 4:
        print(f"Hoiatus: oodati 4 pakendit, saadi {len(packages)}")

    package_details = [extract_detail_areas(pkg, current_product) for pkg in packages]
    time_to_process_ms = (time.perf_counter() - loop_start_t) * 1000.0

    print("\n -------------------------------------------- \n")
    print(f"Takt {takt_index}. Liikumine oli tuvastatud {rel_time:.2f}s juures!")
    print(f"Kontekst: EAN {ean_str} | {product_name} | Aegub {expiry_date.strftime('%d.%m.%Y')}")
    print(f"Tükeldamise eeltöötluse aeg: {time_to_process_ms:.2f} ms")

    if DEBUG_MODE:
        save_slicing_debug(frame, takt_index, packages, package_details)

    return current_product



def run_images_mode(reader: BarcodeProductReader, report_folder: str):
    ensure_dir(TEST_IMAGES_FOLDER)
    files = sorted(
        f for f in os.listdir(TEST_IMAGES_FOLDER)
        if f.startswith(TEST_IMAGE_PREFIX) and f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    results = []
    for filename in files:
        full_path = os.path.join(TEST_IMAGES_FOLDER, filename)
        res = reader.read_image(full_path)
        res["video_time_s"] = None
        results.append(res)
        if not res["barcodes"]:
            print(f"{filename} | Triipkoodi ei leitud | Lugemisaeg: {res['elapsed_ms']:.1f} ms")
        else:
            for bc in res["barcodes"]:
                product = bc.get("product")
                if product:
                    print(
                        f"{filename} | EAN13: {bc['ean']} | Toode: {product['name']} | "
                        f"Säilivusaeg: {product['expiry_date_str']} | Lugemisaeg: {res['elapsed_ms']:.1f} ms"
                    )
                else:
                    print(
                        f"{filename} | EAN13: {bc['ean']} | Toode: andmebaasist puudub | "
                        f"Säilivusaeg: puudub | Lugemisaeg: {res['elapsed_ms']:.1f} ms"
                    )

    report_path = os.path.join(report_folder, "triipkoodi_tulemused.txt")
    write_results_report(results, report_path)
    print(f"Tulemused salvestatud: {report_path}")



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
    report_path = os.path.join(report_folder, "triipkoodi_tulemused.txt")
    write_results_report(results, report_path)
    print(f"Tulemused salvestatud: {report_path}")

    stats = build_statistics(results)
    print("\nStatistika")
    print("==========")
    print(f"Takte kokku: {stats['total_images']}")
    print(f"Tuvastatud taktide arv: {stats['detected_images']}")
    print(f"Tuvastusmäär: {stats['detection_rate_percent']:.2f}%")
    print(f"Keskmine triipkoode pildi kohta: {stats['avg_barcodes_per_image']:.3f}")
    if stats['barcode_reading_time']:
        print(f"Keskmine lugemisaeg: {stats['barcode_reading_time']['mean_ms']:.2f} ms")
        print(f"Maksimaalne lugemisaeg: {stats['barcode_reading_time']['max_ms']:.2f} ms")



def main():
    base_folder = STREAM_URL.split('/')[-1]
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_folder = os.path.join("reports", f"{base_folder}_{date_str}")
    ensure_dir(report_folder)

    reader = BarcodeProductReader(
        license_key=DYNAMSOFT_LICENSE,
        template_path=TEMPLATE_PATH,
        product_db_path=PRODUCT_DB_PATH,
        capture_date_str=CAPTURE_DATE_STR,
    )

    print(f"Mode: {MODE}")
    print(f"Video kuupäev: {CAPTURE_DATE_STR}")
    print(f"DEBUG_MODE: {DEBUG_MODE}")

    if MODE == "images":
        run_images_mode(reader, report_folder)
    elif MODE == "rtsp":
        run_rtsp_mode(reader, report_folder)
    else:
        raise ValueError("MODE peab olema 'rtsp' või 'images'")


if __name__ == "__main__":
    main()
