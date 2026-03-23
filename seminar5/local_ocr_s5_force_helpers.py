import os
import csv
import cv2
import glob
import time
from pathlib import Path
from typing import List, Dict, Callable, Any
import sys
import importlib.util

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Force-load the exact helpers.py file from project root.
helpers_path = BASE_DIR / "helpers.py"
spec = importlib.util.spec_from_file_location("seminar5_helpers", helpers_path)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

get_formatted_date = helpers.get_formatted_date

import numpy as np
import easyocr
import torch
from PIL import Image

# -----------------------------
# Seadistused
# -----------------------------

ENABLED_MODELS = [
    "easyocr",
    "parseq",
    "paddleocr",
]

BATCH_SIZE = 4

PRODUCT_FOLDERS = [
    "rulaad",
    "kalkun",
    "veis",
    "salami",
]

EXPECTED_DATES = {
    "rulaad": "15.03.2026",
    "kalkun": "15.03.2026",
    "veis": "15.03.2026",
    "salami": "18.03.2026",
}

PER_IMAGE_REPORT_CSV = BASE_DIR / "ocr_per_image_report.csv"
SUMMARY_REPORT_CSV = BASE_DIR / "ocr_summary_report.csv"
SUMMARY_REPORT_TXT = BASE_DIR / "ocr_summary_report.txt"

print("Laen OCR-mudelid...")

MODEL_INIT_STATUS: Dict[str, Dict[str, str]] = {}

# EasyOCR
try:
    easyocr_reader = easyocr.Reader(["en"])
    MODEL_INIT_STATUS["easyocr"] = {"status": "ok", "detail": ""}
except Exception as e:
    easyocr_reader = None
    MODEL_INIT_STATUS["easyocr"] = {"status": "init_failed", "detail": str(e)}

# PARSeq
parseq_model = None
parseq_img_transform = None
try:
    from strhub.data.module import SceneTextDataModule
    parseq_model = torch.hub.load("baudm/parseq", "parseq", pretrained=True).eval()
    parseq_device = torch.device("cpu")
    parseq_model = parseq_model.to(parseq_device)
    parseq_img_transform = SceneTextDataModule.get_transform(parseq_model.hparams.img_size)
    MODEL_INIT_STATUS["parseq"] = {"status": "ok", "detail": f"device={parseq_device}"}
    print(f"PARSeq mudel töötab seadmel: {parseq_device}")
except Exception as e:
    MODEL_INIT_STATUS["parseq"] = {"status": "init_failed", "detail": str(e)}

# PaddleOCR / PaddleX
paddle_model = None
try:
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "1"
    import paddlex as pdx  # noqa: E402
    paddle_model = pdx.create_model("PP-OCRv5_server_rec")
    MODEL_INIT_STATUS["paddleocr"] = {"status": "ok", "detail": ""}
except Exception as e:
    MODEL_INIT_STATUS["paddleocr"] = {"status": "init_failed", "detail": str(e)}

print("Mudelite laadimine lõpetatud.\n")


def tuvastus_easyocr(image_batch: List[np.ndarray]) -> List[str]:
    if easyocr_reader is None:
        raise RuntimeError(MODEL_INIT_STATUS["easyocr"]["detail"])
    if not image_batch:
        return []
    raw_texts = []
    for img_np_array in image_batch:
        results = easyocr_reader.readtext(img_np_array)
        raw_text = " ".join([res[1] for res in results]) if results else ""
        raw_texts.append(raw_text)
    return raw_texts


def tuvastus_parseq(image_batch: List[np.ndarray]) -> List[str]:
    if parseq_model is None or parseq_img_transform is None:
        raise RuntimeError(MODEL_INIT_STATUS["parseq"]["detail"])
    if not image_batch:
        return []
    pil_images = [Image.fromarray(img_rgb) for img_rgb in image_batch]
    transformed_imgs = [parseq_img_transform(pil_img) for pil_img in pil_images]
    batch_tensor = torch.stack(transformed_imgs).to(parseq_model.device)
    with torch.no_grad():
        logits = parseq_model(batch_tensor)
    pred = logits.softmax(-1)
    labels, _ = parseq_model.tokenizer.decode(pred)
    return list(labels)


def tuvastus_paddleocr(image_batch: List[np.ndarray]) -> List[str]:
    if paddle_model is None:
        raise RuntimeError(MODEL_INIT_STATUS["paddleocr"]["detail"])
    if not image_batch:
        return []
    predictions = list(paddle_model.predict(image_batch))
    raw_texts = []
    for pred in predictions:
        raw_text = ""
        if pred and "rec_text" in pred:
            raw_text = pred.get("rec_text", "")
        raw_texts.append(raw_text)
    return raw_texts


MODEL_FUNCTIONS: Dict[str, Callable[[List[np.ndarray]], List[str]]] = {
    "easyocr": tuvastus_easyocr,
    "parseq": tuvastus_parseq,
    "paddleocr": tuvastus_paddleocr,
}

MODEL_DISPLAY_NAMES = {
    "easyocr": "EasyOCR",
    "parseq": "PARSeq",
    "paddleocr": "PaddleOCR",
}


def get_image_files() -> List[Dict[str, str]]:
    images = []
    for product_folder in PRODUCT_FOLDERS:
        pattern = str(PROJECT_ROOT / product_folder / "date" / "*.png")
        file_list = sorted(glob.glob(pattern))
        if not file_list:
            print(f"Kaustast {pattern} ei leitud PNG-faile.")
            continue
        for file_path in file_list:
            images.append({
                "product": product_folder,
                "file_path": file_path,
                "expected_date": EXPECTED_DATES.get(product_folder, "")
            })
    return images


def split_into_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def classify_result(expected_date: str, predicted_date: str) -> str:
    if not predicted_date or predicted_date == "NA":
        return "tuvastamata"
    if predicted_date == expected_date:
        return "oige"
    return "vale"


def build_init_failed_summary(model_key: str) -> Dict[str, Any]:
    model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    detail = MODEL_INIT_STATUS.get(model_key, {}).get("detail", "")
    return {
        "model": model_name,
        "rows_total": 0,
        "images_processed_by_ocr": 0,
        "correct": 0,
        "wrong": 0,
        "undetected": 0,
        "image_read_errors": 0,
        "accuracy_all_percent": 0.0,
        "accuracy_processed_percent": 0.0,
        "avg_time_sec": 0.0,
        "model_init_status": "init_failed",
        "model_init_detail": detail,
    }


def run_single_model(model_key: str, images: List[Dict[str, str]]) -> Dict[str, Any]:
    if model_key not in MODEL_FUNCTIONS:
        raise ValueError(f"Tundmatu mudel: {model_key}")

    init_info = MODEL_INIT_STATUS.get(model_key, {"status": "unknown", "detail": ""})
    if init_info["status"] != "ok":
        return {"summary": build_init_failed_summary(model_key), "per_image_rows": []}

    model_function = MODEL_FUNCTIONS[model_key]
    model_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
    print(f"--- Käivitan mudeli: {model_name} ---")

    per_image_rows = []
    processed_images = 0
    read_failures = 0
    processing_times = []
    batches = split_into_batches(images, BATCH_SIZE)

    for batch_index, batch in enumerate(batches, start=1):
        image_batch = []
        meta_batch = []
        for item in batch:
            file_path = item["file_path"]
            img = cv2.imread(file_path)
            if img is None:
                read_failures += 1
                per_image_rows.append({
                    "model": model_name,
                    "product": item["product"],
                    "file_path": file_path,
                    "expected_date": item["expected_date"],
                    "raw_text": "",
                    "predicted_date": "NA",
                    "status": "pildi_lugemise_viga",
                    "is_correct": 0,
                    "processing_time_sec": "",
                })
                print(f"Hoiatus: Ei suutnud lugeda pilti {file_path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_batch.append(img_rgb)
            meta_batch.append(item)

        if not image_batch:
            continue

        start_time = time.perf_counter()
        raw_texts = model_function(image_batch)
        end_time = time.perf_counter()
        batch_processing_time = end_time - start_time
        avg_time_per_image = batch_processing_time / len(image_batch)

        if len(raw_texts) != len(meta_batch):
            print(f"Hoiatus: mudel {model_name} tagastas {len(raw_texts)} tulemust, aga oodati {len(meta_batch)}.")

        for idx, item in enumerate(meta_batch):
            raw_text = raw_texts[idx] if idx < len(raw_texts) else ""
            formatted_date = get_formatted_date(raw_text)
            predicted_date = formatted_date if formatted_date else "NA"
            status = classify_result(item["expected_date"], predicted_date)
            is_correct = 1 if status == "oige" else 0
            per_image_rows.append({
                "model": model_name,
                "product": item["product"],
                "file_path": item["file_path"],
                "expected_date": item["expected_date"],
                "raw_text": raw_text,
                "predicted_date": predicted_date,
                "status": status,
                "is_correct": is_correct,
                "processing_time_sec": f"{avg_time_per_image:.6f}",
            })
            processed_images += 1
            processing_times.append(avg_time_per_image)

        print(f"  Partii {batch_index}/{len(batches)} valmis")

    total_rows = len(per_image_rows)
    correct_count = sum(1 for row in per_image_rows if row["status"] == "oige")
    wrong_count = sum(1 for row in per_image_rows if row["status"] == "vale")
    undetected_count = sum(1 for row in per_image_rows if row["status"] == "tuvastamata")
    image_read_error_count = sum(1 for row in per_image_rows if row["status"] == "pildi_lugemise_viga")
    accuracy_all = (correct_count / total_rows * 100.0) if total_rows > 0 else 0.0
    accuracy_processed = (correct_count / processed_images * 100.0) if processed_images > 0 else 0.0
    avg_time = (sum(processing_times) / len(processing_times)) if processing_times else 0.0

    summary = {
        "model": model_name,
        "rows_total": total_rows,
        "images_processed_by_ocr": processed_images,
        "correct": correct_count,
        "wrong": wrong_count,
        "undetected": undetected_count,
        "image_read_errors": image_read_error_count,
        "accuracy_all_percent": accuracy_all,
        "accuracy_processed_percent": accuracy_processed,
        "avg_time_sec": avg_time,
        "model_init_status": "ok",
        "model_init_detail": init_info.get("detail", ""),
    }
    return {"summary": summary, "per_image_rows": per_image_rows}


def save_per_image_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "model", "product", "file_path", "expected_date", "raw_text",
        "predicted_date", "status", "is_correct", "processing_time_sec",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "model", "rows_total", "images_processed_by_ocr", "correct", "wrong",
        "undetected", "image_read_errors", "accuracy_all_percent",
        "accuracy_processed_percent", "avg_time_sec", "model_init_status",
        "model_init_detail",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_copy = dict(row)
            row_copy["accuracy_all_percent"] = f"{row_copy['accuracy_all_percent']:.2f}"
            row_copy["accuracy_processed_percent"] = f"{row_copy['accuracy_processed_percent']:.2f}"
            row_copy["avg_time_sec"] = f"{row_copy['avg_time_sec']:.6f}"
            writer.writerow(row_copy)


def save_summary_txt(rows: List[Dict[str, Any]], output_path: Path) -> None:
    lines = ["OCR mudelite kokkuvõte\n", "=" * 60 + "\n"]
    for row in rows:
        lines.append(f"Mudel: {row['model']}")
        lines.append(f"Mudeli init staatus: {row['model_init_status']}")
        if row.get("model_init_detail"):
            lines.append(f"Mudeli init detail: {row['model_init_detail']}")
        lines.append(f"Pilte kokku raportis: {row['rows_total']}")
        lines.append(f"OCR-i poolt töödeldud pilte: {row['images_processed_by_ocr']}")
        lines.append(f"Õiged: {row['correct']}")
        lines.append(f"Valed: {row['wrong']}")
        lines.append(f"Tuvastamata: {row['undetected']}")
        lines.append(f"Pildi lugemise vead: {row['image_read_errors']}")
        lines.append(f"Täpsus (kõigi ridade suhtes): {row['accuracy_all_percent']:.2f}%")
        lines.append(f"Täpsus (ainult OCR töödeldud piltide suhtes): {row['accuracy_processed_percent']:.2f}%")
        lines.append(f"Keskmine aeg pildi kohta: {row['avg_time_sec']:.6f} s")
        lines.append("-" * 60)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("KOKKUVÕTE")
    print("=" * 80)
    for row in rows:
        print(f"\nMudel: {row['model']}")
        print(f"  Init staatus: {row['model_init_status']}")
        if row.get("model_init_detail"):
            print(f"  Init detail: {row['model_init_detail']}")
        print(f"  Pildid raportis kokku: {row['rows_total']}")
        print(f"  OCR-i poolt töödeldud: {row['images_processed_by_ocr']}")
        print(f"  Õiged: {row['correct']}")
        print(f"  Valed: {row['wrong']}")
        print(f"  Tuvastamata: {row['undetected']}")
        print(f"  Pildi lugemise vead: {row['image_read_errors']}")
        print(f"  Täpsus (kõik read): {row['accuracy_all_percent']:.2f}%")
        print(f"  Täpsus (ainult OCR töödeldud): {row['accuracy_processed_percent']:.2f}%")
        print(f"  Keskmine aeg pildi kohta: {row['avg_time_sec']:.6f} s")


def main():
    images = get_image_files()
    if not images:
        print("Ühtegi pilti ei leitud. Kontrolli kaustastruktuuri.")
        return
    print(f"Leiti kokku {len(images)} pilti.\n")
    all_per_image_rows = []
    all_summary_rows = []
    for model_key in ENABLED_MODELS:
        result = run_single_model(model_key, images)
        all_per_image_rows.extend(result["per_image_rows"])
        all_summary_rows.append(result["summary"])
    save_per_image_csv(all_per_image_rows, PER_IMAGE_REPORT_CSV)
    save_summary_csv(all_summary_rows, SUMMARY_REPORT_CSV)
    save_summary_txt(all_summary_rows, SUMMARY_REPORT_TXT)
    print_summary(all_summary_rows)
    print("\nFailid salvestatud:")
    print(f"  - {PER_IMAGE_REPORT_CSV.resolve()}")
    print(f"  - {SUMMARY_REPORT_CSV.resolve()}")
    print(f"  - {SUMMARY_REPORT_TXT.resolve()}")


if __name__ == "__main__":
    main()
