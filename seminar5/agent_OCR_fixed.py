import os
import csv
import cv2
import glob
import time
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
import importlib.util

# -----------------------------
# Asukohad
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Kasuta täpselt sama helpers.py faili, mida kasutab local_ocr_s5_force_helpers.py
helpers_path = BASE_DIR / "helpers.py"
spec = importlib.util.spec_from_file_location("seminar5_helpers", helpers_path)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

get_formatted_date = helpers.get_formatted_date
OpenRouterOCR = helpers.OpenRouterOCR

# -----------------------------
# Seadistused
# -----------------------------
PRODUCT_FOLDERS = [
    "rulaad",
    "kalkun",
    "veis",
    "salami",
]

# Võta expected_dates samast loogikast nagu local_ocr_s5_force_helpers.py
EXPECTED_DATES = {
    "rulaad": "15.03.2026",
    "kalkun": "15.03.2026",
    "veis": "15.03.2026",
    "salami": "18.03.2026",
}

BATCH_SIZE = 4

# Raportite nimed erinevad local_ocr failidest
PER_IMAGE_REPORT_CSV = BASE_DIR / "openrouter_per_image_report.csv"
SUMMARY_REPORT_CSV = BASE_DIR / "openrouter_summary_report.csv"
SUMMARY_REPORT_TXT = BASE_DIR / "openrouter_summary_report.txt"

# OpenRouter seadistus
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"
OCR_SYSTEM_PROMPT = (
    "You are an expert OCR system. Your task is to extract the expiry date from the provided image. "
    "The date format is DD.MM.YYYY. If you cannot find a date, respond with 'N/A'."
)
OCR_USER_PROMPT_TEXT = (
    "Extract the expiry date from this image. Respond ONLY with the date in DD.MM.YYYY format, "
    "or 'N/A' if no date is found."
)


def load_openrouter_api_key() -> str:
    """
    Loeb OpenRouter võtme peakaustas olevast openrouter_key.env failist.
    Toetab kujusid:
      OPENROUTER_API_KEY=...
      OPENROUTER_KEY=...
      sk-or-v1-...
    """
    env_path = PROJECT_ROOT / "openrouter_key.env"
    if not env_path.exists():
        raise FileNotFoundError(
            f"OpenRouter võtme faili ei leitud: {env_path}. "
            f"Oodatud asukoht on peakaust/openrouter_key.env"
        )

    content = env_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Võtme fail on tühi: {env_path}")

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key in {"OPENROUTER_API_KEY", "OPENROUTER_KEY", "API_KEY"} and value:
                return value
        elif line.startswith("sk-or-v1-"):
            return line

    raise ValueError(
        f"Failist {env_path} ei leitud sobivat OpenRouter API võtit. "
        f"Kasuta näiteks rida OPENROUTER_API_KEY=sk-or-v1-..."
    )


def get_image_files() -> List[Dict[str, str]]:
    images: List[Dict[str, str]] = []
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
                "expected_date": EXPECTED_DATES.get(product_folder, ""),
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


def save_per_image_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "model", "product", "file_path", "expected_date", "raw_text",
        "predicted_date", "status", "is_correct", "processing_time_sec",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_csv(row: Dict[str, Any], output_path: Path) -> None:
    fieldnames = [
        "model", "rows_total", "images_processed_by_ocr", "correct", "wrong",
        "undetected", "image_read_errors", "api_failures", "accuracy_all_percent",
        "accuracy_processed_percent", "avg_time_sec", "total_api_calls",
        "total_prompt_tokens", "total_completion_tokens", "total_tokens", "total_cost",
        "estimated_daily_images", "estimated_daily_cost",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        row_copy = dict(row)
        row_copy["accuracy_all_percent"] = f"{row_copy['accuracy_all_percent']:.2f}"
        row_copy["accuracy_processed_percent"] = f"{row_copy['accuracy_processed_percent']:.2f}"
        row_copy["avg_time_sec"] = f"{row_copy['avg_time_sec']:.6f}"
        row_copy["total_cost"] = f"{row_copy['total_cost']:.6f}"
        row_copy["estimated_daily_cost"] = f"{row_copy['estimated_daily_cost']:.6f}"
        writer.writerow(row_copy)


def save_summary_txt(summary: Dict[str, Any], date_counts: Counter, output_path: Path) -> None:
    lines = [
        "YL2 OpenRouter OCR kokkuvõte",
        "=" * 60,
        f"Mudel: {summary['model']}",
        f"Pilte kokku raportis: {summary['rows_total']}",
        f"OCR-i poolt töödeldud pilte: {summary['images_processed_by_ocr']}",
        f"Õiged: {summary['correct']}",
        f"Valed: {summary['wrong']}",
        f"Tuvastamata: {summary['undetected']}",
        f"Pildi lugemise vead: {summary['image_read_errors']}",
        f"API vead: {summary['api_failures']}",
        f"Täpsus (kõigi ridade suhtes): {summary['accuracy_all_percent']:.2f}%",
        f"Täpsus (ainult OCR töödeldud piltide suhtes): {summary['accuracy_processed_percent']:.2f}%",
        f"Keskmine aeg pildi kohta: {summary['avg_time_sec']:.6f} s",
        "-" * 60,
        "OpenRouter kasutus:",
        f"API kutsed kokku: {summary['total_api_calls']}",
        f"Prompt tokenid kokku: {summary['total_prompt_tokens']:,}",
        f"Completion tokenid kokku: {summary['total_completion_tokens']:,}",
        f"Tokenid kokku: {summary['total_tokens']:,}",
        f"Hinnanguline maksumus: ${summary['total_cost']:.6f}",
        "-" * 60,
        "Hinnanguline päevane kulu:",
        f"Hinnanguline töödeldud piltide arv päevas: {summary['estimated_daily_images']:.0f}",
        f"Hinnanguline päevane kulu: ${summary['estimated_daily_cost']:.6f}",
        "-" * 60,
        "Leitud kuupäevad (loendus):",
    ]

    if date_counts:
        for date, count in date_counts.most_common():
            lines.append(f"{date}: {count}")
    else:
        lines.append("Kuupäevi ei leitud.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    api_key = load_openrouter_api_key()
    openrouter_ocr = OpenRouterOCR(
        api_key=api_key,
        model=OPENROUTER_MODEL,
        system_prompt=OCR_SYSTEM_PROMPT,
        user_prompt=OCR_USER_PROMPT_TEXT,
        app_referer="",
        app_title="",
    )

    images = get_image_files()
    if not images:
        print("Ühtegi pilti ei leitud. Kontrolli kaustastruktuuri.")
        print(f"Oodatud asukoht: {PROJECT_ROOT / 'seminar5' / '<toode>' / 'date' / '*.png'}")
        return

    print(f"--- OpenRouter ({OPENROUTER_MODEL}) Kuupäevade tuvastamine ---")
    print(f"Leiti kokku {len(images)} pilti.\n")

    per_image_rows: List[Dict[str, Any]] = []
    all_found_dates: List[str] = []
    processed_images = 0
    image_read_errors = 0
    api_failures = 0
    processing_times: List[float] = []
    batches = split_into_batches(images, BATCH_SIZE)

    for batch_index, batch in enumerate(batches, start=1):
        image_batch = []
        meta_batch = []

        for item in batch:
            img = cv2.imread(item["file_path"])
            if img is None:
                image_read_errors += 1
                per_image_rows.append({
                    "model": f"OpenRouter ({OPENROUTER_MODEL})",
                    "product": item["product"],
                    "file_path": item["file_path"],
                    "expected_date": item["expected_date"],
                    "raw_text": "",
                    "predicted_date": "NA",
                    "status": "pildi_lugemise_viga",
                    "is_correct": 0,
                    "processing_time_sec": "",
                })
                print(f"Hoiatus: Ei suutnud lugeda pilti {item['file_path']}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_batch.append(img_rgb)
            meta_batch.append(item)

        if not image_batch:
            continue

        start_time = time.perf_counter()
        raw_texts = openrouter_ocr.tuvastus_openrouter(image_batch)
        end_time = time.perf_counter()
        batch_processing_time = end_time - start_time
        avg_time_per_image = batch_processing_time / len(image_batch)

        for idx, item in enumerate(meta_batch):
            raw_text = raw_texts[idx] if idx < len(raw_texts) else None

            if raw_text is None:
                api_failures += 1
                per_image_rows.append({
                    "model": f"OpenRouter ({OPENROUTER_MODEL})",
                    "product": item["product"],
                    "file_path": item["file_path"],
                    "expected_date": item["expected_date"],
                    "raw_text": "",
                    "predicted_date": "NA",
                    "status": "api_viga",
                    "is_correct": 0,
                    "processing_time_sec": f"{avg_time_per_image:.6f}",
                })
                print(f"Hoiatus: OpenRouter API viga pildi {item['file_path']} puhul.")
                continue

            processed_images += 1
            processing_times.append(avg_time_per_image)
            formatted_date = get_formatted_date(raw_text)
            predicted_date = formatted_date if formatted_date else "NA"
            if predicted_date not in {"NA", "N/A", ""}:
                all_found_dates.append(predicted_date)
            status = classify_result(item["expected_date"], predicted_date)
            is_correct = 1 if status == "oige" else 0

            per_image_rows.append({
                "model": f"OpenRouter ({OPENROUTER_MODEL})",
                "product": item["product"],
                "file_path": item["file_path"],
                "expected_date": item["expected_date"],
                "raw_text": raw_text,
                "predicted_date": predicted_date,
                "status": status,
                "is_correct": is_correct,
                "processing_time_sec": f"{avg_time_per_image:.6f}",
            })

        print(f"  Partii {batch_index}/{len(batches)} valmis")

    rows_total = len(per_image_rows)
    correct = sum(1 for row in per_image_rows if row["status"] == "oige")
    wrong = sum(1 for row in per_image_rows if row["status"] == "vale")
    undetected = sum(1 for row in per_image_rows if row["status"] == "tuvastamata")
    accuracy_all = (correct / rows_total * 100.0) if rows_total > 0 else 0.0
    accuracy_processed = (correct / processed_images * 100.0) if processed_images > 0 else 0.0
    avg_time = (sum(processing_times) / len(processing_times)) if processing_times else 0.0

    usage_totals = openrouter_ocr.get_usage_totals()
    total_api_calls = int(usage_totals.get("calls", 0))
    total_prompt_tokens = int(usage_totals.get("prompt_tokens", 0))
    total_completion_tokens = int(usage_totals.get("completion_tokens", 0))
    total_tokens = int(usage_totals.get("total_tokens", 0))
    total_cost = float(usage_totals.get("cost", 0.0))

    estimated_daily_images = 0.0
    estimated_daily_cost = 0.0
    if processed_images > 0 and total_cost > 0:
        workday_hours = 16
        takt_interval_seconds = 7
        date_areas_per_takt = 8
        total_takt_per_day = (workday_hours * 3600) / takt_interval_seconds
        estimated_daily_images = total_takt_per_day * date_areas_per_takt
        cost_per_image = total_cost / processed_images
        estimated_daily_cost = cost_per_image * estimated_daily_images

    summary = {
        "model": f"OpenRouter ({OPENROUTER_MODEL})",
        "rows_total": rows_total,
        "images_processed_by_ocr": processed_images,
        "correct": correct,
        "wrong": wrong,
        "undetected": undetected,
        "image_read_errors": image_read_errors,
        "api_failures": api_failures,
        "accuracy_all_percent": accuracy_all,
        "accuracy_processed_percent": accuracy_processed,
        "avg_time_sec": avg_time,
        "total_api_calls": total_api_calls,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "estimated_daily_images": estimated_daily_images,
        "estimated_daily_cost": estimated_daily_cost,
    }

    date_counts = Counter(all_found_dates)

    save_per_image_csv(per_image_rows, PER_IMAGE_REPORT_CSV)
    save_summary_csv(summary, SUMMARY_REPORT_CSV)
    save_summary_txt(summary, date_counts, SUMMARY_REPORT_TXT)

    print("\n" + "=" * 80)
    print("KOKKUVÕTE")
    print("=" * 80)
    print(f"Mudel: {summary['model']}")
    print(f"Pildid raportis kokku: {summary['rows_total']}")
    print(f"OCR-i poolt töödeldud: {summary['images_processed_by_ocr']}")
    print(f"Õiged: {summary['correct']}")
    print(f"Valed: {summary['wrong']}")
    print(f"Tuvastamata: {summary['undetected']}")
    print(f"Pildi lugemise vead: {summary['image_read_errors']}")
    print(f"API vead: {summary['api_failures']}")
    print(f"Täpsus (kõik read): {summary['accuracy_all_percent']:.2f}%")
    print(f"Täpsus (ainult OCR töödeldud): {summary['accuracy_processed_percent']:.2f}%")
    print(f"Keskmine aeg pildi kohta: {summary['avg_time_sec']:.6f} s")
    print(f"API kutsed kokku: {summary['total_api_calls']}")
    print(f"Prompt tokenid kokku: {summary['total_prompt_tokens']:,}")
    print(f"Completion tokenid kokku: {summary['total_completion_tokens']:,}")
    print(f"Tokenid kokku: {summary['total_tokens']:,}")
    print(f"Hinnanguline maksumus: ${summary['total_cost']:.6f}")
    print("\nFailid salvestatud:")
    print(f"  - {PER_IMAGE_REPORT_CSV.resolve()}")
    print(f"  - {SUMMARY_REPORT_CSV.resolve()}")
    print(f"  - {SUMMARY_REPORT_TXT.resolve()}")


if __name__ == "__main__":
    main()
