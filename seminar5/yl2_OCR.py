import os
import cv2
import numpy as np
import time
import glob
from collections import Counter
from typing import List, Dict, Any

from helpers import get_formatted_date, OpenRouterOCR


# --- OpenRouter API Configuration ---
# Määra oma OpenRouter API võti keskkonnamuutujana või asenda kohatäide.
# API võtmete jaoks on tungivalt soovitatav kasutada keskkonnamuutujaid.
OPENROUTER_API_KEY = "sk-or-v1-NNNNNNNNNNNNNNNNNNNNNNNNNNNN"
OPENROUTER_API_KEY = "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"

# Vali oma VLM mudel OpenRouteri kaudu.
# Mudeleid ja nende hinnakujundust leiad aadressilt https://openrouter.ai/docs#models
# Näited:
# "google/gemini-2.0-flash-001" (Gemini 1.5 Flash)
# "qwen/qwen3.5-flash-02-23" (Qwen 3.5 Flash)
# "meta-llama/llama-3.2-11b-vision-instruct" (Llama 3.2 Vision)
OPENROUTER_MODEL = "google/gemini-2.0-flash-001" # Default model, change as needed

# Defineeri LLM-i viip kuupäeva eraldamiseks
OCR_SYSTEM_PROMPT = "You are an expert OCR system. Your task is to extract the expiry date from the provided image. The date format is DD.MM.YYYY. If you cannot find a date, respond with 'N/A'."
OCR_USER_PROMPT_TEXT = "Extract the expiry date from this image. Respond ONLY with the date in DD.MM.YYYY format, or 'N/A' if no date is found."
# ------------------------------------

#OLULINE: vaata helpers.py, seal on defineeritud keelemudeli kasutuse loogika.
# def tuvastus_openrouter() on kõige olulisem funktsioon

def main():
    print(f"--- OpenRouter ({OPENROUTER_MODEL}) Kuupäevade tuvastamine ---")

    # Töödeldavate toote kaustade loend
    product_folders = ["rulaad", "kalkun", "veis", "salami"] # Vastavalt README.md-le

    BATCH_SIZE = 4 # Piltide arv, mida töödelda ühes partiis
    all_found_dates = []
    total_files_in_all_folders = 0
    total_images_processed = 0
    total_ocr_successes = 0
    total_ocr_failures = 0
    total_cv2_read_failures = 0 # Pildid, mida cv2.imread ei suutnud lugeda või API viga
    processing_times = [] # Iga pildi OCR-i ennustamiseks kulunud aja salvestamiseks

    # Initsialiseeri OpenRouter OCR klass
    openrouter_ocr = OpenRouterOCR(
        api_key=OPENROUTER_API_KEY,
        model=OPENROUTER_MODEL,
        system_prompt=OCR_SYSTEM_PROMPT,
        user_prompt=OCR_USER_PROMPT_TEXT,
        app_referer="https://ai-project-course-2025.com", # Asenda oma rakenduse URL-iga
        app_title="AI Project Course OCR",
    )

    for product_folder in product_folders:
        folder_path = f"{product_folder}/date/*.png"
        file_list = glob.glob(folder_path)
        total_files_in_all_folders += len(file_list)

        if not file_list:
            print(f"Kaustast {folder_path} ei leitud PNG-faile.")
            continue

        print(f"\nTöötlen kausta: {product_folder} ({len(file_list)} pilti)")

        current_image_batch = []
        current_image_paths = [] # To keep track of paths for warnings/errors

        for file_path in file_list:
            img = cv2.imread(file_path)
            if img is None:
                total_cv2_read_failures += 1
                print(f"Hoiatus: Ei suutnud lugeda {file_path}, jätan OCR-i vahele.")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            current_image_batch.append(img_rgb)
            current_image_paths.append(file_path)

            if len(current_image_batch) == BATCH_SIZE:
                # Töötle partiid
                start_time = time.perf_counter()
                raw_texts = openrouter_ocr.tuvastus_openrouter(current_image_batch) # Kutsu OpenRouter OCR funktsioon välja
                end_time = time.perf_counter()
                
                batch_processing_time = end_time - start_time
                avg_time_per_image = batch_processing_time / len(current_image_batch)

                for i, raw_text in enumerate(raw_texts):
                    # Töötle iga tulemust partiist
                    if raw_text is None: # API viga selle konkreetse pildi puhul
                        total_cv2_read_failures += 1 # Kasutan seda loendurit ka API vigade jaoks
                        print(f"Hoiatus: OpenRouter API viga pildi {current_image_paths[i]} puhul, jätan OCR-i vahele.")
                        continue

                    processing_times.append(avg_time_per_image) 
                    total_images_processed += 1
                    
                    formatted_date = get_formatted_date(raw_text)
                    if formatted_date:
                        all_found_dates.append(formatted_date)
                        total_ocr_successes += 1
                    else:
                        total_ocr_failures += 1
                
                # Lähtesta partii
                current_image_batch = []
                current_image_paths = []

        # Töötle kõik järelejäänud pildid viimases (osalises) partiis
        if current_image_batch:
            start_time = time.perf_counter()
            raw_texts = openrouter_ocr.tuvastus_openrouter(current_image_batch)
            end_time = time.perf_counter()

            batch_processing_time = end_time - start_time
            avg_time_per_image = batch_processing_time / len(current_image_batch)

            for i, raw_text in enumerate(raw_texts):
                if raw_text is None: # API viga selle konkreetse pildi puhul
                    total_cv2_read_failures += 1
                    print(f"Hoiatus: OpenRouter API viga pildi {current_image_paths[i]} puhul, jätan OCR-i vahele.")
                    continue

                processing_times.append(avg_time_per_image)
                total_images_processed += 1

                formatted_date = get_formatted_date(raw_text)
                if formatted_date:
                    all_found_dates.append(formatted_date)
                    total_ocr_successes += 1
                else:
                    total_ocr_failures += 1

    # Statistika loogika
    date_counts = Counter(all_found_dates)
    success_rate_ocr_attempted = (total_ocr_successes / total_images_processed) * 100 if total_images_processed > 0 else 0

    print(f"\n--- Üldine OCR-statistika (OpenRouter - {openrouter_ocr.get_model_name()}) ---")
    print(f"Kõikidest kaustadest leitud faile kokku: {total_files_in_all_folders}")

    total_ocr_processing_time = sum(processing_times)
    if total_images_processed > 0:
        print(f"Keskmine OCR-i töötlusaeg pildi kohta: {total_ocr_processing_time / total_images_processed:.4f} sekundit")
    else:
        print("OCR-i poolt töödeldud pilte ei olnud, keskmist aega ei saa arvutada.")

    print("\n--- Leitud unikaalsed kuupäevad (loendus) ---")
    for date, count in date_counts.most_common():
        print(f"{date}: {count} times")

    print("\n--- OpenRouter API kasutus kokkuvõte ---")
    print(f"Kasutatud mudel: {openrouter_ocr.get_model_name()}")
    usage_totals = openrouter_ocr.get_usage_totals()
    print(f"Total API Calls: {usage_totals['calls']}")
    print(f"Total Prompt Tokens: {usage_totals['prompt_tokens']:,}")
    print(f"Total Completion Tokens: {usage_totals['completion_tokens']:,}")
    print(f"Total Tokens: {usage_totals['total_tokens']:,}")
    print(f"Hinnanguline maksumus (OpenRouter krediidid): ${usage_totals['cost']:.6f}")
    # print(f"Viimase kõne info: {usage_totals['last_call_info']}") # See võib olla väga verbaalne, ehk ainult silumiseks

    if total_images_processed > 0 and usage_totals['cost'] > 0:
        cost_per_image = usage_totals['cost'] / total_images_processed
        
        # Eeldused päevase kulu arvutamiseks (README.md-st)
        workday_hours = 16 # Tööpäeva pikkus tundides
        takt_interval_seconds = 7 # Taktide vahe sekundites
        date_areas_per_takt = 8 # Eeldades, et taktis töödeldakse 8 kuupäeva ala

        total_takt_per_day = (workday_hours * 3600) / takt_interval_seconds # Taktide arv päevas
        estimated_images_per_day = total_takt_per_day * date_areas_per_takt # Hinnanguline piltide arv päevas
        estimated_daily_cost = cost_per_image * estimated_images_per_day # Hinnanguline päevane kulu

        print("\n--- Hinnanguline päevane kulu (praeguse käivituse põhjal) ---")
        print(f"Eeldused: {workday_hours}h tööpäev, 1 takt iga {takt_interval_seconds}s järel, {date_areas_per_takt} kuupäeva ala takti kohta.")
        print(f"Hinnanguline töödeldud piltide arv päevas: {estimated_images_per_day:.0f}")
        print(f"Hinnanguline päevane kulu: ${estimated_daily_cost:.6f}")
    else:
        print("\n--- Hinnanguline päevane kulu ---")
        print("Päevast kulu ei saa hinnata, kuna pilte ei töödeldud või kuluteave puudub.")

if __name__ == "__main__":
    main()
