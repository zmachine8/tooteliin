import cv2
import os
import time
import json
import numpy as np
from datetime import datetime, timedelta
from dynamsoft_barcode_reader_bundle import *
from helpers import RTSPStreamReader, is_green_screen, measure_global_change

# --- KONFIGURATSIOON ---
STREAM_URL = "rtsp://172.17.37.81:8554/salami"
MOTION_THRESHOLD = 15.0
CAPTURE_DELAY = 2.5
DYNAMSOFT_LICENSE = "t0084YQEAAIUx4hU4EqEOu9FaT9GprNtmXmbGA7IcvmG7V7l1yrR4WjV1JWPPrLuJoJN4HXVvqroIag2MeSFUJlbpkh0vhl8/Nrk3lffN1GzB7BvBtkl5"
DEBUG_MODE = True  # Silumisrežiim piltide salvestamiseks
capture_date = datetime(2026, 2, 14)

# Initsialiseerimine
LicenseManager.init_license(DYNAMSOFT_LICENSE)
router = CaptureVisionRouter()
template_path = "medium_template.json"
err_code, err_msg = router.init_settings_from_file(template_path)
if err_code != EnumErrorCode.EC_OK:
    print(f"Failed to load JSON settings from file: {err_msg}")
    exit()

with open('barcode_data.json', 'r') as f:
    barcode_data = json.load(f)

folder_name = STREAM_URL.split('/')[-1]
os.makedirs(folder_name, exist_ok=True)

stream = RTSPStreamReader(STREAM_URL)
time.sleep(2)

# --- Statistika muutujad ---
stats_times = []
success_count = 0
stats_counts = []
total_triggers = 0
started = False
green_cooldown = False
motion_triggered = False
trigger_time = 0
cycle_start_time = 0
current_product = None # we maintain a "current product", if a given frame does not have barcode, we proceed with last ean
required_keys = ["rois", "date_area", "label1_below", "label2_above", "product_area_between"]

print(f"Ühendatud vooga {STREAM_URL}. Ootan rohelist märguannet...")

try:
    while True:
        start_t = time.perf_counter()

        ret1, frame1 = stream.read()
        time.sleep(0.02) # Väike paus, et kaadrid jõuaksid muutuda
        ret2, frame2 = stream.read()
        if not ret1 or not ret2: break

        now = time.time()
        current_is_green = is_green_screen(frame2)
        
        if not started:
            if current_is_green:
                print(">>> Roheline ekraan tuvastatud! Alustame tsüklit.")
                started = True
                green_cooldown = True
        else:
            process_this_frame = False
            if not current_is_green and green_cooldown:
                print(">>> Roheline ekraan lõppes. Alustame monitooringut.")
                green_cooldown = False
                cycle_start_time = now
                total_triggers += 1
                process_this_frame = True
            elif not green_cooldown:
                if current_is_green:
                    print(">>> Järgmine roheline ekraan tuvastatud. Lõpetan tsükli.")
                    break

                # Mõõdame liikumist
                mae = measure_global_change(frame1, frame2)
                if mae > MOTION_THRESHOLD and not motion_triggered:
                    motion_triggered = True
                    trigger_time = now
                    total_triggers += 1
                    print(f"Liikumine tuvastatud {now - cycle_start_time:.2f}s juures! Ootan {CAPTURE_DELAY}s...")

                if motion_triggered and (now - trigger_time >= CAPTURE_DELAY):
                    process_this_frame = True
                    motion_triggered = False

            if process_this_frame:
                elapsed = now - cycle_start_time
                result = router.capture(frame2, "ReadBarcodes_Default")

                items = result.get_items() if result is not None else None
                barcodes = [item.get_text() for item in items or [] if item.get_type() == EnumCapturedResultItemType.CRIT_BARCODE]
                stats_counts.append(len(barcodes))

                # Get info from barcode_data.json
                if len(barcodes)>0:
                    success_count += 1
                    ean = barcodes[0] #just take the first one
                    product = barcode_data.get(ean) 
                    if product is None:
                        print(f"[{elapsed:.2f}s] EAN {ean}: tooteinfot ei leitud.")
                        continue
                    # Uuendame globaalset tooteinfot ja salvestame EAN-i printimiseks
                    current_product = product.copy()
                    current_product["_ean"] = ean                        
                else:
                    print(f"[{elapsed:.2f}s] Triipkoode ei leitud.")

                # Kui meil pole varasemast taktist toodet ja praegu ei leidnud, siis katkestame
                if current_product is None:
                    print("Meil pole current_product teada. Katkestame")
                    continue
                # Kontrollime, et kõik vajalikud koordinaadid on olemas
                if not all(k in current_product for k in required_keys):
                    print(f"VIGA: Tooteinfot EAN {current_product.get('_ean')} on puudulik!")
                    continue
                    
                product_name = current_product.get("ITEMNAME", "Tundmatu toode")
                expiry_duration = current_product.get("BESTBEFOREDAYS", 0)
                expiry_date = capture_date + timedelta(days=expiry_duration)
                ean_str = current_product.get("_ean", "Tundmatu")
                print(f"[{elapsed:.2f}s] Kontekst: EAN {ean_str} | {product_name} | Aegub {expiry_date.strftime('%d.%m.%Y')}")

                # Siit algab kaadri juppideks lõikamise loogika
                temp_slices = []
                for pkg_id, coords in current_product["rois"].items():
                    x1, y1 = coords[0]
                    x2, y2 = coords[1]
                    roi_crop = frame2[y1:y2, x1:x2]
                    # Roteerime 90 kraadi vastupäeva
                    rotated = cv2.rotate(roi_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    temp_slices.append(rotated)                    

                max_h = max(s.shape[0] for s in temp_slices)
                max_w = max(s.shape[1] for s in temp_slices)
                final_slices = [cv2.resize(s, (max_w, max_h)) for s in temp_slices]
                
                end_t = time.perf_counter()
                stats_times.append((end_t - start_t) * 1000)

                if DEBUG_MODE:
                    images_to_save = []
                    for sub in ["date", "label1", "label2", "product_area","individual_products","full_frames"]:
                        os.makedirs(os.path.join(folder_name, sub), exist_ok=True)
                    images_to_save.append((os.path.join(folder_name,"full_frames", f"takt_{total_triggers}_full.png"), frame2))

                    for i, s_img in enumerate(final_slices):
                        s_idx = i + 1
                        images_to_save.append((os.path.join(folder_name,"individual_products", f"takt_{total_triggers}_slice_{s_idx}.png"), s_img))
                            
                        # All-in-one lõikamine ja salvestusnimekirja lisamine
                        dx1, dy1 = current_product["date_area"][0]
                        dx2, dy2 = current_product["date_area"][1]
                        images_to_save.append((os.path.join(folder_name, "date", f"takt_{total_triggers}_s{s_idx}_date.png"), s_img[dy1:dy2, dx1:dx2]))
                                
                        images_to_save.append((os.path.join(folder_name, "label1", f"takt_{total_triggers}_s{s_idx}_l1.png"), s_img[current_product["label1_below"]:, :]))
                        images_to_save.append((os.path.join(folder_name, "label2", f"takt_{total_triggers}_s{s_idx}_l2.png"), s_img[:current_product["label2_above"], :]))
                            
                        py1, py2 = current_product["product_area_between"]
                        images_to_save.append((os.path.join(folder_name, "product_area", f"takt_{total_triggers}_s{s_idx}_prod.png"), s_img[py1:py2, :]))
                
                    save_start_t = time.perf_counter()
                    for path, img in images_to_save:
                        if img is not None and img.size > 0:
                            cv2.imwrite(path, img)
                    print(f"Piltide salvestamine võttis {(time.perf_counter() - save_start_t)*1000:.2f} ms")
                print(f"Takt {total_triggers} töödeldud.")

except KeyboardInterrupt:
    print("Peatatud.")

finally:
    stream.stop()
    
    # --- KODUTÖÖ: Kokkuvõte ---
    success_rate = (success_count / total_triggers * 100) if total_triggers else 0.0
    avg_time = (sum(stats_times) / len(stats_times)) if stats_times else 0.0
    max_time = max(stats_times) if stats_times else 0.0
    avg_barcodes = (sum(stats_counts) / len(stats_counts)) if stats_counts else 0.0
    print("\n--- KOKKUVÕTE ---")
    print(f"Total triggers: {total_triggers}")
    print(f"Triipkoodi tuvastamise õnnestumise protsent: {success_rate:.2f}%")
    print(f"Keskmine tuvastamise aeg: {avg_time:.2f} ms")
    print(f"Maksimaalne tuvastamise aeg: {max_time:.2f} ms")
    print(f"Keskmine leitud triipkoodide arv takti kohta: {avg_barcodes:.2f}")    # 1. Tuvastamise õnnestumise protsent (success_rate).
    
