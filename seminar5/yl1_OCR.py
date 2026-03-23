import os
import re
import cv2
import numpy as np
import time
import glob
from collections import Counter
from typing import List

import easyocr
import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from helpers import get_formatted_date


# --- Kolme katsetatava mudeli defineerimised ---

# 1. EasyOCR Reader
# Võimalik, et pead määrama keelte loendi (lang_list) vastavalt oodatavatele keeltele, nt ['en', 'et']
easyocr_reader = easyocr.Reader(['en']) 

# 2. PARSeq Model
# Jäta vahele ühenduvuse kontrollid ja telemeetria
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
parseq_model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
device = torch.device("cpu") # Määra seade selgesõnaliselt CPU-le
parseq_model = parseq_model.to(device) # Liiguta mudel CPU-le 
print(f"PARSeq mudel töötab seadmel: {device}")
parseq_img_transform = SceneTextDataModule.get_transform(parseq_model.hparams.img_size)

# 3. PaddleOCR Model
import paddlex as pdx
# Näide seadistamisest skriptis (ei pruugi olla efektiivne kõigi PaddlePaddle komponentide jaoks):
os.environ['OMP_NUM_THREADS'] = "1" # Määra kõikidele saadaolevatele CPU tuumadele
#os.environ['MKL_NUM_THREADS'] = '8'
#os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0' # Inteli CPU-de jaoks, kui MKL-i kasutatakse
paddle_model = pdx.create_model("PP-OCRv5_server_rec") # proovi, mis juhtub mobiiliversiooniga

#----------------------------------------------

# --- OCR Mudeli Valik ---
# Vali, millist OCR-mudelit selleks käivituseks kasutada: "easyocr", "parseq" või "paddleocr"
OCR_MODEL_CHOICE = "paddleocr" # <--- MUUDA SEDA MUDELITE VAHETAMISEKS
BATCH_SIZE = 4 # Piltide arv, mida töödelda ühes partiis


# Töötle kaustu eraldi või kõiki koos (NB! Salamil on erinev aegumiskuupäev!)
product_folders = ["rulaad", "kalkun", "veis", "salami"] 
#--------------------------------------------------
def tuvastus_easyocr(image_batch: List[np.ndarray]) -> List[str]:
    """
    Performs OCR using EasyOCR on a batch of images.
    Returns a list of raw extracted texts, one for each image in the batch.
    """
    if not image_batch:
        return []
    
    raw_texts = []
    # EasyOCR-i readtext ootab töötlemiseks korraga ühte pilti (numpy massiivi).
    # Me itereerime läbi partii ja töötleme iga pilti eraldi.
    for img_np_array in image_batch:
        results = easyocr_reader.readtext(img_np_array)
        # Ühenda kõik tuvastatud tekstid praeguse pildi jaoks.
        raw_texts.append(" ".join([res[1] for res in results]))
    return raw_texts

def tuvastus_parseq(image_batch: List[np.ndarray]) -> List[str]:
    """
    Performs OCR using the pre-loaded PARSeq model on a batch of images.
    Returns a list of raw extracted texts, one for each image in the batch.
    """
    if not image_batch:
        return []

    pil_images = [Image.fromarray(img_rgb) for img_rgb in image_batch]
    
    # Rakenda teisendus ja lisa partii dimensioon (mudel ootab partiid).
    transformed_imgs = [parseq_img_transform(pil_img) for pil_img in pil_images]
    batch_tensor = torch.stack(transformed_imgs).to(parseq_model.device) # Liiguta sisendtenser mudeliga samale seadmele.
    
    with torch.no_grad():
        logits = parseq_model(batch_tensor)
    
    # Dekodeeri kõrgeima tõenäosusega märgid kogu partii jaoks.
    pred = logits.softmax(-1)
    labels, _ = parseq_model.tokenizer.decode(pred)
    
    return labels # labels is already a list of strings


def tuvastus_paddleocr(image_batch: List[np.ndarray]) -> List[str]:
    """
    Performs OCR using the pre-loaded PaddleOCR recognition model on a batch of images.
    Returns a list of raw extracted texts, one for each image in the batch.
    """
    if not image_batch:
        return []
    
    # 'predict' PaddleX-is ootab partiitöötluseks piltide loendit (NumPy massiive).
    predictions = list(paddle_model.predict(image_batch)) # Kasuta globaalselt defineeritud paddle_model-it.
    
    raw_texts = []
    for pred in predictions:
        raw_text = ""
        if pred and 'rec_text' in pred: # Kontrolli, kas ennustus on olemas ja sisaldab 'rec_text' võtit
            raw_text = pred.get('rec_text', "") # Ekstraheeri tuvastatud tekst.
        raw_texts.append(raw_text)
    return raw_texts

def main():
    ocr_function = None
    model_name_for_print = ""

    if OCR_MODEL_CHOICE == "easyocr":
        ocr_function = tuvastus_easyocr
        model_name_for_print = "EasyOCR"
    elif OCR_MODEL_CHOICE == "parseq":
        ocr_function = tuvastus_parseq
        model_name_for_print = "PARSeq"
    elif OCR_MODEL_CHOICE == "paddleocr":
        ocr_function = tuvastus_paddleocr
        model_name_for_print = "PaddleOCR"
    else:
        print(f"Viga: Kehtetu OCR_MODEL_CHOICE '{OCR_MODEL_CHOICE}'. Palun vali 'easyocr', 'parseq' või 'paddleocr'.")
        return
    
    print(f"--- {model_name_for_print} Kuupäevade tuvastamine ---")


    kõik_leitud_kuupäevad = []
    töödeldud_piltide_arv_kokku = 0
    töötlusajad = [] # Iga pildi OCR-i ennustamiseks kulunud aja salvestamiseks

    for product_folder in product_folders:
        folder_path = f"{product_folder}/date/*.png"
        file_list = glob.glob(folder_path)

        if not file_list:
            print(f"Kaustast {folder_path} ei leitud PNG-faile.")
            continue

        print(f"\nTöötlen kausta: {product_folder} ({len(file_list)} pilti)")

        current_image_batch = []
        current_image_paths = [] # To keep track of paths for warnings/errors

        for file_path in file_list:
            img = cv2.imread(file_path)
            if img is None:
                print(f"Hoiatus: Ei suutnud lugeda {file_path}, jätan OCR-i vahele.")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            current_image_batch.append(img_rgb)
            current_image_paths.append(file_path)

            if len(current_image_batch) == BATCH_SIZE:
                # Töötle partiid
                start_time = time.perf_counter()
                raw_texts = ocr_function(current_image_batch)
                end_time = time.perf_counter()
                
                batch_processing_time = end_time - start_time
                avg_time_per_image = batch_processing_time / len(current_image_batch)
                for i, raw_text in enumerate(raw_texts):
                    töötlusajad.append(avg_time_per_image) 
                    töödeldud_piltide_arv_kokku += 1
                    
                    formatted_date = get_formatted_date(raw_text)
                    if formatted_date:
                        kõik_leitud_kuupäevad.append(formatted_date)
                    else:
                        # Kui kuupäeva ei leitud, lisa see "NA" või muu tähisega, et loenduses kajastuks
                        kõik_leitud_kuupäevad.append("NA")
                
                # Lähtesta partii
                current_image_batch = []
                current_image_paths = []

        # Töötle kõik järelejäänud pildid viimases (osalises) partiis.
        if current_image_batch:
            start_time = time.perf_counter()
            raw_texts = ocr_function(current_image_batch)
            end_time = time.perf_counter()

            batch_processing_time = end_time - start_time
            avg_time_per_image = batch_processing_time / len(current_image_batch)

            for i, raw_text in enumerate(raw_texts):
                töötlusajad.append(avg_time_per_image)
                töödeldud_piltide_arv_kokku += 1

                formatted_date = get_formatted_date(raw_text)
                if formatted_date:
                    kõik_leitud_kuupäevad.append(formatted_date)
                else:
                    kõik_leitud_kuupäevad.append("NA")

    # Statistika loogika
    kuupäevade_loendused = Counter(kõik_leitud_kuupäevad)

    print(f"\n--- Üldine OCR-statistika ({model_name_for_print}) ---")
    print(f"OCR-i poolt töödeldud pilte kokku: {töödeldud_piltide_arv_kokku}")

    if töötlusajad:
        ocr_töötlusaeg_kokku = sum(töötlusajad)
        if töödeldud_piltide_arv_kokku > 0:
            print(f"Keskmine OCR-i töötlusaeg pildi kohta: {ocr_töötlusaeg_kokku / töödeldud_piltide_arv_kokku:.4f} sekundit")
        else:
            print("OCR-ile ei edastatud ühtegi pilti, seega keskmist aega pildi kohta ei saa arvutada.")

    print("\n--- Leitud unikaalsed kuupäevad (loendus) ---")
    for kuupäev, loendus in kuupäevade_loendused.most_common():
        print(f"{kuupäev}: {loendus} korda")
if __name__ == "__main__":
    main()
