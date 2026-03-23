# Praktikum 5: OCR-mudelite võrdlus ja kuluanalüüs

Selles praktikumi osas keskendume erinevate optilise märgituvastuse (OCR) mudelite võrdlemisele kuupäevade tuvastamisel piltidelt. Lisaks analüüsime ka pilvepõhiste VLM-mudelite (Vision Language Models) kasutamise kulusid reaalajas süsteemis.

## 1. Paigaldamine ja ettevalmistus

Enne alustamist veendu, et sul on vajalikud teegid paigaldatud. 

```bash
# Aktiveeri oma conda keskkond
conda activate YOURENV

# Paigalda vajalikud teegid
pip install opencv-python numpy pillow

# EasyOCR
pip install easyocr

# PARSeq (strhub)
# strhub vajab PyTorchi. Kui sul on GPU, paigalda see vastavalt oma süsteemile:
# nt. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 (CUDA 11.8)
# CPU versioon:
pip install torch torchvision torchaudio
pip install timm einops strhub

# PaddleOCR (paddlex)
# PaddleX on ehitatud PaddlePaddle peale.
# CPU versioon:
pip install paddlepaddle==2.6.0 -f https://www.paddlepaddle.org.cn/whl/cpu/stable.html # Täpsustatud versioon
pip install paddlex

# OpenRouter (OpenAI-ühilduv API)
pip install openai
```

**Andmete struktuur:**
Veendu, et sul on olemas pildid kaustades `[toode]/date/`, näiteks:
- `rulaad/date/*.png`
- `kalkun/date/*.png`
- `veis/date/*.png`
- `salami/date/*.png`

## 2. Ülesanne 1: Kohalike OCR-mudelite võrdlus (EasyOCR, PARSeq, PaddleOCR)
Sinu ülesanne on käivitada ja võrrelda kolme erinevat kohalikku OCR-mudelit (EasyOCR, PARSeq, PaddleOCR) kuupäevade tuvastamisel.

**Failid:**
- `seminar5/yl1_OCR.py`

**Töövoog:**
1.  Failis `yl1_OCR.py` on implementeeritud kolm tuvastusfunktsiooni: `tuvastus_easyocr()`, `tuvastus_parseq()` ja `tuvastus_paddleocr()`. Need võtavad sisendiks pildi tee ja tagastavad tuvastatud toorteksti.
2.  `main()` funktsioonis on koht, kus saad valida, millist neist kolmest funktsioonist kasutada. Muutes vaid ühte rida koodis, saad vahetada aktiivset OCR-mudelit.
3.  `main()` funktsioon itereerib läbi kõigi `[toode]/date/` kaustade, kutsub välja valitud tuvastusfunktsiooni iga pildi kohta ja kogub statistikat.
4.  Skript kasutab ühist `get_formatted_date` funktsiooni tuvastatud teksti post-töötlemiseks ja valideerimiseks.


**Analüüs:**
Pärast iga skripti käivitamist vaata väljastatud statistikat:
-   **Tuvastatud kuupäevade arv ja sagedus:** Milliseid kuupäevi leiti ja kui mitu korda?
-   **Õnnestumise määr:** Kui suur protsent piltidest andis ÕIGE kuupäeva? Kui suur protsent tuvastatud kuupäevadest on õiged?
-   **Keskmine tuvastusaeg:** Kui kiiresti iga mudel kuupäeva tuvastas?

**Arutelu:**
-   Milline mudel oli kõige täpsem?
-   Milline mudel oli kõige kiirem?
-   Kuidas võiksid need mudelid sobida reaalajas tööstuslikku keskkonda?
-   Soovi korral lisa ise mudelite eksimuste võrdlus - kas need eksivad alati samadel piltidel?



## 3. Ülesanne 2: Pilvepõhiste VLM-mudelite (OpenRouter) võrdlus ja kuluanalüüs

Selles ülesandes kasutame OpenRouteri platvormi kaudu ligipääsetavaid Vision Language mudeleid (VLM) kuupäevade tuvastamiseks ja analüüsime nende kulusid. See skript testib ühte OpenRouteri mudelit korraga kõigi tootekataloogide lõikes.

**Fail:**
- `/home/ardi/projects/ai_project_course_2025/projekt2/seminar5/yl2_OCR.py`

**Ettevalmistus:**
1.  **OpenRouter API võti:** Tuleta meelde oma OpenRouteri API võti.
2.  **Määra API võti:** Asenda `OPENROUTER_API_KEY` muutuja failis `yl2_OCR.py` oma API võtmega või määra see keskkonnamuutujana.
3.  **Vali VLM mudel:** Mine OpenRouteri mudelite lehele, filtreeri "text+image to text" mudelite järgi ja reasta need hinna järgi (odavaimad enne). Vali mõni mudel ja asenda see `OPENROUTER_MODEL` muutuja väärtusega.

**Töövoog:**
1.  Failis `yl2_OCR.py` kasutatakse `helpers.py` failis defineeritud `OpenRouterOCR` klassi, mille meetod `tuvastus_openrouter()` teostab tuvastuse, saates pildi OpenRouteri API-le koos spetsiifilise promptiga kuupäeva ekstraheerimiseks. 
2.  Lisaks tavalisele statistikale kogub skript ka OpenRouteri API kasutusstatistikat (tokenid, maksumus).
3.  Skript arvutab ka hinnangulise päevase kulu, eeldades 16-tunnist tööpäeva, ühte takti iga 7 sekundi järel ja 8 kuupäeva-ala tuvastamist taktis.

**Analüüs:**
-   **Tuvastusstatistika:** Võrdle VLM-i tuvastuse täpsust ja kiirust kohalike mudelitega.
-   **API kasutus:** Mitu tokenit kulus ühe pildi kohta? Mis oli ühe pildi tuvastamise maksumus?
-   **Päevane kulu:** Kui palju maksaks selle mudeli kasutamine 16 tundi päevas antud töökoormuse juures?

**Arutelu:**
-   Kas VLM-mudelid pakuvad paremat täpsust kui kohalikud mudelid? Mis hinnaga?
-   Kas VLM-i kasutamine on reaalajas süsteemi jaoks majanduslikult otstarbekas?
-   Millised on VLM-ide eelised ja puudused võrreldes kohalike OCR-lahendustega?
-   Kuidas mõjutab prompti kvaliteet VLM-i tulemusi?

## 4. Üldine kokkuvõte ja võrdlus

Koosta kokkuvõte kõigi nelja mudeli (EasyOCR, PARSeq, PaddleOCR, OpenRouter VLM) tulemustest. Võrdle nende tugevusi ja nõrkusi täpsuse, kiiruse ja kulude osas. Tee järeldused, milline lahendus sobiks kõige paremini antud ülesande jaoks ja miks.