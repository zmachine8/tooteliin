# Seminar 7: DINOv2 tootetuvastus ja RTSP toru

Selles seminaris kasutame DINOv2 pildivektoreid, et võrrelda tootepilte omavahel, vähendada dimensioone visualiseerimiseks ning ehitada kaks lihtsat klassifikaatorit: KNN ja logistiline regressioon. Lõpus ühendame sildituvastuse, tootetuvastuse ja kuupäevatuvastuse RTSP töövoogu.

Selles kaustas on 5 tooteklassi:
- `empty`
- `kalkun`
- `rulaad`
- `salami`
- `veis`

Iga klassi sees on olemas `product_area` pildid. Lisaks on toodetel olemas ka `label1` ja `label2` näited, mida RTSP töövoos kasutatakse.



## 2. KNN ehk lähima naabri klassifikaator

Selles osas kasutame iga toote kohta ühte või mitut näidispilti. Kõik ülejäänud pildid klassifitseeritakse selle järgi, millistele näidistele nad vektoresituse ruumis kõige lähemal on.

**Fail:**
- `toodete_tuvastamine.py`

Olulised seadistused faili alguses:
- `template_count` määrab, mitu esimest pilti võetakse igast klassist näidisteks
- `k` määrab, mitu lähimat naabrit võetakse ennustamisel arvesse

Käivitus:

```bash
python toodete_tuvastamine.py
```

Skript:
- kasutab mudelit `facebook/dinov2-small`
- võtab igast klassist esimesed `template_count` pilti näidisteks
- võrdleb ülejäänud pilte kõigi näidistega kosinussarnasuse abil
- prindib iga klassi tulemused
- salvestab täpsuse graafiku faili

Näiteks:

```text
classification_accuracy_templates4_knn1.png
```

**Ülesanne:**
- käivita kood erinevate `k` väärtustega
- proovi erinevaid `template_count` väärtusi
- vaata, kui kiiresti tulemus paraneb, kui näidiseid juurde lisada
- mõtle, kuidas muutub töö kiirus, kui võrdlemiseks on rohkem näidiseid


## 3. Logistiline regressioon vektoresituste peal

Selles osas kasutame samu DINOv2 esitusi, aga nüüd õpetame nende peal eraldi klassifitseerija.

**Fail:**
- `logistic_regression.py`

Oluline seadistus faili alguses:
- `template_count` määrab, mitu esimest pilti läheb igast klassist treenimisandmestikku.

Käivitus:

```bash
python logistic_regression.py
```

Skript:
- teeb treeningpiltidest DINOv2 embeddingud
- õpetab `LogisticRegression` mudeli
- testib ülejäänud piltidel
- salvestab mudeli `.joblib` faili
- salvestab täpsuse graafiku faili

Näiteks:

```text
logistic_regression_templates20.joblib
classification_accuracy_logistic_regression_templates20.png
```

**Ülesanne:**
- käivita logistilise regressiooni kood
- proovi erinevaid `template_count` väärtusi
- võrdle täpsust ja kiirust KNN-iga

## 5. RTSP töövoog

Lõpuks paneme kokku:
- triipkoodi lugemise
- kuupäeva OCR-i
- sildituvastuse
- tootetuvastuse

**Fail:**
- `RTSP_threadded.py`

See skript:
- loeb RTSP striimi
- leiab taktid liikumise ja rohelise ekraani abil
- loeb triipkoodi
- lõikab igast taktis 4 tootepilti välja
- kontrollib kuupäeva, silte ja toodet
- väljastab koondraporti

Enne käivitamist vaata üle faili alguses olevad seadistused:
- `STREAM_URL`
- `DATE_OCR_METHOD`
- `OPENROUTER_API_KEY`, kui kasutad VLM-i
- `OPENROUTER_MODEL`
- `PRODUCT_MODEL_PATH`
- `DEBUG_MODE`

Käivitus:

```bash
python RTSP_threadded.py
```

**Ülesanne:**
- aruta, milline võiks olla sildikauguse mõistlik lävend
- käivita fail
- vaata tööaegu
- vaata väljundeid
- võrdle tulemusi video tegeliku sisuga


