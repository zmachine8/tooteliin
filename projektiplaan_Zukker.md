# Projektiplaan – tooteliini pildituvastus (CRISP-DM)

## 1. Äritegevuse mõistmine

### 1.1 Probleem ja kasutaja
Projekti eesmärk on teha tootmisliini kaadripõhine kontroll automaatseks. Kasutaja on tootmisliini operaator või protsessi jälgija, kellel on vaja saada kiiresti teada, kas tootel on õige kuupäev, kas pakendis on toode olemas, kas täiskaadrist leitavad triipkoodid seostuvad õigete toodetega ja kas sildid/toode vastavad oodatule.

Praegu ei ole mõistlik kogu kontrolli teha ainult inimese silmaga, sest takt on kiire ja ühes kaadris on korraga 4 pakki. Inimene võib märgata suuri vigu, kuid väiksemad vead või ajutised probleemid jäävad kergesti märkamata. Süsteemi mõte on anda iga takti kohta ühtlane automaatne hinnang ning teha sellest ka statistika.

### 1.2 Mida lahendus peab tegema
Lahendus peab töötama tootmisliini videovoo peal. Kui liinil toimub liikumine ja sellele järgneb stabiilne hetk, tehakse üks täiskaader. Sellest kaadrist:
- lõigatakse välja 4 pakki,
- iga pakk jagatakse aladeks `date`, `label1`, `label2`, `product_area`,
- kuupäeva ala pealt loetakse säilivuskuupäeva,
- triipkoodid otsitakse täiskaadri pealt ning seotakse pärast barcode asukoha järgi vastava slotiga,
- `label1`, `label2` ja `product_area` pealt tehakse DINOv2 põhine sarnasuse kontroll.

### 1.3 Edukuse mõõdikud
Olulised mõõdikud on:
- mitu takti süsteem üldse ära töötleb,
- mitu pakki ühe takti kohta korrektselt lõigatakse,
- kuupäeva saagis ja kuupäeva täpsus,
- DINOv2 põhise `label1`, `label2` ja `product_area` tuvastuse täpsus,
- triipkoodi lugemisaeg ja OCR aeg ühe takti kohta,
- kui korrektselt leitakse täiskaadri barcode põhjal slot `S1..S4`.

Lisaks on oluline, et lahendus annaks tulemused piisavalt kiiresti, et seda saaks kasutada praktilise kontrollisüsteemina, mitte ainult offline analüüsina.

### 1.4 Piirangud
Peamised piirangud on:
- kaadris võib olla liikumisudu,
- mõni pakk võib olla tühi või osaliselt nähtav,
- triipkood ei pruugi alati olla hästi loetav,
- kõik vead ei ole ühesugused ja valede näidete andmestik on piiratud,
- arendus toimub ühe striimi ja piiratud tootehulga peal.

---

## 2. Andmete mõistmine

### 2.1 Andmeallikad
Andmed tulevad RTSP videovoo pealt. Projekti praeguses seisus kasutatakse ühte striimi, kus ühes taktis on 4 pakki. Lisaks on olemas käsitsi kogutud näidispildid DINOv2 reference-galerii jaoks:
- `salami`
- `veis`
- `kalkun`
- `rulaad`
- `empty_label1`
- `empty_label2`
- `empty_product_area`

### 2.2 Andmete struktuur
Iga töödeldud takt annab:
- 1 täiskaadri,
- kuni 4 täiskaadrist leitud barcode tulemust koos asukohapunktidega,
- 4 pakipilti ehk slotid `S1..S4`,
- igast pakist 4 detailiala: `date`, `label1`, `label2`, `product_area`.

See tähendab, et iga takti kohta on üks globaalne barcode-otsingu tulemus täiskaadri peal ja lisaks 16 detailpilti, mille peal tehakse edasi OCR-i või sarnasuse mõõtmist.

### 2.3 Andmete kvaliteet
Andmete kvaliteet ei ole täielikult ühtlane. Probleeme tekitavad:
- liikumisest tekkinud udusus,
- juhuslikud valguse erinevused,
- kaadrisse jäävad käed või muud segavad objektid,
- mõnel taktil tühi pakend,
- triipkoodi halb asend, osaline nähtavus või olukord, kus barcode leitakse täiskaadrist, kuid selle seostamine õige slotiga ebaõnnestub.

Samas on andmete tugevus see, et kaamera vaade ja tooteliini paigutus on üsna püsivad, mis võimaldab kasutada ette määratud piirkondi.

### 2.4 Mida on vaja kirjeldada
Iga toote puhul on vaja kirjeldada:
- pakendite ROI-d täiskaadri peal,
- kuupäeva ala,
- `label1` ala,
- `label2` ala,
- `product_area` ala,
- triipkoodi ja toote seos,
- oodatav kuupäeva loogika.

---

## 3. Andmete ettevalmistamine

### 3.1 Kaadri valimine
Süsteem ei tööta kõigi videokaadrite peal, vaid otsib liikumise algust ning sellele järgnevat stabiilset hetke. Selle põhjal valitakse üks kaader, mida nimetatakse taktiks. See vähendab dubleerimist ja võimaldab analüüsida kaadreid, kus pakkide asend on stabiilsem.

### 3.2 Tükeldamine
Iga täiskaader lõigatakse neljaks pakiks ehk slotiks `S1..S4`. Slotid määratakse ette defineeritud `package_1..package_4` ROI-de põhjal. Seejärel lõigatakse iga pakk omakorda järgmisteks osadeks:
- `date`
- `label1`
- `label2`
- `product_area`

Triipkoodi jaoks tehakse peamist täiskaadri pealt. Pärast barcode leidmist arvutatakse selle asukoha keskpunkt ja seotakse see vastava sloti ROI-ga. See etapp on kriitiline, sest kogu järgnev OCR, barcode-slot sidumine ja DINOv2 analüüs sõltuvad sellest, et ROI-d oleksid õiged.

### 3.3 OCR ettevalmistus
Kuupäeva ala jaoks tehakse mitu eeltöötluse varianti, näiteks:
- originaal,
- CLAHE,
- teravdamine,
- Otsu lävendus,
- adaptive threshold.

OCR-i tulemus valitakse nende variantide hulgast parima kandidaadina.

### 3.4 DINOv2 ettevalmistus
`label1`, `label2` ja `product_area` alad muudetakse DINOv2 mudeli abil vektoriesitusteks. Neid võrreldakse reference-galerii piltidega. Tulemuseks saadakse:
- lähima näite klass,
- kaugus lähima positiivse näiteni,
- kaugus empty-klassi näiteni,
- otsus, kas ala on pigem `empty_like` või mitte.

`empty_like` tähendab, et ala meenutab reference-galerii järgi pigem tühja või mittesisulist näidet kui päris toodet/silti. See ei ole barcode’i tulemus, vaid DINOv2 sarnasuse tulemus.

---

## 4. Tehisintellekti rakendamine

### 4.1 Süsteemi komponendid
Lahendus koosneb mitmest järjest töötavast komponendist:
1. liikumise tuvastus,
2. stabiilse takti valimine,
3. täiskaadri lõikamine 4 pakiks,
4. detailalade lõikamine,
5. triipkoodi otsing täiskaadri pealt,
6. leitud barcode seostamine vastava slotiga `S1..S4`,
7. kuupäeva OCR `date` alalt,
8. DINOv2 põhine sarnasuse mõõtmine `label1`, `label2` ja `product_area` aladel,
9. raporti ja statistika koostamine.

### 4.2 Miks need mudelid ja meetodid
Triipkoodi jaoks kasutatakse Dynamsofti lugejat, sest eesmärk ei ole triipkoodi nullist treenida, vaid saada see praktiliselt loetuks olemasoleva tööriistaga. Praeguses lahenduses loetakse triipkoode täiskaadri pealt, sest see on kiirem kui iga toote peal eraldi mitme kandidaadiga otsimine.

Kuupäeva jaoks kasutatakse OCR-i, sest kuupäeva ala on juba ette teada ning ülesanne on konkreetne tekstituvastus.

Sildi ja toote olemasolu/õigsuse jaoks kasutatakse DINOv2-small mudelit. Selle eelis on, et see annab pildist üldise vektoriesituse ja võimaldab võrrelda näidispilte uute kaadritega ilma suure treenitud klassifikaatorita.

### 4.3 Kuidas süsteemi hinnatakse
Süsteemi headust hinnatakse moodulite kaupa:
- kas takt valiti õigel hetkel,
- kas 4 pakki lõigati õigesti,
- kas kuupäev saadi loetud,
- kas loetud kuupäev oli õige,
- kas `label1`, `label2` ja `product_area` klassifitseerimine andis õige tulemuse,
- kas `empty_like` tuvastus eristas tühje ja mittetühje piirkondi,
- kas täiskaadrist leitud barcode seoti õige slotiga.

### 4.4 Arenduse suund
Praeguse lahenduse järgmised loogilised parandused on:
- täiskaadrist loetud barcode-slot sidumise parandamine,
- vajadusel ROI-de või barcode-slot sidumise reeglite täpsustamine,
- nõrgema `label1` tuvastuse parandamine või väiksema kaaluga kasutamine,
- `label2` ja `product_area` põhjal tugevama lõppotsuse tegemine,
- reference-galerii laiendamine rohkemate heade näidetega,
- vajadusel ROI-de täpsustamine.

### 4.5 Riskid
Peamised riskid on:
- vale lõige põhjustab kõikide järgmiste moodulite vea,
- OCR võib anda vale kuupäeva isegi siis, kui kuupäev on osaliselt nähtav,
- barcode võib täiskaadrist küll leitud saada, kuid jääda sidumata, kui ROI-d ja barcode asukohapunktid ei klapi,
- `label1` ala ei pruugi olla piisavalt stabiilne tugevaks klassifitseerimiseks,
- empty-klassi näited võivad olla liiga erinevad või liiga sarnased päris näidetele,
- lõppotsuse reeglid tuleb valida nii, et valehäireid ei tekiks liiga palju.

---

## 5. Tulemuste hindamine

### 5.1 Vastavus eesmärgile
Lahendus vastab eesmärgile siis, kui see suudab iga takti kohta anda struktureeritud info:
- kas triipkood leiti,
- millise slotiga triipkood seoti,
- kas kuupäev oli olemas,
- kas kuupäev oli õige,
- milline oli `label1`, `label2` ja `product_area` ennustus,
- kas toode oli pigem olemas või empty-like.

### 5.2 Praktiline väärtus
Rakenduse väärtus ei ole ainult üksiku vea leidmine. Sama oluline on see, et süsteem kogub ka statistikat:
- kui palju takte töödeldi,
- kui palju pakke jäi OCR-is lugemata,
- kui sageli mingi ala läheb `empty_like`,
- kui sageli barcode leitakse, kuid jääb ilma slotita,
- kui stabiilne on tuvastus eri toodete peal.

See võimaldab hiljem hinnata, kas lahendus sobib päris tootmisliini seireks.

---

## 6. Juurutamine

### 6.1 Kasutusviis
Kasutaja ei pea nägema mudeli sisemist loogikat. Talle piisab sellest, et iga takti kohta on olemas:
- täiskaader,
- lõiked,
- automaatsed tulemused,
- kokkuvõttefailid ja statistika.

### 6.2 Hooldus ja edasiarendus
Süsteemi hooldus tähendab peamiselt:
- uute toodete lisamist reference-galeriisse,
- ROI-de kohandamist, kui kaamera paigutus muutub,
- OCR, barcode-slot sidumise ja barcode töökindluse testimist uute tingimuste peal,
- DINOv2 reference-piltide uuendamist, kui sildid või pakendid muutuvad.

### 6.3 Kokkuvõte
See projekt ei ole üldine videovalve lahendus, vaid konkreetse tootmisliini konkreetsete toodete kontrollsüsteem. Selle tugevus on stabiilne kaamera vaade, ette teada piirkonnad ja üsna selgelt piiritletud ülesanded. Seetõttu on realistlik teha süsteem, mis annab iga takti kohta kasuliku automaatse hinnangu ja mille täpsust saab samm-sammult parandada.
