# Problémás esetek és megoldásaik

Gyűjtőlista azokról az adat-, címke- és pipeline-problémákról, amelyekbe a projekt során
belefutottunk, és arról, mi lett velük. Részletek a hivatkozott fájlokban.

Állapotjelzők: **megoldva** (kódban / adatban rendezve), **elfogadva** (ismert, szándékosan
nem kezeljük), **nyitott** (nincs döntés vagy terv van rá), **lezárva** (kipróbáltuk, nem hozott).

Utolsó frissítés: 2026-10-15.

---

## A. Adat- és DICOM-szintű esetek

### A1. Térd és boka egy study-ban — megoldva (2026-10-13)

- **Study:** `1.2.826.0.1.3680043.8.498.62924835279557724526138406447993672103`
- **Tünet:** a 11 sorozatból 5 boka. A radiológus (404 ref) jelezte: „Ankle series shown”.
  A kiválasztás mindhárom folyadékérzékeny slotba (sagittal, coronal, axial) bokát tett, mert
  azoknak jobb a felbontása és a pontszáma.
- **Felismerés:** a `BodyPartExamined` mindegyiken `KNEE`. Az egyetlen megbízható jel a
  tekercs: `SENSE_ANKLE_8_AC` (boka) vs. `SENSE_KNEE_16_AC` (térd).
- **Megoldás:**
  - új manifest-oszlop: `coil_name` (`ReceiveCoilName` / `TransmitCoilName`);
  - új beállítás: `selection.non_knee_patterns`. Ha a tekercsnév vagy a `SeriesDescription`
    `ankle`, `foot`, `tobillo`, `cheville`, `tornozelo`, `sprungg` vagy `achill` mintát
    tartalmaz, és nincs benne `knee`, a sorozat nem választható. A knee-foot kombinált
    tekercs térdnek számít.
  - Hatás a train-en: csak ennek a study-nak a 3 slotja változott (SAG DP SPIR, COR DP SPAIR,
    STIR_aTSE_TR lett). A 224-es (`prep_v1_7848…`) és a 320-as (`prep_v1_9b43…`) cache
    újraépítve.
- **Teendő:** a régi cache-ek (`72b9…`, `3737…`, `f659…`) ennél a study-nál még bokát
  tartalmaznak; használat előtt `build-cache`. A Kaggle-re feltöltött `knee_mri` kódot
  frissíteni kell.
- **Korlát:** a manifest kb. 1200 kötetjelöltjénél (a 24 372-ből) üres a tekercsnév; ott csak
  a leírás szűr. Képalapú
  térd/nem-térd osztályozó nem készült (a gyakoriság nem indokolja).
- Kód: [manifest.py](../src/knee_mri/manifest.py) (`DEFAULT_NON_KNEE_PATTERNS`, `add_coil_name`),
  selftest: `check_non_knee_selection`. Leírás: [pipeline_lepesek.md](pipeline_lepesek.md) 7. lépés.

### A2. A `BodyPartExamined` megbízhatatlan — elfogadva

- 16 study `ANKLE` címkéjű, de mind térd (térdtekercs, térdprotokoll; 3 vizuálisan
  ellenőrizve). A valódi boka (A1) viszont `KNEE`. Előfordul még `EXTREMITY`, `WRIST`,
  `HEAD` stb. is.
- **Döntés:** a kiválasztás nem használja a `BodyPartExamined`-et.

### A3. 90°-ban elforgatott láb: sagittalis ↔ coronalis csere — megoldva (2026-10-13)

- **Study:** `1.2.826.0.1.3680043.8.498.41698150452794914665945455397334605778`
- **Tünet:** a radiológus megjegyzése: „Severe deformity with misaligned imaging orientation”.
- **Ok:** súlyos flexiós kontraktúra, posterior luxáció. A lábat oldalára fektetve vagy
  kifordítva vizsgálták, a DICOM-ban viszont `FFS` (hanyatt) fekvés szerepel. A szeleteket az
  anatómiához tervezték, ezért a `SeriesDescription` jó, de a geometria (és a Kaggle
  `Anatomical_Plane`) felcseréli a síkokat:
  - a `LT_*_sag` sorozatok normálvektora (0, 1, 0) → „coronal”, a képen valódi sagittalis;
  - a `LT_*_cor` sorozatok normálvektora (−1, 0, 0.2) → „sagittal”, a képen coronalis;
  - az axiálisokon a patella oldalt van.
  - Emiatt a mi slotjaink is felcserélődnek (sagittal slot = `LT_pd_tse_fs_cor`).
- **Egyedi eset:** a train-ben csak itt mond ellent a leírás síkja a geometriának
  rendszerszinten (7/10 sorozat; máshol legfeljebb egy-egy sorozat).
- **Címkezaj is:** a 404 ref szerint OA 0/0/0, Effusion 0, Contusion 1, Fracture 1; a lelet
  háromkompartmentes OA-t és synovitist ír, törést nem.
- **Döntés (2026-10-13):** pipeline-szabályt nem írunk rá, a study kikerül. Felvéve a
  [notebooks/label_mask_studies.csv](../notebooks/label_mask_studies.csv)-be: mind a 12 target
  ismeretlen, súlya 0, így kimarad a tanításból és minden belőle képzett referenciából. A 404
  ref fájlját (F:) nem módosítottuk; ha később értékelésre használjuk, ezt a study-t ki kell
  hagyni belőle.
- **2026-10-15:** a study benne van az új 450-es referencia `new_242` részében (Andrew/YC
  olvasás: ACL, MCL, MM, LM, Contusion, Fracture = 1). A run-validációban
  (`EvaluationReference.for_run`) a maszk miatt mind a 12 cellája érvénytelen, így a 450-ből
  449 értékelhető, és 6 targetnél eggyel kevesebb a pozitív, mint a csomag `label_counts.csv`-jében.
  Szándékos, a maszk marad.

### A4. Csonka DICOM fájlok — elfogadva

- `1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035` / axial és
  `1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680` / coronal.
- A `PixelData` pontosan fele a vártnak (félbevágott fájl). A pydicom hibaüzenete („transfer
  syntax may be incorrect”) félrevezető.
- A manifest dekódolási próbája csak a középső szeletet nézi, a hibás szelet mindkettőnél
  szélső, ezért `usable=True` lett.
- **Döntés:** a slot kiesik, nem pótoljuk nullákkal. Minden `build-cache` 2 `failed`-del és
  exit code 1-gyel zárul; ez várt, nem regresszió.
- Részletek: [pipeline_lepesek.md](pipeline_lepesek.md#ismert-adathiba-csonka-dicom-fájlok).

### A5. Mindkét térd egy study-ban — megoldva

- **Tünet:** a slotok a két térdet keverhették (pl. sagittal a jobb, axial a bal térdből).
- **Megoldás:** `selection.knee_consistency` + `center_x_mm` manifest-oszlop. Ha a jelöltek a
  középvonal két oldalán, legalább 150 mm-re vannak, minden slot a legjobb sagittalis jelölt
  térdéből jön; ha ott egy sík hiányzik, a slot üres marad.
- **Címke:** 7 study-nál a lelet nem köthető a kiválasztott térdhez (YC review, C csoport) →
  `data.label_mask_csv` = [notebooks/label_mask_studies.csv](../notebooks/label_mask_studies.csv),
  mind a 12 target maszkolva, a képek betöltődnek. (Ugyanitt van az A3 study is, összesen 8.)

### A6. Hibás lateralitás-tag — megoldva (commit 2396ed0)

- **Tünet:** 2134 study-ból, ahol tag és geometria is van, 31-ben ellentmondanak. Egyes
  helyek a protokollból bélyegzik a taget (pl. `COR T1 CS` = L, minden más R, bármelyik
  térdre), így egy study-n belül is eltér.
- **Felismerés:** YC radiológiai review 36 study-n (A: tag sorozatonként eltér; B: tag
  egységes, de a képpozícióval ellentétes; C: két térd). A 29 egytérdes esetben YC mind a
  29-ben a geometriát igazolta; a régi tag-first szabály mind a 29-et elrontotta.
- **Megoldás:** először geometria (kiválasztott coronal/axial képközéppontok medián x-e),
  csak utána tag, többségi szavazással, kizárólag a kiválasztott sorozatokon.
- Regressziós teszt: [test_laterality_yc_review.py](../src/tests/test_laterality_yc_review.py).
- **Kiegészítés (2026-10-15):** a tag-szavazásban csak a síkslotok (sagittal / coronal /
  axial) sorozatai vesznek részt, az extra slotok (`coronal_t1`, `sagittal_second`, ...)
  nem. Korábban egy 5 slotos kiválasztás mellett az extra sorozatok is szavaztak volna. A
  Kaggle-notebook a slotok uniójából építi a laterality táblát, így az oldal függhetett a
  modellcsoporttól. A geometria eddig is csak a `coronal` és `axial` slotot nézte. Mérve:
  az 5 slotos táblából a régi szabály 2 study tag-oldalát döntötte volna el (döntetlenből),
  de mindkettőnek van geometriai oldala, a végső oldal 0 study-ban változott volna. Selftest:
  `check_laterality_derivation` (extra slot nem szavaz, nem lokalizál).
- **A `work/manifest/laterality.csv` (2026-10-12) újraépítve összevetve (2026-10-15):** a
  `side` 4407/4407 study-ban azonos (3 és 5 slotos kiválasztásból is). Eltérés csak
  tájékoztató oszlopokban: `tag_side` 3 geometriai oldalú study-ban (a tábla a mostani
  szavazási szabály előtti állapotból maradt), az A1-es boka-study `center_x_mm` /
  `sagittal_normal_x` értéke a régi, boka-sorozatokból számolt (−76.73 → −113.87, mindkettő R).
  A táblát a 5 slotos C0 két seedjének lefutása után érdemes cserélni (a tanítás
  szempontjából azonos).
- **Ellenőrzés (2026-10-19):** a régi szabályhoz képest 31 train_v8 és 1 ref208 study oldala fordult meg. Mind
  benne van a YC reviewban: 28 egytérdes és 3 kéttérdes; a 3 kéttérdes a label maskban van. A 28-ból 11-nél
  nevez meg oldalt a lelet: 9 a geometriával egyezik, 2 a taggel (…6743784387, …7729459197: tag = lelet = L,
  geometria = R).
  - A canonical frame-ben a geometriai oldal a helyes mindkettőnél: a vastus medialis a MED oldalon van, a
    Baker-ciszta posteromediálisan.
  - Valószínű ok: tükrözött DICOM-koordinátarendszer. A tükrözésnek a kép tengelyeihez kell igazodnia, ezért a
    geometry-first szabály akkor is helyes, ha a lelet a másik oldalt írja.
  - A címkék oldalfüggetlenek (medial/lateral anatómiai fogalom), rossz oldalnál csak a kép tükröződik.
  - A régi és az új laterality hatása a C0-tanításra zajszintű (+0.001 a régi javára). **A javítás marad.**
  - Részletek: [tovabblepesi_javaslat_2026-10-06.md](tovabblepesi_javaslat_2026-10-06.md), 13. pont vége.

### A7. A sagittalis crop levágja a patellát — nyitott (újrapróbálandó)

- **Tünet:** a `foreground` (területi súlypont) crop-közép sagittalisan a hátsó izomtömeg felé
  húzódik; a sagittalis képek 33,8%-ánál valamelyik szél vágott (anterior 33,4%).
- **Kipróbált megoldás:** `data.crop_center=foreground_extent` → 13,6% (anterior 10,6%).
  Az I1 tanítás (224 px, régi címkék) semleges volt (0.9168 vs 0.9169), ezért nem vettük át.
- **Állapot:** a felhasználó a mostani setupban (320 px, train_v8) újra ki akarja próbálni;
  ehhez új 320-as cache kell az összes slottal.

### A8. Intenzitás-inhomogenitás — nyitott (parkolva)

- Vegyes scanner-oldali korrekció (Siemens NORM vs. nem NORM; Philips/GE ismeretlen).
- Terv: N4 korrekció és/vagy bias-field augmentáció, alapból kikapcsolva:
  [terv_inhomogenitas_korrekcio.md](terv_inhomogenitas_korrekcio.md).

### A9. Szeletközép-mintavétel — lezárva

- 30 szeletnél a 24-centrumos „középpont” mindig a bin első szelete (a 4., 9., 14. … szelet
  sosem centrum).
- Inferálás oldali c48 / TTA: a ref208-on +0.0008, LB 0.942 = c24 → nincs nyereség.
  Lásd [slice_coverage_check.md](slice_coverage_check.md).

---

## B. Címkeproblémák

### B1. A frozen v1 validációs referencia gyenge — megoldva

- A v1 címkék a radiológushoz mérve macro 0.679 (OA targetek ~0.5). A frozen OOF AUC többször
  félrevezetett (pl. train_v3 rosszabbnak látszott, az LB-n jobb volt).
- **Megoldás:** radiológusi referenciák: 158 → 208 (158 + Andrew 50), riport: new-100, 158,
  58, Andrew50. A 0.008 alatti lokális különbséget csak az LB dönti el.

### B2. „Nem említett” cellák 0.25-ös értéke — megoldva (train_v8)

- A v4 blend két LLM-forrás átlaga (az egyik 0.5-öt, a másik 0-t ad a nem említett
  leletre) → gyakori 0.25 (train_v7 cellák 19%-a; Baker's 48%, Fracture 43%). A ref208-on ezek
  közül csak 7,1% pozitív.
- **Megoldás:** train_v8 = ezek 0.07-re állítva (kivéve a Synovitist). Eredmény: H3_v8 LB
  **0.942** (v6: 0.936). Szkript: `notebooks/build_train_v8.py`.

### B3. Lateral OA túlpozitív címkék — nyitott (terv kész)

- A train 29%-án magas a címke, a radiológusi prevalencia ~16%. A magas címkéknek csak ~45%-a
  pozitív. Minták: általános „tricompartmental OA”, felszíni chondropathia OA-nak véve,
  a patella laterális facettája laterális kompartmentnek véve.
- Kulcsszavas szabállyal nem javítható (sok nyelv).
- **Terv:** új LLM-prompt csak a Lateral OA-ra (és opcionálisan a PF OA-ra), először a 208
  leleten. Lásd [tovabblepesi_javaslat_2026-10-06.md](tovabblepesi_javaslat_2026-10-06.md).

### B4. Enyhe effúzió kalibrációja — elfogadva

- A hamis pozitívok szinte mind „mild / trace / mínimo” effúziók. A Kaggle-58 viszont
  engedékenyebb, mint a mi radiológusunk, ezért a szigorítás az LB-n kockázatos. A monoton
  átskálázás a címke-AUC-t nem változtatja → kihagyva.

### B5. Referencia-kontamináció — megoldva

- Az Andrew 50 study benne volt a train_v4/v6-ban → train_v7/v8 nélkülük (4199 study),
  ref208 + holdout208 split.
- A self-distillation tanárát nem láthatta a ref208: `paths.exclude_from_training_csv` +
  `check-splits` a `run_cv`-ben; 3-fold CV (`cv3_train_v8`).
- **2026-10-15, ref450:** az új 242 referencia-study mind benne volt a train_v8-ban → train_v11
  (train_v8 − 242 = 3957), a kizárási lista a 450 (minden configban). A train_v8-on tanított
  régi checkpointok látták a 242-t, ezért a 450-en (és a new_242-n) nem vethetők össze az új
  runokkal; csak az original_208 hasonlítható a régi számokhoz.

### B6. A második olvasó nem független — elfogadva

- A YC-100 olvasás szinte azonos Andrew-éval (kappa ≥ 0.93, kivéve Contusion 0.69 és
  Fracture 0.85) → nem használjuk független második olvasóként.

### B7. Self-distillation diverzitásvesztés — lezárva

- train_v9 (0.7·v8 + 0.3·V2S tanár): az egyes modellek javultak, de az ensemble nem
  (H3_v9 0.9392 vs H3_v8 0.9393; LB 0.941), mert minden diák ugyanahhoz a tanárhoz húzott.
  Lásd [v8_vs_v9_trainings.md](v8_vs_v9_trainings.md).

### B8. A train_v10 184 radiológus-sora pozitívban kiürült — nyitott (megfigyelés, 2026-10-14)

- A ref220 válogatása (Σ 1/pos + 1/neg minimum) a ritka pozitívok szinte mindegyikét a 220-ba
  tette. A maradék 184, ami a tanításba ment, emiatt majdnem csupa negatív: ACL 1%, Lateral OA
  1%, MCL/Contusion/Fracture 0%, Effusion 10% pozitív.
- A 184-ből 90 a train_v8-ban LLM-címkével már benne volt. Ezeken a változás szinte csak
  pozitív → negatív (v8 ≥ 0.5, radiológus 0): Effusion 39, ACL 18, Lateral OA 17, Medial OA 16,
  PF OA 16, MCL 14 study. A radiológus szerint ezek LLM-hamis-pozitívok, a képük viszont
  sokszor pozitívnak „látszik” (nehéz negatívok).
- A v10-modellek a tiszta 114-en: az ACL (−0.014 / −0.028 / −0.030) és a Lateral OA
  (−0.018 / −0.011 / −0.030) mindhárom architektúrában (CNT12, R50, C0) romlott, az Effusion
  (+0.027 / +0.004 / +0.028) javult. A macro nem változott.
- A negyedik modellben (V2S v10) viszont az ACL +0.013, a Lateral OA +0.005 lett, így a
  mintázat 4-ből csak 3 modellben áll fenn (négy modell átlaga: ACL −0.015, Lateral OA −0.014).
- Nem eldöntött, hogy ez a nehéz negatívok hatása, a laterality-javításé, vagy közös zaj a
  114-en (ACL 40, Lateral OA 32 pozitív). A macro-hatás mind a négy architektúrán kb. nulla. Szétválasztás: train_v10 a 90 study v8-as
  LLM-címkéjével (csak a címkecsere hatása).
- Tanulság: kiegyensúlyozott holdout-válogatás után a maradék nem véletlen minta, a
  tanításba csak ennek tudatában szabad tenni.

### B9. OA: egy telephely-csoport riportjai rosszul egyeznek a radiológussal — nyitott (2026-10-21)

- Forrás: `notebooks/oa_error_analysis.py`, a `cv545_v12` OOF-ja az 545-ön.
- **Telephely-csoport:** a sagittalis leírás `Sag_PDW_TSE_SPAIR` / `SAG PDW_TSE_SPAIR` / `pd_tse_fs_sag` (Philips-szerű, `RT.` / `LT.` előtag). Ide tartozik a 4407 study 29%-a (1288) és a new_95 mind a 95 studyja. Az axiális sorozat itt szinte mindig T2W SPAIR, PD-axiális csak 9 studyban van, tehát nem a sorozatválasztás hibája.
- **A riport alapú LLM-címke (train_v8) itt sokkal rosszabbul egyezik a radiológussal.** A 242 + 95 studyn az LLM-vs-radiológus AUC:

  | | Medial OA | Lateral OA | PF OA |
  |---|---|---|---|
  | ez a csoport | 0.775 | 0.722 | 0.763 |
  | a többi | 0.878 | 0.856 | 0.872 |

- **A modell OA-ja is itt a leggyengébb:**
  - PF OA AUC a 208-on 0.704 (a többi 0.929), a 242-n 0.828 (0.895).
  - Lateral OA a 208-on 0.807 (0.899).
  - A new_95 (mind ebből a csoportból) Medial / Lateral / PF OA: 0.858 / 0.809 / 0.788.
- **A modell a riport-konvenciót követi.** Ahol az LLM-címke ellentmond a radiológusnak, ott a modell is hibázik (átlagos félrerangsorolás 0.22–0.35, egyező címkénél 0.05–0.12).
  - Pozitívok, ahol az LLM negatív: a new_95-ön a Medial OA 25 pozitívjából 11 ilyen, ezeken a félrerangsorolás 0.35, a PF OA-n 0.52.
  - A 242-n a tipikus eltérés LLM+ / radiológus− (Medial 45, Lateral 47, PF 43).
- **Egyéb:**
  - A hibák a határfokozatoknál sűrűsödnek: a radiológus 1-es negatívjainál és 3-as pozitívjainál.
  - A Medial / Lateral felcserélés nem gyakori (132 egyoldali esetből 0 erősen fordított), laterality-hiba nincs.
- **Lehetséges lépések:**
  1. Ennél a csoportnál az LLM OA-címkék súlyának csökkentése vagy kalibrálása a radiológushoz.
  2. OA-prompt a radiológus kritériumaival (B3).
  3. Több radiológus-címke ebből a csoportból.
- Az esetlista olvasásra: `work/analysis/oa_error_cases.csv` (targetenként 15 FN és 15 FP, riportszöveggel).

---

## C. Pipeline- és infrastruktúra-csapdák

| Probléma | Ok | Megoldás / szabály |
| --- | --- | --- |
| A build rossz cache-gyökérbe ír (2026-10-06-án és 2026-10-13-án is) | A cache-hash függ az `image_size`-tól; a config alapértéke 224, a futások 320-at használnak | Mindig `--set data.image_size=320` (gyökér: `prep_v1_9b43435e7ac6`); a log `Cache root` sorát ellenőrizni |
| Normalizálási keveredés ConvNeXt-futásoknál | A parancsból kimaradt a `data.encoder_normalization` | ConvNeXt/CoAtNet parancsokba mindig explicit `--set data.encoder_normalization=...` |
| A DINOv3 ConvNeXt `lr` 1e-4-gyel tönkrement | A súlyok RMS-e 4–5× kisebb | Az encoder lr-t a súly-RMS-hez skálázni (in12k → 4e-5, DINOv3 → 2e-5) |
| Windows commit-limit OOM (error 1455) a val-workerek indításakor; 2026-10-14-én újra (C0 c48, 8. epoch, `MemoryError` a train-workerben, `build_bag`) | ~68 GB commitot tartottak a tétlen appok (10-14: 46 régi `claude` folyamat ~21 GB, VS Code ~12 GB, Chrome ~8 GB); a c48 kétszeres bagje; mellette indított selftest/smoke Python-folyamatok (~10 + 4 GB) | Tanítás előtt régi Claude-sessionök és appok bezárása, tanítás közben nem futtatni tesztet; a pagefile növelése; resume `last.pt`-ből |
| A `run_holdout.bat` a végén megáll PowerShellből indítva | A bat dupla kattintásnak hiszi | `$env:NOPAUSE = "1"`; a `$env:TRAIN_CSV` a sessionben megmarad |
| A Linux gépen hiányzó CSV-k | A `DATA_ROOT` külön van | Új train/ref CSV-ket (train_v8, ref208) át kell másolni |
| DDP resume | Más GPU-számmal nem folytatható | Ugyanazzal a `--gpu` beállítással folytatni |
| Kaggle-beküldés „Out of Memory” (`holdout5_v11`, 2026-10-19/20, kétszer; a második ~2 óra futás után) | Host-RAM-szivárgás a 2 GPU-s inferenciaúton (63c0d46 óta): a `predict_on_devices` chunkonként és bemeneti csoportonként új GPU-szálakat indított, és minden CUDA-t használt szál ~6 MB-ot hagyott hátra (lokálisan mérve: +38 MB/chunk 3 csoport × 2 szállal; 1 eszközzel vízszintes; DataLoader-workertől és `pin_memory`-tól független). A rejtett teszt méretével nő, ezért a 3 publikus studyn nem látszik | GPU-nként egy tartós munkaszál (`GpuWorkers` a `build_04_notebook.py`-ban); mérve 15 chunkon vízszintes, a predikciók bitre egyeznek. Új inferenciakódot beküldés előtt a scratchpad `leak_probe.py` mintájára sok chunkon át mérni (a rejtett futásról nincs napló). A javított notebook a Kaggle-ön lefutott (2026-10-20, LB 0.941) |
| Külön kiválasztási tábla felülírta az alap summary-t (2026-10-15) | A `select-series` a summary JSON-t fix `series_selection_summary.json` néven írta a CSV mellé | A summary neve a CSV-ből képződik (`<stem>_summary.json`); az alap táblánál a név nem változott |
| Cool-down a csúcs után rontott | Túlillesztés, nem lr-zaj | Nincs cool-down fázis; `best.pt`-nél megállni, multi-seed átlagolás |
| Betelt a H: meghajtó tanítás közben (2026-10-19, `V2S_FToa_s2`, `No space left on device` a checkpoint-mentésnél) | A `work/runs` ~92 GB; minden epoch `best.pt` + `last.pt` (~400 MB) + epoch-checkpoint (~100 MB), a mentés átmenetileg még egy példányt kér | Resume `last.pt`-ből pontos volt. Hosszú sorozat előtt legyen ≥5 GB szabad hely; kész runok `last.pt`-je archiválható (I:/Kaggle/oa_ft_lastpt_backup/) |

---

## D. Mérési csapdák (röviden)

- Egy modell run-to-run zaja kb. ±0.007; a radiológusi referenciák ~0.008 alatt nem döntenek.
- A kiegyensúlyozott ref220 sem segít az egyseedes döntésen: CNT12 v10 s42 − s2 = +0.0089
  [+0.0014, +0.0167]. Ez a modell szórása, nem a mérésé → legalább két seed átlaga.
- A 158/208 referencián ismételt modellválasztás felfelé torzítja a nyertest → rangsorolásra
  igen, súlyok/küszöbök hangolására nem.
- Per-target modellsúlyozás a referencián (CV-vel is) negatív → lezárva.
- Finomhangolás olyan checkpointból, amely látta a validációs studykat (2026-10-19): a train_v8-as
  checkpointok a new_242 mind a 242 studyján tanultak. `train.init_weights` ezért most failure
  gate: a forrás tanító-studyjait (`train_study_ids` / `train_studies.csv`) a validációs és kizárt
  studykkal veti össze; régi runokhoz `src/tools/write_train_studies.py`.
- Specialistánál (`train.loss_targets`) a 12 targetes makró és a 12 targetes referencia-BCE a nem
  tanított, sodródó kimeneteket is átlagolja: a választás csak a csoporton (`eval.selection_targets`
  kötelező), a loss-görbe `val_loss_group_ref_bce`. A referencia-BCE a csoporton is nőhet, miközben a
  csoport-AUC javul (kalibrációs elmozdulás a soft címkék felé) → túlilleszkedést AUC-görbéből ítélni.
- A full_450 macro AUC-t mindig közvetlenül a 450 predikcióból számoljuk, nem az original_208
  és a new_242 átlagából (`subset_summary.csv`). A 242 összetétele nem véletlen (YC hat
  szekvenciás, Zoli patológiadús válogatás), ezért ellentmondó változásnál a részhalmazokat is
  nézni kell.
