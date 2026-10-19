# Továbblépési javaslat (2026-10-06), a saját memórialogjaim alapján

## Kontextus – hol tartunk
- **2026-10-15-től új elsődleges validáció: holdout450** (eredeti 208 + 242 új) és **train_v11** (3957 study); a riport külön adja a full_450 / original_208 / new_242 eredményt (13. pont). Az alábbi számok még a holdout208-ra vonatkoznak.
- **LB legjobb: 0.945** = `holdout5_v8_cnt12` (train_v8, holdout208, E3): R50 + V2S + C0 + CNT12 s42 + CNT12 s2 (lásd 8. pont, „LB-próba”). Előtte `holdout4_v8_c0` 0.943 (H3_v8 + CoAtNet-0, 7. pont), `holdout3_v8` 0.942, H3_v6 0.936.
- **Lokális baseline: H3_v7** (train_v7 = v6 mínusz Andrew-50; holdout208): 208 0.9341, new-100 0.9384, Andrew50 0.9150. Egyedi modellek: B0 0.9224, R50 0.9194, V2S 0.9278 (208, best ep).
- **Ami bizonyítottan működött:** címkeminőség. v4→v6 (csak a címkék cseréje) +0.012 LB.
- **Ami nem működött (lezárva):** per-target modellsúlyozás, logit/rank átlagolás (v8-on és v9-en újramérve, lásd 6. pont), Effusion-oszlop v2-ből, v2/v3 keverése a blenddel, 4. slotként coronal_t1 és sagittal_nonfs, valamint az 5 slotos Raptor-elrendezés (14. pont). Extra slotot többet nem javaslok. Inferálás oldali sűrűbb szeletmintavétel (c48) és slice-offset TTA (4a. pont, LB 0.942 = c24). CoAtNet rmlp_1_rw2 (CR1) ötödik vagy csere-tagként a H4 mellett (7. pont).
- **Mérési korlát:** egy modell run-to-run zaja kb. ±0.007, a referenciák ~0.008 alatt nem döntenek. Ensemble-szinten és LB-vel kell megerősíteni.
- **A modellek már jobbak, mint a címkék** (H3 new-100 0.938 vs. v6 blend címke 0.898). A címkezaj a szűk keresztmetszet.

## Javasolt sorrend

### 1. „Nem említett” cellák alapértékének korrekciója: 0.25 → 0.07 (train_v8, pseudo-labeling nélkül)
**Miért:** a blend képlete `0.5·llm_labels_v2 (stevenleehans) + 0.5·labels_llm_gpt56sol (lixin73)`, mind a 4407 sorban bitpontosan ellenőrizve. A v2 a nem említett eltérésre 0.5-öt ad, a gpt56sol 0-t, ezért lesz belőle 0.25.
- A train_v7 cellák 19%-a 0.25 (Baker's 48%, Fracture 43%, Lateral OA 24%, Contusion 23%).
- A 9563 ilyen cellából 9556 a v2 = 0.5 és gpt56sol = 0 párosból jön. A maradék 7 véletlen egybeesés.
- A ref208-on ezeknek a celláknak csak **7.1%-a** pozitív (524 cella). A <0.1-es explicit negatívoknál ez 4–5%. A 0.25 tehát kb. 3,5-szeresen túlbecsüli a pozitivitást, és a BCE-target kalibrációját torzítja.
- **Döntés (felhasználó):** egyetlen globális érték, **0.07**. Targetenkénti értéket nem használunk, mert az az értékelő halmazra hangolás lenne.
- **Kivétel: Synovitis** (megvalósításkor derült ki). Ez az egyetlen oszlop, ahol a v2 eltér a llm_labels_full-tól. A leletek ritkán említik a synovitist (a full-ban 3689 db 0.5), és a v2 ezeket újrabecsli (~0.27 / ~0.65). A megmaradt 0.5 nála „nem eldönthető”, nem „nem említett”: a ref208-on a 0.25-ös csoport 19%-a pozitív, a 0.13-as csoportnak 4%-a. A 0.07 itt rontana a rangsoron (−0.021 címke AUC), ezért a Synovitis 0.25 marad.
- **Eredmény:** `F:/Kaggle/data/train_v8.csv`, 9165 cella változott (a cellák 18.2%-a); 398 db 0.25 maradt (Synovitis és a véletlen egybeesések).

**Megvalósítás:** új script `notebooks/build_train_v8.py`, a `notebooks/build_ref208_train_v7.py` mintájára.
- Bemenet: `F:/Kaggle/data/train_v7.csv` (dtype=str, keep_default_na=False, hogy a nem érintett cellák szövege ne változzon), valamint a két forrás: `F:/Kaggle/llm_labeling/output/labels_lixin_gpt56sol_V2/llm_labels_v2.csv` és `.../labels_lixin_gpt56sol/labels_llm_gpt56sol.csv`.
- Maszk: a forrásból azonosítjuk a cellákat, `v2 == 0.5 & gpt56sol == 0`, nem a `== 0.25` egyezésből. Így a 7 véletlen 0.25-ös cella érintetlen marad.
- Assertek: a maszkolt cellák a train_v7-ben mind 0.25; a blend-képlet egyezik a train_v7-tel (max. eltérés < 1e-9); a sorok és oszlopok változatlanok (4199); nincs átfedés a ref208-cal.
- Kimenet: `F:/Kaggle/data/train_v8.csv`. Kiírja targetenként, hány cella változott.
- A tanítókód és a config nem változik. Az `allow_soft_targets` kezeli a folytonos értékeket, a validáció a radiológus-referencián megy, a Kaggle notebookot nem érinti.
- Linuxon a train_v8.csv-t a DATA_ROOT-ba is át kell másolni.

**Futtatás** (PowerShell; a run mappája a TRAIN_CSV-ből automatikusan `work/runs/train_v8/` lesz):
```powershell
$env:NOPAUSE = "1"; $env:TRAIN_CSV = "F:\Kaggle\data\train_v8.csv"; $env:HOLDOUT = "208"
$E3 = "--set data.image_size=320 --set data.laterality_canonical=true --set model.side_pooling=true --set model.target_attention=true --set train.ema_decay=0.999 --set train.num_workers=4 --set train.eval_num_workers=2 --set train.prefetch_factor=2".Split(" ")
.\run_holdout.bat B0_E3_img320_trainv8 @E3
.\run_holdout.bat R50_E3_img320_trainv8 --config src/config.radimagenet_resnet50.yaml --set model.encoder_trainable=all @E3
.\run_holdout.bat V2S_E3_img320_trainv8 --set model.backbone=efficientnet_v2_s @E3
```
(az E3 beállítások a v7 runok mentett config.yaml-jaiból; a `$env:TRAIN_CSV` a sessionben megmarad)

**Döntési sorrend:**
1. Szanity-check tanítás nélkül: a train_v8 címke AUC-ja a 208-on ≥ a v7-é (a rangsor alig változik: csak a 0.07–0.25 közötti értékek kerülnek a nem említettek fölé). **Kész:** 208 0.8928 → 0.8933, new-100 0.8981 → 0.8996, Andrew50 0.8418 → 0.8416 (Lateral OA +0.009, Baker's −0.001, Contusion −0.001).
2. B0 v8 vs. B0 v7 (0.9224): curve mean ep6–10 (v7: 0.9200) és best, per-target páros bootstrap, elsődlegesen a new-100-on. Várható nyertesek a magas 0.25-arányú targetek: Baker's, Fracture, Lateral OA, Contusion.
3. Ha a B0 nem negatív: R50 + V2S v8 → H3_v8 vs. H3_v7 (0.9341). Ensemble-szinten döntünk, mert az egyedi modellek zaja ±0.007.
4. H3_v8 LB-próba (a build_04_notebook.py-ban új ensemble-bejegyzéssel), összevetve a 0.936-tal.

**Eredmény (2026-10-07): siker.**
- **Modellenként (208, best epoch):** B0 0.9224 → 0.9292, R50 0.9194 → 0.9240, V2S 0.9278 → 0.9332. A görbe ep6–10 átlaga mindhárom modellnél 0.005–0.007-tel nőtt.
- **H3_v8:** 208 0.9393 (+0.0053 [+0.0009, +0.0098]), new-100 0.9440 (+0.0057), Andrew50 0.9203 (+0.0048).
- **Targetenként:** Fracture +0.025 mindhárom modellben; Lateral OA és Lateral Meniscus −0.006; a Synovitis (kontroll) +0.005.
- **LB: `holdout3_v8` = 0.942** (előtte 0.936). A train_v8 az új címke-alap.

### 2. Self-distillation / pseudo-label kör (az 1. lépés eredményére építve)
A tanítócímke `α·címke + (1−α)·modell_OOF`, ahol a címke a jobbik változat (v7 vagy v8).
- Teacher OOF a teljes train-re: 3-fold CV (holdout208 kizárva), elég a B0. A split a `work/splits/cv3_train_v8` (a train_v8.csv-ből, 4199 study), ez a `run_cv.bat` alapértelmezése. A ref208 kizárását a `paths.exclude_from_training_csv` és a `check-splits` kényszeríti ki.
- Szűrés tanítás nélkül: a kevert címke AUC-ja a radiológushoz mérve a 208-on (a holdout runok predikciói ott valódi OOF-ok). α ∈ {0.3, 0.5, 0.7}.
- Csak a legjobb α-val tanítunk H3-at, az eredményt holdout208-on és LB-n nézzük.

**Megvalósítás (2026-10-07 – 10-09), a tervtől eltérően:**
- **Teacher: V2S** (nem B0), 3-fold CV a `cv3_train_v8` spliten, Linuxon: `V2S_E3_img320_teacher_cv3_20261007_232654` (fold best ep 11–12; egyedi foldok a 208-on 0.919 / 0.924 / 0.920). A 208-ra a teacher-predikció a 3 fold `best.pt` átlaga, külön `evaluate`-tel.
- **α = 0.7, előre rögzítve** (α ≥ 0.5, hogy az LLM-címke maradjon a domináns), nem a címke-AUC szűrés alapján. A szűrés csak szanity-check volt. Címke AUC a 208-on: α=1 (v8) 0.8933, 0.85: 0.9313, **0.7: 0.9358**, 0.5: 0.9389, 0 (csak teacher): 0.9304. A keverék mindkét forrást veri. A legnagyobb címke-nyereség: Baker's +0.13, Fracture +0.08, Effusion +0.07, Lateral Meniscus +0.06; MCL −0.017.
- `notebooks/build_train_v9.py` → `F:/Kaggle/data/train_v9.csv` (4199 study, `0.7·v8 + 0.3·teacher OOF`).
- Studentek holdout208-on, a v8-cal azonos E3 beállításokkal: B0 és R50 Windowson, V2S Linuxon.
- **Előre rögzített siker:** LB ≥ 0.943.

**Eredmény (2026-10-09): nincs ensemble-nyereség, lezárva.** Részletes táblázat: `doc/v8_vs_v9_trainings.md` (generálja: `notebooks/compare_v8_v9.py`).

| Modell (208, best ep) | v8 | v9 | Δ [95% CI], P(>0) |
|---|---|---|---|
| B0 | 0.9292 | 0.9324 | +0.0032 [−0.0009, +0.0075], 0.94 |
| R50 | 0.9240 | 0.9275 | +0.0035 [−0.0007, +0.0077], 0.94 |
| V2S | 0.9332 | 0.9352 | +0.0019 [−0.0034, +0.0076], 0.74 |
| **H3** | **0.9393** | **0.9392** | −0.0001 [−0.0034, +0.0031], 0.46 |

- **Egyenként mindhárom modell javult.** A B0 v9 görbéje minden epochban kb. 0.003-mal a v8 fölött van, és a csúcs után lassabban esik (ep12: 0.9270 vs. 0.9195), ami a soft címke regularizáló hatása.
- **Ensemble-szinten a javulás eltűnik**, mert a studentek egymáshoz hasonultak, hiszen mindegyik ugyanazt a V2S teachert követte. Páronkénti Spearman-korreláció, v8 → v9: B0–V2S 0.904 → 0.933, B0–R50 0.870 → 0.888, R50–V2S 0.856 → 0.878.
- **Targetenként (H3, v9 − v8):** ACL +0.006 és Lateral OA +0.007; Medial Meniscus −0.006 és Baker's −0.005; a többi ±0.004-en belül.

**LB-próba (a felhasználó hipotézise: a két student megtartja a javulását, a régi V2S pedig a saját erősségeit és a változatosságot):**
- `holdout3_v9_v2sv8` = B0 v9 + R50 v9 + V2S v8 (új datasetek: `knee-b0-holdout-v9`, `knee-r50-holdout-v9`): 208-on 0.9388 (−0.0005 [−0.0028, +0.0020] a H3_v8-hoz képest), **LB 0.941** (`holdout3_v8`: 0.942).
- A régi V2S tényleg változatosabb partner (B0 v9–V2S v8: 0.918 vs. V2S v9-cel 0.933), és a Baker's-erőssége megmaradt. A két student átlaga viszont csak +0.0012 (egyenként kb. +0.003), mert egymáshoz is közelebb kerültek, és a régi V2S felé is elmozdultak (B0–V2S v8: 0.904 → 0.918).
- A `holdout3_v9` (mindhárom v9) notebook is elkészült, de nem lett beküldve.

**Kapcsolódó ensemble-próbák (208, tanítás nélkül):**
- **H6** (3 × v8 + 3 × v9, valószínűségátlag): 0.9397, +0.0004 [−0.0016, +0.0024]. Ugyanannak az architektúrának a v8 és v9 változata szinte azonos (Spearman B0 0.978, R50 0.978, V2S 0.948), ezért kétszeres futásidőért zajszintű nyereség; nem küldtük be.
- **Targetenkénti modellsúlyok újramérve a teljes 208-on** (5-fold CV × 10 ismétlés, 0.1-es súlyrács, shrinkage az egyenlő súly felé): H3_v8 sima átlag 0.9393; illesztett súlyok −0.0038, 50% shrinkage −0.0009, 75% −0.0002. A csalásos (ugyanazon illesztett és mért) plafon is csak +0.0034. Lezárva; egyetlen kipróbálatlan változat a 4199 study OOF-ján, LLM-címkén illesztett súly (ehhez B0/R50 3-fold CV kellene).

**Döntés:** a train_v8 marad a címke-alap, a `holdout3_v8` (LB 0.942) a legjobb beküldés. Ha a self-distillation irányt folytatjuk, a teacher a **H3 OOF** legyen (B0 + R50 + V2S 3-fold CV), ne egyetlen modell, hogy a studentek ne egy modell felé hasonuljanak.

### 3. Célzott címkejavítás a gyenge targetekre (olcsó, nincs GPU-igénye)
**Eredeti javaslat:** **Effusion** (Andrew50 0.744, kalibrációs gond) és **Lateral OA** (0.782): LLM prompt- vagy küszöbváltozat, címke AUC-val szűrve a new-100-on és az Andrew50-en. Csak akkor tanítunk, ha a macro címke AUC ≥ +0.01.

**Hibaelemzés (2026-10-09, tanítás nélkül)** – script: `notebooks/weak_target_label_check.py` (az xlsx-olvasáshoz openpyxl kell, ez a `kaggle_2026` env-ben nincs, a `fov_review` env-vel fut); az eltérő esetek leletszöveggel: `notebooks/weak_target_disagreements.csv` (78 eset: radiológus 0–4, v8/v2/gpt56sol címke, H3_v8 pontszám, megjegyzés, angol és eredeti lelet).

| címke-AUC (v8) | 208 | 58 (Kaggle) | new-100 | Andrew50 | H3_v8 modell, 208 |
|---|---|---|---|---|---|
| macro | 0.893 | 0.893 | 0.900 | 0.842 | 0.939 |
| Effusion | 0.878 | 0.877 | 0.907 | 0.744 | **0.970** |
| Lateral OA | 0.868 | 0.833 | 0.932 | 0.782 | **0.859** |

- **Effusion: kimarad.**
  - A modell ezen a targeten nem gyenge: 0.970 a 208-on, 0.987 a new-100-on, 0.972 az Andrew50-en. A macróhoz legfeljebb ~+0.0025-öt hozhatna.
  - A hamis pozitívok (44 eset, radiológus 0, címke ~0.9) szinte mind „mild / minor / minimal / trace / leve / mínimo / geringer” effúziók.
  - A Kaggle-58 viszont engedékenyebb. Az LLM „enyhe” sávjában (v2 0.6–0.84) a pozitív arány a 58-on 0.43, a new-100-on 0.16, az Andrew50-en 0.26. Az Effusion-prevalencia 0.60, 0.20, illetve 0.32. Az LB-t a Kaggle annotátorai pontozzák, ezért a mi radiológusunk szigorúbb definíciója felé húzni a címkét kockázatos.
  - A javítás monoton átskálázás lenne (mild 0.9 → 0.3), ami a címke-AUC-t nem változtatja, így a szűrés nem is tudná mérni.
- **Lateral OA: ez a jelölt.**
  - Ez a modell leggyengébb targetje (0.859 a 208-on). A címke legalább ilyen jó (0.868), a new-100-on jobb is (0.932, a modell 0.865), tehát itt a címkezaj valóban korlátozhat.
  - A train 29%-án magas a címke (v2 > 0.6, 1277 study), a radiológusi prevalencia ~16%. A magas címkéjű eseteknek csak ~45%-a pozitív, a 58-on és a saját referenciákon egyaránt, tehát ez címkehiba, nem definíciós eltérés.
  - A hamis pozitívok (25 eset) három mintába esnek:
    1. általános „OA of all three compartments” / tricompartmentalis OA, a laterális rész külön említése nélkül;
    2. enyhe vagy felszíni chondropathia (grade I–II, „condropatía superficial”) OA-nak véve;
    3. a patella laterális facettája laterális kompartmentnek értve.
  - Kulcsszavas szabállyal nem javítható. A regex-csoportokban a pozitív arány 0.45 és 0.46, és a magas címkéjű leletek közül 581 bolgár, görög, horvát, török vagy más megfogalmazású.
- **PF OA** a második leggyengébb modell-target (0.881), hasonló túlpozitív mintával (v2 > 0.9 sávban az Andrew50-en csak 0.61 pozitív). Ugyanabba a promptkörbe bevehető.

**Terv (még nem indult el):**
1. **Új LLM-prompt, csak a Lateral OA-ra (opcionálisan a PF OA-ra).** A leletszövegből dolgozik, a többi oszlop változatlan marad. Szabályok:
   - csak a laterális femorotibiális kompartment számít; a patella-facetta a PF OA-hoz tartozik;
   - pozitívhoz osteophyta, mély vagy teljes vastagságú porcvesztés, ízületirés-szűkület vagy subchondralis eltérés kell; a felszíni / grade I–II chondropathia negatív;
   - a fokozat nélküli „tricompartmental OA” köztes értéket kap.
   A kimenet 0–4 skála (mint a radiológusé), amit soft címkévé képzünk le. Így a sorrend is változik, nem csak a skála.
2. **A promptba nem kerülhet példa a ref208-ból.** A szabályokat általánosan fogalmazzuk meg; ha kellenek példák, azok a 4199-es train leleteiből jönnek.
3. **Először csak a 208 leleten futtatjuk** (~208 hívás). Csak javulás esetén jön a teljes train (~4400 lelet).
4. **A beillesztés módja előre rögzített 2–3 változat közül választható:** az új érték lecseréli a Lateral OA oszlopot, vagy belekeverjük a régi blendbe (például 0.5 új + 0.5 régi).
5. **A kapu módosul.** A macro +0.01 egyetlen targetre +0.12-t jelentene, ami irreális. Helyette: a Lateral OA (és PF OA) címke-AUC-ja a 208-on páros bootstrappal, a javulás pozitív irányú a 58-on és a new-100-on is. Az Andrew50 csak 6 pozitívval tájékoztató jellegű.
6. **Siker esetén** train_v10 (= train_v8 + új Lateral OA oszlop), H3 E3 a v8-cal azonos beállításokkal, holdout208, majd LB-próba a `holdout3_v8`-hoz (0.942) képest.

**Nyitott kérdés:** melyik LLM-mel és API-val fut a saját címkéző pipeline (`F:/Kaggle/llm_labeling`, `input/rsna_knee_labeling_system_prompt_v1.txt`, ebből készült a GT_v2 és a GT_v3_soft).

### 4. TTA a Kaggle notebookban (olcsó post-processing)
A build_04_notebook.py most TTA nélkül futtat. Jelöltek: slice-offset TTA és enyhe intenzitás-TTA. Bal–jobb flip nem (medial/lateral szemantika). Mérés a meglévő checkpointokkal a 208-on, tanítás nélkül.

**4a. Szeletlefedettség és slice-offset TTA (2026-10-09 – 10-10): lezárva, nincs nyereség.**

**Módszer:** csak az inferálás változott, a checkpointok a train_v8 holdout208 B0/R50/V2S `best.pt`-i (24 középponttal tanítva). Két változattípust mértünk:
- **cN:** `data.centers_per_series` = N, bin-középponttal. 48 középpontnál, 48 szeletig minden szelet bekerül.
- **TTA:** 24 középpont, a bin elejére / közepére / végére tolt nézetekkel (`bin_centers` offset 0 / 0.5 / 1), a logitok átlagával.

A 208-on a sorozatok 76–82%-a több mint 24 szeletes (medián 30–32), tehát a változás a többségüket érinti. Páros bootstrap a c24-hez képest, 1000 ismétlés.

| Változat | H3 hard macro AUC (208) | Δ [95% CI], P(>0) | Idő |
|---|---|---|---|
| c24 (tanításkori) | **0.9393** | – | 1.00× |
| c16 | 0.9382 | −0.0010 [−0.0034, +0.0009], 0.16 | 0.85× |
| c32 | 0.9398 | +0.0005 [−0.0009, +0.0020], 0.76 | 1.16× |
| c48 | 0.9401 | +0.0008 [−0.0006, +0.0023], 0.86 | 1.48× |
| c24 TTA2 (0, 1) | 0.9397 | +0.0005 [−0.0006, +0.0017], 0.79 | 2.06× |
| c24 TTA3 (0, 0.5, 1) | 0.9400 | +0.0007 [−0.0005, +0.0019], 0.86 | 2.98× |

- **Monoton, de zajszintű trend.** c32-nél és c48-nál mindhárom modell külön is javul (+0.0002 … +0.0015). A max-pooling eloszlás-eltolódása nem rontott.
- **Targetenként (c48):** a Lateral OA +0.004, a Synovitis +0.004 és a Baker's +0.004 javul; az MCL −0.005 (15 pozitív) és a Lateral Meniscus −0.002 romlik.
- **Mellékes megfigyelés:** 30 szeletnél a „bin-közép” mindig a bin első szeletét választja, így a 4., 9., 14., … szelet sosem középpont.
- **LB: `holdout3_v8_c48` = 0.942**, ugyanannyi, mint a `holdout3_v8` (c24).
- **Döntés:** marad a c24. Inferálás oldali c48-at vagy slice-offset TTA-t nem javaslunk újra. Kipróbálatlan maradt a tanítás több (48 / összes) középponttal; ezt alacsony prioritásúnak tartjuk.
- **Kód:**
  - `bin_centers(offset=)` és `StudyBagDataset(center_offset=)` (`dataset.py`);
  - `predict_studies(center_offsets=)` (`evaluate.py`);
  - `CENTERS_PER_SERIES` beállítás és `--ensemble holdout3_v8_c48` (`build_04_notebook.py`);
  - script: `python notebooks/slice_coverage_check.py [variant ...]` → `doc/slice_coverage_check.md`.
- **Nyitott:** az enyhe intenzitás-TTA még nincs megmérve.

### 5. Inhomogenitás-korrekció (parkolt terv, doc/terv_inhomogenitas_korrekcio.md)
Először a bias-field augmentáció, B0-n, 2 seeddel. N4 csak akkor, ha az aug pozitív.

### 6. Rangátlagolás a v8 és v9 H3-on (2026-10-09, tanítás nélkül): lezárva, nincs nyereség
**Módszer:** modellenként és targetenként rangsoroltuk a study-kat a sigmoid-valószínűség alapján, a 0/1 cellákon és a soft cellákon együtt (`rank(pct=True)`, egyező értékeknél átlagos rang). A három modell (B0, R50, V2S) normalizált rangjait egyenlő súllyal átlagoltuk. A kontroll a logit-átlag. Mérés a holdout208 predikcióin; a különbséget 1000-szeres páros, study-szintű bootstrappal becsültük a valószínűségátlaghoz képest.

| Ensemble | v8 macro AUC | v9 macro AUC | Δ v8 [95% CI], P(>0) | Δ v9 [95% CI], P(>0) |
|---|---|---|---|---|
| valószínűségátlag (jelenlegi) | **0.9393** | **0.9392** | – | – |
| rangátlag | 0.9386 | 0.9386 | −0.0007 [−0.0021, +0.0006], 0.13 | −0.0006 [−0.0020, +0.0007], 0.21 |
| logit-átlag | 0.9394 | 0.9391 | +0.0001 [−0.0007, +0.0009], 0.56 | −0.0001 [−0.0006, +0.0005], 0.44 |

- A soft macro AUC mindkét verzión ugyanezt adja.
- **Targetenként (rangátlag):** v8-on a Baker's −0.005 és az Effusion +0.003. v9-en a Lateral Meniscus +0.006, az Effusion +0.003 és az ACL −0.005; az ACL-en a V2S egyedül 0.990-et ér el, a rangátlag a gyengébb R50 felé húzza.
- **Miért nem segít:** a három modell kalibrációja gyakorlatilag azonos (az átlagos valószínűség 0.31–0.33, a targetenkénti szórás 0.30–0.32), így a rangátlagnak nincs mit kiegyenlítenie. Közben elveszíti a magabiztossági információt.
- **Döntés:** marad a valószínűségátlag, LB-próba nem kell. Ez megerősíti a korábbi, logit/rank átlagolásra vonatkozó negatív eredményt (lásd Kontextus).
- Script: `python notebooks/rank_average_check.py [v8|v9]` (ugyanazokat a runokat tölti be, mint a `notebooks/compare_v8_v9.py`).

### 7. CoAtNet encoder (hibrid CNN + attention): C0 negyedik tagként LB 0.943, új legjobb; CR1 nem hoz többet
**Miért:** a `holdout3_v8` (LB 0.942) mindhárom tagja tiszta CNN (B0, R50, V2S). A v9-kör tanulsága, hogy az ensemble-nyereséget a tagok diverzitása korlátozza: a v9-es studentek egymáshoz hasonultak, és az egyenkénti javulás ensemble-szinten eltűnt. A CoAtNet az első két szakaszában MBConv, az utolsó kettőben a teljes térképen globális, relatív-pozíciós self-attention. Ez más induktív torzítás, mint a három CNN-é, ezért kétféleképpen hozhat nyereséget: erősebb egyedi modellként, vagy diverzebb ensemble-tagként.

**Változatok (timm 1.0.30, nincs új függőség):**
- `coatnet_0_rw` = `timm/coatnet_0_rw_224.sw_in1k` (27M, ImageNet-1k). Először ezzel.
- `coatnet_rmlp_1_rw2` = `timm/coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k` (41M, ImageNet-12k előtanítás, MLP-alapú relatív pozíció). Csak akkor, ha a 0-s ígéretes.

**Beépítés (2026-10-10, kész):**
- **Adapter.** `CoAtNetEncoder` az `encoders.py`-ban. A timm `forward_features` kimenete `[N, 768, H/32, W/32]`, 320 px-en 10×10, mint a CNN-eknél; a fej, a side pooling és a target attention változatlan.
- **Fix bemeneti méret, mint a DINOv2-nél.** A modell pontosan a `data.image_size`-ra épül, ami 32 többszöröse kell legyen; más méretet elutasít.
  - Betöltéskor a timm a `coatnet_0_rw` 9 relatív-pozíciós tábláját 224-ről 320-ra méretezi át (729 → 1521 sor). A többi 231 tenzor bitre azonos a fájllal, csak az osztályozó fej marad ki.
  - Az rmlp-változatban nincs tábla: mind a 447 tenzor bitre azonos.
  - Az inferencia (evaluate és a Kaggle notebook) a checkpointot mindig a betanított méretre építi újra (`FIXED_INPUT_BACKBONES`).
- **`last_n` mód:** a 4 stage tanulható, a végső LayerNorm2d velük együtt tanul, a stem csak `all` módban.
- **Normalizálás:** új `inception` profil (mean = std = 0.5, a `sw_in*` statisztikája).
- **Súlyok:** HF hubról egyszer töltődnek le, utána a cache-ből offline mennek. A Windows-gépen mindkettő megvan; Linuxon az első futás letölti, vagy a `model.safetensors` átmásolható, és `model.weights=<útvonal>`.
- **Tesztek:**
  - `tests/test_backbones.py`: 24/24, köztük a grad-checkpointing ekvivalencia, a freeze-szabályok, az offline reload és a méretszabályok;
  - `tests/test_pretrained_encoders.py`: 7/7;
  - teljes selftest: 97/97;
  - szintetikus tanítás 320 px-en, E3-fejjel: lefut, a loss csökken.
- **Kaggle:** a notebookhoz kódváltozás nem kell. A `srcknee2` kód-datasetet frissíteni kell, és a Kaggle-image timm-verziója is kell; ha eltér, a timm 1.0.30 wheel datasetként csatolható. A `load_state_dict` szigorú, egy kulcseltérés hangosan elhasal.

**GPU-memória.** Egy teljes tanítólépés: 1 study × 3 slot × 24 valódi ablak, 320 px, bf16, E3-fej, AdamW és EMA. Eszköz: `python src/tools/memory_probe.py <--set kapcsolók>`.

| encoder (RTX 5080, 16 GB) | csúcs, allocated / reserved | lépésidő |
|---|---|---|
| V2S + grad checkpointing (a v8 beállítása) | 4.4 / 8.4 GiB | 261 ms |
| CoAtNet-0 | 12.7 / 13.9 GiB | 225 ms |
| CoAtNet-0 + grad checkpointing | 4.3 / 5.3 GiB | 285 ms |
| rmlp_1_rw2 + grad checkpointing | 6.1 / 7.8 GiB | 498 ms |
| rmlp_1_rw2 | OOM | – |

**GPU-elosztás:**
- **24 GB (Linux, RTX 3090):** CoAtNet-0 checkpointing nélkül, ez a leggyorsabb. Az rmlp_1_rw2 checkpointing nélkül becslés szerint ~22 GB, ezért előbb a `memory_probe.py`-jal ki kell mérni, és ha nem fér el, checkpointinggal megy.
- **16 GB (Windows, RTX 5080):** CoAtNet-0 checkpointinggal. Checkpointing nélkül 13.9 GiB lenne lefoglalva a 15.9-ből, ami Windows alatt túl kevés tartalék; a checkpointing csak +27% időt jelent.
- **A két gép párhuzamosan a CoAtNet-0 két seedjét tanítja**, amit a mérési protokoll egymodelles döntésnél amúgy is előír.

**Futtatás** (a v8 E3 beállításaival; a run mappája `work/runs/train_v8/`):
```powershell
# Windows, 16 GB (seed 2, grad checkpointing)
$env:NOPAUSE = "1"; $env:TRAIN_CSV = "F:\Kaggle\data\train_v8.csv"; $env:HOLDOUT = "208"
$E3 = "--set data.image_size=320 --set data.laterality_canonical=true --set model.side_pooling=true --set model.target_attention=true --set train.ema_decay=0.999 --set train.num_workers=4 --set train.eval_num_workers=2 --set train.prefetch_factor=2".Split(" ")
$C0 = "--set model.backbone=coatnet_0_rw --set model.weights=sw_in1k --set data.encoder_normalization=inception".Split(" ")
.\run_holdout.bat C0_E3_img320_trainv8_s2 @C0 --set model.grad_checkpointing=true --set seed=2 @E3
```
```bash
# Linux, 24 GB (seed 42, grad checkpointing nélkül); a TRAIN_CSV alapértéke már a train_v8
./run_holdout.sh C0_E3_img320_trainv8 --set model.backbone=coatnet_0_rw --set model.weights=sw_in1k --set data.encoder_normalization=inception --set model.grad_checkpointing=false --set data.image_size=320 --set data.laterality_canonical=true --set model.side_pooling=true --set model.target_attention=true --set train.ema_decay=0.999 --set train.num_workers=4 --set train.eval_num_workers=2 --set train.prefetch_factor=2
```

**Kísérleti terv és előre rögzített döntési szabály:**
1. **CoAtNet-0, két seed.** Összevetés a 208-on (best epoch és a görbe ep6–10 átlaga), páros bootstrappal a V2S v8-hoz (0.9332) és a B0 v8-hoz (0.9292). A két seed szórása a run-to-run zajt is mutatja.
2. **Ensemble-szint (ez dönt).** Páronkénti Spearman a három CNN-nel, és két változat:
   - **H4** = H3_v8 + C0;
   - **H3-csere:** a C0 a legkevésbé hozzájáruló tag helyett.
   Viszonyítás: H3_v8 (0.9393).
3. **LB-próba csak akkor**, ha a H4 vagy a csere legalább +0.003 a 208-on, és a C0 korrelációja a tagokkal alacsonyabb, mint a tagoké egymással.
4. **coatnet_rmlp_1_rw2** csak akkor, ha a C0 egyedül legalább a B0 szintjén van (≥ 0.929), vagy ensemble-szinten pozitív. A 24 GB-os gépen fut; közben a 16 GB-os szabad.

**Eredmény (2026-10-10 – 10-11): egyedül V2S-szintű, negyedik tagként LB 0.943.**
- **Runok.** A seedek a tervhez képest fordítva futottak:
  - seed 42 Windowson, grad checkpointinggal: `C0_E3_img320_trainv8_holdout208_20261010_151746`, 17 epoch;
  - seed 2 Linuxon, checkpointing nélkül: `C0_E3_img320_trainv8_s2_holdout208_20261010_153116`, 19 epoch.

  Mindkettő később tetőzik, mint a CNN-ek (ep 11 és 13, azok ep 7–10).

| Modell | 208 | new-100 | 58 | Andrew50 | görbe ep6–10 | best ep |
|---|---|---|---|---|---|---|
| B0 v8 | 0.9292 | 0.9317 | 0.8985 | 0.9256 | 0.9272 | 7 |
| V2S v8 | 0.9332 | 0.9371 | 0.9125 | 0.9193 | 0.9317 | 10 |
| C0 s42 | 0.9314 | 0.9386 | 0.9008 | 0.9317 | 0.9287 | 11 |
| C0 s2 | 0.9358 | 0.9412 | 0.9091 | 0.9376 | 0.9301 | 13 |
| C0 a két seed átlaga | 0.9376 | 0.9451 | 0.9112 | 0.9338 | – | – |

- **Páros bootstrap (208).**
  - V2S-hoz képest: s42 −0.0018 (P>0 = 0.30), s2 +0.0026 (0.74), a két seed átlaga +0.0044 (0.91).
  - s2 − s42: +0.0044 [−0.0026, +0.0116], ez seed-zaj.
- **Targetenként a két seed ±0.02–0.05-tel tér el, rendszerszerű mintával.**
  - Az s2 a Lateral Meniscuson (+0.027), a Lateral OA-n (+0.036), a Medial OA-n (+0.021) és a PF OA-n (+0.014) jobb.
  - Az s42 a Synovitisen (−0.018), a Baker's-en (−0.015) és a Medial Meniscuson (−0.013) jobb.
  - A minta a new-100-on és a 58-on is megmarad; ezért jobb a két seed átlaga bármelyik seednél.
  - A CNN-ekhez képest a C0 a MCL-en erős (0.91 vs. 0.88–0.90); a Medial OA-n a B0 a legjobb (0.965).

| Ensemble (208) | 208 | Δ a H3_v8-hoz [95% CI], P(>0) | new-100 | Andrew50 |
|---|---|---|---|---|
| H3_v8 | 0.9393 | – | 0.9440 | 0.9203 |
| H4 = H3 + C0 s42 | 0.9400 | +0.0007 [−0.0010, +0.0027], 0.79 | 0.9456 | 0.9268 |
| **H4 = H3 + C0 s2** | **0.9417** | +0.0024 [+0.0002, +0.0047], 0.99 | 0.9484 | 0.9320 |
| H4, a két C0 átlaga egy helyen | 0.9411 | +0.0018 [+0.0000, +0.0036], 0.98 | 0.9474 | 0.9286 |
| H5 = H3 + mindkét C0 | 0.9414 | +0.0021 [−0.0005, +0.0049], 0.94 | 0.9486 | 0.9348 |
| B0 cseréje C0 s2-re | 0.9417 | +0.0024 [−0.0004, +0.0053], 0.95 | 0.9485 | 0.9301 |
| R50 / V2S cseréje C0 s2-re | 0.9406 / 0.9409 | +0.0013 / +0.0016 | – | – |
| B0 / R50 / V2S cseréje C0 s42-re | 0.9397 / 0.9383 / 0.9381 | +0.0004 / −0.0010 / −0.0012 | – | – |

- **Spearman (targetenként átlagolva, a 208-on).**
  - C0 a CNN-ekkel: 0.80–0.86; a CNN-ek egymással: 0.81–0.87. A C0 tehát csak kicsit diverzebb.
  - Ez a számítás más, mint a 2. pontbeli (ott B0–V2S 0.904, itt 0.868), ezért csak ezen a táblán belül vethető össze.
- **Előre rögzített szabály: nem teljesült.**
  - Egyik változat sem érte el a +0.003-at.
  - A legjobb +0.0024 a jobbik seedből jön, tehát seed-kiválasztást tartalmaz. Seed-kiválasztás nélkül a nyereség +0.0018 – +0.0021.
  - Az LB-próbáról a felhasználó ennek ellenére döntött.
- **LB: `holdout4_v8_c0` (B0 + R50 + V2S v8 + C0 s2) = 0.943**, a `holdout3_v8` 0.942 volt. Ez az új legjobb. A lokális +0.0024 iránya tartott, a nagysága kb. a felére esett, ami a 3 tizedesre kerekített LB-n zajszintű.
  - Új Kaggle dataset: `knee-c0-holdout-v8` (`best.pt` + `config.yaml`).
  - A `srcknee2` kód-datasetet frissíteni kellett (CoAtNet-adapter), és a Kaggle-en kell a timm.
  - A notebook 4 checkpointja 3 input-csoportban fut, mert a C0 `inception` normalizálással olvas. A lokális próbán a 3 publikus studyn lefutott.
- **Döntés:** a `holdout4_v8_c0` a legjobb beküldés. Az rmlp_1_rw2 feltétele teljesül (C0 ≥ 0.929), ez a következő lépés. Viszonyítás: a H4 (0.9417), és a B0 cseréje is.
- **Kód:**
  - `--ensemble holdout4_v8_c0` (`build_04_notebook.py`);
  - scriptek: `python notebooks/c0_check.py` (egyedi modellek, bootstrap, Spearman, ensemble-ek) és `python notebooks/c0_per_target.py` (kategóriánkénti AUC a négy részhalmazon).

**rmlp_1_rw2 futtatás** (Windows, 16 GB, seed 42, grad checkpointing; becslés ~35 perc/epoch):
```powershell
$env:NOPAUSE = "1"; $env:TRAIN_CSV = "F:\Kaggle\data\train_v8.csv"; $env:HOLDOUT = "208"
$E3 = "--set data.image_size=320 --set data.laterality_canonical=true --set model.side_pooling=true --set model.target_attention=true --set train.ema_decay=0.999 --set train.num_workers=4 --set train.eval_num_workers=2 --set train.prefetch_factor=2".Split(" ")
$CR1 = "--set model.backbone=coatnet_rmlp_1_rw2 --set model.weights=sw_in12k_ft_in1k --set data.encoder_normalization=inception".Split(" ")
.\run_holdout.bat CR1_E3_img320_trainv8 @CR1 --set model.grad_checkpointing=true @E3
```

**CR1 (coatnet_rmlp_1_rw2) eredménye (2026-10-11): nincs nyereség, lezárva.** Részletes táblázat: `doc/coatnet_vs_cnn_trainings.md` (generálja: `python notebooks/coatnet_compare.py`).
- **Runok:**
  - seed 42 Windowson, checkpointinggal: `CR1_E3_img320_trainv8_holdout208_20261010_224418`, best ep11, 2116 s/epoch, ami a C0 ~1.7-szerese;
  - seed 2 Linuxon, szintén checkpointinggal: `CR1_E3_img320_trainv8_s2_holdout208_20261010_225409`, best ep10, 2599 s/epoch.

| Modell (208, best ep) | 208 | new-100 | 58 | Andrew50 | görbe ep6–10 |
|---|---|---|---|---|---|
| V2S v8 | 0.9332 | 0.9371 | 0.9125 | 0.9193 | 0.9317 |
| C0 s42 / s2 | 0.9314 / 0.9358 | 0.9386 / 0.9412 | 0.9008 / 0.9091 | 0.9317 / 0.9376 | 0.9287 / 0.9301 |
| CR1 s42 / s2 | 0.9332 / 0.9303 | 0.9381 / 0.9413 | 0.9072 / 0.9113 | 0.9328 / 0.8954 | 0.9312 / 0.9273 |
| a két seed átlaga: C0 / CR1 | 0.9376 / 0.9350 | 0.9451 / 0.9432 | 0.9112 / 0.9124 | 0.9338 / 0.9199 | – |

- **Egyedül V2S-szintű, a C0-nál kicsit gyengébb.**
  - CR1 s42 − V2S: −0.0000 [−0.0066, +0.0066].
  - A két seed átlaga, CR1 − C0: −0.0026 [−0.0074, +0.0023], P>0 = 0.13.
  - Sem a 12k előtanítás, sem a nagyobb modell nem hozott előnyt.
- **Spearman:** a CR1 a C0-hoz hasonlít (0.83–0.905), a CNN-ekkel 0.825–0.861 a korreláció. Új diverzitást nem hoz egy olyan ensemble-nek, amelyben már van CoAtNet.
- **Ensemble a H4-hez (0.9417) képest:**
  - H3 + CR1 a C0 s2 helyett: −0.0012 (s42) és −0.0019 (s2);
  - H4 + CR1 s42: +0.0001 [−0.0014, +0.0015];
  - H4 + mindkét CR1 seed átlaga: +0.0000;
  - mind a 7 modell: −0.0006;
  - **H4, a B0 helyett CR1 s42-vel: 0.9422, +0.0005 [−0.0014, +0.0022].** Ez zajszintű, a jobbik seedet választja, és a CR1 lassabb a B0-nál.
- **Targetenként a CR1 erősségei:** ACL (s42 0.986), Effusion (s2 0.978) és Synovitis (s2 0.911, a legjobb egyedi érték). **Gyengeségei:** Fracture (0.951–0.953) és az s2-ben a PF OA / Lateral OA (0.843 / 0.846). A H4-ben a nyereségek és a veszteségek kioltják egymást.
- **Döntés:** a CR1-et nem tesszük az ensemble-be, LB-próba nem kell; a `holdout4_v8_c0` (LB 0.943) marad a legjobb.
- **A CoAtNet-irány tanulsága:** egy architekturálisan eltérő tag egyszer hoz nyereséget, egy második CoAtNet már nem.

### 8. ConvNeXt-T encoder (2026-10-12): első két run gyenge, a DINOv3-é hibás beállítás miatt
Backbone-támogatás: `convnext_tiny` (`in12k_ft_in1k` / `fb_in22k_ft_in1k` / `dinov3_lvd1689m`), `convnext_nano`, `convnextv2_tiny`; `imagenet` normalizálás, `model.drop_path_rate`. Mindkét run E3, img320, train_v8, drop path 0.1, encoder_lr 1e-4 / head_lr 3e-4, warmup 2, 25 epoch, early stopping 5.

| Run | Gép | 208 best (ep) | ep0 → ep2 | train loss ep10 |
|---|---|---|---|---|
| `CNT22_E3_img320_trainv8_holdout208_20261011_225628` (fb_in22k_ft_in1k) | Windows, 1 GPU | 0.9251 (10) | 0.797 → 0.890 | 0.363 |
| `CNTD3_E3_img320_trainv8_holdout208_20261011_224246` (dinov3_lvd1689m) | Linux, **2 GPU DDP (az első teljes DDP-run)** | 0.8884 (18) | 0.757 → 0.776 | 0.465 |
| összevetés: C0 s2 | Linux, 1 GPU | 0.9358 (13) | 0.814 → 0.905 | 0.378 |

- **CNT22: tanul, de B0/R50-szinten áll meg** (0.925), ep8–10 után túltanul (train loss tovább esik, a 208 csökken). Nem hibás, csak nem jobb. Ensemble-hozzájárulás (Spearman, H4 + CNT22) még nincs mérve.
- **CNTD3: alultanul**: lassan indul, a 0.888 messze a többi alatt. Diagnózis (mérés 8 valódi val-studyn, a run configjával, ugyanaz a bemenet és loss mindkét súlyra):
  - a DINOv3 súlyok RMS-e stage-enként 0.014–0.018, az fb22k-é 0.066–0.071 (**4–5× kisebb**). Az AdamW lépése paraméterenként ~lr, így 1e-4 relatíve 4–5× nagyobb változás: ugyanaz a lr a DINOv3-nak ~4–5e-4-es „fb22k-ekvivalens" lr, ami az első epochokban szétszedi az előtanított feature-öket;
  - a kezdeti encoder grad norm 15.7 (fb22k 0.53, C0 4.3, B0 0.67), vagyis a DINOv3-háló sokkal érzékenyebb; a végső LayerNorm kimenete is ~2× nagyobb skálájú (head.norm súly max 13.6);
  - a `grad_clip=1.0` minden lépést ~1/16-ra vág, de AdamW mellett a konstans skálázás alig számít, ezért ez nem önálló ok.
  - **Zavaró tényező:** ez volt az első teljes DDP-run, a DDP-útvonal valódi adaton még nincs összevetve egy GPU-s runnal.
- **Következő lépések** (rövid, 4–5 epoches próbák elég, a görbe ep0–2 különbsége egyértelmű):
  1. DINOv3 egy GPU-n Windowson `train.encoder_lr=2e-5`-tel (≈ az fb22k relatív lépésmérete 1e-4-nél). Ha így ep2-re ~0.89 körül van, a lr volt a hiba.
  2. DDP-kontroll: fb22k 2 GPU-n Linuxon, 3 epoch; a CNT22 görbéjével (0.797 / 0.845 / 0.890) kell egyeznie zajon belül.
  3. CNT22-höz: előbb Spearman / H4-ensemble-hozzájárulás; hangolni (alacsonyabb lr, drop path 0.2) csak akkor érdemes, ha diverzitást hoz.

**Második kör (2026-10-12): a DDP rendben van, a DINOv3-nál a lr volt a hiba; ensemble-ben zajszintű.** Script: `python notebooks/convnext_compare.py`.
- **Runok:**
  - `CNT22_DDPcheck_E3_img320_trainv8_holdout208_20261012_064352`: fb22k, Linux 2 GPU DDP, ugyanaz a config és seed (42), mint a CNT22-é; végigfutott (15 epoch, best ep9), 468 s/epoch (a CNT22 637 s-a Windowson);
  - `CNTD3_lr2e-5_E3_img320_trainv8_holdout208_20261012_064601`: dinov3, Windows 1 GPU, `train.encoder_lr=2e-5`; **még fut**, az alábbi számok az ep10-es (eddigi legjobb) állapotot mutatják.

| Modell | 208 | new-100 | 58 | Andrew50 | görbe ep6–10 | ep0 / 1 / 2 | best ep |
|---|---|---|---|---|---|---|---|
| V2S v8 | 0.9332 | 0.9371 | 0.9125 | 0.9193 | 0.9317 | – | 10 |
| C0 s2 | 0.9358 | 0.9412 | 0.9091 | 0.9376 | 0.9301 | 0.814 / – / 0.905 | 13 |
| CNT22 (Windows) | 0.9251 | 0.9282 | 0.8947 | 0.9247 | 0.9236 | 0.797 / 0.845 / 0.890 | 10 |
| CNT22 DDP (Linux) | 0.9327 | 0.9343 | 0.9011 | **0.9470** | 0.9295 | 0.796 / 0.852 / 0.895 | 9 |
| a két CNT22 átlaga | 0.9314 | 0.9338 | 0.9000 | 0.9370 | – | – | – |
| CNTD3 lr 2e-5 (ep10-ig) | 0.9286 | 0.9342 | 0.8999 | 0.9248 | 0.9229 | 0.790 / 0.813 / 0.859 | 10, még emelkedik |

- **DDP-kontroll: átment.** Az első 3 epoch zajon belül egyezik a CNT22 görbéjével. A végeredmény a DDP-runban jobb: +0.0077 [+0.0020, +0.0139], P>0 = 1.00. Ugyanaz a seed, de más a mintasorrend (DistributedSampler) és az augmentáció RNG-je, tehát ez gyakorlatilag egy második seed. A C0 két seedje +0.0044-gyel tért el; a +0.0077 a seed-zaj felső széle. A Linux/Windows korábban nem mutatott rendszerszerű eltérést (C0 s2 > s42, de CR1 s42 > s2). **Következtetés:** a DDP-útvonal használható, de abból, hogy itt jobb lett, nem következik, hogy a DDP jobb. A 2 GPU csak 1.36× gyorsulást hoz (468 vs. 637 s/epoch, 3090 vs. 5080).
- **CNTD3: a lr volt a hiba.** 2e-5-tel ep2-ben 0.859 (1e-4-gyel 0.776 volt), ep10-ben 0.9286 (korábban a best 0.8884). Lassabban tanul, mint az fb22k, de még emelkedik, és a train loss ep10-ben 0.402 (a CNT22-é 0.363), vagyis a túltanulás még nem kezdődött el.
- **Egyedül (208, páros bootstrap):** CNT22 DDP − V2S −0.0005 [−0.0099, +0.0086]; − C0 s2 −0.0031 (P>0 = 0.23). CNTD3 − V2S −0.0047; − CNT22 DDP −0.0042 (P>0 = 0.15). A ConvNeXt-T legfeljebb V2S-szintű. Az Andrew50-en viszont a CNT22 DDP 0.9470, ez az eddigi legjobb egyedi érték.
- **Targetenként (208).**
  - A CNT22 DDP a Windows-futásnál szinte minden targeten jobb: Synovitis +0.018, Lateral Meniscus +0.014, Effusion +0.013.
  - A CNTD3 a CNT22 DDP-hez képest a Medial Meniscuson (+0.012, 0.962, ez a legjobb egyedi érték) és a Synovitisen (+0.011, 0.912, szintén a legjobb) erősebb. Gyengébb a Lateral OA-n (−0.029), a Contusionön (−0.025), a PF OA-n (−0.015) és a MCL-en (−0.014).
  - Mindkét ConvNeXt gyenge a Fracture-ön (0.938–0.946, a többieké 0.966) és az ACL-en (0.959–0.964, a V2S-é 0.981).
- **Spearman (208).**
  - A két CNT22 egymással 0.899. Ez azonos architektúra mellett a run-to-run korreláció.
  - A CNT22 DDP a többiekkel 0.815–0.866, vagyis nem diverzebb, mint a CNN-ek egymással.
  - A CNTD3 a V2S-sel 0.776, a B0-val 0.804, a C0-val 0.809, a ConvNeXt-ek közül ez a legdiverzebb.

| Ensemble (208) | 208 | Δ a H4-hez [95% CI], P(>0) | new-100 | Andrew50 |
|---|---|---|---|---|
| H4 (LB 0.943) | 0.9417 | – | 0.9484 | 0.9320 |
| H4 + CNT22 (Windows) | 0.9415 | −0.0003, 0.36 | 0.9473 | 0.9333 |
| H4 + CNT22 DDP | 0.9428 | +0.0011 [−0.0007, +0.0030], 0.88 | 0.9490 | 0.9381 |
| H4 + CNTD3 (ep10) | 0.9426 | +0.0008 [−0.0006, +0.0024], 0.86 | 0.9494 | 0.9338 |
| H4 + CNT22 DDP + CNTD3 | 0.9434 | +0.0017 [−0.0005, +0.0040], 0.94 | 0.9495 | 0.9342 |
| H4, a B0 helyett CNT22 DDP | 0.9434 | +0.0017 [−0.0001, +0.0039], 0.96 | 0.9492 | 0.9374 |
| H4, a B0 helyett CNTD3 | 0.9425 | +0.0008, 0.76 | 0.9497 | 0.9303 |

- **A +0.003-as szabály nem teljesül.** A legjobb változat +0.0017, és a CNT22-nél a jobbik „seedet" használja; a Windows-os CNT22 semmit nem hoz.
- **CNTD3 végeredménye (lefutott, 16 epoch):** a best az ep10 maradt, 0.9286. Ep11–12-ben 0.9285 a plató, utána lassan esik (ep15: 0.9266), közben a train loss tovább csökken (0.402 → 0.353). A fenti táblák számai így véglegesek. Az előre rögzített 0.933 nem teljesült, az ensemble-nyereség +0.0008: **a DINOv3-as ConvNeXt-T lezárva.**

**Még egy CNT22-próba (a felhasználó döntése, 2026-10-12): rövidebb ütemezés és erősebb regularizáció, két seeddel.**
- **Miért:** mindkét CNT22 ep9–10-ben tetőzik. Ekkor az encoder lr még ~7e-5, a 25 epochos cosine csak ~35%-ig ért le, utána a modell túltanul. Magas lr-nél a végpont erősen függ a mintasorrendtől, ez magyarázhatja a két futás közötti 0.008-at.
- **Változtatások egyben:**
  - `train.max_epochs=12`, így a cosine a csúcs környékén ér le nullára;
  - `train.weight_decay=0.05`, a ConvNeXt fine-tune receptje szerint (az 1e-4 AdamW mellett gyakorlatilag nulla);
  - `model.drop_path_rate=0.2`.
- **Két seed:** egy futásból a zaj miatt nem lehet dönteni. Viszonyítás: a két eddigi CNT22 átlaga, 0.9314.
- **Siker:** ha a két új futás átlaga ≥ 0.934, és a két futás egymástól < 0.004-gyel tér el, jöhet a H4-ensemble próba. Ha nem javul, a következő lépés: `model.weights=in12k_ft_in1k`, előtte súlyskála-ellenőrzés; utána layer-wise lr decay, ami kódváltozást igényel a `parameter_groups`-ban.
- **Workerek:** Linuxon nem kell felülírni őket (alap 8/4). A két DDP-s convnext_tiny run 8/4-gyel 468 s/epoch, 4/2-vel 467 s/epoch volt, tehát nem az adatbetöltés a szűk keresztmetszet.
```bash
# Linux, 2 GPU DDP, seed 2
./run_holdout.sh --gpu 0,1 CNT22reg_E3_img320_trainv8_s2 --set model.backbone=convnext_tiny --set model.weights=fb_in22k_ft_in1k --set model.drop_path_rate=0.2 --set train.weight_decay=0.05 --set train.max_epochs=12 --set seed=2 --set model.grad_checkpointing=false --set data.image_size=320 --set data.laterality_canonical=true --set model.side_pooling=true --set model.target_attention=true --set train.ema_decay=0.999
```
```powershell
# Windows, seed 42 (a CNTD3 után)
$env:NOPAUSE = "1"; $env:TRAIN_CSV = "F:\Kaggle\data\train_v8.csv"; $env:HOLDOUT = "208"
$E3 = "--set data.image_size=320 --set data.laterality_canonical=true --set model.side_pooling=true --set model.target_attention=true --set train.ema_decay=0.999 --set train.num_workers=4 --set train.eval_num_workers=2 --set train.prefetch_factor=2".Split(" ")
$CNT = "--set model.backbone=convnext_tiny --set model.weights=fb_in22k_ft_in1k --set model.drop_path_rate=0.2 --set train.weight_decay=0.05 --set train.max_epochs=12".Split(" ")
.\run_holdout.bat CNT22reg_E3_img320_trainv8 @CNT --set model.grad_checkpointing=false @E3
```

**CNT22reg eredménye (2026-10-12): nincs javulás, a ConvNeXt-irány lezárva.**
- **Runok:**
  - `CNT22reg_E3_img320_trainv8_s2_holdout208_20261012_093739`: Linux DDP, seed 2, lefutott; best ep9, **0.9248**;
  - `CNT22reg_E3_img320_trainv8_holdout208_20261012_102741`: Windows, seed 42, lefutott (12 epoch); best ep8, **0.9265** (new-100 0.9381, 58 0.8902, Andrew50 0.9073).
  - A két seed közötti különbség 0.0017 (s2 − s42: −0.0017 [−0.0079, +0.0049]). A CNT22-nél ez 0.0077 volt, tehát a rövid cosine a szórást csökkentette, a szintet nem.
  - A két seed predikció-átlaga 0.9283, a két CNT22-é 0.9314: −0.0030 [−0.0108, +0.0059]. H4 + a REG két seedjének átlaga: +0.0004.
  - A négy fb22k-futás: 0.9251 / 0.9327 / 0.9248 / 0.9265, átlaguk 0.927.
- **A görbe.**
  - A rövidebb cosine a túltanulás okozta esést megszüntette: ep7–11 között 0.923–0.925 a plató, a végén nincs visszaesés.
  - A csúcs viszont nem lett magasabb. A train loss a végén 0.373; a CNT22-é ugyanennyi epoch alatt kb. 0.35.
- **Páros bootstrap (208):**
  - REG s2 − CNT22 Windows: −0.0003;
  - REG s2 − CNT22 DDP: −0.0079 (P>0 = 0.03);
  - a REG két futásának átlaga − a két CNT22 átlaga: −0.0052 (P>0 = 0.09; a s42 itt ep6-os állapot).
  - Az Andrew50-en a REG s2 0.893 (a CNT22 DDP-é 0.947).
- **Targetenként** a REG s2 az ACL-en (+0.007) és a Medial Meniscuson (+0.008) jobb, a MCL-en (−0.024) és a PF OA-n (−0.014) gyengébb. Rendszerszerű nyereség nincs.
- **Ensemble:** H4 + REG s2: −0.0001; a B0 cseréje: −0.0001.
- **Értelmezés:** négy fb22k-futásból három 0.925 körül van (0.9251, 0.9248, és a s42 is ezen a pályán halad). A CNT22 DDP 0.9327-je a szerencsés seed volt, nem a ConvNeXt-T valódi szintje. A ConvNeXt-T fb22k ~0.925-ön, B0/R50-szinten áll.
- **Sikerkritérium:** a két futás átlagának el kellene érnie a 0.934-et, ehhez a s42-nek ≥ 0.943 kellene. Ez elérhetetlen, a Windows-run eredménye már nem változtat a döntésen.
- **Döntés:** a ConvNeXt-irány (fb22k, DINOv3, regularizált) lezárva; a H4 (LB 0.943) marad a legjobb.

**Utolsó ConvNeXt-próba (a felhasználó döntése, 2026-10-12): in12k előtanítás, `encoder_lr=4e-5`.**
- **Súlyskála.** A ConvNeXt-T súlyainak RMS-e stage-enként, és a layer-scale γ átlaga:

| súly | RMS stage-enként | γ átlag |
|---|---|---|
| fb_in22k_ft_in1k | 0.066–0.071 | 2.51 |
| in12k_ft_in1k | 0.025–0.028 | 1.75 |
| dinov3_lvd1689m | 0.014–0.018 | 1.32 |

  Az in12k súlyok ~2.6×-kal kisebbek, mint az fb22k-éi. A DINOv3-nál tanultak alapján ezért az encoder lr 1e-4 / 2.6 ≈ 4e-5.
- **Config:** minden más az eredeti CNT22-é (25 epoch, wd 1e-4, drop path 0.1, warmup 2, early stopping 5), csak a súly és az encoder lr más.
- **Miért érdemes:** az előtanítás eddig nagyobb hatással volt a diverzitásra, mint a regularizáció. A CNTD3 a legdiverzebb ConvNeXt (Spearman a V2S-sel 0.776). A várható nyereség kicsi, mert a CoAtNet-nél a 12k előtanítás (CR1) nem hozott semmit.
- **Layer-wise lr decay: kimarad.** A CNT22reg-ben a túltanulás megszűnt, a csúcs mégis 0.925 maradt, tehát nem a felejtés a szűk keresztmetszet. Kódváltozást is igényelne.
- **Döntési szabály (két seed után):** siker, ha a két futás átlaga ≥ 0.934 (a két CNT22 átlaga 0.9314), vagy ha a H4-hez adva ensemble-ben legalább +0.003. Ha egyik sem teljesül, a ConvNeXt végleg lezárva.
```bash
# Linux, 2 GPU DDP, seed 2
./run_holdout.sh --gpu 0,1 CNT12_lr4e-5_E3_img320_trainv8_s2 --set model.backbone=convnext_tiny --set model.weights=in12k_ft_in1k --set train.encoder_lr=4e-5 --set model.drop_path_rate=0.1 --set seed=2 --set model.grad_checkpointing=false --set data.image_size=320 --set data.laterality_canonical=true --set model.side_pooling=true --set model.target_attention=true --set train.ema_decay=0.999
```
```powershell
# Windows, seed 42
$env:NOPAUSE = "1"; $env:TRAIN_CSV = "F:\Kaggle\data\train_v8.csv"; $env:HOLDOUT = "208"
$E3 = "--set data.image_size=320 --set data.laterality_canonical=true --set model.side_pooling=true --set model.target_attention=true --set train.ema_decay=0.999 --set train.num_workers=4 --set train.eval_num_workers=2 --set train.prefetch_factor=2".Split(" ")
$CNT = "--set model.backbone=convnext_tiny --set model.weights=in12k_ft_in1k --set train.encoder_lr=4e-5 --set model.drop_path_rate=0.1".Split(" ")
.\run_holdout.bat CNT12_lr4e-5_E3_img320_trainv8 @CNT --set model.grad_checkpointing=false @E3
```

**CNT12 közbenső eredménye (2026-10-12; az s2 lefutott, az s42 ep6-nál tart): ez a legjobb ConvNeXt, de ensemble-ben még mindig kicsi.**
- **Runok:**
  - `CNT12_lr4e-5_E3_img320_trainv8_s2_holdout208_20261012_122058`: Linux DDP, 15 epoch; best ep9, **0.9314**, görbe ep6–10 0.9303. A new-100-on **0.9428**, ez az eddigi legjobb egyedi érték (C0 s2: 0.9412).
  - `CNT12_lr4e-5_E3_img320_trainv8_holdout208_20261012_133340`: Windows, seed 42, még fut; ep6-ban 0.9303 (az s2 ugyanitt 0.9283-on járt).
- **Összevetés a fb22k-val:** a négy fb22k-futás átlaga 0.927, a CNT12 s2 0.9314. Az előtanítás és a súlyskálához igazított lr együtt kb. +0.004.
  - A Fracture javult a legtöbbet: 0.952–0.954 a fb22k 0.933–0.946-jához képest. Az ACL 0.97.
  - A Lateral OA (0.853–0.859) és a MCL (s42: 0.861) gyenge maradt.
- **Két seed predikció-átlaga (s42 ep6-tal):** 0.9351. Ugyanígy számolva a két CNT22-é 0.9314 (+0.0037 [−0.0030, +0.0108], P>0 = 0.84), a két C0 seedé 0.9376.
- **Spearman a többi modellel:** 0.79–0.84, nem diverzebb, mint a CNN-ek egymással.
- **Ensemble a H4-hez képest:**
  - H4 + C12 s2: +0.0011 (P>0 = 0.88);
  - H4 + a két seed átlaga: +0.0012 [−0.0001, +0.0026], P>0 = 0.96;
  - a B0 cseréje: +0.0011–0.0012.
- **Döntési szabály most:** az egyedi feltétel (a két seed átlaga ≥ 0.934) a predikció-átlaggal már teljesül, a futásonkénti best-ek átlagával (~0.931) még nem. Az ensemble-feltétel (+0.003) messze nem teljesül.

**CNT12 végeredménye (mindkét seed lefutott, 2026-10-12).**
- **s42 (Windows):** best ep9, **0.9347**; görbe ep6–10 0.9330; new-100 **0.9482**, ez az eddigi legjobb egyedi érték; 58: 0.8956; Andrew50: 0.9204.

| Modell | 208 | new-100 | 58 | Andrew50 |
|---|---|---|---|---|
| V2S v8 | 0.9332 | 0.9371 | 0.9125 | 0.9193 |
| C0 s2 | 0.9358 | 0.9412 | 0.9091 | 0.9376 |
| CNT12 s2 / s42 | 0.9314 / 0.9347 | 0.9428 / 0.9482 | 0.9000 / 0.8956 | 0.9091 / 0.9204 |
| a két seed predikció-átlaga: C12 / C0 | 0.9360 / 0.9376 | 0.9477 / 0.9451 | 0.9019 / 0.9112 | 0.9184 / 0.9338 |

- **Egyedül:** V2S- és C0-szintű. A két seed átlaga a V2S-hez képest +0.0028 [−0.0050, +0.0111], a két CNT22 átlagához képest +0.0046.
  - A fb22k-hoz képest a Fracture, a MCL (s42: 0.915) és a Lateral Meniscus javult.
  - A két seed egymással 0.919 Spearman-korrelációjú, a többi modellel 0.80–0.84.
- **Ensemble a H4-hez képest:**

| Változat | 208 | new-100 | Andrew50 |
|---|---|---|---|
| H4 + a két C12 seed átlaga | +0.0014 [−0.0002, +0.0030], 0.95 | +0.0025 [−0.0003, +0.0061], 0.96 | −0.0015 |
| B0 cseréje a két C12 seed átlagára | +0.0016 [−0.0003, +0.0037], 0.95 | **+0.0034** [−0.0005, +0.0078], 0.96 | −0.0033 |
| B0 cseréje C12 s42-re | +0.0013, 0.91 | +0.0032, 0.95 | −0.0040 |

- **Döntési szabály:**
  - Az egyedi feltétel teljesül: a predikció-átlag 0.9360, a best-ek átlaga 0.9331.
  - Az ensemble-feltétel nem: a 208-on +0.0016, a +0.003 kevesebb mint fele.
  - A new-100-on (a fő mérce) a B0 cseréje +0.0034, de az Andrew50-en −0.003.
  - Összevetésül a C0-nál: a 208-on +0.0024, a new-100-on +0.0044 volt, és ebből az LB-n +0.001 lett. Ugyanilyen arányban itt LB-n ~0.0005–0.001 várható, ami a 3 tizedesre kerekített LB-n valószínűleg nem látszik.
- **Döntés:** a ConvNeXt-hangolás lezárva. A CNT12 a legjobb ConvNeXt, és egyedül teljes értékű tag (V2S/C0-szint). LB-próba csak akkor, ha a felhasználó beküldést szán rá; a jelölt a B0 helyett a két CNT12 seed átlaga (5 checkpoint).
- **Tanulság:** az encoder lr-t a súlyok skálájához kell igazítani (in12k: 1e-4 / 2.6). Ez nagyobb hatású volt, mint a regularizáció vagy az ütemezés.

**LB-próba: `holdout5_v8_cnt12` = 0.945, ez az új legjobb** (2026-10-19-én rögzítve). A H4 a B0 helyett a két CNT12 seeddel, 5 checkpoint, sigmoid-átlag, mind train_v8, holdout208, E3, img320:
- R50 `R50_E3_img320_trainv8_holdout208_20261007_003442`;
- V2S `V2S_E3_img320_trainv8_holdout208_20261007_053337`;
- C0 s2 `C0_E3_img320_trainv8_s2_holdout208_20261010_153116` (Linux; dataset `knee-c0-holdout-v8`, ugyanaz, mint a H4-ben);
- CNT12 `CNT12_lr4e-5_E3_img320_trainv8_holdout208_20261012_133340` (s42);
- CNT12 s2 `CNT12_lr4e-5_E3_img320_trainv8_s2_holdout208_20261012_122058`.

Néhány megjegyzés:
- A beküldés a `build_04_notebook.py` `--ensemble holdout5_v8_cnt12` bejegyzéséből készült. Az öt modell mind a 2026-10-12-i laterality-javítás (2396ed0) előtt tanult, a régi, tag-first kerettel.
- **A H4-hez (0.943) képest +0.002.** A lokális előrejelzés +0.0016 volt (208), illetve +0.0034 (new-100). A +0.003-as szabályt a 208-on nem érte el, az LB-n mégis nyert. Ezúttal a new-100 jelezte jobban előre az irányt.
- **A v11-es összetételeket is veri** (holdout6_v11: 0.941). A különbség (−0.004) valószínűleg a 242 tanító study kiesésének ára, ami lokálisan még a seed-zajon belül maradt (13. pont). Az összetétel azonban nem teljesen azonos: a v11-es hatosból hiányzik a R50.

**Lehűtési próba (CNT12cool, 2026-10-12): negatív, a csúcs utáni romlás túltanulás, nem lr-zaj.**
- **Kérdés (a felhasználó felvetése):** ha a csúcsnál esne az lr, javulna-e még a modell?
- **Beállítás:** `train.init_weights` = a CNT12 best.pt-je (ep9, EMA-súlyok); utána 4 epoch, warmup nélkül, cosine lefutással; encoder lr 1e-5 → 0, head lr 3e-5 → 0. Ez a csúcsbeli lr kb. harmada.
- **Runok:**
  - `CNT12cool_E3_img320_trainv8_s2_holdout208_20261012_182722` (Linux DDP);
  - `CNT12cool_E3_img320_trainv8_holdout208_20261012_183342` (Windows).

| | kiindulás | ep0 | ep1 | ep2 | ep3 (utolsó) | Δ az utolsó epochra |
|---|---|---|---|---|---|---|
| s2 | 0.9314 | 0.9242 | 0.9244 | 0.9227 | 0.9224 | −0.0090 |
| s42 | 0.9347 | 0.9312 | 0.9289 | 0.9282 | 0.9274 | −0.0073 |

- **Alhalmazok** (a lehűtött run legjobb epochja a kiinduláshoz képest; ezen a szelekciós torzítás a lehűtésnek kedvez):
  - s2: new-100 0.9428 → 0.9329, Andrew50 0.9091 → 0.8942;
  - s42: new-100 0.9482 → 0.9421, Andrew50 0.9204 → 0.9232.
- **Értelmezés:**
  - Nagyon alacsony lr mellett is minden epochban romlik a 208, miközben a train loss tovább esik (0.354 → 0.326).
  - Tehát a csúcs után a modell a tanítóhalmazt tanulja tovább (túltanul), nem az lr-zaj a gond. Az alacsony lr ezt csak lassítja, nem állítja meg.
  - Ezzel egybevág, hogy a CNT22reg-ben a rövid cosine sem emelte a csúcsot, és hogy az EMA már eleve kisimítja a súlyokat.
- **Döntés:** lehűtési fázist a többi modellre (V2S, C0) sem érdemes tenni. A csúcsnál megállni (best.pt) a helyes stratégia.
  - Ami eddig rendre javított, az több seed predikcióinak átlaga: C0 +0.002, CNT12 +0.001–0.005 az egyedi seedekhez képest.
  - Ugyanaz a modell több seeddel olcsóbb nyereség, mint az ütemezés hangolása.

**Javítás: a ConvNeXt-runok egy része más normalizálással tanult (2026-10-12, a beküldés előkészítésekor derült ki).**
- A CNT22, a CNT22 DDP és a két CNTD3 `data.encoder_normalization=imagenet`-tel tanult. A CNT22reg, a CNT12 és a CNT12cool `mri_scalar`-ral, mert a parancsaikból kimaradt a `--set data.encoder_normalization=imagenet`.
- **A beküldést ez nem érinti:** a notebook minden checkpointot a saját `config.yaml`-ja szerinti normalizálással olvas, vagyis pontosan úgy, ahogy tanult. A lokális 208-as számok is ugyanezekre a checkpointokra vonatkoznak.
- **Az összevetéseket kis mértékben torzítja:**
  - `mri_scalar` = (x − 0.5) / 0.25, `imagenet` ≈ (x − 0.456) / 0.225; ez kb. 0.2-es eltolás és ~10% skálakülönbség a normalizált bemeneten, a LayerNorm-os ConvNeXt-nek kis zavar;
  - a CNT22reg és a CNT22 különbsége részben ebből is jöhet, ugyanígy a CNT12 (in12k + lr 4e-5) előnye a fb22k-val szemben. A lehűtési próbát nem érinti, mert ott a kiindulással azonos volt;
  - a DINOv3 lr-tanulsága tiszta, ott mindkét run `imagenet`-tel ment.
- Hogy az `imagenet` vagy az `mri_scalar` jobb-e a CNT12-nek, azt nem mértük.

### 9. Új validációs halmaz: ref220 (kiegyensúlyozott) + train_v10 (2026-10-13)
**Miért:** elkészült a 404 studys radiológus-referencia (`F:/Kaggle/data/train_labeled_404_reference.csv` = 208 régi + 96 új YC + 100 új Andrew; összesítő: `reference_404_summary.xlsx`). A ritka targetek (MCL, Contusion, Fracture) AUC-ja a 208-on kevés pozitív miatt zajos volt.
- **Válogatás (felhasználói döntések):** 220 study a 404-ből, szabad válogatással (a 158/208 nem kötelező elem). Cél: a Σ_target (1/pos + 1/neg) minimuma, ami az AUC-becslések varianciájával arányos. 20 véletlen indulás, 1-1 csere hegymászással, seed 2026. Költség: 0.275 (véletlen 220: ~0.395, mind a 404: 0.212).
- **Eredmény:** minden ritka pozitív bekerült, a gyakori targetek ~50% körül vannak.

| | ACL | MCL | MedMen | LatMen | MedOA | LatOA | PFOA | Eff | Syn | Baker | Contus | Fract |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pozitív a 220-ban | 70 | 23 | 110 | 94 | 82 | 66 | 95 | 110 | 106 | 82 | 45 | 58 |
| pozitív a 404-ben | 71 | 23 | 161 | 115 | 91 | 67 | 133 | 127 | 121 | 95 | 45 | 58 |
| arány a 220-ban | .32 | .11 | .50 | .44 | .38 | .30 | .43 | .51 | .49 | .38 | .21 | .26 |

- Forrás szerint a 220: 92 a 158-ból, 22 Andrew 50, 106 az új 196-ból. 28 maszkolt cella, 2612 érvényes.
- **train_v10** (felhasználói döntés: a maradék 184 a tanításba megy radiológus-címkével): a train_v8 a 404 nélkül (4003 sor, változatlan) + a 184 radiológus-címkével (kemény 0/1). Maszkolt radiológus-cella: a v8 LLM-értéke, ha volt (20 cella), különben üres = weight 0 (30 cella). Összesen 4187 study.
- **Figyelem:** a 220 pozitívban dúsított (sok többpatológiás térd), a holdout208 számaival nem vethető össze. A régi (train_v8-as) modellek a 220-ból 106 studyn tanultak (LLM-címkével), ezért a teljes 220-on nem mérhetők tisztán, csak a 208-ból jövő 114-es részén. A v8 → v10 összevetéshez újra kell tanítani; a train_v10-ben a 184 radiológus-címke maga is változás.
- Megvalósítás: `notebooks/build_ref220_train_v10.py`; kimenet `F:/Kaggle/data/train_labeled_220_reference.csv` (+ másolat a `notebooks/`-ban), `F:/Kaggle/data/train_v10.csv`, `notebooks/train_v10_radiologist_184.csv`. A split (`work/splits/holdout220/`) és a frozen ref (`work/labels/frozen_reference_ref220.csv`) elkészült, a check-fixed-split rendben van. Pipeline-kód nem változott, a `run_holdout` alapértékei maradtak (HOLDOUT=208, train_v8).

**Futtatás** (PowerShell):
```
$env:HOLDOUT = "220"; $env:TRAIN_CSV = "F:\Kaggle\data\train_v10.csv"; $env:NOPAUSE = "1"
.\run_holdout.bat E3_img320 --set data.image_size=320
```
Linuxon a két új CSV-t át kell másolni a DATA_ROOT-ba: `HOLDOUT=220 TRAIN_CSV=$DATA_ROOT/train_v10.csv ./run_holdout.sh ...`

**Első v10-run: CNT12 s42 (2026-10-13), ígéretes, de egy seed.**
- **Run:** `work/runs/train_v10/CNT12_lr4e-5_E3_img320_trainv10_holdout220_20261013_180539` (Windows, seed 42, a CNT12 v8-cal azonos beállítás, 17 epoch).
- **Nem csak a címke változott** a v8-as CNT12 s42-höz képest:
  - train_v10: 4187 study (a v8-run 4199-en tanult, ebből 196 az új refekből, LLM-címkével);
  - laterality-javítás (2396ed0, előbb geometria, aztán tag);
  - 8 maszkolt study (`notebooks/label_mask_studies.csv`, minden target weight 0); ebből 1 a 220-ban, 12 cella, így a 220-on 2600 érvényes cella van.
- **Görbe a 220-on:** best ep11, **0.9005** (ep10 0.9004, ep12 0.8990); ep6–10 átlag 0.8946. A csúcs két epochhal később van, mint a v8-nál (ep9).
  - A 220 részei: a 114 a 208-ból 0.9102, a 106 új 0.8864. A 106-hoz nincs tiszta v8-as összevetés.
  - Gyenge: Lateral OA 0.803, PF OA 0.864, Synovitis 0.856, MCL 0.881.
- **Tiszta összevetés a 114-en** (a 220 ∩ 208: egyik modell sem tanult rajta; ref220-címkék, a 208-hoz képest 4 cella eltérés). Szkript: scratchpad `v10_check.py`.

| Modell | 114 | n100-rész (43) | 58-rész (49) | A50-rész (22) |
|---|---|---|---|---|
| **CNT12 v10 s42** | **0.9102** | 0.9128 | 0.9004 | **0.9050** |
| CNT12 v8 s42 / s2 | 0.9022 / 0.9021 | 0.9121 / 0.9130 | 0.8897 / 0.8921 | 0.8815 / 0.8635 |
| CNT12 v8 két seed átlaga | 0.9061 | 0.9162 | 0.8951 | 0.8772 |
| C0 v8 s42 / V2S v8 | 0.9007 / 0.9028 | 0.9080 / 0.9029 | 0.8980 / 0.9050 | 0.8715 / 0.8644 |

- **Páros bootstrap a 114-en:**
  - v10 − v8 s42 (azonos seed): **+0.0080** [−0.0014, +0.0173], P>0 = 0.95;
  - v10 − v8 két seed átlaga: +0.0041 [−0.0041, +0.0126], P>0 = 0.83;
  - a new-100-részen (43 study) nincs különbség (+0.0008); a nyereség a 58- és az A50-részből jön.
- **Per target (114, v10 − v8 s42):** Synovitis +0.049, MCL +0.029, Lateral Meniscus +0.025, Contusion +0.017, Effusion +0.011; Lateral OA −0.016, Fracture −0.009, ACL −0.006.
- **Ensemble a 114-en** (H4 = B0 + R50 + V2S + C0 s2, v8, prob-átlag; 0.9119):

| Változat | 114 | Δ a H4-hez képest |
|---|---|---|
| H4 + a két v8 CNT12 seed átlaga | 0.9141 | +0.0022 [−0.0002, +0.0048], 0.97 |
| H4 + CNT12 v10 | 0.9163 | **+0.0044** [+0.0017, +0.0074], 1.00 |
| B0 cseréje a két v8 CNT12 átlagára | 0.9146 | +0.0027 [−0.0004, +0.0059], 0.96 |
| B0 cseréje CNT12 v10-re | 0.9172 | **+0.0053** [+0.0017, +0.0090], 1.00 |

  - A B0 cseréjénél a v10-es CNT12 a két v8 seed átlagánál is jobb: +0.0026 [+0.0001, +0.0053], P>0 = 0.98.
- **Értelmezés:**
  - Egy seeden +0.008, ensemble-ben +0.003 a v8-hoz képest, és a CI alsó határa ensemble-ben pozitív. A <0.008-as szabály szerint ez is csak LB-próba után számít.
  - Három változás keveredik (radiológus-címke a 184-en, laterality-javítás, maszk), ezeket ez a run nem választja szét. A Lateral Meniscus és a MCL javulása illik a laterality-javításhoz; a Synovitis és az Effusion a radiológus-címkékhez.
  - A 114 kicsi (A50-rész 22 study), a seedzaj a 208-on ~0.003 volt.
- **Javasolt következő lépés (akkor):** CNT12 v10 s2, utána V2S/C0 v10-en, majd LB-próba. A s2 és a R50 lefutott, lásd lent: a s42 előnye seedszerencse volt.

**v10 végeredmény (2026-10-14): CNT12 s2 + R50. A train_v10 a 114-en nem javít; a s42 +0.008-a seedszerencse volt.**
- **Runok** (mind Windows; a config a v8-as párjával azonos, csak az adatbeállítások térnek el):

| Run | best | 220 | ep6–10 |
|---|---|---|---|
| `CNT12_lr4e-5_E3_img320_trainv10_holdout220_20261013_180539` (s42) | ep11 | 0.9005 | 0.8946 |
| `CNT12_lr4e-5_E3_img320_trainv10_s2_holdout220_20261013_215639` | ep9 | 0.8916 | 0.8877 |
| `R50_E3_img320_trainv10_holdout220_20261014_062316` | ep8 | 0.8825 | 0.8809 |

- **A 114-en** (tiszta rész; szkript: scratchpad `v10_check2.py`):

| Modell | v10 | v8 | v10 − v8 [95% CI], P>0 |
|---|---|---|---|
| CNT12 s42 | 0.9102 | 0.9022 | +0.0080 [−0.0014, +0.0173], 0.95 |
| CNT12 s2 | 0.8955 | 0.9021 | −0.0066 [−0.0157, +0.0029], 0.09 |
| CNT12 két seed átlaga | 0.9077 | 0.9061 | **+0.0016** [−0.0054, +0.0084], 0.64 |
| R50 | 0.8845 | 0.8874 | **−0.0029** [−0.0106, +0.0050], 0.23 |

- **Ensemble a 114-en, azonos tagokkal v10 vs v8:**
  - B0 → CNT12-átlag cseréje (R50, V2S, C0 v8): v10 0.9152, v8 0.9146, **+0.0006** [−0.0016, +0.0028];
  - ugyanez R50 v10-zel: **−0.0002** [−0.0033, +0.0029];
  - a H4-hez képest a B0 → CNT12 csere v8-as CNT12-kkel is +0.0027. A nyereség tehát a CNT12-ből jön, nem a v10-ből.
- **Per target (114, v10 − v8),** mindkét modellben azonos irányban: ACL −0.014 / −0.028, Lateral OA −0.018 / −0.011. A CNT12-ben Effusion +0.027, Synovitis +0.041, Fracture −0.026, a R50-ben ezek ±0.005 körül. A 114-en a ritkább targetek 15–40 pozitívval mérhetők, a per-target különbség zaj-szintű.
- **Seedzaj a 220-on:** CNT12 s42 − s2 = **+0.0089** [+0.0014, +0.0167], P>0 = 0.99. A két seed predikcióinak Spearman-korrelációja 0.935.
  - Ez a modell szórása, nem a mérésé, ezért a kiegyensúlyozott 220 sem teszi megbízhatóvá az egyseedes döntést.
  - Az első s42-es összevetés (+0.008) ennek a mintapéldája.
- **Csak v10-modellek a 220-on:** CNT12-átlag 0.9012; + R50 (1:1) 0.9032 (+0.0020 [−0.0027, +0.0072]); három tag (s42, s2, R50 egyenlő súllyal) 0.9051.
- **Értelmezés:**
  - A train_v10 a radiológus-címkéket a 184 studyn (a tanítóhalmaz 4.4%-án), a laterality-javítást és a 8 maszkot együtt hozta. Ez a 114-en mérhető nyereséget nem adott (CNT12 +0.0016, R50 −0.003, ensemble ±0.0006).
  - Ez nem jelenti, hogy a v10 rosszabb. A változás kicsi a tanítóhalmazhoz képest, és a hatása a seedzaj alatt van.
  - A v10 fő haszna a mérés: a 220 a ritka targeteken pontosabb, mint a 208.
- **Döntés / javaslat:**
  - A V2S-t és a C0-t címkenyereség miatt nem érdemes v10-en újratanítani. Ha a további kísérleteket a 220-on mérjük, akkor bázisként kellenek, de ez mérési, nem javítási ok.
  - Az LB-próba a B0 → CNT12 cseréjére v8-as és v10-es CNT12-kkel ugyanazt várja; ennek LB-hatását korábban ~0.0005–0.001-re becsültük.
  - Új kísérletnél egy seed nem elég, a 220-on sem: legalább két seed átlaga, vagy ensemble-szintű döntés.

**C0 v10 (2026-10-14, Linux DDP 2 GPU, seed 2): ugyanaz a kép.** A V2S v10 még fut.
- **Run:** `C0_E3_img320_trainv10_s2_holdout220_20261013_230729`; best ep9, **0.8995** a 220-on, ep6–10 átlag 0.8948. A v8-as C0 s2 egy GPU-n futott, a pipeline szerint a DDP ugyanaz a kísérlet (8 study / lépés).
- **A 114-en:** v10 0.9043, v8 0.9025, **+0.0018** [−0.0081, +0.0119], P>0 = 0.62. Szkript: scratchpad `v10_check3.py`.
- **Ensemble a 114-en:**
  - a H4-ben csak a C0-t cserélve v10-re: −0.0007 [−0.0043, +0.0031];
  - CNT12-átlag + R50 + V2S + C0 mindhárom v10-es taggal, a v8-as változathoz képest: −0.0012 [−0.0057, +0.0032].
- **Per target (114, v10 − v8), CNT12 / R50 / C0:**
  - mindhárom modellben romlott: ACL −0.014 / −0.028 / −0.030 és Lateral OA −0.018 / −0.011 / −0.030;
  - javult: Effusion +0.027 / +0.004 / +0.028; Synovitis (CNT12 +0.041, C0 +0.031).
  - Lehetséges ok: a 184 radiológus-sor a ref220 válogatása miatt majdnem csupa negatív, és a 90 v8-ban is szereplő studyn a csere szinte csak pozitív → negatív (Effusion 39, ACL 18, Lateral OA 17). Részletek: [problemas_esetek.md](problemas_esetek.md) B8.
- **Csak v10-modellek a 220-on:**

| Ensemble | 220 | 114 | 106 új |
|---|---|---|---|
| CNT12-átlag | 0.9012 | 0.9077 | 0.8911 |
| C0 s2 | 0.8995 | 0.9043 | 0.8930 |
| CNT12-átlag + C0 (1:1) | **0.9060** | 0.9129 | 0.8971 |
| CNT12-átlag + C0 + R50 (1/3) | 0.9064 | 0.9109 | 0.9005 |

  - CNT12-átlag + C0 a CNT12-átlaghoz képest: **+0.0048** [+0.0011, +0.0090], P>0 = 0.99. A C0 és a CNT12-átlag Spearman-korrelációja 0.925.
  - A R50 hozzáadása: +0.0003 [−0.0025, +0.0034]. A v10-es ensemble-ben a R50 nem ad hozzá.
- **Következő (akkor):** a V2S v10, utána a teljes v10-es ensemble a 220-on.

**V2S v10 és a teljes v10-es ensemble (2026-10-14): a train_v10 négy architektúrán is semleges, a v8 ↔ v10 kérdés lezárva.**
- **Run:** `V2S_E3_img320_trainv10_s2_holdout220_20261014_090914` (Linux DDP 2 GPU, seed 2; a v8-as V2S seed 42-es volt, és ott be volt kapcsolva a grad checkpointing). Best ep9, **0.9013** a 220-on; ep6–10 átlag **0.9004**, a v10-es runok közül a legsimább görbe.
- **v10 − v8 a 114-en, egyedi modellek:** CNT12-átlag +0.0016, R50 −0.0029, C0 +0.0018, V2S **+0.0005** [−0.0062, +0.0080]. Szkript: scratchpad `v10_check4.py`.
- **Per target:** a V2S-ben az ACL (+0.013) és a Lateral OA (+0.005) nem romlott. A B8-ban leírt mintázat így csak 3 modellből 3-ban áll fenn, 4-ből nem; a négy modell átlaga ACL −0.015, Lateral OA −0.014, Effusion +0.016, Synovitis +0.017.
- **Azonos tagú ensemble-k a 114-en, v10 vs v8:**

| Tagok | v10 | v8 | Δ [95% CI], P>0 |
|---|---|---|---|
| CNT12-átlag, V2S, C0 | 0.9154 | 0.9132 | +0.0022 [−0.0035, +0.0078], 0.79 |
| CNT12-átlag, R50, V2S, C0 | 0.9142 | 0.9146 | −0.0004 [−0.0059, +0.0046], 0.44 |
| V2S, C0 | 0.9103 | 0.9106 | −0.0003 [−0.0073, +0.0070], 0.47 |

  - A H4 v8 (B0, R50, V2S, C0 s2; LB 0.943) a 114-en 0.9119. A v10-es CNT12 + V2S + C0 ehhez képest +0.0034 [−0.0027, +0.0098]; ennek nagy része az összetételből jön (ugyanez v8-cal +0.0013).
- **Csak v10-modellek a 220-on:**

| Ensemble | 220 | 114 | 106 új |
|---|---|---|---|
| CNT12-átlag + V2S + C0 (1/3) | **0.9101** | 0.9154 | 0.9033 |
| + R50 (1/4) | 0.9096 | 0.9142 | 0.9034 |
| V2S + C0 | 0.9068 | 0.9103 | 0.9027 |
| CNT12-átlag + C0 | 0.9060 | 0.9129 | 0.8971 |

  - A V2S hozzáadása a CNT12 + C0-hoz: **+0.0040** [+0.0014, +0.0070]. A R50 hozzáadása: −0.0005 [−0.0029, +0.0018].
  - A legjobb v10-es összetétel a CNT12-átlaghoz képest +0.0089 [+0.0043, +0.0139].
  - Spearman a 220-on: V2S–CNT12 0.909, V2S–C0 0.908, C0–CNT12 0.925, V2S–R50 0.858. A R50 a legdiverzebb, mégsem ad hozzá, mert egyedül gyenge (0.8825).
- **Döntés:**
  - A train_v10 a train_v8-hoz képest se nem jobb, se nem rosszabb (egyedi modellek: −0.003 … +0.002; ensemble: −0.0004 … +0.0022). A v8 ↔ v10 kérdés lezárva.
  - Ensemble-összetételre a 220 és a 114 egyet mond: CNT12 (két seed) + V2S + C0; a B0 és a R50 kimarad.
  - LB-jelölt: CNT12-átlag + V2S + C0, v10-es vagy v8-as checkpointokkal (a kettő ugyanazt várja). A H4-hez képest lokálisan +0.001–0.003, ez a <0.008-as szabály alatt van, tehát csak LB dönt.
  - **Beküldési notebook kész (2026-10-14):** `python notebooks/build_04_notebook.py --ensemble holdout4_v10_cnt12`. A négy v10-es checkpoint (CNT12 s42, CNT12 s2, V2S s2, C0 s2) egyenlő súllyal megy, így a CNT12 súlya 1/2. A 220-on ez 0.9094, az 1/3-os változat 0.9101, a különbség elhanyagolható. A tiszta 114-en a holdout4_v8_c0-hoz képest +0.0036.
    - Kaggle-datasetek: `knee-cnt12-holdout-v10`, `knee-cnt12s2-holdout-v10`, `knee-v2s-holdout-v10`, `knee-c0-holdout-v10`. Forrás: `work/export_v10/runs/<run>/` (karcsúsított best.pt, 102–124 MB, bitre azonos súlyok) + config.yaml.
    - A `srcknee2` datasetben a laterality-javítás (2396ed0) és a b92f900 utáni kód kell, mert a v10-modellek ezzel tanultak.
    - Lokális próbafuttatás a 3 publikus teszt-studyn rendben: 4 checkpoint, egyező prep-hash (9b43435e7ac6), a C0 `inception`, a többi `mri_scalar` normalizálással, ahogy tanultak.
  - A holdout404-re váltás (a 4003-on tanítva) a következő kísérletek mérési alapja lehet, lásd a beszélgetésben felvetett indoklást: kb. 12%-kal szűkebb CI, nincs B8-torzítás, a v8-bázis a teljes 208-on összevethető.

### 10. Inferencia-pontosság a Kaggle T4-en: az fp16 rendben van (2026-10-14)

- **Kérdés:** a Kaggle T4 (sm_75) nem tud natív bf16-ot, ezért a notebook `AMP = "auto"` beállítása ott **fp16** autocastot ad, a modellek viszont bf16 autocasttal tanultak (3090, 5080). Kell-e emiatt fp32-vel tanítani?
- **Válasz: nem.** A checkpoint mindig fp32 súlyokat tárol (az autocast csak a számítást érinti), így a tanítási pontosság nem dönti el az inferenciáét. Az fp16 egyetlen valódi kockázata a túlcsordulás (|x| > 65504), ezen az fp32 tanítás nem segítene.
- **Mérés:** CNT12 v10 s42 `best.pt`, ref220, ugyanaz a checkpoint fp32 / fp16 / bf16 inferenciával (RTX 5080), forward hookokkal minden levélmodul legnagyobb |kimenetére|. Szkript: `notebooks/amp_check.py`.

| amp | hard macro AUC | nem véges | max \|Δ\| az fp32-höz | átlag \|Δ\| | max \|aktiváció\| | csúcsmemória |
|---|---|---|---|---|---|---|
| fp32 | 0.9006 | 0 | – | – | 937 | 5.42 GB |
| fp16 | 0.9006 | 0 | 4.7e-3 | 1.5e-4 | 937 | 3.45 GB |
| bf16 | 0.9005 | 0 | 2.5e-2 | 9.9e-4 | 942 | 3.45 GB |

- A bf16 menet a tanításkor mentett predikciókat 6e-8-ra reprodukálja (a mérés helyes). Célpontonként az fp16 és az fp32 AUC-ja legfeljebb 0.0002-vel tér el.
- Az fp16 **közelebb van az fp32-höz, mint a bf16** (hosszabb mantissza): a validáláskor használt bf16-hoz képest a Kaggle fp16-os inferenciája inkább pontosabb.
- **Tartalék a túlcsordulásig:** a legnagyobb aktiváció a residual streamben van (stage 2, block 8, ~940; a residual összeg ≤ ~1530), ez ~40–70× alatta marad a 65504-es határnak.
- **Döntés:** fp32 tanítás nem kell, a notebook `AMP = "auto"` marad. Ha egy új encodernél kétség merül fel, ugyanezzel a szkripttel mérhető.

### 11. Tanítás 48 slice-centrummal (C0, train_v10, 2026-10-14): nincs nyereség, lezárva

- **Kérdés:** az inferenciaoldali c48 nem hozott semmit (LB 0.942 = c24, lásd `doc/slice_coverage_check.md`). Segít-e, ha a modell már a tanításnál is 48 centrummal (144 triplet / study) látja a sorozatokat?
- **Runok** (C0 E3 img320, train_v10, holdout220, `data.centers_per_series=48`, grad checkpointing, mindegyik a saját beállításával validál):
  - s2, Linux DDP 2×3090: `C0_E3_img320_c48_trainv10_s2_holdout220_20261014_143747`, best ep10 **0.8959**, 1086 s/epoch (a c24 párja 719 s), 7.7 GB/GPU;
  - s42, Windows 5080: `C0_E3_img320_c48_trainv10_holdout220_20261014_144235`, best ep11 **0.8948**, 1634 s/epoch, 7.6 GB. A 8. epochnál host-oldali `MemoryError` (problemas_esetek C), resume-mal folytatva; a 15. epoch közben kézzel leállítva (3 epoch javulás nélkül, csökkenő görbe).
  - Bázis: `C0_E3_img320_trainv10_s2_holdout220_20261013_230729` (c24 s2), 0.8995.
- **Eredmény** (hard macro AUC, páros bootstrap; szkript: scratchpad `c48_check.py`):

| | 220 | 114 | 106 új |
|---|---|---|---|
| C0 c24 s2 | 0.8995 | 0.9043 | 0.8930 |
| C0 c48 s2 | 0.8959 | 0.9024 | 0.8874 |
| C0 c48 s42 | 0.8948 | 0.8981 | 0.8897 |
| C0 c48 2-seed átlag | 0.9001 | 0.9051 | 0.8920 |

  - c48 s2 − c24 s2 (azonos seed): **−0.0036** [−0.0099, +0.0021] a 220-on, −0.0019 a 114-en. A c48 s42 is a c24 s2 alatt van (−0.0047).
  - A 2-seed c48 átlag +0.0006 a c24 s2-höz képest, de ez két modell átlaga egy ellen; a seed-ensemble önmagában is ennyit hoz.
  - Görbe ep6–10: c24 0.8948, c48 s2 0.8930, c48 s42 0.8916.
  - Ensemble (CNT12-átlag + V2S + C0, 1/3-onként, 0.9101): a C0 cseréje c48 s2-re −0.0016, c48 s42-re −0.0001, a c48 átlagra −0.0007; a c48 átlag negyedik tagként −0.0009. Egyik sem pozitív.
- **Értelmezés:** a c24 binenként véletlen centrummal tanít, így epochokon át már most is a teljes térfogatot látja; a sűrűbb mintavétel csak redundáns, szomszédos tripleteket ad a pooling elé. Ehhez ~1.5× (Linux) – 2.3× (Windows, checkpointing) epochidő jár.
- **Döntés:** a slice-lefedettség iránya (inferencia és tanítás) **lezárva**, `centers_per_series: 24` marad.

### 12. Augmentáció kikapcsolva (C0 és CNT12, train_v8, holdout208, 2026-10-15): nem jobb, az augmentáció marad
- **Kérdés:** mennyit ér a klasszikus augmentáció (±10° forgatás, ±5% eltolás, 0.9–1.1 skála, gain/bias/gamma, σ=0.01 zaj)? `--set augment.enabled=false`. A véletlen slice-centrum mintavétel ezzel nem kapcsol ki.
- **Runok** (seed 2, Linux 2×3090 DDP):
  - `C0_E3_img320_noaug_trainv8_s2_holdout208_20261014_234918`, best ep10, **0.9335**. Pár: `C0_E3_img320_trainv8_s2_holdout208_20261010_153116`, best ep13, 0.9358.
  - `CNT12_lr4e-5_noaug_trainv8_s2_holdout208_20261015_042831`, best ep5, **0.9251**, ep10-nél early stop, közben resume ep9-nél. Pár: `CNT12_lr4e-5_E3_img320_trainv8_s2_holdout208_20261012_122058`, best ep9, 0.9314.
  - **Zavaró tényezők** (mindegyik kisebb a seed-szórásnál): a noaug runok az újabb kóddal futottak, `label_mask_csv`-vel (8 study minden targetje súly 0) és a knee_consistency/non_knee kulcsokkal. A C0 párja 1 GPU-s, grad checkpointing nélkül futott, a noaug DDP-vel és checkpointinggel.
- **Eredmény** (hard macro AUC, best epoch; szkript: scratchpad `noaug_check.py`):

| | 208 | new-100 | 58 | A50 |
|---|---|---|---|---|
| C0 s2 aug | 0.9358 | 0.9412 | 0.9091 | 0.9376 |
| C0 s2 noaug | 0.9335 | 0.9420 | 0.9000 | 0.9144 |
| CNT12 s2 aug | 0.9314 | 0.9428 | 0.9000 | 0.9091 |
| CNT12 s2 noaug | 0.9251 | 0.9366 | 0.8898 | 0.9141 |

  - noaug − aug, páros bootstrap (208): C0 **−0.0023** [−0.0097, +0.0048], P>0 = 0.29; CNT12 **−0.0063** [−0.0148, +0.0015], P>0 = 0.06. Összevetésül a seed-zaj: C0 s2 − s42 +0.0044, CNT12 s2 − s42 −0.0033.
  - Mindkét architektúrán az MCL esik a legtöbbet (C0 −0.016, CNT12 −0.044). A CNT12-nél az ACL, a Lateral Meniscus, a Medial OA és a PF OA is −0.012…−0.019. Az Effusion, a Synovitis, a Contusion és a Fracture enyhén javul.
  - Gyorsabb túltanulás augmentáció nélkül. A train loss ugyanabban az epochban alacsonyabb (C0 ep10: 0.336 vs 0.378; CNT12 ep5: 0.403 vs 0.424), és a csúcs korábban jön (C0 ep10 vs ep13, CNT12 ep5 vs ep9). A görbe ep6–10 átlaga: C0 0.9316 vs 0.9301 (a C0 a görbe közepén nem rosszabb, csak a csúcsa alacsonyabb), CNT12 0.9241 vs 0.9303.
  - Diverzitás: a Spearman aug ~ noaug 0.885 (C0) és 0.902 (CNT12), ugyanaz a nagyságrend, mint két seed között (0.840 és 0.919).
  - Ensemble (H5 = R50 + V2S + C0 s2 + CNT12 s42 + CNT12 s2): ha a két s2 tagot noaug-ra cseréljük, **−0.0009** [−0.0033, +0.0015]; ha a két noaug-ot hetedik tagként hozzáadjuk, **+0.0000**.
- **Értelmezés:** az augmentáció kis, de konzisztens regularizáció. Nélküle egyik architektúra sem jobb. A CNT12-n majdnem szignifikáns a romlás (P>0 = 0.06), és a túltanulás korábban kezdődik. A noaug modell diverzitást sem ad az ensemble-hez.
- **Döntés:** `augment.enabled: true` marad, a noaug iránya **lezárva**. Ha az augmentációval még foglalkozunk, inkább erősítés jöhet szóba (pl. nagyobb skála/forgatás, bias-field, lásd 5.), mint gyengítés.

### 13. Új elsődleges validáció: holdout450 + train_v11 (2026-10-15)
- **Forrás:** `F:\Kaggle\data\rsna_reference_450_package.zip` (README a készítés módjával).
  450 = az eredeti 208 (bájtra azonos, a 3 átfedő study 6 eltérése a 208 javára döntve) +
  242 új (96 YC hat szekvenciás, 100 Andrew második véletlen 100, 46 Zoli patológiadús).
  Csak a Review status = 1 Reading sorok; 0/1 → 0, 3/4 → 1, 2 / X / üres → maszk.
- **train_v11** = train_v8 a 450 nélkül (értékek változatlanok): 4199 − 242 = **3957 study**.
  Script: `notebooks/build_ref450_train_v11.py` (a zipből olvas, ellenőriz, a 450/242/manifest
  fájlokat bájtra másolja `F:/Kaggle/data`-ba és `notebooks/`-ba).
- **Alapértékek:** `run_holdout.*` `HOLDOUT=450`, `TRAIN_CSV=train_v11.csv`; minden config
  `train_csv` = train_v11, `exclude_from_training_csv` = 450, új
  `paths.reference_subsets_csv: notebooks/reference_450_manifest.csv`. Teacher CV:
  `cv3_train_v11` (automatikusan készül).
- **Riport:** minden run végén (és `cli report <run>`-nal régiekre) `subset_summary.csv` +
  `subset_metrics_per_class.csv` + `report.html` táblázat: full_450 (közvetlenül), original_208,
  new_242; macro soft/ROC-AUC, 12 per-target AUC, pos/neg. A checkpoint-választás a full_450-en.
- **Maszk:** az A3 (90°-ban forgatott) study a new_242-ben van → 449 értékelhető study.
- **Teendő:** friss baseline-ok train_v11-en (a régi train_v8-as checkpointok látták a 242-t;
  velük csak az original_208 vethető össze). Linuxon a `train_v11.csv` és a
  `train_labeled_450_reference.csv` a DATA_ROOT-ba másolandó. Régi train_v8 runok folytatása:
  `HOLDOUT=208 TRAIN_CSV=…/train_v8.csv`, CV-nél
  `--set paths.exclude_from_training_csv=notebooks/train_labeled_208_reference.csv`.
- **Első v11 runok (C0 E3 img320, mindkettő seed 2!)**, 2026-10-15. A Linuxos (3090, grad
  checkpointing nélkül) és a Windowsos (5080, checkpointinggal) ugyanaz a kísérlet, nem két seed.
  A Windowsos az összevetéskor még futott: ep11 kész, best ep10. Szkript: scratchpad `compare_v11.py`.

| run (best ep) | full_450 | original_208 | new_242 |
|---|---|---|---|
| `C0_E3_img320_trainv11_holdout450_20261015_linux` (ep10) | 0.9198 | 0.9296 | 0.9130 |
| `C0_E3_img320_trainv11_s2_holdout450_20261015_170342` (ep10, futás közben) | 0.9210 | 0.9297 | 0.9151 |
| a kettő átlaga | 0.9244 | 0.9341 | 0.9176 |
| v8 C0 s2 / s42 (holdout208, a 208-on választva) | – | 0.9358 / 0.9314 | – |

  - Win − Linux: full +0.0012 [−0.0037, +0.0056], a 208-on +0.0001. A per-target Spearman
    0.946, vagyis a hardveres nemdeterminizmus magában is seed-szerű zajt ad (két valódi seed
    a v8-on: 0.840). Targetenként nagy a szórás (Baker's 0.954 vs 0.921, PF OA 0.853 vs 0.882).
  - A new_242 nehezebb, mint a 208 (−0.015…−0.017), főleg a Medial OA (0.87 vs 0.93–0.95),
    a PF OA, a Lateral OA és a Baker's miatt.
  - v11 − v8 az original_208-on: egy modell −0.006 [−0.013, +0.001] (P>0 = 0.04), az átlagok
    −0.0036 [−0.0078, +0.0005]. Ez nem tiszta összevetés: a v8 242 studyval többön tanult
    (4199 vs 3957), és a best.pt-t magán a 208-on választotta (optimista), a v11 a 450-en.
  - **Értelmezés:** a v11 C0 az új baseline. A 208-on mért kis visszaesés a seed-zaj határán
    van, és két ismert torzítás is magyarázza. A következő lépés: egy valódi második seed
    (pl. s42) a v11-en, és a többi architektúra v11-es baseline-ja.
- **CNT12 v11 (lr 4e-5, in12k, drop path 0.1, `mri_scalar`), két seed, Windows, 2026-10-16:**

| run (best ep) | full_450 | original_208 | new_242 |
|---|---|---|---|
| `CNT12_lr4e-5_E3_img320_s42_trainv11_holdout450_20261015_222807` (ep10) | 0.9185 | 0.9329 | 0.9039 |
| `CNT12_lr4e-5_E3_img320_s2_trainv11_holdout450_20261016_012610` (ep10, early stop ep13) | 0.9188 | 0.9318 | 0.9076 |

  - A full_450-en a C0 v11 szintje (0.920–0.921). A 208-on kicsit jobb (v8 CNT12 s2/s42: 0.9314 / 0.9347), a new_242-n gyengébb a C0-nál.
- **C0 v11 s42 (Linux, best ep11): 0.9192 / 0.9286 / 0.9122** (full / 208 / 242). Összevetés mind az 5 v11 runnal
  (scratchpad `compare_v11b.py`, páros study-bootstrap 1000×):

| | full_450 | original_208 | new_242 |
|---|---|---|---|
| C0 átlag (s2 Linux + s42) | 0.9235 | 0.9336 | 0.9157 |
| C0 átlag3 (+ s2 Windows) | 0.9251 | 0.9347 | 0.9183 |
| CNT12 átlag (s2 + s42) | 0.9224 | 0.9354 | 0.9098 |
| **C0-átlag + CNT12-átlag (1:1)** | **0.9273** | **0.9386** | 0.9175 |
| C0-átlag3 + CNT12-átlag (1:1) | 0.9280 | 0.9390 | 0.9188 |
| mind az 5 run egyenlő súllyal | 0.9281 | 0.9386 | 0.9192 |

  - Seed-zaj kicsi: C0 s2 − s42 +0.0007, CNT12 s2 − s42 +0.0003 (full). Spearman: C0 s2 Linux ~ Windows 0.946,
    C0 s2 ~ s42 0.934, CNT12 s2 ~ s42 0.931 → a Windowsos „ugyanaz a seed" gyakorlatilag harmadik seed.
  - Két seed átlaga az egyedihez képest: C0 +0.0036 [+0.0011, +0.0067].
  - C0 vs CNT12 (átlagok): full +0.0011, 208 −0.0018, 242 +0.0059 [−0.0027, +0.0157]; a CNT12 a new_242-n gyengébb.
  - **C0 + CNT12** a C0-átlaghoz képest +0.0038 [+0.0014, +0.0063], a CNT12-átlaghoz +0.0049 [+0.0021, +0.0081]
    (full); Spearman a két architektúra között 0.923. Ez a legjobb v11-es összetétel; a harmadik C0 (Windows s2)
    még +0.0007 [+0.0001, +0.0014].
  - A new_242 mindenhol ~0.02-vel nehezebb, főleg Medial OA (0.88–0.89 vs 0.95), Contusion, Baker's, PF OA.
  - Javaslat: V2S v11 (két seed) — a v10-en V2S + C0 + CNT12 volt a legjobb; utána LB-jelölt v11-es checkpointokkal.
- **v11 vs v8 párok az original_208-on** (a v8 runok csak ezt prediktálták; scratchpad `compare_v11_v8.py`, 2000×):

| | v11 | v8 | v11 − v8 [95% CI], P>0 |
|---|---|---|---|
| C0 s2 / s42 | 0.9296 / 0.9286 | 0.9358 / 0.9314 | −0.0062 (0.04) / −0.0029 (0.19) |
| CNT12 s2 / s42 | 0.9318 / 0.9329 | 0.9314 / 0.9347 | +0.0004 (0.55) / −0.0018 (0.28) |
| C0-átlag | 0.9336 | 0.9376 | −0.0040 [−0.0086, +0.0001], 0.03 |
| CNT12-átlag | 0.9354 | 0.9360 | −0.0006 [−0.0045, +0.0039], 0.40 |
| C0 + CNT12 | 0.9386 | 0.9414 | −0.0029 [−0.0059, +0.0001], 0.03 |

  - Per target (C0 + CNT12): Lateral OA −0.012 (mindkét architektúrában), MCL −0.012 (csak C0: −0.022), PF OA −0.008;
    Synovitis +0.011, Effusion +0.005. A v11 ~ v8 pár Spearman 0.925–0.944, a seed-párok szintje.
  - **Értelmezés:** a v11 a 208-on kb. 0.003-mal gyengébb, a CI éppen súrolja a 0-t. Két ismert, a v8 felé húzó
    torzítás: (1) a v8 best.pt-jét a 208-on választottuk (optimista), a v11-ét a 450-en; (2) a v8 242 studyval
    (5.8%) többön tanult, köztük a patológiadús Zoli-46-on — ez a ritka pozitívú targeteknél (Lateral OA, MCL)
    látszik. A kettő együtt bőven kiadhatja a −0.003-at; tiszta (torzítás nélküli) v8-mérés a 450-en nincs.
  - **Döntés-javaslat:** a v11 marad az alap (csak így tiszta a 450-es validáció). Hogy a 242 study elvesztése
    LB-n mennyit ér, azt csak beküldés mutatja meg (v11 C0+CNT12 vs ugyanez v8-cal); <0.008 lokálisan nem dönt.
- **V2S v11, két seed (2026-10-16):** `V2S_E3_img320_s42_trainv11_holdout450_20261016_062318` (ep8) és
  `V2S_E3_img320_s2_trainv11_holdout450_20261016_091902` (ep8). Scratchpad `compare_v11_all.py`.

| | full_450 | original_208 | new_242 |
|---|---|---|---|
| V2S s42 / s2 | 0.9249 / 0.9248 | 0.9342 / 0.9331 | 0.9159 / 0.9172 |
| V2S-átlag | **0.9282** | 0.9366 | 0.9204 |
| C0-átlag / CNT12-átlag | 0.9235 / 0.9224 | 0.9336 / 0.9354 | 0.9157 / 0.9098 |
| C0 + CNT12 | 0.9273 | 0.9386 | 0.9175 |
| V2S + C0 | 0.9294 | 0.9386 | 0.9220 |
| V2S + CNT12 | 0.9303 | 0.9413 | 0.9208 |
| **V2S + C0 + CNT12 (seed-átlagok, 1/3)** | **0.9303** | 0.9403 | 0.9215 |
| mind a 7 run egyenlő súllyal | 0.9303 | 0.9402 | 0.9220 |

  - A V2S a legjobb egyedi architektúra a v11-en: V2S-átlag − C0-átlag +0.0047 [+0.0005, +0.0086], − CNT12-átlag
    +0.0058 [+0.0004, +0.0115]; a new_242-n a CNT12-nél +0.0106. Seed-különbség −0.0001.
  - V2S + C0 + CNT12 a C0 + CNT12-höz +0.0030 [+0.0015, +0.0045]; a V2S-átlaghoz +0.0021 [−0.0010, +0.0050].
    A háromból bármelyik pár közel van (V2S + CNT12 −0.0000, V2S + C0 −0.0009); a 7 egyenlő súly sem jobb.
  - **v8-párral (208):** V2S s42 +0.0010 [−0.0047, +0.0072] (a V2S nem veszít, a C0 −0.004, a CNT12 −0.0006);
    V2S + C0 + CNT12 (V2S s42 mindkét oldalon) −0.0015 [−0.0047, +0.0017], P>0 = 0.17 — a C0+CNT12-nél látott
    −0.0029 a V2S-sel a felére csökken. v8-on csak egy V2S holdout208 run van (s42).
  - LB-jelölt: V2S-átlag + C0-átlag + CNT12-átlag (6 checkpoint, 1/3 súly architektúránként), v11-es checkpointokkal.
  - **Beküldési notebook kész (2026-10-16):** `python notebooks/build_04_notebook.py --ensemble holdout6_v11`
    (egyenlő súly checkpointonként = 1/3 architektúránként; C0-ként a két Linuxos run, s42 és s2). Lokális próba a
    3 publikus teszt studyn rendben (6 checkpoint, prep hash 9b43435e7ac6, 0.5 perc). Tartalék: a három s42 modell,
    a holdout450-en 0.9293 (−0.0010). Kaggle datasetek: `knee-{v2s,v2ss2,c0,c0s2,cnt12,cnt12s2}-holdout-v11`.
  - **Feltöltésre: `work/export_v11/`** (`python src/tools/export_checkpoints.py --ensemble holdout6_v11 --out
    work/export_v11`): datasetenként egy mappa, karcsúsított `best.pt` (EMA-súlyok + metaadat, optimizer és
    nyers súlyok nélkül) + `config.yaml`; 2630 MB → 658 MB. Inferencia az exportból = a run-mappás (max. 1e-6).
    Minden beküldés előtt így.
  - **LB 2026-10-16: holdout6_v11 = 0.941** (holdout4_v8_c0: 0.943). A 208 ezt előre jelezte: a v11-es hatos ott
    0.9403, a H4 0.9417, −0.0015 [−0.0049, +0.0019]; ugyanez az összetétel v8-as modellekből 0.9418 (= H4), tehát nem
    az összetétel, hanem a v11-es modellek (−242 tanító study, főleg Medial/Lateral OA, Fracture). Tanulság: LB-jelölt
    előtt a v8-as referenciához (208) is mérni kell, nem csak a v11-en belül. Ellenőrizendő: a Kaggle-ön csatolt
    `srcknee2` verziója (az új sorozatválasztás nélkül eltérne a v11 tanításától). Szkript: scratchpad `lb_check_v11.py`.
  - **R50 v11, két seed (2026-10-16):** `R50_E3_img320_s42_trainv11_holdout450_20261016_123429` (ep7, 0.9079) és
    `..._s2_..._123439` (ep8, 0.9061); átlag 0.9084 / 0.9182 / 0.9006 (full / 208 / 242), a V2S-átlagnál −0.020
    [−0.028, −0.012]. Spearman a többi taggal 0.87 (a legdiverzebb), mégsem ad hozzá: V2S + C0 + CNT12 + R50 0.9302
    (−0.0001 a beküldött hármashoz). **v8-párral (208): R50 −0.0059 [−0.0108, −0.0011]** — a v11-en a R50 veszít a
    legtöbbet (C0 −0.004, CNT12 −0.0006, V2S +0.001); MCL −0.034, Lateral OA −0.029, ACL −0.016. Négyes összetétel a
    208-on: v8 0.9426, v11 0.9405, −0.0020 [−0.0051, +0.0010]; a H4 B0 nélkül: −0.0025. Minden azonos összetételű
    párban ~0.002-vel gyengébb a v11 → a 242 tanító study kiesése a valószínű ok. Szkript: scratchpad `compare_v11_r50.py`.
  - **Epochonkénti részhalmaz-mérés (2026-10-16):** új runoknál a `history.csv`-ben `val_original_208_macro_soft_auc` /
    `val_new_242_macro_soft_auc`, és `epoch_predictions/validation_predictions_ep<NN>.csv` minden epochról — ebből
    látszik, melyik epochot választaná a 208 és melyiket a 450. A meglévő runokra visszamenőleg nem érhető el.
- **Új seedek (s7, s123) v11-en: V2S, CNT12, CNT22 (Linux, 2026-10-16/17)**, összevetve a v8-as párjaikkal.
  CNT12 = in12k, enc lr 4e-5, `mri_scalar`; CNT22 = fb22k, enc lr 1e-4, `imagenet` (mint a v8 CNT22/CNT22_DDPcheck);
  early stopping patience 3 (v8: 5), a best epoch mindig a stop előtt van. Scratchpad `compare_v11_v8_seeds.py`
  (páros study-bootstrap 2000×).

| run (best ep) | full_450 | original_208 | new_242 |
|---|---|---|---|
| V2S s7 (ep9) / s123 (ep8) | 0.9232 / 0.9193 | 0.9319 / 0.9309 | 0.9174 / 0.9111 |
| CNT12 s7 (ep10) / s123 (ep8) | 0.9197 / 0.9211 | 0.9284 / 0.9318 | 0.9112 / 0.9128 |
| CNT22 s7 (ep10) / s123 (ep8) | 0.9216 / 0.9199 | 0.9319 / 0.9276 | 0.9133 / 0.9151 |
| V2S 4 seed átlag | 0.9285 | 0.9374 | 0.9214 |
| CNT12 4 seed átlag | 0.9251 | 0.9367 | 0.9146 |
| CNT22 2 seed átlag | 0.9233 | 0.9326 | 0.9162 |
| H6_v11 (beküldött, LB 0.941) | 0.9303 | 0.9403 | 0.9215 |
| V2S4 + C0 + CNT12_4 + CNT22 (archonként 1/4) | 0.9308 | 0.9413 | 0.9230 |

  - **v11 vs v8 a 208-on:**

| | v11 | v8 | v11 − v8 [95% CI], P>0 |
|---|---|---|---|
| V2S s7 / s123 (vs v8 s42, egyetlen v8 V2S) | 0.9319 / 0.9309 | 0.9332 | −0.0013 (0.34) / −0.0023 (0.20) |
| CNT12 s7+s123 átlag vs v8 s42+s2 átlag | 0.9347 | 0.9360 | −0.0013 [−0.0055, +0.0034], 0.28 |
| CNT12 4 seed vs v8 2 seed | 0.9367 | 0.9360 | +0.0007 [−0.0032, +0.0049], 0.65 |
| CNT22 s7+s123 vs v8 CNT22 + DDPcheck | 0.9326 | 0.9314 | +0.0012 [−0.0045, +0.0067], 0.70 |
| V2S + CNT12 + CNT22 (új seedek vs v8) | 0.9396 | 0.9398 | −0.0002 [−0.0030, +0.0027], 0.43 |
| V2S4 + C0 + CNT12_4 + CNT22 vs ugyanez v8-ból | 0.9413 | 0.9409 | +0.0003 [−0.0021, +0.0029], 0.61 |
| ugyanez vs H4_v8 (LB 0.943) | 0.9413 | 0.9417 | −0.0005 [−0.0041, +0.0032], 0.42 |

  - **A 208-on választott epoch (orákulum, a v8-éval azonos optimizmus)** a hat új runból négyben ugyanaz, mint a
    450-es best.pt; a másik kettőben +0.0009 (V2S s123) és +0.0012 (CNT22 s123). A szelekciós torzítás tehát kicsi;
    vele együtt a háromas v11 − v8 +0.0004.
  - A v11-es V2S mind a 4 seedje a v8-as s42 alatt vagy körül van (0.9342 / 0.9331 / 0.9319 / 0.9309 vs 0.9332;
    átlagos egyedi különbség −0.0007). Egyedi párok szórása nagy (CNT22: +0.0068 és −0.0051) — ez seed-zaj, a
    v8 két CNT22-je is 0.0076-tal tért el.
  - Per target (háromas, 208): PF OA −0.009, Fracture −0.006, MCL −0.004; Lateral Meniscus +0.009, Effusion,
    Synovitis +0.005. A korábbi Lateral OA / MCL veszteség itt nem ismétlődik (Lateral OA +0.004).
  - v11-en belül (full_450): az új V2S-pár −0.0031 [−0.0062, +0.0001] a régi (s42/s2) alatt, az új CNT12-pár
    +0.0023 felette → seed-zaj ~0.003 páronként. 4 seed a 2-höz: V2S +0.0003, CNT12 +0.0027 [+0.0012, +0.0044].
    CNT22 a CNT12-nél −0.0014, Spearman 0.950 (nem diverzebb); a négyes összetételben +0.0004 [−0.0006, +0.0016].
    V2S4 + C0 + CNT12_4 = H6 + 0.0001.
  - **Értelmezés:** a korábbi „minden azonos összetételű párban ~−0.002" a több seeddel ~0-ra zsugorodik
    (−0.0002 … +0.0003). A v11 − v8 különbség a 208-on a seed-zajon belül van; a 242 tanító study kiesésének ára
    lokálisan nem mérhető. Az LB-n mért −0.002 (H6 vs H4) is 3 tizedesjegyen belüli. A legjobb v11 összetétel a
    208-on a H4 alatt −0.0005 → LB-n kb. a H4 szintje várható, nem jobb; a +0.003-as szabály nem teljesül.
- **C0 v11 s7 / s123 (Linux, 2026-10-17) + összesítés: minden v11 run vs v8-párok.** Scratchpad
  `compare_v11_all_seeds.py` (2000×). A `C0_..._s2_..._20261015_214415` mappa valójában a s42 Linux run bájtra azonos
  másolata (seed 42) → kihagyva; a Windowsos C0 s2 az átlagokban nincs benne (a v11 C0 = s42, s2, s7, s123).
  - C0 s7 (ep10) 0.9194 / 0.9325 / 0.9077, s123 (ep8) 0.9190 / 0.9271 / 0.9115 (full / 208 / 242); a 208-orákulum
    epoch mindkettőnél ugyanaz. Az új pár a régihez (s42+s2) −0.0005 [−0.0040, +0.0029]; a C0 a legstabilabb arch
    (4 egyedi full_450: 0.9190–0.9198, sd 0.0004), a 4 seed átlaga a 2-höz +0.0015.
  - **v11 archonként (full_450, összes seed átlaga):** V2S 0.9285, CNT12 0.9251, C0 0.9250, CNT22 0.9233, R50 0.9084.
    V2S − C0 +0.0035 [−0.0006, +0.0074], V2S − CNT12 +0.0034; C0 = CNT12.
  - **Ensemble (full_450):** V2S + C0 + CNT12 minden seeddel 0.9309 / 0.9414 / 0.9220, a H6-hoz +0.0006
    [−0.0004, +0.0015]; + CNT22 ±0.0000, + R50 +0.0002, mind a 16 run egyenlő súllyal 0.9312 (+0.0003) — a seedek
    száma után már semmi sem mozdít.

| 208-on, v11 − v8 | egyedi runok átlaga | 2-seed átlagok (v11 összes párja) | összes seed átlaga |
|---|---|---|---|
| V2S (v8: 1 run) | −0.0007 (n=4 vs 1) | — | +0.0042 (4 vs 1 seed, v11-nek kedvez) |
| **C0** | **−0.0042** | **−0.0039 (−0.0055 … −0.0019)** | −0.0019 [−0.0062, +0.0023] (4 vs 2) |
| CNT12 | −0.0018 | −0.0011 (−0.0023 … +0.0002) | +0.0007 (4 vs 2) |
| CNT22 | +0.0008 | +0.0012 | +0.0012 |
| **R50** (v8: 1 run) | **−0.0071** | — | −0.0059 [−0.0112, −0.0010] |

  - Azonos seedes párok (208): V2S s42 +0.0010, C0 s42 −0.0029, C0 s2 −0.0062, CNT12 s42 −0.0018, s2 +0.0004,
    R50 s42 −0.0067 [−0.0130, −0.0009].
  - **A v11 veszteség a C0-ra és a R50-re koncentrálódik**, a V2S / CNT12 / CNT22 a seed-zajon belül egyezik.
    C0 targetenként (összes seed): PF OA −0.019, Lateral OA −0.015, Medial OA −0.007; R50: MCL −0.034, Lateral OA
    −0.029, ACL −0.016. A 208-orákulum epoch a C0 új seedjeinél nem változtat → nem szelekciós torzítás.
  - **Ensemble a 208-on:** V2S + C0 + CNT12 seed-illesztve (V2S s42, C0/CNT12 s42+s2) −0.0015 [−0.0049, +0.0016];
    minden v11 seeddel −0.0004; + CNT22 0.0000 / +0.0003; + R50 −0.0015 / −0.0012; R50 + V2S + C0 (H4 B0 nélkül)
    −0.0024 / −0.0018. A H4-hez (0.9417): H6 −0.0015 (LB −0.002), v11 V2S + C0 + CNT12 összes seeddel −0.0003
    [−0.0037, +0.0031], + CNT22 −0.0005.
  - Ensemble per target (V2S + C0 + CNT12 + CNT22, 208): PF OA −0.011, MCL / Medial Meniscus / Medial OA −0.003;
    Effusion, Synovitis +0.008, Lateral Meniscus +0.006.
  - **Értelmezés:** a 242 kiesett tanító study a C0-nál (és a R50-nél) mérhetően ront, főleg az OA-targeteken;
    a V2S és a ConvNeXt-ek nem érzik. Több seed ezt az ensemble szintjén kiegyenlíti: a 12-runos v11 V2S + C0 + CNT12
    a 208-on a H4 szintje (−0.0003), így LB-n ~0.942–0.943 várható (a H6 −0.0015-ből −0.002 lett). A +0.003-as
    szabály nem teljesül; LB-próba csak akkor éri meg, ha a 12 checkpoint inferenciaideje belefér.
- **Epochválasztás 208 vs 450 szerint, minden modellre (2026-10-17;** scratchpad `selection_table.py`). „208-sel."
  = a 208-on legjobb epoch, „450-sel." = best.pt. A 208 szerinti választás csak az `epoch_predictions/`-ös
  (s7/s123) runoknál számolható; a v8-nál a 242 / 450 nem értékelhető (a 242 tanító adat volt).

| modell | seed | train | 208-sel. ep / AUC 208 | 450-sel. ep | AUC 208 | AUC 242 | AUC 450 |
|---|---|---|---|---|---|---|---|
| V2S | 42 / 2 | v11 | — | 8 / 8 | 0.9342 / 0.9331 | 0.9159 / 0.9172 | 0.9249 / 0.9248 |
| V2S | 7 | v11 | 9 / 0.9319 | 9 | 0.9319 | 0.9174 | 0.9232 |
| V2S | 123 | v11 | 7 / 0.9318 (242: 0.9087, 450: 0.9187) | 8 | 0.9309 | 0.9111 | 0.9193 |
| V2S | 42 | v8 | 10 / 0.9332 | — | | | |
| C0 | 42 / 2 / 2 Win | v11 | — | 11 / 10 / 10 | 0.9286 / 0.9296 / 0.9297 | 0.9122 / 0.9130 / 0.9151 | 0.9192 / 0.9198 / 0.9210 |
| C0 | 7 | v11 | 10 / 0.9325 | 10 | 0.9325 | 0.9077 | 0.9194 |
| C0 | 123 | v11 | 8 / 0.9271 | 8 | 0.9271 | 0.9115 | 0.9190 |
| C0 | 42 / 2 | v8 | 11 / 0.9314, 13 / 0.9358 | — | | | |
| CNT12 | 42 / 2 | v11 | — | 10 / 10 | 0.9329 / 0.9318 | 0.9039 / 0.9076 | 0.9185 / 0.9188 |
| CNT12 | 7 | v11 | 10 / 0.9284 | 10 | 0.9284 | 0.9112 | 0.9197 |
| CNT12 | 123 | v11 | 8 / 0.9318 | 8 | 0.9318 | 0.9128 | 0.9211 |
| CNT12 | 42 / 2 | v8 | 9 / 0.9347, 9 / 0.9314 | — | | | |
| CNT22 | 7 | v11 | 10 / 0.9319 | 10 | 0.9319 | 0.9133 | 0.9216 |
| CNT22 | 123 | v11 | 10 / 0.9288 (242: 0.9127, 450: 0.9193) | 8 | 0.9276 | 0.9151 | 0.9199 |
| CNT22 | 42 Win / DDP | v8 | 10 / 0.9251, 9 / 0.9327 | — | | | |
| R50 | 42 / 2 | v11 | — | 7 / 8 | 0.9173 / 0.9165 | 0.9005 / 0.8979 | 0.9079 / 0.9061 |
| R50 | 42 | v8 | 9 / 0.9240 | — | | | |
| B0 | 42 | v8 | 7 / 0.9292 | — | | | |

  - A 8 mérhető v11 runból 6-ban a két választás azonos; a másik kettőben a 208-as epoch a 208-on +0.0009 / +0.0012,
    a 450-en −0.0006 / −0.0006 → a választási torzítás ~0.001, a v11 − v8 különbséget nem magyarázza.
  - A v8 best epochjai (7–13) kicsit későbbiek, mint a v11-éi (7–11). Az early stopping türelem eltérése (v8: 5,
    v11: 3) nem számít: a v8 history-ján 3-as türelemmel mind a 9 run ugyanazt az epochot választotta volna.
  - **Mi más tér el a v8 és a v11 runok között a 242-n kívül** (config-diff, 2026-10-17): a v11-ben új
    sorozatválasztás (`selection.knee_consistency`, `non_knee_patterns`, `bilateral_min_separation_mm`) és
    `paths.label_mask_csv`. A 3 alapslotban a régi (3slot/4slot backup, before_knee_consistency) és a mostani
    `series_selection.csv` csak 2 study 4 slotjában tér el (1 train_v11, 1 new_242, 0 a 208-ban); a maszk 8 study
    (7 train_v11, 1 new_242). A prep hash azonos (9b43435e7ac6). → A v11 − v8 különbség gyakorlatilag csak a 242.
  - **Tervezett ellenőrzés:** train_v8 (a 242 a régi LLM-címkéivel) a mostani v11-es beállításokkal, holdout208,
    s42/s2: `HOLDOUT=208 TRAIN_CSV=$DATA_ROOT/train_v8.csv ./run_holdout.sh ...` (kódváltozás nem kell). Ugyanaz a seed
    itt sem ad valódi párosítást (a futások nem determinisztikusak, a Linux/Windows „azonos seed" Spearman 0.946).

### 14. 5 slotos C0 a publikus Raptor-elrendezéssel (train_v8, holdout208, 2026-10-15): rosszabb, lezárva
- **Kérdés:** segít-e a publikus Raptor-notebook (rsna-versia-7) 5 slotja: Sag fluid, Sag nem-fluid,
  Cor fluid, Cor nem-fluid, Axial? Adaptálva: a 3 alapslot bitre azonos, két új puha slot,
  `sagittal_second` és `coronal_second` (`<sík>_second` szűrő: a sík másik series-e,
  nonfs PD > T1 > T2 > CSV non-fluid > tartalék). Tábla: `work/manifest/series_selection_5slot.csv`.
  Lefedettség: sag2 4288, cor2 3283 (ebből T1 2836, vagyis nagyrészt a régi coronal_t1), mind az 5
  slot 3166 / 4407.
- **Runok** (C0 E3 img320, grad checkpointing):
  - `C0_S5_img320_trainv8_holdout208_20261015_020654`: s42, Windows, best ep10, **0.9300** (3 slot: 0.9314).
  - `C0_S5_img320_trainv8_s2_holdout208_20261015_113340`: s2, Linux 2 GPU, **0.9251** a 9. epochban
    (3 slot: 0.9358). A helyi másolat a 11. epoch közepén készült, a végleges mappával újra kell futtatni.
  - Költség: kb. 1.55× epochidő (1910 s vs 1235 s), GPU-csúcs 6.6 GB vs 4.2 GB.
- **Eredmény** (hard macro AUC, best epoch; szkript: `notebooks/slot5_compare.py`):

| | 208 | new-100 | 58 | A50 |
|---|---|---|---|---|
| C0 3 slot átlag (s42+s2) | 0.9376 | 0.9451 | 0.9112 | 0.9338 |
| C0 5 slot átlag (s42+s2) | 0.9305 | 0.9382 | 0.9049 | 0.9177 |

  - 5 slot − 3 slot, páros bootstrap (208): s42 −0.0015 [−0.0077, +0.0044]; s2 −0.0107
    [−0.0184, −0.0027]; átlag **−0.0071 [−0.0127, −0.0019]**, P>0 = 0.00.
  - Targetenként (208, átlagok): MCL −0.031, Lateral OA −0.023, PF OA −0.019, ACL −0.009,
    Medial OA −0.009; Effusion +0.007, a többi ±0.003.
  - Ensemble a H4-hez (0.9417) képest: C0 s2 → 5 slot s2 csere −0.0024 [−0.0047, −0.0001];
    → 5 slot átlag −0.0018; 5. tagként +S5 s42 −0.0002. A kapu (+0.003) messze nincs meg.
  - Az 5 slotos modell jobban hasonlít a CNN-ekre (Spearman a B0/V2S-sel ~0.87–0.88 vs a 3 slotos
    C0 0.83–0.86), vagyis diverzitást sem hoz.
- **Értelmezés:** a harmadik extra slotos próbálkozás (coronal_t1, sagittal_nonfs után) is negatív.
  A nem folyadékérzékeny második series főleg a szalag- és OA-targeteken ront; valószínűleg
  hígítja a pooling-ot és a head-et, miközben a FS-szekvenciák információját nem egészíti ki.
- **Döntés:** **lezárva**. Extra slotot a jelenlegi architektúrával nem javaslok. A `<sík>_second`
  szűrő a kódban marad (selftest `second_slot_selection`).

### Mérési protokoll (minden lépésre)
- 2026-10-15-től: elsődleges a **full_450** (holdout450, train_v11), mellette külön az original_208 és a new_242.
- Korábban: elsődleges a new-100, mellette 208/158/58/Andrew50, OOF predikciókból, páros bootstrap, curve mean ep6–10.
- Egymodelles összehasonlításnál 2 seed vagy ensemble-szintű döntés.
- <0.008 lokális nyereség csak LB-próba után számít.
- A ref208-at rangsorolásra használjuk, hangolásra nem. Az 1. lépés 0.07-es értéke az egyetlen kivétel: egy skalár.

## Kritikus fájlok / meglévő elemek
- Új: `notebooks/build_train_v8.py` (minta: `notebooks/build_ref208_train_v7.py`)
- Új: `notebooks/build_ref220_train_v10.py` (ref220 + train_v10, 9. pont)
- Új: `notebooks/build_ref450_train_v11.py` (ref450 + train_v11, 13. pont)
- `run_holdout.bat/.sh` (TRAIN_CSV, HOLDOUT=450 alapból; a kimeneti mappát a TRAIN_CSV nevéből képzi)
- `notebooks/build_04_notebook.py` (LB-próbához ensemble-bejegyzés)
- Eval: scratchpad `eval_holdout.py`; elemzés: scratchpad `blend_check*.py`
- A repóban lévő másolat (`doc/tovabblepesi_javaslat_2026-10-06.md`) frissítendő ezzel a verzióval.

## Ellenőrzés
- A `build_train_v8.py` assertjei lefutnak, és a változott cellák száma 9556.
- A train_v8 címke AUC-ja a 208-on nem rosszabb a v7-nél.
- A B0 v8 run elindul, a `work/runs/train_v8/` alatt jön létre, és a run.log szerint a train_v8.csv-t olvassa.
- Az eredményeket a B0/H3 v7 baseline-nal vetjük össze a fenti protokoll szerint.
