# Terv – 2. kör (Astra javaslatai alapján)

## ÁLLAPOT (2026-09-21) – mi készült el, mi jön

### Elkészült és ellenőrzött (pytest: 26/26)
- **1. lépés – `cli diagnose`** (`src/knee_mri/diagnose.py`). Baseline eredmény: `work/runs/cv3_20260920_002647_diagnose/`.
  - PASS: célváltozó-sorrend, best checkpoint = history argmax, újratöltés után azonos score (1e-16),
    loss-maszk valódi batchen, OOF-diszjunkció, notebook sigmoid-átlag + prep-hash, checkpoint-egyezés.
  - **Valódi problémák**:
    - (a) túlillesztés: a train loss ×0.012–0.032-re esik, a val AUC a 12–18. epoch után lapos;
    - (b) score-szaturáció: a logitok ±20–40 között vannak (MCL medián −19.9), bf16-kvantáltak (0.25-ös lépés),
      célváltozónként csak ~21% egyedi score, holtversenyek;
    - (c) 82–102 tanítóstudy epochonként teljesen felügyelet nélkül.
- **3. lépés előkészítése**:
  - `cli merge-label-details` → `work/labels/labels_details_all.csv` (52 884 sor, a 58 ref studynál a radiológus-címke, basis=`reference`);
  - `labels.unmentioned_weight_per_target`;
  - `cli freeze-reference` → `work/labels/frozen_reference.csv`, alapértelmezett a `config.yaml`-ban (`paths.frozen_reference_csv`);
  - `EvaluationReference.for_run/from_csv` a `train.py` és az `evaluate.py` számára;
  - ellenőrizve: az A0 (details, `borderline_policy=as_negative`) tanítótáblája bitre egyezik a train_v1-gyel,
    és a fold 0 referencia bitre egyezik a baseline-nal az A1 policy alatt is.
  - Két új selftest: `unmentioned_weight_per_target`, `frozen_reference_roundtrip`.
- **2. lépés eszköze**: a `cli label-audit sample|summarize` elkészült.
  - A kitöltendő lap: `work/audit/label_audit.csv` (100 study / 1200 sor: random 40, synovitis 20, mcl 20, many_empty 20).
  - Az áttekintő oldal: `work/audit/label_audit.html` (kiemelt evidencia).
  - A `summarize` egy kitalált kitöltésen tesztelve.
- **4. lépés eszköze**: a `cli qc --studies --out-dir` elkészült, és elkészült a `qc_checklist.csv` is.
  - A galéria: `work/qc_round2/` (30 study, köztük 12 ref-pozitív MCL/Synovitis/Lateral OA eset); a numeric_ok igaz.
  - **Első vizuális lelet** (3/30 kép megnézve): a `qc_4538994436903404.png`-n (179 mm-es FOV) a 150 mm-es
    foreground-kivágás levágja a patellát és az elülső lágyrészt sagittal síkon, és egy oldalsávot coronal síkon.
    Ez a PF OA-t, az Effusiont (suprapatellaris recessus) és a Synovitist érinti.

### Következő lépések (sorrendben) – TERVEZET, még nincs implementálva
0. **HIBAJAVÍTÁS ELŐSZÖR: az epochváltás nem jut el a tartós DataLoader-workerekhez.** (Külső vélemény, a kódban ellenőrizve.)
   - **ELKÉSZÜLT (2026-09-21):** `EpochSampler` + `resolve_item_key` a `dataset.py`-ban, a `train.py` tanító loadere és a `train_epoch` bekötve;
     új selftest: `epoch_reaches_workers` (pytest 27/27). A régi viselkedés reprodukálva: 2 tartós workerrel a studyk 4/4-e bitre azonos
     volt az epochok között; javítás után 0/4. Szintetikus tanítás `num_workers=2`-vel végigfut. **Következő: új A0 baseline 2 seeddel.**
   - A hiba:
     - a `train.py:232-234` `persistent_workers=True`-t állít be (a futások `num_workers: 8`-cal mentek);
     - a `train.py:248-249` a `set_epoch()`-ot csak a főfolyamat datasetjén hívja meg, a workerek másolata `epoch=0`-n marad;
     - a `dataset.py:357` véletlenszám-generátora `default_rng([seed, epoch, index, slot])`, ezért minden study minden epochban
       **ugyanazt** kapja: ugyanazokat a szeletközéppontokat (`bin_centers`) és ugyanazokat a térbeli/intenzitás-augmentációs
       paramétereket (a GPU-s útnál is, mert a paramétereket a worker sorsolja, `dataset.py:400-406`);
     - csak a keverés sorrendje, a GPU-zaj és a dropout változik.
   - Valószínűleg ez is hozzájárul a túlillesztéshez és a szaturációhoz: a train loss 0.4-ről 0.007-re esik, a val AUC lapos.
   - Javítás: a sampler `(index, epoch)` párokat ad át. Egy új `EpochAwareSampler` (vagy egy `RandomSampler` köré írt wrapper)
     a `set_epoch`-ot a főfolyamatban kapja meg, és a `StudyBagDataset.__getitem__` elfogadja a párt; az `int` indexet is
     megtartjuk a validációhoz és a `predict_studies`-hez. A véletlenszám-generátor továbbra is `(seed, epoch, index, slot)`,
     így determinisztikus marad. Tartalék megoldás, ha ez nem megy: `persistent_workers=False`.
   - Új selftest/pytest: DataLoader `num_workers=2`, `persistent_workers=True`, 2 epoch mellett ugyanarra a studyra
     (a) az epochok között eltérő középpontok és augmentációs paraméterek jönnek, (b) ugyanabban az epochban ismételve azonosak,
     (c) a `num_workers=0` ugyanazt adja, mint a `num_workers=2`.
   - Következmény: a régi baseline (fold 0: 0.903, OOF: 0.891, 58-ref: 0.803) hibás augmentációval készült, ezért **nem
     összehasonlítási alap**. A javítás után az új A0 (2 seed) lesz a referencia, és minden további kísérlet ehhez mér.
1. **QC befejezése:** a maradék 27 kép átnézése és a `work/qc_round2/qc_checklist.csv` kitöltése.
   - **A0 eredmény (s42):** fold 0 macro AUC 0.903 → **0.917**, 11/12 célváltozó javult, az MCL szaturációja 55.6% → 1.4%,
     a legjobb epoch train loss-a 0.024 → 0.141. Maradék: bf16-kvantált logitok (MCL 315/1469 egyedi score) → „fej fp32” kísérlet.
   - **`cli qc-edges` ELKÉSZÜLT** (`qc.py:crop_edge_audit`, selftest `crop_edge_fill`, pytest 28/28). Kalibráció a 30 QC-studyn:
     a sagittal **anterior** él 30%-ban érintett (> 0.3), a posterior 0%-ban, a coronal oldalélek 3–7%-ban, az axial 0%-ban.
     Vizuálisan megerősítve (0126… 0.597: az elülső rész levágva; 4773… 0.287: határeset → a küszöb inkább óvatos).
     Valószínű ok: a foreground-súlypont középpontot a hátsó izomtömeg hátrafelé húzza. **Javasolt I1: középpont az
     előtér kiterjedésének (bounding box) közepére**, változatlan 150 mm-es FOV mellett (így a felbontás sem romlik).
     **500 véletlen study (`work/qc_edges/`):** a sagittal anterior él 33.2%-ban érintett (> 0.3; > 0.2: 44%, > 0.5: 9%),
     a posterior 3.2%-ban, a coronal oldalélek 4–6%-ban, az axial 2–3%-ban. → **Az I1 indokolt, és ez a következő kísérlet.**
   - **A0 s43:** macro 0.9172 (s42: 0.9166) → az új baseline **0.917**; a seedzaj macro szinten ~±0.001, célváltozónként ~±0.007
     (a Synovitis ±0.03, mert csak 21 negatívja van). A s43-ban az ACL 24%-a újra szaturált → a fej fp32 + regularizáció továbbra is kell.
   Eredeti tervezet a mérőszámhoz:
   - Hely: új függvény a `src/knee_mri/qc.py`-ban (`crop_edge_audit(cfg, study_ids)`), és egy CLI alparancs
     (`qc-edges --n-studies 500 --out-dir ...`). A meglévő `cache_path` és `read_cache_entry` függvényekre épül (`qc.py:numeric_qc` mintájára).
   - Mérés: minden cache-elt (study, slot) kötet középső szeletharmadán a 3–4 px széles szélsávokban megnézzük,
     mekkora arányban vannak nem-háttér pixelek (érték > ~0.15 a [0,1] skálán). Ezt oldalanként számoljuk.
   - Az anatómiai irányok a kanonikus orientációból (`geometry.py:24-26`) jönnek:
     - sagittal: oszlopok → anterior (a jobb szél = patella/elülső oldal, a bal = poplitealis oldal);
       a felső és alsó szél (comb, lábszár) mindig „érintett”, ezért ezeket nem értékeljük;
     - coronal: oszlopok → a beteg bal oldala (a két oldalsáv a mediális és a laterális kollaterálisok);
     - axial: sorok → posterior, oszlopok → a beteg bal oldala (mind a négy él releváns).
   - Kimenet: `qc_edges.csv` ((study, slot) soronként az élenkénti kitöltési arány és a `crop_pad_fraction`), valamint
     egy összesítő: síkonként és élenként a „levágott” studyk aránya (küszöb pl. > 0.3).
   - Futtatás: 500 véletlen studyn, nem a teljes 50 GB-os cache-en. A küszöböt a 30 kézzel átnézett QC-képen kalibráljuk
     (pl. a 4538… esetnek jeleznie kell).
   - Döntés: ha a sagittal anterior élen vagy a coronal oldaléleken a studyk jelentős része (pl. > 10%) érintett,
     jöhet az I1 kísérlet. (Eredetileg `fov_mm=170–180` vagy `crop_center=geometric` volt a javaslat; a mérés után a
     bounding-box középpont lett a javaslat, lásd fent és a 3. pont I1 sorát.)
2. **Andrew kitölti a `label_audit.csv`-t** (és opcionálisan egy Qwen második vélemény a `llm_second_opinion` oszlopba,
   a `02_label_train_reports_*` notebookok API-kódjával). Utána: `cli label-audit summarize`.
3. **Fold 0 kísérletsor** (B0, 224, egyszerre egy változtatás, 2 seed (42, 43); értékelés: `cli diagnose <run>`):
   - **A0 = KÉSZ, ez a baseline:** macro AUC 0.917 (s42: 0.9166, s43: 0.9172). Parancs: az epochjavítás utáni alapbeállítás
     (a `train_v1.csv`-ből épített tanítótábla bitre egyezik a details-forrás A0-jával).
   - **Az I1 lezárva (nem nyert), a következő az L1.** Eredeti sorrend: (1) az I1 cache-építésének elindítása → (2) közben az L1 implementálása és tesztelése → (3) az I1 tanítása
     → (4) az L1 tanítása → (5) az A1/A2 csak az L1 után.
   - **I1 = a kivágás középpontja az előtér kiterjedésének (bounding box) közepe**, a súlypont helyett; a FOV marad 150 mm.
     - **ELKÉSZÜLT:** `data.crop_center=foreground_extent` (`preprocess.py:estimate_center` + `_extent_midpoint`, 0.5–99.5% robusztus
       kiterjedés; config-validáció; selftest `foreground_extent_center`; pytest 29/29). Megjegyzés: a `foreground` valójában a bináris
       maszk TERÜLETI súlypontja (nem intenzitással súlyozott), ezt a docstring most helyesen írja.
     - **Előzetes mérés a 30 QC-studyn (memóriában, cache nélkül):** a sagittal anterior él érintettsége 30% → 7%, a posterior 0% → 7%
       (ugyanaz a 2 túl nagy térd: 0126…, 1217…, ahol a 150 mm egyik irányban sem elég); a coronal és az axial gyakorlatilag változatlan.
       Vizuálisan ellenőrizve: `work/qc_round2/edges/extent_vs_centroid_sagittal.png`.
     - Cache-építés: `cd src; python -m knee_mri.cli build-cache --set data.crop_center=foreground_extent --workers 8`
       (a config-hash miatt külön mappába, ~50 GB, órák); utána `qc-edges --set data.crop_center=foreground_extent --out-dir ../work/qc_edges_extent`.
     - **Cache KÉSZ** (`I:/Kaggle/data/cache/prep_v1_f659a959c53d`, 13 219 npz; a `paths.cache_dir` átállítva az I:-re, mindkét cache ott van).
     - **`qc-edges` az új cache-en, ugyanazon az 500 studyn (`work/qc_edges_extent/`):** sagittal bármely él 33.6% → **13.6%**;
       anterior 33.2% → 10.6% (> 0.5: 9.4% → 2.0%), posterior 3.2% → 10.8%; mindkét él 7.8% (a térd nagyobb a 150 mm-nél);
       a coronal (6.0% → 5.4%) és az axial (3.2% → 2.8%) nem romlott. → **Mehet az I1 tanítás.**
       A maradék ~8% (mindkét oldalon csonka) csak nagyobb vagy adaptív FOV-val kezelhető → későbbi, külön kísérlet.
     - **I1 EREDMÉNY: NEM FOGADJUK EL.** Fold 0, 2 seed: macro 0.9168 (s42: 0.9143, s43: 0.9193) vs az A0 0.9169 → nincs különbség.
       Célváltozónként: ACL −0.009 (mindkét seeden, a seedzaj felett), PF OA −0.011, Synovitis −0.014 (zajos), MCL +0.019 (zajos),
       Fracture +0.014. A hipotézis közvetlen tesztje: a fold 0 validációjának 31.5%-a (463 study) volt csonka a régi kivágással;
       ezeken az I1 macro −0.0035, a nem csonkákon +0.001 → a csonkítás megszüntetése helyben nem javított. Az A0 a csonka
       studykon eleve jobb volt (0.931), mint a többin (0.915), tehát a csonkulás nem nehezítette a modellt. Valószínű ok: a patella
       és a suprapatellaris recessus az axial síkon (0–3% csonka) is látszik, a síkok redundánsak. Kikötés: a helyi referencia
       riportalapú; a teszt képalapú címkéin más lehet, de ez nem indokol külön beküldést.
       **Az A0 (`crop_center=foreground`) marad a baseline.** A `foreground_extent` opció és a cache (`prep_v1_f659a959c53d`, ~23 GB) megmarad.
     - (Csak ha az I1 nyerne:) a tanításnál és a Kaggle-notebookban is `data.crop_center=foreground_extent` kellene; a notebook a prep-hash eltérést jelzi (`REQUIRE_PREP_MATCH`),
       és a frissített `knee_mri` csomagot fel kell tölteni.
     - Előbb `cli qc-edges` az új cache-en: a sagittal anterior érintettség 33% helyett érdemben csökkenjen, a többi él ne romoljon.
     - Utána fold 0 tanítás 2 seeddel, az A0 (0.917) ellen.
   - **L1 = javított loss + gradiensakkumuláció, 0/1 súlyokkal** (a `RSNA_loss_es_gradiensakkumulacio_javitas.md` szerint).
     - **IMPLEMENTÁLVA (2026-09-21):** `train.loss_normalization: microbatch | window` (alapértelmezés: `microbatch`, így az A0 változatlan).
       `loss.py`: `WindowNormalizer` + `window_normalized_bce` (a régi `masked_class_normalized_bce` érintetlen); `train.py`: a `train_epoch`
       szétbontva `_train_microbatches` (régi, változatlan sorrend) és `_train_windows` (az ablak pufferelése, nevezők előre, az üres
       mikrobatch forward nélkül kimarad, az üres ablaknál nincs lépés) ágra; a history új oszlopa `train_objective` (a `train_loss` marad diagnosztikai).
       Elfogadási tesztek (selftest + pytest, 33/33): `window_loss_matches_full` (érték és logitgradiens, max eltérés 1.9e-9),
       `window_weight_scaling` (pontosan 0.2×, a nevezők változatlanok), `window_empty_cases`, `window_training_step` (szintetikus).
       Valódi adatos próba (`L1_smoke_real`, 64 study, 2 epoch, 8 worker) rendben. Mellékjavítás: `utils.remove_file_logging`.
       Futtatás: `--set train.loss_normalization=window`, nevek `L1_window_fold0_s42/s43`.
     - **L1 EREDMÉNY (2026-09-22): AUC-SEMLEGES, infrastruktúraként ELFOGADVA** (az A1/A2 előfeltétele).
       Fold 0: s42 0.9191 (ep. 17), s43 0.9154 (ep. 11), átlag **0.9172** vs A0 0.9169. Páros bootstrap (1000):
       +0.0004, SE 0.0030, 95% CI [−0.0055, +0.0066], P(>0) = 0.54. Célváltozónként: Fracture +0.015 (CI [+0.003, +0.030],
       mindkét seeden), MCL +0.021 (mindkét seeden, CI átfedi a 0-t), Synovitis −0.028 (mindkét seeden, CI [−0.066, +0.004]),
       a többi ±0.006-on belül. 58-ref (14 study): 0.822/0.820 vs A0 0.832/0.806. A szaturáció megmaradt (s42: Fracture 37.8%).
       (Előtte egy hibás futás: `L1_extent_fold0_s42/s43` = microbatch + foreground_extent, bitre az I1 s42 → törölhető.)
       **Az A1/A2 innentől `loss_normalization=window`-val fut, az L1 a baseline.**
     - A hiba: a `loss.py` mikrobatchenként normalizál; `microbatch_studies=1` mellett `w·BCE / w = BCE`, vagyis a súly nagysága
       kiesik, és egy osztály súlya attól függ, hány másik címke van ugyanabban a studyban.
     - Az új célfüggvény: `L_c = Σ_i m_ic·w_ic·BCE_ic / D_c`, ahol `D_c = Σ_i m_ic` a teljes 8-as ablakon (`m = w > 0`);
       `L = átlag_{c: D_c>0} L_c`. A mikrobatch-hozzájárulás ugyanezekkel az ablak-nevezőkkel normalizálódik, és nem osztódik tovább 8-cal.
     - Implementáció: a tanító ciklus az ablak mikrobatcheit előre beolvassa egy pufferbe (CPU, ~8×45 MB), kiszámolja a `D_c`-t
       és az aktív osztályok számát, majd egyenként forward/backward. Az ablak végén egyszer: unscale → clip → optimizer → scheduler.
       Teljesen üres ablaknál nincs lépés; a rövidebb utolsó ablak a tényleges számokkal fut.
     - Naplózás: az optimalizált célfüggvény külön oszlop; a mostani `LossAccumulator`-os epoch-loss diagnosztikai néven marad.
     - Tesztek (a dokumentum 4. pontja): az egyben számolt ablak-loss logitgradiense egyezzen a 8 részletben akkumulálttal (FP32);
       az `1.0 → 0.2` súlyváltás pontosan ötödöl, a többi nevező változatlan; a kizárt címkék gradiense 0; üres mikrobatch,
       üres ablak és rövid utolsó ablak. A `class_normalized_loss` és a `gradient_accumulation` selftestet át kell írni.
     - Futtatás: a jelenlegi cache-en, 2 seeddel, az A0 ellen (vagy az I1 ellen, ha az I1 nyert, és akkor annak cache-én).
   - **A1 / A2 = gyenge negatívok (`not_mentioned`), CSAK az L1 után.** Feltétel: a súlyhígítás tudatos kezelése.
     Mivel a gyenge címke a nevezőben teljes 1-gyel szerepel (`m=1`), sok gyenge címke **növeli `D_c`-t, és felhígítja a valódi címkéket**.
     Synovitis példa (~1 valódi + ~7 `not_mentioned` egy ablakban): a valódi címke `BCE/8` a mostani `BCE/1` helyett, a 7 gyenge
     negatív együtt `7·0.2/8 ≈ 0.175` > a valódi `0.125`. A dokumentum állítása („1.0 → 0.2 pontosan ötödöl”) csak meglévő címke
     átsúlyozására igaz, új címke bekapcsolására nem. Lehetőségek:
     - osztályonként kisebb súly (`unmentioned_weight_per_target`, a Synovitisre külön);
     - a nevezőben súlyösszeg, ablakszinten;
     - globális (a tanítóhalmazból előre számolt) nevező.
     A választást az audit (A2) és egy előzetes számítás (az ablakonkénti valódi vs. gyenge hozzájárulás osztályonként) indokolja.
     - **Előzetes számítás KÉSZ (2026-09-22, fold 0 train, 2938 study, 60 kevert epoch, 8-as ablak, w = 0.2):**
       - Gyenge (`not_mentioned`) cellák: Synovitis 2505 (valódi: 339 poz / **53 neg**), Baker's 1407, Fracture 1232,
         Contusion 932, Lateral OA 875, Medial OA 688, PF OA 563; a többi 159–298.
       - A gyenge rész aránya az osztály gradiensében (≈ w·n_weak / (n_real + w·n_weak)): Synovitis **56%**, Baker's 17%,
         Fracture 13%, a többi ≤ 10%.
       - **A mostani (maszkszámos) nevezővel a valódi címkék gradiensét a gyenge címkék w-től függetlenül hígítják**
         (a nevezőben 1-gyel számítanak): valódi tömeg A1/L1 = Synovitis **0.20**, Baker's 0.48, Fracture 0.56, Contusion 0.63,
         Lateral OA 0.64; összesen 0.71. Ez tehát egy nem szándékos osztály-átsúlyozás is.
       - A súlyösszeg-nevező (ablakon belül) a csak gyenge címkét tartalmazó ablakokban visszahozza a régi hibát (w/w = 1):
         a Synovitisnél az ablakok ~1/3-a ilyen. **Elvetve.**
       - **Javaslat: globális osztálynevező** `D_c = K · Σ_i w_ic / N_train` (a tanítóhalmazból előre, konstans): arányos,
         nincs kioltás, nincs w-független hígítás. Ez kódváltozás (`loss_normalization=global`), és előbb w = 0 mellett
         külön összevetendő az L1-gyel (egy változtatás egyszerre).
       - A Synovitis súlyát az audit P(pozitív | not_mentioned) aránya döntse el; 20%-os gyenge részarányhoz w ≈ 0.04 kellene,
         a Baker'shez ≈ 0.24, a többinél a 0.2 rendben van.
     - **`loss_normalization=global` IMPLEMENTÁLVA (2026-09-22):** `loss.py: GlobalNormalizer` (`D_c = K·Σw/N_train`, K = 8)
       + `global_normalized_bce`; `dataset.label_weight_matrix()`; a `train.py` a `_train_windows` útvonalat használja fix
       nevezőkkel (a D_c-t induláskor naplózza). Tesztek (36/36): `global_loss_matches_formula`, `global_no_dilution`
       (gyenge címke be: a többi gradiens bitre azonos; egy magányos 0.2-es címke pontosan 0.2×; kontrasztként a window
       policy felezné a valódi címkét), `global_training_step`. Valódi adatos próba (64 study, A1-beállítás) rendben.
     - **Következő futások:** G0 = `global`, w = 0 (az L1 ellen, 2 seed) → A1 = G0 + details-forrás + `unmentioned_weight=0.2`.
     - **G0 EREDMÉNY (2026-09-22): ELFOGADVA, ez az új baseline.** Fold 0: s42 0.9194 (ep. 10), s43 0.9224 (ep. 11),
       átlag **0.9209** vs L1 0.9172 / A0 0.9169. Páros bootstrap (1000): G0−L1 +0.0036, SE 0.0021, CI [−0.0006, +0.0077],
       P(>0) = 0.96; G0−A0 +0.0040, CI [−0.0008, +0.0092]. A küszöb (~0.005) alatt, de konzisztens, és az A1-hez amúgy is ez kell.
       Célváltozónként (vs L1, mindkét seeden azonos irányban): PF OA +0.012 és Effusion +0.006 (CI > 0), Synovitis +0.020,
       MCL +0.013, Medial OA +0.005; **ACL −0.008 (CI < 0, mindkét seeden; az A0-hoz képest is)**, Fracture −0.010.
       58-ref (14 study): 0.832 / 0.820. Szaturáció: max 4.3% / 11.6%.
     - **A1 s42 (2026-09-22), előzetes, az s43 fut:** macro (valódi címkék) 0.9211 vs G0 átlag 0.9209 → semleges
       (CI [−0.0073, +0.0074]); Fracture +0.014 (CI > 0), Synovitis −0.016 (zajos), Lateral Meniscus −0.010. 58-ref (14): 0.816.
       **Kiegészítő mérés (`scratchpad/weak_auc.py`): a validáció `not_mentioned` cellái** (~4800 cella, a befagyasztott
       referenciában NINCSENEK benne): pozitív vs not_mentioned AUC macro 0.872/0.874 (G0) → **0.906** (A1), mind a 12
       célváltozón javul; pozitív vs (negatív + not_mentioned) 0.908/0.909 → **0.921**. A not_mentioned cellák átlagos
       score-ja 0.125/0.148 → 0.069; Synovitis 0.40/0.63 → 0.11. Kikötés: ezt az A1-et pont erre tanítottuk, így csak akkor
       ér valamit, ha a not_mentioned valóban többnyire negatív (a 58-as referencián 157 neg / 27 poz ≈ 85% neg).
       **Hipotézis a helyi 0.92 vs LB 0.866 résre:** a helyi mérés kihagyja a not_mentioned cellákat (a cellák ~25%-a),
       a teszt képalapú címkéi viszont minden cellát tartalmaznak; a G0 ezeken magas score-t ad (Synovitis 0.4–0.6).
     - **A1 VÉGEREDMÉNY (2 seed):** s42 0.9211, s43 0.9116 → átlag **0.9164** vs G0 0.9209: **−0.0045**, CI [−0.0110, +0.0017],
       P(>0) = 0.07 → a valódi címkéken valószínűleg kis veszteség. Mindkét seeden romlik: Synovitis −0.023 (a várt hígítás:
       a gyenge rész az osztály gradiensének 56%-a), Lateral Meniscus −0.016 (CI < 0), MCL −0.012, PF OA −0.010; javul: Fracture +0.009.
       A not_mentioned cellákon viszont mindkét seeden javul: poz vs nm 0.873 → **0.900**, poz vs (neg+nm) 0.909 → **0.917**.
       58-ref (14): 0.816 / 0.806. **Helyben nem dönthető el** (melyik mérés tükrözi a tesztet) → Kaggle-beküldéssel döntünk.
     - **Javasolt döntő teszt, új tanítás nélkül:** két beküldés a meglévő fold 0 checkpointokkal (seedpáronként ensemble):
       `CHECKPOINT_GLOB` = `G0_global_fold0_s4*/best.pt` ill. `A1_weak02_fold0_s4*/best.pt`. Előtte a Kaggle-datasetben a
       `knee_mri` csomagot frissíteni kell (az új `loss_normalization: global` configot a régi validáció elutasítaná).
       Ha az A1 nyer az LB-n → A2 (Synovitis kisebb súly) és 3 fold; ha nem → G0 3 fold.
   - **S1 = `data.image_size=320` (2026-09-22), s42 előzetes: 0.9240** vs G0 átlag 0.9209 → +0.0031, SE 0.0028,
     CI [−0.0022, +0.0085], P(>0) = 0.87; egy seednél a küszöb ~0.007, tehát kell a s43. Indok: a natív felbontás 0.31–0.33 mm/px,
     a 150 mm-es kivágás natívan ~460 px, a 224 px tehát ~2× kicsinyítés. Célváltozónként a várt helyen javul:
     Lateral Meniscus +0.015 és Medial Meniscus +0.010 (CI > 0), MCL +0.012, ACL +0.005; **PF OA −0.018 (CI < 0)**.
     58-ref (14): **0.837** (eddigi legjobb). Cache: `prep_v1_<320>` (~47 GB, I:), tanítás ~1.7 helyett ~3 óra, GPU 6.4 GB.
     - **KAGGLE LB (2026-09-23): 0.877** az `S1_img320_fold0_s42` EGYETLEN checkpointjával (a korábbi 0.866 a régi,
       hibás epochkezelésű 3 foldos együttes volt). Tehát az epoch-javítás + `global` loss + 320 px legalább +0.011,
       úgy, hogy közben elvesztettük a 3 foldos ensemble-t és a tanítóadat 1/3-át. Helyi fold 0: 0.9240 → a rés ~0.047,
       ami illik a címkezaj/not_mentioned hipotézishez.
     - **Következő beküldések (olcsók, nincs új tanítás):** A1 vs G0 fold 0 (224) párban → eldönti a gyenge címke kérdést;
       majd az S1 s43 + 3 fold.
   - **P1 = `model.spatial_pool` (2026-09-23, IMPLEMENTÁLVA, tanítás még nem futott).** Indok: az enkóder után minden szelet
     H×W térképe EGY globális átlaggá lapul (`AdaptiveAvgPool2d(1)`), 320 px-nél 10×10 → egy 2×2-es gócos lelet amplitúdójának
     4%-a marad. Ez a 320-as kísérlet korlátja is. Opciók: `avg` (eredeti, alapértelmezés) | `avgmax` (átlag + maximum,
     feature_dim 2560, nincs új paraméter) | `attention` (1×1 konvolúciós pontozó + softmax; a záró réteg NULLA kezdőértékkel,
     tehát a tanítás pontosan az `avg` viselkedéséből indul; +164 k paraméter, a head learning rate csoportjában).
     Tesztek (pytest + selftest 42/42): `attention_pool_starts_as_average` (init = átlag, 8.9e-8), `spatial_pool_shapes`,
     `focal_signal_survives_pooling` (átlag 0.04 / max 1.00 / fókuszált attention 1.00), `spatial_pool_training_step`
     (1×1 térképnél a gradiens helyesen 0). Futtatás a 320-as cache-en, az S1 (0.9240) ellen, 2 seeddel.
   - **Windows commit-limit probléma és javítás (2026-09-22):** 320 px-nél az 1. epoch elején elfogyott a commit keret
     (RuntimeError 1455, „a lapozófájl túl kicsi”); a RAM nem volt tele, a commit igen (117 GB / 119 GB). Ok: a tanító ÉS a
     validációs loader is 8 tartós workert tart (a validációsak a tanítás alatt végig élnek, ~15 GB), 320-nál egy study
     float32 bagje 88 MB. Javítás (pytest+selftest 38/38): (a) `dataset.transport_dtype` – a késleltetett (GPU-s) úton a
     workerek float16-ban adják át a bagot, ami a float16 cache mellett bitre veszteségmentes és felezi a megosztott memóriát;
     a `train._forward` a transzformáció előtt szélesíti float32-re; (b) `train.eval_num_workers: 4` / `eval_prefetch_factor: 2`
     + a validációs workerek NEM tartósak (`eval_loader_settings`, az `evaluate.predict_studies` is ezt használja).
     Új selftestek: `float16_transport`, `eval_loader_budget`. A számok nem változnak tőle.
     Az A1/A2 helyi validációs értéke korlátozott: a befagyasztott referencia nem tartalmaz `not_mentioned` cellát, a hatás főleg
     a beküldésen és a 58-as referencián látszik.
   - **Később, külön kísérletként:** a túlillesztés és a szaturáció kezelése (a fej fp32-ben a bf16-kvantálás ellen,
     erősebb regularizáció, rövidebb tanítás); globális vs. ablakos osztálynevező összevetése.
4. A nyertes változat 3 foldon → `merge-oof` → `diagnose` → Kaggle-beküldés.

## Kontextus

A Kaggle LB 0.866 macro ROC-AUC, a cél ~0.95. Astra öt lépést javasol ebben a sorrendben:
(1) az implementáció és az osztályonkénti eredmények ellenőrzése, (2) célzott címkeaudit,
(3) a „nem említett” és a „bizonytalan” szétválasztása, (4) a képbemenet ellenőrzése, (5) mérés egy
rögzített foldon, egyszerre egy változtatással, EfficientNet-B0 alapon.

Az előzetes vizsgálat eredményei, amelyek ezt megalapozzák:

- A `train_v1.csv` pontosan a `labels_details.csv` leképezése. A `positive` → 1. A `negative`
  (explicit_absence, below_threshold **és a 205 borderline is**) → 0. A `not_mentioned` (14 289) és az
  `uncertain` (4 565) → üres. A wide CSV-ből így már **nem** különíthető el a két státusz, a
  részletes export viszont megvan: `F:/Kaggle/llm_labeling/output/{GT,remaining}/runs/<hash>/labels_details.csv`.
  A `status` értékei megegyeznek a `constants.py` STATUS_* értékeivel. Minden sornak van `evidence` mezője.
- A 3-fold baseline (`work/runs/cv3_20260920_002647_*`) számai:
  - OOF macro AUC az LLM-címkéken: 0.891;
  - OOF a 58 radiológus-studyn: **0.803**. Gyenge célváltozók: MCL 0.63, Synovitis 0.70, Lateral OA 0.70,
    Lateral Meniscus 0.73, PF OA 0.78, Fracture 0.78.
- A 58 studyn az LLM `basis` mezője és a radiológus-címke így viszonyul egymáshoz:

  | basis | ref neg / pos |
  |---|---|
  | not_mentioned | 157 / 27 |
  | insufficient_detail | 25 / 48 |
  | below_threshold | 51 / 28 |
  | explicit_absence | 192 / 6 |

  Tehát a „nem említett” többnyire negatív, a „nem eldönthető” többnyire pozitív. Összemosni vagy mindet
  kihagyni egyaránt hibás. A Synovitisnek csak 43 LLM-negatívja van (+31 a referenciából).
- Az inferencia (`notebooks/build_04_notebook.py`) a checkpointonkénti sigmoidok átlagát veszi (`:353-379`),
  és ellenőrzi a preprocessing-hash egyezését (`:255`). Ez a rész rendben van, csak rögzíteni kell a diagnosztikában.
- A `LabelTable.binary_reference()` (`labels.py:95-104`) csak a positive/negative cellákat veszi be
  a validációs referenciába. A not_mentioned cellák tanítósúlya ezért nem változtatja meg a referenciát.
  Ez Astra 3. pontjához elengedhetetlen.

---

## 1. lépés – Implementáció- és eredmény-diagnosztika

Új script: `src/knee_mri/diagnose.py`, `cli diagnose <run-group>` alparanccsal. A meglévő
`metrics.py` függvényeit használja, és csak olvas.

- Foldonként és OOF-ra, célváltozónként: AUC, n_pos, n_neg, a kizárt (üres) cellák száma státusz szerint
  bontva, és az AUC a 58 radiológus-studyn. Kimenet: `diagnose_per_class.csv`.
- Ellenőrzések, mindegyik PASS/FAIL sorral:
  - a célváltozók sorrendje a checkpointban, a configban, a `train_v1.csv`-ben és a `sample_submission.csv`-ben;
  - a loss-maszk: egy batchen a nulla súlyú cellák gradiense 0 (a `selftest.py` masked_label_gradients
    ellenőrzését futtatjuk valódi batch-csel);
  - a `best.pt` epochja egyezik a `val_summary.json` legjobb epochjával, és újratöltés után ugyanazok a validációs logitok jönnek ki;
  - az OOF score folytonos: az egyedi értékek száma célváltozónként nagy legyen, és a szaturált (<1e-6 vagy >1−1e-6) predikciók
    arányát is kiírjuk, mert az ACL score-ok ~1e-13-ig mennek;
  - a három checkpoint egymás közötti predikció-korrelációja a 58 studyn.
- A notebook ensemble-logikájáról (sigmoid-átlag) egy sor igazolást írunk a riportba.

## 2. lépés – Célzott címkeaudit (~100 lelet)

Új script: `src/knee_mri/label_audit.py`, `cli label-audit`.

- Mintavétel fix seeddel, a 4349 „remaining” studyból:
  - 40 véletlen study;
  - 20 study, ahol a Synovitis nem üres vagy synovitis-kulcsszó van a szövegben;
  - 20 study, ahol az MCL positive vagy below_threshold, vagy MCL-kulcsszó van a szövegben;
  - 20 study, ahol legalább 8 cella üres.
- Kimenet: `work/audit/label_audit.csv` + HTML (a `report.py` mintájára). Soronként: study, célváltozó, LLM status/basis,
  `evidence`, `reason`, a riport releváns részlete kiemelve, és üres mezők a döntésnek:
  - `audit_label` (0/1/undecidable);
  - `error_type` ∈ {`ok`, `llm_misread`, `definition_misapplied`, `not_decidable_from_report`};
  - `note`.
- Második vélemény segédjelként: a Qwen egy független promptot kap ugyanarra a ~1200 cellára
  (`llm_second_opinion` oszlop). Ez **nem** referencia, csak arra való, hogy Andrew figyelmét az ütközésekre irányítsa.
- A kitöltést Andrew végzi. Utána egy összesítés fut célváltozónként és basis-onként: hibaarány hibatípus szerint,
  és P(audit=1 | basis) not_mentioned, insufficient_detail és below_threshold esetén.
  Ez a 3. lépés osztályonkénti döntésének az alapja.

## 3. lépés – A „nem említett” és a „bizonytalan” szétválasztása

- Új összefűző script: `src/knee_mri/merge_label_details.py`. A GT és a remaining `labels_details.csv`-ből
  egyetlen `work/labels/labels_details_all.csv` fájlt készít. A 58 referencia-studynál a radiológus-címke
  lép a helyére (positive/negative, basis=`reference`), ahogy a `train_v1.csv` is csinálta.
- Tanítás: `labels.source=details`, `paths.labels_details_csv=work/labels/labels_details_all.csv`.
  A `labels.py:_resolve_status` már most is külön kezeli a `not_mentioned` (`unmentioned_weight`) és az
  `uncertain` (`uncertain_weight`) státuszt.
- **Kötelező invariáns:** a validációs referencia minden kísérletben ugyanaz. Az A0 és a train_v1 baseline
  `validation_reference.csv` fájljának egyeznie kell. A borderline most 0-ként szerepel a train_v1-ben, a details
  útvonalon `exclude`. Ezért az A0-ban `borderline_policy=as_negative` kell, hogy a referencia bitre egyezzen.
  Új teszt: `tests/test_pipeline.py` → a referencia nem függ az `unmentioned_weight`/`uncertain_weight` értékétől.
- Új config-kulcs, osztályonkénti súlyhoz: `labels.unmentioned_weight_per_target: {Synovitis: 0.0, Fracture: 0.2, ...}`.
  A `_resolve_status` a globális érték helyett ezt használja, ha meg van adva. Az értékeket a 2. lépés
  audit-arányai indokolják, és a választás a run configjába kerül.
- Kísérletek (4. lépés protokollja szerint):
  - A0 = details-forrás, minden üres kimarad (ez reprodukálja a baseline-t);
  - A1 = not_mentioned gyenge negatív, egységesen 0.2 súllyal;
  - A2 = osztályonkénti súlyok az audit alapján.
  - Az `uncertain` minden kísérletben kimarad, mert a 58-as adat szerint többségük pozitív, így negatívként ártana.

## 4. lépés – A modell tényleges képbemenetének ellenőrzése

- A meglévő `cli qc` parancs futtatása `--n-studies 30`-cal. A mintában legyen legalább 10 study a 58 referenciából
  (MCL-, Synovitis- és Lateral OA-pozitívak), hogy a kivágás a releváns anatómián ellenőrizhető legyen
  (MCL: a mediális oldal coronal síkon, a synovium: suprapatellaris recessus).
  Ehhez a `qc.py` kiválasztójának kell egy `--studies fajl.txt` opció.
- Ellenőrzőlista (`work/qc/qc_checklist.csv`, study × pont: OK/hiba):
  - a sík helyes;
  - a szeletsorrend monoton;
  - a triplet valódi szomszédokból áll;
  - az intenzitás nem szaturált;
  - a 150 mm-es FOV kivágás nem vágja le a mediális vagy laterális kollaterálist, a patellát és a suprapatellaris recessust;
  - kiírjuk a `crop_pad_fraction` értéket.
- Egyezés az inferenciával: a teszt-notebook cache-éből 3 studyra ugyanaz a QC fut. Összevetjük a
  preprocessing-hash-t és a pixelstatisztikákat a lokális cache-sel (`04_kaggle_test_local.ipynb`).
- Ha a kivágás anatómiát vág le, a javítás (pl. `fov_mm=170` vagy `crop_center=geometric`) **egy külön
  kísérlet** az 5. lépés protokolljában.

## 5. lépés – Mérési protokoll: egy fold, egy változtatás

- Rögzített alap: EfficientNet-B0, 224 px, `split.fold=0`, seed 42, a jelenlegi config.
- A kísérletek ebben a sorrendben jönnek, mindegyik csak egy dologban tér el az előzőleg elfogadott állapottól:
  1. A0 (details-forrás, reprodukció);
  2. A1;
  3. A2;
  4. javított címkék (ha az audit után lesz korrekciós fájl);
  5. képfeldolgozás-módosítás (ha a 4. lépés indokolja).
- Minden futás után a `diagnose` a fold 0 validációján ezeket adja ki:
  - macro AUC és osztályonkénti AUC a befagyasztott LLM-referencián;
  - AUC a fold 0-ba eső radiológus-studykon (~19 study, csak irányjelzőnek);
  - n_pos/n_neg/kizárt darabszámok.
- Döntési szabály: egy változtatást akkor fogadunk el, ha a macro AUC javul, és egyik gyenge célváltozó sem
  romlik számottevően. A zajszint előzetes becsléséhez az A0-t 2 seeddel futtatjuk.
- A legjobb változat utána 3 foldon fut (`run_cv.bat`). Merge-oof, a teljes 58-as OOF, és csak ezután jön
  a Kaggle-beküldés.
- Nagyobb backbone, felbontás, attention-pooling és TTA **csak ezután** kerül sorra, ugyanezzel a protokollal.

---

## Kritikus fájlok

- Új: `src/knee_mri/diagnose.py`, `label_audit.py`, `merge_label_details.py`
- Módosul:
  - `src/knee_mri/cli.py` (3 alparancs);
  - `labels.py` (osztályonkénti unmentioned súly);
  - `config.py` / `src/config.yaml` (új kulcs);
  - `qc.py` (`--studies`);
  - `src/tests/test_pipeline.py`.
- Újrahasznosítva:
  - `metrics.py` (AUC, NA-kezelés);
  - `labels.py:_resolve_status`, `binary_reference`;
  - `selftest.py` (masked gradient check);
  - `report.py` (HTML);
  - `cli qc`;
  - `merge-oof`.

## Ellenőrzés

- `python -m knee_mri.cli selftest` és `python -m pytest src/tests -q` minden kódmódosítás után.
- `cli diagnose work/runs/cv3_20260920_002647`: a baseline számait reprodukálja (OOF 0.891, 58-ref 0.803),
  és minden ellenőrzés PASS.
- `validate-schema` a details-forrással: `n_not_mentioned` = 14 289 + a GT-ben lévő darabszám,
  és `n_uncertain` külön jelenik meg.
- Az A0 fold 0 `validation_reference.csv` fájlja bitre egyezik a baseline fold 0 referenciájával, és az A0 AUC-ja
  a seedzajon belül van a baseline 0.903-hoz képest.
