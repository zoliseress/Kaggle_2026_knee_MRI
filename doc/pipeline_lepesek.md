# A pipeline lépései — mit csinál pontosan az egyes parancsok

Ez a dokumentum a `src/knee_mri` csomag parancsait írja le részletesen: mit olvasnak, mit
számolnak, mit írnak ki, milyen döntéseket hoznak, és mikor állnak meg hibával.

Az angol nyelvű `src/README.md` az áttekintés és a futtatási sorrend; ez a fájl az egyes
lépések belső működését magyarázza.

Minden parancs a `src/` könyvtárból fut, és minden config-kulcs felülírható:

```powershell
python -m knee_mri.cli <parancs> --set data.image_size=288
```

---

## Tartalom

| Parancs | Mire való |
| --- | --- |
| [`selftest`](#selftest) | 22 hibamód-ellenőrzés, adat nélkül |
| [`validate-schema`](#validate-schema) | bemeneti fájlok és címke-lefedettség ellenőrzése |
| [`build-manifest`](#build-manifest) | DICOM-leltár, geometria, sorozatválasztás |
| [`select-series`](#select-series) | csak a sorozatválasztás újrafuttatása |
| [`build-cache`](#build-cache) | determinisztikus előfeldolgozás és cache |
| [`qc`](#qc) | QC galéria — mit lát ténylegesen a modell |
| [`make-splits`](#make-splits) | állandó, csoport-diszjunkt foldok |
| [`train`](#train) | tanítás három módban |
| [`evaluate`](#evaluate) | mentett checkpoint kiértékelése |
| [`report`](#report) | offline HTML riport |
| [`merge-oof`](#merge-oof) | fold-előrejelzések összefűzése |

---

## `selftest`

```powershell
python -m knee_mri.cli selftest [--quick]
```

### Mit csinál

22 célzott ellenőrzést futtat azokra a hibamódokra, amelyek ezt a pipeline-t reálisan el
tudják rontani. Szintetikus adatokon dolgozik, **de a valódi modellt és a valódi gradiens-utat
használja** — semmi nincs kimockolva abból, amit éppen ellenőriz.

Az ellenőrzések három csoportba esnek:

**Geometria és adatépítés**
- `geometric_slice_sorting` — összekevert sorrendű szeletek a vetített `ImagePositionPatient`
  szerint állnak-e helyre; a szabálytalan szeletköz felismerése.
- `plane_assignment` — sagittal/coronal/axial helyes-e a normálvektorból, és a 45°-os ferde
  eset ambiguousként jelölődik-e.
- `inplane_transform_is_reorder` — a kanonikus síkbeli orientáció tényleg csak tömb-átrendezés
  (a pixelértékek halmaza változatlan), nem interpoláció.
- `true_neighbour_triplets` — a triplet `[i-1, i, i+1]` az **eredeti** stackből jön, a szélek
  levágódnak, és megjelölt fizikai hézagon nem lép át.
- `short_stack_padding` — rövid stackek invalid centrumokkal töltődnek fel.
- `dataset_contract` — az item és a batch alakja megfelel a dokumentált kontraktusnak.

**Modell és veszteség**
- `masked_pooling` — az átlag nevezőjébe és a maximumba nem kerül bele a padding; üres slot
  pontosan nulla (nem `-inf`, nem NaN).
- `padding_invariance` — ha a maszkolt helyekre szemetet írunk, a logit **bitre azonos** marad.
- `missing_slot_is_zero` — hiányzó sorozat nem befolyásolja a kimenetet, és véges marad.
- `masked_label_gradients` — nulla súlyú célváltozó nem kap gradienst; üres felügyelet esetén
  differenciálható nulla jön vissza.
- `nan_target_rejected` — NaN célérték nulla súllyal is hibát dob (a `NaN * 0` nem biztonságos).
- `class_normalized_loss` — a veszteség számszerűen egyezik a referenciaképlettel.
- `gradient_accumulation` — az ablak-skálázás helyes, a rövid utolsó ablakot is beleértve.
- `epoch_loss_accounting` — az epoch-veszteség akkumulált osztály-számlálókból jön.
- `sigmoid_not_softmax` — a modell logitot ad, a sigmoid kimenetek nem összegződnek 1-re.
- `train_step_reduces_loss` — valódi forward/backward/optimizer lépés csökkenti a veszteséget.
- `checkpoint_roundtrip` — újratöltés után azonos logitok.

**Címkék és felosztás**
- `single_class_auc_is_na` — egyosztályos célváltozó `NA`-t ad, **soha nem 0.5-öt**.
- `label_join_by_key` — a címke-összekapcsolás kulcs alapú, nem sorpozíció szerinti.
- `empty_numeric_is_unknown` — üres numerikus cella ismeretlen marad (0 súly), nem lesz negatív.
- `patient_group_separation` — egy betegcsoport nem eshet szét több foldba.
- `pseudonymous_patient_id` — a study-nként egyedi `PatientID` study-szintű csoportosításként
  jelenik meg, nem beteg-szintűként.

### Bemenet / kimenet

Semmilyen adatot nem olvas és semmilyen fájlt nem ír. Kilépési kód: 0, ha minden teszt átment,
különben 1.

A `--quick` kihagyja azt a 7 ellenőrzést, amely encodert épít (a modell- és gradiens-alapúakat),
így 15 ellenőrzés marad.

> Ugyanezek pytest alatt is futnak: `python -m pytest tests -q`.

---

## `validate-schema`

```powershell
python -m knee_mri.cli validate-schema [--dicom-sample-limit N] [--keep-going]
```

### Mit csinál

1. **Környezet naplózása** — Python-, torch-, torchvision-, pydicom-, sklearn-verziók és a GPU
   adatai bekerülnek a logba.
2. **`train.csv` ellenőrzése** — a `StudyInstanceUID` egyedi-e, van-e üres azonosító, és a 12
   célváltozó-oszlop jelen van-e a projekt sorrendjében. Az azonosítókat **stringként** olvassa,
   soha nem float-ként.
3. **`train_series.csv` ellenőrzése** — duplikált `(study, series)` kulcsok; olyan sorozat,
   amely több study-hoz tartozik (a series→study kapcsolatnak sok-az-egyhez kell lennie);
   a `Anatomical_Plane` váratlan értékei; a `Fluid_Sensitive`/`Fat_Suppression` oszlopok
   számszerűsége.
4. **Kereszt-ellenőrzés a `train.csv`-vel** — olyan study a sorozatlistában, amely nincs a
   `train.csv`-ben (hiba), és olyan study a `train.csv`-ben, amelyhez nincs sorozat (hiba).
5. **Lemez-leltár** — végigjárja a `train_series/<Study>/<Series>/` könyvtárakat, és összeveti
   a CSV-vel: hiányzó könyvtár (hiba), CSV-ben nem szereplő könyvtár (figyelmeztetés), üres
   könyvtár (figyelmeztetés).
6. **Opcionális címke-exportok** — mindegyikről megmondja, hogy nincs beállítva, nem létezik,
   vagy hiányoznak belőle mezők; a hiányzó fájl **mindig a várt oszloplistával együtt** jelenik
   meg, hogy ne kelljen találgatni.
7. **Címketábla felépítése és a lefedettség kiszámítása**, majd a *readiness gate* futtatása.

### Mit ír

A `work/schema/` könyvtárba:

| Fájl | Tartalom |
| --- | --- |
| `schema_findings.csv` | minden megállapítás: `level` (info/warning/error), `code`, `message`, `count` |
| `schema_summary.json` | darabszámok: study-k, sorozatok, síkonkénti eloszlás, lefedettség |
| `labels_table.csv` | study-nként a 12 `y::<cél>`, `w::<cél>`, `kind::<cél>` oszlop |
| `labels_counts.csv` | célváltozónként pozitív/negatív/borderline/bizonytalan/ismeretlen darabszámok |
| `readiness.json` | átment-e a readiness gate, és ha nem, miért |

Ha valamelyik ellenőrzés hibát talál, további részletező táblákat is kiír, például
`schema_orphan_series_studies.csv`, `schema_studies_without_series.csv`,
`schema_series_missing_on_disk.csv`, `schema_series_to_many_studies.csv`,
`schema_empty_series_dirs.csv`, `schema_series_not_in_csv.csv`.

### Kilépési kódok

- `0` — nincs hiba
- `2` — sémahiba (a `--keep-going` átlépteti a címke-ellenőrzésre)
- `3` — a címketábla nem volt felépíthető

### A readiness gate

Ez dönti el, szabad-e valódi fold-tanítást indítani. Két feltétel:

- legalább `labels.min_labeled_studies` (alapból 500) study-nak legyen felügyelete;
- legyen olyan célváltozó, amelynek van legalább `labels.min_positives_per_target` (alapból 10)
  ismert pozitív **és** negatív esete.

**A jelenlegi exporton ez szándékosan elbukik**, mert a `train.csv` csak 58 study-ra tartalmaz
címkét a 4407-ből. Ez nem hiba, hanem a beépített védelem: egy 58 study-s export elég egy
egyértelműen megjelölt füstteszthez, de nem elég hiteles CV-eredményhez.

### Futásidő

A teljes exporton kb. 4 másodperc (a lemez-leltár a legdrágább része).

---

## `build-manifest`

```powershell
python -m knee_mri.cli build-manifest [--limit N] [--studies fajl.txt]
```

Ez a pipeline legösszetettebb lépése. A teljes exporton kb. **75 perc** 8 worker mellett
(HDD-ről olvasva).

### 1. Fejlécek beolvasása

Sorozatkönyvtáranként minden fájl fejlécét beolvassa (`stop_before_pixels=True`), és kigyűjti:
`ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`, mátrixméret, `EchoTime`,
`EchoNumbers`, `TemporalPositionIdentifier`, `ImageType`, szekvencia-metaadatok, `PatientID`,
`Laterality`, transfer syntax.

**Multiframe objektumok**: ha `NumberOfFrames > 1`, a valódi képkockánkénti geometriát bontja
ki a `PerFrameFunctionalGroupsSequence`-ből. Ha ez nem elérhető, a volume
`multiframe_unsupported` flaget kap — **soha nem számolódik egyetlen szeletnek**.

### 2. Akvizíciós csoportokra bontás

Egy `SeriesInstanceUID` alatt gyakran több, egymással nem összefésülhető stack van (más echo,
más időpont, más orientáció). A kód ezeket **külön volume-jelöltekre** bontja a következő kulcs
szerint: kerekített `ImageOrientationPatient` + mátrixméret + `EchoTime` + `EchoNumbers` +
`TemporalPositionIdentifier` + `ImageType` harmadik eleme.

> A teljes exporton 24371 sorozatból **24372 volume-jelölt** lett — vagyis egy sorozat valóban
> két akvizícióra bomlott. Ezért nem sorozat-, hanem volume-szinten dolgozik a pipeline.

### 3. Geometriai rendezés

- Ellenőrzi, hogy a szeletek orientációja kompatibilis-e (5°-os tűrés) → különben
  `inconsistent_orientation`.
- Közös normálvektort képez a sorirány és oszlopirány keresztszorzatából.
- Minden `ImagePositionPatient`-et rávetít a normálra, és **a vetület szerint rendez** —
  soha nem fájlnév vagy `InstanceNumber` szerint.
- A szomszédos vetületek különbségéből számolja a tényleges szeletközt. **A `SliceThickness`-t
  soha nem használja helyette.**
- Jelöli: `duplicate_positions`, `irregular_spacing`, `large_gap`, `mixed_matrix_size`,
  `short_stack`.

### 4. Sík meghatározása

A normálvektor és a három fő tengely szöge alapján: ±x → sagittal, ±y → coronal, ±z → axial.
Ha a szög nagyobb `manifest.obliquity_tolerance_deg`-nél (35°), vagy a két legjobb tengely
közel azonos, `ambiguous_plane` flaget kap.

### 5. Kanonikus síkbeli orientáció

Meghatározza, milyen transzponálás és tükrözés viszi a sorozatot egységes megjelenítési
irányba (pl. sagittalnál: sorok lefelé = inferior, oszlopok = anterior felé). **Ez kizárólag
tömb-átrendezés, nem fizikai újramintavételezés** — a ferde adat nem lesz csendben
"kiegyenesítve", csak megkapja az `oblique_inplane` flaget.

### 6. Dekódolási próba

Ha `manifest.decode_probe` igaz (alapértelmezés), sorozatonként megpróbálja dekódolni a középső
szeletet. Sikertelenség esetén `decode_failed` flag és a hibaüzenet kerül a manifestbe.

### 7. Sorozatválasztás (slotonként egy)

Síkonként (sagittal/coronal/axial) egy volume-ot választ, átlátható pontszám alapján:

| Komponens | Súly | Jelentés |
| --- | --- | --- |
| `fluid_sensitive` | 3.0 | a `train_series.csv` jelzése |
| `fat_suppression` | 0.5 | ugyanonnan |
| `te_pd_t2` | 2.0 | `EchoTime` alapú PD/T2-valószínűség (≥60 ms → 1.0; ≥25 → 0.85; ≥15 → 0.4; alatta 0) |
| `slice_count` | 1.0 | szeletszám a kívánt centrumszámhoz mérve |
| `inplane_resolution` | 0.5 | finomabb pixelméret jobb |
| `plane_confidence` | 1.0 | mennyire tengelyre esik a normál |
| `quality_penalty` | −2.0 | minőségi flagenként levonás |

A lokalizátorokat a leírás mintái alapján kizárja (`localiz`, `scout`, `survey`, `3-pl`, stb.).
Holtverseny esetén a `SeriesInstanceUID` szerint dönt, így a választás **determinisztikus és
azonos tanításkor és validáláskor**.

> Fontos: a `SeriesDescription` önmagában nem azonosítja megbízhatóan a szekvenciát — ezért csak
> egy jel a több közül. A `Fluid_Sensitive` és `Fat_Suppression` ezen az exporton **minden sorban
> azonos**, tehát egy jelet hordoznak, nem kettőt.

### 8. PatientID-audit

Study-nként összegyűjti a `PatientID` értékeket, és jelöli: hiányzó, study-n belül inkonzisztens,
sok study-ra újrahasznált (site-reuse), illetve hány study tartozik az adott azonosítóhoz.

### Mit ír

A `work/manifest/` könyvtárba:

| Fájl | Tartalom |
| --- | --- |
| `manifest.csv` | volume-jelöltenként egy sor, 43 oszlop: sík, szög, szeletszám, pixelméret, szeletköz, `inplane_ops`, normálvektor, szekvencia-metaadatok, `patient_id`, transfer syntax, `decode_ok`, `quality_flags`, `usable` |
| `manifest_meta.json` | verzió, darabszámok, a használt paraméterek |
| `series_selection.csv` | `(study, slot)`-onként egy sor: kiválasztott `volume_id`, `score`, `runner_up_score`, `n_candidates`, szöveges `reason`, `quality_flags`, `path` |
| `series_selection_summary.json` | hány study kapott mindhárom slotot, slotonkénti darabszám |
| `patient_audit.csv` | study-nkénti PatientID-audit |

### Mikor áll meg hibával

Ha a volume-ok kevesebb mint fele dekódolható, `RuntimeError`-t dob azzal az üzenettel, hogy ez
rendszerszintű dekóder-probléma (hiányzó `pylibjpeg`/`gdcm` plugin tömörített transfer
syntaxhoz), nem néhány sérült sorozat. Így nem lehet véletlenül üres bemenetekből álló
adathalmazon tanítani.

---

## `select-series`

```powershell
python -m knee_mri.cli select-series
```

Csak a fenti 7. lépést futtatja újra a **meglévő** `manifest.csv`-n, és felülírja a
`series_selection.csv`-t. Akkor hasznos, ha a `selection.weights` súlyokon állítasz, és nem
akarod újra végigolvasni a DICOM-fejléceket (ami 75 perc lenne).

---

## `build-cache`

```powershell
python -m knee_mri.cli build-cache [--limit-studies N] [--workers N]
```

### Mit csinál

A kiválasztott sorozatokat egyetlen, verziózott, determinisztikus úton dolgozza fel, és **teljes
sorozatokat** ment ki — nem egy véletlenül kiválasztott bag-et. Így minden epoch más mintát vehet
ugyanabból a cache-elt pixelhalmazból.

Sorozatonként:

1. **Dekódolás** — pydicom, `apply_modality_lut` (rescale slope/intercept). A
   `PixelPaddingValue`-val egyező pixelek NaN-ra állnak, hogy kimaradjanak az intenzitás-
   statisztikából. `MONOCHROME1` esetén **egyszeri, explicit** invertálás.
2. **Kanonikus síkbeli orientáció** — a manifestben meghatározott transzponálás/tükrözés
   alkalmazása (a sor- és oszloptávolság ennek megfelelően cserélődik).
3. **Fizikai kivágás** — rögzített `data.fov_mm` (alapból 150 mm) oldalú négyzet. A középpont
   `data.crop_center` szerint:
   - `foreground` (alapértelmezés): a középső szeletharmad intenzitás-súlyozott súlypontja,
     a kép 25–75%-os sávjára szorítva, hogy egy kiugró érték ne rántsa el a kivágást;
   - `geometric`: a kép mértani közepe.
   A mm→pixel átváltás **soronként és oszloponként külön** történik, tehát az anizotróp
   pixelméret helyesen kezelt. A mátrixon kívülre eső rész NaN-nal töltődik ("nem felvett"),
   és a kitöltési arány bekerül a metaadatba.
4. **Robusztus intenzitás-skálázás** — a sorozat saját előterének p1/p99 percentilisére vág és
   [0,1]-re skáláz. Előtér = a p99 tizedénél nagyobb véges értékek; ha ez túl kevés, minden
   véges értékre vált (`weak_foreground` flag). Konstans vagy elfajult kép esetén nullákat ad
   vissza `constant_image` flaggel — nem oszt nullával.
5. **Négyzetes újramintavételezés** — antialiasolt bilineáris interpoláció `data.image_size`-ra.
   Mivel a kivágás már fizikailag négyzetes, ez nem torzítja az anatómiát. **ImageNet
   center-crop nem történik utána** — az levágna anatómiát.

### Mit ír

A `<cache_dir>/prep_v1_<hash>/` könyvtárba, ahol a `<hash>` az összes pixelt befolyásoló
config-mezőből képződik:

| Fájl | Tartalom |
| --- | --- |
| `<StudyUID>/<slot>.npz` | `image` tömb `[Z, méret, méret]` float16-ban, plusz JSON metaadat |
| `cache_meta.json` | az előfeldolgozás aláírása (mely beállításokkal készült) |
| `cache_report.csv` | `(study, slot)`-onként: `ok` / `cached` / `failed` és a hibaüzenet |

Az `.npz` metaadata tartalmazza: `volume_id`, sík, szeletszám, sor-/oszlop-/szelettávolság,
`output_mm_per_px`, `inplane_ops`, `crop_pad_fraction`, a használt p1/p99 értékeket,
minőségi flageket, az előfeldolgozás hash-ét, a forrásfájlok ujjlenyomatát, és a `gap_ok`
tömböt (szomszédonként: átléphető-e a hézag).

**Címke és riportszöveg soha nem kerül a képcache-be.**

### Cache-érvényesség

Minden belépő hordozza a config-hash-t és a forrásfájlok ujjlenyomatát. Ha bármelyik eltér,
az adott belépő elavultnak minősül és újraépül. Az írás atomi (ideiglenes fájl + átnevezés),
tehát megszakított futás nem hagy félkész fájlt. Más `image_size` vagy `fov_mm` **külön cache-
könyvtárat** kap, nem írja felül a meglévőt.

### Mikor áll meg hibával

Ha a belépők több mint 20%-a sikertelen, `RuntimeError` — ez rendszerszintű probléma, nem
néhány sérült sorozat.

### Ismert adathiba: csonka DICOM fájlok

A teljes, 13 221 sorozatos cache-építés 2 belépőn bukott el. Mindkettő **fizikailag csonka
DICOM fájl** a forrásadatban — nem kódhiba, és nem is dekóder-probléma:

| Study / slot | Sorozat | `PixelData` mérete | Várt | A sérült szelet |
| --- | --- | --- | --- | --- |
| `…34685905…83035` / axial | `…39396636…28348` (36 szelet, 640×640) | 409 600 B | 819 200 B | `InstanceNumber=36` (utolsó) |
| `…37833587…95680` / coronal | `…31160785…15703` (38 szelet, 416×416) | 173 056 B | 346 112 B | `InstanceNumber=1` (első) |

Mindkét fájl Explicit VR Little Endian (`1.2.840.10008.1.2.1`), `BitsAllocated=16`,
`SamplesPerPixel=1`, egyetlen frame — tehát se tömörítés, se hibás transfer syntax, se hibás
0028-as csoportérték nincs bennük. A `PixelData` elem pontosan **feleannyi bájtot** tartalmaz,
mint amennyit a `Rows × Columns × 2` megkövetel: a fájlok félbevágva kerültek a diszkre. A
sorozat összes többi szelete ép. A pydicom hibaüzenete (`…the transfer syntax may be
incorrect`) emiatt félrevezető: a transfer syntax rendben van, a fájl rövid.

**Miért esik ki a teljes kötet.** A dekódolás szigorú: az első szelethiba felszáll, és az egész
`(study, slot)` belépő `failed` lesz. Ez szándékos — a hiányzó pixeleket nem pótoljuk csendben
nullákkal.

**Miért nem szúrta ki a `build-manifest`.** A dekódolási próba (6. lépés) csak a **középső**
szeletet dekódolja, a két sérült fájl viszont mindkét esetben szélső szelet. Ezért kapta meg
mindkét sorozat a `usable=True` jelölést.

**Hatás: elhanyagolható.** 13 221-ből 2 belépő (0,015%), és egyetlen study sem vész el, csak
egy-egy sík hiányzik belőlük:

- `…34685905…83035`: sagittal + coronal megvan, **axial hiányzik** (train_pool, fold 2)
- `…37833587…95680`: sagittal + axial megvan, **coronal hiányzik** (train_pool, fold 1)

A dataset ezt natívan kezeli: hiányzó cache-fájl esetén az adott slot `present` maszkja nulla
marad, és a tanítás a maradék két síkkal megy tovább — pont erre való a slot-maszkolás. A 20%-os
cache-kapu meg sem közelíti a küszöböt, és a lefedettségi kapu sem szólal meg, mert egyik study
sem maradt mind a három slot nélkül.

**Lehetséges enyhítés, jelenleg nincs implementálva.** A dekódolás kaphatna egy opciót, ami a
dekódolhatatlan szeletet kihagyja a stackből (nem nullákkal tölti ki), flageli a kötetet a
kihagyott szeletek számával, és csak akkor bukik el, ha a megmaradt szeletszám a
`manifest.min_slices` alá esik. Mellé a dekódolási próba a középső helyett az első/középső/utolsó
szeletet nézhetné, hogy az ilyen hiba már a manifest-fázisban kiderüljön. Mivel mindkét sérült
fájl szélső szelet, ezzel a két slot anatómiai veszteség nélkül visszanyerhető lenne.

### Méret és futásidő

Mérve: **kb. 12 MB / study** (3 slot, 224 px, float16). A teljes 4407 study-ra ez
**nagyjából 50–55 GB**. Az 58 study-s ellenőrző cache 713 MB. Érdemes gyors lemezre tenni a
`paths.cache_dir`-t.

---

## `qc`

```powershell
python -m knee_mri.cli qc [--n-studies N]
```

### Mit csinál

Study-nként és slotonként hat panelt rajzol:

1. az **eredeti** (kivágás előtti) középső szelet, rárajzolva a fizikai kivágás sárga kerete,
   a mátrixmérettel és a pixelmérettel;
2–4. a **feldolgozott** első, középső és utolsó szelet;
5. a **valódi triplet**, amit a modell megkap (három szomszédos szelet egymás mellett);
6. ugyanaz **RGB-ként** összerakva — így a csatornák közti eltolódás szemmel látható.

A study-kat szándékosan vegyesen választja: hiányos slotú, minőségi flaggel megjelölt, rövid
stackű, ferde/ambiguous síkú, és normál esetek.

Emellett numerikus ellenőrzést is futtat, ami kép nélkül is működik: méret, végesség,
értéktartomány, konstans-e a kép, kitöltési arány.

### Mit ír

A `work/qc/` könyvtárba: `qc_<studyvég>.png` képek, `qc_numeric.csv`, `qc_summary.json`.

A `qc_summary.json` `numeric_ok` mezője **csak a ténylegesen cache-elt slotokat** veszi
figyelembe — egy jogosan hiányzó slot nem rontja el a verdiktet, azt a `n_slots_missing` külön
számolja.

Ha a képek nem állíthatók elő, a numerikus ellenőrzés akkor is lefut, és a riport azt írja,
hogy vizuális ellenőrzés nem történt — nem tesz úgy, mintha megtörtént volna.

---

## `make-splits`

```powershell
python -m knee_mri.cli make-splits
```

### 1. Csoportosítási kulcs eldöntése

A `PatientID`-t **nem fogadja el vakon**. Az auditból ellenőrzi:

- hiányzik-e valahol,
- study-n belül inkonzisztens-e,
- konstans-e,
- site-szinten újrahasznált-e (`split.max_studies_per_patient_id` fölött),
- **egyedi-e minden study-ra**.

Az utolsó eset a fontos ezen az exporton: **mind a 4407 study külön `PatientID`-t kapott**,
egyetlen megosztott érték sincs. Ez study-szintű álnév, nem beteg-azonosító — a rá épített
csoportosítás pontosan ugyanaz lenne, mint a study-szintű bontás, miközben beteg-szintű
védelemnek *látszana*. A kód ezt felismeri, és a felosztást **study-szintűként jelenti**, két
explicit figyelmeztetéssel.

> Következmény, amit minden CV-szám mellé oda kell írni: jelenleg semmi nem garantálja, hogy
> ugyanannak a személynek két vizsgálata (két térd, kontroll) egy foldba kerül. A lokális CV
> emiatt optimista lehet.

### 2. Referencia-halmaz kivétele

Ha `split.holdout_reference` igaz (alapértelmezés), a radiológus által címkézett study-k és a
**teljes csoportjuk** kikerül a tanításból. Ezek később opcionális diagnosztikai auditként
használhatók, de **soha nem early stopping kritériumként** — a promptolás ezeken a riportokon
történt, tehát nem független arany standard.

### 3. Foldokba osztás

Mohó, több-címkés, csoport-elsődleges kiegyensúlyozás: a csoportokat a pozitív esetek száma
szerint csökkenő sorrendben veszi, és mindegyiket ahhoz a foldhoz adja, amelyik után a
célváltozónkénti pozitív-darabszámok szórása a legkisebb.

Ez azért saját implementáció, mert a `StratifiedGroupKFold` nem kezel több-címkés mátrixot — és
a hiányzó célértékeket **nem kódolja negatívnak** csak azért, hogy a rétegzés kényelmes legyen.

### 4. Duplikált riportok auditja

Normalizálja a riportszöveget (kisbetűs, összevont szóközök), hash-eli, és megnézi, hány study
osztozik ugyanazon a szövegen, illetve hány ilyen klaszter nyúlik át több foldon.

> Ezen az exporton **54 klaszter, 204 study**, a legnagyobb 37 study-val. A szövegek
> megvizsgálva: ezek rövid **"normál lelet" sablonok** (`Sin anomalías`, `ACL normal. MCL
> normal…`) különböző álnevek alatt — vagyis különböző egészséges térdek, nem egy beteg. Ezért
> a kód csak jelenti őket, **nem vonja össze**: az összevonás sok egészséges beteget dobna egy
> foldba, ami rosszabb lenne. Akkor érdemes újragondolni, ha egy klaszterben *specifikus,
> szokatlan* lelet szerepel.

### Mit ír

A `work/splits/` könyvtárba:

| Fájl | Tartalom |
| --- | --- |
| `splits.csv` | `StudyInstanceUID`, `group`, `group_source`, `role` (`train_pool` / `reference_holdout`), `fold`, `report_hash`, `n_studies_with_same_report` |
| `splits_meta.json` | verzió, seed, foldszám, csoportosítási forrás, **a figyelmeztetések szövege**, a PatientID-audit, foldonkénti darabszámok és célváltozónkénti pozitív-számok |

A jelenlegi állapot: 58 study `reference_holdout`-ban, a maradék három foldban 1450 / 1450 / 1449.

### Megjegyzés

Ha létező `splits.csv`-t írna felül, figyelmeztet — mert azzal a **kiértékelési referencia is
megváltozik**. A foldokat a tanítási kísérletek előtt kell befagyasztani.

---

## `train`

```powershell
python -m knee_mri.cli train --mode {synthetic|overfit|fold} [--n-studies N] [--epochs N]
                             [--name NÉV] [--allow-unready]
```

### A három mód

| Mód | Adat | Mire jó |
| --- | --- | --- |
| `synthetic` | generált képek | a teljes tanítási út ellenőrzése adat nélkül |
| `overfit` | 8–16 valódi study, augmentáció nélkül | **in-sample hibakeresés** — tud-e a modell egyáltalán illeszkedni |
| `fold` | egy teljes fold | a valódi futtatás |

Az `overfit` módban a tanító és a "validációs" halmaz **szándékosan azonos**. A kód ezt
figyelmeztetésben kimondja, és a `run_summary.json`-be is beleírja, hogy ezek a számok
in-sample hibakeresési eredmények, nem tarthatók held-out teljesítménynek.

### Indítás előtti ellenőrzések (`fold` és `overfit`)

1. Betölti a `splits.csv`-t, és **kemény ellenőrzést** futtat: a fold csoport-diszjunkt-e,
   nincs-e átfedés a tanító és a validációs halmaz között.
2. Felépíti a címketáblát, és lefuttatja a readiness gate-et. Ha elbukik, `fold` módban
   megáll — kivéve `--allow-unready`, ekkor a `run_summary.json`-be bekerül, hogy az eredmény
   csak diagnosztikai.
3. **Lefedettségi kapu**: megnézi, mely study-knak nincs egyetlen használható sorozata sem.
   Alapból (`data.allow_all_missing_studies=false`) ez leállítja a futást. Ez azért fontos, mert
   ha a validációból csendben kiesnének ezek a study-k, a metrika a "túlélőkön" jobbnak látszana.
4. **Befagyasztja a kiértékelési referenciát** (`validation_reference.csv`) — a célértékeket, a
   maszkokat és a study-sorrendet — *mielőtt* bármilyen tanítási címke-politikát alkalmazna.
5. Kiírja a **célváltozónkénti és felosztásonkénti** címke-darabszámokat, és külön megnevezi
   azokat a célváltozókat, amelyekre nincs felügyelet, vagy csak egy osztály van jelen.

### A tanítási hurok

- **Optimalizáló**: AdamW, külön paramétercsoport az encodernek és a fejnek, saját tanulási
  rátával (`train.encoder_lr`, `train.head_lr`).
- **Ütemező**: `train.warmup_epochs` epochnyi lineáris felfutás, utána koszinusz-lecsengés.
  **Optimizer-lépésenként** lép, nem batch-enként.
- **Gradiens-akkumuláció**: `train.accumulation_steps` mikrobatch alkot egy ablakot. Minden
  mikrobatch vesztesége az **ablak valódi méretével** osztódik — így a rövidebb utolsó ablak is
  helyesen skálázódik.
- **Üres felügyelet**: ha egy egész ablakban nincs egyetlen felügyelt cella sem, az optimizer- és
  ütemező-lépés **kimarad**, és ezt naplózza. Egyetlen mikrobatch üres felügyelete
  differenciálható nullát ad.
- **Gradiens-vágás**: `train.grad_clip` szerint, **a skálázás visszavonása után**.
- **AMP**: automatikus választás — bf16, ha a GPU támogatja (nincs szükség GradScalerre),
  különben fp16 + GradScaler, CPU-n float32.
- **BatchNorm**: alapból befagyasztott futó statisztikák az előtanított encoderben (az affin
  paraméterek taníthatók maradnak), és ez a politika minden `model.train()` után újra
  érvényesül.

### Validáció epochonként

`model.eval()` és `torch.inference_mode()`. Az előrejelzéseket **a teljes foldra összegyűjti**,
és csak utána számol rangsor-metrikát — soha nem átlagol batch-enkénti AUC-t. Ha bármelyik
validációs study-ra nem születik előrejelzés, hibát dob.

### Checkpoint-választás és early stopping

Kizárólag a **befagyasztott lokális macro ROC-AUC** alapján. Ha egyetlen célváltozónak sincs
definiált AUC-ja, a kód ezt hibaként naplózza, és **nem választ checkpointot NaN alapján** —
és nem is helyettesíti a radiológus-referencia auditjával. Két egymást követő ilyen epoch után
leáll azzal, hogy előbb a validációs címke-lefedettséget kell rendbe tenni.

### Mit ír

A `work/runs/<mód>_fold<N>_<időbélyeg>/` könyvtárba:

| Fájl | Tartalom |
| --- | --- |
| `run.log` | a futás teljes naplója |
| `environment.json` | csomagverziók, GPU-adatok |
| `config.yaml` | a ténylegesen használt konfiguráció |
| `coverage.json` | hány study-nak hiányzott minden slotja |
| `label_counts_per_split.csv` | célváltozónkénti darabszámok tanító/validációs bontásban |
| `validation_reference.csv` | a befagyasztott bináris referencia és érvényességi maszk |
| `history.csv` | epochonként: veszteség, üres mikrobatch-ek, optimizer-lépések, tanulási ráták, idő, study/s, **csúcs GPU-memória**, validációs macro ROC-AUC, definiált célváltozók száma, macro AP, macro F1 |
| `metrics_per_class.csv` | célváltozónként: `n_pos`, `n_neg`, `n_unresolved`, `coverage`, `roc_auc`, `average_precision`, `prevalence`, `ap_degenerate`, `tp/fp/tn/fn`, `precision`, `recall`, `specificity`, `f1` |
| `validation_predictions.csv` | `(study, célváltozó)`-nként: fold, pontszám, referencia, érvényesség, epoch, mód, checkpoint, címkeforrás |
| `val_summary.json` | a legjobb epoch összefoglalója |
| `run_summary.json` | mód, fold, legjobb pontszám, eszköz, provenance, és a módhoz tartozó figyelmeztetés |
| `best.pt`, `last.pt` | checkpointok |

A checkpoint tartalmazza a modell-, optimizer-, ütemező- és scaler-állapotot, az epochot, a
legjobb pontszámot, az RNG-állapotot, a **célváltozó-sorrendet**, a konfigurációt, valamint a
címke-, split- és előfeldolgozás-verziókat. Újratöltéskor ezeket ellenőrzi.

### Folytatás

```powershell
python -m knee_mri.cli train --mode fold --set train.resume=work/runs/<futás>/last.pt
```

A folytatás **epoch-határon pontos**, és a következő epochtól folytatódik. Epoch közbeni
folytatás nincs implementálva, és a kód ezt nem is állítja.

---

## `evaluate`

```powershell
python -m knee_mri.cli evaluate --checkpoint <út>/best.pt
                                [--partition {validation|reference_holdout}] [--out-dir ÚT]
```

### Mit csinál

1. Betölti a checkpointot, és **ellenőrzi a célváltozó-sorrendet** — eltérés esetén hibát dob.
   Az architektúrát előtanított súlyok újraletöltése nélkül építi fel.
2. Figyelmeztet, ha a checkpoint `image_size`, `series_slots`, `centers_per_series` vagy
   normalizálási beállítása eltér a jelenlegitől.
3. A `validation` partíció a fold held-out study-it pontozza az extrakcióból származó
   befagyasztott referencián. A `reference_holdout` a radiológus-referencián — ez
   **kizárólag diagnosztikai jelzés**, nem arany standard, és az `summary.json` `note` mezője ezt
   ki is mondja.
4. Determinisztikus inferencia: `eval()`, `inference_mode()`, augmentáció és TTA nélkül.

### Mit ír

Az `eval_<partíció>/` alkönyvtárba: `predictions.csv`, `metrics_per_class.csv`, `summary.json`,
`reference.csv`. A `summary.json` összeveti a checkpoint és a jelenlegi cache előfeldolgozás-
hash-ét is.

---

## `report`

```powershell
python -m knee_mri.cli report <futás-könyvtár> [--qc-dir ÚT]
```

Egyetlen offline HTML fájlt állít elő a futás már meglévő melléktermékeiből:
`learning_curves.png` (veszteség, validációs AUC, tanulási ráták), `per_class_metrics.png`
(ROC-AUC oszlopdiagram — a definiálatlan célváltozók **szürkén, `NA` felirattal**; AP a
prevalenciával összevetve), a metrikatáblák, a QC galéria és a nagy magabiztosságú eltérések
listája.

Az eltérés-lista fejlécében ott van, hogy ez **fejlesztési segédeszköz**: a esetek átnézése
megváltoztatja a referenciát, ezért ki kell maradnia bármilyen lezárt végső kiértékelésből, és
egy javított címketábla nem bizonyítéka annak, hogy az őt előállító modell javult.

---

## `merge-oof`

```powershell
python -m knee_mri.cli merge-oof <fold0>/validation_predictions.csv <fold1>/... <fold2>/...
```

Összefűzi a foldonkénti előrejelzéseket `oof_predictions.csv`-be, és kiszámolja a teljes OOF
metrikákat.

**Három ellenőrzésen kell átmennie, különben visszautasítja a műveletet:**

1. minden várt fold jelen van (egyetlen held-out fold **nem** OOF-lefedettség);
2. nincs duplikált `(study, célváltozó)` sor — vagyis a foldok valóban diszjunktak;
3. a tanítási halmaz minden study-jára van előrejelzés.

Kimenet: `oof_predictions.csv`, `oof_metrics_per_class.csv`, `oof_summary.json`.

---

## Mit jelentenek a metrikák

- **ROC-AUC** — célváltozónként, a befagyasztott bináris referencián. Ha csak egy osztály van
  jelen, **`NA`, soha nem 0.5**.
- **AP (average precision)** — a nevén nevezve, a prevalenciával együtt; elfajult támogatás
  esetén megjelölve, nem csendben beleátlagolva.
- **Precision / recall / specificity / F1** — rögzített 0.5-ös küszöbön, a konfúziós
  darabszámokkal. Definiálatlan nevező esetén `NA` (pl. ha egyetlen pozitív előrejelzés sincs,
  a precision `NA`).
- **Macro értékek** — mindig `n_definiált / 12` formában jelentve. Ez a **lokális, értékelhető
  célváltozókra vett** macro; nem azonos semmilyen hivatalos verseny-metrikával.

Ha a validációs címkék között lágy (soft) célértékek vannak, azok **kimaradnak a bináris
referenciából** — a küszöbölésük kitalált referenciát hozna létre. Helyettük a kompatibilis
súlyozott veszteség jelentendő.

---

## Leállító kapuk összefoglalva

Ezek szándékosan állítják meg a futást:

| Kapu | Mikor |
| --- | --- |
| readiness gate | a címke-lefedettség vagy az osztálytámogatás a küszöb alatt |
| lefedettségi kapu | van olyan study, amelynek egyetlen slotjában sincs használható sorozat |
| dekóder-kapu | a volume-ok több mint fele nem dekódolható |
| cache-kapu | a belépők több mint 20%-a sikertelen |
| AUC-kapu | két egymást követő epochban egyetlen definiált validációs ROC-AUC sincs |
| felosztás-kapu | nem csoport-diszjunkt fold, vagy tanító/validációs átfedés |
| checkpoint-kapu | a checkpoint célváltozó-sorrendje eltér a projekt sorrendjétől |
| OOF-kapu | hiányos vagy nem diszjunkt fold-lefedettség |

---

## A jelenlegi állapot

| Lépés | Állapot |
| --- | --- |
| `validate-schema` | lefutott a teljes exportra, sémahiba nincs |
| `build-manifest` | lefutott: 24372 volume-jelölt, mind dekódolható, mind a 4407 study megkapta mindhárom slotot |
| `build-cache` | lefutott a **teljes** exportra: 13 221 sorozatból 13 219 `ok`, 2 `failed` — mindkettő csonka forrás-DICOM, lásd [Ismert adathiba](#ismert-adathiba-csonka-dicom-fájlok) |
| `qc` | lefutott a cache-elt study-kra, a képek vizuálisan ellenőrizve |
| `make-splits` | lefutott: 58 holdout + 1450/1450/1449 |
| `train --mode synthetic` | lefutott GPU-n, bf16 AMP |
| `train --mode overfit` | lefutott valódi képeken (8 study): veszteség 0.67 → 0.44, in-sample AUC → 1.0 |
| `train --mode fold` | **blokkolva** a readiness gate által, amíg nincs valódi extrakciós export |

Mért teljesítmény RTX 5080-on, alapbeállításokkal (224 px, 24 centrum, 3 slot,
`microbatch_studies=1`): **3.3 GB csúcs-VRAM**, **kb. 7 study/s**, azaz nagyjából
**11 perc / epoch** 4407 study-ra.
