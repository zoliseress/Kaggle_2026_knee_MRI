# RSNA knee MRI — 2.5D MIL (EfficientNet-B0 baseline)

Study-level, multi-label knee MRI classification (12 targets) from DICOM series, using a
2D slice encoder (**torchvision EfficientNet-B0** by default; EfficientNetV2-S, RadImageNet
ResNet-50 and DINOv2 ViT-S/14 as alternatives) on **2.5D adjacent-slice triplets** with
**masked multiple-instance pooling** over series slots.

The model consumes images and explicit input-validity masks. No LLM is called during
image training or validation; report-derived labels enter only through the label export
files described below.

## Scope

Implemented on top of the baseline, each behind its own switch and off by default:
spatial avg/max/attention pooling of the feature map, per-target attention, side
(medial/lateral) pooling, a canonical laterality frame, weight EMA, extra series slots
(`coronal_t1`, `sagittal_nonfs`) and the alternative encoders. **Not** implemented: 3D
networks, segmentation, anatomical locators, pseudo-labelling / distillation training
(only its leakage guard is in place, see *Keeping the reference out of every training
run*), TTA. Fold checkpoints are averaged only at inference. Externally supplied soft
targets are supported by the loss/data contract, but no soft value is ever invented.

## Install

```powershell
conda activate kaggle_2026
pip install -r ..\requirements.txt
```

`torch`/`torchvision` must match your CUDA runtime — install them with the index URL
that PyTorch documents for *your* driver rather than a generic command. Verified here
with torch 2.13.0+cu130, torchvision 0.28.0+cu130, pydicom 3.0.2, scikit-learn 1.9.1,
numpy 2.4.6, pandas 3.0.5 on Python 3.11 / Windows, RTX 5080 (16 GB).
Every run writes its own `environment.json` with these versions and the GPU information.

Compressed DICOM transfer syntaxes need a decoder plugin (`pylibjpeg` + `pylibjpeg-libjpeg`,
or `gdcm`). The manifest step probe-decodes one slice per volume and **fails loudly** if a
large share of volumes cannot be decoded, instead of producing a dataset of missing inputs.
The current export decodes with plain pydicom (Explicit VR Little Endian).

## Files outside the repository

A fresh clone holds the code, the configs, the runner scripts and the 58-study reference
(`notebooks/train_labeled_58_reference.csv`). Everything below has to be supplied or
regenerated. Paths are the ones in `config.yaml` / the runner scripts; `<data>` is
`paths.data_root` (here `F:/Kaggle/data`).

**1. Competition data** — from the Kaggle competition page, unpacked under `<data>`:
`train.csv`, `train_series.csv`, `train_series/`, and for inference `test.csv`,
`test_series.csv`, `test_series/`.

**2. Project label tables** — not on Kaggle and not in git; they come out of the LLM report
labelling (`notebooks/02_label_train_reports_*.ipynb`) and the build scripts:

| file (in `<data>`) | how it is made | inputs it needs |
| --- | --- | --- |
| `train_v6.csv` | blend `0.5 · llm_labels_v2 + 0.5 · labels_llm_gpt56sol` | the two LLM label exports |
| `train_v7.csv` | `notebooks/build_ref208_train_v7.py`: `train_v6` minus the 50 new reference studies | `train_v6.csv`, `train_labeled_158_reference.csv`, `RSNA_Andrew 50.xlsx` |
| `train_v8.csv` *(default `paths.train_csv`)* | `notebooks/build_train_v8.py`: not-mentioned cells 0.25 → 0.07 | `train_v7.csv`, `F:/Kaggle/llm_labeling/output/labels_lixin_gpt56sol_V2/llm_labels_v2.csv`, `.../labels_lixin_gpt56sol/labels_llm_gpt56sol.csv` |

The build scripts have their paths hard-coded (`D = 'F:/Kaggle/data/'`). Older tables
(`train_v1`–`train_v5`) are needed only to reproduce old runs.

**3. Radiologist reference sets** — not in git; needed in **two** places:

| file | where | used by |
| --- | --- | --- |
| `train_labeled_158_reference.csv` | `<data>` and `notebooks/` | `paths.reference_csv`; `HOLDOUT=158`; input of the 208 build |
| `train_labeled_208_reference.csv` | `<data>` and `notebooks/` | `run_holdout.*` (`VAL_CSV`), and `paths.exclude_from_training_csv` — **`train.py` refuses to start when this file is missing** |

`build_ref208_train_v7.py` writes the 208 file to both places (the 158 plus the 0–4 scores of
`RSNA_Andrew 50.xlsx`, binarised: 0–1 → 0, 3–4 → 1, 2 / X → empty). There is no build script
for the 158 file in the repository. Setting `--set paths.exclude_from_training_csv=` turns
the leakage check off, but only do that when you know the training table holds no
reference study.

**4. Generated under `work/`** (gitignored, rebuilt by the pipeline commands):
`manifest/` (`build-manifest`, `build-laterality`), the cache (`build-cache`, at
`paths.cache_dir`), `splits/` (`make-splits`, or created by the runner scripts) and
`labels/frozen_reference_*.csv` (`freeze-reference`). `run_holdout.*` creates its frozen
reference itself. `config.yaml`, which is what `run_cv.*` uses, points at
`work/labels/frozen_reference_v3.csv`. That file is the train_v3 soft labels frozen over the
4407-study `work/splits/splits.csv`, and it has to exist before a CV run:

```powershell
python -m knee_mri.cli freeze-reference --set paths.train_csv=<data>/train_v3.csv `
    --set paths.splits_csv=work/splits/splits.csv --out work/labels/frozen_reference_v3.csv
```

**5. Pretrained weights.** EfficientNet (torchvision) and DINOv2 (`lvd142m`, Hugging Face)
download on first use into their caches. RadImageNet ResNet-50 does not: put `ResNet50.pt`
at `work/pretrained/RadImageNet_pytorch/ResNet50.pt` (see *Encoder backbones*).

**6. Kaggle inference** additionally needs the trained run directories (`best.pt` +
`config.yaml` each) and this `src/knee_mri` package uploaded as Kaggle datasets (see
*Step 7*).

## Inputs and contracts

| File | Role |
| --- | --- |
| `train.csv` | study metadata, report text, and — in this export — the 12 target columns |
| `train_v<N>.csv` | the same columns with LLM-derived (soft) labels; the one in use is `paths.train_csv` (currently `train_v8.csv`) |
| `train_series.csv` | study↔series map plus `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` |
| `train_series/<StudyUID>/<SeriesUID>/*.dcm` | image data |
| `labels_details.csv` *(optional, preferred)* | one row per (study, target) with `status` and `basis` |
| `labels_statuses.csv` *(optional)* | wide text statuses |
| `labels_predictions.csv` *(optional)* | wide numeric export; empty cell = unresolved |
| `labels_predictions_exclude_borderline.csv` *(optional)* | same, borderline also emptied |
| `train_labeled_158_reference.csv` *(optional)* | radiologist reference set (158 studies = the original 58 + 100) — audit only |
| `train_labeled_208_reference.csv` | the current evaluation set: the 158 plus 50 more radiologist-read studies; never a training study (`paths.exclude_from_training_csv`) |

Identifiers are read as strings, joins are key-based (never positional), and
`validate-schema` checks unique study ids, unique (study, series) keys, many-to-one
series→study, orphans in both directions, unexpected values, and labelled coverage.
Missing optional exports are reported with their expected field lists — the code never
guesses a column. Patient ids, source paths and report text are never used as features.

**Target order** (frozen; stored in every checkpoint and prediction file):

```
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture
```

### Current data status

The original `train.csv` carries numeric targets for only **58 of 4407 studies (1.3 %)**,
the same set as the first radiologist reference file; on it the readiness gate **fails full
training by design** and says why. Training therefore runs on the LLM-labelled
`train_v<N>.csv` tables, whose 12 target columns hold soft values in [0, 1]
(`labels.source=auto` picks them up when no other export is configured):

| table | content |
| --- | --- |
| `train_v1.csv` | binary labels |
| `train_v2.csv`, `train_v3.csv` | soft labels (QWen; QWen + GPT) |
| `train_v7.csv` | soft-label blend without the 208 reference studies |
| `train_v8.csv` *(default)* | `train_v7` with not-mentioned findings at 0.07 instead of 0.25 (Synovitis kept); **4199 studies** = 4407 − 208 |

The generator scripts are under `notebooks/` (`build_ref208_train_v7.py`,
`build_train_v8.py`). A wide or details export (`paths.labels_details_csv` etc.) can still
replace the training table's own columns.

## Label policy

| Extraction status | target | weight |
| --- | --- | --- |
| `positive` | 1 | 1 |
| `negative`, basis `explicit_absence` / `below_threshold` | 0 | 1 |
| `negative`, basis `borderline` | 0 | **0** (`labels.borderline_policy=exclude`, default) or 1 (`as_negative` ablation) |
| `uncertain` (conflict / insufficient detail / not assessed) | 0 (placeholder) | `labels.uncertain_weight` (default 0) |
| `not_mentioned` | 0 (placeholder) | `labels.unmentioned_weight` (default 0; 0.2 / 1.0 are explicit weak-negative experiments), per target via `labels.unmentioned_weight_per_target` (details source only) |
| failed extraction, missing row, empty numeric cell | 0 (placeholder) | **0 — unknown, never a silent negative** |

Placeholders are always finite: `NaN * 0` is not a safe way to mask a loss. A wide numeric
export cannot express borderline status, and lost status information is never
reconstructed from a binary value — to exclude borderline cases, use the matching export.
Soft (continuous) targets in [0, 1] are accepted only with `labels.allow_soft_targets=true`
and are kept separate from confidence weights. The BCE loss takes them as they are. They
enter the evaluation reference (`LabelTable.continuous_reference`, `freeze-reference`) and are
scored by the soft ROC-AUC (`metrics.soft_roc_auc`): a weighted concordance index where a
pair counts with weight (y_i − y_j)+. It equals ROC-AUC exactly on a binary reference.
`eval.selection_metric` (`macro_soft_auc` by default, or `macro_roc_auc`) selects the
checkpoint. ROC-AUC, AP and the threshold counts keep using the hard 0/1 cells only.

## Pipeline

All commands run from `src/`. Add `--set key.sub=value` to override any config entry.

> A step-by-step description of what each command does internally — inputs, outputs, decisions
> and failure conditions — is in **[`doc/pipeline_lepesek.md`](../doc/pipeline_lepesek.md)**
> (Hungarian).

Anything in `<angle brackets>` is a placeholder you must fill in — those lines are not
copy-pasteable as they stand.

### Step 0 — checks that need no data at all

Run these first. They require no DICOMs, no cache and no splits, so they tell you the code
works before you spend hours on the cache build.

```powershell
python -m knee_mri.cli selftest                                   # synthetic failure-mode checks
python -m knee_mri.cli train --mode synthetic --n-studies 16 --epochs 3
```

### Steps 1–5 — the real chain, in this order

Run them in this order. The `Needs` column states what each command actually requires, so you
can see which dependencies are hard (the command fails without them) and which are advisory:

| Step | Command | Needs |
| --- | --- | --- |
| 1 | `validate-schema` | the input CSVs only |
| 2a | `build-manifest` | the DICOM root — writes `series_selection.csv` **and** `patient_audit.csv` |
| 2a′ | `select-series` *(optional)* | the manifest — re-runs only the selection, e.g. after changing `data.series_slots` or `selection.*` |
| 2a″ | `build-laterality` *(optional)* | the manifest — only for `data.laterality_canonical=true` |
| 2b | `build-cache` | **the manifest** — fails without `series_selection.csv` |
| 3a | `qc`, `qc-edges` | **the cache** — otherwise every panel just says "not cached" |
| 3b | `make-splits` | soft: runs without `patient_audit.csv`, but then warns and falls back to study-level grouping |
| 4 | `train --mode overfit` | **the cache and `splits.csv`** |
| 5 | `train --mode fold` | **the cache and `splits.csv`**, and a passing label-readiness gate |

```powershell
# 1. Configure paths, validate schemas and label coverage
python -m knee_mri.cli validate-schema

# 2. Build the DICOM manifest (selection + patient audit), then the deterministic cache
python -m knee_mri.cli build-manifest
python -m knee_mri.cli build-cache

# 3. Inspect QC (gallery; how often the crop cuts anatomy per edge), then create the immutable folds
python -m knee_mri.cli qc --n-studies 12
python -m knee_mri.cli qc-edges --n-studies 500
python -m knee_mri.cli make-splits

# 4. Tiny in-sample overfit check on real images (debugging output, not performance)
python -m knee_mri.cli train --mode overfit --n-studies 8

# 5. Train fold 0, then build its report
python -m knee_mri.cli train --mode fold --set split.fold=0
python -m knee_mri.cli report work/runs/<dataset>/<run-name>
```

### Step 6 — after a fold has finished: pick what you need

**This is a menu, not a sequence.** These commands are alternatives; run only the one that
matches your situation.

*The run was interrupted and you want to continue it* (resumes at the next epoch boundary):

```powershell
python -m knee_mri.cli train --mode fold --set train.resume=work/runs/<dataset>/<run>/last.pt
```

*Score a saved checkpoint on its held-out fold:*

```powershell
python -m knee_mri.cli evaluate --checkpoint work/runs/<dataset>/<run>/best.pt
```

*Run the diagnostic audit against the radiologist reference set* — a development signal only,
never a gold benchmark and never a selection criterion:

```powershell
python -m knee_mri.cli evaluate --checkpoint work/runs/<dataset>/<run>/best.pt --partition reference_holdout
```

*Train the remaining folds* (one command per fold, each writing its own run directory):

```powershell
python -m knee_mri.cli train --mode fold --set split.fold=1
python -m knee_mri.cli train --mode fold --set split.fold=2
```

*Merge the out-of-fold predictions* — only once **every** fold above has finished. The command
refuses anything less than complete, verified-disjoint coverage:

```powershell
python -m knee_mri.cli merge-oof `
  work/runs/<dataset>/<run-fold0>/validation_predictions.csv `
  work/runs/<dataset>/<run-fold1>/validation_predictions.csv `
  work/runs/<dataset>/<run-fold2>/validation_predictions.csv
```

In day-to-day use steps 4–6 go through the runner scripts in the repository root:
`run_holdout.bat` / `run_holdout.sh` (one run on a fixed train/validation split) and
`run_cv.bat` / `run_cv.sh` (all folds, then `merge-oof`). Both take a run-name prefix as
first argument, pass every further argument to `train`, and continue an interrupted run with
`--resume <run name>` — see *Fixed hold-out instead of CV* below.

### Where this repository currently stands

All steps have been run on the full export. The cache was built for every selected series
(13 221 series, 13 219 `ok`; the 2 failures are truncated source DICOM files, see
[`doc/pipeline_lepesek.md`](../doc/pipeline_lepesek.md)). Training uses `train_v8.csv`, mostly
as a fixed hold-out against the 208 reference studies, and as a 3-fold CV
(`work/splits/cv3_train_v8`) for teacher runs. The working image size is **320 px**
(`--set data.image_size=320`, with its own cache); `config.yaml` still says 224.

`notebooks/03_knee_mri_efficientnet_walkthrough.ipynb` runs the same functions
step by step with prints, QC images and a tiny training run. It imports the modules; it
does not duplicate the implementation.

### Step 7 — inference on the competition test set

`notebooks/04_kaggle_test_online.ipynb` is the Kaggle submission notebook, and
`notebooks/04_kaggle_test_local.ipynb` is the same notebook wired to the local export for
a dry run. They differ only in the five paths of their first cell and in the package name
(`srcknee2` is what the Kaggle code dataset calls `knee_mri`); both are generated by
`python notebooks/build_04_notebook.py`, so edit the cells there, not in the `.ipynb`.

The notebook points
`paths.dicom_root` at `test_series/`, runs manifest → selection → cache → prediction in
study chunks (caching and deleting as it goes, so the working directory stays bounded on a
hidden test set of unknown size), averages the sigmoid outputs of the fold checkpoints and
writes `submission.csv`. It imports the same functions used in training and refuses to run
when a checkpoint's stored preprocessing hash differs from the one the notebook produces.

The manifest and cache workers run under the **spawn** start method
(`MP_START_METHOD`): the cache worker calls torch (`preprocess.resample_square` uses
`F.interpolate`), and forking it out of a process that already holds CUDA models and
torch's thread pools deadlocks on the child's first torch call. That is what a fork-based
Linux kernel hit while the fork-free manifest stage passed.

Paths are fixed settings in its first cell, not a search: `COMPETITION_DIR`
(`/kaggle/input/competitions/rsna-knee-abnormality-detection`), `CODE_DIR`,
`CHECKPOINT_GLOB`, `WORK_DIR` and `SUBMISSION_PATH`. Each is checked before anything runs
and names itself in the error when it is wrong. The `config.yaml` beside the first matched
checkpoint supplies the image parameters, so every run directory must be uploaded with its
own config.

Verified on the three public test studies (all three slots selected for each) and
cross-checked by re-predicting fold-0 validation studies through the notebook path: the
scores reproduce the training-time `validation_predictions.csv` to a maximum absolute
difference of 1.0e-3 (mean 2.9e-5), i.e. bf16 autocast noise.

## How the images are built

1. **Inventory.** Every series directory is split into geometrically coherent *volume
   candidates*: mixed echoes, time points and orientations become separate candidates
   rather than one scrambled stack. `ImageOrientationPatient` compatibility is checked, the
   common normal comes from the cross product, and slices are sorted by projected
   `ImagePositionPatient` — never by filename or `InstanceNumber`. Duplicate positions,
   irregular spacing and large gaps are flagged (centre-to-centre spacing is measured, not
   taken from `SliceThickness`). The plane comes from the normal with an explicit
   obliquity tolerance; ambiguous assignments are logged. Multiframe objects are expanded
   with their real per-frame geometry or flagged `multiframe_unsupported` — never counted
   as a single slice.
2. **Selection.** One series per slot (sagittal / coronal / axial), ranked by a transparent
   score: fluid-sensitive flag, fat suppression, TE-based PD/T2 likelihood, slice count,
   in-plane resolution, plane confidence, minus quality-flag penalties. Localizers are
   excluded by description pattern. Ties break on the series UID, so the choice is
   deterministic and identical for training and validation. The score components and the
   runner-up are written to `series_selection.csv`. Note: `SeriesDescription` alone does
   not reliably identify a sequence — it is one input among several, and
   `Fluid_Sensitive`/`Fat_Suppression` are identical in this export, so they carry one
   signal, not two. `selection.fluid_te_preference` decides between the fluid-sensitive
   echoes: `t2` (default) ranks TE ≥ 60 ms above PD-like 25–60 ms, `pd` the other way round
   (the radiologists' PDFS preference).

   **Extra slots.** `data.series_slots` may add a *filtered* slot `<plane>_<filter>` to the
   three plane slots. It takes the best volume of that plane that passes the filter and is
   not already the plane slot's own series:
   * `coronal_t1` — non fat-suppressed spin-echo T1 (0 < TR < 1000 ms, 0 < TE < 30 ms, no
     GR/IR, no inversion time; missing headers never pass);
   * `sagittal_nonfs` (any `<plane>_nonfs`) — non fat-suppressed, non fluid-sensitive
     spin-echo, ranked PD > T1 > T2 before the score; any other weighting is refused.

   A changed slot list needs its own selection csv and cache directory.
3. **Preprocessing** (one versioned deterministic path, used for train and validation):
   modality LUT applied via pydicom, `PixelPaddingValue` turned into NaN, MONOCHROME1
   inverted once explicitly; a canonical series-local in-plane orientation built from
   transposes and flips only (a pure array reordering — oblique data is *not* silently
   straightened, and `oblique_inplane` is flagged); a fixed **150 × 150 mm** physical crop
   around a foreground-derived centre (the scan centre is not assumed to be the knee
   centre), honouring anisotropic row/column spacing, padded where the FOV exceeds the
   matrix. `data.crop_center` picks the centre: `foreground` (default, area centroid of
   the foreground mask), `foreground_extent` (midpoint between the skin lines — the
   centroid is pulled towards the posterior musculature and pushed the patella off the
   anterior edge on sagittal images in about a third of the studies, `qc-edges`) or `geometric`;
   `tests2/crop_center_policy_comparison.py` compares them. Robust p1/p99 foreground intensity scaling to [0, 1] with documented fallbacks
   for empty foreground and constant images; antialiased bilinear resample to a square
   output. No ImageNet centre-crop is applied afterwards — that would cut away anatomy.
   The original through-plane order is preserved; no isotropic volume is fabricated.
4. **Cache.** Whole preprocessed *series* are cached (float16 npz + metadata), before any
   stochastic centre sampling — every epoch draws a fresh bag from the same cached pixels.
   Entries carry the preprocessing hash and a source fingerprint; stale entries are
   detected and rebuilt, writes are atomic. Labels and report text never enter the cache.
5. **Bags.** For each slot the index range is divided into `S` bins: one random centre per
   bin while training, deterministic bin midpoints in validation. A triplet is
   `[i-1, i, i+1]` of the *original* stack, clipped at the boundaries; where a physical gap
   was flagged, the centre slice is repeated instead of crossing the gap. Stacks shorter
   than `S` use each slice once and pad the rest as invalid. Missing slots get padded
   inputs and false masks.

Augmentation draws **one** spatial transform (rotation / translation / scale) per series
and applies it identically to every slice and channel, plus mild series-consistent
intensity jitter. No laterality flips, no independent per-channel colour jitter.
Validation has no augmentation and no TTA.

**Laterality frame** (`data.laterality_canonical`, default off). The canonical in-plane
orientation follows patient coordinates, so a right knee's medial compartment sits on the
opposite image side from a left knee's. `build-laterality` decides the side per study —
from the DICOM `Laterality` / `ImageLaterality` tag (~50 % of the studies), otherwise from
the median x of the coronal/axial image centres (agrees with the tag in 97.4 % of the
tagged studies; within `data.laterality_min_offset_mm` of the midline it stays unresolved
and is not flipped). With the switch on, right knees are mirrored on coronal/axial and
sagittal slices are ordered towards lateral when a bag is read; the cache is not affected.

`augment.device` chooses **where** that transform is applied, not what it is:

| value | what happens |
|---|---|
| `cpu` | the DataLoader worker augments, normalises and masks the bag |
| `cuda` | the worker emits the raw `[0, 1]` bag plus an `augment_params` `[P, 7]` row, and the training loop calls `augment_batch` on the device |
| `auto` | *(default)* `cuda` when one is visible, otherwise `cpu` |

The parameters are drawn in the worker either way, from the same
`(seed, epoch, index, slot)` generator, so a given study gets the same transform in a
given epoch regardless of placement. On the device path the whole `[S, K]` stack of a
series rides through `grid_sample` as channels, so it costs one sampling grid instead of
`S × K` identical ones.

**On equivalence.** Run on the same device with the noise field off, the batched call is
*bit-identical* to the per-slice one — `augment_device_equivalence` in the selftest asserts
`max abs diff == 0`, and it holds on real cached studies too. Two things still differ in a
real `cuda` run, and neither is a property of this code:

* the noise field — numpy draws it on the CPU path, torch on the device path (both
  reproducible from `seed`);
* float32 `grid_sample` rounding between PyTorch's CPU and CUDA kernels: ~1e-4 max,
  ~5e-7 mean, on ~0.5–0.7 % of pixels.

An epoch of training compounds that into a different trajectory — the same kind of
difference a different seed produces, not a better or worse one. Short runs here (2 epochs,
600 studies) tracked each other on `train_loss` to ~2e-4 but landed 2–4 AUC points apart on
a 300-study validation set, which is within what that statistic does at that run length.
**Nothing here establishes that the two placements reach the same final metric**; if that
matters for a result, compare them at full length, or set `augment.device=cpu` to reproduce
an earlier run exactly.

## Model

```
[B, P, S, 3, H, W] → gather valid triplets → [N_valid, 3, H, W]
   → encoder adapter → feature map [N_valid, C, Hf, Wf]   (B0: C = 1280)
   → spatial pooling (model.spatial_pool)  → [N_valid, F] (avg: F = C)
   → scatter back to [B, P, S, F]                 (index_copy, differentiable)
   → masked mean ⊕ feature-wise masked max over S → [B, P, 2F]
   → concat fixed-order slots + P presence flags  → [B, P·2F + P]   (B0, avg, P = 3: 7683)
   → Linear(.., 256) → ReLU → Dropout(0.2) → Linear(256, 12)  → logits
```

`P` is the number of slots in `data.series_slots` (3 by default, 4 with an extra slot).

* Padding never enters the mean denominator or the max; an all-masked max returns exact
  zeros via a finite sentinel, never ±inf or NaN.
* A fully missing series produces an exactly zero, finite feature vector, and no fake image
  is ever encoded — otherwise BatchNorm would learn from padding.
* Logits during training; sigmoid only for metrics and exports. No softmax across targets.
* `model.freeze_bn_running_stats=true` (default) keeps the pretrained BatchNorm running
  statistics frozen while affine parameters stay trainable, and the policy is reapplied
  after every `model.train()`. Trainable statistics remain available as an option.
* Pretrained weights are explicit: `IMAGENET1K_V1`, a local checkpoint path, or `none`.
  A download failure raises with instructions — it never falls back to random init silently.
* The encoder is chosen with `model.backbone`; see *Encoder backbones* below.
* `model.spatial_pool`: `avg` (default, global average), `avgmax` (average ⊕ maximum, 2C)
  or `attention` (softmax-weighted average over the map, zero-initialised so it starts as
  `avg`).
* `model.target_attention` (needs `spatial_pool=avg`, meant with
  `data.laterality_canonical`): every window also yields its two column-half vectors, and a
  window's depth zone (`model.depth_zones` thirds of the valid windows) is
  medial/central/lateral on sagittal. Each target has its own query over all
  (slot, zone, half) tokens; the pooled vector adds a per-target logit to the head's. The
  output weights start at zero, so training starts from exactly the mean/max model.
* `model.side_pooling` (needs `spatial_pool=avg` and `data.laterality_canonical`): the head
  gets one mean/max per *side* of a slot instead of one per slot — the column halves of
  coronal/axial (medial | lateral) and the depth zones of sagittal — so a linear head can
  tell a medial from a lateral finding.
* `model.encoder_chunk_size` limits the size of one encoder call. **It does not guarantee
  lower training memory**: every chunk's autograd graph is retained until backward. To cut
  peak memory, reduce `train.microbatch_studies` or `data.centers_per_series`. Features are
  never detached (that would be a head-only diagnostic, not this model). Peak GPU memory is
  logged every epoch.

### Encoder backbones

The encoder is an adapter (`knee_mri/encoders.py`) that turns one slice triplet
`[N, 3, H, W]` into one spatial map `[N, C, Hf, Wf]`. Pooling, side pooling, target attention
and the head take their width from the adapter's `out_channels` — nothing is projected to a
common width.

| `model.backbone` | weights (`model.weights`) | map at 224 px | `last_n` units | example config |
|---|---|---|---|---|
| `efficientnet_b0` (default) | `IMAGENET1K_V1` \| file \| `none` | `[N, 1280, 7, 7]` | 9 feature stages | `config.yaml` |
| `efficientnet_v2_s` | `IMAGENET1K_V1` \| file \| `none` | `[N, 1280, 7, 7]` | 8 feature stages | — |
| `radimagenet_resnet50` | RadImageNet `ResNet50.pt` \| `none` | `[N, 2048, 7, 7]` | `layer1`…`layer4` | `config.radimagenet_resnet50.yaml` |
| `dinov2_vits14` | `lvd142m` \| file \| `none` | `[N, 384, 16, 16]` | 12 transformer blocks | `config.dinov2_vits14.yaml` |

```powershell
python -m knee_mri.cli train --config src/config.dinov2_vits14.yaml --mode fold --set split.fold=0
python -m knee_mri.cli train --config src/config.radimagenet_resnet50.yaml --mode fold --set split.fold=0
```

Switching the backbone is a new experiment. Train it from that backbone's pretrained weights;
`train.init_weights` from a checkpoint of another backbone is refused, and a resume detects a
changed backbone, freeze policy or adapter version.

* **EfficientNet** keeps its exact state_dict keys (`features.<i>.…`), so every earlier
  checkpoint loads strictly and predicts bit-identically (verified on six real runs, and by
  `tests/test_backbones.py` against the pre-adapter `model.py` from git).
* **DINOv2 ViT-S/14** — timm `vit_small_patch14_dinov2.lvd142m`. The adapter keeps the
  final-LayerNorm patch tokens of `forward_features()`, drops the CLS token (and any register
  tokens) and rearranges the row-major tokens into the `H/14 × W/14` grid. The model is built
  for exactly `data.image_size`; timm resamples the pretrained 518 px position embeddings when
  the weights load. There is no padding, cropping or resizing: the size must be a multiple of
  14 — **224 (16×16) and 336 (24×24) work, 320 and 384 are refused**. Use
  `data.encoder_normalization=imagenet` (its pretraining statistics). No CLS classifier branch.
  `lvd142m` downloads `timm/vit_small_patch14_dinov2.lvd142m/model.safetensors` once into the
  Hugging Face cache (`HF_HUB_OFFLINE=1` works afterwards), or point `model.weights` at a local
  copy of that file.
* **RadImageNet ResNet-50** — torchvision ResNet-50 up to `layer4` (no pooling, no fc) with the
  official RadImageNet PyTorch weights: `RadImageNet_pytorch/ResNet50.pt` from
  `RadImageNet_pytorch.zip` (Google Drive link in the README of
  [BMEII-AI/RadImageNet](https://github.com/BMEII-AI/RadImageNet); sha256 of the verified file
  `08629f7e…0734`). That file is the state_dict of the demo's `Backbone` wrapper,
  `nn.Sequential(*list(resnet50().children())[:9])`, so its keys are `backbone.{0,1,4,5,6,7}.*`;
  the loader renames them explicitly to `conv1/bn1/layer1…layer4` and refuses any missing, extra,
  mis-shaped or non-finite tensor (all 318 parameters and BN buffers must load). There is no
  download and no ImageNet substitute. Use `data.encoder_normalization=radimagenet_torch`: the demo
  feeds `(uint8 − 127.5)·2/255`, i.e. `[0, 1] → [−1, 1]` (mean 0.5, std 0.5). Checked on 60 real
  cached triplets: with that profile the batch statistics entering the 53 BatchNorm layers match
  their running statistics (mean |Δμ|/σ 0.10), with `mri_scalar` or `imagenet` they do not (≈ 4).

**Normalisation.** `data.encoder_normalization` (`mri_scalar` default | `imagenet` |
`radimagenet_torch`) is applied exactly once, by `dataset.normalization_stats()` — in the
worker or in `augment_batch` on the GPU, identically, and the same in training, validation and
inference. The robust intensity scaling and the geometry are unchanged. The profile, its
mean/std, the encoder id, adapter version and weight provenance (path + sha256) are stored in
every checkpoint (`payload["encoder"]`, `model_description`).

**Freezing** (`model.encoder_trainable`): `all` (default), `frozen` (head only) or `last_n`
with `model.encoder_trainable_units` — the last N transformer blocks plus the final norm
(DINOv2), the last N residual stages (ResNet), the last N feature stages (EfficientNet). The
example configs start from DINOv2 `last_n=4` and RadImageNet `layer4` only: first experiments,
not tuned optima. Frozen units get `requires_grad=False`, are left out of the optimizer
groups and run in eval mode after every `model.train()` (their BatchNorm statistics never
move); `model.freeze_bn_running_stats` separately decides the BN statistics of trainable units.

**Memory.** `model.grad_checkpointing` is backbone-specific: per EfficientNet stage, per
trainable ResNet stage, per DINOv2 block (non-reentrant, so trainable blocks after frozen ones
still get gradients). Frozen units keep no autograd graph. `model.encoder_chunk_size` still
works but bounds only the size of one encoder call, not the training memory.

**Inference** rebuilds the architecture from the checkpoint's own config with
`model.weights=none` — no network and no pretrained file — then loads the full project
checkpoint strictly. A DINOv2 checkpoint must be run at its trained `data.image_size`.

## Loss

`L_c = Σᵢ wᵢ꜀ · BCE(zᵢ꜀, yᵢ꜀) / Σᵢ wᵢ꜀` for classes with positive total weight, averaged
over those classes, computed in float32 from `binary_cross_entropy_with_logits(reduction='none')`.
An empty-supervision micro-batch returns a differentiable zero, and an accumulation window
with no supervision at all skips the optimizer and scheduler step (both are counted and
logged).

Accumulating class-normalised micro-batch losses is **not** algebraically identical to
normalising once over the whole effective batch, so the denominator is a stated policy,
`train.loss_normalization`:

| value | denominator `D_c` | note |
| --- | --- | --- |
| `microbatch` | per micro-batch, averaged over the window | the original formula above; with one-study micro-batches `w·BCE/w = BCE`, so a weight acts only as zero / non-zero |
| `window` | valid-label (mask) count of the whole accumulation window | a 0.2 label counts 1/5 of a 1.0 label, but switching a weak label on dilutes the class's other labels |
| `global` *(default)* | fixed per class from the whole training set: `K · Σ w / N_train` | every label keeps the same weight in every window — needed for weak and soft labels |

The last (possibly shorter) window is scaled by its real size. The reported epoch loss is
computed separately from accumulated class numerators/denominators — not from unweighted
batch averages. No positive-class weighting or oversampling is used.

## Training options

* **Weight EMA** (`train.ema_decay`, 0 = off; 0.999 ≈ 1000 steps): an exponential moving
  average of the weights, with the warmup `d_t = min(decay, (1+t)/(10+t))`. The EMA weights
  are validated, selected and saved as `best.pt` `"model"`; the raw weights are scored too
  (`*_raw` columns in `history.csv`).
* **`train.resume`** continues *the same* run exactly from its `last.pt` (checked against the
  stored signature: backbone, freeze policy, EMA, …). **`train.init_weights`** starts a *new*
  run from a checkpoint's model weights with a fresh optimizer, best score and history.
  Setting both is refused.

## Splits and leakage control

`splits.csv` is created before training and persisted with its metadata. DICOM `PatientID`
is **audited** first (missing, inconsistent within a study, constant, or reused by a site
above `split.max_studies_per_patient_id`); a failed audit falls back to study-level
splitting with a prominent warning, because a StudyInstanceUID is not proof of patient
independence. Fold assignment is a greedy, multilabel-aware, group-first balance over
per-target positive counts — `StratifiedGroupKFold` is not called with an unsupported
multi-label matrix, and missing targets are never encoded as negatives for stratification.
Identical report texts are audited across folds and reported, not merged (a repeated report
is not proof of the same patient). `splits.csv` carries `report_hash` and
`n_studies_with_same_report` so the check is repeatable.

### What the audit found on this export

* **`PatientID` is unique for every one of the 4407 studies** — 4407 distinct ids, none
  shared. It is a per-study pseudonym, not a patient key: grouping by it would be identical
  to study-level splitting while *looking* like patient-level protection. The audit detects
  this (`unique_per_study`) and reports the grouping as **study-level**, with the limitation
  stated in `splits_meta.json`. There is therefore currently **no way to guarantee** that two
  studies of the same person (both knees, a follow-up) land in the same fold — the local CV
  can be optimistic, and that caveat belongs next to any score from it.
* **54 identical-report clusters cover 204 studies** (largest: 37). Inspection shows these
  are short *normal-finding* boilerplate texts (`Sin anomalías`, `Diz eklemi içi sıvı miktarı
  normal…`, `ACL normal. MCL normal. …`) across different pseudonyms — i.e. different normal
  knees, not one patient. Merging them into one group would wrongly collapse many healthy
  patients into a single fold, so the default (audit and report, never merge) is the right
  call here. Revisit only if a cluster with a *specific, unusual* report appears.

With `split.holdout_reference=true` the radiologist reference studies (`paths.reference_csv`)
and their whole groups are held out of the CV folds (role `reference_holdout`, fold -1) and used as an **optional diagnostic audit**
(`evaluate --partition reference_holdout`), never as the early-stopping criterion — prompt
tuning happened on those reports, so they are not an independent gold benchmark.

### Fixed hold-out instead of CV (`run_holdout.sh` / `run_holdout.bat`)

One run, no folds: `train_v8.csv` (4199 studies) trains, and the 208 radiologist-labelled
studies of `train_labeled_208_reference.csv` validate (`HOLDOUT=208`, the default; 208 = the
158 plus 50 more). `train_v8` contains none of those 208 studies. The scripts pass everything
with `--set`, so the configs and `run_cv.*` stay as they were. On first use the scripts also
create the two files they need:

```bash
# splits.csv schema: validation studies fold 0, training studies fold 1
python -m knee_mri.cli make-fixed-split --set paths.train_csv=<data>/train_v8.csv \
    --set paths.splits_csv=work/splits/holdout208/splits.csv \
    --validation-csv <data>/train_labeled_208_reference.csv
# 0/1 radiologist labels frozen as the validation reference; empty cells are invalid
python -m knee_mri.cli freeze-reference --from-csv <data>/train_labeled_208_reference.csv \
    --out work/labels/frozen_reference_ref208.csv
```

and on every start `check-fixed-split` refuses a split that does not match `TRAIN_CSV` /
`VAL_CSV` (e.g. a `TRAIN_CSV` left over in the terminal that still holds hold-out studies).
Training then runs with `split.fold=0`, `paths.splits_csv` and `paths.frozen_reference_csv`
pointing at those files, and with `paths.exclude_from_training_csv` set to the validation CSV.

```powershell
.\run_holdout.bat E3_img320 --set data.image_size=320          # run E3_img320_holdout208_<timestamp>
.\run_holdout.bat --resume E3_img320_holdout208_20261005_143722 --set data.image_size=320
$env:HOLDOUT="158"; $env:TRAIN_CSV="F:\Kaggle\data\train_v6.csv"; .\run_holdout.bat   # the old 158 hold-out
```

`DATA_ROOT`, `TRAIN_CSV`, `VAL_CSV`, `SPLITS`, `FROZEN_REF` and `HOLDOUT` are environment
overrides; `NOPAUSE=1` skips the final key press. The Linux scripts also take `--gpu <id>`
before everything else. A resume needs the same extra arguments and
environment as the original start — the resume check stops on a difference.

Here the 208 studies **are** the early-stopping and `best.pt` criterion, so the score
measured on them is slightly optimistic, and with this few positives per target (e.g. MCL)
the macro AUC is also noisy. The "validation: NO supervision" warning in the label counts
is expected: the reference studies are not in the training table, and the metric comes
from the frozen reference.

### Keeping the reference out of every training run (teachers included)

The 208-study reference (`train_labeled_208_reference.csv`) is the evaluation set. A teacher
that trained on part of it would carry it into the pseudo-labels of its students, so no
training run may see it, CV teachers included. Three checks enforce this:

* **`paths.exclude_from_training_csv`** (default in every config:
  `notebooks/train_labeled_208_reference.csv`). `train.py` stops before the first epoch when a
  training study of the fold is on that list, whatever `splits.csv` is in use. A configured but
  missing or empty file is an error. An empty value turns the check off, e.g. to resume a legacy
  run. `run_holdout.*` sets it to `VAL_CSV`.
* **`make-splits` with `split.holdout_reference=true`** refuses to write a split with an empty
  hold-out (missing `reference_csv`, or none of its studies in `train_csv`). With `train_v8.csv`
  the flag is not needed: the CSV has no reference rows to begin with.
* **`run_cv.*`** no longer uses the legacy `work/splits/splits.csv` (4407 studies, the reference
  inside the folds). Its default split is `work/splits/cv<NFOLDS>_<train_csv stem>/splits.csv`.
  The script creates it with `make-splits` when it is missing and checks it on every start with
  `check-splits --n-folds N` (fold count, training CSV, exclusion list). The fold count comes
  from `NFOLDS` (default 3), the training CSV from `TRAIN_CSV` or the config, and `SPLITS`
  overrides the path. `merge-oof` gets the same split.

```powershell
.\run_cv.bat B0_teacher --set data.image_size=320   # teacher OOF: 3 folds (default), split work/splits/cv3_train_v8, ref208 never trains
.\run_cv.bat --resume B0_teacher_cv3_<timestamp> --set data.image_size=320
```

`run_cv.*` names the runs `<prefix>_cv<N>_<timestamp>_fold<k>` and writes the merged OOF to
`..._oof`. On `--resume` a fold with `run_summary.json` is skipped, a fold with only
`last.pt` continues from it, and a fold with neither starts over; the OOF is merged again
at the end.

## Metrics

`model.eval()` + `torch.inference_mode()`, predictions accumulated over the whole fold
before any ranking metric is computed (never per-batch AUC averaging). Per target:
ROC-AUC with known positive/negative counts, average precision reported as **AP** with
prevalence and a degeneracy flag, and precision / recall / specificity / F1 at a fixed
0.5 threshold with the confusion counts. Undefined denominators return NA.
**A single-class target returns NA — never 0.5.** Macro values state `n_defined_targets/12`.
This is the local evaluable-target macro; any official evaluator adapter is a separate,
unverified concern.

Thresholds are not tuned on the data they are reported on. `validation_predictions.csv`
carries one row per (study, target) with fold, score, reference, validity and checkpoint
provenance. `merge-oof` **refuses** to merge unless all intended folds are present,
disjoint and complete — a single held-out fold is not OOF coverage. Patient-group bootstrap
intervals are available (`evaluate.bootstrap_intervals`) and optional.

## Outputs per run

Runs are grouped by training table: `paths.output_dir` defaults to
`<work_dir>/runs/<train_csv stem>`, so a run on `train_v3.csv` lands in
`work/runs/train_v3/<run>` (an explicit `paths.output_dir` replaces the whole default).
`python -m knee_mri.cli output-dir [--config ...] [--set ...]` prints the resolved folder; the
`run_cv.*` / `run_holdout.*` scripts use it to find the fold runs for the OOF merge.
`src/tools/migrate_runs_by_dataset.py` sorted the earlier flat `work/runs/<run>` folders.

`history.csv`, `metrics_per_class.csv`, `validation_predictions.csv`,
`validation_reference.csv`, `run_summary.json`, `coverage.json`, `environment.json`,
`config.yaml`, `run.log`, `best.pt`, `last.pt`, learning-curve and per-class plots, and an
offline `report.html` linking the QC gallery, label counts, metrics and high-confidence
disagreements. Disagreement review is a development aid and is kept out of any locked final
evaluation.

Checkpoints store model/optimizer/scheduler/scaler state, epoch, best score, RNG state,
target order, config and the label/split/preprocessing versions. Resume is exact **at epoch
boundaries only**; mid-epoch resume is not implemented and is not claimed.

## Diagnostics and label tools

These read finished runs or label exports; none of them trains.

| command | what it does |
| --- | --- |
| `diagnose <run dir or run-group prefix>` | per-class ROC-AUC on the frozen reference and on the radiologist studies of each validation set, excluded cells by LLM status, and PASS/WARN/FAIL implementation checks → `<group>_diagnose/`. `--runs` lists the fold runs explicitly (several seeds per fold), `--deep` also reloads checkpoints and pushes real studies through them |
| `evaluate --allow-data-override KEY` | scores a checkpoint with a `data.*` setting that differs from its training config (repeatable); otherwise such a difference is refused |
| `merge-label-details <details.csv …>` | stitches the per-run LLM `labels_details.csv` exports into one details file; a gap or a duplicate (study, target) row is an error, reference studies take the reference label |
| `label-audit sample` / `summarize` | draws a stratified ~100-report audit sheet (random, Synovitis, MCL, many-empty groups), then summarises the reviewer's verdicts as error rates per target and basis |
| `freeze-reference [--from-csv]` | freezes the validation reference once, so every label experiment is scored against the same file |

## Failure gates (these stop the run on purpose)

* label coverage or class support below `labels.min_labeled_studies` / `min_positives_per_target`;
* studies with no usable series in any slot (`data.allow_all_missing_studies=false`) — the
  fallback is explicit and separately reported, so validation cannot quietly improve by
  dropping them;
* more than half of the volumes failing to decode (systematic decoder problem);
* more than 20 % of cache entries failing;
* two consecutive epochs without a single defined validation ROC-AUC;
* a non-group-disjoint fold, a train/validation overlap, or a checkpoint whose target order
  differs from the project order;
* a training study listed in `paths.exclude_from_training_csv`, or a split that does not
  match the run (`check-splits`, `check-fixed-split`).

## Verification status

`python -m knee_mri.cli selftest` — all checks pass on this machine. They cover geometric
slice sorting, plane assignment, the in-plane transform being a pure reordering, true
neighbour triplets and gap handling, short-stack padding, masked mean/max, padding
invariance of the logits, missing-slot zeroing, masked-label gradients, NaN-target
rejection, the class-normalisation formula, accumulation window scaling, epoch-loss
accounting, sigmoid-not-softmax outputs, single-class AUC returning NA, key-based label
joins, patient-group separation, pseudonymous-PatientID detection, the dataset contract,
checkpoint round-trip and a real forward/backward that reduces the loss. The same checks run
under pytest (`tests/`).

Encoder adapters (`tests/test_backbones.py`, random weights, no network): the documented
map of every backbone, finite `[B, 12]` logits, padding and absent slots never encoded (NaN
padding leaves the logits identical), pooling / side pooling / target attention on the new
maps, the DINOv2 patch grid (CLS dropped, row-major order, equal to timm's own NCHW output,
320/384 refused), gradients only in unfrozen units, complete non-duplicated optimizer groups,
single normalisation identical on the worker and GPU paths, resume / init-weights guards,
exact gradient checkpointing, the strict RadImageNet loader, a synthetic train → `best.pt` →
offline reload with identical predictions, and old EfficientNet checkpoints. Real weights:
`tests/test_pretrained_encoders.py` (skips when the weights are absent).

`tests2/` holds the visual preprocessing checks: data-independent PNG examples with numeric
assertions (`visual_preprocessing_examples.py`, also under pytest), the 150 mm crop of the
flagged wide-FOV series as NRRD + PNG (`flagged_wide_fov_crops.py`), and the crop-centre
policy comparison over 500 studies (`crop_center_policy_comparison.py`). See
[`tests2/README.md`](tests2/README.md).

Also executed here on the real export (the first verification round, on the 58 labelled
studies of the original `train.csv`):

* `validate-schema` over all 4407 studies / 24371 series — no schema errors; the readiness
  gate correctly refuses full training at 58/4407 labelled studies;
* **the full manifest**: 24371 series → 24372 volume candidates (one series really does split
  into two acquisition groups), **24372/24372 decoded and usable**, and **all 4407 studies
  received all three slots**. Flags: 1353 `oblique_inplane`, 251 `ambiguous_plane`;
* **`make-splits` over all 4407 studies**: 58 reference studies held out, three folds of
  1450/1450/1449, grouping honestly reported as study-level (see the audit findings above);
* cache build for those 174 selected series (174/174 ok) and the QC gallery, inspected
  visually — anatomy centred, orientation consistent, triplets are true neighbours;
* the in-sample overfit diagnostic on 8 real studies: train loss 0.67 → 0.44, in-sample macro
  ROC-AUC 0.94 → 1.00 over 6 epochs. That is a debugging signal that the gradient path works
  **and nothing else** — train and "validation" studies are identical there;
* a synthetic GPU run with bf16 AMP, plus the HTML report and plots.

Measured on an RTX 5080 at the default settings (224 px, 24 centres, 3 slots,
`microbatch_studies=1`): **3.3 GB peak GPU memory** and **≈7 studies/s**, i.e. roughly
11 minutes per epoch over 4407 studies, excluding the one-off cache build. At 320 px
(72 triplets, bf16, one study): B0 6.3 GB and 10.4 training studies/s, V2-S 11.1 GB and
5.6 studies/s, V2-S with `model.grad_checkpointing` 4.1 GB and 4.0 studies/s.

Since then the full-dataset cache has been built and fold / hold-out runs train on the
LLM-labelled tables (see *Where this repository currently stands*).

## Performance notes

Cache size measured here: **≈12 MB per study** (3 slots, 224 px, float16 npz), i.e. roughly
**50–55 GB for all 4407 studies**; a 320 px cache holds about twice the pixels. Image size and crop
policy are part of the preprocessing hash (a change rebuilds the cache); a different series
selection should get its own selection csv and cache directory. Put `paths.cache_dir` on a disk
with room (here `I:/Kaggle/data/cache`); the 58-study verification cache alone is 713 MB.
One study at the default settings is `3 × 24 × 3 × 224 × 224` floats ≈ 43 MB, and one
optimizer step encodes `72 × microbatch_studies` triplets. Start with
`train.microbatch_studies=1`, `accumulation_steps=8`, raise `num_workers` on Linux
(keep it at 0 or use the `main()` guard on Windows), and read the logged peak GPU memory
before increasing anything.

### Where the time actually goes

Measured on this machine (RTX 5080, Xeon W-2245, 8 physical cores, `num_workers=8`,
`microbatch_studies=1`, 224 px, cache on a SATA SSD):

| per study, one worker | `augment.device=cpu` | `augment.device=cuda` |
|---|---|---|
| cache read (npz, cold) | ~10 ms | ~10 ms |
| zlib decompress + cast | ~65 ms | ~65 ms |
| gather + augment + normalise | ~470 ms | — |
| **worker total** | **548 ms** | **90 ms** |
| augmentation on the GPU | — | ~1.2 ms |

Steady-state training epoch, 600 studies: **61.1 s → 42.5 s**, i.e. 9.8 → 14.1 studies/s.
The ceiling with everything already resident on the GPU is 16.5 studies/s, so with
`augment.device=cuda` the run is GPU-bound at ~86 % of that ceiling; on the CPU path it
was dataloader-bound at ~59 %. More `num_workers` will not help past 8 physical cores —
raise `microbatch_studies` instead, and watch the logged peak GPU memory.

Disk placement barely matters for training: the loop only ever reads the npz cache, and
that read is ~1 % of the work. It is the one-off manifest and cache build that reads the
DICOM export, and *that* is worth putting on an SSD.
