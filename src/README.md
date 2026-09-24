# RSNA knee MRI — EfficientNet-B0 2.5D MIL baseline

Study-level, multi-label knee MRI classification (12 targets) from DICOM series, using
**torchvision EfficientNet-B0** on **2.5D adjacent-slice triplets** with **masked
multiple-instance pooling** over plane slots.

The model consumes images and explicit input-validity masks. No LLM is called during
image training or validation; report-derived labels enter only through the label export
files described below.

## Scope

In this version, deliberately: one strong reproducible baseline. **Not** implemented:
3D networks, segmentation, anatomical locators, CoAtNet/DINO, attention pooling,
pseudo-labelling, ensembling, TTA. Externally supplied soft targets are supported by the
loss/data contract, but no soft value is ever invented.

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

## Inputs and contracts

| File | Role |
| --- | --- |
| `train.csv` | study metadata, report text, and — in this export — the 12 target columns |
| `train_series.csv` | study↔series map plus `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` |
| `train_series/<StudyUID>/<SeriesUID>/*.dcm` | image data |
| `labels_details.csv` *(optional, preferred)* | one row per (study, target) with `status` and `basis` |
| `labels_statuses.csv` *(optional)* | wide text statuses |
| `labels_predictions.csv` *(optional)* | wide numeric export; empty cell = unresolved |
| `labels_predictions_exclude_borderline.csv` *(optional)* | same, borderline also emptied |
| `train_labeled_58_reference.csv` *(optional)* | radiologist reference set — audit only |

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

`train.csv` in this export carries numeric targets for **58 of 4407 studies (1.3 %)**, the
same set as the radiologist reference file. The readiness gate therefore **fails full
training by design** and says why. That is enough for the synthetic smoke test and for a
clearly labelled in-sample overfit check (`--mode overfit`), and not enough for a
credible CV result. Point `paths.labels_details_csv` (or one of the wide exports) at a
real extraction export to unlock fold training.

## Label policy

| Extraction status | target | weight |
| --- | --- | --- |
| `positive` | 1 | 1 |
| `negative`, basis `explicit_absence` / `below_threshold` | 0 | 1 |
| `negative`, basis `borderline` | 0 | **0** (`labels.borderline_policy=exclude`, default) or 1 (`as_negative` ablation) |
| `uncertain` (conflict / insufficient detail / not assessed) | 0 (placeholder) | `labels.uncertain_weight` (default 0) |
| `not_mentioned` | 0 (placeholder) | `labels.unmentioned_weight` (default 0; 0.2 / 1.0 are explicit weak-negative experiments) |
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
python -m knee_mri.cli selftest                                   # 22 failure-mode checks
python -m knee_mri.cli train --mode synthetic --n-studies 16 --epochs 3
```

### Steps 1–5 — the real chain, in this order

Run them in this order. The `Needs` column states what each command actually requires, so you
can see which dependencies are hard (the command fails without them) and which are advisory:

| Step | Command | Needs |
| --- | --- | --- |
| 1 | `validate-schema` | the input CSVs only |
| 2a | `build-manifest` | the DICOM root — writes `series_selection.csv` **and** `patient_audit.csv` |
| 2b | `build-cache` | **the manifest** — fails without `series_selection.csv` |
| 3a | `qc` | **the cache** — otherwise every panel just says "not cached" |
| 3b | `make-splits` | soft: runs without `patient_audit.csv`, but then warns and falls back to study-level grouping |
| 4 | `train --mode overfit` | **the cache and `splits.csv`** |
| 5 | `train --mode fold` | **the cache and `splits.csv`**, and a passing label-readiness gate |

```powershell
# 1. Configure paths, validate schemas and label coverage
python -m knee_mri.cli validate-schema

# 2. Build the DICOM manifest (selection + patient audit), then the deterministic cache
python -m knee_mri.cli build-manifest
python -m knee_mri.cli build-cache

# 3. Inspect QC, then create the immutable folds
python -m knee_mri.cli qc --n-studies 12
python -m knee_mri.cli make-splits

# 4. Tiny in-sample overfit check on real images (debugging output, not performance)
python -m knee_mri.cli train --mode overfit --n-studies 8

# 5. Train fold 0, then build its report
python -m knee_mri.cli train --mode fold --set split.fold=0
python -m knee_mri.cli report work/runs/<run-name>
```

### Step 6 — after a fold has finished: pick what you need

**This is a menu, not a sequence.** These commands are alternatives; run only the one that
matches your situation.

*The run was interrupted and you want to continue it* (resumes at the next epoch boundary):

```powershell
python -m knee_mri.cli train --mode fold --set train.resume=work/runs/<run>/last.pt
```

*Score a saved checkpoint on its held-out fold:*

```powershell
python -m knee_mri.cli evaluate --checkpoint work/runs/<run>/best.pt
```

*Run the diagnostic audit against the radiologist reference set* — a development signal only,
never a gold benchmark and never a selection criterion:

```powershell
python -m knee_mri.cli evaluate --checkpoint work/runs/<run>/best.pt --partition reference_holdout
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
  work/runs/<run-fold0>/validation_predictions.csv `
  work/runs/<run-fold1>/validation_predictions.csv `
  work/runs/<run-fold2>/validation_predictions.csv
```

### Where this repository currently stands

Steps 1, 2a and 3b have been run on the full export; the cache (2b) exists only for the 58
labelled studies, so step 3a covered those. Step 5 is blocked by the label-readiness gate
until a real extraction export is configured — see *Current data status* above.

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
   signal, not two.
3. **Preprocessing** (one versioned deterministic path, used for train and validation):
   modality LUT applied via pydicom, `PixelPaddingValue` turned into NaN, MONOCHROME1
   inverted once explicitly; a canonical series-local in-plane orientation built from
   transposes and flips only (a pure array reordering — oblique data is *not* silently
   straightened, and `oblique_inplane` is flagged); a fixed **150 × 150 mm** physical crop
   around a foreground-derived centre (the scan centre is not assumed to be the knee
   centre), honouring anisotropic row/column spacing, padded where the FOV exceeds the
   matrix; robust p1/p99 foreground intensity scaling to [0, 1] with documented fallbacks
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
   → EfficientNet-B0 features + global average pool → [N_valid, 1280]
   → scatter back to [B, P, S, 1280]            (index_copy, differentiable)
   → masked mean ⊕ feature-wise masked max over S → [B, P, 2560]
   → concat fixed-order slots + P presence flags  → [B, 7683]   (P = 3)
   → Linear(7683, 256) → ReLU → Dropout(0.2) → Linear(256, 12)  → logits
```

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
* `model.encoder_chunk_size` limits the size of one encoder call. **It does not guarantee
  lower training memory**: every chunk's autograd graph is retained until backward. To cut
  peak memory, reduce `train.microbatch_studies` or `data.centers_per_series`. Features are
  never detached (that would be a head-only diagnostic, not this model). Peak GPU memory is
  logged every epoch.

## Loss

`L_c = Σᵢ wᵢ꜀ · BCE(zᵢ꜀, yᵢ꜀) / Σᵢ wᵢ꜀` for classes with positive total weight, averaged
over those classes, computed in float32 from `binary_cross_entropy_with_logits(reduction='none')`.
An empty-supervision micro-batch returns a differentiable zero, and an accumulation window
with no supervision at all skips the optimizer and scheduler step (both are counted and
logged).

Accumulating class-normalised micro-batch losses is **not** algebraically identical to
normalising once over the whole effective batch. The policy chosen here: normalise per
micro-batch, average over the window, scale the last (possibly shorter) window by its real
size. The reported epoch loss is computed separately from accumulated class
numerators/denominators — not from unweighted batch averages. No positive-class weighting
or oversampling is used to start with.

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

The radiologist reference studies and their whole groups are held out of training by
default (`split.holdout_reference`) and used as an **optional diagnostic audit**
(`evaluate --partition reference_holdout`), never as the early-stopping criterion — prompt
tuning happened on those reports, so they are not an independent gold benchmark.

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

`history.csv`, `metrics_per_class.csv`, `validation_predictions.csv`,
`validation_reference.csv`, `run_summary.json`, `coverage.json`, `environment.json`,
`config.yaml`, `run.log`, `best.pt`, `last.pt`, learning-curve and per-class plots, and an
offline `report.html` linking the QC gallery, label counts, metrics and high-confidence
disagreements. Disagreement review is a development aid and is kept out of any locked final
evaluation.

Checkpoints store model/optimizer/scheduler/scaler state, epoch, best score, RNG state,
target order, config and the label/split/preprocessing versions. Resume is exact **at epoch
boundaries only**; mid-epoch resume is not implemented and is not claimed.

## Failure gates (these stop the run on purpose)

* label coverage or class support below `labels.min_labeled_studies` / `min_positives_per_target`;
* studies with no usable series in any slot (`data.allow_all_missing_studies=false`) — the
  fallback is explicit and separately reported, so validation cannot quietly improve by
  dropping them;
* more than half of the volumes failing to decode (systematic decoder problem);
* more than 20 % of cache entries failing;
* two consecutive epochs without a single defined validation ROC-AUC;
* a non-group-disjoint fold, a train/validation overlap, or a checkpoint whose target order
  differs from the project order.

## Verification status

`python -m knee_mri.cli selftest` — 22/22 checks pass on this machine. They cover geometric
slice sorting, plane assignment, the in-plane transform being a pure reordering, true
neighbour triplets and gap handling, short-stack padding, masked mean/max, padding
invariance of the logits, missing-slot zeroing, masked-label gradients, NaN-target
rejection, the class-normalisation formula, accumulation window scaling, epoch-loss
accounting, sigmoid-not-softmax outputs, single-class AUC returning NA, key-based label
joins, patient-group separation, pseudonymous-PatientID detection, the dataset contract,
checkpoint round-trip and a real forward/backward that reduces the loss. The same checks run
under pytest (`tests/`).

Also executed here on the real export:

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
11 minutes per epoch over 4407 studies, excluding the one-off cache build.

**Not yet executed:** the full-dataset cache build (~50 GB, hours of decoding) and a real
fold training run. The fold run is blocked by label coverage, not by the code.

## Performance notes

Cache size measured here: **≈12 MB per study** (3 slots, 224 px, float16 npz), i.e. roughly
**50–55 GB for all 4407 studies**. Put `paths.cache_dir` on a disk with room; the 58-study
verification cache alone is 713 MB.
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
