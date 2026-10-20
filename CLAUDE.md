# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Kaggle "RSNA Knee Abnormality Detection": study-level multi-label classification of 12 findings
(order fixed in `knee_mri/constants.py::TARGETS`) from knee MRI DICOMs. Two halves:

1. **LLM-extracted labels** from radiology reports (`notebooks/02_*`, `notebooks/build_*.py`) →
   soft-target tables `train_v<N>.csv` (current: `train_v11.csv`, 3957 studies = `train_v8`
   minus the 242 new reference studies).
2. **2.5D multiple-instance image model** in `src/knee_mri/` (one series per plane slot →
   adjacent-slice triplets → 2D encoder → masked mean/max over slices → MLP head → 12 logits).

`README.md` is the overview; `src/README.md` is the authoritative technical reference (label
policy, model, loss, DDP, splits, metrics, failure gates) — read the relevant section before
changing behaviour. Many notebooks, shell-script comments/messages and `doc/` notes are in
**Hungarian**; code, configs and `src/README.md` are in English.

## Commands

Run from `src/` (or set `PYTHONPATH=src`). Conda env: `kaggle_2026` (Python 3.11).
Any config key can be overridden with `--set key.sub=value`; pick a config with `--config`.

```bash
cd src
python -m knee_mri.cli selftest [--quick]           # synthetic failure-mode checks, no data needed
python -m pytest tests tests2 -q                    # pytest wrappers around the same checks + backbones
python -m pytest tests/test_pipeline.py::test_plane_assignment -q   # single test
python -m knee_mri.cli train --mode synthetic --n-studies 16 --epochs 3   # end-to-end smoke test, no data

# Real chain (needs data paths set in the config)
python -m knee_mri.cli validate-schema | build-manifest | build-cache | qc | make-splits
python -m knee_mri.cli train --mode fold --set split.fold=0 --set data.image_size=320
python -m knee_mri.cli report work/runs/<train_csv stem>/<run>
python -m knee_mri.cli diagnose <run dir or prefix>
```

Training is normally launched via the root runner scripts (they create/verify splits and the
frozen reference, resolve the output dir, and support `--gpu 0,1` for DDP via torchrun and
`--resume <run name>`):

```bash
./run_holdout.sh [--gpu N] <prefix> --set data.image_size=320   # train_v11 trains, 450 reference validates
./run_cv.sh      [--gpu N] <prefix> --set data.image_size=320   # 3-fold CV + merge-oof
./run_cv545.sh   [--gpu N] <prefix> --set data.image_size=320   # 3-fold CV on the 545 reference (LABEL_MODE=soft|binary)
```

Linux uses `src/config.linux.yaml`; `src/config.yaml` is the Windows config (paths like
`F:/Kaggle/data`). The working image size is 320 even though configs default to 224.

Hardware: the Windows machine has 1× RTX 5080 (16 GB); the Linux machine has 2× RTX 3090
(24 GB each), and the default GPU assignment there is `--gpu 2,3` (physical IDs 2 and 3).
Size batch/memory and DDP commands accordingly.

## Architecture notes (cross-file)

- **Entry point**: `knee_mri/cli.py` (`cmd_*` per subcommand). `src/train.py` is a thin wrapper
  equal to `cli train` (needed for torchrun and Windows DataLoader workers).
- **Config**: `knee_mri/config.py` loads YAML + `--set` overrides; relative paths resolve against
  the repo root. `paths.output_dir` defaults to `work/runs/<train_csv stem>/`.
- **Data path**: `manifest.py` (DICOM inventory + per-slot series selection → `series_selection.csv`)
  → `laterality.py` / `geometry.py` → `preprocess.py` (deterministic canonical orientation, physical
  crop, robust scaling, resample) → cache under `<cache_dir>/<PREPROCESS_VERSION>_<preprocess_hash>`
  → `dataset.py` (`StudyBagDataset`, `[B, P, S, 3, H, W]` bags with masks). Any change to a
  preprocessing setting changes the hash and so builds a separate cache; stale entries are rejected.
  If you change preprocessing semantics, update `preprocess_signature`/`PREPROCESS_VERSION`.
- **Model**: `encoders.py` (backbone adapters: EfficientNet B0/V2-S, ConvNeXt, RadImageNet ResNet-50,
  DINOv2 ViT-S/14; each has its own normalization) + `model.py` (gather valid triplets, scatter back,
  masked pooling, slot presence flags, optional target-attention / side pooling). Padding and missing
  slots must never be encoded or leak into pooling.
- **Labels**: `labels.py` maps extraction statuses to (target, weight); unknown = weight 0, never a
  silent negative. Placeholders must stay finite (no `NaN * 0` masking). `loss.py` is weighted BCE
  with a configurable normalization policy (`train.loss_normalization`, default `global`).
- **Training**: `train.py` (accumulation windows of 8 studies, EMA, exact resume at epoch boundaries
  only, signature-checked) + `distributed.py` (DDP designed to reproduce the single-GPU experiment
  exactly; `accumulation_steps` must be divisible by GPU count).
- **Evaluation**: `metrics.py` — whole-fold predictions, single-class target → NA (never 0.5),
  soft ROC-AUC on continuous references; checkpoint selection by `eval.selection_metric`.
  `report.py` adds a per-subset breakdown (`full_450` computed directly, `original_208`,
  `new_242`; `subset_summary.csv`) from `paths.reference_subsets_csv`.
- **Inference**: `notebooks/04_kaggle_test_*.ipynb` are generated by `notebooks/build_04_notebook.py`
  and import the `knee_mri` package directly. Edit the builder and regenerate; never edit the
  `.ipynb` directly. The notebook checks each checkpoint's stored preprocessing hash.
- **Tests**: `selftest.py` holds the `check_*` functions; `tests/test_pipeline.py` wraps them for
  pytest. A new check goes in `selftest.py` plus a wrapper.

## Invariants not to break

- The 450-study radiologist reference (`notebooks/train_labeled_450_reference.csv` = the
  original 208 + 242 new, from `rsna_reference_450_package.zip`) is the evaluation set and
  must never enter training (incl. CV teachers). Enforced via `paths.exclude_from_training_csv`,
  `check-splits`, `check-fixed-split`. Report full_450, original_208 and new_242 separately.
  Exception: the reference-545 3-fold CV (`run_cv545.sh`, inputs from `notebooks/build_ref545_cv.py`)
  trains each fold on the other two folds' radiologist soft labels and validates on its own fold's
  binary labels; a study's own fold never sees it. Models/pseudo-labels from that CV must not feed
  a run validated on the 450/545. Report full_545, original_208, new_242, new_95 there.
- Studies in `notebooks/label_mask_studies.csv` (bilateral / rotated scans) have all labels masked.
- `splits.csv` is immutable once created; folds are patient-group disjoint.
- Failure gates (see `src/README.md` → *Failure gates*) stop runs on purpose; don't weaken them to
  make a run pass.
- `work/` (manifests, caches, splits, runs) is git-ignored; data is not in the repo.
