# Implementation prompt: RSNA knee MRI, EfficientNet-B0 2.5D MIL

You are a senior PyTorch engineer experienced in medical imaging. Implement a complete, runnable training and validation pipeline for study-level, multi-label knee MRI classification. Use **torchvision EfficientNet-B0 with 2.5D adjacent-slice inputs and masked multiple-instance pooling**.

Deliver working code, a practical notebook, configuration, and run instructions. Do not stop at an architecture description or pseudocode. Keep the first implementation small and understandable. Use English for code, comments, documentation and notebook explanations.

## 1. Goal and boundaries

The training data contain DICOM MRI studies with multiple series. Labels have already been extracted from radiology reports. The image model consumes images and explicit input-validity masks, not report text. Do not call an LLM during image training or validation.

Build one strong, reproducible baseline first. Do not implement a 3D network, segmentation model, anatomical locator, CoAtNet, DINO, attention pooling, pseudo-label generation or ensemble in this version. Support externally supplied soft targets through the loss/data contract, but do not invent their values.

The plan is informed by participant reports that small CNNs at 224-288 pixels can perform well. Those reports are not reproducible benchmarks or promised scores. Report extraction can be correct while the report still omits an image finding. Do not require perfect agreement with a small radiologist reference before enabling image training.

## 2. Inspect inputs and establish explicit contracts

Inspect any attached project files before choosing adapters. Likely inputs are:

- `train.csv`: study metadata and original reports.
- `train_series.csv`: mapping of studies to series and available series metadata.
- DICOM directories such as `F:\Kaggle\data\train_series\<StudyInstanceUID>\<SeriesInstanceUID>\...`.
- `labels_predictions.csv`: `StudyInstanceUID` plus the 12 numeric targets; unresolved values are empty.
- `labels_predictions_exclude_borderline.csv`: the alternative export with borderline decisions empty as well.
- `labels_statuses.csv`: the four text statuses.
- `labels_details.csv`: one row per study and target, including extraction details; it may also contain reference fields that MUST NOT silently become training labels.
- Optional `train_labeled_58_reference.csv`: the small radiologist-reference set.
- Optional `label_train_reports.ipynb`: inspect it to understand the export schema; do not execute its API calls.

Some exports may currently cover only 58 studies. Check coverage against the full study inventory. Do not pretend that a 58-study export labels the full dataset. A small export is enough for a clearly identified smoke test; full training requires adequate labeled coverage and class support.

Use this confirmed **project target order**, and check it against the actual input headers:

```python
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
```

Keep target order in configuration and checkpoints. Do not invent clinical definitions or claim this prompt verifies the current official scoring interface. Implement local macro ROC-AUC as the selection metric; keep any official evaluator adapter separate and identify unverified competition requirements.

Read identifiers as strings. Validate unique study IDs, unique study-series keys, many-to-one series-to-study joins, label duplicates, orphan series, unexpected labels and coverage. Never join by row position. Never use patient IDs, source paths or report text as predictive features.

If files are absent, implement explicit adapters and schema-validation commands with documented expected fields. Do not guess missing columns. Ask only for indispensable unresolved mappings, while completing all independent implementation work.

## 3. Configuration and environment

Use plain PyTorch and torchvision, not a high-level training framework. Prefer pandas, NumPy, scikit-learn, pydicom and matplotlib; add a DICOM decoder or imaging dependency only where needed and explain it.

Support local Windows/Jupyter and Linux/Kaggle with configurable paths using pathlib. Design for one CUDA GPU, initially an RTX 3090 with 24 GB VRAM, with CPU smoke-test support. This is a design target, not a promise that every setting fits in memory. The separate LLM server is not a training dependency.

Provide YAML configuration and CLI overrides. Starting settings:

```yaml
seed: 42
n_folds: 3
fold: 0
image_size: 224
fov_mm: 150.0
series_slots: [sagittal, coronal, axial]
centers_per_series: 24
adjacent_slices: 3
microbatch_studies: 1
accumulation_steps: 8
max_epochs: 25
warmup_epochs: 2
early_stopping_patience: 5
encoder_lr: 0.0001
head_lr: 0.0003
weight_decay: 0.0001
head_dropout: 0.2
freeze_bn_running_stats: true
unmentioned_weight: 0.0
borderline_policy: exclude
num_workers: 0
amp: auto
```

These are starting hypotheses, not optimal settings. Expose 288-pixel input and other FOV values for later controlled experiments. Validate image size, slot count and channel count against the architecture. Log effective microbatch/accumulation behavior.

Use compatible, documented package versions. Provide installation instructions appropriate to the user's CUDA environment instead of inventing a universal CUDA wheel command. Log installed versions and GPU information. Use importable dataset/worker functions and a main guard for Windows multiprocessing; keep notebook defaults reliable.

## 4. DICOM inventory, geometry and series selection

Build a reusable manifest before training. Record source IDs, paths, dimensions, frame count, orientation, pixel spacing, slice positions, sequence metadata, transfer syntax, decode status and quality flags.

For ordinary single-frame stacks:

1. Group compatible images; separate mixed echoes, time points, orientations or other acquisition dimensions when present.
2. Verify compatible `ImageOrientationPatient` vectors. Form a common normal from their cross product.
3. Project each `ImagePositionPatient` onto that normal and sort geometrically. Do not use filenames or InstanceNumber as the primary order.
4. Detect duplicate positions and irregular spacing; avoid blindly deduplicating different echoes at the same location. Do not substitute SliceThickness for center-to-center spacing.
5. Infer the plane from the normal with an explicit obliquity tolerance. Log ambiguous plane assignment.

Handle enhanced/multiframe objects with their actual per-frame geometry when supported. If a format cannot be handled faithfully, explicitly flag it and report coverage; never silently treat a multiframe file as one slice. Missing required decoding dependencies must fail clearly. A rare corrupt series can be quarantined with a reason and a fallback candidate; a systematic decoder failure must not turn into a dataset full of missing inputs.

Select one usable series per slot deterministically. Prefer suitable fluid-sensitive PD/T2-like acquisitions as a configurable initial heuristic. Use orientation plus available sequence metadata; do not claim SeriesDescription alone reliably identifies a sequence. Exclude localizers. Rank candidates transparently and use a stable tie-break. Export selected series and reasons.

Use only image-side information for selection. Freeze selection rules across train and validation. Preserve source geometry and laterality; do not create voxel-level alignment assumptions between different planes.

## 5. Image preprocessing and cache

Implement one versioned deterministic preprocessing path reused for training and validation:

- Decode using pydicom and required plugins. Handle applicable pixel-value transforms, padding values and photometric interpretation deliberately. Do not apply an arbitrary display window or convert MRI to lossy 8-bit JPEG.
- For MONOCHROME1 or unusual scaling, use an explicit, documented policy checked against representative images; prevent double transforms.
- Construct a consistent series-local in-plane orientation. Distinguish array reordering from physical resampling of oblique data. Record the transformation and use row/column spacing correctly.
- Use a fixed physical in-plane crop, initially 150 x 150 mm, with a documented center derived from image geometry or a conservative foreground estimate. Use a consistent crop across slices of a series. Do not claim the scan center is guaranteed to be the knee center. Support padding and a broader-FOV option.
- Respect anisotropic pixel spacing when mapping physical distances to array coordinates. Resample to the requested square output without arbitrary anatomical stretching. Do not apply the torchvision ImageNet center-crop transform afterward: that could remove more anatomy.
- Keep the original through-plane slice ordering for this 2.5D baseline. Do not fabricate isotropic volumes merely to build adjacent triplets. Flag physical gaps that make nominally adjacent slices unsuitable.
- Estimate robust p1/p99 intensities from valid foreground within each series, excluding padding/background. Clip and scale to [0,1], guarding empty foreground, constant images and nonfinite values. Document fallback behavior.
- Apply an explicit final encoder normalization. Default to the pretrained weight normalization mean/std for the three channels, while documenting that these channels are MRI neighbors rather than RGB. Offer a consistent scalar MRI normalization alternative as a later ablation. Save the chosen policy in the checkpoint.

Cache deterministic processed **full series**, before stochastic center sampling and augmentation. Do not cache one randomly sampled bag and reuse it for every epoch. Use an efficient array format with metadata, source fingerprints and preprocessing hash. Detect stale caches and write atomically. Do not store labels or reports in the image cache.

Generate a QC gallery before full training: original and processed views, crop boundaries, spacing/FOV, first/middle/last slices, series selection and quality flags. Include missing planes, short stacks, oblique and problematic cases. Show actual example triplets the model receives. At minimum, inspect/report dimensions, finite values and coverage; do not fabricate visual verification if images are unavailable.

Per-series normalization can use the current series at inference. Any population-level learned preprocessing statistic must be fitted on the training fold only.

## 6. Build 2.5D bags and coherent augmentation

The dataset returns:

- `images`: `[P, S, K, H, W]`, with P=3, S=24 and K=3 initially.
- `slice_valid_mask`: `[P, S]` boolean.
- `series_present_mask`: `[P]` boolean.
- `targets`: `[12]` finite values in [0,1].
- `label_weights`: `[12]` nonnegative values.
- Study ID and metadata needed for logging, separately from model inputs.

Choose center indices from the geometrically sorted original stack. For center i, use `[i-1, i, i+1]`, clipped at the original stack boundaries. Do not stack neighbors from the sparsely sampled list of centers.

For a series with at least S slices, divide the full index range into S bins: sample one center per bin during training and use deterministic bin-midpoint centers in validation. For shorter series, use each original center once and pad extra center slots with `slice_valid_mask=False`. Edge repetition within a valid triplet is allowed; padded centers must not receive extra pooling weight.

Missing series receive padded input slots and false masks. Record or exclude all-missing studies explicitly; never train them as normal examples. Validation must not silently drop them and report improved metrics on survivors. Default to failing the run's data-quality gate until the issue is resolved or an explicit, separately reported fallback policy is selected.

Use modest configurable in-plane rotation, translation and scaling plus mild intensity/noise augmentation. Draw one spatial transform per series and apply it consistently across all slices and channels. Avoid unreviewed laterality flips and independent RGB color jitter. Validation has no random augmentation or TTA. Seed workers and samplers reproducibly, without promising bitwise equivalence across hardware.

## 7. Exact model: EfficientNet-B0 plus masked mean/max pooling

Use `torchvision.models.efficientnet_b0` with explicit approved ImageNet weights, or an explicitly supplied local checkpoint. Support offline weight loading. Never silently fall back to random initialization if pretrained weights are unavailable; expose `weights=none` as an intentional option.

Remove the original classifier and use the feature extractor plus global spatial average pooling to obtain 1280 features per valid triplet.

Tensor flow:

1. Batched input: `[B, P, S, 3, H, W]`.
2. Gather only valid triplets and encode as `[N_valid, 3, H, W]`.
3. Restore features to `[B, P, S, 1280]` using differentiable operations.
4. Compute masked mean and feature-wise masked maximum over S, per slot.
5. Concatenate mean and max: `[B, P, 2560]`.
6. Concatenate fixed-order slot vectors and P presence flags. At P=3: 7683 features.
7. Head: `Linear(7683,256) -> ReLU -> Dropout(0.2) -> Linear(256,12)`.

Return logits, not sigmoid probabilities, during training. Apply sigmoid for metrics and exported predictions. There is no softmax across targets.

Padding must not enter the mean denominator or max. A completely missing series must produce an exactly zero finite feature vector. Do not compute an all-masked max/softmax that creates infinities/NaNs. Avoid encoding fake missing images; otherwise BatchNorm may learn from padding.

For small study batches, default to frozen BatchNorm running statistics in the pretrained encoder, with affine parameters still trainable. Reapply that policy after `model.train()`. Keep other training behavior explicit. Allow trainable running statistics as a controlled configuration option.

Support encoder chunking for inference. Explain that naive chunking during training may retain all computation graphs and does not guarantee low memory. Do not detach features to save memory unless explicitly implementing a head-only diagnostic mode. Profile peak memory; expose smaller microbatches and optional activation checkpointing if actually required.

## 8. Labels, uncertainty and masked BCE

Adapt the current extraction statuses:

- `positive`: target 1, weight 1.
- `negative` with `basis=explicit_absence` or `below_threshold`: target 0, weight 1.
- `negative` with `basis=borderline`: configurable; default excluded with weight 0, with a documented `as_negative` ablation.
- `uncertain` (including conflict/insufficient detail/not assessed): finite placeholder target 0, weight 0.
- `not_mentioned`: placeholder 0, weight 0 by default. Expose weak-negative weights 0.2 and 1 as explicit later experiments.
- Failed extraction, missing row or empty numeric value: unknown, never silently negative.

Prefer the actual details/status schema when available. If only a wide numeric export is supplied, missing remains unknown; require the appropriate export to exclude borderline cases. Do not reconstruct lost status information from a binary value. A generic review flag or evidence-format repair is not by itself a clinical uncertainty label.

Allow explicitly provided soft targets in [0,1] separately from confidence weights. Do not invent soft values from LLM confidence, statuses or the validation reference. Do not overwrite source labels.

Use `binary_cross_entropy_with_logits(..., reduction='none')` in float32, then apply label weights. Implement class-wise normalization:

`L_c = sum_i(w_ic * BCE(z_ic,y_ic)) / sum_i(w_ic)` for classes with positive total weight; average over those valid classes.

Use finite targets before evaluating BCE; multiplying NaN by zero is not safe. Return a differentiable zero for an empty-supervision microbatch, and skip optimizer/scheduler updates when the whole accumulation window has no supervision. Track empty supervision explicitly. Start without additional positive-class weighting or oversampling.

Document that accumulating class-normalized microbatch losses is not exactly equivalent to class normalization over one large batch with different masks. Choose a consistent policy, scale partial final accumulation windows correctly and log epoch loss using accumulated class numerators/denominators rather than misleading unweighted batch averages.

Report known/unknown, positive/negative, borderline and weighted counts by target and split before training. Detect targets with no usable supervision or no learnable positive/negative support. Do not silently report them as successfully trained.

## 9. Splits and leakage control

Create and persist `splits.csv` before training. Use reliable patient groups when available, keeping both knees, repeated studies and all series in one partition. Do not assume DICOM PatientID is globally trustworthy: audit missing, constant, site-reused and inconsistent values. Do not use StudyInstanceUID as proof of patient independence.

If no trustworthy grouping exists, use study-level splitting with duplicate checks and prominently document the limitation. Identical images and exact duplicated reports should be audited across partitions; a repeated report is not automatically proof of the same patient.

Start with three group-disjoint folds and run fold 0 first. Use multilabel-aware group balancing where feasible, with explicit per-target support checks. Do not call StratifiedGroupKFold with an unsupported multi-label matrix. Do not encode missing targets as negatives just for stratification. Provide an all-fold option without requiring it for the first run.

Freeze evaluation targets, masks and study IDs before testing alternative training-label policies. Do not let an experiment's training weights silently redefine its validation reference.

If the 58 reference studies are supplied, reserve them and their known patient groups from main image training by default and use them as an optional diagnostic audit, not the early-stopping criterion. Note that prior prompt tuning on these reports limits independence. If the user explicitly changes this policy, record it and do not report in-sample predictions as held-out gold performance.

If only the 58-study labeling export is available, fail the full-training readiness check with a clear explanation and retain the synthetic smoke-test path. An explicit diagnostic override may use those images for a tiny overfit check, but must label the run as in-sample debugging, disable independent gold-performance claims, and not alter the normal split manifest. Do not fabricate a credible competition CV result from that limited input.

## 10. Training loop

Implement a complete loop with:

- AdamW parameter groups for encoder and head, configurable LR and weight decay.
- Two warmup epochs followed by cosine decay, with explicit scheduler-step semantics.
- Gradient accumulation, correct last-window scaling, zero_grad and gradient clipping after unscaling when applicable.
- Hardware-aware AMP: bf16 when supported, fp16 with GradScaler otherwise, float32 fallback. Use APIs appropriate for pinned PyTorch versions.
- Validation once per epoch; early stopping and best checkpoint based only on the frozen local macro ROC-AUC.
- A clear failure/alternative policy if no validation target has a defined AUC; do not silently choose a checkpoint using NaN or the gold audit.
- Progress logging: loss, learning rates, completed optimizer steps, throughput, elapsed time, class support, data failures and peak GPU memory.
- `best.pt` and `last.pt`, including model, optimizer, scheduler, scaler, epoch, best score, RNG state, target order, split/label/preprocessing versions and configuration.
- Epoch-boundary resume with documented reproducibility limits and compatibility checks. No claim of exact mid-epoch resume unless implemented.

Provide three modes: synthetic smoke test, 8-16-study overfit check without augmentation, and normal fold training. Do not present tiny-subset overfit metrics as validation performance.

## 11. Validation metrics and outputs

Use `model.eval()` and `torch.inference_mode()`. Accumulate predictions for the entire fold before computing ranking metrics. Do not average per-batch AUC.

Evaluate each target on its fixed, valid binary reference entries:

- ROC-AUC and known positive/negative counts.
- Average precision, named explicitly as AP, plus prevalence.
- Precision, recall/sensitivity, specificity and F1 at a fixed diagnostic threshold of 0.5, with confusion counts and explicit undefined-denominator handling.
- Coverage and unresolved counts.

Return NA for ROC-AUC when there is only one class. Never substitute 0.5. For AP, flag degenerate support and apply a documented policy, rather than silently averaging misleading values. Report macro metrics with `n_defined_targets / 12` and a fixed included-target policy. Distinguish the local evaluable-target macro from any unverified official metric.

Do not pass soft targets directly to binary ROC-AUC/AP or threshold them into an invented reference. With only soft validation targets, report the compatible loss and explain that a fixed binary evaluation reference is needed for binary ranking metrics.

Save one prediction per study and target, including fold, score, reference, validity and checkpoint provenance. A single held-out fold is not complete OOF coverage; merge into `oof_predictions.csv` only when all intended folds are complete and verified disjoint. Optionally export patient-group bootstrap intervals with fixed seed; do not make them a prerequisite for a first run.

Save `history.csv`, `metrics_per_class.csv`, `validation_predictions.csv`, `run_summary.json`, learning curves and per-class metric plots. Include a simple offline HTML report linking QC images, class counts, metrics and selected high-confidence disagreements. Keep disagreement review out of any locked final evaluation process.

Do not tune thresholds on the same data and present the resulting threshold metrics as independent performance. Do not use a repaired/pseudo-labeled target table as evidence that its producing model improved.

## 12. Deliverables and verification

Provide a small package with clear responsibilities, for example:

- `config.yaml`, dependency file and `README.md`.
- Source modules for schema/labels, manifest, preprocessing/cache, splits, dataset, model, loss/metrics and training.
- CLI entry points for manifest building, preprocessing, split creation, training and checkpoint evaluation/prediction.
- A walkthrough notebook that imports those modules, prints input/schema checks, creates a QC gallery, runs the tiny checks and starts one fold. Avoid duplicating the whole implementation in notebook cells.

Provide focused checks for material failure modes: geometric slice sorting, true neighboring triplets, padding invariance, missing slots, masked-label gradients, accumulation behavior, sigmoid outputs, single-class AUC, patient-group separation, label joins and checkpoint reload. Use synthetic data for these checks when real images are absent. Do not mock away the model/gradient path being checked.

Verify syntax/imports and execute a small forward/backward pass if the environment permits. If a GPU or real dataset is unavailable, state exactly what was checked and what still needs a local run. Never fabricate training curves, scores, timings or successful full-data processing.

Finish with exact commands for this sequence:

1. Configure paths and validate schemas/label coverage.
2. Build the manifest and deterministic cache.
3. Inspect QC and create the immutable folds.
4. Run synthetic and tiny-overfit checks.
5. Train fold 0 and inspect fixed-reference metrics.
6. Resume or evaluate a saved checkpoint; optionally run remaining folds.

Implement the entire baseline, not only the easy parts. Keep unknown dataset-specific requirements explicit and fail with actionable messages instead of silently guessing.

## Technical reference links

Check these against the installed versions when implementing:

- [Torchvision EfficientNet-B0 and pretrained normalization](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [scikit-learn ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [pydicom pixel data and decoding](https://pydicom.github.io/pydicom/stable/tutorials/pixel_data/introduction.html)
- [DICOM image geometry](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html)
