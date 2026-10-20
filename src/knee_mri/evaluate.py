"""Evaluate a saved checkpoint, export predictions, and merge verified OOF folds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .config import Config
from .constants import CHECKPOINT_VERSION, DEFAULT_BACKBONE, FIXED_INPUT_BACKBONES, STUDY_ID, TARGETS
from .dataset import StudyBagDataset, check_study_coverage, collate_studies, enforce_coverage_gate
from .encoders import canonical_encoder_model_id
from .labels import build_label_table, load_reference_table
from .metrics import evaluate_predictions, read_reference_subsets, soft_roc_auc, subset_metrics
from .model import build_model
from .preprocess import preprocess_hash
from .splits import ROLE_REFERENCE_HOLDOUT, ROLE_TRAIN_POOL, fold_study_ids, load_splits
from .train import EvaluationReference
from .utils import LOG, atomic_write_dataframe, atomic_write_json, autocast_ctx, select_device


# Input keys the dataset applies at inference time that the model's weights depend on. A
# mismatch feeds the model other inputs than it was trained on - with identical tensor
# shapes for a reordered slot list or another normalisation, so nothing would crash.
INFERENCE_DATA_KEYS = (
    "data.image_size", "data.series_slots", "data.centers_per_series", "data.encoder_normalization",
    "data.laterality_canonical",
)
# Model keys that are not architecture: the pretrained source (the checkpoint replaces it)
# and the memory-only chunking / gradient checkpointing, which the inference host may tune.
# A checkpoint without model.backbone predates the key and is B0 (the config default).
RUNTIME_MODEL_KEYS = ("weights", "encoder_chunk_size", "grad_checkpointing")
ARCHITECTURE_KEYS = (
    "architecture", "n_slots", "n_targets", "spatial_pool", "target_attention", "side_pooling", "feature_dim", "head_in",
    # Encoder identity (absent from checkpoints written before the adapters: then not compared).
    "encoder_model_id", "encoder_adapter_version", "encoder_out_channels", "encoder_input_size",
)
# What a checkpoint written before a key existed was trained with: absent means the default,
# never "skip the check" - an old model must not silently receive mirrored inputs.
INFERENCE_DATA_DEFAULTS = {"data.laterality_canonical": False}


def load_checkpoint_for_inference(
    cfg: Config, path: str | Path, allow_data_overrides: Sequence[str] = ()
) -> tuple[torch.nn.Module, dict]:
    """Rebuild a checkpoint's model for inference.

    The architecture comes from the checkpoint's own config, so any run can be loaded
    whatever `cfg` says about the model. The inputs come from `cfg` (the dataset is built
    from it), so a difference in INFERENCE_DATA_KEYS is refused unless that key is named
    in `allow_data_overrides` - an explicit, logged decision.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Checkpoint version {payload.get('version')} != {CHECKPOINT_VERSION}")
    if payload.get("target_order") != list(TARGETS):
        raise ValueError(
            f"Checkpoint target order differs from the project order.\n"
            f"  checkpoint: {payload.get('target_order')}\n  project:    {list(TARGETS)}"
        )
    unknown = sorted(set(allow_data_overrides) - set(INFERENCE_DATA_KEYS))
    if unknown:
        raise ValueError(f"allow_data_overrides names {unknown}; only {list(INFERENCE_DATA_KEYS)} can be overridden")

    stored_tree = payload.get("config") or {}
    stored_cfg = Config(stored_tree)
    mismatched = {}
    for key in INFERENCE_DATA_KEYS:
        stored = stored_cfg.get_dotted(key, INFERENCE_DATA_DEFAULTS.get(key))
        current = cfg.get_dotted(key, INFERENCE_DATA_DEFAULTS.get(key))
        if stored is None:
            continue
        differs = list(stored) != list(current) if isinstance(stored, (list, tuple)) else stored != current
        if differs:
            mismatched[key] = (stored, current)
    refused = {k: v for k, v in mismatched.items() if k not in allow_data_overrides}
    if refused:
        details = "; ".join(f"{k}: trained {s!r}, now {c!r}" for k, (s, c) in refused.items())
        raise ValueError(
            f"{path}: the inference config would feed the model other inputs than in training ({details}). "
            "Use the training run's config.yaml, or name the keys in allow_data_overrides to accept this."
        )
    for key, (stored, current) in mismatched.items():
        LOG.warning("%s: %s override accepted - trained with %r, inferring with %r", path, key, stored, current)

    # Architecture from the checkpoint, runtime knobs from the caller. weights=none: the bare
    # architecture, no download and no pretrained file - the checkpoint supplies every weight.
    build_cfg = cfg.copy()
    stored_model = stored_tree.get("model")
    if stored_model:
        build_cfg.set_dotted("model.backbone", stored_model.get("backbone", DEFAULT_BACKBONE))
        build_cfg.set_dotted("model.target_attention", bool(stored_model.get("target_attention", False)))
        build_cfg.set_dotted("model.side_pooling", bool(stored_model.get("side_pooling", False)))
        for key, value in stored_model.items():
            if key not in RUNTIME_MODEL_KEYS:
                build_cfg.set_dotted(f"model.{key}", value)
    else:
        LOG.warning("%s carries no model config; building the architecture from the current config", path)
    build_cfg.model.weights = "none"
    backbone = str(build_cfg.model.get("backbone", DEFAULT_BACKBONE))
    if backbone in FIXED_INPUT_BACKBONES:
        # Built for one input size (DINOv2 position embeddings, CoAtNet relative-position
        # tables); no override can change it.
        trained_size = int(stored_cfg.get_dotted("data.image_size"))
        if int(cfg.data.image_size) != trained_size:
            raise ValueError(
                f"{path}: the {backbone} encoder was trained at data.image_size={trained_size}; the inference "
                f"config uses {cfg.data.image_size}. {backbone} inputs must have the trained size."
            )
        build_cfg.data.image_size = trained_size
    model = build_model(build_cfg, n_slots=len(stored_cfg.get_dotted("data.series_slots") or cfg.data.series_slots))
    stored_description, built = dict(payload.get("model_description") or {}), model.describe()
    if "encoder_model_id" in stored_description:
        # Early ConvNeXt checkpoints stored the pretrained tag in the id (see canonical_encoder_model_id).
        stored_description["encoder_model_id"] = canonical_encoder_model_id(backbone, stored_description["encoder_model_id"])
    differing = [k for k in ARCHITECTURE_KEYS if k in stored_description and stored_description[k] != built[k]]
    if differing:
        raise ValueError(
            f"{path}: rebuilt architecture differs in {differing}: "
            + ", ".join(f"{k} {stored_description[k]!r} vs {built[k]!r}" for k in differing)
        )
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


@torch.inference_mode()
def predict_studies(
    cfg: Config,
    model: torch.nn.Module,
    study_ids: Sequence[str],
    label_table=None,
    center_offsets: Sequence[float | None] = (None,),
) -> tuple[list[str], np.ndarray]:
    """Deterministic inference: eval mode, no augmentation.

    `center_offsets`: one slice-window view per entry (the in-bin centre position, None = the
    bin midpoint, see `bin_centers`); the views' logits are averaged before the sigmoid. The
    default single midpoint view is the plain, TTA-free inference.
    """
    from torch.utils.data import DataLoader

    from .train import eval_loader_settings

    if not center_offsets:
        raise ValueError("center_offsets needs at least one view")
    spec = select_device(cfg.train.amp)
    model = model.to(spec.device)
    model.eval()
    num_workers, prefetch_factor, _ = eval_loader_settings(cfg)
    ids: list[str] = []
    logit_sum: np.ndarray | None = None
    for offset in center_offsets:
        dataset = StudyBagDataset(cfg, list(study_ids), label_table, train=False, center_offset=offset)
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(cfg.train.eval_batch_studies)),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_studies,
            pin_memory=spec.device.type == "cuda",
            **({"prefetch_factor": prefetch_factor} if num_workers > 0 and prefetch_factor else {}),
        )
        view_ids: list[str] = []
        view_logits: list[np.ndarray] = []
        for batch in loader:
            with autocast_ctx(spec):
                logits = model(
                    batch["images"].to(spec.device),
                    batch["slice_valid_mask"].to(spec.device),
                    batch["series_present_mask"].to(spec.device),
                )
            view_logits.append(logits.float().cpu().numpy())
            view_ids.extend(batch["study_ids"])
        stacked = (
            np.concatenate(view_logits, axis=0) if view_logits else np.zeros((0, len(TARGETS)), dtype=np.float32)
        )
        if logit_sum is None:
            ids, logit_sum = view_ids, stacked.astype(np.float64)
        else:
            assert view_ids == ids, "views must visit the studies in the same order"
            logit_sum += stacked
    assert logit_sum is not None
    mean_logits = torch.from_numpy(logit_sum / len(center_offsets))
    return ids, torch.sigmoid(mean_logits).numpy().astype(np.float32)


def evaluate_checkpoint(
    cfg: Config,
    checkpoint: str | Path,
    out_dir: str | Path | None = None,
    partition: str = "validation",
    allow_data_overrides: Sequence[str] = (),
) -> dict:
    """Score a checkpoint on its fold's validation studies, or on the reference hold-out."""
    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir) if out_dir else checkpoint.parent / f"eval_{partition}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, payload = load_checkpoint_for_inference(cfg, checkpoint, allow_data_overrides)
    splits = load_splits(cfg)
    all_ids = sorted(set(splits[STUDY_ID].astype(str)))
    label_table = build_label_table(cfg, all_ids)

    if partition == "reference_holdout":
        study_ids = sorted(splits[splits["role"] == ROLE_REFERENCE_HOLDOUT][STUDY_ID].astype(str))
        reference_table = load_reference_table(cfg, all_ids)
        if reference_table is None:
            raise FileNotFoundError("partition='reference_holdout' needs paths.reference_csv")
        reference = EvaluationReference.from_table(reference_table, study_ids)
        note = (
            "Radiologist-reference audit. Prompt tuning happened on these reports, so this is a "
            "diagnostic signal, not an independent gold benchmark, and it is never the early-stopping "
            "criterion."
        )
    else:
        _, study_ids = fold_study_ids(splits, int(cfg.split.fold))
        reference = EvaluationReference.for_run(cfg, label_table, study_ids)
        note = "Held-out fold of the local CV, scored against the frozen extraction-derived reference."

    if not study_ids:
        raise ValueError(f"No studies for partition={partition}")

    coverage = check_study_coverage(cfg, study_ids)
    excluded = enforce_coverage_gate(cfg, coverage, partition)
    if excluded:
        keep = [s for s in study_ids if s not in set(excluded)]
        reference = EvaluationReference(
            study_ids=keep,
            values=reference.values[[study_ids.index(s) for s in keep]],
            valid=reference.valid[[study_ids.index(s) for s in keep]],
            note=reference.note,
        )
        study_ids = keep

    ids, scores = predict_studies(cfg, model, study_ids, label_table)
    order = {sid: i for i, sid in enumerate(ids)}
    aligned = scores[[order[s] for s in reference.study_ids]]

    table, summary = evaluate_predictions(
        aligned,
        reference.values,
        reference.valid,
        threshold=float(cfg.eval.threshold),
        min_support=int(cfg.eval.min_class_support),
    )
    summary.update(
        {
            "partition": partition,
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_best_score": payload.get("best_score"),
            "training_mode": payload.get("mode"),
            "prep_hash_current": preprocess_hash(cfg),
            "prep_hash_checkpoint": payload.get("versions", {}).get("prep_hash"),
            "note": note,
        }
    )

    rows = []
    for i, study in enumerate(reference.study_ids):
        for c, target in enumerate(TARGETS):
            rows.append(
                {
                    STUDY_ID: study,
                    "target": target,
                    "fold": int(cfg.split.fold),
                    "partition": partition,
                    "score": float(aligned[i, c]),
                    "reference": float(reference.values[i, c]),
                    "reference_valid": bool(reference.valid[i, c]),
                    "checkpoint": str(checkpoint),
                    "training_mode": payload.get("mode", ""),
                }
            )
    predictions = pd.DataFrame(rows)
    atomic_write_dataframe(predictions, out_dir / "predictions.csv")
    atomic_write_dataframe(table, out_dir / "metrics_per_class.csv")
    atomic_write_json(out_dir / "summary.json", summary)
    reference.save(out_dir / "reference.csv")
    LOG.info("Evaluation written to %s: %s", out_dir, json.dumps(summary, indent=2, default=str))
    return summary


def merge_oof(
    cfg: Config,
    fold_prediction_files: Sequence[str | Path],
    out_path: str | Path | None = None,
    fold_assignments: str | Path | None = None,
    reference_csv: str | Path | None = None,
) -> Path:
    """Merge per-fold validation predictions into `oof_predictions.csv`.

    Only complete, verified-disjoint coverage is merged: a single held-out fold is not OOF.
    By default the folds and the study pool come from `splits.csv` (`train_pool` role). With
    `fold_assignments` (a STUDY_ID, fold csv, e.g. the reference-545 manifest) they come from that
    table instead: the i-th prediction file is fold i (its own `fold` column - 0 in a fixed-holdout
    run - is overwritten) and must hold exactly that fold's studies.
    With `reference_csv` (a wide STUDY_ID + 12 targets csv, empty = masked) the predictions are
    re-scored against that reference instead of the one the runs validated on: a cell stays
    valid only if it was valid in the run (label mask) and is non-empty in the csv.
    """
    frames = [pd.read_csv(path, dtype={STUDY_ID: "string"}) for path in fold_prediction_files]
    if not frames:
        raise ValueError("No prediction files given")

    if fold_assignments is not None:
        assignment = pd.read_csv(fold_assignments, dtype={STUDY_ID: "string"}, encoding="utf-8-sig")
        if not assignment[STUDY_ID].is_unique:
            raise ValueError(f"OOF merge refused: duplicated studies in {fold_assignments}")
        fold_of = assignment.set_index(STUDY_ID)["fold"].astype(int)
        expected_folds = sorted(fold_of.unique())
        if len(frames) != len(expected_folds):
            raise ValueError(
                f"OOF merge refused: {len(frames)} prediction files for folds {expected_folds}. "
                "A single held-out fold is not complete out-of-fold coverage."
            )
        for fold, (path, frame) in zip(expected_folds, zip(fold_prediction_files, frames)):
            got = set(frame[STUDY_ID].astype(str))
            want = set(fold_of.index[fold_of == fold])
            if got != want:
                raise ValueError(
                    f"OOF merge refused: {path} holds {len(got - want)} studies outside fold {fold} and "
                    f"misses {len(want - got)} of its studies (files must be given in fold order)."
                )
            frame["fold"] = fold
        pool_studies = set(fold_of.index)
    else:
        splits = load_splits(cfg)
        expected_folds = sorted(int(f) for f in splits[splits["role"] == ROLE_TRAIN_POOL]["fold"].unique())
        pool_studies = set(splits[splits["role"] == ROLE_TRAIN_POOL][STUDY_ID].astype(str))
    merged = pd.concat(frames, ignore_index=True)
    if reference_csv is not None:
        override = pd.read_csv(reference_csv, dtype={STUDY_ID: "string"}, encoding="utf-8-sig")
        long = override.melt(id_vars=STUDY_ID, value_vars=list(TARGETS), var_name="target", value_name="_ref")
        missing_ref = set(merged[STUDY_ID].astype(str)) - set(override[STUDY_ID].astype(str))
        if missing_ref:
            raise ValueError(f"{reference_csv}: {len(missing_ref)} predicted studies are missing")
        merged = merged.merge(long, on=[STUDY_ID, "target"], how="left", validate="many_to_one")
        if not merged["_ref"].dropna().isin([0.0, 1.0]).all():
            raise ValueError(f"{reference_csv}: reference values must be 0, 1 or empty")
        merged["reference_valid"] = merged["reference_valid"].astype(bool) & merged["_ref"].notna()
        merged["reference"] = merged["_ref"].fillna(0.0)
        merged = merged.drop(columns="_ref")

    present_folds = sorted(int(f) for f in merged["fold"].unique())
    if present_folds != expected_folds:
        raise ValueError(
            f"OOF merge refused: folds {present_folds} present, {expected_folds} expected. "
            "A single held-out fold is not complete out-of-fold coverage."
        )
    duplicated = merged.duplicated(subset=[STUDY_ID, "target"]).sum()
    if duplicated:
        raise ValueError(f"OOF merge refused: {duplicated} duplicated (study, target) rows - folds are not disjoint.")

    covered = set(merged[STUDY_ID].astype(str))
    missing = pool_studies - covered
    if missing:
        raise ValueError(f"OOF merge refused: {len(missing)} training-pool studies have no prediction.")

    out_path = Path(out_path) if out_path else Path(cfg.paths.output_dir) / "oof_predictions.csv"
    atomic_write_dataframe(merged, out_path)

    def _evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        wide = {
            column: frame.pivot(index=STUDY_ID, columns="target", values=column)[list(TARGETS)].to_numpy()
            for column in ("score", "reference", "reference_valid")
        }
        return evaluate_predictions(
            wide["score"].astype(float),
            wide["reference"].astype(float),
            wide["reference_valid"].astype(bool),
            threshold=float(cfg.eval.threshold),
            min_support=int(cfg.eval.min_class_support),
        )

    table, summary = _evaluate(merged)
    atomic_write_dataframe(table, out_path.with_name("oof_metrics_per_class.csv"))
    atomic_write_json(out_path.with_name("oof_summary.json"), summary)

    # Per-fold macros next to the pooled OOF value: a different statistic (the pooled one also
    # mixes the fold models' calibration), so both are reported.
    fold_rows = []
    for fold in expected_folds:
        _, fold_summary = _evaluate(merged[merged["fold"] == fold])
        fold_rows.append(
            {"fold": str(fold), "n_studies": fold_summary["n_studies"],
             "macro_soft_auc": fold_summary["macro_soft_auc"], "macro_roc_auc": fold_summary["macro_roc_auc"]}
        )
    metrics = ("macro_soft_auc", "macro_roc_auc")
    per_fold = pd.DataFrame(fold_rows)
    fold_rows.append({"fold": "mean", **{m: float(np.nanmean(per_fold[m])) for m in metrics}})
    fold_rows.append({"fold": "std", **{m: float(np.nanstd(per_fold[m], ddof=1)) for m in metrics}})
    fold_rows.append({"fold": "pooled_oof", "n_studies": summary["n_studies"], **{m: summary[m] for m in metrics}})
    fold_table = pd.DataFrame(fold_rows)
    atomic_write_dataframe(fold_table, out_path.with_name("fold_summary.csv"))

    subsets = read_reference_subsets(cfg.paths.get("reference_subsets_csv"))
    subset_per_class, subset_summary = subset_metrics(
        merged, subsets, threshold=float(cfg.eval.threshold), min_support=int(cfg.eval.min_class_support)
    )
    atomic_write_dataframe(subset_summary, out_path.with_name("oof_subset_summary.csv"))
    atomic_write_dataframe(subset_per_class, out_path.with_name("oof_subset_metrics_per_class.csv"))
    LOG.info("OOF merged: %s (%s)\n%s\n%s", out_path, summary["defined_fraction"],
             fold_table.to_string(index=False), subset_summary.to_string(index=False))
    return out_path


def bootstrap_intervals(
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
    n_boot: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Patient-group bootstrap CIs for per-target soft ROC-AUC. Optional, never a prerequisite.

    The soft ROC-AUC equals ROC-AUC on a binary reference and keeps continuous reference
    values instead of truncating them to 0.
    """
    merged = predictions.merge(splits[[STUDY_ID, "group"]], on=STUDY_ID, how="left")
    rng = np.random.default_rng(seed)
    groups = merged["group"].dropna().unique()
    rows = []
    for target in TARGETS:
        subset = merged[(merged["target"] == target) & merged["reference_valid"].astype(bool)]
        point = soft_roc_auc(subset["reference"].to_numpy(float), subset["score"].to_numpy(float))
        if not np.isfinite(point):
            rows.append({"target": target, "soft_auc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0})
            continue
        samples = []
        for _ in range(n_boot):
            picked = rng.choice(groups, size=len(groups), replace=True)
            chunk = pd.concat([subset[subset["group"] == g] for g in picked], ignore_index=True)
            value = soft_roc_auc(chunk["reference"].to_numpy(float), chunk["score"].to_numpy(float))
            if np.isfinite(value):
                samples.append(value)
        if samples:
            low, high = np.percentile(samples, [2.5, 97.5])
        else:
            low = high = np.nan
        rows.append(
            {"target": target, "soft_auc": point, "ci_low": float(low), "ci_high": float(high), "n_boot": len(samples)}
        )
    return pd.DataFrame(rows)
