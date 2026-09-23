"""Evaluate a saved checkpoint, export predictions, and merge verified OOF folds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .config import Config
from .constants import CHECKPOINT_VERSION, STUDY_ID, TARGETS
from .dataset import StudyBagDataset, check_study_coverage, collate_studies, enforce_coverage_gate
from .labels import build_label_table, load_reference_table
from .metrics import evaluate_predictions
from .model import build_model
from .preprocess import preprocess_hash
from .splits import ROLE_REFERENCE_HOLDOUT, ROLE_TRAIN_POOL, fold_study_ids, load_splits
from .train import EvaluationReference
from .utils import LOG, atomic_write_dataframe, atomic_write_json, autocast_ctx, select_device


def load_checkpoint_for_inference(cfg: Config, path: str | Path) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Checkpoint version {payload.get('version')} != {CHECKPOINT_VERSION}")
    if payload.get("target_order") != list(TARGETS):
        raise ValueError(
            f"Checkpoint target order differs from the project order.\n"
            f"  checkpoint: {payload.get('target_order')}\n  project:    {list(TARGETS)}"
        )
    stored_cfg = Config(payload.get("config", {}))
    for key in ("data.image_size", "data.series_slots", "data.centers_per_series", "data.encoder_normalization"):
        stored, current = stored_cfg.get_dotted(key), cfg.get_dotted(key)
        if stored is None:
            continue
        differs = list(stored) != list(current) if isinstance(stored, list) else stored != current
        if differs:
            LOG.warning("Checkpoint %s=%s differs from the current config value %s", key, stored, current)
    # Rebuild the architecture without re-downloading pretrained weights.
    build_cfg = cfg.copy()
    build_cfg.model.weights = "none"
    model = build_model(build_cfg, n_slots=len(stored_cfg.get_dotted("data.series_slots", cfg.data.series_slots)))
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


@torch.inference_mode()
def predict_studies(
    cfg: Config,
    model: torch.nn.Module,
    study_ids: Sequence[str],
    label_table=None,
) -> tuple[list[str], np.ndarray]:
    """Deterministic inference: eval mode, no augmentation, no TTA."""
    from torch.utils.data import DataLoader

    spec = select_device(cfg.train.amp)
    model = model.to(spec.device)
    model.eval()
    dataset = StudyBagDataset(cfg, list(study_ids), label_table, train=False)
    from .train import eval_loader_settings

    num_workers, prefetch_factor, _ = eval_loader_settings(cfg)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(cfg.train.eval_batch_studies)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_studies,
        pin_memory=spec.device.type == "cuda",
        **({"prefetch_factor": prefetch_factor} if num_workers > 0 and prefetch_factor else {}),
    )
    ids: list[str] = []
    scores: list[np.ndarray] = []
    for batch in loader:
        with autocast_ctx(spec):
            logits = model(
                batch["images"].to(spec.device),
                batch["slice_valid_mask"].to(spec.device),
                batch["series_present_mask"].to(spec.device),
            )
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        ids.extend(batch["study_ids"])
    stacked = np.concatenate(scores, axis=0) if scores else np.zeros((0, len(TARGETS)), dtype=np.float32)
    return ids, stacked


def evaluate_checkpoint(
    cfg: Config,
    checkpoint: str | Path,
    out_dir: str | Path | None = None,
    partition: str = "validation",
) -> dict:
    """Score a checkpoint on its fold's validation studies, or on the reference hold-out."""
    checkpoint = Path(checkpoint)
    out_dir = Path(out_dir) if out_dir else checkpoint.parent / f"eval_{partition}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, payload = load_checkpoint_for_inference(cfg, checkpoint)
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


def merge_oof(cfg: Config, fold_prediction_files: Sequence[str | Path], out_path: str | Path | None = None) -> Path:
    """Merge per-fold validation predictions into `oof_predictions.csv`.

    Only complete, verified-disjoint coverage is merged: a single held-out fold is not OOF.
    """
    splits = load_splits(cfg)
    expected_folds = sorted(int(f) for f in splits[splits["role"] == ROLE_TRAIN_POOL]["fold"].unique())
    frames = [pd.read_csv(path, dtype={STUDY_ID: "string"}) for path in fold_prediction_files]
    if not frames:
        raise ValueError("No prediction files given")
    merged = pd.concat(frames, ignore_index=True)

    present_folds = sorted(int(f) for f in merged["fold"].unique())
    if present_folds != expected_folds:
        raise ValueError(
            f"OOF merge refused: folds {present_folds} present, {expected_folds} expected. "
            "A single held-out fold is not complete out-of-fold coverage."
        )
    duplicated = merged.duplicated(subset=[STUDY_ID, "target"]).sum()
    if duplicated:
        raise ValueError(f"OOF merge refused: {duplicated} duplicated (study, target) rows - folds are not disjoint.")

    pool_studies = set(splits[splits["role"] == ROLE_TRAIN_POOL][STUDY_ID].astype(str))
    covered = set(merged[STUDY_ID].astype(str))
    missing = pool_studies - covered
    if missing:
        raise ValueError(f"OOF merge refused: {len(missing)} training-pool studies have no prediction.")

    out_path = Path(out_path) if out_path else Path(cfg.paths.output_dir) / "oof_predictions.csv"
    atomic_write_dataframe(merged, out_path)

    pivot_scores = merged.pivot(index=STUDY_ID, columns="target", values="score")[list(TARGETS)].to_numpy()
    pivot_ref = merged.pivot(index=STUDY_ID, columns="target", values="reference")[list(TARGETS)].to_numpy()
    pivot_valid = merged.pivot(index=STUDY_ID, columns="target", values="reference_valid")[list(TARGETS)].to_numpy()
    table, summary = evaluate_predictions(
        pivot_scores.astype(float),
        pivot_ref.astype(float),
        pivot_valid.astype(bool),
        threshold=float(cfg.eval.threshold),
        min_support=int(cfg.eval.min_class_support),
    )
    atomic_write_dataframe(table, out_path.with_name("oof_metrics_per_class.csv"))
    atomic_write_json(out_path.with_name("oof_summary.json"), summary)
    LOG.info("OOF merged: %s (%s)", out_path, summary["defined_fraction"])
    return out_path


def bootstrap_intervals(
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
    n_boot: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Patient-group bootstrap CIs for per-target ROC-AUC. Optional, never a prerequisite."""
    from sklearn.metrics import roc_auc_score

    merged = predictions.merge(splits[[STUDY_ID, "group"]], on=STUDY_ID, how="left")
    rng = np.random.default_rng(seed)
    groups = merged["group"].dropna().unique()
    rows = []
    for target in TARGETS:
        subset = merged[(merged["target"] == target) & merged["reference_valid"].astype(bool)]
        if subset["reference"].nunique() < 2:
            rows.append({"target": target, "roc_auc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0})
            continue
        point = float(roc_auc_score(subset["reference"].astype(int), subset["score"].astype(float)))
        samples = []
        for _ in range(n_boot):
            picked = rng.choice(groups, size=len(groups), replace=True)
            chunk = pd.concat([subset[subset["group"] == g] for g in picked], ignore_index=True)
            if chunk["reference"].nunique() < 2:
                continue
            samples.append(float(roc_auc_score(chunk["reference"].astype(int), chunk["score"].astype(float))))
        if samples:
            low, high = np.percentile(samples, [2.5, 97.5])
        else:
            low = high = np.nan
        rows.append(
            {"target": target, "roc_auc": point, "ci_low": float(low), "ci_high": float(high), "n_boot": len(samples)}
        )
    return pd.DataFrame(rows)
