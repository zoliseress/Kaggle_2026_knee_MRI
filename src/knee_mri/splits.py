"""Immutable cross-validation folds with explicit leakage control.

Grouping policy
---------------
DICOM `PatientID` is audited before it is trusted: missing, inconsistent within a
study, or reused by a whole site (too many studies for one id) disqualifies it. If
no trustworthy grouping exists we fall back to study-level splitting and say so
prominently - a StudyInstanceUID is not proof of patient independence.

Balancing is multilabel-aware but group-first: a greedy assignment keeps every group
intact while evening out per-target known-positive counts across folds. We never
encode missing targets as negatives just to make stratification convenient.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .constants import SPLITS_VERSION, STUDY_ID, TARGETS
from .labels import KIND_POSITIVE, KIND_SOFT, LabelTable
from .utils import LOG, atomic_write_dataframe, atomic_write_json

ROLE_TRAIN_POOL = "train_pool"
ROLE_REFERENCE_HOLDOUT = "reference_holdout"


def splits_path(cfg: Config) -> Path:
    """paths.splits_csv, or the historical <work_dir>/splits/splits.csv when it is unset."""
    configured = cfg.paths.get("splits_csv")
    return Path(configured) if configured else Path(cfg.paths.work_dir) / "splits" / "splits.csv"


@dataclass
class GroupingDecision:
    source: str  # "patient" | "study"
    groups: dict[str, str]
    audit: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def audit_patient_ids(audit_df: pd.DataFrame, max_studies_per_id: int) -> dict:
    """Summarise how trustworthy DICOM PatientID is as a grouping key."""
    if audit_df is None or not len(audit_df):
        return {"available": False}
    total = len(audit_df)
    missing = int(audit_df["patient_id_missing"].sum())
    inconsistent = int(audit_df["patient_id_inconsistent"].sum())
    reused = int(audit_df.get("patient_id_site_reused", pd.Series(dtype=bool)).sum())
    distinct = int(audit_df.loc[audit_df["patient_id"] != "", "patient_id"].nunique())
    multi_study = int((audit_df["n_studies_for_id"] > 1).sum()) if "n_studies_for_id" in audit_df else 0
    return {
        "available": True,
        "studies": total,
        "missing_patient_id": missing,
        "inconsistent_within_study": inconsistent,
        "site_reused_ids": reused,
        "distinct_patient_ids": distinct,
        "studies_sharing_an_id": multi_study,
        "constant_patient_id": bool(distinct == 1 and total > 1),
        # A PatientID that never repeats is a per-study pseudonym, not a patient key: it
        # produces groups that are 1:1 with studies and therefore protects nothing.
        "unique_per_study": bool(missing == 0 and multi_study == 0 and distinct == total and total > 1),
        "max_studies_per_id": max_studies_per_id,
    }


def decide_grouping(cfg: Config, study_ids: list[str], patient_audit: pd.DataFrame | None) -> GroupingDecision:
    mode = cfg.split.grouping
    max_per_id = int(cfg.split.max_studies_per_patient_id)
    audit = audit_patient_ids(patient_audit, max_per_id)
    warnings: list[str] = []

    def study_level() -> GroupingDecision:
        warnings.append(
            "No trustworthy patient grouping: splitting at STUDY level. Two studies of the same "
            "patient (e.g. both knees or a follow-up) can therefore land in different folds, which "
            "inflates the local CV. Treat the reported score accordingly."
        )
        return GroupingDecision("study", {s: s for s in study_ids}, audit, warnings)

    if mode == "study":
        return study_level()

    if not audit.get("available"):
        if mode == "patient":
            raise ValueError("split.grouping='patient' requires a manifest with PatientID; none was found.")
        return study_level()

    if audit.get("unique_per_study"):
        warnings.append(
            f"DICOM PatientID is unique for every one of the {audit['studies']} studies "
            f"({audit['distinct_patient_ids']} distinct ids, none shared). It is a per-study "
            "pseudonym, not a patient key: grouping by it is identical to study-level splitting "
            "and gives NO protection against two studies of the same person landing in different "
            "folds. Reporting the grouping as study-level so the limitation stays visible."
        )
        if mode == "auto":
            return study_level()

    trustworthy = (
        audit["missing_patient_id"] == 0
        and audit["inconsistent_within_study"] == 0
        and audit["site_reused_ids"] == 0
        and not audit["constant_patient_id"]
    )
    if not trustworthy and mode == "auto":
        warnings.append(f"PatientID audit rejected the key: {audit}")
        return study_level()
    if not trustworthy and mode == "patient":
        warnings.append(f"split.grouping='patient' forced despite a failed audit: {audit}")

    mapping = {}
    lookup = patient_audit.set_index(STUDY_ID)["patient_id"].to_dict()
    uncovered = [s for s in study_ids if s not in lookup]
    if uncovered:
        warnings.append(
            f"The PatientID audit covers only {len(study_ids) - len(uncovered)} of {len(study_ids)} studies. "
            f"The remaining {len(uncovered)} fall back to study-level groups, so studies of the same patient "
            "can land in different folds. Build the manifest for every study to close this gap."
        )
        audit["studies_not_in_audit"] = len(uncovered)
    for study in study_ids:
        pid = str(lookup.get(study, "") or "")
        mapping[study] = f"pid::{pid}" if pid else f"study::{study}"
    return GroupingDecision("patient", mapping, audit, warnings)


def duplicate_report_audit(train_df: pd.DataFrame) -> pd.DataFrame:
    """Find studies that share an identical (normalised) report text."""
    if train_df is None or "Report" not in train_df.columns:
        return pd.DataFrame(columns=[STUDY_ID, "report_hash", "n_studies_with_same_report"])
    text = train_df["Report"].fillna("").astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    digest = text.map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()[:16] if t else "")
    frame = pd.DataFrame({STUDY_ID: train_df[STUDY_ID].astype(str), "report_hash": digest})
    counts = frame[frame["report_hash"] != ""]["report_hash"].value_counts()
    frame["n_studies_with_same_report"] = frame["report_hash"].map(counts).fillna(0).astype(int)
    return frame


def _group_positive_counts(
    groups: dict[str, str], study_ids: list[str], table: LabelTable | None
) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    members: dict[str, list[str]] = {}
    for study in study_ids:
        members.setdefault(groups[study], []).append(study)
    stats: dict[str, np.ndarray] = {}
    index = table.index if table is not None else {}
    for group, studies in members.items():
        vector = np.zeros(len(TARGETS) + 1, dtype=np.float64)
        vector[-1] = len(studies)
        if table is not None:
            for study in studies:
                row = index.get(study)
                if row is None:
                    continue
                vector[: len(TARGETS)] += (table.kinds[row] == KIND_POSITIVE).astype(np.float64)
                # A soft cell adds its positive mass; with no soft cells the folds are unchanged.
                soft = (table.kinds[row] == KIND_SOFT) & (table.weights[row] > 0)
                vector[: len(TARGETS)] += np.where(soft, table.targets[row], 0.0)
        stats[group] = vector
    return members, stats


def assign_folds(
    groups: dict[str, str],
    study_ids: list[str],
    n_folds: int,
    table: LabelTable | None,
    seed: int,
) -> dict[str, int]:
    """Greedy multilabel-aware, group-disjoint fold assignment."""
    members, stats = _group_positive_counts(groups, study_ids, table)
    rng = np.random.default_rng(seed)
    order = sorted(members.keys())
    rng.shuffle(order)  # deterministic tie-break given the seed
    order.sort(key=lambda g: (-float(stats[g][: len(TARGETS)].sum()), -float(stats[g][-1])))

    totals = np.zeros((n_folds, len(TARGETS) + 1), dtype=np.float64)
    assignment: dict[str, int] = {}
    for group in order:
        vector = stats[group]
        costs = []
        for fold in range(n_folds):
            candidate = totals.copy()
            candidate[fold] += vector
            # Balance per-target positives and group sizes: penalise the spread.
            costs.append(float(np.sum(candidate.std(axis=0))))
        best = int(np.argmin(costs))
        totals[best] += vector
        assignment[group] = best

    return {study: assignment[groups[study]] for study in study_ids}


def make_splits(
    cfg: Config,
    study_ids: list[str],
    label_table: LabelTable | None = None,
    patient_audit: pd.DataFrame | None = None,
    train_df: pd.DataFrame | None = None,
    reference_study_ids: list[str] | None = None,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Create and persist `splits.csv`. Existing files are never silently overwritten."""
    study_ids = sorted(set(str(s) for s in study_ids))
    decision = decide_grouping(cfg, study_ids, patient_audit)
    for message in decision.warnings:
        LOG.warning("%s", message)

    holdout: set[str] = set()
    if bool(cfg.split.holdout_reference) and reference_study_ids:
        # Keep the reference studies and their whole groups out of image training.
        ref_groups = {decision.groups[s] for s in reference_study_ids if s in decision.groups}
        holdout = {s for s in study_ids if decision.groups[s] in ref_groups}
        LOG.info(
            "Reference hold-out: %d studies (%d reference + %d same-group) reserved from training.",
            len(holdout),
            len([s for s in reference_study_ids if s in decision.groups]),
            len(holdout) - len([s for s in reference_study_ids if s in decision.groups]),
        )

    pool = [s for s in study_ids if s not in holdout]
    if not pool:
        raise ValueError(
            "Every study ended up in the reference hold-out. Set split.holdout_reference=false only if "
            "you accept that the reference studies are no longer an independent audit set."
        )
    fold_of = assign_folds(decision.groups, pool, int(cfg.split.n_folds), label_table, int(cfg.seed))

    rows = []
    for study in study_ids:
        in_holdout = study in holdout
        rows.append(
            {
                STUDY_ID: study,
                "group": decision.groups[study],
                "group_source": decision.source,
                "role": ROLE_REFERENCE_HOLDOUT if in_holdout else ROLE_TRAIN_POOL,
                "fold": -1 if in_holdout else int(fold_of[study]),
            }
        )
    splits = pd.DataFrame(rows).sort_values(STUDY_ID, kind="stable").reset_index(drop=True)

    if train_df is not None:
        duplicates = duplicate_report_audit(train_df)
        splits = splits.merge(duplicates, on=STUDY_ID, how="left")
        crossing = _duplicate_reports_across_folds(splits)
        if crossing:
            LOG.warning(
                "%d identical-report clusters span more than one fold. A repeated report is not proof "
                "of the same patient, so they are only reported, not merged.",
                crossing,
            )

    out_path = Path(out_path) if out_path else splits_path(cfg)
    if out_path.exists():
        LOG.warning("Overwriting existing %s - evaluation references change with it.", out_path)
    atomic_write_dataframe(splits, out_path)
    summary = split_summary(splits, label_table)
    atomic_write_json(
        out_path.with_name("splits_meta.json"),
        {
            "version": SPLITS_VERSION,
            "seed": int(cfg.seed),
            "n_folds": int(cfg.split.n_folds),
            "group_source": decision.source,
            "grouping_warnings": decision.warnings,
            "patient_id_audit": decision.audit,
            "holdout_reference": bool(cfg.split.holdout_reference),
            "n_reference_holdout": int(len(holdout)),
            "summary": summary,
        },
    )
    LOG.info("Splits written: %s (%s)", out_path, summary["per_fold"])
    return splits


FIXED_VALIDATION_FOLD = 0
FIXED_TRAIN_FOLD = 1


def make_fixed_split(
    cfg: Config,
    train_ids: list[str],
    validation_ids: list[str],
    out_path: str | Path | None = None,
    sources: dict[str, str] | None = None,
) -> pd.DataFrame:
    """One fixed train/validation partition instead of CV folds.

    Written in the splits.csv schema: validation studies get fold 0, training studies
    fold 1, so `split.fold=0` trains on the first set and validates on the second. The
    two sets must be disjoint - an overlap is a label leak, never silently resolved.
    """
    train_set = set(str(s) for s in train_ids)
    val_set = set(str(s) for s in validation_ids)
    if not train_set or not val_set:
        raise ValueError(f"Both sets need studies: {len(train_set)} training, {len(val_set)} validation.")
    overlap = train_set & val_set
    if overlap:
        raise ValueError(
            f"{len(overlap)} studies are in both the training and the validation set, e.g. {sorted(overlap)[:3]}. "
            "Remove them from the training CSV first."
        )

    rows = [
        {
            STUDY_ID: study,
            "group": f"study::{study}",
            "group_source": "study",
            "role": ROLE_TRAIN_POOL,
            "fold": FIXED_VALIDATION_FOLD if study in val_set else FIXED_TRAIN_FOLD,
        }
        for study in sorted(train_set | val_set)
    ]
    splits = pd.DataFrame(rows)

    out_path = Path(out_path) if out_path else splits_path(cfg)
    atomic_write_dataframe(splits, out_path)
    atomic_write_json(
        out_path.with_name("splits_meta.json"),
        {
            "version": SPLITS_VERSION,
            "mode": "fixed_holdout",
            "group_source": "study",
            "validation_fold": FIXED_VALIDATION_FOLD,
            "n_train": len(train_set),
            "n_validation": len(val_set),
            "sources": sources or {},
        },
    )
    LOG.info("Fixed split written: %s (%d training, %d validation studies)", out_path, len(train_set), len(val_set))
    return splits


def verify_fixed_split(
    splits: pd.DataFrame,
    meta: dict[str, Any],
    train_ids: list[str],
    validation_ids: list[str],
    train_csv: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """Check that an existing fixed split still matches the CSVs a run is started with.

    make-fixed-split runs once, so a later run with a different training or validation
    CSV would otherwise reuse the old split silently. Returns (errors, warnings).
    """
    errors: list[str] = []
    warnings: list[str] = []
    if meta.get("mode") != "fixed_holdout":
        errors.append(f"the split is not a fixed hold-out (mode={meta.get('mode')!r})")
        return errors, warnings

    train_set = set(str(s) for s in train_ids)
    val_set = set(str(s) for s in validation_ids)
    fold = splits["fold"].astype(int)
    split_val = set(splits.loc[fold == FIXED_VALIDATION_FOLD, STUDY_ID].astype(str))
    split_train = set(splits.loc[fold == FIXED_TRAIN_FOLD, STUDY_ID].astype(str))

    overlap = train_set & val_set
    if overlap:
        errors.append(
            f"{len(overlap)} validation studies also have a row in the training CSV, e.g. {sorted(overlap)[:2]} "
            "(wrong training CSV for this hold-out?)"
        )
    if split_val != val_set:
        errors.append(
            f"the split validates {len(split_val)} studies, the validation CSV lists {len(val_set)} "
            f"({len(split_val - val_set)} only in the split, {len(val_set - split_val)} only in the CSV)"
        )
    missing = split_train - train_set
    if missing:
        errors.append(f"{len(missing)} training studies of the split have no row in the training CSV, e.g. {sorted(missing)[:2]}")
    unused = train_set - split_train - val_set
    if unused:
        warnings.append(f"{len(unused)} studies of the training CSV are not in the split and will not train")

    source = meta.get("sources", {}).get("train_csv")
    if train_csv and source and Path(source).resolve() != Path(train_csv).resolve():
        warnings.append(f"the split was made from {source}, this run trains on {train_csv}")
    return errors, warnings


def _duplicate_reports_across_folds(splits: pd.DataFrame) -> int:
    if "report_hash" not in splits.columns:
        return 0
    real = splits[(splits["report_hash"].fillna("") != "") & (splits["n_studies_with_same_report"] > 1)]
    if not len(real):
        return 0
    per_hash = real.groupby("report_hash")["fold"].nunique()
    return int((per_hash > 1).sum())


def split_summary(splits: pd.DataFrame, table: LabelTable | None) -> dict:
    per_fold = splits.groupby("fold").size().to_dict()
    summary: dict[str, Any] = {"per_fold": {int(k): int(v) for k, v in per_fold.items()}}
    if table is not None:
        index = table.index
        rows = []
        for fold, chunk in splits.groupby("fold"):
            positions = [index[s] for s in chunk[STUDY_ID] if s in index]
            if not positions:
                continue
            kinds = table.kinds[positions]
            entry = {"fold": int(fold), "n_studies": len(positions)}
            for c, target in enumerate(TARGETS):
                entry[f"pos::{target}"] = int((kinds[:, c] == KIND_POSITIVE).sum())
            rows.append(entry)
        summary["per_fold_positives"] = rows
    return summary


def load_splits(cfg: Config, path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else splits_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"splits.csv not found: {path}. Run `python -m knee_mri.cli make-splits` first.")
    return pd.read_csv(path, dtype={STUDY_ID: "string"})


def fold_study_ids(splits: pd.DataFrame, fold: int) -> tuple[list[str], list[str]]:
    """(train studies, validation studies) for one fold. The hold-out never trains."""
    pool = splits[splits["role"] == ROLE_TRAIN_POOL]
    train = sorted(pool[pool["fold"] != fold][STUDY_ID].astype(str))
    val = sorted(pool[pool["fold"] == fold][STUDY_ID].astype(str))
    overlap = set(train) & set(val)
    if overlap:
        raise AssertionError(f"Train/validation overlap of {len(overlap)} studies in fold {fold}")
    return train, val


def assert_group_disjoint(splits: pd.DataFrame, fold: int) -> None:
    """Hard check: no group may appear in both partitions of a fold."""
    pool = splits[splits["role"] == ROLE_TRAIN_POOL]
    train_groups = set(pool[pool["fold"] != fold]["group"])
    val_groups = set(pool[pool["fold"] == fold]["group"])
    shared = train_groups & val_groups
    if shared:
        raise AssertionError(f"Fold {fold} is not group-disjoint: {len(shared)} shared group(s), e.g. {list(shared)[:3]}")
