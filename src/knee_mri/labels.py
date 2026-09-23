"""Label construction: extraction statuses -> (target, weight) pairs.

Contract
--------
For every study and every one of the 12 targets we produce
  * `targets[i, c]`  finite value in [0, 1]  (a placeholder 0.0 when unknown), and
  * `weights[i, c]`  >= 0                    (0.0 means "no supervision").

Nothing here invents a label. Unknown stays unknown; a missing row, a failed
extraction and an empty numeric cell are all UNKNOWN with weight 0 - never a
silent negative. Soft targets are only accepted when they are explicitly supplied
and `labels.allow_soft_targets` is on; they are never derived from LLM confidence,
from statuses or from the radiologist reference set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .constants import (
    LABELS_VERSION,
    STATUS_NEGATIVE,
    STATUS_NOT_MENTIONED,
    STATUS_POSITIVE,
    STATUS_UNCERTAIN,
    STUDY_ID,
    TARGETS,
)
from .schema import read_id_csv
from .utils import LOG, atomic_write_dataframe

# Kinds recorded per (study, target) cell for reporting and auditing.
KIND_POSITIVE = "positive"
KIND_NEGATIVE = "negative"
KIND_BORDERLINE = "borderline"
KIND_UNCERTAIN = "uncertain"
KIND_NOT_MENTIONED = "not_mentioned"
KIND_UNKNOWN = "unknown"
KIND_SOFT = "soft"

# `reference`: radiologist-reference cells merged in by merge_label_details.
NEGATIVE_BASES_FIRM = {"explicit_absence", "below_threshold", "reference"}
BORDERLINE_BASES = {"borderline"}


@dataclass
class LabelTable:
    """Study-aligned label matrices plus their provenance."""

    study_ids: list[str]
    targets: np.ndarray  # [N, 12] float32, finite, in [0, 1]
    weights: np.ndarray  # [N, 12] float32, >= 0
    kinds: np.ndarray  # [N, 12] object, see KIND_* above
    source: str
    policy: dict = field(default_factory=dict)
    version: str = LABELS_VERSION

    def __post_init__(self) -> None:
        n, c = len(self.study_ids), len(TARGETS)
        for name, arr in (("targets", self.targets), ("weights", self.weights), ("kinds", self.kinds)):
            if arr.shape != (n, c):
                raise ValueError(f"LabelTable.{name} must have shape {(n, c)}, got {arr.shape}")
        if not np.isfinite(self.targets).all():
            raise ValueError("LabelTable.targets contains non-finite values; placeholders must be finite")
        if (self.targets < 0).any() or (self.targets > 1).any():
            raise ValueError("LabelTable.targets must lie in [0, 1]")
        if (self.weights < 0).any() or not np.isfinite(self.weights).all():
            raise ValueError("LabelTable.weights must be finite and non-negative")

    @property
    def index(self) -> dict[str, int]:
        return {sid: i for i, sid in enumerate(self.study_ids)}

    def subset(self, study_ids: list[str]) -> "LabelTable":
        idx_map = self.index
        missing = [s for s in study_ids if s not in idx_map]
        if missing:
            raise KeyError(f"{len(missing)} study id(s) absent from the label table, e.g. {missing[:3]}")
        rows = [idx_map[s] for s in study_ids]
        return LabelTable(
            study_ids=list(study_ids),
            targets=self.targets[rows].copy(),
            weights=self.weights[rows].copy(),
            kinds=self.kinds[rows].copy(),
            source=self.source,
            policy=dict(self.policy),
            version=self.version,
        )

    def binary_reference(self) -> tuple[np.ndarray, np.ndarray]:
        """The fixed binary evaluation reference: (values in {0,1}, validity mask).

        Only cells that are unambiguously positive or negative are valid. Soft and
        zero-weight cells are excluded - a soft target is never thresholded into an
        invented binary reference.
        """
        valid = np.isin(self.kinds, [KIND_POSITIVE, KIND_NEGATIVE]) & (self.weights > 0)
        values = np.where(self.targets >= 0.5, 1.0, 0.0)
        return values.astype(np.float32), valid

    def to_frame(self) -> pd.DataFrame:
        frames = {STUDY_ID: self.study_ids}
        for c, target in enumerate(TARGETS):
            frames[f"y::{target}"] = self.targets[:, c]
            frames[f"w::{target}"] = self.weights[:, c]
            frames[f"kind::{target}"] = self.kinds[:, c]
        return pd.DataFrame(frames)

    def counts_frame(self) -> pd.DataFrame:
        """Known/unknown, positive/negative, borderline and weighted counts per target."""
        rows = []
        binary_values, binary_valid = self.binary_reference()
        for c, target in enumerate(TARGETS):
            kinds = self.kinds[:, c]
            weights = self.weights[:, c]
            rows.append(
                {
                    "target": target,
                    "n_studies": len(self.study_ids),
                    "n_positive": int((kinds == KIND_POSITIVE).sum()),
                    "n_negative": int((kinds == KIND_NEGATIVE).sum()),
                    "n_borderline": int((kinds == KIND_BORDERLINE).sum()),
                    "n_uncertain": int((kinds == KIND_UNCERTAIN).sum()),
                    "n_not_mentioned": int((kinds == KIND_NOT_MENTIONED).sum()),
                    "n_unknown": int((kinds == KIND_UNKNOWN).sum()),
                    "n_soft": int((kinds == KIND_SOFT).sum()),
                    "n_supervised": int((weights > 0).sum()),
                    "total_weight": float(weights.sum()),
                    "n_eval_positive": int(((binary_values[:, c] == 1) & binary_valid[:, c]).sum()),
                    "n_eval_negative": int(((binary_values[:, c] == 0) & binary_valid[:, c]).sum()),
                }
            )
        df = pd.DataFrame(rows)
        df["prevalence"] = df["n_eval_positive"] / (df["n_eval_positive"] + df["n_eval_negative"]).replace(0, np.nan)
        return df


def _blank_matrices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.zeros((n, len(TARGETS)), dtype=np.float32)
    weights = np.zeros((n, len(TARGETS)), dtype=np.float32)
    kinds = np.full((n, len(TARGETS)), KIND_UNKNOWN, dtype=object)
    return targets, weights, kinds


def _resolve_status(
    status: str | None,
    basis: str | None,
    borderline_policy: str,
    unmentioned_weight: float,
    uncertain_weight: float,
) -> tuple[float, float, str]:
    """Map one extraction status to (target, weight, kind)."""
    status = (status or "").strip().lower()
    basis = (basis or "").strip().lower()

    if status == STATUS_POSITIVE:
        return 1.0, 1.0, KIND_POSITIVE
    if status == STATUS_NEGATIVE:
        if basis in BORDERLINE_BASES:
            if borderline_policy == "as_negative":
                return 0.0, 1.0, KIND_BORDERLINE
            return 0.0, 0.0, KIND_BORDERLINE
        if basis in NEGATIVE_BASES_FIRM or basis == "":
            # An empty basis is treated as a firm negative only for exports that do
            # not carry the basis column at all; see build_from_statuses().
            return 0.0, 1.0, KIND_NEGATIVE
        # Unknown basis vocabulary: do not guess.
        return 0.0, 0.0, KIND_UNKNOWN
    if status == STATUS_UNCERTAIN or status in ("conflict", "insufficient_detail", "not_assessed"):
        return 0.0, float(uncertain_weight), KIND_UNCERTAIN
    if status == STATUS_NOT_MENTIONED:
        return 0.0, float(unmentioned_weight), KIND_NOT_MENTIONED
    return 0.0, 0.0, KIND_UNKNOWN


def unmentioned_weights(cfg: Config) -> dict[str, float]:
    """Per-target weight of a `not_mentioned` cell.

    `labels.unmentioned_weight_per_target` overrides the global `labels.unmentioned_weight`
    for the targets it names; every other target keeps the global value.
    """
    default = float(cfg.labels.unmentioned_weight)
    per_target = cfg.labels.get("unmentioned_weight_per_target") or {}
    return {target: float(per_target.get(target, default)) for target in TARGETS}


def build_from_details(df: pd.DataFrame, study_ids: list[str], cfg: Config) -> LabelTable:
    """Preferred path: long export with one row per (study, target) and a status/basis."""
    idx = {sid: i for i, sid in enumerate(study_ids)}
    targets, weights, kinds = _blank_matrices(len(study_ids))
    target_idx = {t: c for c, t in enumerate(TARGETS)}
    allow_soft = bool(cfg.labels.allow_soft_targets)
    unmentioned = unmentioned_weights(cfg)
    has_basis = "basis" in df.columns
    if not has_basis:
        LOG.warning(
            "labels_details export has no `basis` column: borderline negatives cannot be identified, "
            "so every negative is treated as a firm negative. Supply the basis column or use the "
            "exclude_borderline export to apply labels.borderline_policy."
        )

    for row in df.itertuples(index=False):
        sid = str(getattr(row, STUDY_ID, "") or "")
        target = str(getattr(row, "target", "") or "")
        if sid not in idx or target not in target_idx:
            continue
        i, c = idx[sid], target_idx[target]
        value, weight, kind = _resolve_status(
            getattr(row, "status", None),
            getattr(row, "basis", None) if has_basis else None,
            cfg.labels.borderline_policy,
            unmentioned[target],
            float(cfg.labels.uncertain_weight),
        )
        soft = getattr(row, "soft_target", None)
        if allow_soft and soft is not None and pd.notna(soft):
            soft_value = float(soft)
            if not 0.0 <= soft_value <= 1.0:
                raise ValueError(f"soft_target must be in [0, 1], got {soft_value} for study {sid} target {target}")
            value, weight, kind = soft_value, max(weight, 1.0), KIND_SOFT
        targets[i, c], weights[i, c], kinds[i, c] = value, weight, kind

    return LabelTable(
        study_ids=list(study_ids),
        targets=targets,
        weights=weights,
        kinds=kinds,
        source="details",
        policy=_policy(cfg, has_basis=has_basis),
    )


def build_from_statuses(df: pd.DataFrame, study_ids: list[str], cfg: Config) -> LabelTable:
    """Wide text-status export. Without a basis column every negative is firm."""
    idx = {sid: i for i, sid in enumerate(study_ids)}
    targets, weights, kinds = _blank_matrices(len(study_ids))
    if cfg.labels.borderline_policy == "exclude":
        LOG.warning(
            "labels_statuses export carries no borderline basis; borderline_policy='exclude' cannot be "
            "honoured from this file. Use labels_details_csv or the exclude_borderline numeric export."
        )
    unmentioned = unmentioned_weights(cfg)
    for row in df.itertuples(index=False):
        sid = str(getattr(row, STUDY_ID, "") or "")
        if sid not in idx:
            continue
        i = idx[sid]
        for c, target in enumerate(TARGETS):
            if target not in df.columns:
                continue
            raw = getattr(row, _attr_name(target), None)
            value, weight, kind = _resolve_status(
                None if raw is None or pd.isna(raw) else str(raw),
                None,
                cfg.labels.borderline_policy,
                unmentioned[target],
                float(cfg.labels.uncertain_weight),
            )
            targets[i, c], weights[i, c], kinds[i, c] = value, weight, kind
    return LabelTable(
        study_ids=list(study_ids),
        targets=targets,
        weights=weights,
        kinds=kinds,
        source="statuses",
        policy=_policy(cfg, has_basis=False),
    )


def build_from_wide_numeric(df: pd.DataFrame, study_ids: list[str], cfg: Config, source: str) -> LabelTable:
    """Wide numeric export (`labels_predictions*.csv` or the train.csv target columns).

    Empty cells are UNKNOWN, never negatives. Status information that the export
    dropped cannot be reconstructed from a binary value, so borderline handling has
    to come from choosing the right export file.
    """
    idx = {sid: i for i, sid in enumerate(study_ids)}
    targets, weights, kinds = _blank_matrices(len(study_ids))
    allow_soft = bool(cfg.labels.allow_soft_targets)
    frame = df.set_index(STUDY_ID)
    non_binary_seen = False

    for c, target in enumerate(TARGETS):
        if target not in frame.columns:
            continue
        values = pd.to_numeric(frame[target], errors="coerce")
        for sid, raw in values.items():
            key = str(sid)
            if key not in idx or pd.isna(raw):
                continue
            i = idx[key]
            value = float(raw)
            if value in (0.0, 1.0):
                targets[i, c] = value
                weights[i, c] = 1.0
                kinds[i, c] = KIND_POSITIVE if value == 1.0 else KIND_NEGATIVE
            elif 0.0 < value < 1.0:
                non_binary_seen = True
                if not allow_soft:
                    raise ValueError(
                        f"{source} contains the non-binary value {value} for study {key}, target {target}. "
                        "Set labels.allow_soft_targets=true only if these really are explicitly supplied "
                        "soft targets; otherwise fix the export."
                    )
                targets[i, c] = value
                weights[i, c] = 1.0
                kinds[i, c] = KIND_SOFT
            else:
                raise ValueError(f"{source} value {value} outside [0, 1] for study {key}, target {target}")

    policy = _policy(cfg, has_basis=False)
    policy["non_binary_values_present"] = non_binary_seen
    policy["borderline_note"] = (
        "A wide numeric export cannot express borderline negatives. Point "
        "paths.labels_predictions_exclude_borderline_csv at the matching export when "
        "labels.borderline_policy='exclude'."
    )
    return LabelTable(
        study_ids=list(study_ids),
        targets=targets,
        weights=weights,
        kinds=kinds,
        source=source,
        policy=policy,
    )


def _attr_name(column: str) -> str:
    """itertuples() sanitises column names; mirror that mapping."""
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in column)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe


def _policy(cfg: Config, has_basis: bool) -> dict:
    return {
        "borderline_policy": cfg.labels.borderline_policy,
        "unmentioned_weight": float(cfg.labels.unmentioned_weight),
        "unmentioned_weight_per_target": unmentioned_weights(cfg),
        "uncertain_weight": float(cfg.labels.uncertain_weight),
        "allow_soft_targets": bool(cfg.labels.allow_soft_targets),
        "basis_available": has_basis,
        "version": LABELS_VERSION,
        "target_order": list(TARGETS),
    }


def _pick_source(cfg: Config) -> tuple[str, Path]:
    """Decide which export to use, honouring labels.source and the borderline policy."""
    paths = cfg.paths
    requested = cfg.labels.source
    candidates: list[tuple[str, str | None]] = [
        ("details", paths.get("labels_details_csv")),
        ("statuses", paths.get("labels_statuses_csv")),
    ]
    if cfg.labels.borderline_policy == "exclude":
        candidates.append(("wide_exclude_borderline", paths.get("labels_predictions_exclude_borderline_csv")))
        candidates.append(("wide", paths.get("labels_predictions_csv")))
    else:
        candidates.append(("wide", paths.get("labels_predictions_csv")))
        candidates.append(("wide_exclude_borderline", paths.get("labels_predictions_exclude_borderline_csv")))
    candidates.append(("train_csv", paths.get("train_csv")))

    def _available(name: str, raw: str | None) -> bool:
        if not raw or not Path(raw).exists():
            return False
        if name == "train_csv":
            header = pd.read_csv(raw, nrows=0)
            return set(TARGETS).issubset(header.columns)
        return True

    if requested == "auto":
        for name, raw in candidates:
            if _available(name, raw):
                return name, Path(str(raw))
        raise FileNotFoundError(
            "No usable label export found. Configure one of paths.labels_details_csv, "
            "paths.labels_statuses_csv, paths.labels_predictions_csv, "
            "paths.labels_predictions_exclude_borderline_csv, or supply a train.csv that carries "
            f"the 12 target columns {TARGETS}."
        )

    wanted = {"details": ["details"], "wide": ["wide_exclude_borderline", "wide"], "train_csv": ["train_csv"]}[requested]
    if requested == "details":
        wanted = ["details", "statuses"]
    for name, raw in candidates:
        if name in wanted and _available(name, raw):
            return name, Path(str(raw))
    raise FileNotFoundError(f"labels.source={requested} requested but no matching export is available.")


def build_label_table(cfg: Config, study_ids: list[str]) -> LabelTable:
    """Build the study-aligned label table for the configured source."""
    source, path = _pick_source(cfg)
    LOG.info("Label source: %s (%s)", source, path)
    df = read_id_csv(path)

    if source == "details":
        table = build_from_details(df, study_ids, cfg)
    elif source == "statuses":
        table = build_from_statuses(df, study_ids, cfg)
    else:
        table = build_from_wide_numeric(df, study_ids, cfg, source=source)
    table.policy["source_file"] = str(path)
    return table


def load_reference_table(cfg: Config, study_ids: list[str]) -> LabelTable | None:
    """Load the small radiologist reference set as a separate, audit-only table."""
    raw = cfg.paths.get("reference_csv")
    if not raw or not Path(raw).exists():
        return None
    df = read_id_csv(raw)
    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        raise ValueError(f"reference_csv {raw} is missing target columns {missing}")
    ref_cfg = cfg.copy()
    ref_cfg.labels.allow_soft_targets = False
    table = build_from_wide_numeric(df, study_ids, ref_cfg, source="reference")
    table.policy["source_file"] = str(raw)
    table.policy["usage"] = "diagnostic audit only; never a training label source, never the early-stopping criterion"
    return table


@dataclass
class ReadinessReport:
    ok: bool
    reasons: list[str]
    details: dict[str, Any]

    def log(self) -> None:
        if self.ok:
            LOG.info("Label readiness check passed: %s", self.details)
        else:
            for reason in self.reasons:
                LOG.error("Readiness: %s", reason)

    def raise_if_failed(self) -> None:
        if not self.ok:
            joined = "\n".join(f"  - {r}" for r in self.reasons)
            raise RuntimeError(
                "Full-training readiness check failed:\n"
                f"{joined}\n"
                "Options: (a) supply a label export with adequate coverage, "
                "(b) run the synthetic smoke test (`--mode synthetic`), or "
                "(c) run the explicitly labelled in-sample overfit diagnostic (`--mode overfit`), "
                "whose metrics must not be reported as held-out performance."
            )


def check_training_readiness(cfg: Config, table: LabelTable, n_total_studies: int) -> ReadinessReport:
    """Decide whether the labelled coverage and class support allow a real fold run."""
    counts = table.counts_frame()
    supervised_studies = int((table.weights.sum(axis=1) > 0).sum())
    reasons: list[str] = []

    min_studies = int(cfg.labels.min_labeled_studies)
    if supervised_studies < min_studies:
        reasons.append(
            f"only {supervised_studies} of {n_total_studies} studies carry any supervision "
            f"(labels.min_labeled_studies={min_studies}). A small export is enough for a clearly "
            "identified smoke test, not for full training."
        )

    min_pos = int(cfg.labels.min_positives_per_target)
    weak = counts[(counts["n_eval_positive"] < min_pos) | (counts["n_eval_negative"] < min_pos)]
    if len(weak) == len(TARGETS):
        reasons.append(
            f"no target reaches {min_pos} known positives and {min_pos} known negatives; "
            "nothing is learnable or evaluable yet."
        )
    elif len(weak):
        LOG.warning(
            "Targets without usable support (min %d pos/neg): %s",
            min_pos,
            ", ".join(f"{r.target}(+{r.n_eval_positive}/-{r.n_eval_negative})" for r in weak.itertuples()),
        )

    dead = counts[counts["n_supervised"] == 0]
    if len(dead):
        LOG.warning("Targets with no supervision at all: %s", list(dead["target"]))

    details = {
        "supervised_studies": supervised_studies,
        "total_studies": n_total_studies,
        "coverage_fraction": round(supervised_studies / max(n_total_studies, 1), 4),
        "targets_without_support": list(weak["target"]),
        "targets_without_supervision": list(dead["target"]),
        "label_source": table.source,
    }
    return ReadinessReport(ok=not reasons, reasons=reasons, details=details)


def save_label_artifacts(table: LabelTable, out_dir: str | Path, prefix: str = "labels") -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "table": atomic_write_dataframe(table.to_frame(), out_dir / f"{prefix}_table.csv"),
        "counts": atomic_write_dataframe(table.counts_frame(), out_dir / f"{prefix}_counts.csv"),
    }
    return paths
