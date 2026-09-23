"""Merge the per-run LLM `labels_details.csv` exports into one training details file.

The wide `train_v1.csv` collapsed `not_mentioned` and `uncertain` into the same empty
cell. The labelling runs kept both apart in their long `labels_details.csv` export
(one row per study x target with `status` and `basis`); this module stitches those
exports together into the `labels_details_csv` contract of `labels.build_from_details`:

    StudyInstanceUID, target, status, basis, evidence, reason, needs_review, source

Rules:
  * every training study must carry exactly one row per target - a gap or a duplicate
    is an error, never silently filled;
  * the radiologist reference studies take the reference label instead of the LLM one
    (status positive/negative, basis `reference`), exactly as `train_v1.csv` did;
  * the LLM columns are passed through unchanged, so the audit can trace every cell
    back to its evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import Config
from .constants import STATUS_NEGATIVE, STATUS_POSITIVE, STUDY_ID, TARGETS
from .schema import read_id_csv
from .utils import LOG, atomic_write_dataframe, atomic_write_json

KNOWN_STATUSES = {"positive", "negative", "uncertain", "not_mentioned"}
PASSTHROUGH = ["evidence", "reason", "needs_review"]
REFERENCE_BASIS = "reference"


def _read_llm_details(path: Path) -> pd.DataFrame:
    df = read_id_csv(path)
    if "target" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "target"})
    required = [STUDY_ID, "target", "status", "basis"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    keep = required + [c for c in PASSTHROUGH if c in df.columns]
    out = df[keep].copy()
    out["status"] = out["status"].astype(str).str.strip().str.lower()
    out["basis"] = out["basis"].astype("string").str.strip().str.lower()
    out["source"] = str(path)
    unknown_status = sorted(set(out["status"]) - KNOWN_STATUSES)
    if unknown_status:
        raise ValueError(f"{path}: unknown status values {unknown_status}")
    unknown_target = sorted(set(out["target"]) - set(TARGETS))
    if unknown_target:
        raise ValueError(f"{path}: unknown targets {unknown_target}")
    return out


def _reference_rows(reference_csv: Path) -> pd.DataFrame:
    ref = read_id_csv(reference_csv)
    missing = [t for t in TARGETS if t not in ref.columns]
    if missing:
        raise ValueError(f"reference_csv {reference_csv} is missing target columns {missing}")
    long = ref[[STUDY_ID] + TARGETS].melt(id_vars=STUDY_ID, var_name="target", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    bad = long[~long["value"].isin([0.0, 1.0])]
    if len(bad):
        raise ValueError(f"reference_csv has {len(bad)} non-binary cells, e.g. {bad.head(3).to_dict('records')}")
    long["status"] = long["value"].map({1.0: STATUS_POSITIVE, 0.0: STATUS_NEGATIVE})
    long["basis"] = REFERENCE_BASIS
    long["source"] = str(reference_csv)
    return long.drop(columns="value")


def merge_label_details(
    cfg: Config,
    detail_paths: Sequence[str | Path],
    out_path: str | Path | None = None,
) -> Path:
    """Stitch the LLM detail exports together and apply the radiologist reference."""
    if not detail_paths:
        raise ValueError("At least one labels_details.csv is required")
    frames = [_read_llm_details(Path(p)) for p in detail_paths]
    llm = pd.concat(frames, ignore_index=True)

    dup = llm.duplicated(subset=[STUDY_ID, "target"], keep=False)
    if dup.any():
        examples = llm.loc[dup, [STUDY_ID, "target", "source"]].head(6).to_dict("records")
        raise ValueError(f"{int(dup.sum())} duplicated (study, target) rows across the inputs, e.g. {examples}")

    train_ids = sorted(set(read_id_csv(cfg.paths.train_csv)[STUDY_ID].dropna().astype(str)))
    reference_csv = cfg.paths.get("reference_csv")
    reference_ids: set[str] = set()
    if reference_csv and Path(reference_csv).exists():
        ref = _reference_rows(Path(reference_csv))
        reference_ids = set(ref[STUDY_ID])
        llm = llm[~llm[STUDY_ID].isin(reference_ids)]
        merged = pd.concat([llm, ref], ignore_index=True)
    else:
        LOG.warning("No reference_csv configured: the merged file carries LLM labels only.")
        merged = llm

    merged = merged[merged[STUDY_ID].isin(train_ids)]
    counts = merged.groupby(STUDY_ID)["target"].nunique()
    incomplete = sorted(set(train_ids) - set(counts[counts == len(TARGETS)].index))
    if incomplete:
        raise ValueError(
            f"{len(incomplete)} training studies lack a complete set of {len(TARGETS)} targets "
            f"after the merge, e.g. {incomplete[:3]}"
        )

    order = {t: i for i, t in enumerate(TARGETS)}
    merged = merged.assign(_t=merged["target"].map(order)).sort_values([STUDY_ID, "_t"]).drop(columns="_t")
    columns = [STUDY_ID, "target", "status", "basis"] + [c for c in PASSTHROUGH if c in merged.columns] + ["source"]
    merged = merged[columns].reset_index(drop=True)

    out_path = Path(out_path) if out_path else Path(cfg.paths.work_dir) / "labels" / "labels_details_all.csv"
    atomic_write_dataframe(merged, out_path)

    summary = {
        "n_studies": len(train_ids),
        "n_rows": len(merged),
        "n_reference_studies": len(reference_ids & set(train_ids)),
        "inputs": [str(p) for p in detail_paths],
        "status_by_basis": {
            f"{s}/{b}": int(n) for (s, b), n in merged.groupby(["status", "basis"], dropna=False).size().items()
        },
    }
    atomic_write_json(out_path.with_name(out_path.stem + "_summary.json"), summary)
    LOG.info("Merged label details: %s (%d rows, %d studies)", out_path, len(merged), len(train_ids))
    return out_path
