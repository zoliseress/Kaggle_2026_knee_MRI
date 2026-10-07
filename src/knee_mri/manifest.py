"""DICOM inventory (manifest) and deterministic per-slot series selection.

The manifest has one row per *volume candidate* - a geometrically coherent stack
inside a series, so mixed echoes or time points are separate rows. Selection then
picks at most one volume per plane slot using image-side information only, with a
transparent score and a stable tie-break.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Config
from .constants import MANIFEST_VERSION, SERIES_ID, STUDY_ID, slot_filter, slot_plane
from .dicom_io import VolumeCandidate, build_volume_candidates, probe_decode
from .utils import LOG, atomic_write_dataframe, atomic_write_json

MANIFEST_COLUMNS = [
    STUDY_ID,
    SERIES_ID,
    "volume_key",
    "volume_id",
    "path",
    "plane",
    "plane_angle_deg",
    "plane_ambiguous",
    "plane_runner_up",
    "n_slices",
    "n_files",
    "rows",
    "cols",
    "row_spacing_mm",
    "col_spacing_mm",
    "slice_spacing_mm",
    "max_gap_ratio",
    "duplicate_positions",
    "irregular_spacing",
    "inplane_ops",
    "oblique_inplane",
    "normal_x",
    "normal_y",
    "normal_z",
    "series_description",
    "scanning_sequence",
    "sequence_variant",
    "scan_options",
    "mr_acquisition_type",
    "echo_time",
    "repetition_time",
    "inversion_time",
    "field_strength",
    "patient_id",
    "laterality",
    "body_part",
    "photometric",
    "transfer_syntax",
    "modality",
    "decode_ok",
    "decode_error",
    "quality_flags",
    "usable",
]


def _volume_row(volume: VolumeCandidate, min_slices: int, decode_probe: bool, monochrome1_policy: str) -> dict:
    header = volume.header
    decode_ok, decode_error = True, ""
    if decode_probe:
        decode_ok, decode_error = probe_decode(volume, monochrome1_policy)
        if not decode_ok:
            volume.flags = sorted(set(volume.flags) | {"decode_failed"})
    return {
        STUDY_ID: volume.study_uid,
        SERIES_ID: volume.series_uid,
        "volume_key": volume.volume_key,
        "volume_id": volume.volume_id,
        "path": str(Path(volume.slices[0].path).parent),
        "plane": volume.plane.plane,
        "plane_angle_deg": round(volume.plane.angle_deg, 2),
        "plane_ambiguous": bool(volume.plane.ambiguous),
        "plane_runner_up": volume.plane.runner_up,
        "n_slices": volume.n_slices,
        "n_files": header.n_files,
        "rows": volume.slices[0].rows,
        "cols": volume.slices[0].cols,
        "row_spacing_mm": round(volume.row_spacing, 5),
        "col_spacing_mm": round(volume.col_spacing, 5),
        "slice_spacing_mm": round(volume.order.median_spacing, 5),
        "max_gap_ratio": round(volume.order.max_gap_ratio, 3),
        "duplicate_positions": volume.order.duplicate_positions,
        "irregular_spacing": bool(volume.order.irregular),
        "inplane_ops": volume.inplane.describe(),
        "oblique_inplane": bool(volume.inplane.oblique),
        "normal_x": round(float(volume.normal[0]), 5),
        "normal_y": round(float(volume.normal[1]), 5),
        "normal_z": round(float(volume.normal[2]), 5),
        "series_description": header.series_description,
        "scanning_sequence": header.scanning_sequence,
        "sequence_variant": header.sequence_variant,
        "scan_options": header.scan_options,
        "mr_acquisition_type": header.mr_acquisition_type,
        "echo_time": header.echo_time,
        "repetition_time": header.repetition_time,
        "inversion_time": header.inversion_time,
        "field_strength": header.magnetic_field_strength,
        "patient_id": header.patient_id,
        "laterality": header.laterality,
        "body_part": header.body_part,
        "photometric": header.photometric,
        "transfer_syntax": header.transfer_syntax,
        "modality": header.modality,
        "decode_ok": bool(decode_ok),
        "decode_error": decode_error,
        "quality_flags": ";".join(volume.flags),
        "usable": bool(volume.usable(min_slices) and decode_ok),
    }


def scan_series_dir(args: tuple[str, dict]) -> list[dict]:
    """Importable worker: inventory one series directory. Safe for ProcessPoolExecutor."""
    series_dir_str, params = args
    series_dir = Path(series_dir_str)
    try:
        candidates, header, series_flags = build_volume_candidates(
            series_dir,
            obliquity_tolerance_deg=float(params["obliquity_tolerance_deg"]),
            max_gap_ratio=float(params["max_gap_ratio"]),
            min_slices=int(params["min_slices"]),
        )
    except Exception as exc:  # a broken series must not kill the whole inventory
        return [
            {
                STUDY_ID: series_dir.parent.name,
                SERIES_ID: series_dir.name,
                "volume_key": "g0",
                "volume_id": f"{series_dir.name}#g0",
                "path": str(series_dir),
                "n_slices": 0,
                "quality_flags": "scan_failed",
                "decode_ok": False,
                "decode_error": f"{exc.__class__.__name__}: {exc}",
                "usable": False,
            }
        ]
    if not candidates:
        return [
            {
                STUDY_ID: series_dir.parent.name,
                SERIES_ID: series_dir.name,
                "volume_key": "g0",
                "volume_id": f"{series_dir.name}#g0",
                "path": str(series_dir),
                "n_slices": 0,
                "n_files": header.n_files,
                "quality_flags": ";".join(series_flags) or "no_candidates",
                "decode_ok": False,
                "decode_error": "",
                "usable": False,
            }
        ]
    return [
        _volume_row(
            volume,
            min_slices=int(params["min_slices"]),
            decode_probe=bool(params["decode_probe"]),
            monochrome1_policy=str(params["monochrome1_policy"]),
        )
        for volume in candidates
    ]


def _series_dirs(dicom_root: Path, study_ids: Iterable[str] | None = None) -> list[Path]:
    dirs: list[Path] = []
    wanted = set(study_ids) if study_ids is not None else None
    for study_dir in sorted(p for p in dicom_root.iterdir() if p.is_dir()):
        if wanted is not None and study_dir.name not in wanted:
            continue
        dirs.extend(sorted(p for p in study_dir.iterdir() if p.is_dir()))
    return dirs


def build_manifest(
    cfg: Config,
    study_ids: Iterable[str] | None = None,
    limit: int | None = None,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Inventory every series directory and persist `manifest.csv`."""
    dicom_root = Path(cfg.paths.dicom_root)
    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root not found: {dicom_root}")

    series_dirs = _series_dirs(dicom_root, study_ids)
    if limit is not None:
        series_dirs = series_dirs[:limit]
    params = {
        "obliquity_tolerance_deg": cfg.manifest.obliquity_tolerance_deg,
        "max_gap_ratio": cfg.manifest.max_gap_ratio,
        "min_slices": cfg.manifest.min_slices,
        "decode_probe": cfg.manifest.decode_probe,
        "monochrome1_policy": cfg.data.monochrome1_policy,
    }
    LOG.info("Building manifest for %d series directories (workers=%s)", len(series_dirs), cfg.manifest.workers)

    rows: list[dict] = []
    workers = int(cfg.manifest.workers)
    jobs = [(str(d), params) for d in series_dirs]
    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            rows.extend(scan_series_dir(job))
            if i % 200 == 0:
                LOG.info("  %d/%d series scanned", i, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(scan_series_dir, job): job[0] for job in jobs}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    rows.extend(future.result())
                except Exception as exc:  # pragma: no cover - defensive
                    LOG.error("Series scan crashed for %s: %s", futures[future], exc)
                if i % 200 == 0:
                    LOG.info("  %d/%d series scanned", i, len(jobs))

    manifest = pd.DataFrame(rows)
    for column in MANIFEST_COLUMNS:
        if column not in manifest.columns:
            manifest[column] = pd.NA
    manifest = manifest[MANIFEST_COLUMNS]
    manifest = manifest.sort_values([STUDY_ID, SERIES_ID, "volume_key"], kind="stable").reset_index(drop=True)

    out_path = Path(out_path) if out_path else Path(cfg.paths.work_dir) / "manifest" / "manifest.csv"
    atomic_write_dataframe(manifest, out_path)
    atomic_write_json(
        out_path.with_name("manifest_meta.json"),
        {
            "version": MANIFEST_VERSION,
            "n_rows": int(len(manifest)),
            "n_series": int(manifest[SERIES_ID].nunique()),
            "n_studies": int(manifest[STUDY_ID].nunique()),
            "params": params,
            "dicom_root": str(dicom_root),
        },
    )
    LOG.info("Manifest written: %s (%d volume candidates)", out_path, len(manifest))
    log_manifest_quality(manifest)
    return manifest


def log_manifest_quality(manifest: pd.DataFrame) -> dict:
    """Summarise decode status and quality flags; a systematic decoder failure is an error."""
    total = len(manifest)
    decode_ok = int(manifest["decode_ok"].fillna(False).astype(bool).sum())
    usable = int(manifest["usable"].fillna(False).astype(bool).sum())
    # Drop the empty strings a split of "" produces by filtering, not by replacing them
    # with NaN: replace() on an object column triggers a pandas downcasting FutureWarning.
    exploded = manifest["quality_flags"].fillna("").str.split(";").explode()
    flags = exploded[exploded != ""].value_counts()
    summary = {
        "volumes": total,
        "decode_ok": decode_ok,
        "usable": usable,
        "flag_counts": flags.to_dict(),
    }
    LOG.info("Manifest quality: %s", summary)
    if total and decode_ok / total < 0.5:
        raise RuntimeError(
            f"Only {decode_ok}/{total} volumes decoded. This looks like a systematic decoder problem "
            "(missing pylibjpeg/gdcm plugin for a compressed transfer syntax), not a few corrupt series. "
            "Fix the decoding dependency instead of training on missing inputs."
        )
    return summary


def load_manifest(cfg: Config, path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else Path(cfg.paths.work_dir) / "manifest" / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}. Run `python -m knee_mri.cli build-manifest` first.")
    return pd.read_csv(path, dtype={STUDY_ID: "string", SERIES_ID: "string", "volume_id": "string"})


# --------------------------------------------------------------------------------------
# Series selection
# --------------------------------------------------------------------------------------


def _is_localizer(description: str, patterns: list[str]) -> bool:
    text = (description or "").lower()
    return any(pattern in text for pattern in patterns)


TE_PREFERENCES = ("t2", "pd")


def _te_score(echo_time: float | None, preference: str = "t2") -> float:
    """Fluid-sensitive PD/T2-like acquisitions have a longer TE. Heuristic, not a sequence oracle.

    preference (selection.fluid_te_preference): "t2" ranks T2-like TE >= 60 ms above PD-like
    25-60 ms (original); "pd" swaps the two (the radiologists prefer PDFS over T2FS).
    """
    if preference not in TE_PREFERENCES:
        raise ValueError(f"selection.fluid_te_preference must be one of {TE_PREFERENCES}, got {preference!r}")
    if echo_time is None or not np.isfinite(echo_time):
        return 0.3  # unknown: neither rewarded nor punished
    if echo_time >= 60:
        return 1.0 if preference == "t2" else 0.85
    if echo_time >= 25:
        return 0.85 if preference == "t2" else 1.0
    if echo_time >= 15:
        return 0.4
    return 0.0


def _resolution_score(row_spacing: float, col_spacing: float) -> float:
    spacing = max(float(row_spacing), float(col_spacing))
    if not np.isfinite(spacing) or spacing <= 0:
        return 0.0
    return float(np.clip((1.2 - spacing) / 0.9, 0.0, 1.0))


# T1 weighting: short TR and short TE (spin echo). On the training data the coronal T1s have
# TR 270-911 ms and TE 5.6-29.2 ms; PD/T2 have TR >= 1000 ms.
T1_MAX_TR_MS = 1000.0
T1_MAX_TE_MS = 30.0


def _header_tokens(value) -> set[str]:
    """Multi-valued DICOM strings as stored in the manifest ('SE', "['SE', 'IR']") -> {'SE', 'IR'}."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return set()
    return set(re.findall(r"[A-Z0-9_]+", str(value).upper()))


def passes_slot_filter(df: pd.DataFrame, slot_filter_name: str) -> pd.Series:
    """Which scored candidates a filtered slot (constants.SLOT_FILTERS) may take.

    Every condition must be positively established from the headers: a missing value
    (TR, TE, ScanningSequence) never passes, so an unreadable header cannot sneak in.
    """
    h = _sequence_headers(df)
    if slot_filter_name == "t1":
        short_tr_te = h["tr"].between(0, T1_MAX_TR_MS, inclusive="neither") & h["te"].between(
            0, T1_MAX_TE_MS, inclusive="neither"
        )
        return h["spin_echo"] & short_tr_te & ~h["inversion"] & ~h["fat_suppressed"]
    if slot_filter_name == "nonfs":
        # Strict: the series csv must say "no" to both; unknown (no row, no csv) never passes.
        # Only a recognised contrast (PD, T1, T2) is admitted; an intermediate weighting such as
        # TR 890 / TE 30 is none of them and stays out.
        known = (h["tr"] > 0) & (h["te"] > 0)
        base = h["spin_echo"] & known & ~h["inversion"] & ~h["fat_suppressed"] & h["meta_not_fs_not_fluid"]
        return base & (slot_filter_priority(df, "nonfs") >= 0)
    raise ValueError(f"unknown slot filter {slot_filter_name!r}")


def slot_filter_priority(df: pd.DataFrame, slot_filter_name: str) -> pd.Series:
    """Rank class of a filtered slot's candidates, compared before the score (higher first).

    nonfs: PD 2 > T1 1 > T2 0 (the radiologists' meniscus/cartilage preference for non-FS PD,
    T1 next for fracture lines and marrow); -1 = none of them, which the nonfs filter refuses.
    Other filters: 0.
    """
    if slot_filter_name != "nonfs":
        return pd.Series(0, index=df.index, dtype=int)
    h = _sequence_headers(df)
    long_tr = h["tr"] >= T1_MAX_TR_MS
    priority = pd.Series(-1, index=df.index, dtype=int)
    priority[long_tr & (h["te"] >= 60)] = 0
    priority[passes_slot_filter(df, "t1")] = 1
    # PD has a long TR and any TE below the T2 range: many sagittal PD TSEs run at TE 9-10 ms.
    priority[long_tr & (h["te"] > 0) & (h["te"] < 60)] = 2
    return priority


def _sequence_headers(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Sequence-type evidence of scored candidates; a missing header is never positive evidence."""
    def column(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    sequence = column("scanning_sequence").map(_header_tokens)
    options = column("scan_options").map(_header_tokens)
    ti = pd.to_numeric(column("inversion_time"), errors="coerce")
    return {
        "tr": pd.to_numeric(column("repetition_time"), errors="coerce"),
        "te": pd.to_numeric(column("echo_time"), errors="coerce"),
        # Spin echo only: GR (also GR+SE hybrids), IR and research/unknown sequences are out.
        "spin_echo": sequence.map(lambda s: "SE" in s and not s & {"GR", "IR"}).astype(bool),
        "inversion": ti > 0,  # an inversion pulse without the IR token (STIR, FLAIR-like)
        "fat_suppressed": (pd.to_numeric(column("fat_suppression"), errors="coerce").fillna(0) > 0)
        | options.map(lambda s: "FS" in s).astype(bool),
        # Raw csv flags (score_candidates' meta_*): only a known 0 counts as "no".
        "meta_not_fs_not_fluid": (pd.to_numeric(column("meta_fat_suppression"), errors="coerce") == 0)
        & (pd.to_numeric(column("meta_fluid_sensitive"), errors="coerce") == 0),
    }


NON_BLOCKING_PENALTY_FLAGS = {
    "irregular_spacing",
    "duplicate_positions",
    "large_gap",
    "ambiguous_plane",
    "oblique_inplane",
    "mixed_matrix_size",
    "missing_pixel_spacing",
    "header_read_failed",
}


def score_candidates(manifest: pd.DataFrame, series_meta: pd.DataFrame | None, cfg: Config) -> pd.DataFrame:
    """Attach a transparent score and its components to every usable volume candidate."""
    weights = cfg.selection.weights
    patterns = [p.lower() for p in cfg.selection.localizer_patterns]
    df = manifest.copy()

    df["is_localizer"] = df["series_description"].fillna("").map(lambda d: _is_localizer(d, patterns))
    # meta_* keep the series csv as given (NaN = unknown) for the slot filters; the score
    # columns read unknown as 0 and may be switched off without touching the filters.
    df["meta_fluid_sensitive"] = np.nan
    df["meta_fat_suppression"] = np.nan
    df["csv_plane"] = pd.NA
    if series_meta is not None and SERIES_ID in series_meta.columns:
        meta = series_meta.drop_duplicates(subset=[SERIES_ID]).set_index(SERIES_ID)
        for column, out in (("Fluid_Sensitive", "meta_fluid_sensitive"), ("Fat_Suppression", "meta_fat_suppression")):
            if column in meta.columns:
                df[out] = pd.to_numeric(df[SERIES_ID].map(meta[column]), errors="coerce")
        if "Anatomical_Plane" in meta.columns:
            df["csv_plane"] = df[SERIES_ID].map(meta["Anatomical_Plane"]).astype("string").str.lower()
    df["fluid_sensitive"] = df["meta_fluid_sensitive"].fillna(0.0)
    df["fat_suppression"] = df["meta_fat_suppression"].fillna(0.0)

    if not bool(cfg.selection.prefer_fluid_sensitive):
        df["fluid_sensitive"] = 0.0

    te_preference = str(cfg.selection.get("fluid_te_preference", "t2"))
    df["score_te"] = df["echo_time"].map(lambda v: _te_score(float(v) if pd.notna(v) else None, te_preference))
    df["score_slices"] = np.clip(
        pd.to_numeric(df["n_slices"], errors="coerce").fillna(0) / float(cfg.data.centers_per_series), 0.0, 1.0
    )
    df["score_resolution"] = [
        _resolution_score(r, c)
        for r, c in zip(
            pd.to_numeric(df["row_spacing_mm"], errors="coerce").fillna(9.9),
            pd.to_numeric(df["col_spacing_mm"], errors="coerce").fillna(9.9),
        )
    ]
    df["score_plane"] = 1.0 - pd.to_numeric(df["plane_angle_deg"], errors="coerce").fillna(90.0) / 90.0
    df["n_penalty_flags"] = (
        df["quality_flags"]
        .fillna("")
        .map(lambda s: len(NON_BLOCKING_PENALTY_FLAGS & set(f for f in s.split(";") if f)))
    )

    df["score"] = (
        float(weights.fluid_sensitive) * df["fluid_sensitive"]
        + float(weights.fat_suppression) * df["fat_suppression"]
        + float(weights.te_pd_t2) * df["score_te"]
        + float(weights.slice_count) * df["score_slices"]
        + float(weights.inplane_resolution) * df["score_resolution"]
        + float(weights.plane_confidence) * df["score_plane"]
        - float(weights.quality_penalty) * df["n_penalty_flags"]
    )
    return df


def select_series(
    cfg: Config,
    manifest: pd.DataFrame,
    series_meta: pd.DataFrame | None = None,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Choose at most one volume per (study, slot). Rules are frozen across train and val.

    A plane slot ("coronal") takes the best-scoring volume of its plane. A filtered slot
    ("coronal_t1") takes the best-scoring volume of its plane that passes the filter and is
    not the volume already chosen for the plane slot, so the two slots never duplicate.
    """
    slots = list(cfg.data.series_slots)
    scored = score_candidates(manifest, series_meta, cfg)
    usable = scored[scored["usable"].fillna(False).astype(bool) & ~scored["is_localizer"]]
    filter_names = {slot_filter(s) for s in slots} - {None}
    filter_masks = {f: passes_slot_filter(usable, f) for f in filter_names}
    priority_by_filter = {f: slot_filter_priority(usable, f) for f in filter_names}
    # Plane slots first: a filtered slot must know what its plane slot took.
    resolution_order = sorted(slots, key=lambda s: slot_filter(s) is not None)

    rows: list[dict] = []
    all_studies = sorted(set(manifest[STUDY_ID].dropna().astype(str)))
    for study in all_studies:
        study_rows = usable[usable[STUDY_ID] == study]
        by_slot: dict[str, dict] = {}
        for slot in resolution_order:
            plane, filter_name = slot_plane(slot), slot_filter(slot)
            candidates = study_rows[study_rows["plane"] == plane]
            if filter_name is not None:
                candidates = candidates[filter_masks[filter_name].loc[candidates.index]]
                taken = by_slot.get(plane, {}).get("volume_id")
                if taken is not None and not pd.isna(taken):
                    candidates = candidates[candidates["volume_id"].astype(str) != str(taken)]
            n_candidates = int(len(candidates))
            if not n_candidates:
                if filter_name is not None:
                    reason = f"no usable {filter_name} candidate besides the {plane} slot's volume"
                else:
                    n_localizers = int(
                        ((scored[STUDY_ID] == study) & (scored["plane"] == plane) & scored["is_localizer"]).sum()
                    )
                    n_unusable = int(
                        ((scored[STUDY_ID] == study) & (scored["plane"] == plane) & ~scored["usable"].fillna(False)).sum()
                    )
                    reason = f"no usable candidate (localizers={n_localizers}, unusable={n_unusable})"
                by_slot[slot] = {
                    STUDY_ID: study,
                    "slot": slot,
                    SERIES_ID: pd.NA,
                    "volume_id": pd.NA,
                    "selected": False,
                    "score": np.nan,
                    "n_candidates": 0,
                    "reason": reason,
                    "quality_flags": "",
                    "n_slices": 0,
                }
                continue
            # Stable deterministic ranking: filter priority desc (constant for plane slots and
            # t1), score desc, then lexicographic ids.
            if filter_name is not None:
                candidates = candidates.assign(filter_priority=priority_by_filter[filter_name].loc[candidates.index])
            else:
                candidates = candidates.assign(filter_priority=0)
            ranked = candidates.sort_values(
                ["filter_priority", "score", SERIES_ID, "volume_key"],
                ascending=[False, False, True, True],
                kind="stable",
            )
            best = ranked.iloc[0]
            by_slot[slot] = {
                STUDY_ID: study,
                "slot": slot,
                SERIES_ID: str(best[SERIES_ID]),
                "volume_id": str(best["volume_id"]),
                "volume_key": str(best["volume_key"]),
                "selected": True,
                "score": float(best["score"]),
                "runner_up_score": float(ranked.iloc[1]["score"]) if len(ranked) > 1 else np.nan,
                "n_candidates": n_candidates,
                "reason": (
                    f"fluid_sensitive={best['fluid_sensitive']:.0f} te={best['echo_time']} "
                    f"slices={best['n_slices']} res={best['row_spacing_mm']:.2f}x{best['col_spacing_mm']:.2f}mm "
                    f"plane_angle={best['plane_angle_deg']:.1f}deg penalties={best['n_penalty_flags']}"
                ),
                "series_description": best["series_description"],
                "filter_priority": int(best["filter_priority"]),
                "normal_x": float(best["normal_x"]) if "normal_x" in best and pd.notna(best["normal_x"]) else np.nan,
                "quality_flags": best["quality_flags"],
                "n_slices": int(best["n_slices"]),
                "path": best["path"],
            }
        rows.extend(by_slot[slot] for slot in slots)

    selection = pd.DataFrame(rows)
    out_path = Path(out_path) if out_path else selection_path(cfg)
    atomic_write_dataframe(selection, out_path)

    present = selection[selection["selected"]]
    per_study = present.groupby(STUDY_ID).size() if len(present) else pd.Series(dtype=int)
    coverage = {
        "studies": len(all_studies),
        "studies_with_all_slots": int((per_study == len(slots)).sum()),
        "studies_with_no_slot": int(len(all_studies) - per_study.gt(0).sum()),
        "selected_per_slot": present.groupby("slot").size().to_dict() if len(present) else {},
    }
    atomic_write_json(out_path.with_name("series_selection_summary.json"), coverage)
    LOG.info("Series selection written: %s | coverage: %s", out_path, coverage)
    return selection


def selection_path(cfg: Config) -> Path:
    configured = cfg.paths.get("series_selection_csv")
    return Path(configured) if configured else Path(cfg.paths.work_dir) / "manifest" / "series_selection.csv"


def load_selection(cfg: Config, path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else selection_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Series selection not found: {path}. Run `build-manifest` first.")
    return pd.read_csv(path, dtype={STUDY_ID: "string", SERIES_ID: "string", "volume_id": "string"})


def patient_group_audit(manifest: pd.DataFrame, max_studies_per_id: int = 20) -> pd.DataFrame:
    """Audit DICOM PatientID before trusting it as a grouping key."""
    df = manifest.dropna(subset=[STUDY_ID]).copy()
    df["patient_id"] = df["patient_id"].fillna("").astype(str).str.strip()
    per_study = df.groupby(STUDY_ID)["patient_id"].agg(lambda s: sorted({v for v in s if v}))
    rows = []
    for study, ids in per_study.items():
        rows.append(
            {
                STUDY_ID: study,
                "patient_id": ids[0] if len(ids) == 1 else "",
                "n_distinct_patient_ids": len(ids),
                "patient_id_missing": len(ids) == 0,
                "patient_id_inconsistent": len(ids) > 1,
            }
        )
    audit = pd.DataFrame(rows)
    if len(audit):
        counts = audit["patient_id"].replace("", np.nan).value_counts()
        oversized = set(counts[counts > max_studies_per_id].index)
        audit["patient_id_site_reused"] = audit["patient_id"].isin(oversized)
        audit["n_studies_for_id"] = audit["patient_id"].map(counts).fillna(0).astype(int)
    else:
        audit["patient_id_site_reused"] = []
        audit["n_studies_for_id"] = []
    return audit
