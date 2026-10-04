"""Deterministic preprocessing of full series, and the on-disk cache.

One versioned path is used for both training and validation:
  decode -> canonical in-plane array orientation -> fixed physical crop ->
  robust per-series intensity scaling -> square resample.

The cache stores whole preprocessed *series*, before any stochastic centre sampling
or augmentation, so every epoch can draw a different bag from the same cached data.
Labels and report text never enter the image cache.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Config
from .constants import PREPROCESS_VERSION, STUDY_ID
from .dicom_io import VolumeCandidate, build_volume_candidates, decode_volume
from .utils import LOG, atomic_save_npz, atomic_write_dataframe, atomic_write_json, load_npz_meta, stable_hash

# Config fields that change the pixels. Any change invalidates the cache.
PREPROCESS_KEYS = [
    "data.image_size",
    "data.fov_mm",
    "data.crop_center",
    "data.intensity_percentiles",
    "data.monochrome1_policy",
    "data.cache_dtype",
    "manifest.obliquity_tolerance_deg",
    "manifest.max_gap_ratio",
    "manifest.min_slices",
]


def preprocess_signature(cfg: Config) -> dict:
    payload = {key: cfg.get_dotted(key) for key in PREPROCESS_KEYS}
    payload["version"] = PREPROCESS_VERSION
    return payload


def preprocess_hash(cfg: Config) -> str:
    return stable_hash(preprocess_signature(cfg))


def cache_root(cfg: Config) -> Path:
    return Path(cfg.paths.cache_dir) / f"{PREPROCESS_VERSION}_{preprocess_hash(cfg)}"


def cache_path(cfg: Config, study_uid: str, slot: str) -> Path:
    return cache_root(cfg) / study_uid / f"{slot}.npz"


def source_fingerprint(volume: VolumeCandidate) -> str:
    """Fingerprint the actual source files so a changed export invalidates the cache."""
    entries = []
    for ref in volume.slices:
        path = Path(ref.path)
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        entries.append((path.name, ref.frame_index, size))
    return stable_hash(sorted(entries), length=16)


# --------------------------------------------------------------------------------------
# Geometry-aware crop and resample
# --------------------------------------------------------------------------------------


def estimate_center(stack: np.ndarray, mode: str) -> tuple[float, float]:
    """One crop centre per series, in (row, col) array coordinates.

    `foreground` uses the centroid of the thresholded foreground mask of the middle slices
    (an area centroid: every pixel above 0.25 x p99 counts once, whatever its intensity).
    `foreground_extent` uses the midpoint of the same foreground mask's extent instead: the
    area centroid is pulled towards the larger tissue mass (on sagittal images the posterior
    musculature), which pushed the patella and the suprapatellar recess off the anterior
    edge of the crop in about a third of the studies (`qc-edges`). The extent midpoint sits
    halfway between the skin lines, so neither side is favoured.
    The scan centre is NOT guaranteed to be the knee centre, which is exactly why this
    is configurable and reported in the QC gallery.
    """
    z, h, w = stack.shape
    geometric = ((h - 1) / 2.0, (w - 1) / 2.0)
    if mode == "geometric":
        return geometric

    lo, hi = max(0, z // 3), max(1, (2 * z) // 3)
    middle = stack[lo:hi] if hi > lo else stack
    finite = np.isfinite(middle)
    if not finite.any():
        return geometric
    values = np.where(finite, middle, np.nan)
    high = np.nanpercentile(values, 99.0)
    if not np.isfinite(high) or high <= 0:
        return geometric
    mask = np.nan_to_num(values, nan=0.0) > 0.25 * high
    if mask.sum() < 0.01 * mask.size:
        return geometric
    weights = mask.sum(axis=0).astype(np.float64)  # [H, W] foreground votes across slices
    total = weights.sum()
    if total <= 0:
        return geometric
    if mode == "foreground_extent":
        row_center = _extent_midpoint(weights.sum(axis=1))
        col_center = _extent_midpoint(weights.sum(axis=0))
    else:
        row_idx = np.arange(h, dtype=np.float64)
        col_idx = np.arange(w, dtype=np.float64)
        row_center = float((weights.sum(axis=1) * row_idx).sum() / total)
        col_center = float((weights.sum(axis=0) * col_idx).sum() / total)
    # Keep the centroid inside the image so an outlier cannot drag the crop off anatomy.
    row_center = float(np.clip(row_center, 0.25 * (h - 1), 0.75 * (h - 1)))
    col_center = float(np.clip(col_center, 0.25 * (w - 1), 0.75 * (w - 1)))
    return row_center, col_center


def _extent_midpoint(profile: np.ndarray, tail: float = 0.005) -> float:
    """Midpoint of a 1-D foreground profile's robust extent (its `tail` / `1 - tail` quantiles).

    The quantiles are taken over the foreground votes, so a few stray bright pixels at the
    border cannot stretch the extent the way a plain first/last non-zero index would.
    """
    cumulative = np.cumsum(profile, dtype=np.float64)
    total = cumulative[-1]
    first = int(np.searchsorted(cumulative, tail * total, side="left"))
    last = int(np.searchsorted(cumulative, (1.0 - tail) * total, side="left"))
    return (first + last) / 2.0


def physical_crop(
    stack: np.ndarray,
    row_spacing: float,
    col_spacing: float,
    fov_mm: float,
    center_rc: tuple[float, float],
) -> tuple[np.ndarray, dict]:
    """Crop a fixed physical square, honouring anisotropic in-plane spacing.

    Regions outside the acquired matrix are padded with NaN (i.e. "not acquired"),
    which keeps them out of the intensity statistics.
    """
    z, h, w = stack.shape
    row_px = max(1, int(round(fov_mm / max(float(row_spacing), 1e-6))))
    col_px = max(1, int(round(fov_mm / max(float(col_spacing), 1e-6))))
    row0 = int(round(center_rc[0] - (row_px - 1) / 2.0))
    col0 = int(round(center_rc[1] - (col_px - 1) / 2.0))

    out = np.full((z, row_px, col_px), np.nan, dtype=np.float32)
    src_r0, src_r1 = max(0, row0), min(h, row0 + row_px)
    src_c0, src_c1 = max(0, col0), min(w, col0 + col_px)
    if src_r1 > src_r0 and src_c1 > src_c0:
        dst_r0, dst_c0 = src_r0 - row0, src_c0 - col0
        out[:, dst_r0 : dst_r0 + (src_r1 - src_r0), dst_c0 : dst_c0 + (src_c1 - src_c0)] = stack[
            :, src_r0:src_r1, src_c0:src_c1
        ]
    padded = float(np.isnan(out).mean())
    info = {
        "crop_rows_px": row_px,
        "crop_cols_px": col_px,
        "crop_row0": row0,
        "crop_col0": col0,
        "crop_pad_fraction": round(padded, 4),
        "crop_center_row": round(float(center_rc[0]), 2),
        "crop_center_col": round(float(center_rc[1]), 2),
        "fov_mm": float(fov_mm),
    }
    return out, info


def robust_scale(stack: np.ndarray, percentiles: tuple[float, float]) -> tuple[np.ndarray, dict]:
    """Clip to robust per-series foreground percentiles and scale to [0, 1]."""
    finite = np.isfinite(stack)
    flags: list[str] = []
    if not finite.any():
        return np.zeros_like(stack, dtype=np.float32), {"p_low": 0.0, "p_high": 0.0, "flags": ["empty_foreground"]}

    values = stack[finite]
    high_ref = float(np.percentile(values, 99.0))
    foreground = values[values > 0.1 * high_ref] if high_ref > 0 else values
    if foreground.size < max(64, int(0.01 * values.size)):
        foreground = values
        flags.append("weak_foreground")

    p_low = float(np.percentile(foreground, percentiles[0]))
    p_high = float(np.percentile(foreground, percentiles[1]))
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        # Constant or degenerate image: emit zeros rather than dividing by ~0.
        flags.append("constant_image")
        return np.zeros_like(stack, dtype=np.float32), {"p_low": p_low, "p_high": p_high, "flags": flags}

    scaled = (np.clip(stack, p_low, p_high) - p_low) / (p_high - p_low)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    return scaled, {"p_low": p_low, "p_high": p_high, "flags": flags}


def resample_square(stack: np.ndarray, size: int) -> np.ndarray:
    """Resample `[Z, H, W]` to `[Z, size, size]` with antialiased bilinear interpolation.

    The crop is already a physical square, so this rescales both axes by the same
    physical factor - no anatomical stretching is introduced.
    """
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(np.ascontiguousarray(stack, dtype=np.float32)).unsqueeze(1)
    if tensor.shape[-2:] == (size, size):
        return tensor.squeeze(1).numpy()
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return resized.squeeze(1).clamp_(0.0, 1.0).numpy()


@dataclass
class PreprocessedSeries:
    image: np.ndarray  # [Z, size, size] float32 in [0, 1]
    meta: dict

    @property
    def n_slices(self) -> int:
        return int(self.image.shape[0])


def preprocess_volume(volume: VolumeCandidate, cfg: Config) -> PreprocessedSeries:
    """Run the full deterministic path for one selected volume."""
    stack = decode_volume(volume, monochrome1_policy=cfg.data.monochrome1_policy)
    row_spacing, col_spacing = volume.inplane.apply_spacing(volume.row_spacing, volume.col_spacing)

    center = estimate_center(stack, cfg.data.crop_center)
    cropped, crop_info = physical_crop(stack, row_spacing, col_spacing, float(cfg.data.fov_mm), center)
    scaled, scale_info = robust_scale(cropped, tuple(cfg.data.intensity_percentiles))
    image = resample_square(scaled, int(cfg.data.image_size))

    spacings = np.asarray(volume.order.spacings, dtype=np.float32)
    median = float(volume.order.median_spacing)
    gap_ok = (
        (spacings <= median * float(cfg.manifest.max_gap_ratio)) if median > 0 else np.ones(spacings.shape, dtype=bool)
    )
    meta = {
        "volume_id": volume.volume_id,
        "study_uid": volume.study_uid,
        "series_uid": volume.series_uid,
        "plane": volume.plane.plane,
        "plane_angle_deg": round(volume.plane.angle_deg, 2),
        "n_slices": int(image.shape[0]),
        "image_size": int(cfg.data.image_size),
        "row_spacing_mm": round(float(row_spacing), 5),
        "col_spacing_mm": round(float(col_spacing), 5),
        "slice_spacing_mm": round(median, 5),
        "output_mm_per_px": round(float(cfg.data.fov_mm) / float(cfg.data.image_size), 5),
        "inplane_ops": volume.inplane.describe(),
        "oblique_inplane": bool(volume.inplane.oblique),
        "quality_flags": sorted(set(volume.flags) | set(scale_info.pop("flags", []))),
        "prep_hash": preprocess_hash(cfg),
        "prep_signature": preprocess_signature(cfg),
        "source_fingerprint": source_fingerprint(volume),
        "normalization": "per_series_robust_p{}_p{}".format(*cfg.data.intensity_percentiles),
        **crop_info,
        **scale_info,
    }
    return PreprocessedSeries(image=image.astype(np.float32), meta={**meta, "gap_ok": gap_ok.astype(bool).tolist()})


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------


def write_cache_entry(path: Path, series: PreprocessedSeries, dtype: str) -> Path:
    array = series.image
    if dtype == "float16":
        stored = array.astype(np.float16)
    elif dtype == "uint8":
        stored = np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        stored = array.astype(np.float32)
    meta = {**series.meta, "stored_dtype": dtype}
    return atomic_save_npz(path, {"image": stored}, meta)


def read_cache_entry(path: Path, expected_hash: str | None = None) -> tuple[np.ndarray, dict]:
    """Load a cached series as float32 in [0, 1]; raise on a stale entry."""
    with np.load(path, allow_pickle=False) as npz:
        meta = load_npz_meta(npz)
        stored = npz["image"]
    if expected_hash is not None and meta.get("prep_hash") != expected_hash:
        raise ValueError(
            f"Stale cache entry {path}: prep_hash {meta.get('prep_hash')} != {expected_hash}. "
            "Rebuild the cache or point paths.cache_dir at the matching one."
        )
    dtype = meta.get("stored_dtype", "float32")
    if dtype == "uint8":
        image = stored.astype(np.float32) / 255.0
    else:
        image = stored.astype(np.float32)
    return image, meta


def _find_volume(cfg: Config, series_path: str, volume_key: str) -> VolumeCandidate | None:
    candidates, _, _ = build_volume_candidates(
        Path(series_path),
        obliquity_tolerance_deg=float(cfg.manifest.obliquity_tolerance_deg),
        max_gap_ratio=float(cfg.manifest.max_gap_ratio),
        min_slices=int(cfg.manifest.min_slices),
    )
    for candidate in candidates:
        if candidate.volume_key == volume_key:
            return candidate
    return None


def build_cache_entry(args: tuple[dict, dict]) -> dict:
    """Importable worker: preprocess and cache one (study, slot) entry."""
    row, cfg_dict = args
    from .config import Config as _Config

    cfg = _Config(cfg_dict)
    study, slot = row[STUDY_ID], row["slot"]
    out_path = cache_path(cfg, study, slot)
    result = {STUDY_ID: study, "slot": slot, "path": str(out_path), "status": "ok", "error": "", "n_slices": 0}

    try:
        if out_path.exists():
            try:
                image, meta = read_cache_entry(out_path, expected_hash=preprocess_hash(cfg))
                if meta.get("source_fingerprint") and meta.get("volume_id") == row["volume_id"]:
                    result.update(status="cached", n_slices=int(image.shape[0]))
                    return result
            except Exception:
                pass  # stale or corrupt: rebuild below

        volume = _find_volume(cfg, row["path"], row["volume_key"])
        if volume is None:
            result.update(status="failed", error=f"volume {row['volume_id']} not found under {row['path']}")
            return result
        series = preprocess_volume(volume, cfg)
        if series.n_slices == 0:
            result.update(status="failed", error="no slices after preprocessing")
            return result
        write_cache_entry(out_path, series, str(cfg.data.cache_dtype))
        result.update(n_slices=series.n_slices)
    except Exception as exc:
        result.update(status="failed", error=f"{exc.__class__.__name__}: {exc}")
    return result


def remove_unselected_entries(cfg: Config, unselected: pd.DataFrame) -> int:
    """Delete the cache files of (study, slot) rows with selected=False; returns how many existed."""
    removed = 0
    for study, slot in zip(unselected[STUDY_ID].astype(str), unselected["slot"].astype(str)):
        path = cache_path(cfg, study, slot)
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def build_cache(
    cfg: Config,
    selection: pd.DataFrame,
    study_ids: Iterable[str] | None = None,
    workers: int | None = None,
) -> pd.DataFrame:
    """Preprocess and cache every selected (study, slot) series.

    A (study, slot) the selection leaves empty must not keep an entry from an earlier
    selection: the dataset reads whatever file exists, so such an entry is deleted.
    """
    in_scope = selection
    if study_ids is not None:
        in_scope = selection[selection[STUDY_ID].isin(set(study_ids))]
    selected = in_scope["selected"].fillna(False).astype(bool)
    rows = in_scope[selected].copy()
    if not len(rows):
        raise ValueError("Nothing to cache: the selection table has no selected series.")

    root = cache_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    removed = remove_unselected_entries(cfg, in_scope[~selected])
    if removed:
        LOG.warning("Removed %d cache entries of (study, slot) pairs the selection now leaves empty", removed)
    atomic_write_json(root / "cache_meta.json", preprocess_signature(cfg))
    LOG.info("Building cache at %s for %d selected series", root, len(rows))

    cfg_dict = cfg.to_dict()
    jobs = [
        (
            {
                STUDY_ID: str(r[STUDY_ID]),
                "slot": str(r["slot"]),
                "path": str(r["path"]),
                "volume_key": str(r["volume_key"]),
                "volume_id": str(r["volume_id"]),
            },
            cfg_dict,
        )
        for _, r in rows.iterrows()
    ]

    workers = int(workers if workers is not None else cfg.manifest.workers)
    results: list[dict] = []
    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            results.append(build_cache_entry(job))
            if i % 100 == 0:
                LOG.info("  %d/%d series preprocessed", i, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(build_cache_entry, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if i % 100 == 0:
                    LOG.info("  %d/%d series preprocessed", i, len(jobs))

    report = pd.DataFrame(results)
    atomic_write_dataframe(report, root / "cache_report.csv")
    status_counts = report["status"].value_counts().to_dict()
    LOG.info("Cache build finished: %s", status_counts)
    failed = report[report["status"] == "failed"]
    if len(failed):
        LOG.error("Failed cache entries: %d (see %s)", len(failed), root / "cache_report.csv")
        for _, row in failed.head(5).iterrows():
            LOG.error("  %s/%s: %s", row[STUDY_ID], row["slot"], row["error"])
        if len(failed) > 0.2 * len(report):
            raise RuntimeError(
                f"{len(failed)}/{len(report)} cache entries failed. This is a systematic problem "
                "(decoder plugin, path or geometry), not a few corrupt series."
            )
    return report


def cache_index(cfg: Config) -> pd.DataFrame:
    """Index the cache on disk: one row per (study, slot) with its slice count."""
    root = cache_root(cfg)
    rows = []
    if not root.exists():
        return pd.DataFrame(columns=[STUDY_ID, "slot", "path", "n_slices"])
    for study_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for entry in sorted(study_dir.glob("*.npz")):
            try:
                with np.load(entry, allow_pickle=False) as npz:
                    meta = load_npz_meta(npz)
                    n_slices = int(npz["image"].shape[0])
            except Exception as exc:
                LOG.warning("Unreadable cache entry %s: %s", entry, exc)
                continue
            rows.append(
                {
                    STUDY_ID: study_dir.name,
                    "slot": entry.stem,
                    "path": str(entry),
                    "n_slices": n_slices,
                    "volume_id": meta.get("volume_id", ""),
                    "prep_hash": meta.get("prep_hash", ""),
                }
            )
    return pd.DataFrame(rows)
