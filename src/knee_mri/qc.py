"""QC gallery: what the model actually receives, and why that series was chosen.

The gallery deliberately over-samples the awkward cases (missing planes, short stacks,
oblique geometry, quality flags) instead of showing only clean examples. If images are
unavailable the numeric checks still run - dimensions, finite values, coverage - and the
report says that no visual verification was possible rather than pretending otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import Config
from .constants import STUDY_ID
from .dataset import StudyBagDataset, bin_centers, build_bag
from .dicom_io import build_volume_candidates, decode_volume
from .preprocess import cache_path, estimate_center, physical_crop, preprocess_hash, read_cache_entry
from .utils import LOG, atomic_write_dataframe, atomic_write_json


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def pick_qc_studies(selection: pd.DataFrame, manifest: pd.DataFrame, n: int = 12) -> list[str]:
    """Pick a mix of ordinary and problematic studies."""
    selected = selection[selection["selected"].fillna(False).astype(bool)]
    per_study = selected.groupby(STUDY_ID).size()
    picks: list[str] = []

    incomplete = sorted(per_study[per_study < selection["slot"].nunique()].index)[: max(1, n // 4)]
    flagged = sorted(selected[selected["quality_flags"].fillna("") != ""][STUDY_ID].unique())[: max(1, n // 4)]
    short = sorted(selected.sort_values("n_slices").head(max(1, n // 4))[STUDY_ID].unique())
    oblique_ids = sorted(
        manifest[manifest["quality_flags"].fillna("").str.contains("oblique|ambiguous", regex=True)][STUDY_ID].unique()
    )[: max(1, n // 4)]
    ordinary = sorted(per_study[per_study == selection["slot"].nunique()].index)

    for group in (incomplete, flagged, short, oblique_ids, ordinary):
        for study in group:
            if study not in picks:
                picks.append(str(study))
            if len(picks) >= n:
                return picks
    return picks


def series_qc_figure(cfg: Config, study: str, out_path: Path, selection: pd.DataFrame) -> dict:
    """Original vs processed views, crop boundary, and the actual triplets the model sees."""
    plt = _matplotlib()
    slots = list(cfg.data.series_slots)
    rows = selection[(selection[STUDY_ID] == study) & selection["selected"].fillna(False).astype(bool)]
    info: dict = {STUDY_ID: study, "slots": {}, "figure": str(out_path)}

    fig, axes = plt.subplots(len(slots), 6, figsize=(20, 3.4 * len(slots)), squeeze=False)
    for p, slot in enumerate(slots):
        row = rows[rows["slot"] == slot]
        for ax in axes[p]:
            ax.axis("off")
        if not len(row):
            axes[p][0].set_title(f"{slot}: MISSING SLOT", color="crimson", fontsize=11)
            info["slots"][slot] = {"present": False}
            continue

        entry = row.iloc[0]
        cached = cache_path(cfg, study, slot)
        if not cached.exists():
            axes[p][0].set_title(f"{slot}: not cached", color="crimson")
            info["slots"][slot] = {"present": False, "reason": "not cached"}
            continue
        image, meta = read_cache_entry(cached, expected_hash=preprocess_hash(cfg))

        # Original (pre-crop) middle slice, decoded on the fly for comparison.
        original = None
        try:
            candidates, _, _ = build_volume_candidates(
                Path(entry["path"]),
                obliquity_tolerance_deg=float(cfg.manifest.obliquity_tolerance_deg),
                max_gap_ratio=float(cfg.manifest.max_gap_ratio),
                min_slices=int(cfg.manifest.min_slices),
            )
            volume = next((c for c in candidates if c.volume_key == entry["volume_key"]), None)
            if volume is not None:
                stack = decode_volume(volume, monochrome1_policy=cfg.data.monochrome1_policy)
                row_sp, col_sp = volume.inplane.apply_spacing(volume.row_spacing, volume.col_spacing)
                center = estimate_center(stack, cfg.data.crop_center)
                _, crop_info = physical_crop(stack, row_sp, col_sp, float(cfg.data.fov_mm), center)
                original = (stack, crop_info)
        except Exception as exc:  # QC must not fail the pipeline
            LOG.warning("QC could not re-decode %s/%s: %s", study, slot, exc)

        if original is not None:
            stack, crop_info = original
            mid = stack[stack.shape[0] // 2]
            finite = np.nan_to_num(mid, nan=0.0)
            axes[p][0].imshow(finite, cmap="gray")
            axes[p][0].add_patch(
                plt.Rectangle(
                    (crop_info["crop_col0"], crop_info["crop_row0"]),
                    crop_info["crop_cols_px"],
                    crop_info["crop_rows_px"],
                    fill=False,
                    edgecolor="yellow",
                    linewidth=1.5,
                )
            )
            axes[p][0].set_title(
                f"{slot} original {mid.shape[0]}x{mid.shape[1]}\n"
                f"{meta['row_spacing_mm']:.2f}x{meta['col_spacing_mm']:.2f} mm, crop {cfg.data.fov_mm:.0f} mm",
                fontsize=8,
            )

        n_slices = image.shape[0]
        for column, (index, label) in enumerate(
            [(0, "first"), (n_slices // 2, "middle"), (n_slices - 1, "last")], start=1
        ):
            axes[p][column].imshow(image[index], cmap="gray", vmin=0, vmax=1)
            axes[p][column].set_title(f"{slot} processed {label} ({index + 1}/{n_slices})", fontsize=8)

        centers, valid = bin_centers(n_slices, int(cfg.data.centers_per_series), None)
        gap_ok = np.asarray(meta.get("gap_ok", []), dtype=bool)
        bag, _ = build_bag(image, centers, valid, gap_ok if gap_ok.size else None)
        pick = int(np.flatnonzero(valid)[len(np.flatnonzero(valid)) // 2]) if valid.any() else 0
        triplet = bag[pick]
        axes[p][4].imshow(np.concatenate(list(triplet), axis=1), cmap="gray", vmin=0, vmax=1)
        axes[p][4].set_title(f"{slot} model triplet (centre {centers[pick]})", fontsize=8)
        axes[p][5].imshow(triplet.transpose(1, 2, 0), vmin=0, vmax=1)
        axes[p][5].set_title(f"{slot} triplet as RGB\nflags: {','.join(meta.get('quality_flags', [])) or 'none'}", fontsize=8)

        info["slots"][slot] = {
            "present": True,
            "n_slices": int(n_slices),
            "series_uid": meta.get("series_uid"),
            "quality_flags": meta.get("quality_flags", []),
            "row_spacing_mm": meta.get("row_spacing_mm"),
            "col_spacing_mm": meta.get("col_spacing_mm"),
            "slice_spacing_mm": meta.get("slice_spacing_mm"),
            "crop_pad_fraction": meta.get("crop_pad_fraction"),
            "output_mm_per_px": meta.get("output_mm_per_px"),
            "finite": bool(np.isfinite(image).all()),
            "value_range": [float(image.min()), float(image.max())],
        }

    fig.suptitle(f"QC: {study}", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return info


def numeric_qc(cfg: Config, study_ids: Sequence[str]) -> pd.DataFrame:
    """Dimension/finite/coverage checks that work without producing any image."""
    rows = []
    for study in study_ids:
        for slot in cfg.data.series_slots:
            path = cache_path(cfg, str(study), slot)
            if not path.exists():
                rows.append({STUDY_ID: study, "slot": slot, "cached": False})
                continue
            try:
                image, meta = read_cache_entry(path, expected_hash=preprocess_hash(cfg))
            except Exception as exc:
                rows.append({STUDY_ID: study, "slot": slot, "cached": True, "error": str(exc)})
                continue
            rows.append(
                {
                    STUDY_ID: study,
                    "slot": slot,
                    "cached": True,
                    "n_slices": int(image.shape[0]),
                    "height": int(image.shape[1]),
                    "width": int(image.shape[2]),
                    "size_ok": bool(image.shape[1] == image.shape[2] == int(cfg.data.image_size)),
                    "all_finite": bool(np.isfinite(image).all()),
                    "min": float(image.min()),
                    "max": float(image.max()),
                    "mean": float(image.mean()),
                    "constant": bool(float(image.max() - image.min()) < 1e-6),
                    "crop_pad_fraction": meta.get("crop_pad_fraction"),
                    "quality_flags": ";".join(meta.get("quality_flags", [])),
                }
            )
    return pd.DataFrame(rows)


def build_qc_gallery(
    cfg: Config,
    selection: pd.DataFrame,
    manifest: pd.DataFrame,
    n_studies: int = 12,
    out_dir: str | Path | None = None,
    study_ids: Sequence[str] | None = None,
) -> dict:
    """QC figures + numeric checks. `study_ids` replaces the automatic mixed pick."""
    out_dir = Path(out_dir) if out_dir else Path(cfg.paths.work_dir) / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    studies = list(study_ids) if study_ids else pick_qc_studies(selection, manifest, n_studies)
    LOG.info("QC gallery for %d studies -> %s", len(studies), out_dir)

    infos = []
    for study in studies:
        try:
            infos.append(series_qc_figure(cfg, study, out_dir / f"qc_{study[-16:]}.png", selection))
        except Exception as exc:
            LOG.warning("QC figure failed for %s: %s", study, exc)
            infos.append({STUDY_ID: study, "error": f"{exc.__class__.__name__}: {exc}"})

    numeric = numeric_qc(cfg, studies)
    atomic_write_dataframe(numeric, out_dir / "qc_numeric.csv")
    # A slot that is legitimately absent is reported separately; it must not make the
    # numeric verdict look like a preprocessing failure.
    cached = numeric[numeric["cached"].fillna(False).astype(bool)] if len(numeric) else numeric
    checked = cached[cached.get("error", pd.Series(dtype=object)).isna()] if "error" in cached.columns else cached
    summary = {
        "n_studies": len(studies),
        "studies": studies,
        "figures": [i.get("figure") for i in infos if i.get("figure")],
        "visual_verification": bool(any(i.get("figure") for i in infos)),
        "details": infos,
        "n_slots_checked": int(len(checked)),
        "n_slots_missing": int(len(numeric) - len(cached)) if len(numeric) else 0,
        "n_slots_unreadable": int(len(cached) - len(checked)) if len(cached) else 0,
        "numeric_ok": bool(
            len(checked)
            and checked["all_finite"].fillna(False).all()
            and checked["size_ok"].fillna(False).all()
            and not checked["constant"].fillna(True).any()
        ),
    }
    atomic_write_json(out_dir / "qc_summary.json", summary)
    write_qc_checklist(numeric, out_dir / "qc_checklist.csv")
    return summary


# Points a reviewer ticks per study while looking at the gallery (OK / hiba / note).
QC_CHECKLIST_POINTS = [
    "plane_correct",
    "slice_order_monotonic",
    "triplet_true_neighbours",
    "intensity_not_saturated",
    "crop_keeps_mcl_lcl",
    "crop_keeps_patella",
    "crop_keeps_suprapatellar_recess",
]


def write_qc_checklist(numeric: pd.DataFrame, path: Path) -> Path | None:
    """Empty review sheet, one row per (study, slot), pre-filled with the numeric facts."""
    if numeric.empty:
        return None
    facts = [c for c in (STUDY_ID, "slot", "cached", "n_slices", "crop_pad_fraction", "quality_flags") if c in numeric.columns]
    sheet = numeric[facts].copy()
    for point in QC_CHECKLIST_POINTS:
        sheet[point] = ""
    sheet["note"] = ""
    if path.exists():
        LOG.warning("%s exists; not overwriting a checklist that may hold review work", path)
        return None
    sheet.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def batch_sanity_check(cfg: Config, study_ids: Sequence[str], label_table=None) -> dict:
    """Assert the dataset contract on real items before a long run starts."""
    dataset = StudyBagDataset(cfg, list(study_ids)[:4], label_table, train=False)
    if not len(dataset):
        raise ValueError("No studies to check")
    item = dataset[0]
    p, s, k, h, w = item["images"].shape
    checks = {
        "images_shape": list(item["images"].shape),
        "expected_shape": [len(cfg.data.series_slots), int(cfg.data.centers_per_series), 3, int(cfg.data.image_size), int(cfg.data.image_size)],
        "slice_valid_shape": list(item["slice_valid_mask"].shape),
        "series_present_shape": list(item["series_present_mask"].shape),
        "targets_finite": bool(item["targets"].isfinite().all()),
        "weights_non_negative": bool((item["label_weights"] >= 0).all()),
        "images_finite": bool(item["images"].isfinite().all()),
        "padding_is_zero": bool(
            item["images"][~item["slice_valid_mask"]].abs().sum().item() == 0 if (~item["slice_valid_mask"]).any() else True
        ),
    }
    checks["shape_ok"] = checks["images_shape"] == checks["expected_shape"]
    return checks


# --------------------------------------------------------------------------------------
# Crop-edge audit: does the physical crop cut anatomy off?
# --------------------------------------------------------------------------------------

# Canonical in-plane orientation (geometry.CANONICAL_AXES): which array edge is which
# anatomical direction. `None` marks an edge that is always tissue by design (the thigh and
# the calf run out of every sagittal/coronal image) and is therefore not evaluated.
# Coronal/axial columns run towards the PATIENT's left; medial/lateral depends on the knee's
# laterality, so those edges are named by patient side.
CROP_EDGES: dict[str, dict[str, str | None]] = {
    "sagittal": {"top": None, "bottom": None, "left": "posterior", "right": "anterior"},
    "coronal": {"top": None, "bottom": None, "left": "patient_right", "right": "patient_left"},
    "axial": {"top": "anterior", "bottom": "posterior", "left": "patient_right", "right": "patient_left"},
}


def edge_fill_fractions(image: np.ndarray, band_px: int = 4, tissue_threshold: float = 0.1) -> dict[str, float]:
    """Fraction of tissue pixels in each border band, over the middle third of the slices.

    The cache is scaled to [0, 1] and padding is 0, so air and padding fall below the
    threshold; subcutaneous fat and everything brighter lie above it. A band that is mostly
    tissue means the skin line - and possibly more - lies outside the crop.
    """
    z = image.shape[0]
    middle = image[z // 3 : max(z // 3 + 1, 2 * z // 3)]
    tissue = middle > tissue_threshold
    bands = {
        "top": tissue[:, :band_px, :],
        "bottom": tissue[:, -band_px:, :],
        "left": tissue[:, :, :band_px],
        "right": tissue[:, :, -band_px:],
    }
    return {edge: float(band.mean()) for edge, band in bands.items()}


def crop_edge_audit(
    cfg: Config,
    study_ids: Sequence[str],
    out_dir: str | Path,
    cut_threshold: float = 0.3,
    band_px: int = 4,
    tissue_threshold: float = 0.1,
) -> dict:
    """Per (study, slot) edge fill, plus the share of studies whose crop touches each edge."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_hash = preprocess_hash(cfg)
    rows = []
    for study in study_ids:
        for slot in cfg.data.series_slots:
            path = cache_path(cfg, str(study), slot)
            if not path.exists():
                continue
            try:
                image, meta = read_cache_entry(path, expected_hash=expected_hash)
            except Exception as exc:
                LOG.warning("Crop-edge audit: cannot read %s/%s: %s", study, slot, exc)
                continue
            fills = edge_fill_fractions(image, band_px=band_px, tissue_threshold=tissue_threshold)
            row = {STUDY_ID: study, "slot": slot, "crop_pad_fraction": meta.get("crop_pad_fraction")}
            for edge, name in CROP_EDGES.get(slot, {}).items():
                if name is not None:
                    row[f"fill_{name}"] = round(fills[edge], 4)
            rows.append(row)

    table = pd.DataFrame(rows)
    atomic_write_dataframe(table, out_dir / "qc_edges.csv")

    summary: dict = {
        "n_studies": int(table[STUDY_ID].nunique()) if len(table) else 0,
        "cut_threshold": cut_threshold,
        "band_px": band_px,
        "tissue_threshold": tissue_threshold,
        "fov_mm": float(cfg.data.fov_mm),
        "crop_center": cfg.data.crop_center,
        "share_cut": {},
    }
    for slot, edges in CROP_EDGES.items():
        subset = table[table["slot"] == slot] if len(table) else table
        if subset.empty:
            continue
        names = [n for n in edges.values() if n is not None]
        shares = {n: round(float((subset[f"fill_{n}"] > cut_threshold).mean()), 4) for n in names}
        shares["any_edge"] = round(float((subset[[f"fill_{n}" for n in names]] > cut_threshold).any(axis=1).mean()), 4)
        summary["share_cut"][slot] = {"n": int(len(subset)), **shares}
    atomic_write_json(out_dir / "qc_edges_summary.json", summary)
    return summary
