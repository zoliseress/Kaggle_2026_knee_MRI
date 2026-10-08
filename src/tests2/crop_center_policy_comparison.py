"""Compare the crop-centre policies (foreground, foreground_extent, geometric) on real data.

For a seeded random sample of studies (by default the same 500 studies as the ``qc-edges``
audit) every selected series is decoded once and run through the production preprocessing
path - centre policy -> 150 mm physical crop -> robust scaling -> square resample - with
each policy. The crop-edge metric of ``qc-edges`` (``knee_mri.qc.edge_fill_fractions``)
then measures how often the crop cuts anatomy at each evaluated edge, and the NaN padding
fraction measures how much of the crop falls outside the acquired matrix.

Run from ``src`` (torch and pydicom are needed)::

    python tests2\\crop_center_policy_comparison.py --workers 12

Outputs under ``tests2/artifacts/crop_center_comparison/``: ``per_series.csv`` (one row per
series and policy), ``summary.csv`` (cut shares per slot and policy), ``paired.csv`` (how
often only one of two policies cuts) and ``divergent_<slot>.png`` galleries of the series
whose policies disagree most.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knee_mri.config import Config, load_config
from knee_mri.constants import STUDY_ID
from knee_mri.dicom_io import build_volume_candidates, decode_volume
from knee_mri.manifest import load_selection
from knee_mri.preprocess import estimate_center, physical_crop, resample_square, robust_scale
from knee_mri.qc import CROP_EDGES, edge_fill_fractions

POLICIES = ("foreground", "foreground_extent", "geometric")
COLORS = {"foreground": "orange", "foreground_extent": "lime", "geometric": "cyan"}
CUT_THRESHOLD = 0.3  # the qc-edges default


def _plane_of(slot: str) -> str:
    return "sagittal" if slot.startswith("sagittal") else "coronal" if slot.startswith("coronal") else "axial"


def _load(row: dict, cfg: Config):
    candidates, _, flags = build_volume_candidates(
        Path(row["path"]),
        obliquity_tolerance_deg=float(cfg.manifest.obliquity_tolerance_deg),
        max_gap_ratio=float(cfg.manifest.max_gap_ratio),
        min_slices=int(cfg.manifest.min_slices),
    )
    volume = next((c for c in candidates if c.volume_key == row["volume_key"]), None)
    if volume is None:
        raise ValueError(f"no {row['volume_key']} candidate ({', '.join(flags)})")
    stack = decode_volume(volume, monochrome1_policy=cfg.data.monochrome1_policy)
    row_spacing, col_spacing = volume.inplane.apply_spacing(volume.row_spacing, volume.col_spacing)
    return stack, float(row_spacing), float(col_spacing)


def evaluate_series(args: tuple[dict, dict]) -> list[dict]:
    """Importable worker: one selected series through every policy -> one row per policy."""
    row, cfg_dict = args
    cfg = Config(cfg_dict)
    base = {STUDY_ID: row[STUDY_ID], "slot": row["slot"], "SeriesInstanceUID": row["SeriesInstanceUID"]}
    try:
        stack, row_spacing, col_spacing = _load(row, cfg)
    except Exception as exc:  # unreadable series: reported, not fatal
        return [{**base, "policy": None, "error": str(exc)[:200]}]
    edges = {edge: name for edge, name in CROP_EDGES[_plane_of(row["slot"])].items() if name is not None}
    centers = {policy: estimate_center(stack, policy) for policy in POLICIES}
    out = []
    for policy, center in centers.items():
        cropped, info = physical_crop(stack, row_spacing, col_spacing, float(cfg.data.fov_mm), center)
        scaled, _ = robust_scale(cropped, tuple(cfg.data.intensity_percentiles))
        image = resample_square(scaled, int(cfg.data.image_size))
        fills = edge_fill_fractions(image)
        offset = np.subtract(center, centers["foreground"]) * (row_spacing, col_spacing)
        result = {
            **base,
            "policy": policy,
            "crop_pad_fraction": info["crop_pad_fraction"],
            "center_row": round(center[0], 2),
            "center_col": round(center[1], 2),
            "offset_from_foreground_mm": round(float(np.hypot(*offset)), 2),
        }
        for edge, name in edges.items():
            result[f"fill_{name}"] = round(fills[edge], 4)
        result["any_cut"] = any(fills[edge] > CUT_THRESHOLD for edge in edges)
        out.append(result)
    return out


def summarize(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut shares per (slot, policy), and paired 'only A cuts / only B cuts' counts."""
    rows = []
    for (slot, policy), group in table.groupby(["slot", "policy"], sort=True):
        names = [c for c in group.columns if c.startswith("fill_") and group[c].notna().all()]
        rows.append({
            "slot": slot,
            "policy": policy,
            "n": len(group),
            "share_any_cut": round(float(group["any_cut"].mean()), 4),
            **{f"share_cut_{c[5:]}": round(float((group[c] > CUT_THRESHOLD).mean()), 4) for c in names},
            "mean_pad_fraction": round(float(group["crop_pad_fraction"].mean()), 4),
            "share_pad_over_5pct": round(float((group["crop_pad_fraction"] > 0.05).mean()), 4),
            "median_offset_from_foreground_mm": round(float(group["offset_from_foreground_mm"].median()), 1),
        })
    cut = table.pivot_table(index=[STUDY_ID, "slot"], columns="policy", values="any_cut", aggfunc="first").dropna()
    paired = []
    for slot, group in cut.groupby(level="slot"):
        for a, b in (("foreground", "geometric"), ("foreground", "foreground_extent"), ("foreground_extent", "geometric")):
            ga, gb = group[a].astype(bool), group[b].astype(bool)
            paired.append({"slot": slot, "a": a, "b": b, "n": len(group), "both_cut": int((ga & gb).sum()),
                           f"only_a_cuts": int((ga & ~gb).sum()), f"only_b_cuts": int((~ga & gb).sum())})
    return pd.DataFrame(rows), pd.DataFrame(paired)


def divergent_gallery(table: pd.DataFrame, selection: pd.DataFrame, cfg: Config, slot: str, path: Path, k: int = 8) -> None:
    """Middle slices of the k series of ``slot`` whose geometric and foreground crops differ most."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    sub = table[(table["slot"] == slot) & (table["policy"] == "geometric")]
    top = sub.nlargest(k, "offset_from_foreground_mm")
    if top.empty:
        return
    cols = 4
    nrows = int(np.ceil(len(top) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(5 * cols, 5.3 * nrows))
    for axis in np.atleast_1d(axes).ravel():
        axis.axis("off")
    for axis, (_, r) in zip(np.atleast_1d(axes).ravel(), top.iterrows()):
        srow = selection[(selection[STUDY_ID] == r[STUDY_ID]) & (selection["slot"] == slot)].iloc[0].to_dict()
        stack, row_spacing, col_spacing = _load(srow, cfg)
        middle = np.nan_to_num(stack[len(stack) // 2])
        low, high = np.percentile(middle, (1, 99))
        axis.imshow(middle, cmap="gray", vmin=low, vmax=high if high > low else None)
        cuts = table[(table[STUDY_ID] == r[STUDY_ID]) & (table["slot"] == slot)].set_index("policy")["any_cut"]
        for policy in POLICIES:
            center = estimate_center(stack, policy)
            _, info = physical_crop(stack, row_spacing, col_spacing, float(cfg.data.fov_mm), center)
            axis.add_patch(Rectangle((info["crop_col0"], info["crop_row0"]), info["crop_cols_px"], info["crop_rows_px"],
                                     fill=False, edgecolor=COLORS[policy], linewidth=1.4))
            axis.plot(center[1], center[0], "+", color=COLORS[policy], ms=12, mew=2)
        axis.set_title(f"..{r[STUDY_ID][-8:]}  offset {r['offset_from_foreground_mm']:.0f} mm\n"
                       + " ".join(f"{p[:6]}:{'CUT' if cuts.get(p) else 'ok'}" for p in POLICIES), fontsize=9)
    fig.suptitle(f"{slot}: largest geometric-vs-foreground centre offsets "
                 "(orange = foreground, lime = foreground_extent, cyan = geometric)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Config YAML (default: src/config.yaml).")
    parser.add_argument("--n-studies", type=int, default=500, help="Seeded random studies (same pick as qc-edges).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("artifacts") / "crop_center_comparison")
    args = parser.parse_args()

    cfg = load_config(args.config)
    selection = load_selection(cfg)
    selection = selection[selection["selected"].fillna(False).astype(bool)].copy()
    selection[STUDY_ID] = selection[STUDY_ID].astype(str)
    pool = sorted(set(selection[STUDY_ID]))
    rng = np.random.default_rng(int(cfg.seed))  # identical draw to `cli qc-edges`
    studies = set(rng.choice(pool, size=min(args.n_studies, len(pool)), replace=False))
    jobs = selection[selection[STUDY_ID].isin(studies)].to_dict("records")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool_exec:
        futures = [pool_exec.submit(evaluate_series, (job, dict(cfg))) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if done % 250 == 0:
                print(f"{done}/{len(futures)} series", flush=True)
    table = pd.DataFrame(rows)
    errors = table[table["policy"].isna()] if "error" in table else table.iloc[0:0]
    table = table[table["policy"].notna()].drop(columns=["error"], errors="ignore")
    table.to_csv(args.out_dir / "per_series.csv", index=False)
    summary, paired = summarize(table)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    paired.to_csv(args.out_dir / "paired.csv", index=False)
    for slot in sorted(table["slot"].unique()):
        divergent_gallery(table, selection, cfg, slot, args.out_dir / f"divergent_{slot}.png")

    pd.set_option("display.width", 250)
    print(f"{len(studies)} studies, {len(jobs)} series, {len(errors)} unreadable")
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
