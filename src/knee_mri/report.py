"""Offline plots and a self-contained HTML run report.

The report links the QC gallery, the label counts, the fixed-reference metrics and a
short list of high-confidence disagreements for review. Disagreement review is a
development aid: it is kept out of any locked final evaluation, and the report says so.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .constants import STUDY_ID, TARGETS
from .metrics import subset_metrics
from .utils import LOG, atomic_write_dataframe, atomic_write_text


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _best_epoch(history: pd.DataFrame, summary: dict) -> int | None:
    if summary.get("best_epoch") is not None and int(summary["best_epoch"]) >= 0:
        return int(summary["best_epoch"])
    column = "val_selection_score" if "val_selection_score" in history else "val_macro_roc_auc"
    if column not in history:
        return None
    values = pd.to_numeric(history[column], errors="coerce")
    return int(history.loc[values.idxmax(), "epoch"]) if values.notna().any() else None


def _plot_series(ax, history: pd.DataFrame, column: str, label: str, **style: Any) -> bool:
    """One history column against epoch; absent (older runs) or all-NaN columns are skipped."""
    if column not in history:
        return False
    values = pd.to_numeric(history[column], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float))
    if not finite.any():
        return False
    ax.plot(history["epoch"][finite], values[finite], label=label, **style)
    return True


def plot_learning_curves(history: pd.DataFrame, out_path: Path, summary: dict | None = None) -> Path:
    plt = _matplotlib()
    summary = summary or {}
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ax_loss, ax_rank, ax_ap, ax_calib, ax_lr, ax_speed = axes.ravel()

    _plot_series(ax_loss, history, "train_loss", "train loss", marker="o")
    _plot_series(ax_loss, history, "val_loss", "val loss (training labels)", marker="s")
    _plot_series(ax_loss, history, "val_ref_bce", "val BCE (reference)", marker="^")
    ax_loss.set_title("Loss (class-normalised BCE)")

    # Older runs have no val_selection_score column; their selection score was macro ROC-AUC.
    selection = summary.get("selection_metric", "selection score")
    _plot_series(ax_rank, history, "val_selection_score", f"selection ({selection})", marker="o", lw=2.5, color="tab:green")
    _plot_series(ax_rank, history, "val_selection_score_raw", "selection, raw (no EMA)", ls="--", color="tab:green", alpha=0.6)
    _plot_series(ax_rank, history, "val_macro_roc_auc", "ROC-AUC (0/1 cells)", marker="s", color="tab:blue")
    _plot_series(ax_rank, history, "val_macro_soft_auc", "soft ROC-AUC", marker="^", color="tab:purple")
    ax_rank.set_title("Validation macro ranking metrics")

    _plot_series(ax_ap, history, "val_macro_ap", "AP", marker="o", color="tab:orange")
    _plot_series(ax_ap, history, "val_macro_f1", "F1 @ threshold", marker="s", color="tab:red")
    ax_ap.set_title("Validation macro AP / F1")

    _plot_series(ax_calib, history, "val_macro_brier", "Brier (lower is better)", marker="o", color="tab:brown")
    _plot_series(ax_calib, history, "val_macro_spearman", "Spearman", marker="s", color="tab:cyan")
    ax_calib.set_title("Validation calibration / correlation")

    # Linear: the cosine schedule ends at exactly 0, which a log axis cannot show.
    _plot_series(ax_lr, history, "encoder_lr", "encoder", marker="o")
    _plot_series(ax_lr, history, "head_lr", "head", marker="s")
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_lr.set_title("Learning rate (end of epoch)")

    _plot_series(ax_speed, history, "studies_per_second", "train studies/s", marker="o", color="tab:gray")
    twin = ax_speed.twinx()
    if _plot_series(twin, history, "peak_gpu_gb", "peak GPU GB", marker="s", color="tab:olive"):
        twin.set_ylabel("GB")
        twin.set_ylim(0, twin.get_ylim()[1] * 1.1)
        twin.legend(loc="lower right", fontsize=8)
    ax_speed.set_ylim(0, ax_speed.get_ylim()[1] * 1.1)  # from 0: run-to-run jitter must not look like a trend
    ax_speed.set_title("Throughput / memory")

    from matplotlib.ticker import MaxNLocator

    best = _best_epoch(history, summary)
    for ax in axes.ravel():
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if best is not None:
            ax.axvline(best, ls=":", c="k", lw=1, label=f"best epoch {best}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

    fig.suptitle(summary.get("run_dir", out_path.parent.name).replace("\\", "/").rsplit("/", 1)[-1])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_per_class_metrics(metrics: pd.DataFrame, out_path: Path) -> Path:
    plt = _matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = list(metrics["target"])
    y = np.arange(len(order))

    auc = metrics["roc_auc"].to_numpy(dtype=float)
    axes[0].barh(y, np.nan_to_num(auc, nan=0.0), color=["tab:blue" if np.isfinite(v) else "lightgrey" for v in auc])
    axes[0].axvline(0.5, ls="--", c="grey", lw=1)
    axes[0].set_yticks(y, order)
    axes[0].set_xlim(0, 1)
    axes[0].set_title("ROC-AUC (grey = undefined / NA)")
    for i, value in enumerate(auc):
        axes[0].text(0.02, i, "NA" if not np.isfinite(value) else f"{value:.3f}", va="center", fontsize=8)

    ap = metrics["average_precision"].to_numpy(dtype=float)
    prevalence = metrics["prevalence"].to_numpy(dtype=float)
    axes[1].barh(y, np.nan_to_num(ap, nan=0.0), color="tab:orange", label="AP")
    axes[1].plot(np.nan_to_num(prevalence, nan=0.0), y, "k.", label="prevalence")
    axes[1].set_yticks(y, order)
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Average precision vs prevalence")
    axes[1].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def high_confidence_disagreements(predictions: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Confident predictions that contradict the fixed reference - review material only."""
    valid = predictions[predictions["reference_valid"].astype(bool)].copy()
    if not len(valid):
        return valid
    valid["margin"] = np.abs(valid["score"] - valid["reference"])
    return valid.sort_values("margin", ascending=False).head(n)


def _table_html(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    if df is None or not len(df):
        return "<p><em>No data.</em></p>"
    return df.to_html(index=False, float_format=lambda v: float_format.format(v), na_rep="NA", border=0)


def write_subset_metrics(
    run_dir: Path,
    predictions: pd.DataFrame,
    subsets: pd.DataFrame | None,
    threshold: float = 0.5,
    min_support: int = 1,
) -> str:
    """`subset_summary.csv` + `subset_metrics_per_class.csv` in the run dir; returns the HTML section.

    Without a subset table there is only the full slice, which `metrics_per_class.csv` already
    covers, so nothing is written.
    """
    if subsets is None or not len(predictions):
        return ""
    per_class, summary = subset_metrics(predictions, subsets, threshold=threshold, min_support=min_support)
    atomic_write_dataframe(per_class, run_dir / "subset_metrics_per_class.csv")
    atomic_write_dataframe(summary, run_dir / "subset_summary.csv")

    names = list(summary["subset"])
    wide = pd.DataFrame({"target": list(TARGETS)})
    for name in names:
        part = per_class[per_class["subset"] == name].set_index("target").loc[list(TARGETS)]
        wide[f"{name} AUC"] = part["soft_auc"].to_numpy(float)
        wide[f"{name} pos/neg"] = [f"{p}/{n}" for p, n in zip(part["n_pos"], part["n_neg"])]
    macro = {"target": "macro (soft AUC)"}
    for name, row in summary.set_index("subset").iterrows():
        macro[f"{name} AUC"] = row["macro_soft_auc"]
        macro[f"{name} pos/neg"] = f"{row['n_evaluable_studies']}/{row['n_studies']} studies"
    wide = pd.concat([pd.DataFrame([macro]), wide], ignore_index=True)
    return (
        "<h2>Validation metrics by reference subset</h2><p>Same best-checkpoint predictions and per-cell "
        f"mask in every column. <code>{html.escape(names[0])}</code> is computed directly from all "
        "predictions, not averaged from the subsets. AUC = soft ROC-AUC (equals ROC-AUC on a binary "
        "reference); pos/neg = hard reference cells; the macro row shows evaluable/all studies.</p>"
        + _table_html(wide)
    )


def _backbone(run_dir: Path) -> str:
    """The run's encoder from its saved config.yaml; a neutral label when unreadable."""
    try:
        import yaml

        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        return str(cfg["model"]["backbone"])
    except Exception:
        return "encoder"


def build_html_report(
    run_dir: str | Path,
    qc_dir: str | Path | None = None,
    label_counts: pd.DataFrame | None = None,
    out_name: str = "report.html",
    subsets: pd.DataFrame | None = None,
    threshold: float = 0.5,
    min_support: int = 1,
) -> Path:
    """Assemble one offline HTML file from the artefacts a run already wrote.

    `subsets` (STUDY_ID, subset) adds the per-subset validation breakdown (`write_subset_metrics`).
    """
    run_dir = Path(run_dir).resolve()
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8")) if (run_dir / "run_summary.json").exists() else {}
    history = pd.read_csv(run_dir / "history.csv") if (run_dir / "history.csv").exists() else pd.DataFrame()
    metrics = pd.read_csv(run_dir / "metrics_per_class.csv") if (run_dir / "metrics_per_class.csv").exists() else pd.DataFrame()
    predictions = (
        pd.read_csv(run_dir / "validation_predictions.csv", dtype={STUDY_ID: "string"})
        if (run_dir / "validation_predictions.csv").exists()
        else pd.DataFrame()
    )

    figures: list[tuple[str, Path]] = []
    if len(history):
        figures.append(("Learning curves", plot_learning_curves(history, run_dir / "learning_curves.png", summary)))
    if len(metrics):
        figures.append(("Per-class metrics", plot_per_class_metrics(metrics, run_dir / "per_class_metrics.png")))

    qc_images: list[Path] = []
    if qc_dir and Path(qc_dir).exists():
        # Absolute, so the file:// links in the HTML work from any working directory.
        qc_images = sorted(Path(qc_dir).resolve().glob("qc_*.png"))[:12]

    warning_block = ""
    for key in ("warning", "reference_note"):
        if summary.get(key):
            warning_block += f"<div class='warn'>{html.escape(str(summary[key]))}</div>"

    def relative(path: Path) -> str:
        try:
            return str(path.relative_to(run_dir)).replace("\\", "/")
        except ValueError:
            return path.as_uri()

    sections = [
        f"<h2>Run summary</h2><pre>{html.escape(json.dumps(summary, indent=2, default=str))}</pre>",
        "<h2>Metric scope</h2><p>Local, evaluable-target macros on the frozen reference. ROC-AUC uses the "
        "0/1 cells; soft ROC-AUC uses every valid cell, continuous ones included, and equals ROC-AUC on a "
        "binary reference. The run selects on <code>"
        + html.escape(str(summary.get("selection_metric", "macro_roc_auc")))
        + "</code>. Targets without both classes are reported as <code>NA</code>, never as 0.5. This is not a "
        "verified official competition metric.</p>",
    ]
    if label_counts is not None and len(label_counts):
        sections.append("<h2>Label counts per target</h2>" + _table_html(label_counts, "{:.3f}"))
    if len(metrics):
        sections.append("<h2>Validation metrics per target</h2>" + _table_html(metrics))
    subset_section = write_subset_metrics(run_dir, predictions, subsets, threshold, min_support)
    if subset_section:
        sections.append(subset_section)
    for title, path in figures:
        sections.append(f"<h2>{title}</h2><img src='{relative(path)}' style='max-width:100%'>")
    if len(history):
        sections.append("<h2>History</h2>" + _table_html(history))
    if qc_images:
        gallery = "".join(
            f"<figure><img src='{Path(p).as_uri()}' style='max-width:100%'><figcaption>{html.escape(p.name)}</figcaption></figure>"
            for p in qc_images
        )
        sections.append(f"<h2>QC gallery</h2>{gallery}")
    if len(predictions):
        disagreements = high_confidence_disagreements(predictions)
        sections.append(
            "<h2>High-confidence disagreements (review aid)</h2>"
            "<p class='warn'>Development aid only. Reviewing these cases changes the reference, so it must "
            "stay outside any locked final evaluation, and a repaired label table is not evidence that the "
            "model producing it improved.</p>" + _table_html(disagreements)
        )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Knee MRI run report - {html.escape(run_dir.name)}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; line-height: 1.45; color: #1a1a1a; }}
 table {{ border-collapse: collapse; font-size: 0.85rem; }}
 th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 pre {{ background: #f6f8fa; padding: 1rem; overflow-x: auto; font-size: 0.8rem; }}
 .warn {{ background: #fff4e5; border-left: 4px solid #e69100; padding: 0.6rem 1rem; margin: 1rem 0; }}
 figure {{ margin: 1rem 0; }}
</style></head>
<body>
<h1>Knee MRI - {html.escape(_backbone(run_dir))} 2.5D MIL</h1>
<p>Run directory: <code>{html.escape(str(run_dir))}</code></p>
{warning_block}
{''.join(sections)}
</body></html>
"""
    out_path = run_dir / out_name
    atomic_write_text(out_path, document)
    LOG.info("HTML report written: %s", out_path)
    return out_path


def build_run_report(cfg: Any, run_dir: str | Path, qc_dir: str | Path | None = None) -> Path:
    """`build_html_report` with the project's label counts and QC gallery: the `report`
    command and the end of every training run share it."""
    counts_path = Path(cfg.paths.work_dir) / "schema" / "labels_counts.csv"
    counts = pd.read_csv(counts_path) if counts_path.exists() else None
    qc_dir = qc_dir or Path(cfg.paths.work_dir) / "qc"
    subsets_path = cfg.paths.get("reference_subsets_csv")
    subsets = pd.read_csv(subsets_path, dtype={STUDY_ID: "string"}, encoding="utf-8-sig") if subsets_path else None
    return build_html_report(
        run_dir,
        qc_dir=qc_dir,
        label_counts=counts,
        subsets=subsets,
        threshold=float(cfg.eval.threshold),
        min_support=int(cfg.eval.min_class_support),
    )
