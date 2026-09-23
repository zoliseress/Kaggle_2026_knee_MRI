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
from .utils import LOG, atomic_write_text


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_learning_curves(history: pd.DataFrame, out_path: Path) -> Path:
    plt = _matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o")
    axes[0].set_title("Training loss (class-normalised)")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)

    finite = history[np.isfinite(history["val_macro_roc_auc"])]
    axes[1].plot(finite["epoch"], finite["val_macro_roc_auc"], marker="o", color="tab:green")
    axes[1].axhline(0.5, ls="--", c="grey", lw=1)
    axes[1].set_title("Validation macro ROC-AUC (evaluable targets)")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["epoch"], history["encoder_lr"], marker="o", label="encoder")
    axes[2].plot(history["epoch"], history["head_lr"], marker="s", label="head")
    axes[2].set_title("Learning rate")
    axes[2].set_xlabel("epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

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


def build_html_report(
    run_dir: str | Path,
    qc_dir: str | Path | None = None,
    label_counts: pd.DataFrame | None = None,
    out_name: str = "report.html",
) -> Path:
    """Assemble one offline HTML file from the artefacts a run already wrote."""
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
        figures.append(("Learning curves", plot_learning_curves(history, run_dir / "learning_curves.png")))
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
        "<h2>Metric scope</h2><p>Local, evaluable-target macro ROC-AUC on the frozen binary reference. "
        "Targets without both classes are reported as <code>NA</code>, never as 0.5. This is not a verified "
        "official competition metric.</p>",
    ]
    if label_counts is not None and len(label_counts):
        sections.append("<h2>Label counts per target</h2>" + _table_html(label_counts, "{:.3f}"))
    if len(metrics):
        sections.append("<h2>Validation metrics per target</h2>" + _table_html(metrics))
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
<h1>Knee MRI - EfficientNet-B0 2.5D MIL</h1>
<p>Run directory: <code>{html.escape(str(run_dir))}</code></p>
{warning_block}
{''.join(sections)}
</body></html>
"""
    out_path = run_dir / out_name
    atomic_write_text(out_path, document)
    LOG.info("HTML report written: %s", out_path)
    return out_path
