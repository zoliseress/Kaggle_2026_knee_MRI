"""Validation metrics on a fixed binary reference.

Rules enforced here:
  * ranking metrics are computed once over the whole accumulated fold, never averaged
    per batch;
  * a single-class target returns NA - never the misleading 0.5;
  * AP is reported under its own name together with prevalence, and degenerate support
    is flagged instead of silently averaged in;
  * threshold metrics use one fixed diagnostic threshold and report the confusion
    counts, with undefined denominators returned as NA;
  * the macro values state `n_defined_targets / 12`. This is the *local* evaluable-target
    macro, which is not the same object as any official competition metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import N_TARGETS, TARGETS


@dataclass
class ClassMetrics:
    target: str
    n_known: int
    n_pos: int
    n_neg: int
    n_unresolved: int
    coverage: float
    roc_auc: float
    average_precision: float
    prevalence: float
    ap_degenerate: bool
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    specificity: float
    f1: float

    def as_row(self) -> dict:
        return self.__dict__.copy()


def _safe_divide(numerator: float, denominator: float) -> float:
    """Undefined denominators return NaN (reported as NA), not an invented value."""
    return float(numerator) / float(denominator) if denominator > 0 else float("nan")


def class_metrics(
    scores: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    target: str,
    threshold: float = 0.5,
    min_support: int = 1,
) -> ClassMetrics:
    from sklearn.metrics import average_precision_score, roc_auc_score

    valid = valid.astype(bool)
    y_true = reference[valid].astype(int)
    y_score = scores[valid].astype(float)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    n_unresolved = int((~valid).sum())
    coverage = float(valid.mean()) if valid.size else 0.0

    auc = float("nan")
    if n_pos >= min_support and n_neg >= min_support:
        auc = float(roc_auc_score(y_true, y_score))

    ap, prevalence, degenerate = float("nan"), float("nan"), True
    if n_pos > 0 and n_neg > 0:
        ap = float(average_precision_score(y_true, y_score))
        prevalence = n_pos / (n_pos + n_neg)
        degenerate = n_pos < min_support

    predicted = (y_score >= threshold).astype(int)
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    tn = int(((predicted == 0) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)

    return ClassMetrics(
        target=target,
        n_known=n_pos + n_neg,
        n_pos=n_pos,
        n_neg=n_neg,
        n_unresolved=n_unresolved,
        coverage=round(coverage, 4),
        roc_auc=auc,
        average_precision=ap,
        prevalence=prevalence,
        ap_degenerate=bool(degenerate),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
    )


def evaluate_predictions(
    scores: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    threshold: float = 0.5,
    min_support: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Per-target metrics plus the local evaluable-target macro summary."""
    if scores.shape != reference.shape or scores.shape != valid.shape:
        raise ValueError("scores, reference and valid must share the same [N, 12] shape")
    if scores.shape[1] != N_TARGETS:
        raise ValueError(f"expected {N_TARGETS} target columns, got {scores.shape[1]}")
    if scores.size and (np.nanmin(scores) < 0.0 or np.nanmax(scores) > 1.0):
        raise ValueError("scores must be sigmoid probabilities in [0, 1], not logits")

    rows = [
        class_metrics(scores[:, c], reference[:, c], valid[:, c], TARGETS[c], threshold, min_support).as_row()
        for c in range(N_TARGETS)
    ]
    table = pd.DataFrame(rows)

    defined_auc = table["roc_auc"].dropna()
    defined_ap = table.loc[~table["ap_degenerate"], "average_precision"].dropna()
    summary = {
        "macro_roc_auc": float(defined_auc.mean()) if len(defined_auc) else float("nan"),
        "n_defined_targets": int(len(defined_auc)),
        "n_targets": N_TARGETS,
        "defined_fraction": f"{len(defined_auc)}/{N_TARGETS}",
        "macro_average_precision": float(defined_ap.mean()) if len(defined_ap) else float("nan"),
        "n_ap_targets": int(len(defined_ap)),
        "macro_f1_at_threshold": float(table["f1"].dropna().mean()) if table["f1"].notna().any() else float("nan"),
        "threshold": float(threshold),
        "n_studies": int(scores.shape[0]),
        "mean_coverage": float(table["coverage"].mean()),
        "undefined_targets": list(table.loc[table["roc_auc"].isna(), "target"]),
        "metric_scope": (
            "local evaluable-target macro ROC-AUC over the fixed binary reference; "
            "not a verified official competition metric"
        ),
    }
    return table, summary


def soft_target_warning(kinds: np.ndarray) -> str | None:
    """Binary ranking metrics need a binary reference; soft targets cannot supply one."""
    from .labels import KIND_SOFT

    if np.any(kinds == KIND_SOFT):
        return (
            "Soft targets are present in the validation labels. They are excluded from the binary "
            "reference: ROC-AUC/AP need fixed binary entries, and thresholding a soft value would "
            "invent a reference. Report the compatible weighted loss for those cells instead."
        )
    return None


def selection_metric(summary: dict) -> float:
    """The single number early stopping and checkpoint selection may look at."""
    value = summary.get("macro_roc_auc", float("nan"))
    return float(value) if value is not None else float("nan")
