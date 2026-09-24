"""Validation metrics on a fixed reference whose cells are 0, 1 or continuous in (0, 1).

Rules enforced here:
  * ranking metrics are computed once over the whole accumulated fold, never averaged
    per batch;
  * a single-class target returns NA - never the misleading 0.5;
  * AP is reported under its own name together with prevalence, and degenerate support
    is flagged instead of silently averaged in;
  * threshold metrics use one fixed diagnostic threshold and report the confusion
    counts, with undefined denominators returned as NA;
  * the macro values state `n_defined_targets / 12`. This is the *local* evaluable-target
    macro, which is not the same object as any official competition metric;
  * ROC-AUC, AP and the threshold counts use the hard (0/1) reference cells only. The
    soft ROC-AUC (`soft_roc_auc`) uses every valid cell, continuous ones included, and
    equals ROC-AUC exactly when the reference is binary;
  * a non-finite score (NaN, +-Inf) is an error, never imputed or dropped: NaN sorts as
    the largest score and would otherwise rank "perfectly".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import N_TARGETS, TARGETS


@dataclass
class ClassMetrics:
    target: str
    n_known: int  # every valid cell: hard positives + hard negatives + soft
    n_pos: int
    n_neg: int
    n_soft: int
    eff_positive: float  # sum of y over the valid cells
    eff_negative: float  # sum of 1 - y over the valid cells
    n_unresolved: int
    coverage: float
    roc_auc: float
    soft_auc: float
    spearman: float
    brier: float
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


def soft_roc_auc(y: np.ndarray, scores: np.ndarray, min_support: float = 1.0) -> float:
    """ROC-AUC generalised to reference values in [0, 1] (a weighted concordance index).

        softAUC = sum_{i,j} (y_i - y_j)+ H(s_i - s_j) / sum_{i,j} (y_i - y_j)+,   H(0) = 1/2

    A pair counts in proportion to how far apart its references are; pairs with equal
    references do not count. On a binary reference this is exactly the ROC-AUC; a score
    that orders the reference perfectly gets 1, a reversed one 0.

    Computed in O(n log n) through the probabilistic pair sum
        P = sum_{i != j} y_i (1 - y_j) H(s_i - s_j)
    which splits, per unordered pair with y_i >= y_j, into a symmetric part
    a = y_j (1 - y_i) and the concordance term (y_i - y_j) H(s_i - s_j). Hence
    softAUC = (P - A) / D with A = sum a and D = sum |y_i - y_j| over unordered pairs.

    Returns NaN when the positive or the negative mass (sum y, sum 1 - y) is below
    `min_support`, or every reference value is the same.
    """
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y and scores must be 1-D arrays of the same length")
    if y.size and (not np.isfinite(y).all() or y.min() < 0.0 or y.max() > 1.0):
        raise ValueError("soft_roc_auc reference values must be finite and in [0, 1]")
    if not np.isfinite(s).all():
        bad = np.flatnonzero(~np.isfinite(s))
        raise ValueError(f"soft_roc_auc scores must be finite; {bad.size} non-finite at positions {bad[:5].tolist()}")
    pos_mass, neg_mass = float(y.sum()), float((1.0 - y).sum())
    if pos_mass < min_support or neg_mass < min_support:
        return float("nan")
    ordered = np.sort(y)
    ranks = np.arange(y.size, dtype=np.float64)
    spread = float((ordered * (2.0 * ranks - y.size + 1.0)).sum())  # D = sum_{i<j} |y_i - y_j|
    if spread <= 1e-12:
        return float("nan")
    self_pairs = float((y * (1.0 - y)).sum())
    all_weight = pos_mass * neg_mass - self_pairs  # sum_{i != j} y_i (1 - y_j) = 2A + D
    symmetric = 0.5 * (all_weight - spread)  # A
    # P: group tied scores; a positive beats every negative strictly below it and half-beats ties.
    _, group = np.unique(s, return_inverse=True)
    pos_by_score = np.bincount(group, weights=y)
    neg_by_score = np.bincount(group, weights=1.0 - y)
    neg_below = np.cumsum(neg_by_score) - neg_by_score
    all_pairs = float((pos_by_score * (neg_below + 0.5 * neg_by_score)).sum())
    # Each cell is tied with itself and contributed y_i (1 - y_i) / 2 above; drop the self-pairs.
    probabilistic = all_pairs - 0.5 * self_pairs
    return float(np.clip((probabilistic - symmetric) / spread, 0.0, 1.0))


def _spearman(y: np.ndarray, scores: np.ndarray) -> float:
    """Rank correlation of reference and score; NaN when either side is constant."""
    if y.size < 2:
        return float("nan")
    ranked_y = pd.Series(y).rank().to_numpy()
    ranked_s = pd.Series(scores).rank().to_numpy()
    if np.ptp(ranked_y) == 0 or np.ptp(ranked_s) == 0:
        return float("nan")
    return float(np.corrcoef(ranked_y, ranked_s)[0, 1])


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
    y_all = reference[valid].astype(float)
    s_all = scores[valid].astype(float)
    hard = (y_all == 0.0) | (y_all == 1.0)
    # The binary metrics below see the hard cells only; a soft value is never thresholded.
    y_true = y_all[hard].astype(int)
    y_score = s_all[hard]
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    n_soft = int((~hard).sum())
    n_unresolved = int((~valid).sum())
    coverage = float(valid.mean()) if valid.size else 0.0

    soft_auc = soft_roc_auc(y_all, s_all, min_support=min_support)
    spearman = _spearman(y_all, s_all)
    brier = float(np.mean((s_all - y_all) ** 2)) if y_all.size else float("nan")

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
        n_known=n_pos + n_neg + n_soft,
        n_pos=n_pos,
        n_neg=n_neg,
        n_soft=n_soft,
        eff_positive=float(y_all.sum()),
        eff_negative=float((1.0 - y_all).sum()),
        n_unresolved=n_unresolved,
        coverage=round(coverage, 4),
        roc_auc=auc,
        soft_auc=soft_auc,
        spearman=spearman,
        brier=brier,
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
    finite = np.isfinite(scores)
    if not finite.all():
        rows, cols = np.nonzero(~finite)
        examples = ", ".join(f"row {r}/{TARGETS[c]}={scores[r, c]}" for r, c in zip(rows[:5], cols[:5]))
        raise ValueError(f"scores contain {rows.size} non-finite values (e.g. {examples}); the model output is broken")
    if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
        raise ValueError("scores must be sigmoid probabilities in [0, 1], not logits")

    rows = [
        class_metrics(scores[:, c], reference[:, c], valid[:, c], TARGETS[c], threshold, min_support).as_row()
        for c in range(N_TARGETS)
    ]
    table = pd.DataFrame(rows)

    defined_auc = table["roc_auc"].dropna()
    defined_ap = table.loc[~table["ap_degenerate"], "average_precision"].dropna()
    defined_soft = table["soft_auc"].dropna()
    summary = {
        "macro_roc_auc": float(defined_auc.mean()) if len(defined_auc) else float("nan"),
        "n_defined_targets": int(len(defined_auc)),
        "macro_soft_auc": float(defined_soft.mean()) if len(defined_soft) else float("nan"),
        "n_soft_auc_targets": int(len(defined_soft)),
        "macro_spearman": float(table["spearman"].dropna().mean()) if table["spearman"].notna().any() else float("nan"),
        "macro_brier": float(table["brier"].dropna().mean()) if table["brier"].notna().any() else float("nan"),
        "n_soft_cells": int(table["n_soft"].sum()),
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
            "local evaluable-target macros: ROC-AUC over the hard (0/1) reference cells, soft ROC-AUC "
            "over every valid cell including continuous ones; not a verified official competition metric"
        ),
    }
    return table, summary


def soft_target_warning(kinds: np.ndarray) -> str | None:
    """State how continuous reference cells enter the metrics."""
    from .labels import KIND_SOFT

    if np.any(kinds == KIND_SOFT):
        return (
            "Continuous (soft) targets are present in the validation labels. They enter the soft "
            "ROC-AUC, Spearman and Brier values; ROC-AUC, AP and the threshold counts use the hard "
            "0/1 cells only, because thresholding a soft value would invent a reference."
        )
    return None


SELECTION_METRICS = ("macro_soft_auc", "macro_roc_auc")


def selection_metric(summary: dict, name: str = "macro_soft_auc") -> float:
    """The single number early stopping and checkpoint selection may look at."""
    if name not in SELECTION_METRICS:
        raise ValueError(f"selection metric must be one of {SELECTION_METRICS}, got {name!r}")
    value = summary.get(name, float("nan"))
    return float(value) if value is not None else float("nan")
