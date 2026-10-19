"""Blend a category-group specialist into a base ensemble, on its own targets only.

    p_final[t] = (1 - alpha) * p_base[t] + alpha * p_specialist[t]    for t in the group
    p_final[t] = p_base[t]                                            for every other target

A specialist (train.loss_targets) keeps all 12 outputs, but only its loss targets were
supervised; the others drifted with the shared backbone and are never used. Rows are matched
by StudyInstanceUID and columns by target name, never by position.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .config import target_subset
from .constants import STUDY_ID, TARGETS


def long_to_wide(predictions: pd.DataFrame) -> pd.DataFrame:
    """validation_predictions.csv (one row per study x target) -> one row per study, TARGETS columns."""
    if predictions.duplicated([STUDY_ID, "target"]).any():
        raise ValueError("predictions hold a (study, target) pair more than once")
    wide = predictions.pivot(index=STUDY_ID, columns="target", values="score")
    missing = [t for t in TARGETS if t not in wide.columns]
    if missing:
        raise ValueError(f"predictions lack targets {missing}")
    wide = wide[list(TARGETS)].reset_index()
    wide.columns.name = None
    return wide


def blend_specialist(
    base: pd.DataFrame, specialist: pd.DataFrame, targets: Sequence[str], alpha: float
) -> pd.DataFrame:
    """`base` with the `targets` columns mixed toward `specialist` (both wide: STUDY_ID + TARGETS).

    Both frames must hold exactly the same studies; the result keeps `base`'s row order.
    """
    names = target_subset(list(targets), "specialist targets")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    for label, frame in (("base", base), ("specialist", specialist)):
        if STUDY_ID not in frame.columns:
            raise ValueError(f"{label} predictions have no {STUDY_ID} column")
        if frame[STUDY_ID].duplicated().any():
            raise ValueError(f"{label} predictions hold a study more than once")
        absent = [t for t in names if t not in frame.columns]
        if absent:
            raise ValueError(f"{label} predictions lack the target columns {absent}")
    base_ids, spec_ids = set(base[STUDY_ID].astype(str)), set(specialist[STUDY_ID].astype(str))
    if base_ids != spec_ids:
        raise ValueError(
            f"base and specialist cover different studies ({len(base_ids - spec_ids)} only in base, "
            f"{len(spec_ids - base_ids)} only in the specialist)"
        )
    aligned = specialist.assign(**{STUDY_ID: specialist[STUDY_ID].astype(str)}).set_index(STUDY_ID)
    aligned = aligned.loc[base[STUDY_ID].astype(str), names]
    out = base.copy()
    for t in names:
        mixed = (1.0 - alpha) * base[t].to_numpy(dtype=np.float64) + alpha * aligned[t].to_numpy(dtype=np.float64)
        if not np.isfinite(mixed).all():
            raise ValueError(f"non-finite blended score for {t}")
        out[t] = mixed
    return out
