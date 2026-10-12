"""Regression test of the laterality rule against a radiologist's review (real data).

`fixtures/laterality_yc_review.csv` holds YC's verdict on the 36 training studies whose
laterality information was inconsistent: groups A (tag differs between series) and B (tag
uniform, opposite to the image position) show one knee each, group C both knees. For the
29 single-knee studies `derive_laterality` must give YC's side; the old tag-first rule got
all 29 wrong.

Needs the manifest, the series selection and the DICOM files of these studies; skipped
when they are not available.

    cd src && python -m pytest tests/test_laterality_yc_review.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knee_mri.config import load_config  # noqa: E402
from knee_mri.constants import STUDY_ID  # noqa: E402
from knee_mri.laterality import derive_laterality  # noqa: E402
from knee_mri.manifest import load_manifest, load_selection  # noqa: E402

REVIEW = Path(__file__).with_name("fixtures") / "laterality_yc_review.csv"


def test_laterality_matches_radiologist_review():
    review = pd.read_csv(REVIEW, dtype=str)
    single_knee = review[review["group"].isin(["A", "B"])].set_index(STUDY_ID)
    assert len(single_knee) == 29

    try:
        cfg = load_config()
        manifest = load_manifest(cfg)
        selection = load_selection(cfg)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"manifest / series selection not available: {exc}")
    studies = set(single_knee.index)
    selection = selection[selection[STUDY_ID].astype(str).isin(studies)]
    paths = selection.loc[selection["selected"].fillna(False).astype(bool), "path"].dropna().astype(str)
    if not len(paths) or not all(Path(p).is_dir() for p in paths):
        pytest.skip("DICOM files of the reviewed studies not available")

    table = derive_laterality(
        manifest[manifest[STUDY_ID].astype(str).isin(studies)],
        selection,
        min_offset_mm=float(cfg.data.get("laterality_min_offset_mm", 20.0)),
        workers=8,
    ).set_index(STUDY_ID)

    got = table["side"].reindex(single_knee.index).fillna("")
    wrong = single_knee.assign(rule_side=got, source=table["side_source"].reindex(single_knee.index))
    wrong = wrong[wrong["rule_side"] != wrong["yc_laterality"]]
    assert wrong.empty, f"laterality differs from the radiologist review:\n{wrong[['group', 'yc_laterality', 'rule_side', 'source']]}"
