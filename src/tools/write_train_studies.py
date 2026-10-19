"""Backfill run_dir/train_studies.csv for runs trained before checkpoints recorded their studies.

`train.init_weights` refuses a checkpoint whose training studies it cannot check against the new
run's validation studies (Trainer._init_leak_check). Newer checkpoints carry `train_study_ids`;
for an older run this tool reconstructs them from the run's own config: the training studies of
`split.fold` in its `paths.splits_csv`. That is a superset of what was trained (the coverage gate
may have dropped a few studies), which only makes the leak check stricter.

    python src/tools/write_train_studies.py work/runs/train_v11/V2S_E3_img320_s42_trainv11_holdout450_20261016_062318 ...

A run trained on another machine names that machine's splits path; the tool then falls back to
<repo>/work/splits/<same folder>/splits.csv, or to --splits. It refuses to write when the
reconstruction does not reproduce the run's own validation studies (validation_reference.csv)
or its training-study count (label_counts_per_split.csv).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from knee_mri.config import load_config  # noqa: E402
from knee_mri.constants import STUDY_ID, TRAIN_STUDIES_FILE  # noqa: E402
from knee_mri.splits import fold_study_ids, load_splits  # noqa: E402


def splits_for(run_dir: Path, configured: str | None, override: str | None) -> Path:
    candidates = [Path(override)] if override else []
    if configured:
        candidates += [Path(configured), REPO / "work" / "splits" / Path(configured).parent.name / "splits.csv"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{run_dir.name}: no splits file among {[str(c) for c in candidates]}; pass --splits")


def reconstruct(run_dir: Path, override: str | None) -> tuple[list[str], Path]:
    cfg = load_config(run_dir / "config.yaml", resolve=False)
    path = splits_for(run_dir, cfg.paths.get("splits_csv"), override)
    train_ids, val_ids = fold_study_ids(load_splits(cfg, path), int(cfg.split.fold))

    reference = run_dir / "validation_reference.csv"
    if reference.exists():
        validated = set(pd.read_csv(reference, dtype={STUDY_ID: "string"})[STUDY_ID].astype(str))
        if not validated <= set(val_ids):
            raise ValueError(f"{run_dir.name}: {len(validated - set(val_ids))} validation studies are not in fold "
                             f"{cfg.split.fold} of {path}; this is not the split the run used")
        if validated & set(train_ids):
            raise ValueError(f"{run_dir.name}: the reconstructed training set holds validation studies")
    counts = run_dir / "label_counts_per_split.csv"
    if counts.exists():
        table = pd.read_csv(counts)
        trained = int(table.loc[table["split"] == "train", "n_studies"].iloc[0])
        if not 0 <= len(train_ids) - trained <= max(5, len(train_ids) // 100):
            raise ValueError(f"{run_dir.name}: {len(train_ids)} reconstructed vs {trained} trained studies")
    return train_ids, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="run directories")
    parser.add_argument("--splits", default=None, help="splits.csv to use instead of the run's configured one")
    parser.add_argument("--force", action="store_true", help="overwrite an existing train_studies.csv")
    args = parser.parse_args()

    failed = 0
    for run in args.runs:
        run_dir = Path(run)
        target = run_dir / TRAIN_STUDIES_FILE
        if target.exists() and not args.force:
            print(f"skip   {run_dir.name}: {TRAIN_STUDIES_FILE} exists")
            continue
        try:
            train_ids, path = reconstruct(run_dir, args.splits)
        except (ValueError, FileNotFoundError) as error:
            print(f"FAILED {error}")
            failed += 1
            continue
        pd.DataFrame({STUDY_ID: train_ids}).to_csv(target, index=False)
        print(f"wrote  {run_dir.name}: {len(train_ids)} training studies (from {path})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
