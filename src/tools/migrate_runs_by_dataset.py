"""One-off: move flat work/runs/<run> folders into work/runs/<train_csv stem>/<run>.

A training run's dataset is the stem of `paths.train_csv` in its config.yaml. Folders
without a config (`*_oof`, `*_diagnose`) follow their fold runs: the `runs` list of
diagnose_summary.json when present, else the sibling folders sharing the name prefix.
Anything ambiguous or unresolved is listed and left in place.

    python src/tools/migrate_runs_by_dataset.py            # dry run: prints the plan
    python src/tools/migrate_runs_by_dataset.py --apply    # moves the folders
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
# The stopped CV of 2026-10-01: moved by hand.
DEFAULT_EXCLUDE = ["R50_E3_img320_trainv3_cv3_20261001_103636"]
DERIVED_SUFFIXES = ("_oof", "_diagnose")


def _basename(path: str) -> str:
    """Last component of a Windows or POSIX path, whichever OS wrote it."""
    return re.split(r"[\\/]", str(path).rstrip("\\/"))[-1]


def _dataset_of_config(config_path: Path) -> str | None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    train_csv = (cfg.get("paths") or {}).get("train_csv")
    return Path(_basename(train_csv)).stem if train_csv else None


def _sibling_candidates(name: str, trained: dict[str, str]) -> list[str]:
    base = name
    for suffix in DERIVED_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    found = [run for run in trained if run.startswith(base + "_")]
    seed = re.fullmatch(r"(.+)_(s\d+)", base)
    if not found and seed:  # E1_ema0999_s42_oof <- E1_ema0999_fold<N>_s42
        core, s = seed.groups()
        found = [run for run in trained if run.startswith(core + "_fold") and run.endswith("_" + s)]
    return found


def plan(runs_root: Path, exclude: list[str]) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, str]]]:
    dirs = sorted(p for p in runs_root.iterdir() if p.is_dir())
    trained: dict[str, str] = {}
    for d in dirs:
        if (d / "config.yaml").exists():
            dataset = _dataset_of_config(d / "config.yaml")
            if dataset:
                trained[d.name] = dataset

    moves: list[tuple[Path, str, str]] = []
    skipped: list[tuple[Path, str]] = []
    for d in dirs:
        if any(d.name.startswith(prefix) for prefix in exclude):
            skipped.append((d, "excluded"))
            continue
        if not (d / "config.yaml").exists() and any((c / "config.yaml").exists() for c in d.iterdir() if c.is_dir()):
            continue  # already a dataset folder
        if d.name in trained:
            moves.append((d, trained[d.name], "config.yaml"))
            continue
        if (d / "config.yaml").exists():
            skipped.append((d, "config.yaml has no paths.train_csv"))
            continue

        sources, how = [], "name prefix"
        summary = d / "diagnose_summary.json"
        if summary.exists():
            listed = json.loads(summary.read_text(encoding="utf-8")).get("runs") or []
            sources, how = [_basename(r) for r in listed], "diagnose_summary.json"
        if not sources:
            sources = _sibling_candidates(d.name, trained)
        datasets = {trained.get(s) for s in sources}
        if not sources or None in datasets:
            skipped.append((d, f"source runs not found ({how}: {sources})"))
        elif len(datasets) > 1:
            skipped.append((d, f"source runs disagree: {sorted(datasets)}"))
        else:
            moves.append((d, datasets.pop(), f"{how}: {len(sources)} run(s)"))
    return moves, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", default=str(REPO / "work" / "runs"))
    parser.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE, help="Run-name prefixes to leave in place")
    parser.add_argument("--apply", action="store_true", help="Move the folders (default: dry run)")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    moves, skipped = plan(runs_root, args.exclude)
    for src, dataset, how in moves:
        print(f"{src.name:<55} -> {dataset:<10} [{how}]")
    for src, why in skipped:
        print(f"{src.name:<55} SKIP  {why}")
    counts: dict[str, int] = {}
    for _, dataset, _ in moves:
        counts[dataset] = counts.get(dataset, 0) + 1
    print(f"\n{len(moves)} to move {dict(sorted(counts.items()))}, {len(skipped)} skipped")

    if not args.apply:
        print("Dry run - nothing moved. Re-run with --apply.")
        return 0
    failed = 0
    for src, dataset, _ in moves:
        dst = runs_root / dataset / src.name
        if dst.exists():
            print(f"EXISTS, not moved: {dst}")
            failed += 1
            continue
        dst.parent.mkdir(exist_ok=True)
        shutil.move(str(src), str(dst))
    print(f"Moved {len(moves) - failed} folder(s) under {runs_root}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
