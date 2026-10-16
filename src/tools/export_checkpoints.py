"""Slim inference copies of an ensemble's checkpoints, one Kaggle-dataset folder per checkpoint.

Before every Kaggle submission: the training checkpoint also carries what only an exact resume
needs (the raw non-EMA weights, the AdamW moments, scheduler / scaler / RNG state, the history),
about three quarters of the file. Inference (`evaluate.load_checkpoint_for_inference` and the
04 notebook) reads only `model` (the selected, EMA weights when EMA was on) and the metadata, so
the export keeps exactly those and drops the rest. The weights are copied bit for bit (same
dtype); every export is reloaded and compared with its source before it counts as written.

    python src/tools/export_checkpoints.py --ensemble holdout6_v11 --out work/export_v11

Layout: <out>/<Kaggle dataset name>/{best.pt, config.yaml}, the names taken from the ensemble's
entry in notebooks/build_04_notebook.py, so each folder uploads as one dataset as it is. An
exported best.pt cannot resume training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from knee_mri.config import load_config  # noqa: E402
from knee_mri.evaluate import load_checkpoint_for_inference  # noqa: E402

# Only an exact resume (train.Trainer.load_checkpoint) reads these.
TRAINING_ONLY_KEYS = (
    "model_raw", "optimizer", "scheduler", "scaler", "rng_state", "rng_state_by_rank", "train_state",
    "resume_signature",
)


def ensemble_members(name: str) -> list[tuple[str, str]]:
    """(run dir under work/runs, Kaggle dataset name) of a single-run ensemble in build_04_notebook."""
    spec = importlib.util.spec_from_file_location("build_04_notebook", REPO / "notebooks" / "build_04_notebook.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    if name not in builder.ENSEMBLES:
        raise SystemExit(f"unknown ensemble {name!r}; choose from {sorted(builder.ENSEMBLES)}")
    members = []
    for run, dataset, *kind in builder.ENSEMBLES[name]["members"]:
        if kind != [builder.SINGLE]:
            raise SystemExit(f"{run}: fold members are not supported, export each fold run separately")
        members.append((run, dataset))
    return members


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def export_one(run_dir: Path, out_dir: Path) -> tuple[float, float]:
    source = run_dir / "best.pt"
    payload = torch.load(str(source), map_location="cpu", weights_only=False)
    slim = {k: v for k, v in payload.items() if k not in TRAINING_ONLY_KEYS}
    slim["export"] = {
        "source": str(source.relative_to(REPO)).replace("\\", "/"),
        "source_sha256": sha256(source),
        "dropped_keys": [k for k in TRAINING_ONLY_KEYS if k in payload],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "best.pt"
    torch.save(slim, str(target))
    shutil.copy2(run_dir / "config.yaml", out_dir / "config.yaml")

    # Reload as inference does and compare every weight with the source.
    cfg = load_config(out_dir / "config.yaml", resolve=False)
    model, reloaded = load_checkpoint_for_inference(cfg, target)
    state = model.state_dict()
    if set(state) != set(payload["model"]):
        raise RuntimeError(f"{target}: state_dict keys differ from the source")
    for key, value in payload["model"].items():
        if value.dtype != state[key].dtype or not torch.equal(value, state[key]):
            raise RuntimeError(f"{target}: weight {key} differs from the source")
    for key in ("version", "target_order", "config", "model_description", "versions", "best_epoch", "best_score"):
        if reloaded.get(key) != payload.get(key):
            raise RuntimeError(f"{target}: {key} differs from the source")
    return source.stat().st_size / 2**20, target.stat().st_size / 2**20


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ensemble", required=True, help="Ensemble name in notebooks/build_04_notebook.py ENSEMBLES")
    parser.add_argument("--out", required=True, help="Output folder, e.g. work/export_v11")
    parser.add_argument("--runs-dir", default=str(REPO / "work" / "runs"), help="Root of the run directories")
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    total_in = total_out = 0.0
    for run, dataset in ensemble_members(args.ensemble):
        before, after = export_one(Path(args.runs_dir) / run, out / dataset)
        total_in, total_out = total_in + before, total_out + after
        print(f"{dataset:28s} {before:7.0f} MB -> {after:6.0f} MB   ({run})")
    print(f"total {total_in:.0f} MB -> {total_out:.0f} MB; verified against the sources; written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
