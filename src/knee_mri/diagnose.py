"""Read-only diagnostics of finished fold runs: per-class results plus implementation checks.

    python -m knee_mri.cli diagnose work/runs/train_v1/cv3_20260920_002647   # a run group (all folds)
    python -m knee_mri.cli diagnose work/runs/train_v1/fold0_A1 --deep       # one run, GPU checks too

Outputs (default `<first run>/../<group>_diagnose/`):
  * `diagnose_per_class.csv` - per run and for the merged OOF, per target: ROC-AUC on the
    frozen extraction reference, n_pos / n_neg, excluded cells broken down by LLM status,
    and ROC-AUC on the radiologist reference studies that fall into that validation set;
  * `diagnose_checks.csv`    - one PASS / FAIL / WARN / INFO / SKIP row per check;
  * `diagnose_summary.json`  - macro values and the check verdicts.

Nothing here trains or rewrites a run. The `--deep` checks load checkpoints and push a
few real studies through the model; everything else only reads the run artefacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import Config, load_config
from .constants import STUDY_ID, TARGETS
from .metrics import soft_roc_auc
from .schema import read_id_csv
from .utils import LOG, atomic_write_dataframe, atomic_write_json

SATURATION_EPS = 1e-6
RELOAD_TOLERANCE = 2e-3  # bf16 autocast: logits are reproducible to ~1e-3 in probability
NOTEBOOK_BUILDER = Path(__file__).resolve().parents[2] / "notebooks" / "build_04_notebook.py"


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


def resolve_run_dirs(target: str | Path | None, runs: Sequence[str | Path] | None = None) -> list[Path]:
    """An explicit list of run directories, a single run directory, or a `<prefix>_fold<N>` group.

    The prefix form takes every matching directory, and it cannot tell a second seed
    (`..._fold0_s43`) from another fold, which then breaks the OOF. Name the runs with
    `runs=` (CLI: `--runs`) whenever one fold has several runs.
    """
    if runs:
        resolved = [Path(r).resolve() for r in runs]
        missing = [str(r) for r in resolved if not (r / "validation_predictions.csv").exists()]
        if missing:
            raise FileNotFoundError(f"No validation_predictions.csv in: {missing}")
        if len(set(resolved)) != len(resolved):
            raise ValueError("the same run directory was listed twice")
        return resolved
    if target is None:
        raise ValueError("pass a run directory / group prefix, or an explicit --runs list")
    target = Path(target).resolve()
    if (target / "validation_predictions.csv").exists():
        return [target]
    found = sorted(
        p for p in target.parent.glob(f"{target.name}_fold*") if (p / "validation_predictions.csv").exists()
    )
    if not found:
        raise FileNotFoundError(f"No run directory with validation_predictions.csv at {target} or {target}_fold*")
    return found


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC on a binary reference, soft ROC-AUC on a continuous one (never truncated to int)."""
    return soft_roc_auc(np.asarray(y, dtype=float), np.asarray(s, dtype=float))


def _hard(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame[frame[column].isin((0.0, 1.0))]


def _load_status_lookup(details_csv: str | Path | None) -> pd.DataFrame | None:
    if not details_csv or not Path(details_csv).exists():
        return None
    df = read_id_csv(details_csv)
    if "target" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "target"})
    return df[[STUDY_ID, "target", "status"]]


# --------------------------------------------------------------------------------------
# Per-class table
# --------------------------------------------------------------------------------------


def per_class_rows(
    name: str,
    preds: pd.DataFrame,
    reference: pd.DataFrame | None,
    statuses: pd.DataFrame | None,
) -> list[dict]:
    """One row per target for one prediction set (a fold, or the merged OOF)."""
    preds = preds.copy()
    preds["reference_valid"] = preds["reference_valid"].astype(bool)
    if statuses is not None:
        preds = preds.merge(statuses, on=[STUDY_ID, "target"], how="left")
    if reference is not None:
        preds = preds.merge(reference, on=[STUDY_ID, "target"], how="left")

    rows = []
    for target in TARGETS:
        g = preds[preds["target"] == target]
        valid = g[g["reference_valid"]]
        excluded = g[~g["reference_valid"]]
        hard = _hard(valid, "reference")
        row = {
            "run": name,
            "target": target,
            "n_studies": len(g),
            "roc_auc": _auc(hard["reference"].to_numpy(), hard["score"].to_numpy()),
            "soft_auc": _auc(valid["reference"].to_numpy(), valid["score"].to_numpy()),
            "n_pos": int((valid["reference"] == 1).sum()),
            "n_neg": int((valid["reference"] == 0).sum()),
            "n_soft": len(valid) - len(hard),
            "n_excluded": len(excluded),
        }
        if statuses is not None:
            for status, n in excluded["status"].fillna("missing").value_counts().items():
                row[f"excluded_{status}"] = int(n)
        if reference is not None:
            ref = g[g["radiologist"].notna()]
            row.update(
                {
                    "ref58_n": len(ref),
                    "ref58_n_pos": int((ref["radiologist"] == 1).sum()),
                    "ref58_roc_auc": _auc(ref["radiologist"].to_numpy(), ref["score"].to_numpy()),
                }
            )
        rows.append(row)
    return rows


def _macro(frame: pd.DataFrame, column: str) -> float:
    values = frame[column].dropna() if column in frame else pd.Series(dtype=float)
    return float(values.mean()) if len(values) else float("nan")


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, check: str, run: str, status: str, detail: str) -> None:
        self.rows.append({"check": check, "run": run, "status": status, "detail": detail})
        log = LOG.error if status == "FAIL" else LOG.warning if status == "WARN" else LOG.info
        log("%-26s %-28s %-4s %s", check, run, status, detail)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["check", "run", "status", "detail"])


def _header(path: str | Path | None) -> list[str] | None:
    if not path or not Path(path).exists():
        return None
    return list(pd.read_csv(path, nrows=0).columns)


def check_target_order(checks: Checks, cfg: Config, run: Path, payload: dict | None, preds: pd.DataFrame) -> None:
    expected = list(TARGETS)
    problems = []
    if payload is not None and payload.get("target_order") != expected:
        problems.append(f"checkpoint {payload.get('target_order')}")
    stored_order = [t for t in pd.unique(preds["target"])]
    if stored_order != expected:
        problems.append(f"validation_predictions order {stored_order}")
    sources = {
        "train_csv": cfg.paths.train_csv,
        "sample_submission": Path(cfg.paths.data_root) / "sample_submission.csv",
        "reference_csv": cfg.paths.get("reference_csv"),
    }
    for label, path in sources.items():
        header = _header(path)
        if header is None:
            continue
        columns = [c for c in header if c in expected]
        if columns != expected:
            problems.append(f"{label} columns {columns}")
    checks.add(
        "target_order",
        run.name,
        "FAIL" if problems else "PASS",
        "; ".join(problems) if problems else "checkpoint, predictions, train_csv, sample_submission and reference agree",
    )


def check_best_epoch(checks: Checks, run: Path, payload: dict | None) -> None:
    history_path, summary_path = run / "history.csv", run / "val_summary.json"
    if not history_path.exists() or not summary_path.exists():
        checks.add("best_checkpoint", run.name, "SKIP", "history.csv or val_summary.json missing")
        return
    history = pd.read_csv(history_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # Runs from before eval.selection_metric existed logged the selection score as val_macro_roc_auc.
    if "val_selection_score" in history:
        column, key = "val_selection_score", (payload or {}).get("selection_metric", "macro_soft_auc")
    else:
        column, key = "val_macro_roc_auc", "macro_roc_auc"
    best_row = history.loc[history[column].idxmax()]
    epochs = {
        "history_argmax": int(best_row["epoch"]),
        "val_summary": int(summary["epoch"]),
    }
    if payload is not None:
        epochs["best.pt epoch"] = int(payload.get("epoch", -1))
        epochs["best.pt best_epoch"] = int(payload.get("best_epoch", -1))
    agree = len(set(epochs.values())) == 1
    score_gap = abs(float(best_row[column]) - float(summary[key]))
    checks.add(
        "best_checkpoint",
        run.name,
        "PASS" if agree and score_gap < 1e-9 else "FAIL",
        f"{epochs}; history max {key} {best_row[column]:.4f} vs val_summary {summary[key]:.4f}",
    )

    first, best, last = history.iloc[0], best_row, history.iloc[-1]
    ratio = float(last["train_loss"]) / max(float(first["train_loss"]), 1e-12)
    checks.add(
        "train_loss_trajectory",
        run.name,
        "WARN" if ratio < 0.05 else "INFO",
        f"train loss epoch0 {first['train_loss']:.4f} -> best epoch {int(best['epoch'])} {best['train_loss']:.4f} "
        f"-> last epoch {int(last['epoch'])} {last['train_loss']:.4f} (x{ratio:.3f}); "
        f"val {key} at last epoch {last[column]:.4f}",
    )
    if "train_empty_microbatches" in history:
        checks.add(
            "empty_microbatches",
            run.name,
            "INFO",
            f"{int(last['train_empty_microbatches'])} training studies per epoch carry no supervision at all",
        )


def check_continuity(checks: Checks, name: str, preds: pd.DataFrame) -> None:
    worst = []
    saturated_total = 0
    for target in TARGETS:
        s = preds.loc[preds["target"] == target, "score"].to_numpy(dtype=float)
        unique_frac = len(np.unique(s)) / max(len(s), 1)
        saturated = float(((s < SATURATION_EPS) | (s > 1 - SATURATION_EPS)).mean())
        saturated_total += saturated
        worst.append((target, unique_frac, saturated))
    min_unique = min(w[1] for w in worst)
    most_saturated = max(worst, key=lambda w: w[2])
    status = "FAIL" if min_unique < 0.5 else "WARN" if most_saturated[2] > 0.05 else "PASS"
    checks.add(
        "continuous_scores",
        name,
        status,
        f"min unique fraction {min_unique:.3f}; most saturated {most_saturated[0]} "
        f"{most_saturated[2]:.1%} of scores outside [{SATURATION_EPS:g}, 1-{SATURATION_EPS:g}] "
        "(ties at the saturation floor make the ranking arbitrary)",
    )


def check_notebook_ensemble(checks: Checks) -> None:
    if not NOTEBOOK_BUILDER.exists():
        checks.add("inference_ensemble", "notebook", "SKIP", f"{NOTEBOOK_BUILDER} not found")
        return
    text = NOTEBOOK_BUILDER.read_text(encoding="utf-8")
    sigmoid = re.search(r"probs\s*=\s*torch\.sigmoid\(logits\.float\(\)\)", text)
    mean = re.search(r"total\s*/\s*len\(models\)", text)
    prep = "REQUIRE_PREP_MATCH" in text
    ok = bool(sigmoid and mean and prep)
    checks.add(
        "inference_ensemble",
        "notebook",
        "PASS" if ok else "FAIL",
        "build_04_notebook.py averages per-checkpoint sigmoid probabilities and enforces the preprocessing hash"
        if ok
        else f"sigmoid={bool(sigmoid)} mean={bool(mean)} prep_check={prep}",
    )


# --------------------------------------------------------------------------------------
# Deep checks (load checkpoints, run real studies)
# --------------------------------------------------------------------------------------


def check_reload_and_mask(checks: Checks, run: Path, preds: pd.DataFrame, n_studies: int) -> None:
    import torch

    from .dataset import StudyBagDataset, collate_studies
    from .evaluate import load_checkpoint_for_inference, predict_studies
    from .labels import build_label_table
    from .loss import masked_class_normalized_bce
    from .utils import select_device

    run_cfg = load_config(run / "config.yaml")
    model, _ = load_checkpoint_for_inference(run_cfg, run / "best.pt")

    # 1. Reload reproduces the stored validation scores.
    study_ids = list(dict.fromkeys(preds[STUDY_ID].astype(str)))[:n_studies]
    ids, scores = predict_studies(run_cfg, model, study_ids)
    stored = preds.pivot(index=STUDY_ID, columns="target", values="score").loc[ids, list(TARGETS)].to_numpy()
    diff = float(np.abs(scores - stored).max())
    checks.add(
        "reload_reproduces_scores",
        run.name,
        "PASS" if diff < RELOAD_TOLERANCE else "FAIL",
        f"max |p_reload - p_stored| = {diff:.2e} over {len(ids)} studies (tolerance {RELOAD_TOLERANCE:g})",
    )

    # 2. The loss mask on a real batch: zero-weight cells receive exactly zero gradient.
    label_table = build_label_table(run_cfg, sorted(set(preds[STUDY_ID].astype(str))))
    index = label_table.index
    mixed = [s for s in study_ids if 0 < (label_table.weights[index[s]] > 0).sum() < len(TARGETS)]
    if not mixed:
        checks.add("loss_mask_real_batch", run.name, "SKIP", "no study with partly missing labels among the sample")
        return
    spec = select_device(run_cfg.train.amp)
    # predict_studies() moved the first copy to the device inside inference_mode, which turns its
    # parameters into inference tensors; autograd needs a fresh copy.
    model, _ = load_checkpoint_for_inference(run_cfg, run / "best.pt")
    model = model.to(spec.device).eval()
    batch = collate_studies([StudyBagDataset(run_cfg, mixed[:1], label_table, train=False)[0]])
    logits = model(
        batch["images"].to(spec.device),
        batch["slice_valid_mask"].to(spec.device),
        batch["series_present_mask"].to(spec.device),
    ).float()
    logits.retain_grad()
    weights = batch["label_weights"].to(spec.device)
    out = masked_class_normalized_bce(logits, batch["targets"].to(spec.device), weights)
    out.loss.backward()
    grad = logits.grad.detach()
    masked = weights == 0
    leak = float(grad[masked].abs().max()) if bool(masked.any()) else 0.0
    supervised = ~masked
    live = int((grad[supervised].abs() > 0).sum())
    # A supervised cell can lose its gradient legitimately: sigmoid(z) rounds to exactly y in float32
    # once |z| is large and the prediction agrees with the label. That is saturation, not masking.
    saturated = int(((grad[supervised] == 0) & (logits.detach()[supervised].abs() > 16)).sum())
    unexplained = int(supervised.sum()) - live - saturated
    checks.add(
        "loss_mask_real_batch",
        run.name,
        "PASS" if leak == 0.0 and unexplained == 0 else "FAIL",
        f"study {mixed[0][-12:]}: {int(masked.sum())} masked cells, max |grad| on them {leak:.1e}; "
        f"{live}/{int(supervised.sum())} supervised cells have a gradient, {saturated} lost it to logit saturation",
    )
    model.zero_grad(set_to_none=True)
    del logits, grad
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_checkpoint_agreement(checks: Checks, runs: Sequence[Path], reference_ids: list[str]) -> None:
    from scipy.stats import spearmanr

    from .evaluate import load_checkpoint_for_inference, predict_studies

    if len(runs) < 2 or not reference_ids:
        checks.add("checkpoint_agreement", "group", "SKIP", "needs >= 2 runs and the radiologist reference")
        return
    matrices = []
    for run in runs:
        run_cfg = load_config(run / "config.yaml")
        model, _ = load_checkpoint_for_inference(run_cfg, run / "best.pt")
        ids, scores = predict_studies(run_cfg, model, reference_ids)
        order = {s: i for i, s in enumerate(ids)}
        matrices.append(scores[[order[s] for s in reference_ids]])
    per_target = []
    for c, target in enumerate(TARGETS):
        rhos = [
            spearmanr(matrices[a][:, c], matrices[b][:, c]).statistic
            for a in range(len(matrices))
            for b in range(a + 1, len(matrices))
        ]
        per_target.append((target, float(np.nanmean(rhos))))
    low = [f"{t} {r:.2f}" for t, r in per_target if r < 0.5]
    checks.add(
        "checkpoint_agreement",
        "group",
        "WARN" if low else "PASS",
        "mean pairwise Spearman on the reference studies: "
        + ", ".join(f"{t} {r:.2f}" for t, r in per_target)
        + ". NOTE: each reference study is in the training set of all but one checkpoint, so this is a "
        "sanity check for a broken checkpoint, not an independent agreement estimate.",
    )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def diagnose(
    cfg: Config,
    target: str | Path | None = None,
    out_dir: str | Path | None = None,
    details_csv: str | Path | None = None,
    deep: bool = False,
    n_reload_studies: int = 16,
    runs: Sequence[str | Path] | None = None,
    group_name: str | None = None,
) -> dict:
    import torch

    runs = resolve_run_dirs(target, runs)
    group_name = group_name or (runs[0].name.rsplit("_fold", 1)[0] if len(runs) > 1 else runs[0].name)
    out_dir = Path(out_dir) if out_dir else runs[0].parent / f"{group_name}_diagnose"
    out_dir.mkdir(parents=True, exist_ok=True)

    details_csv = details_csv or cfg.paths.get("labels_details_csv")
    default_details = Path(cfg.paths.work_dir) / "labels" / "labels_details_all.csv"
    if not details_csv and default_details.exists():
        details_csv = default_details
    statuses = _load_status_lookup(details_csv)
    if statuses is None:
        LOG.warning("No labels_details file: excluded cells are counted but not broken down by status.")

    reference = None
    reference_ids: list[str] = []
    ref_path = cfg.paths.get("reference_csv")
    if ref_path and Path(ref_path).exists():
        ref = read_id_csv(ref_path)
        reference_ids = sorted(ref[STUDY_ID].astype(str))
        reference = ref[[STUDY_ID] + TARGETS].melt(id_vars=STUDY_ID, var_name="target", value_name="radiologist")

    checks = Checks()
    rows: list[dict] = []
    all_preds = []
    for run in runs:
        preds = pd.read_csv(run / "validation_predictions.csv", dtype={STUDY_ID: "string"})
        all_preds.append(preds)
        payload = None
        if (run / "best.pt").exists():
            payload = torch.load(str(run / "best.pt"), map_location="cpu", weights_only=False)
            payload.pop("optimizer", None)
        rows.extend(per_class_rows(run.name, preds, reference, statuses))
        check_target_order(checks, cfg, run, payload, preds)
        check_best_epoch(checks, run, payload)
        check_continuity(checks, run.name, preds)
        if deep:
            check_reload_and_mask(checks, run, preds, n_reload_studies)
        del payload

    if len(runs) > 1:
        merged = pd.concat(all_preds, ignore_index=True)
        if merged.duplicated(subset=[STUDY_ID, "target"]).any():
            checks.add("oof_disjoint", "group", "FAIL", "a (study, target) pair is predicted by two runs")
        else:
            checks.add("oof_disjoint", "group", "PASS", f"{merged[STUDY_ID].nunique()} studies, each predicted once")
            rows.extend(per_class_rows("oof", merged, reference, statuses))
            check_continuity(checks, "oof", merged)

    check_notebook_ensemble(checks)
    if deep:
        check_checkpoint_agreement(checks, runs, reference_ids)

    table = pd.DataFrame(rows)
    atomic_write_dataframe(table, out_dir / "diagnose_per_class.csv")
    atomic_write_dataframe(checks.frame(), out_dir / "diagnose_checks.csv")

    macro = {
        name: {
            "macro_roc_auc": _macro(g, "roc_auc"),
            "macro_soft_auc": _macro(g, "soft_auc"),
            "ref58_macro_roc_auc": _macro(g, "ref58_roc_auc"),
            "ref58_n_studies": int(g["ref58_n"].max()) if "ref58_n" in g else 0,
        }
        for name, g in table.groupby("run", sort=False)
    }
    verdicts = checks.frame()["status"].value_counts().to_dict()
    summary = {
        "runs": [str(r) for r in runs],
        "macro": macro,
        "check_counts": verdicts,
        "details_csv": str(details_csv) if details_csv else None,
        "deep": deep,
        "note": (
            "roc_auc is scored on the hard (0/1) cells and soft_auc on every valid cell of each run's "
            "frozen extraction-derived reference; ref58_roc_auc on the "
            "radiologist reference studies inside that validation set (small: a directional signal only)."
        ),
    }
    atomic_write_json(out_dir / "diagnose_summary.json", summary)

    printable = table[["run", "target", "roc_auc", "soft_auc", "n_pos", "n_neg", "n_soft", "n_excluded"] + (["ref58_roc_auc"] if reference is not None else [])]
    LOG.info("Per-class results:\n%s", printable.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    LOG.info("Macro: %s", json.dumps(macro, indent=1))
    LOG.info("Diagnostics written to %s", out_dir)
    return summary
