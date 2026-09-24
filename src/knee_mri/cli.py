"""Command line entry points.

    python -m knee_mri.cli <command> [--config src/config.yaml] [--set key.sub=value ...]

Every command is importable as a function too, so the notebook can call the same code
without duplicating it. A `main()` guard is used throughout for Windows multiprocessing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .constants import STUDY_ID
from .utils import LOG, atomic_write_json, log_environment, seed_everything


def _common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: src/config.yaml)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override, e.g. --set data.image_size=288",
    )
    return parser


def _load(args: argparse.Namespace):
    cfg = load_config(args.config, args.overrides)
    seed_everything(int(cfg.seed))
    return cfg


# --------------------------------------------------------------------------------------


def cmd_validate_schema(args: argparse.Namespace) -> int:
    from .labels import build_label_table, check_training_readiness, save_label_artifacts
    from .schema import read_id_csv, validate_inputs

    cfg = _load(args)
    log_environment()
    report = validate_inputs(cfg, dicom_sample_limit=args.dicom_sample_limit)
    report.log()
    out_dir = Path(cfg.paths.work_dir) / "schema"
    report.save(out_dir)
    LOG.info("Schema artefacts written to %s", out_dir)

    if not report.ok():
        LOG.error("Schema validation found %d error(s).", len(report.errors))
        if not args.keep_going:
            return 2

    try:
        train_df = read_id_csv(cfg.paths.train_csv)
        study_ids = sorted(set(train_df[STUDY_ID].dropna().astype(str)))
        table = build_label_table(cfg, study_ids)
        save_label_artifacts(table, out_dir)
        counts = table.counts_frame()
        LOG.info("Label counts per target:\n%s", counts.to_string(index=False))
        readiness = check_training_readiness(cfg, table, len(study_ids))
        readiness.log()
        atomic_write_json(out_dir / "readiness.json", {"ok": readiness.ok, "reasons": readiness.reasons, **readiness.details})
    except Exception as exc:
        LOG.error("Label table could not be built: %s", exc)
        return 3
    return 0


def cmd_build_manifest(args: argparse.Namespace) -> int:
    from .manifest import build_manifest, patient_group_audit, select_series
    from .schema import read_id_csv
    from .utils import atomic_write_dataframe

    cfg = _load(args)
    study_ids = None
    if args.studies:
        study_ids = [line.strip() for line in Path(args.studies).read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = build_manifest(cfg, study_ids=study_ids, limit=args.limit)

    series_meta = None
    if Path(cfg.paths.train_series_csv).exists():
        series_meta = read_id_csv(cfg.paths.train_series_csv)
    select_series(cfg, manifest, series_meta)

    audit = patient_group_audit(manifest, int(cfg.split.max_studies_per_patient_id))
    atomic_write_dataframe(audit, Path(cfg.paths.work_dir) / "manifest" / "patient_audit.csv")
    LOG.info("Patient-ID audit written (%d studies)", len(audit))
    return 0


def cmd_select_series(args: argparse.Namespace) -> int:
    from .manifest import load_manifest, select_series
    from .schema import read_id_csv

    cfg = _load(args)
    manifest = load_manifest(cfg)
    series_meta = read_id_csv(cfg.paths.train_series_csv) if Path(cfg.paths.train_series_csv).exists() else None
    select_series(cfg, manifest, series_meta)
    return 0


def cmd_build_cache(args: argparse.Namespace) -> int:
    from .manifest import load_selection
    from .preprocess import build_cache, cache_root

    cfg = _load(args)
    selection = load_selection(cfg)
    study_ids = None
    if args.limit_studies:
        study_ids = sorted(set(selection[selection["selected"].fillna(False).astype(bool)][STUDY_ID]))[: args.limit_studies]
    report = build_cache(cfg, selection, study_ids=study_ids, workers=args.workers)
    LOG.info("Cache root: %s", cache_root(cfg))
    LOG.info("Status counts: %s", report["status"].value_counts().to_dict())
    return 0 if not (report["status"] == "failed").any() else 1


def cmd_make_splits(args: argparse.Namespace) -> int:
    from .labels import build_label_table, load_reference_table
    from .schema import read_id_csv
    from .splits import make_splits

    cfg = _load(args)
    train_df = read_id_csv(cfg.paths.train_csv)
    study_ids = sorted(set(train_df[STUDY_ID].dropna().astype(str)))
    label_table = build_label_table(cfg, study_ids)

    audit_path = Path(cfg.paths.work_dir) / "manifest" / "patient_audit.csv"
    patient_audit = pd.read_csv(audit_path, dtype={STUDY_ID: "string", "patient_id": "string"}) if audit_path.exists() else None
    if patient_audit is None:
        LOG.warning("No patient_audit.csv (run build-manifest first); falling back to study-level grouping.")

    reference_ids = None
    reference_table = load_reference_table(cfg, study_ids)
    if reference_table is not None:
        reference_ids = [
            sid for sid, weight in zip(reference_table.study_ids, reference_table.weights.sum(axis=1)) if weight > 0
        ]
        LOG.info("Radiologist reference covers %d studies", len(reference_ids))

    make_splits(
        cfg,
        study_ids=study_ids,
        label_table=label_table,
        patient_audit=patient_audit,
        train_df=train_df,
        reference_study_ids=reference_ids,
    )
    return 0


def cmd_qc(args: argparse.Namespace) -> int:
    from .manifest import load_manifest, load_selection
    from .qc import build_qc_gallery

    cfg = _load(args)
    study_ids = None
    if args.studies:
        study_ids = [line.strip() for line in Path(args.studies).read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = build_qc_gallery(
        cfg,
        load_selection(cfg),
        load_manifest(cfg),
        n_studies=args.n_studies,
        out_dir=args.out_dir,
        study_ids=study_ids,
    )
    LOG.info("QC summary: %s", json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))
    return 0


def cmd_qc_edges(args: argparse.Namespace) -> int:
    import numpy as np

    from .manifest import load_selection
    from .qc import crop_edge_audit

    cfg = _load(args)
    if args.studies:
        study_ids = [line.strip() for line in Path(args.studies).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        selection = load_selection(cfg)
        pool = sorted(set(selection[selection["selected"].fillna(False).astype(bool)][STUDY_ID].astype(str)))
        rng = np.random.default_rng(int(cfg.seed))
        study_ids = sorted(rng.choice(pool, size=min(args.n_studies, len(pool)), replace=False))
    out_dir = args.out_dir or Path(cfg.paths.work_dir) / "qc_edges"
    summary = crop_edge_audit(
        cfg, study_ids, out_dir, cut_threshold=args.cut_threshold, band_px=args.band_px, tissue_threshold=args.tissue_threshold
    )
    LOG.info("Crop-edge audit (%s): %s", out_dir, json.dumps(summary["share_cut"], indent=1))
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from .selftest import run_all_checks

    cfg = _load(args)
    results = run_all_checks(cfg, quick=args.quick)
    failures = [name for name, ok, _ in results if not ok]
    for name, ok, detail in results:
        (LOG.info if ok else LOG.error)("%-38s %s %s", name, "PASS" if ok else "FAIL", detail or "")
    LOG.info("%d/%d checks passed", len(results) - len(failures), len(results))
    return 1 if failures else 0


def cmd_train(args: argparse.Namespace) -> int:
    from .train import run_fold, run_overfit, run_synthetic

    cfg = _load(args)
    if args.mode == "synthetic":
        summary = run_synthetic(cfg, n_studies=args.n_studies or 16, epochs=args.epochs, name=args.name)
    elif args.mode == "overfit":
        summary = run_overfit(cfg, n_studies=args.n_studies or 8, name=args.name)
    else:
        summary = run_fold(cfg, study_limit=args.n_studies, name=args.name, allow_unready=args.allow_unready)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_checkpoint

    cfg = _load(args)
    summary = evaluate_checkpoint(
        cfg,
        args.checkpoint,
        out_dir=args.out_dir,
        partition=args.partition,
        allow_data_overrides=args.allow_data_override or (),
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_merge_oof(args: argparse.Namespace) -> int:
    from .evaluate import merge_oof

    cfg = _load(args)
    path = merge_oof(cfg, args.predictions, out_path=args.out)
    print(str(path))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_html_report

    cfg = _load(args)
    counts_path = Path(cfg.paths.work_dir) / "schema" / "labels_counts.csv"
    counts = pd.read_csv(counts_path) if counts_path.exists() else None
    qc_dir = args.qc_dir or Path(cfg.paths.work_dir) / "qc"
    path = build_html_report(args.run_dir, qc_dir=qc_dir, label_counts=counts)
    print(str(path))
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    from .diagnose import diagnose

    cfg = _load(args)
    summary = diagnose(
        cfg,
        args.target,
        out_dir=args.out_dir,
        details_csv=args.details,
        deep=args.deep,
        n_reload_studies=args.n_reload_studies,
        runs=args.runs,
        group_name=args.group_name,
    )
    return 1 if summary["check_counts"].get("FAIL", 0) else 0


def cmd_merge_label_details(args: argparse.Namespace) -> int:
    from .merge_label_details import merge_label_details

    cfg = _load(args)
    print(str(merge_label_details(cfg, args.details, out_path=args.out)))
    return 0


def cmd_label_audit(args: argparse.Namespace) -> int:
    from .label_audit import sample_audit, summarize_audit

    cfg = _load(args)
    if args.action == "summarize":
        summarize_audit(cfg, args.sheet)
        return 0
    groups = None
    if args.groups:
        groups = {name.strip(): int(size) for name, size in (item.split("=") for item in args.groups.split(","))}
    try:
        sample_audit(cfg, out_dir=args.out_dir, details_csv=args.details, group_sizes=groups, seed=args.seed, force=args.force)
    except FileExistsError as exc:
        LOG.error("%s", exc)
        return 1
    return 0


def cmd_freeze_reference(args: argparse.Namespace) -> int:
    from .labels import build_label_table
    from .splits import load_splits
    from .train import EvaluationReference

    cfg = _load(args)
    out = Path(args.out) if args.out else Path(cfg.paths.work_dir) / "labels" / "frozen_reference.csv"
    if out.exists() and not args.force:
        LOG.error("%s exists. A frozen reference must not change between experiments; pass --force to replace it.", out)
        return 1
    study_ids = sorted(set(load_splits(cfg)[STUDY_ID].astype(str)))
    table = build_label_table(cfg, study_ids)
    reference = EvaluationReference.from_table(table, study_ids)
    reference.save(out)
    atomic_write_json(
        out.with_name(out.stem + "_meta.json"),
        {
            "label_source": table.source,
            "label_policy": table.policy,
            "n_studies": len(study_ids),
            "n_valid_cells": int(reference.valid.sum()),
            "n_positive_cells": int(((reference.values == 1.0) & reference.valid).sum()),
            "n_soft_cells": int((~np.isin(reference.values, (0.0, 1.0)) & reference.valid).sum()),
            "eff_positive_mass": float((reference.values * reference.valid).sum()),
        },
    )
    LOG.info("Frozen reference written to %s (%d valid cells). Set paths.frozen_reference_csv to use it.", out, int(reference.valid.sum()))
    return 0


# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="knee_mri", description="RSNA knee MRI EfficientNet-B0 2.5D MIL baseline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = _common(sub.add_parser("validate-schema", help="Validate inputs, label schema and coverage"))
    p.add_argument("--dicom-sample-limit", type=int, default=None, help="Only scan the first N study directories")
    p.add_argument("--keep-going", action="store_true", help="Continue to the label checks despite schema errors")
    p.set_defaults(func=cmd_validate_schema)

    p = _common(sub.add_parser("build-manifest", help="Inventory DICOMs, then select one series per slot"))
    p.add_argument("--limit", type=int, default=None, help="Only scan the first N series directories")
    p.add_argument("--studies", default=None, help="File with one StudyInstanceUID per line")
    p.set_defaults(func=cmd_build_manifest)

    p = _common(sub.add_parser("select-series", help="Re-run series selection on an existing manifest"))
    p.set_defaults(func=cmd_select_series)

    p = _common(sub.add_parser("build-cache", help="Preprocess and cache the selected series"))
    p.add_argument("--limit-studies", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(func=cmd_build_cache)

    p = _common(sub.add_parser("make-splits", help="Create the immutable splits.csv"))
    p.set_defaults(func=cmd_make_splits)

    p = _common(sub.add_parser("qc", help="Build the QC gallery"))
    p.add_argument("--n-studies", type=int, default=12)
    p.add_argument("--studies", default=None, help="File with one StudyInstanceUID per line (replaces the automatic pick)")
    p.add_argument("--out-dir", default=None, help="Default: <work_dir>/qc")
    p.set_defaults(func=cmd_qc)

    p = _common(sub.add_parser("qc-edges", help="Measure how often the physical crop cuts anatomy at each edge"))
    p.add_argument("--n-studies", type=int, default=500, help="Random studies (seeded) when --studies is not given")
    p.add_argument("--studies", default=None, help="File with one StudyInstanceUID per line")
    p.add_argument("--out-dir", default=None, help="Default: <work_dir>/qc_edges")
    p.add_argument("--cut-threshold", type=float, default=0.3, help="Edge band tissue fraction above which the edge counts as cut")
    p.add_argument("--band-px", type=int, default=4)
    p.add_argument("--tissue-threshold", type=float, default=0.1)
    p.set_defaults(func=cmd_qc_edges)

    p = _common(sub.add_parser("selftest", help="Run the synthetic failure-mode checks"))
    p.add_argument("--quick", action="store_true", help="Skip the checks that build an encoder")
    p.set_defaults(func=cmd_selftest)

    p = _common(sub.add_parser("train", help="Train: synthetic smoke test, tiny overfit check, or a fold"))
    p.add_argument("--mode", choices=["synthetic", "overfit", "fold"], default="fold")
    p.add_argument("--n-studies", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="Synthetic mode only")
    p.add_argument("--name", default=None, help="Run directory name")
    p.add_argument(
        "--allow-unready",
        action="store_true",
        help="Train despite a failed label-readiness check (results are diagnostic only)",
    )
    p.set_defaults(func=cmd_train)

    p = _common(sub.add_parser("evaluate", help="Evaluate a saved checkpoint"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--partition", choices=["validation", "reference_holdout"], default="validation")
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--allow-data-override",
        action="append",
        metavar="KEY",
        help="Accept a data.* input setting that differs from the checkpoint's training config (repeatable)",
    )
    p.set_defaults(func=cmd_evaluate)

    p = _common(sub.add_parser("merge-oof", help="Merge complete, verified per-fold predictions"))
    p.add_argument("predictions", nargs="+")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_merge_oof)

    p = _common(sub.add_parser("report", help="Build the offline HTML report for a run directory"))
    p.add_argument("run_dir")
    p.add_argument("--qc-dir", default=None)
    p.set_defaults(func=cmd_report)

    p = _common(sub.add_parser("diagnose", help="Per-class results and implementation checks of finished runs"))
    p.add_argument(
        "target",
        nargs="?",
        default=None,
        help="A run directory, or a run-group prefix such as work/runs/cv3_20260920_002647",
    )
    p.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Explicit run directories, one per fold. Use this when a fold has several seeds: "
        "the prefix form would take them all and skip the OOF.",
    )
    p.add_argument("--group-name", default=None, help="Name of the diagnose output directory (default: the prefix)")
    p.add_argument("--details", default=None, help="labels_details file used to break excluded cells down by status")
    p.add_argument("--deep", action="store_true", help="Also load checkpoints: reload, loss-mask and agreement checks")
    p.add_argument("--n-reload-studies", type=int, default=16)
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_diagnose)

    p = _common(sub.add_parser("merge-label-details", help="Merge LLM labels_details exports into one training file"))
    p.add_argument("details", nargs="+", help="labels_details.csv files of the labelling runs")
    p.add_argument("--out", default=None, help="Default: <work_dir>/labels/labels_details_all.csv")
    p.set_defaults(func=cmd_merge_label_details)

    p = _common(sub.add_parser("label-audit", help="Draw the targeted label-audit sample, or summarise a filled one"))
    p.add_argument("action", choices=["sample", "summarize"])
    p.add_argument("--groups", default=None, help="e.g. random=40,synovitis=20,mcl=20,many_empty=20")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--details", default=None, help="Default: <work_dir>/labels/labels_details_all.csv")
    p.add_argument("--out-dir", default=None, help="Default: <work_dir>/audit")
    p.add_argument("--sheet", default=None, help="summarize: filled label_audit.csv")
    p.add_argument("--force", action="store_true", help="sample: overwrite an existing audit sheet")
    p.set_defaults(func=cmd_label_audit)

    p = _common(sub.add_parser("freeze-reference", help="Freeze the validation reference for all studies"))
    p.add_argument("--out", default=None, help="Default: <work_dir>/labels/frozen_reference.csv")
    p.add_argument("--force", action="store_true", help="Replace an existing frozen reference")
    p.set_defaults(func=cmd_freeze_reference)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
