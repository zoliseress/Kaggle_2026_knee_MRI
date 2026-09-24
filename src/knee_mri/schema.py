"""Input schema validation and coverage checks.

Everything here is read-only and defensive: identifiers are read as strings, joins
are key-based (never positional), and anything unexpected becomes an explicit
finding instead of a silent repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .constants import PLANES, SERIES_ID, STUDY_ID, TARGETS
from .utils import LOG, atomic_write_dataframe, atomic_write_json

TRAIN_CSV_REQUIRED = [STUDY_ID]
TRAIN_SERIES_CSV_REQUIRED = [STUDY_ID, SERIES_ID]

# Expected fields of the optional report-extraction exports. They are documented
# here so a missing file produces an actionable message instead of a guess.
EXPECTED_EXPORT_SCHEMAS: dict[str, dict[str, Any]] = {
    "labels_details_csv": {
        "required": [STUDY_ID, "target", "status"],
        "optional": ["basis", "value", "soft_target", "p_positive", "confidence", "evidence", "review_flag"],
        "note": (
            "Preferred source: one row per (study, target) with the extraction status. "
            "`status` in {positive, negative, uncertain, not_mentioned}; for `negative` the "
            "`basis` column distinguishes explicit_absence / below_threshold / borderline. "
            "`soft_target` or `p_positive` (one of them) is P(positive) in [0, 1], empty on not_mentioned."
        ),
    },
    "labels_statuses_csv": {
        "required": [STUDY_ID],
        "optional": TARGETS,
        "note": "Wide export with the four text statuses per target; no `basis` detail.",
    },
    "labels_predictions_csv": {
        "required": [STUDY_ID] + TARGETS,
        "optional": [],
        "note": "Wide numeric export; empty cells are unresolved and stay UNKNOWN (weight 0).",
    },
    "labels_predictions_exclude_borderline_csv": {
        "required": [STUDY_ID] + TARGETS,
        "optional": [],
        "note": "Same as labels_predictions_csv, with borderline decisions left empty as well.",
    },
    "reference_csv": {
        "required": [STUDY_ID] + TARGETS,
        "optional": [],
        "note": "Small radiologist reference set. Diagnostic audit only, never a training label source.",
    },
}


@dataclass
class Finding:
    level: str  # "info" | "warning" | "error"
    code: str
    message: str
    count: int = 0

    def as_row(self) -> dict:
        return {"level": self.level, "code": self.code, "message": self.message, "count": self.count}


@dataclass
class SchemaReport:
    findings: list[Finding] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def add(self, level: str, code: str, message: str, count: int = 0) -> None:
        self.findings.append(Finding(level, code, message, count))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warning"]

    def ok(self) -> bool:
        return not self.errors

    def log(self) -> None:
        for finding in self.findings:
            log_fn = {"error": LOG.error, "warning": LOG.warning}.get(finding.level, LOG.info)
            suffix = f" (n={finding.count})" if finding.count else ""
            log_fn("[%s] %s%s", finding.code, finding.message, suffix)

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_dataframe(pd.DataFrame([f.as_row() for f in self.findings]), out_dir / "schema_findings.csv")
        atomic_write_json(out_dir / "schema_summary.json", self.summary)
        for name, table in self.tables.items():
            atomic_write_dataframe(table, out_dir / f"schema_{name}.csv")
        return out_dir / "schema_findings.csv"

    def raise_if_failed(self) -> None:
        if self.errors:
            lines = "\n".join(f"  - [{f.code}] {f.message}" for f in self.errors)
            raise ValueError(f"Input schema validation failed with {len(self.errors)} error(s):\n{lines}")


def read_id_csv(path: str | Path, id_columns: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV keeping identifier columns as strings (never as floats)."""
    id_columns = id_columns or [STUDY_ID, SERIES_ID]
    df = pd.read_csv(path, dtype={col: "string" for col in id_columns}, keep_default_na=True)
    for col in id_columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    return df


def _check_columns(df: pd.DataFrame, required: list[str], name: str, report: SchemaReport) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.add("error", f"{name}.missing_columns", f"{name} is missing required columns {missing}", len(missing))


def validate_targets_header(df: pd.DataFrame, name: str, report: SchemaReport) -> list[str]:
    """Check the project target order against the actual file header."""
    present = [t for t in TARGETS if t in df.columns]
    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        report.add("warning", f"{name}.missing_targets", f"{name} has no column for targets {missing}", len(missing))
    if present:
        order_in_file = [c for c in df.columns if c in set(TARGETS)]
        if order_in_file != present:
            report.add(
                "warning",
                f"{name}.target_order",
                f"{name} lists targets in a different order than TARGETS; the project order is enforced by key, "
                f"file order={order_in_file}",
            )
    return present


def validate_train_csv(path: Path, report: SchemaReport) -> pd.DataFrame | None:
    if not path.exists():
        report.add("error", "train_csv.missing", f"train.csv not found at {path}")
        return None
    df = read_id_csv(path)
    _check_columns(df, TRAIN_CSV_REQUIRED, "train_csv", report)
    if STUDY_ID not in df.columns:
        return df
    dupes = int(df[STUDY_ID].duplicated().sum())
    if dupes:
        report.add("error", "train_csv.duplicate_studies", "train.csv contains duplicated StudyInstanceUID rows", dupes)
    blanks = int(df[STUDY_ID].isna().sum() + (df[STUDY_ID] == "").sum())
    if blanks:
        report.add("error", "train_csv.blank_study_id", "train.csv has blank StudyInstanceUID values", blanks)
    validate_targets_header(df, "train_csv", report)
    report.summary["train_csv_rows"] = int(len(df))
    report.summary["train_csv_unique_studies"] = int(df[STUDY_ID].nunique())
    if "Report" in df.columns:
        empty_reports = int(df["Report"].isna().sum())
        report.summary["train_csv_empty_reports"] = empty_reports
        if empty_reports:
            report.add("warning", "train_csv.empty_reports", "studies without report text", empty_reports)
    return df


def validate_train_series_csv(path: Path, train_df: pd.DataFrame | None, report: SchemaReport) -> pd.DataFrame | None:
    if not path.exists():
        report.add("error", "train_series_csv.missing", f"train_series.csv not found at {path}")
        return None
    df = read_id_csv(path)
    _check_columns(df, TRAIN_SERIES_CSV_REQUIRED, "train_series_csv", report)
    if SERIES_ID not in df.columns or STUDY_ID not in df.columns:
        return df

    key_dupes = int(df.duplicated(subset=[STUDY_ID, SERIES_ID]).sum())
    if key_dupes:
        report.add(
            "error",
            "train_series_csv.duplicate_keys",
            "duplicated (StudyInstanceUID, SeriesInstanceUID) rows",
            key_dupes,
        )
    series_in_many_studies = df.groupby(SERIES_ID)[STUDY_ID].nunique()
    violating = series_in_many_studies[series_in_many_studies > 1]
    if len(violating):
        report.add(
            "error",
            "train_series_csv.series_to_many_studies",
            "SeriesInstanceUID mapped to more than one study (series->study must be many-to-one)",
            int(len(violating)),
        )
        report.tables["series_to_many_studies"] = violating.reset_index(name="n_studies")

    if "Anatomical_Plane" in df.columns:
        planes = df["Anatomical_Plane"].astype("string").str.strip().str.lower()
        unexpected = sorted(set(planes.dropna().unique()) - set(PLANES))
        if unexpected:
            report.add(
                "warning",
                "train_series_csv.unexpected_plane",
                f"Anatomical_Plane values outside {PLANES}: {unexpected}",
                len(unexpected),
            )
        report.summary["series_per_plane"] = planes.value_counts().to_dict()

    for flag in ("Fluid_Sensitive", "Fat_Suppression"):
        if flag in df.columns:
            values = pd.to_numeric(df[flag], errors="coerce")
            bad = int(values.isna().sum())
            if bad:
                report.add("warning", f"train_series_csv.{flag}_non_numeric", f"{flag} has non-numeric values", bad)
            report.summary[f"{flag}_positive"] = int((values == 1).sum())
    if {"Fluid_Sensitive", "Fat_Suppression"}.issubset(df.columns):
        same = int(
            (
                pd.to_numeric(df["Fluid_Sensitive"], errors="coerce")
                == pd.to_numeric(df["Fat_Suppression"], errors="coerce")
            ).sum()
        )
        if same == len(df):
            report.add(
                "warning",
                "train_series_csv.flags_identical",
                "Fluid_Sensitive and Fat_Suppression are identical for every row; treat them as one "
                "signal in series selection instead of two independent features",
                len(df),
            )

    if train_df is not None and STUDY_ID in train_df.columns:
        train_studies = set(train_df[STUDY_ID].dropna())
        series_studies = set(df[STUDY_ID].dropna())
        orphan_series_studies = sorted(series_studies - train_studies)
        studies_without_series = sorted(train_studies - series_studies)
        if orphan_series_studies:
            report.add(
                "error",
                "train_series_csv.orphan_studies",
                "train_series.csv references studies that are absent from train.csv",
                len(orphan_series_studies),
            )
            report.tables["orphan_series_studies"] = pd.DataFrame({STUDY_ID: orphan_series_studies})
        if studies_without_series:
            report.add(
                "error",
                "train_csv.studies_without_series",
                "train.csv studies with no series row",
                len(studies_without_series),
            )
            report.tables["studies_without_series"] = pd.DataFrame({STUDY_ID: studies_without_series})

    report.summary["train_series_rows"] = int(len(df))
    report.summary["train_series_unique_series"] = int(df[SERIES_ID].nunique())
    report.summary["train_series_unique_studies"] = int(df[STUDY_ID].nunique())
    return df


def validate_dicom_root(
    dicom_root: Path,
    series_df: pd.DataFrame | None,
    report: SchemaReport,
    sample_limit: int | None = None,
) -> pd.DataFrame:
    """Compare the on-disk study/series inventory with train_series.csv."""
    if not dicom_root.exists():
        report.add("error", "dicom_root.missing", f"DICOM root not found: {dicom_root}")
        return pd.DataFrame(columns=[STUDY_ID, SERIES_ID, "path", "n_files"])

    rows: list[dict] = []
    study_dirs = sorted(p for p in dicom_root.iterdir() if p.is_dir())
    if sample_limit is not None:
        study_dirs = study_dirs[:sample_limit]
    for study_dir in study_dirs:
        for series_dir in sorted(p for p in study_dir.iterdir() if p.is_dir()):
            n_files = sum(1 for _ in series_dir.iterdir())
            rows.append(
                {
                    STUDY_ID: study_dir.name,
                    SERIES_ID: series_dir.name,
                    "path": str(series_dir),
                    "n_files": n_files,
                }
            )
    disk = pd.DataFrame(rows, columns=[STUDY_ID, SERIES_ID, "path", "n_files"])
    report.summary["disk_studies"] = int(disk[STUDY_ID].nunique()) if len(disk) else 0
    report.summary["disk_series"] = int(len(disk))

    empty = disk[disk["n_files"] == 0]
    if len(empty):
        report.add("warning", "dicom_root.empty_series_dirs", "series directories without any file", int(len(empty)))
        report.tables["empty_series_dirs"] = empty

    if series_df is not None and SERIES_ID in series_df.columns and sample_limit is None:
        disk_keys = set(zip(disk[STUDY_ID], disk[SERIES_ID]))
        csv_keys = set(zip(series_df[STUDY_ID].astype(str), series_df[SERIES_ID].astype(str)))
        missing_on_disk = sorted(csv_keys - disk_keys)
        not_in_csv = sorted(disk_keys - csv_keys)
        if missing_on_disk:
            report.add(
                "error",
                "dicom_root.series_missing_on_disk",
                "series listed in train_series.csv without a directory on disk",
                len(missing_on_disk),
            )
            report.tables["series_missing_on_disk"] = pd.DataFrame(missing_on_disk, columns=[STUDY_ID, SERIES_ID])
        if not_in_csv:
            report.add(
                "warning",
                "dicom_root.series_not_in_csv",
                "series directories on disk that train_series.csv does not list",
                len(not_in_csv),
            )
            report.tables["series_not_in_csv"] = pd.DataFrame(not_in_csv, columns=[STUDY_ID, SERIES_ID])
    return disk


def validate_optional_export(key: str, path: Path | None, report: SchemaReport) -> pd.DataFrame | None:
    """Validate one optional label export against its documented expected schema."""
    spec = EXPECTED_EXPORT_SCHEMAS[key]
    if path is None:
        report.add("info", f"{key}.not_configured", f"{key} not configured. Expected fields: {spec['required']}")
        return None
    if not path.exists():
        report.add(
            "warning",
            f"{key}.missing",
            f"{key} configured as {path} but the file does not exist. "
            f"Expected columns {spec['required']} (optional: {spec['optional']}). {spec['note']}",
        )
        return None
    df = read_id_csv(path)
    missing = [c for c in spec["required"] if c not in df.columns]
    if missing:
        report.add(
            "error",
            f"{key}.missing_columns",
            f"{path.name} is missing required columns {missing}; present columns: {list(df.columns)[:20]}",
            len(missing),
        )
        return df
    if STUDY_ID in df.columns:
        if key == "labels_details_csv" and "target" in df.columns:
            dupes = int(df.duplicated(subset=[STUDY_ID, "target"]).sum())
            if dupes:
                report.add("error", f"{key}.duplicate_rows", "duplicated (study, target) rows", dupes)
            unknown_targets = sorted(set(df["target"].dropna().astype(str)) - set(TARGETS))
            if unknown_targets:
                report.add(
                    "error",
                    f"{key}.unknown_targets",
                    f"target values outside the project target list: {unknown_targets[:10]}",
                    len(unknown_targets),
                )
        else:
            dupes = int(df[STUDY_ID].duplicated().sum())
            if dupes:
                report.add("error", f"{key}.duplicate_studies", "duplicated StudyInstanceUID rows", dupes)
            validate_targets_header(df, key, report)
    report.summary[f"{key}_rows"] = int(len(df))
    return df


def coverage_report(
    train_df: pd.DataFrame | None,
    label_frame: pd.DataFrame | None,
    report: SchemaReport,
    label_name: str,
) -> None:
    """Check labelled coverage against the full study inventory."""
    if train_df is None or label_frame is None or STUDY_ID not in getattr(label_frame, "columns", []):
        return
    all_studies = set(train_df[STUDY_ID].dropna())
    labelled = set(label_frame[STUDY_ID].dropna()) & all_studies
    unknown_ids = set(label_frame[STUDY_ID].dropna()) - all_studies
    report.summary[f"{label_name}_studies_covered"] = len(labelled)
    report.summary[f"{label_name}_coverage_fraction"] = round(len(labelled) / max(len(all_studies), 1), 4)
    if unknown_ids:
        report.add(
            "error",
            f"{label_name}.unknown_studies",
            f"{label_name} references studies that are not in train.csv",
            len(unknown_ids),
        )
    if labelled and len(labelled) < len(all_studies):
        report.add(
            "info",
            f"{label_name}.partial_coverage",
            f"{label_name} covers {len(labelled)} of {len(all_studies)} studies "
            f"({100 * len(labelled) / max(len(all_studies), 1):.1f}%)",
            len(labelled),
        )


def validate_inputs(cfg: Config, dicom_sample_limit: int | None = None) -> SchemaReport:
    """Run every input check and return the collected findings."""
    report = SchemaReport()
    paths = cfg.paths
    report.summary["target_order"] = TARGETS

    train_df = validate_train_csv(Path(paths.train_csv), report)
    series_df = validate_train_series_csv(Path(paths.train_series_csv), train_df, report)
    validate_dicom_root(Path(paths.dicom_root), series_df, report, sample_limit=dicom_sample_limit)

    for key in EXPECTED_EXPORT_SCHEMAS:
        raw = paths.get(key)
        export = validate_optional_export(key, Path(raw) if raw else None, report)
        if export is not None:
            coverage_report(train_df, export, report, key)

    # The 12 target columns of train.csv are themselves a wide numeric export.
    if train_df is not None and set(TARGETS).issubset(train_df.columns):
        known = train_df[TARGETS].notna().any(axis=1)
        report.summary["train_csv_labelled_studies"] = int(known.sum())
        report.add(
            "info",
            "train_csv.label_coverage",
            f"train.csv carries numeric targets for {int(known.sum())} of {len(train_df)} studies",
            int(known.sum()),
        )
    return report
