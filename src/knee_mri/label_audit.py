"""Targeted audit of the LLM report labels.

    python -m knee_mri.cli label-audit sample       # draw ~100 reports, write the audit sheet + HTML
    python -m knee_mri.cli label-audit summarize    # after review: error rates per target and basis

The sample is stratified on purpose (Astra, 2nd round): a random part estimates the
overall error rate, the other parts over-sample where the labels are thin or suspect:

  * random         - uniform over the non-reference training studies;
  * synovitis      - Synovitis was decided, or synovitis is mentioned in the report;
  * mcl            - MCL positive / below threshold, or the MCL is mentioned;
  * many_empty     - at least `many_empty_min` of the 12 cells are not_mentioned/uncertain.

Groups are drawn in that order without replacement, so a study appears once. Only
the random group is an unbiased sample; the per-basis rates in the summary say which
group they come from.

The reviewer fills, per (study, target) row of `label_audit.csv`:
  audit_label  1 | 0 | undecidable     (what the REPORT supports under the competition definition)
  error_type   ok | llm_misread | definition_misapplied | not_decidable_from_report
  note         free text

`llm_second_opinion` is a helper column for a second LLM pass. It is never the reference.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .constants import STUDY_ID, TARGETS
from .schema import read_id_csv
from .utils import LOG, atomic_write_dataframe, atomic_write_json

ERROR_TYPES = ["ok", "llm_misread", "definition_misapplied", "not_decidable_from_report"]
AUDIT_LABELS = ["1", "0", "undecidable"]
EMPTY_STATUSES = {"not_mentioned", "uncertain"}

# Multilingual mention patterns (es / en / de / nl / fr / it / pt). They only steer the
# sampling towards reports that talk about the finding; they never set a label.
MENTION_PATTERNS = {
    "Synovitis": r"sinovit|synovit|synoviit|sinovial|synovial|synovium|sinovio",
    "MCL": r"\bmcl\b|\blli\b|colateral (?:medial|interno|tibial)|medial collateral|collat[ée]ral (?:m[ée]dial|interne|tibial)"
    r"|innenband|mediale collateral|collaterale mediale|mediales seitenband",
}


def _load_inputs(cfg: Config, details_csv: str | Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    details_path = Path(details_csv) if details_csv else Path(cfg.paths.work_dir) / "labels" / "labels_details_all.csv"
    if not details_path.exists():
        raise FileNotFoundError(f"{details_path} not found; run `merge-label-details` first")
    details = read_id_csv(details_path)
    reports = read_id_csv(cfg.paths.train_csv)
    if "Report" not in reports.columns:
        raise ValueError(f"{cfg.paths.train_csv} has no Report column")
    return details, reports[[STUDY_ID, "Report"]]


def draw_sample(
    details: pd.DataFrame,
    reports: pd.DataFrame,
    group_sizes: dict[str, int],
    seed: int,
    many_empty_min: int = 8,
) -> pd.DataFrame:
    """Return one row per sampled study with its `audit_group`."""
    rng = np.random.default_rng(seed)
    llm = details[details["basis"] != "reference"]
    wide_status = llm.pivot(index=STUDY_ID, columns="target", values="status")
    wide_basis = llm.pivot(index=STUDY_ID, columns="target", values="basis")
    text = reports.set_index(STUDY_ID)["Report"].fillna("").str.lower().reindex(wide_status.index).fillna("")

    def mentions(target: str) -> pd.Series:
        return text.str.contains(MENTION_PATTERNS[target], regex=True)

    candidates = {
        "random": pd.Series(True, index=wide_status.index),
        "synovitis": (wide_status["Synovitis"] != "not_mentioned") | mentions("Synovitis"),
        "mcl": (wide_status["MCL"] == "positive") | (wide_basis["MCL"] == "below_threshold") | mentions("MCL"),
        "many_empty": wide_status.isin(EMPTY_STATUSES).sum(axis=1) >= many_empty_min,
    }
    unknown = set(group_sizes) - set(candidates)
    if unknown:
        raise ValueError(f"unknown audit groups {sorted(unknown)}; known: {list(candidates)}")

    picked: list[tuple[str, str]] = []
    taken: set[str] = set()
    for group, size in group_sizes.items():
        pool = sorted(s for s in candidates[group][candidates[group]].index if s not in taken)
        if len(pool) < size:
            LOG.warning("Audit group %s: only %d candidates for %d requested", group, len(pool), size)
        chosen = rng.choice(pool, size=min(size, len(pool)), replace=False) if pool else []
        for sid in sorted(chosen):
            picked.append((sid, group))
            taken.add(sid)
        LOG.info("Audit group %-10s %3d studies (from %d candidates)", group, len(chosen), len(pool))
    return pd.DataFrame(picked, columns=[STUDY_ID, "audit_group"])


def build_audit_sheet(sample: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    rows = sample.merge(details, on=STUDY_ID, how="left")
    order = {t: i for i, t in enumerate(TARGETS)}
    rows = rows.assign(_s=rows["audit_group"].map({g: i for i, g in enumerate(sample["audit_group"].unique())}))
    rows = rows.assign(_t=rows["target"].map(order)).sort_values(["_s", STUDY_ID, "_t"]).drop(columns=["_s", "_t"])
    keep = [STUDY_ID, "audit_group", "target", "status", "basis", "evidence", "reason"]
    sheet = rows[[c for c in keep if c in rows.columns]].copy()
    for column in ("llm_second_opinion", "audit_label", "error_type", "note"):
        sheet[column] = ""
    return sheet.reset_index(drop=True)


def _highlight(report: str, snippets: list[str]) -> str:
    """HTML-escape the report and mark every evidence snippet found verbatim."""
    escaped = html.escape(report)
    for snippet in sorted({s.strip() for s in snippets if isinstance(s, str) and len(s.strip()) >= 6}, key=len, reverse=True):
        pattern = re.escape(html.escape(snippet))
        escaped = re.sub(pattern, lambda m: f"<mark>{m.group(0)}</mark>", escaped, flags=re.IGNORECASE)
    return escaped.replace("\n", "<br>")


def _evidence_snippets(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)  # the labelling export stores a JSON list of quotes
        if isinstance(parsed, list):
            return [str(p) for p in parsed]
    except ValueError:
        pass
    parts = re.split(r'"\s*,\s*"|\s\|\s|\n', value.strip().strip("[]"))
    return [p.strip().strip('"').strip("'") for p in parts if p.strip()]


def write_audit_html(sheet: pd.DataFrame, reports: pd.DataFrame, path: Path) -> Path:
    text = reports.set_index(STUDY_ID)["Report"].fillna("")
    cards = []
    for (sid, group), rows in sheet.groupby([STUDY_ID, "audit_group"], sort=False):
        snippets = [s for v in rows.get("evidence", pd.Series(dtype=object)) for s in _evidence_snippets(v)]
        body = "".join(
            "<tr class='{cls}'><td>{t}</td><td>{st}</td><td>{b}</td><td>{e}</td><td>{r}</td></tr>".format(
                cls="empty" if r.status in EMPTY_STATUSES else r.status,
                t=html.escape(r.target),
                st=html.escape(str(r.status)),
                b=html.escape(str(r.basis)),
                e=html.escape(str(r.evidence) if pd.notna(r.evidence) else ""),
                r=html.escape(str(r.reason) if pd.notna(r.reason) else ""),
            )
            for r in rows.itertuples()
        )
        cards.append(
            f"<section><h2>{html.escape(sid[-16:])} <small>{html.escape(group)} · {html.escape(sid)}</small></h2>"
            f"<div class='report'>{_highlight(text.get(sid, ''), snippets)}</div>"
            "<table><tr><th>Target</th><th>Status</th><th>Basis</th><th>Evidence</th><th>Reason</th></tr>"
            f"{body}</table></section>"
        )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Label audit</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#666; --line:#ddd; --mark:#ffe58a; --pos:#e6f4ea; --neg:#fff; --empty:#fdecea; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161618; --fg:#e8e8ea; --muted:#9a9a9f; --line:#333; --mark:#6b5a12; --pos:#173323; --neg:#161618; --empty:#3a1f1c; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0 auto; max-width:1100px; padding:16px; }}
section {{ border-top:1px solid var(--line); padding:12px 0 20px; }}
h2 small {{ color:var(--muted); font-weight:normal; font-size:12px; }}
.report {{ white-space:normal; background:color-mix(in srgb, var(--fg) 4%, transparent); padding:10px; border-radius:6px; }}
mark {{ background:var(--mark); color:inherit; }}
table {{ border-collapse:collapse; width:100%; margin-top:8px; font-size:13px; }}
th, td {{ border-bottom:1px solid var(--line); padding:4px 6px; text-align:left; vertical-align:top; }}
tr.positive {{ background:var(--pos); }} tr.empty {{ background:var(--empty); }}
</style></head><body>
<h1>Label audit — {sheet[STUDY_ID].nunique()} reports</h1>
<p>Fill <code>label_audit.csv</code>: <b>audit_label</b> ∈ {{{", ".join(AUDIT_LABELS)}}},
<b>error_type</b> ∈ {{{", ".join(ERROR_TYPES)}}}. Judge what the report supports under the competition
definition; highlighted text is the LLM's quoted evidence. Red rows are not_mentioned / uncertain.</p>
{''.join(cards)}
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


def sample_audit(
    cfg: Config,
    out_dir: str | Path | None = None,
    details_csv: str | Path | None = None,
    group_sizes: dict[str, int] | None = None,
    seed: int | None = None,
    force: bool = False,
) -> Path:
    out_dir = Path(out_dir) if out_dir else Path(cfg.paths.work_dir) / "audit"
    sheet_path = out_dir / "label_audit.csv"
    if sheet_path.exists() and not force:
        raise FileExistsError(f"{sheet_path} exists and may already hold review work; pass --force to redraw")
    group_sizes = group_sizes or {"random": 40, "synovitis": 20, "mcl": 20, "many_empty": 20}
    seed = int(cfg.seed if seed is None else seed)

    details, reports = _load_inputs(cfg, details_csv)
    sample = draw_sample(details, reports, group_sizes, seed)
    sheet = build_audit_sheet(sample, details)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(sheet_path, index=False, encoding="utf-8-sig")  # utf-8-sig: opens cleanly in Excel
    write_audit_html(sheet, reports, out_dir / "label_audit.html")
    atomic_write_json(
        out_dir / "label_audit_meta.json",
        {"seed": seed, "group_sizes": group_sizes, "n_studies": int(sample[STUDY_ID].nunique()), "n_rows": len(sheet)},
    )
    LOG.info("Audit sheet: %s (%d rows); review page: %s", sheet_path, len(sheet), out_dir / "label_audit.html")
    return sheet_path


def summarize_audit(cfg: Config, sheet_path: str | Path | None = None) -> pd.DataFrame:
    """Error rates per target and basis from a filled audit sheet."""
    sheet_path = Path(sheet_path) if sheet_path else Path(cfg.paths.work_dir) / "audit" / "label_audit.csv"
    sheet = pd.read_csv(sheet_path, dtype=str, encoding="utf-8-sig").fillna("")
    done = sheet[sheet["audit_label"].str.strip() != ""].copy()
    if done.empty:
        raise ValueError(f"{sheet_path}: no row has an audit_label yet")
    bad_label = sorted(set(done["audit_label"].str.strip()) - set(AUDIT_LABELS))
    bad_type = sorted(set(done["error_type"].str.strip()) - set(ERROR_TYPES) - {""})
    if bad_label or bad_type:
        raise ValueError(f"unexpected audit values: audit_label {bad_label}, error_type {bad_type}")
    done["audit_label"] = done["audit_label"].str.strip()
    done["error_type"] = done["error_type"].str.strip().replace("", "ok")
    done["audit_pos"] = (done["audit_label"] == "1").astype(float)
    done.loc[done["audit_label"] == "undecidable", "audit_pos"] = np.nan

    def rates(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        grouped = frame.groupby(keys, dropna=False)
        out = grouped.size().rename("n").to_frame()
        out["p_audit_positive"] = grouped["audit_pos"].mean()
        out["n_undecidable"] = grouped["audit_label"].apply(lambda s: int((s == "undecidable").sum()))
        for error in ERROR_TYPES:
            out[f"rate_{error}"] = grouped["error_type"].apply(lambda s, e=error: float((s == e).mean()))
        return out.reset_index()

    by_basis = rates(done, ["basis"]).assign(level="basis")
    by_target_basis = rates(done, ["target", "basis"]).assign(level="target_basis")
    by_group = rates(done, ["audit_group", "basis"]).assign(level="group_basis")
    summary = pd.concat([by_basis, by_target_basis, by_group], ignore_index=True)
    out = sheet_path.with_name("label_audit_summary.csv")
    atomic_write_dataframe(summary, out)
    LOG.info("Audited %d/%d rows. Per basis:\n%s", len(done), len(sheet), by_basis.to_string(index=False))
    LOG.info("Summary written to %s", out)
    return summary
