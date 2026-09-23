#!/usr/bin/env python3
"""Compare two GT labeling HTML reports. Python 3.9+, standard library only.

Command line:
    python compare_labeling_metrics.py --old GT_label_predictions.html \
        --new labels_review.html --json metrics_comparison.json

Notebook:
    from compare_labeling_metrics import compare_reports, print_comparison
    result = compare_reports("GT_label_predictions.html", "labels_review.html")
    print_comparison(result)

Inputs are the FULL HTML exports from label_train_reports.ipynb, with GT
reference labels. The filtered labels_review.csv is NOT a valid substitute.
The embedded JSON is read without executing any HTML or JavaScript. Exported
summary metrics are not used: everything is calculated from individual rows.

Policies match the comparison in the conversation:
  * positive=1, negative=0, including negative/borderline;
  * uncertain/not_mentioned/unprocessed are excluded from classification
    metrics, but remain in the coverage denominator;
  * all GT labels must be 0/1, and both reports must contain the same studies,
    source report text, and reference labels;
  * scores must be binary. Binary ROC-AUC = (recall + specificity) / 2,
    with half credit for tied positive-negative score pairs;
  * macro averages give each defined label metric equal weight;
    micro metrics pool all evaluated study-label pairs;
  * undefined ratios are None, displayed as NA, never silently set to zero.

This is report-derived vs image-derived GT evaluation, not a leaderboard
score. AUC on binary decisions does not measure continuous score ranking.
"""

import argparse
from collections import Counter
from html.parser import HTMLParser
import json
from pathlib import Path


LABELS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)
RATE_METRICS = (
    "coverage", "positive_coverage", "negative_coverage", "accuracy",
    "precision", "recall", "specificity", "f1", "balanced_accuracy",
    "roc_auc", "reference_positive_detection_rate",
)


class DatasetParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.active = False
        self.parts = []
        self.matches = 0

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("id") == "dataset":
            self.active = True
            self.matches += 1

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script":
            self.active = False


def load_report(path):
    """Load and validate one complete HTML report; return source text and rows."""
    path = Path(path)
    parser = DatasetParser()
    parser.feed(path.read_text(encoding="utf-8-sig"))
    parser.close()
    if parser.matches != 1:
        raise ValueError(f"{path}: expected exactly one script with id='dataset'.")
    data = json.loads("".join(parser.parts))
    if len(data["labels"]) != len(LABELS) or set(data["labels"]) != set(LABELS):
        raise ValueError(f"{path}: expected the 12 knee labels.")
    rows, reports = {}, {}
    for case in data["cases"]:
        uid = case["uid"]
        if not isinstance(uid, str) or not uid or uid in reports:
            raise ValueError(f"{path}: invalid or duplicate study UID: {uid!r}")
        if not isinstance(case["report"], str):
            raise ValueError(f"{path}: report text must be a string for {uid}.")
        reports[uid] = case["report"]
        seen = set()
        for row in case["details"]:
            label = row["label"]
            if label not in LABELS or label in seen:
                raise ValueError(f"{path}: invalid or duplicate label for {uid}: {label}")
            seen.add(label)
            if row["StudyInstanceUID"] != uid:
                raise ValueError(f"{path}: inconsistent UID in label details.")
            gt, pred = row["reference_label"], row["predicted_label"]
            if type(gt) not in (int, float) or gt not in (0, 1):
                raise ValueError(f"{path}: {uid}/{label} has no binary GT reference.")
            if pred is not None and (
                type(pred) not in (int, float) or pred not in (0, 1)
            ):
                raise ValueError(f"{path}: this script expects 0/1/null, not soft scores.")
            processing = row["processing_status"]
            if processing == "success":
                status = row["status"]
                expected = {"positive": 1, "negative": 0,
                            "uncertain": None, "not_mentioned": None}
                if status not in expected or pred != expected[status]:
                    raise ValueError(f"{path}: inconsistent status/prediction: {uid}/{label}")
            elif pred is not None:
                raise ValueError(f"{path}: an unprocessed label must have a null prediction.")
            rows[(uid, label)] = row
        if seen != set(LABELS):
            raise ValueError(f"{path}: missing labels for {uid}.")
    if not reports:
        raise ValueError(f"{path}: no studies found.")
    return {"path": str(path), "reports": reports, "rows": rows}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def calculate_metrics(rows):
    """Metrics for any list of GT rows, retaining abstentions in coverage."""
    rows = list(rows)
    evaluated = [r for r in rows if r["predicted_label"] is not None]
    confusion = Counter(
        (int(r["reference_label"]), int(r["predicted_label"])) for r in evaluated
    )
    tp, fp = confusion[(1, 1)], confusion[(0, 1)]
    tn, fn = confusion[(0, 0)], confusion[(1, 0)]
    n_pos = sum(r["reference_label"] == 1 for r in rows)
    n_neg = len(rows) - n_pos
    recall, specificity = ratio(tp, tp + fn), ratio(tn, tn + fp)
    balanced = None if recall is None or specificity is None else (recall + specificity) / 2
    # Pairwise ROC-AUC: TP beats TN; TP/FP and FN/TN pairs are tied.
    auc = ratio(tp * tn + 0.5 * (tp * fp + fn * tn),
                (tp + fn) * (tn + fp))
    statuses = Counter(r["status"] for r in rows if r["processing_status"] == "success")
    return {
        "n_reference": len(rows), "n_reference_positive": n_pos,
        "n_reference_negative": n_neg, "n_evaluated": len(evaluated),
        "n_positive": tp + fp, "n_negative": tn + fn,
        "n_uncertain": statuses["uncertain"],
        "n_not_mentioned": statuses["not_mentioned"],
        "n_unprocessed": sum(r["processing_status"] != "success" for r in rows),
        "n_borderline": sum(r["processing_status"] == "success" and
                            r.get("basis") == "borderline" for r in rows),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_correct": tp + tn, "n_incorrect": fp + fn,
        "coverage": ratio(len(evaluated), len(rows)),
        "positive_coverage": ratio(tp + fn, n_pos),
        "negative_coverage": ratio(tn + fp, n_neg),
        "accuracy": ratio(tp + tn, len(evaluated)),
        "precision": ratio(tp, tp + fp), "recall": recall,
        "specificity": specificity, "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": balanced, "roc_auc": auc,
        "reference_positive_detection_rate": ratio(tp, n_pos),
    }


def summarize(rows):
    rows = list(rows)
    per_label = {label: calculate_metrics(r for r in rows if r["label"] == label)
                 for label in LABELS}
    macro, n_defined = {}, {}
    for metric in RATE_METRICS:
        values = [m[metric] for m in per_label.values() if m[metric] is not None]
        macro[metric] = ratio(sum(values), len(values))
        n_defined[metric] = len(values)
    return {"micro": calculate_metrics(rows), "macro": macro,
            "macro_n_defined_labels": n_defined, "per_label": per_label}


def outcome(row):
    if row["processing_status"] != "success":
        return "unprocessed"
    if row["predicted_label"] is None:
        return "abstained"
    return "correct" if row["predicted_label"] == row["reference_label"] else "incorrect"


def compare_reports(old_path, new_path):
    """Return all metrics, paired comparisons and changed decisions as a dict."""
    old, new = load_report(old_path), load_report(new_path)
    x, y = old["rows"], new["rows"]
    if x.keys() != y.keys():
        raise ValueError("The two reports contain different study/label pairs.")
    if old["reports"] != new["reports"]:
        raise ValueError("Source report text differs between runs.")
    if any(x[k]["reference_label"] != y[k]["reference_label"] for k in x):
        raise ValueError("GT reference labels differ between runs.")
    keys = sorted(x)
    common = [k for k in keys if x[k]["predicted_label"] is not None
              and y[k]["predicted_label"] is not None]
    transitions = Counter((outcome(x[k]), outcome(y[k])) for k in keys)
    status_changes = Counter((x[k]["status"], y[k]["status"]) for k in keys)
    changed = [
        {"StudyInstanceUID": k[0], "label": k[1], "gt": x[k]["reference_label"],
         "old_prediction": x[k]["predicted_label"], "new_prediction": y[k]["predicted_label"],
         "old_status": x[k]["status"], "new_status": y[k]["status"],
         "old_outcome": outcome(x[k]), "new_outcome": outcome(y[k])}
        for k in keys if x[k]["predicted_label"] != y[k]["predicted_label"]
        or x[k]["status"] != y[k]["status"]
    ]
    return {
        "old_file": old["path"], "new_file": new["path"],
        "n_studies": len(old["reports"]), "n_possible_decisions": len(keys),
        "policy": "Binary decisions only; borderline stays negative; missing is not zero.",
        "auc_note": "Binary-score ROC-AUC, with half credit for ties. Not continuous-score AUC.",
        "own": {"old": summarize(x.values()), "new": summarize(y.values())},
        "common": {"n_decisions": len(common),
                   "old": summarize(x[k] for k in common),
                   "new": summarize(y[k] for k in common)},
        "transitions": [{"old": a, "new": b, "n": n}
                        for (a, b), n in sorted(transitions.items())],
        "status_transitions": [{"old": a, "new": b, "n": n}
                               for (a, b), n in status_changes.items()],
        "n_changed_predictions": sum(x[k]["predicted_label"] != y[k]["predicted_label"] for k in keys),
        "n_changed_statuses": sum(x[k]["status"] != y[k]["status"] for k in keys),
        "changed_decisions": changed,
    }


def display_table(headers, rows):
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    def line(row):
        return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
    print(line(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(line(row))


def format_value(value, percent=False):
    if value is None:
        return "NA"
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def print_comparison(result):
    """Print the same summary, per-label F1/AUC and paired metrics as in chat."""
    old, new = result["own"]["old"], result["own"]["new"]
    print(f"Studies: {result['n_studies']} | Possible decisions: {result['n_possible_decisions']}")
    print("Same study IDs, source text and GT verified. Missing predictions are excluded.")
    print("ROC-AUC uses binary scores (equals balanced accuracy). NA = undefined.\n")
    counts = ("n_evaluated", "n_positive", "n_negative", "n_uncertain", "n_not_mentioned",
              "n_unprocessed", "n_borderline", "tp", "fp", "tn", "fn", "n_correct", "n_incorrect")
    table = [[metric, old["micro"][metric], new["micro"][metric]] for metric in counts]
    for metric in RATE_METRICS:
        table.append([metric + " (micro)",
                      format_value(old["micro"][metric], metric != "roc_auc"),
                      format_value(new["micro"][metric], metric != "roc_auc")])
    for metric in ("f1", "roc_auc"):
        table.append([metric + " (macro)",
                      format_value(old["macro"][metric], metric == "f1"),
                      format_value(new["macro"][metric], metric == "f1")])
        table.append([metric + " macro: defined labels",
                      old["macro_n_defined_labels"][metric], new["macro_n_defined_labels"][metric]])
    display_table(["Metric", "Old", "New"], table)
    print("\nPer-label results on each run's evaluated subset:")
    table = []
    for label in LABELS:
        a, b = old["per_label"][label], new["per_label"][label]
        table.append([label, a["n_evaluated"], b["n_evaluated"],
                      format_value(a["f1"], True), format_value(b["f1"], True),
                      format_value(a["roc_auc"]), format_value(b["roc_auc"])])
    display_table(["Label", "Old N", "New N", "Old F1", "New F1", "Old AUC", "New AUC"], table)
    print(f"\nSame binary-decision subset: {result['common']['n_decisions']} pairs")
    table = []
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        table.append([metric + " (micro)",
                      *(format_value(result["common"][run]["micro"][metric], metric != "roc_auc")
                        for run in ("old", "new"))])
    table.append(["roc_auc (macro)", *(format_value(result["common"][run]["macro"]["roc_auc"])
                                      for run in ("old", "new"))])
    display_table(["Metric", "Old", "New"], table)
    print("\nDecision transitions (GT comparison):")
    display_table(["Old", "New", "Count"],
                  [[t["old"], t["new"], t["n"]] for t in result["transitions"]])
    print(f"\nChanged predictions: {result['n_changed_predictions']}; "
          f"changed statuses: {result['n_changed_statuses']}")
    print("Coverage in the main table uses ALL reference labels as denominator.")
    print("Recall uses ONLY evaluated reference positives; positive detection rate uses ALL GT positives.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", required=True, type=Path, help="Previous full GT HTML report")
    parser.add_argument("--new", required=True, type=Path, help="New full GT HTML report")
    parser.add_argument("--json", type=Path, help="Optional output JSON with all metrics and changed decisions")
    args = parser.parse_args()
    try:
        if args.json and args.json.resolve() in {args.old.resolve(), args.new.resolve()}:
            raise ValueError("The output JSON path must not overwrite either input HTML.")
        result = compare_reports(args.old, args.new)
        print_comparison(result)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
            print(f"\nSaved: {args.json}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
