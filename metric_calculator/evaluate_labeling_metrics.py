#!/usr/bin/env python3
"""Evaluate one knee labeling run from CSV files. Python 3.9+, standard library.

Notebook:
    from evaluate_labeling_metrics import evaluate_csv, print_metrics
    result = evaluate_csv(
        predictions_path="labels_predictions.csv",
        reference_path="train_labeled_58_reference.csv",
        statuses_path="labels_statuses.csv",  # optional; omit or set None
    )
    print_metrics(result)

Command line:
    python evaluate_labeling_metrics.py --predictions labels_predictions.csv \
        --reference train_labeled_58_reference.csv --statuses labels_statuses.csv \
        --json metrics.json

All CSVs are wide: StudyInstanceUID + the 12 exact label columns. The reference
may have additional columns and unlabeled studies, as in original train.csv.
Never use an LLM-filled train_v1.csv as GT. The filtered labels_review.csv is
not a substitute for labels_predictions.csv or labels_statuses.csv.

Evaluation covers ALL known GT cells in the reference. Missing prediction rows
stay in coverage's denominator. Predictions without GT are ignored and counted.
Empty GT cells are excluded. Joins use UID, never row order. Duplicate IDs,
missing columns, nonbinary scores and contradictory statuses raise errors.

Missing cells must be empty. Numeric 0, 1, 0.0 and 1.0 are accepted. Statuses
are positive/negative/uncertain/not_mentioned or empty. UTF-8 BOM is supported.
Use delimiter=";" (or --delimiter ";") for semicolon-delimited files.

Unanswered predictions are excluded from classification metrics, but not from
coverage. Borderline negatives remain negative. Binary ROC-AUC equals balanced
accuracy with half credit for ties. Macro averages weight defined label metrics
equally; micro pools all evaluated pairs. Undefined values are None/NA.

Without a status file, uncertain/unmentioned counts are unavailable (None).
Blank statuses leave the reason for missing predictions unknown. Processing
failures and borderline counts cannot be determined from these wide CSVs and
are returned as None. They are not assumed to be zero.
"""
import argparse

from collections import Counter

import csv

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

def read_wide_csv(path, *, statuses=False, delimiter=","):
    """Read UID + 12 label columns; reject duplicate IDs/headers and invalid cells."""
    path = Path(path)
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate column names.")
        missing = set(("StudyInstanceUID", *LABELS)) - set(fields)
        if missing:
            raise ValueError(f"{path}: missing columns: {sorted(missing)}. Use the full wide CSV.")
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{line}: malformed CSV row.")
            uid = row["StudyInstanceUID"].strip()
            if not uid or uid in result:
                raise ValueError(f"{path}:{line}: empty or duplicate StudyInstanceUID: {uid!r}")
            labels = {}
            for label in LABELS:
                value = row[label].strip()
                if statuses:
                    if value not in ("", "positive", "negative", "uncertain", "not_mentioned"):
                        raise ValueError(f"{path}:{line}/{label}: unknown status {value!r}.")
                    labels[label] = value or None
                elif value == "":
                    labels[label] = None
                else:
                    try:
                        number = float(value)
                    except ValueError:
                        raise ValueError(f"{path}:{line}/{label}: expected 0, 1 or an empty cell.") from None
                    if number not in (0, 1):
                        raise ValueError(f"{path}:{line}/{label}: expected 0/1, received {value!r}.")
                    labels[label] = int(number)
            result[uid] = labels
    if not result:
        raise ValueError(f"{path}: no data rows.")
    return result


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
    statuses = Counter(r["status"] for r in rows)
    has_statuses = bool(rows) and rows[0]["statuses_provided"]
    return {
        "n_reference": len(rows), "n_reference_positive": n_pos,
        "n_reference_negative": n_neg, "n_evaluated": len(evaluated),
        "n_positive": tp + fp, "n_negative": tn + fn,
        "n_missing": len(rows) - len(evaluated),
        "n_uncertain": statuses["uncertain"] if has_statuses else None,
        "n_not_mentioned": statuses["not_mentioned"] if has_statuses else None,
        # Wide CSVs do not contain processing_status or the borderline basis.
        "n_unprocessed": None, "n_borderline": None,
        "n_missing_reason_unknown": sum(r["predicted_label"] is None and r["status"] is None for r in rows),
        "n_missing_prediction_rows": sum(not r["prediction_row_present"] for r in rows),
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


def evaluate_csv(predictions_path, reference_path, statuses_path=None, *, delimiter=","):
    """Evaluate all known GT cells in reference_path, joining exclusively by UID.

    reference_path may be the 58-study reference or the original train.csv.
    Blank GT cells are excluded. Missing prediction rows remain in coverage's
    denominator. Predictions without any GT are ignored and counted explicitly.
    Optional statuses must have exactly the prediction file's study IDs.
    """
    predictions = read_wide_csv(predictions_path, delimiter=delimiter)
    reference = read_wide_csv(reference_path, delimiter=delimiter)
    statuses = None if statuses_path is None else read_wide_csv(statuses_path, statuses=True, delimiter=delimiter)
    if statuses is not None:
        if statuses.keys() != predictions.keys():
            raise ValueError("Status and prediction CSVs must contain exactly the same study IDs.")
        for uid in predictions:
            for label in LABELS:
                status, pred = statuses[uid][label], predictions[uid][label]
                expected = {"positive": 1, "negative": 0, "uncertain": None, "not_mentioned": None}
                if status is not None and pred != expected[status]:
                    raise ValueError(f"Inconsistent status/prediction for {uid}/{label}.")
                if status is None and pred is not None:
                    raise ValueError(f"Missing status for a binary prediction: {uid}/{label}.")
    gt_ids = {uid for uid, labels in reference.items() if any(v is not None for v in labels.values())}
    if not gt_ids:
        raise ValueError("The reference CSV contains no binary GT labels.")
    if not gt_ids.intersection(predictions):
        raise ValueError("No prediction studies overlap with the GT-labeled reference studies.")
    rows = []
    for uid in sorted(gt_ids):
        for label in LABELS:
            gt = reference[uid][label]
            if gt is None:
                continue
            rows.append({
                "StudyInstanceUID": uid, "label": label, "reference_label": gt,
                "predicted_label": predictions.get(uid, {}).get(label),
                "status": statuses.get(uid, {}).get(label) if statuses is not None else None,
                "statuses_provided": statuses is not None,
                "prediction_row_present": uid in predictions,
            })
    return {
        "predictions_file": str(predictions_path), "reference_file": str(reference_path),
        "statuses_file": str(statuses_path) if statuses_path is not None else None,
        "n_studies": len(gt_ids), "n_possible_decisions": len(rows),
        "n_prediction_studies": len(predictions), "n_reference_studies": len(reference),
        "n_missing_prediction_studies": len(gt_ids - predictions.keys()),
        "n_prediction_studies_without_gt": len(predictions.keys() - gt_ids),
        "n_reference_studies_without_gt": len(reference) - len(gt_ids),
        "policy": "Evaluate all known GT cells; retain missing predictions in coverage; never fill missing with zero.",
        "auc_note": "Binary-score ROC-AUC, with half credit for ties.",
        "metadata_note": "Wide CSVs do not establish processing failures or borderline counts. These are null, not zero. Without statuses, uncertainty types are also null.",
        **summarize(rows),
    }


def print_metrics(result):
    """Print aggregate and per-label metrics for one labeling run."""
    print(f"Studies: {result['n_studies']} | Possible decisions: {result['n_possible_decisions']}")
    print("ROC-AUC uses binary scores. NA = undefined. Missing predictions are excluded.")
    print(f"GT studies without a prediction row: {result['n_missing_prediction_studies']}")
    print(f"Prediction studies without GT (ignored): {result['n_prediction_studies_without_gt']}")
    print("\nCounts:")
    counts = (
        "n_reference", "n_reference_positive", "n_reference_negative",
        "n_evaluated", "n_positive", "n_negative", "n_uncertain",
        "n_not_mentioned", "n_missing", "n_missing_reason_unknown", "n_missing_prediction_rows",
        "tp", "fp", "tn", "fn", "n_correct", "n_incorrect",
    )
    display_table(["Metric", "Value"], [[m, "NA" if result["micro"][m] is None else result["micro"][m]] for m in counts])
    print("\nAggregate metrics:")
    display_table(
        ["Metric", "Micro", "Macro", "Defined macro labels"],
        [[m, format_value(result["micro"][m], m != "roc_auc"),
          format_value(result["macro"][m], m != "roc_auc"),
          result["macro_n_defined_labels"][m]] for m in RATE_METRICS],
    )
    print("\nPer-label counts (N = evaluated binary decisions):")
    fields = ("n_evaluated", "n_missing", "n_uncertain", "n_not_mentioned", "tp", "fp", "tn", "fn")
    display_table(
        ["Label", "N", "Missing", "Uncertain", "Unmentioned", "TP", "FP", "TN", "FN"],
        [[label, *("NA" if result["per_label"][label][m] is None else result["per_label"][label][m] for m in fields)] for label in LABELS],
    )
    print("\nPer-label metrics:")
    fields = ("coverage", "accuracy", "precision", "recall", "specificity", "f1", "roc_auc")
    display_table(
        ["Label", "Coverage", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"],
        [[label, *(format_value(result["per_label"][label][m], m != "roc_auc")
                    for m in fields)] for label in LABELS],
    )
    print("\nCoverage denominator: all GT labels, including unanswered/unprocessed labels.")
    print("Recall denominator: evaluated GT positives; reference_positive_detection_rate: all GT positives.")
    print("Macro: equal weight per defined label. Micro: all evaluated study-label pairs pooled.")
    print("Status counts are NA without a status CSV; blank predictions are never assumed negative.")
    print("Processing-error and borderline counts cannot be recovered from these wide CSVs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # labels_predictions.csv
    parser.add_argument("--predictions", required=True, type=Path, help="labels_predictions.csv")
    # train_labeled_58_reference.csv
    parser.add_argument("--reference", required=True, type=Path, help="GT reference CSV or original train.csv")
    # labels_statuses.csv
    parser.add_argument("--statuses", type=Path, help="Optional labels_statuses.csv")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter, default comma")
    parser.add_argument("--json", type=Path, help="Optional JSON output with all metrics")
    args = parser.parse_args()
    try:
        inputs = [args.predictions, args.reference] + ([args.statuses] if args.statuses else [])
        if args.json and args.json.resolve() in {p.resolve() for p in inputs}:
            raise ValueError("The output JSON path must not overwrite any input CSV.")
        result = evaluate_csv(args.predictions, args.reference, args.statuses, delimiter=args.delimiter)
        print_metrics(result)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
            print(f"\nSaved: {args.json}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
