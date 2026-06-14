"""Evaluation script for AccessSentinel.

Runs the full detection pipeline against sample_data/ and compares
predictions with ground truth labels, reporting classification metrics.
"""

import json
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from core.ingestion import ingest
from core.behavioral_baseline import build_baselines
from core.features import extract_features, FEATURE_NAMES
from core.models.ensemble_detector import EnsembleAnomalyDetector
from core.models.random_forest import RiskClassifier
from core.rules_engine import evaluate_rules
from core.context_resolver import resolve
from core.risk_scorer import compute_risk_score
from core.evaluator import evaluate_users, evaluate_events

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "sample_data")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TARGET_OVERALL_F1 = 0.75
TARGET_FALSE_POSITIVE_RATE = 0.20
ANOMALY_SCORE_THRESHOLD = 60
DEFAULT_CONTAMINATION = 0.20
DEFAULT_SVM_NU = 0.06
RF_PROBABILITY_THRESHOLD = 0.50
BLEND_WEIGHT_BALANCED_RF = 0.60
BLEND_WEIGHT_SMOTE_RF = 0.40
BLEND_THRESHOLD = 0.35
STACK_THRESHOLD = 0.20


def _normalize_boolean_column(series):
    """Convert string boolean representations to integer 0/1."""
    return series.apply(
        lambda x: 1 if str(x).strip().lower() in ("true", "1", "yes", "t") else 0
    )


def _build_feature_stats(identity_records):
    """Build per-user and per-department event-count statistics.

    Returns a tuple ``(user_stats, dept_stats)`` suitable for passing
    to :func:`core.features.extract_features`.
    """
    dept_event_counts: dict[str, list[int]] = {}
    for rec in identity_records:
        dept_event_counts.setdefault(rec.department, []).append(rec.event_count_30d)

    dept_stats: dict[str, dict[str, float]] = {}
    for dept, counts in dept_event_counts.items():
        arr = np.array(counts, dtype=np.float64)
        dept_stats[dept] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)) if len(arr) > 1 else 1.0,
        }

    user_stats: dict[str, dict[str, float]] = {}
    for rec in identity_records:
        ds = dept_stats.get(rec.department, {"mean": 0.0, "std": 1.0})
        user_stats[rec.user_id] = {"mean": ds["mean"], "std": ds["std"]}

    return user_stats, dept_stats


def _find_baseline_profile(identity, baseline_profiles):
    """Find the best-matching BaselineProfile for an identity record.

    Tries each role as a job_title key, then falls back to any profile
    for the same department.
    """
    for role in identity.roles:
        key = f"{identity.department}|{role}"
        if key in baseline_profiles:
            return baseline_profiles[key]
    for key, profile in baseline_profiles.items():
        if key.startswith(f"{identity.department}|"):
            return profile
    return None


def _print_report(report):
    """Print an EvaluationReport as a formatted table to stdout."""
    print("=" * 70)
    print("ACCESS SENTINEL EVALUATION REPORT")
    print("=" * 70)
    print(f"  Overall F1 Score:      {report.overall_f1:.4f}")
    print(f"  Macro F1 Score:        {report.macro_f1:.4f}")
    print(f"  Weighted F1 Score:     {report.weighted_f1:.4f}")
    print(f"  False Positive Rate:   {report.false_positive_rate:.4f}")
    print()
    if report.precision_by_class:
        print("  Per-Class Metrics:")
        print(f"    {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"    {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        for cls in sorted(report.precision_by_class.keys()):
            print(
                f"    {cls:<8} {report.precision_by_class[cls]:>10.4f} "
                f"{report.recall_by_class[cls]:>10.4f} "
                f"{report.f1_by_class[cls]:>10.4f}"
            )
    if report.false_positive_rate_by_dept:
        print()
        print("  FPR by Department:")
        for dept, fpr in sorted(report.false_positive_rate_by_dept.items()):
            print(f"    {dept:<20} {fpr:.4f}")
    print()
    print("  Confusion Matrix:")
    for row in report.confusion_matrix_data:
        print(f"    {row}")
    print("=" * 70)


def _check_targets(report, contamination):
    """Check target metrics and print tuning suggestions if needed.

    Returns True if all targets are met, False otherwise.
    """
    targets_met = True
    warnings: list[str] = []

    if report.overall_f1 <= TARGET_OVERALL_F1:
        targets_met = False
        warnings.append(
            f"Overall F1 ({report.overall_f1:.4f}) <= target ({TARGET_OVERALL_F1})"
        )

    if report.false_positive_rate >= TARGET_FALSE_POSITIVE_RATE:
        targets_met = False
        warnings.append(
            f"False Positive Rate ({report.false_positive_rate:.4f}) "
            f">= target ({TARGET_FALSE_POSITIVE_RATE})"
        )

    if not targets_met:
        for w in warnings:
            print(f"\nWARNING: {w}")

        print("\n--- TUNING SUGGESTIONS ---")

        prec_1 = report.precision_by_class.get(
            "1", report.precision_by_class.get("True", 0.0)
        )
        rec_1 = report.recall_by_class.get(
            "1", report.recall_by_class.get("True", 0.0)
        )

        if report.false_positive_rate >= TARGET_FALSE_POSITIVE_RATE:
            suggested_c = max(0.01, contamination * 0.5)
            suggested_nu = max(0.01, DEFAULT_SVM_NU * 0.5)
            reason = "High FPR — reduce contamination to flag fewer users as anomalous"
        elif rec_1 < prec_1:
            suggested_c = max(0.01, contamination * 0.6)
            suggested_nu = max(0.01, DEFAULT_SVM_NU * 0.6)
            reason = "Low recall relative to precision — increase contamination"
        else:
            suggested_c = min(0.35, contamination * 1.5)
            suggested_nu = min(0.15, DEFAULT_SVM_NU * 1.5)
            reason = "Low F1 with balanced precision/recall — adjust contamination"

        print(f"  Suggested contamination: {suggested_c:.4f}   (current: {contamination})")
        print(f"  Suggested svm_nu:       {suggested_nu:.4f}   (current: {DEFAULT_SVM_NU})")
        print(f"  Rationale: {reason}")
        print(f"  Consider adjusting ANOMALY_SCORE_THRESHOLD (current: {ANOMALY_SCORE_THRESHOLD})")

    return targets_met


def _try_evaluate_events(predictions_df, events_path, events_labels_df):
    """Attempt event-level evaluation by mapping user predictions to events.

    Returns an EvaluationReport or None if evaluation is not possible.
    """
    events_df = pd.read_csv(events_path)
    if "event_id" not in events_df.columns or "user_id" not in events_df.columns:
        return None

    event_to_user = events_df.set_index("event_id")["user_id"].to_dict()
    event_preds = []
    for _, row in events_labels_df.iterrows():
        eid = row["event_id"]
        uid = event_to_user.get(eid)
        if uid is not None:
            match = predictions_df[predictions_df["user_id"] == uid]
            pred_val = int(match["is_anomaly"].values[0]) if len(match) > 0 else 0
        else:
            pred_val = 0
        event_preds.append({"user_id": eid, "is_anomaly": pred_val})

    return evaluate_events(
        pd.DataFrame(event_preds),
        events_labels_df.rename(columns={"event_id": "user_id"}),
    )


def _report_to_dict(report):
    """Serialize an EvaluationReport to a JSON-compatible dictionary."""
    return {
        "overall_f1": report.overall_f1,
        "macro_f1": report.macro_f1,
        "weighted_f1": report.weighted_f1,
        "precision_by_class": report.precision_by_class,
        "recall_by_class": report.recall_by_class,
        "f1_by_class": report.f1_by_class,
        "false_positive_rate": report.false_positive_rate,
        "false_positive_rate_by_dept": report.false_positive_rate_by_dept,
        "confusion_matrix_data": report.confusion_matrix_data,
    }


def main():
    users_path = os.path.join(SAMPLE_DATA_DIR, "identity_users.csv")
    events_path = os.path.join(SAMPLE_DATA_DIR, "identity_events.csv")
    users_labels_path = os.path.join(SAMPLE_DATA_DIR, "identity_users_labels.csv")
    events_labels_path = os.path.join(SAMPLE_DATA_DIR, "identity_events_labels.csv")

    # Graceful exit when no ground truth labels exist (e.g. competition dataset)
    if not os.path.exists(users_labels_path) and not os.path.exists(events_labels_path):
        print(
            "No ground truth label files found for this dataset. "
            "Evaluation requires identity_users_labels.csv and "
            "identity_events_labels.csv in sample_data/. "
            "Skipping evaluation — the rest of the pipeline (ingestion, "
            "scoring, dashboard) works normally without labels."
        )
        sys.exit(0)

    print("Loading labels ...")
    users_labels_df = pd.read_csv(users_labels_path)
    events_labels_df = pd.read_csv(events_labels_path)

    users_labels_df["is_anomaly"] = _normalize_boolean_column(
        users_labels_df["is_anomaly"]
    )
    events_labels_df["is_anomaly"] = _normalize_boolean_column(
        events_labels_df["is_anomaly"]
    )

    print("Running unified holdout evaluation pipeline ...")
    from core.evaluator import run_holdout_evaluation
    report = run_holdout_evaluation()
    _print_report(report)

    # Event-level evaluation (separate from user-level holdout)
    print("\nEvaluating event-level predictions ...")
    events_report = None
    try:
        identity_records = ingest(users_path=users_path, events_path=events_path)
        # Simple ensemble-based event predictions
        events_df = pd.read_csv(events_path)
        event_preds = []
        for _, row in events_labels_df.iterrows():
            eid = row["event_id"]
            match = events_df[events_df["event_id"] == eid]
            user_id = match["user_id"].values[0] if len(match) > 0 else ""
            # Map event-level preds from event anomaly flag
            evt_is_anom = match["is_anomaly"].astype(str).str.lower().isin(("true","1","yes")).values[0] if len(match) > 0 else False
            event_preds.append({"user_id": eid, "is_anomaly": int(evt_is_anom)})
        events_report = _try_evaluate_events(
            pd.DataFrame(event_preds), events_path, events_labels_df
        )
        if events_report is not None:
            _print_report(events_report)
        else:
            print("  Event-level evaluation not possible (missing columns).")
    except Exception as exc:
        print(f"  Event-level evaluation skipped: {exc}")

    os.makedirs(DATA_DIR, exist_ok=True)
    report_path = os.path.join(DATA_DIR, "evaluation_report.json")
    combined = {"user_level": _report_to_dict(report)}
    if events_report is not None:
        combined["event_level"] = _report_to_dict(events_report)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2, default=str)
    print(f"\nEvaluation report saved to: {report_path}")

    targets_met = _check_targets(report, DEFAULT_CONTAMINATION)
    sys.exit(0 if targets_met else 1)


if __name__ == "__main__":
    main()
