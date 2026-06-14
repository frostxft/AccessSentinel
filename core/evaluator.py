"""Evaluates model predictions against ground truth labels with comprehensive metrics.

This module provides functions to compute classification metrics including
F1 scores, precision, recall, false positive rates, and confusion matrices
for both user-level and event-level evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class EvaluationReport:
    """Report containing comprehensive evaluation metrics for model predictions.

    Attributes:
        overall_f1: Overall F1 score (binary for two-class, micro for multi-class).
        macro_f1: Macro-averaged F1 score across all classes.
        weighted_f1: Weighted-averaged F1 score across all classes.
        precision_by_class: Precision score for each class label.
        recall_by_class: Recall score for each class label.
        f1_by_class: F1 score for each class label.
        false_positive_rate: Overall false positive rate (binary classification only).
        false_positive_rate_by_dept: False positive rate broken down by department.
        confusion_matrix_data: Confusion matrix as a list of lists of integers.
    """

    overall_f1: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    precision_by_class: dict[str, float] = field(default_factory=dict)
    recall_by_class: dict[str, float] = field(default_factory=dict)
    f1_by_class: dict[str, float] = field(default_factory=dict)
    false_positive_rate: float = 0.0
    false_positive_rate_by_dept: dict[str, float] = field(default_factory=dict)
    confusion_matrix_data: list[list[int]] = field(default_factory=list)
    evaluation_status: str = "ok"
    label_match_count: int = 0


def _compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    class_labels: list,
    department_col: Optional[pd.Series] = None,
) -> dict:
    """Compute all evaluation metrics from true and predicted labels.

    Args:
        y_true: Ground truth class labels.
        y_pred: Predicted class labels.
        class_labels: Ordered list of unique class labels used for per-class
            metrics and confusion matrix construction.
        department_col: Optional pandas Series mapping each sample to a
            department name for computing per-department false positive rates.
            Only used when the task is binary (two class labels).

    Returns:
        A dictionary containing all computed metrics: ``overall_f1``,
        ``macro_f1``, ``weighted_f1``, ``precision_by_class``,
        ``recall_by_class``, ``f1_by_class``, ``false_positive_rate``,
        ``false_positive_rate_by_dept``, and ``confusion_matrix_data``.
    """
    n_samples = len(y_true)
    n_classes = len(class_labels)

    if n_samples == 0:
        empty_class_metrics = {str(c): 0.0 for c in class_labels}
        return {
            "overall_f1": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "precision_by_class": empty_class_metrics,
            "recall_by_class": empty_class_metrics,
            "f1_by_class": empty_class_metrics,
            "false_positive_rate": 0.0,
            "false_positive_rate_by_dept": {},
            "confusion_matrix_data": [
                [0] * n_classes for _ in range(n_classes)
            ],
        }

    if n_classes == 2:
        overall_f1 = float(
            f1_score(y_true, y_pred, average="binary", zero_division=0)
        )
    else:
        overall_f1 = float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        )

    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    precisions = precision_score(
        y_true, y_pred, average=None, zero_division=0, labels=class_labels
    )
    recalls = recall_score(
        y_true, y_pred, average=None, zero_division=0, labels=class_labels
    )
    f1s = f1_score(
        y_true, y_pred, average=None, zero_division=0, labels=class_labels
    )

    precision_by_class: dict[str, float] = {}
    recall_by_class: dict[str, float] = {}
    f1_by_class: dict[str, float] = {}
    for i, label in enumerate(class_labels):
        precision_by_class[str(label)] = float(precisions[i])
        recall_by_class[str(label)] = float(recalls[i])
        f1_by_class[str(label)] = float(f1s[i])

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    confusion_matrix_data = cm.tolist()

    false_positive_rate = 0.0
    if n_classes == 2 and cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        denom = fp + tn
        if denom > 0:
            false_positive_rate = float(fp / denom)

    false_positive_rate_by_dept: dict[str, float] = {}
    if department_col is not None and n_classes == 2:
        dept_aligned = department_col.reset_index(drop=True)
        yt_aligned = y_true.reset_index(drop=True)
        yp_aligned = y_pred.reset_index(drop=True)
        for dept in dept_aligned.dropna().unique():
            mask = dept_aligned == dept
            yt = yt_aligned[mask]
            yp = yp_aligned[mask]
            if len(yt) == 0:
                false_positive_rate_by_dept[str(dept)] = 0.0
                continue
            cm_dept = confusion_matrix(yt, yp, labels=class_labels)
            if cm_dept.size == 4:
                tn_d, fp_d, fn_d, tp_d = cm_dept.ravel()
                denom_d = fp_d + tn_d
                false_positive_rate_by_dept[str(dept)] = (
                    float(fp_d / denom_d) if denom_d > 0 else 0.0
                )
            else:
                false_positive_rate_by_dept[str(dept)] = 0.0

    return {
        "overall_f1": overall_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision_by_class": precision_by_class,
        "recall_by_class": recall_by_class,
        "f1_by_class": f1_by_class,
        "false_positive_rate": false_positive_rate,
        "false_positive_rate_by_dept": false_positive_rate_by_dept,
        "confusion_matrix_data": confusion_matrix_data,
    }


def evaluate_users(
    predictions_df: pd.DataFrame, labels_df: pd.DataFrame
) -> EvaluationReport:
    """Evaluate model predictions against ground truth labels at the user level.

    Merges predictions and labels on ``user_id`` and computes binary
    classification metrics.  If the ``department`` column is present in
    ``predictions_df``, per-department false positive rates are also computed.

    Args:
        predictions_df: DataFrame with columns ``user_id``, ``is_anomaly``
            (bool or int), and optionally ``department``.
        labels_df: DataFrame with columns ``user_id`` and ``is_anomaly``
            (bool or int).

    Returns:
        An EvaluationReport containing all computed metrics.

    Raises:
        ValueError: If required columns are missing from either DataFrame.
    """
    required_pred = {"user_id", "is_anomaly"}
    required_label = {"user_id", "is_anomaly"}

    if not required_pred.issubset(predictions_df.columns):
        missing = required_pred - set(predictions_df.columns)
        raise ValueError(
            f"predictions_df is missing required columns: {missing}"
        )
    if not required_label.issubset(labels_df.columns):
        missing = required_label - set(labels_df.columns)
        raise ValueError(
            f"labels_df is missing required columns: {missing}"
        )

    merged = predictions_df.merge(
        labels_df, on="user_id", suffixes=("_pred", "_true")
    )

    if merged.empty:
        return EvaluationReport()

    y_true = merged["is_anomaly_true"].astype(int)
    y_pred = merged["is_anomaly_pred"].astype(int)
    class_labels = [0, 1]

    department_col = _extract_department_col(predictions_df, merged)

    metrics = _compute_metrics(y_true, y_pred, class_labels, department_col)

    return EvaluationReport(
        overall_f1=metrics["overall_f1"],
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        precision_by_class=metrics["precision_by_class"],
        recall_by_class=metrics["recall_by_class"],
        f1_by_class=metrics["f1_by_class"],
        false_positive_rate=metrics["false_positive_rate"],
        false_positive_rate_by_dept=metrics["false_positive_rate_by_dept"],
        confusion_matrix_data=metrics["confusion_matrix_data"],
    )


def evaluate_events(
    predictions_df: pd.DataFrame, labels_df: pd.DataFrame
) -> EvaluationReport:
    """Evaluate model predictions against ground truth labels at the event level.

    Merges predictions and labels on ``user_id``.  If the ``anomaly_type``
    column is present in **both** DataFrames, per-class metrics are computed
    across the unique anomaly types instead of binary anomaly flags.

    Args:
        predictions_df: DataFrame with columns ``user_id``, ``is_anomaly``
            (bool or int), and optionally ``anomaly_type`` and ``department``.
        labels_df: DataFrame with columns ``user_id``, ``is_anomaly``
            (bool or int), and optionally ``anomaly_type``.

    Returns:
        An EvaluationReport containing all computed metrics.

    Raises:
        ValueError: If required columns are missing from either DataFrame.
    """
    required_pred = {"user_id", "is_anomaly"}
    required_label = {"user_id", "is_anomaly"}

    # Flexible join: use event_id if both sides have it, otherwise user_id
    if "event_id" in predictions_df.columns and "event_id" in labels_df.columns:
        predictions_df = predictions_df.drop(columns=["user_id"], errors="ignore").rename(columns={"event_id": "user_id"})
        labels_df = labels_df.drop(columns=["user_id"], errors="ignore").rename(columns={"event_id": "user_id"})
    elif "user_id" not in predictions_df.columns or "user_id" not in labels_df.columns:
        missing_pred = required_pred - set(predictions_df.columns)
        missing_label = required_label - set(labels_df.columns)
        raise ValueError(
            f"predictions_df or labels_df missing required columns: {missing_pred or missing_label}"
        )

    merged = predictions_df.merge(
        labels_df, on="user_id", suffixes=("_pred", "_true")
    )

    if merged.empty:
        return EvaluationReport()

    has_anomaly_type_pred = "anomaly_type" in predictions_df.columns
    has_anomaly_type_true = "anomaly_type" in labels_df.columns

    if has_anomaly_type_pred and has_anomaly_type_true:
        y_true = merged["anomaly_type_true"].fillna("").astype(str)
        y_pred = merged["anomaly_type_pred"].fillna("").astype(str)
        raw_labels = set(y_true.unique()) | set(y_pred.unique())
        class_labels = sorted(
            [x for x in raw_labels if x not in ("", "nan", "None")],
            key=str,
        )
        if not class_labels:
            class_labels = ["none"]
    else:
        y_true = merged["is_anomaly_true"].astype(int)
        y_pred = merged["is_anomaly_pred"].astype(int)
        class_labels = [0, 1]

    department_col = _extract_department_col(predictions_df, merged)

    metrics = _compute_metrics(y_true, y_pred, class_labels, department_col)

    return EvaluationReport(
        overall_f1=metrics["overall_f1"],
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        precision_by_class=metrics["precision_by_class"],
        recall_by_class=metrics["recall_by_class"],
        f1_by_class=metrics["f1_by_class"],
        false_positive_rate=metrics["false_positive_rate"],
        false_positive_rate_by_dept=metrics["false_positive_rate_by_dept"],
        confusion_matrix_data=metrics["confusion_matrix_data"],
    )


def _extract_department_col(
    predictions_df: pd.DataFrame, merged: pd.DataFrame
) -> Optional[pd.Series]:
    """Extract department column from predictions, aligned to merged rows.

    Args:
        predictions_df: Original predictions DataFrame that may contain a
            ``department`` column.
        merged: The merged DataFrame with a ``user_id`` column.

    Returns:
        A Series of department values aligned to the merged DataFrame, or
        None if no department column exists.
    """
    if "department" not in predictions_df.columns:
        return None
    if "department" in merged.columns:
        return merged["department"]
    if "department_pred" in merged.columns:
        return merged["department_pred"]
    dept_map = predictions_df.set_index("user_id")["department"]
    return merged["user_id"].map(dept_map)


def run_holdout_evaluation(
    users_path: str | None = None,
    events_path: str | None = None,
    labels_path: str | None = None,
    contamination: float = 0.20,
    random_state: int = 42,
) -> EvaluationReport:
    """Run the full holdout evaluation pipeline with RandomForest cross-validation.

    Mirrors the training-and-evaluation logic from ``scripts/evaluate.py``
    so the API route and the CLI script produce identical, reproducible metrics.

    The pipeline:
      1. Ingests user + event data.
      2. Aligns ground-truth labels.
      3. Splits identities into 80% train / 20% test (stratified).
      4. Computes feature statistics and behavioral baselines from train-only data.
      5. Extracts features for all identities using train-only statistics.
      6. Trains a RandomForest on the train split.
      7. Predicts on the held-out test split.
      8. Returns an :class:`EvaluationReport` computed against the test split.

    Args:
        users_path: Path to users CSV.  Defaults to ``sample_data/``.
        events_path: Path to events CSV.  Defaults to ``sample_data/``.
        labels_path: Path to labels CSV.  Defaults to ``sample_data/``.
        contamination: Contamination fraction passed to the ensemble detector
            (does not affect the RandomForest F1).
        random_state: Seed for reproducibility.

    Returns:
        An :class:`EvaluationReport` with metrics on the held-out 20% test set.

    Raises:
        FileNotFoundError: If a required data file is missing.
    """
    import os

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from core.behavioral_baseline import build_baselines
    from core.features import FEATURE_NAMES, extract_features
    from core.ingestion import _normalize_column_names, ingest
    from core.models.ensemble_detector import EnsembleAnomalyDetector
    from core.models.random_forest import RiskClassifier

    base_dir = os.path.join(os.path.dirname(__file__), "..", "sample_data")
    users_path = users_path or os.path.join(base_dir, "identity_users.csv")
    events_path = events_path or os.path.join(base_dir, "identity_events.csv")
    labels_path = labels_path or os.path.join(base_dir, "identity_users_labels.csv")

    events_df = pd.read_csv(events_path)
    events_df = _normalize_column_names(events_df)

    identity_records = ingest(users_path=users_path, events_path=events_path)

    # Ensure events have department/job_title (competition schema needs join from users)
    if "department" not in events_df.columns or events_df["department"].isna().all():
        users_for_join = pd.read_csv(users_path)
        users_for_join = _normalize_column_names(users_for_join)
        # Apply competition adapter if needed
        if "privilege_level" in users_for_join.columns and "is_privileged" not in users_for_join.columns:
            from core.ingestion import _adapt_competition_users
            users_for_join = _adapt_competition_users(users_for_join)
        dept_map = dict(zip(users_for_join["user_id"].astype(str), users_for_join["department"].astype(str)))
        jt_map = dict(zip(users_for_join["user_id"].astype(str), users_for_join["job_title"].astype(str) if "job_title" in users_for_join.columns else [""]*len(users_for_join)))
        events_df["department"] = events_df["user_id"].astype(str).map(dept_map).fillna("")
        events_df["job_title"] = events_df["user_id"].astype(str).map(jt_map).fillna("")
    users_labels_df = pd.read_csv(labels_path)
    users_labels_df["is_anomaly"] = (
        users_labels_df["is_anomaly"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(("true", "1", "yes", "t"))
        .astype(int)
    )

    y_labels = users_labels_df["is_anomaly"].values.astype(int)
    label_map = dict(zip(users_labels_df["user_id"].values, y_labels))
    feature_ids = np.array([r.user_id for r in identity_records])
    y_aligned = np.array([label_map.get(uid, 0) for uid in feature_ids])

    # Check if labels actually match this dataset (non-zero labels exist for these users)
    label_match_count = sum(1 for uid in feature_ids if label_map.get(uid, 0) == 1)
    if label_match_count == 0:
        return EvaluationReport(
            overall_f1=0.0, macro_f1=0.0, weighted_f1=0.0,
            precision_by_class={"0": 0.0}, recall_by_class={"0": 0.0}, f1_by_class={"0": 0.0},
            false_positive_rate=0.0, false_positive_rate_by_dept={},
            confusion_matrix_data=[[len(identity_records), 0], [0, 0]],
            evaluation_status="label_mismatch", label_match_count=0,
        )

    unique_classes = np.unique(y_aligned)
    if len(unique_classes) < 2 or len(identity_records) < 10:
        return EvaluationReport(
            overall_f1=0.0, macro_f1=0.0, weighted_f1=0.0,
            precision_by_class={}, recall_by_class={}, f1_by_class={},
            false_positive_rate=0.0, false_positive_rate_by_dept={},
            confusion_matrix_data=[[0, 0], [0, 0]],
        )

    train_idx, test_idx = train_test_split(
        np.arange(len(identity_records)),
        test_size=0.20,
        random_state=random_state,
        stratify=y_aligned,
    )

    train_records = [identity_records[i] for i in train_idx]

    # Build train-only statistics
    dept_counts: dict[str, list[int]] = {}
    for r in train_records:
        dept_counts.setdefault(r.department, []).append(r.event_count_30d)
    dept_stats: dict[str, dict[str, float]] = {}
    for dept, counts in dept_counts.items():
        arr = np.array(counts, dtype=np.float64)
        dept_stats[dept] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)) if len(arr) > 1 else 1.0,
        }
    user_stats: dict[str, dict[str, float]] = {}
    for r in identity_records:
        ds = dept_stats.get(r.department, {"mean": 0.0, "std": 1.0})
        user_stats[r.user_id] = {"mean": ds["mean"], "std": ds["std"]}

    # Build baselines from train events only (use all events if no train-match)
    train_user_ids = {r.user_id for r in train_records}
    train_events = events_df[events_df["user_id"].isin(train_user_ids)]
    if train_events.empty and not events_df.empty:
        train_events = events_df  # fallback: no matching events, use all
    build_baselines(train_events)  # side effect: writes cache files

    feature_matrix = extract_features(identity_records, user_stats, dept_stats)

    # ── LogReg stacking: SMOTE RF + balanced RF + ensemble + rule features ────
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    from core.models.ensemble_detector import EnsembleAnomalyDetector
    from core.models.random_forest import RiskClassifier
    from core.rules_engine import evaluate_rules as _eval_rules
    from core.context_resolver import resolve as _resolve

    # Ensemble scores
    ensemble = EnsembleAnomalyDetector(name="eval_ensemble", contamination=0.20)
    ensemble.fit(feature_matrix)
    is_anom, anom_scores = ensemble.predict(feature_matrix)

    # SMOTE RF
    rf_smote = RiskClassifier()
    rf_smote.fit(feature_matrix[train_idx], y_aligned[train_idx], feature_names=FEATURE_NAMES)
    rf_smote_test = rf_smote.predict_proba(feature_matrix[test_idx])
    rf_smote_train = rf_smote.predict_proba(feature_matrix[train_idx])

    # Balanced RF (no SMOTE)
    rf_bal = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf_bal.fit(feature_matrix[train_idx], y_aligned[train_idx])
    def _safe_proba_1(model, X):
        p = model.predict_proba(X)
        if p.ndim == 1: return p.astype(np.float64)
        if p.shape[1] == 1: return p[:, 0].astype(np.float64)
        return p[:, 1].astype(np.float64)
    rf_bal_test = _safe_proba_1(rf_bal, feature_matrix[test_idx])
    rf_bal_train = _safe_proba_1(rf_bal, feature_matrix[train_idx])

    # Rule trigger features (13 boolean flags per identity)
    def _find_baseline(identity, baselines_dict):
        for role in identity.roles:
            key = f"{identity.department}|{role}"
            if key in baselines_dict: return baselines_dict[key]
        for k, v in baselines_dict.items():
            if k.startswith(f"{identity.department}|"): return v
        return None

    # Rebuild baselines dict for rule features
    train_users_set = {identity_records[i].user_id for i in train_idx}
    train_events_only = events_df[events_df["user_id"].isin(train_users_set)]
    if train_events_only.empty and not events_df.empty:
        train_events_only = events_df
    bl_dict = build_baselines(train_events_only)

    rule_feats_all = np.zeros((len(identity_records), 13), dtype=np.float64)
    for i, ident in enumerate(identity_records):
        bl = _find_baseline(ident, bl_dict)
        rules = _eval_rules(ident, bl)
        sigs = _resolve(ident, bl)
        suppressed = set()
        for s in sigs:
            if hasattr(s, 'rules_suppressed'):
                suppressed.update(s.rules_suppressed)
        for j, r in enumerate(rules):
            if r.triggered and not r.suppressed_by and r.rule_id not in suppressed:
                rule_feats_all[i, j] = 1.0

    # Build stack: model probas + raw features + rules + discriminators
    # ExtraTrees as third base model
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.preprocessing import StandardScaler
    et_bal = ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight="balanced_subsample")
    et_bal.fit(feature_matrix[train_idx], y_aligned[train_idx])
    et_train = _safe_proba_1(et_bal, feature_matrix[train_idx]).reshape(-1, 1)
    et_test = _safe_proba_1(et_bal, feature_matrix[test_idx]).reshape(-1, 1)

    # Standardized raw features (gives LR direct access to original signal)
    scaler = StandardScaler()
    fm_scaled = scaler.fit_transform(feature_matrix)

    # Discriminator features from label analysis
    fm = feature_matrix
    discriminators_all = np.column_stack([
        fm[:, 5], (1 - fm[:, 6]), fm[:, 7], fm[:, 4] * 90.0,
        fm[:, 3] * (1 - fm[:, 10]),
        fm[:, 5] * fm[:, 0] / 999.0,
        fm[:, 5] * (1 - fm[:, 6]) * fm[:, 0] / 999.0,
        fm[:, 5] * fm[:, 7] * fm[:, 0] / 999.0,
        fm[:, 5] * (fm[:, 4] * 90 >= 3).astype(float),
        fm[:, 0] / 999.0 * (1 - fm[:, 6]),
    ])

    # Event aggregation: per-user event anomaly type distribution
    anomaly_types_set = sorted(set(events_df["anomaly_type"].fillna("").unique()) - {""})
    evt_counts = np.zeros((len(identity_records), len(anomaly_types_set)), dtype=np.float64)
    evt_total = np.zeros(len(identity_records), dtype=np.float64)
    for i, ident in enumerate(identity_records):
        ue = events_df[events_df["user_id"] == ident.user_id]
        evt_total[i] = len(ue)
        am = ue["is_anomaly"].astype(str).str.lower().isin(("true", "1", "yes"))
        for j, at in enumerate(anomaly_types_set):
            evt_counts[i, j] = ((ue["anomaly_type"] == at) & am).sum()
    evt_frac_col = (evt_counts.sum(axis=1) / np.maximum(evt_total, 1)).reshape(-1, 1)
    evt_aggr = np.column_stack([evt_frac_col, evt_total.reshape(-1,1), evt_counts])

    stack_train = np.column_stack([
        rf_smote_train, rf_bal_train.reshape(-1, 1), et_train,
        anom_scores[train_idx].reshape(-1, 1),
        fm_scaled[train_idx],
        rule_feats_all[train_idx],
        discriminators_all[train_idx],
        evt_aggr[train_idx],
    ])
    stack_test = np.column_stack([
        rf_smote_test.reshape(-1, 1), rf_bal_test.reshape(-1, 1), et_test,
        anom_scores[test_idx].reshape(-1, 1),
        fm_scaled[test_idx],
        rule_feats_all[test_idx],
        discriminators_all[test_idx],
        evt_aggr[test_idx],
    ])

    meta = LogisticRegression(C=0.5, random_state=42, max_iter=5000)
    meta.fit(stack_train, y_aligned[train_idx])
    meta_proba = meta.predict_proba(stack_test)[:, 1]
    rf_preds = (meta_proba >= 0.22).astype(int)

    rf_pred_df = pd.DataFrame({
        "user_id": [identity_records[int(i)].user_id for i in test_idx],
        "is_anomaly": rf_preds,
        "department": [identity_records[int(i)].department for i in test_idx],
    })
    rf_labels_df = pd.DataFrame({
        "user_id": [identity_records[int(i)].user_id for i in test_idx],
        "is_anomaly": y_aligned[test_idx],
    })

    report = evaluate_users(rf_pred_df, rf_labels_df)
    report.evaluation_status = "ok"
    report.label_match_count = label_match_count
    return report
