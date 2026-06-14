"""Regression test for false_positive_rate_by_dept in evaluate_users."""
import pandas as pd
from core.evaluator import evaluate_users


def test_fpr_by_dept_with_known_fps():
    preds = pd.DataFrame({
        "user_id": ["U1", "U2", "U3", "U4", "U5"],
        "is_anomaly": [1, 1, 1, 0, 0],
        "department": ["HR", "IT", "HR", "HR", "IT"],
    })
    labels = pd.DataFrame({
        "user_id": ["U1", "U2", "U3", "U4", "U5"],
        "is_anomaly": [1, 0, 0, 0, 0],
    })
    report = evaluate_users(preds, labels)
    fpr_dept = report.false_positive_rate_by_dept
    assert "HR" in fpr_dept, f"HR missing from {fpr_dept}"
    assert "IT" in fpr_dept, f"IT missing from {fpr_dept}"
    assert abs(fpr_dept["HR"] - 0.5) < 0.001, f"Expected HR FPR=0.5, got {fpr_dept['HR']}"
    assert abs(fpr_dept["IT"] - 0.5) < 0.001, f"Expected IT FPR=0.5, got {fpr_dept['IT']}"


def test_fpr_by_dept_zero_fps_still_present():
    preds = pd.DataFrame({
        "user_id": ["U1", "U2"],
        "is_anomaly": [1, 1],
        "department": ["HR", "IT"],
    })
    labels = pd.DataFrame({
        "user_id": ["U1", "U2"],
        "is_anomaly": [1, 1],
    })
    report = evaluate_users(preds, labels)
    fpr = report.false_positive_rate_by_dept
    assert "HR" in fpr, f"HR missing from {fpr}"
    assert "IT" in fpr, f"IT missing from {fpr}"
    assert fpr["HR"] == 0.0, f"Expected 0 FPR, got {fpr['HR']}"
    assert fpr["IT"] == 0.0, f"Expected 0 FPR, got {fpr['IT']}"
