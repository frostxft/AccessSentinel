"""DLP (Data Loss Prevention) integration for AccessSentinel.

Detects exfiltration risk patterns from event logs: bulk downloads,
unusual export activity, and after-hours data access to sensitive resources.
"""

from dataclasses import dataclass

EXFIL_TRIGGER_COUNT = 5
EXFIL_RESOURCE_SENSITIVITY_SCORE = {"critical": 4, "high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class ExfiltrationAlert:
    """An exfiltration-related risk alert for a single identity."""

    user_id: str
    username: str
    risk_level: str  # CRITICAL, HIGH, MEDIUM, LOW
    total_exports: int
    after_hours_exports: int
    sensitive_resource_count: int
    exfiltration_score: int
    description: str


def detect_exfiltration_risk(
    identity_record,
    event_log_entries: list[dict],
) -> ExfiltrationAlert | None:
    """Detect exfiltration risk for a single identity from their event log.

    Args:
        identity_record: IdentityRecord with username, department, etc.
        event_log_entries: List of dicts with action, resource_sensitivity,
            time_classification (or timestamp), and status fields.

    Returns:
        ExfiltrationAlert if risk detected, otherwise None.
    """
    exports = [
        e for e in event_log_entries
        if e.get("action", "").lower() in ("download", "export_data", "export")
    ]
    if len(exports) < EXFIL_TRIGGER_COUNT:
        return None

    after_hours = [
        e for e in exports
        if e.get("time_classification", "") in ("unusual_hours", "night", "weekend")
        or (hasattr(e, "_hour") and e.get("_hour", 12) in (0, 1, 2, 3, 4, 5))
    ]
    sensitive = [
        e for e in exports
        if e.get("resource_sensitivity", "low") in ("critical", "high")
    ]

    score = (
        len(exports)
        + len(after_hours) * 2
        + len(sensitive) * EXFIL_RESOURCE_SENSITIVITY_SCORE.get(
            "critical" if any(e.get("resource_sensitivity") == "critical" for e in sensitive) else "high", 3
        )
    )

    if score >= 30:
        risk = "CRITICAL"
    elif score >= 20:
        risk = "HIGH"
    elif score >= 10:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return ExfiltrationAlert(
        user_id=getattr(identity_record, "user_id", ""),
        username=getattr(identity_record, "username", ""),
        risk_level=risk,
        total_exports=len(exports),
        after_hours_exports=len(after_hours),
        sensitive_resource_count=len(sensitive),
        exfiltration_score=score,
        description=(
            f"{len(exports)} export actions detected "
            f"({len(after_hours)} after-hours, "
            f"{len(sensitive)} on sensitive resources). "
            f"Exfiltration risk score: {score}."
        ),
    )
