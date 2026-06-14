"""Context-aware signal resolver for ambiguous access scenarios.

Disambiguates ambiguous scenarios and suppresses false positives
with documented reasoning.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from core.ingestion import IdentityRecord
from core.behavioral_baseline import BaselineProfile


@dataclass(frozen=True)
class ContextSignal:
    """Resolved context signal that adjusts risk scoring and suppresses rules.

    Attributes:
        signal_type: Unique identifier for this signal category.
        explanation: Human-readable justification for the signal.
        confidence: Confidence score between 0.0 and 1.0.
        score_adjustment: Integer adjustment to the risk score. Negative
            values reduce risk; positive values increase it.
        rules_suppressed: List of rule IDs suppressed by this signal.
        requires_followup: If True, triggers VERIFY_WITH_MANAGER remediation.
    """

    signal_type: str
    explanation: str
    confidence: float
    score_adjustment: int
    rules_suppressed: list[str]
    requires_followup: bool


def _compute_account_age_days(identity: IdentityRecord, now: datetime) -> int:
    """Return the account age in days, defaulting to 365 if unavailable."""
    if identity.created_at is not None:
        return (now - identity.created_at).days
    return 365


def _compute_days_since_login(identity: IdentityRecord, now: datetime) -> int:
    """Return days since last login, defaulting to 999 if unavailable."""
    if identity.last_login is not None:
        return (now - identity.last_login).days
    return 999


def _compute_behavior_zscore(
    identity: IdentityRecord, baseline: BaselineProfile
) -> float:
    """Compute raw z-score of event count against baseline mean and std."""
    if baseline.std_events_30d > 0.0:
        return (
            identity.event_count_30d - baseline.avg_events_30d
        ) / baseline.std_events_30d
    return 0.0


def _sabbatical_possible(
    identity: IdentityRecord, days_since_login: int
) -> ContextSignal | None:
    """Check for possible sabbatical or extended leave scenario.

    Args:
        identity: The identity record to evaluate.
        days_since_login: Days since the account's last login.

    Returns:
        ContextSignal if the sabbatical pattern matches, else None.
    """
    if not (
        30 <= days_since_login <= 180
        and identity.is_privileged
        and identity.employment_status == "active"
        and identity.owner_id
    ):
        return None
    return ContextSignal(
        signal_type="SABBATICAL_POSSIBLE",
        explanation=(
            f"Account with privileged access has been inactive for "
            f"{days_since_login} days while still listed as active with "
            f"a designated owner. Possible sabbatical or extended leave. "
            f"Verify with manager to confirm legitimate absence before flagging."
        ),
        confidence=0.60,
        score_adjustment=-10,
        rules_suppressed=[],
        requires_followup=True,
    )


def _temp_elevation(identity: IdentityRecord) -> ContextSignal | None:
    """Check for temporary role elevation following a recent role change.

    Args:
        identity: The identity record to evaluate.

    Returns:
        ContextSignal if a temporary elevation pattern is detected, else None.
    """
    if identity.role_changes_90d < 1:
        return None
    return ContextSignal(
        signal_type="TEMP_ELEVATION",
        explanation=(
            "Role changed within the past 14 days. Privilege creep and "
            "lateral movement spike rules suppressed for a 30-day "
            "observation window following role change."
        ),
        confidence=0.75,
        score_adjustment=-5,
        rules_suppressed=["PRIVILEGE_CREEP", "LATERAL_MOVEMENT_SPIKE"],
        requires_followup=False,
    )


def _new_hire_ramp(
    identity: IdentityRecord,
    baseline: BaselineProfile | None,
    account_age_days: int,
) -> ContextSignal | None:
    """Check for new-hire ramp-up period where access is still being scoped.

    Args:
        identity: The identity record to evaluate.
        baseline: The behavioral baseline, or None if unavailable.
        account_age_days: Age of the account in days.

    Returns:
        ContextSignal if the new-hire ramp pattern matches, else None.
    """
    if baseline is None:
        return None
    if account_age_days > 30:
        return None
    if identity.systems_count <= baseline.avg_unique_resources_30d:
        return None
    return ContextSignal(
        signal_type="NEW_HIRE_RAMP",
        explanation=(
            f"Account is {account_age_days} days old. Access scoping is "
            f"typically incomplete in the first 30 days for new hires."
        ),
        confidence=0.80,
        score_adjustment=-10,
        rules_suppressed=["OVER_PRIVILEGED"],
        requires_followup=False,
    )


def _batch_job_pattern(
    identity: IdentityRecord,
    baseline: BaselineProfile | None,
    behavior_zscore: float,
) -> ContextSignal | None:
    """Check for consistent off-hours automation (batch job) pattern.

    Args:
        identity: The identity record to evaluate.
        baseline: The behavioral baseline, or None if unavailable.
        behavior_zscore: The computed raw behavior z-score.

    Returns:
        ContextSignal if a batch job pattern is detected, else None.
    """
    if baseline is None:
        return None
    if not (
        identity.account_type == "service"
        and identity.off_hours_access_pct > 0.70
        and not identity.interactive_login
        and identity.event_count_30d > 20
        and behavior_zscore < 2.0
    ):
        return None
    return ContextSignal(
        signal_type="BATCH_JOB_PATTERN",
        explanation=(
            "Consistent off-hours automation pattern detected. No "
            "interactive login flag. Verify against scheduled job "
            "registry before treating as abuse."
        ),
        confidence=0.80,
        score_adjustment=-15,
        rules_suppressed=["SERVICE_ACCT_ABUSE"],
        requires_followup=False,
    )


def _month_end_finance(
    identity: IdentityRecord,
    baseline: BaselineProfile | None,
    current_day_of_month: int,
    behavior_zscore: float,
) -> ContextSignal | None:
    """Check for Finance department month-end close activity surge.

    Args:
        identity: The identity record to evaluate.
        baseline: The behavioral baseline, or None if unavailable.
        current_day_of_month: The current day of the month (1-31).
        behavior_zscore: The computed raw behavior z-score.

    Returns:
        ContextSignal if month-end Finance pattern matches, else None.
    """
    if baseline is None:
        return None
    if not (
        identity.department == "Finance"
        and behavior_zscore > 2.0
        and current_day_of_month >= 28
    ):
        return None
    multiplier = baseline.month_end_multiplier
    return ContextSignal(
        signal_type="MONTH_END_FINANCE",
        explanation=(
            f"Finance department shows elevated activity consistent with "
            f"month-end close. Baseline month_end_multiplier is "
            f"{multiplier:.1f}x. This is an expected pattern."
        ),
        confidence=0.90,
        score_adjustment=-20,
        rules_suppressed=["EXCESSIVE_ACCESS"],
        requires_followup=False,
    )


def _impossible_travel_confirmed(
    identity: IdentityRecord,
) -> ContextSignal | None:
    """Check for confirmed impossible travel indicator of compromise.

    Args:
        identity: The identity record to evaluate.

    Returns:
        ContextSignal if impossible travel is detected, else None.
    """
    if not identity.impossible_travel_detected and not identity.cross_system_impossible_travel:
        return None
    is_cross_system = bool(identity.cross_system_impossible_travel)
    adjustment = 30 if is_cross_system else 25
    explanation = (
        "Two authentications from geographically distant locations across "
        "different source systems within hours. Cross-system corroboration "
        "increases confidence of credential compromise."
        if is_cross_system
        else (
            "Two authentications from geographically impossible locations "
            "within hours. High-confidence indicator of credential compromise."
        )
    )
    return ContextSignal(
        signal_type="IMPOSSIBLE_TRAVEL_CONFIRMED",
        explanation=explanation,
        confidence=0.95,
        score_adjustment=adjustment,
        rules_suppressed=[],
        requires_followup=False,
    )


def _contractor_norm(identity: IdentityRecord) -> ContextSignal | None:
    """Check for contractor off-hours access which may be normal behaviour.

    Args:
        identity: The identity record to evaluate.

    Returns:
        ContextSignal documenting contractor norms, or None.
    """
    if not (
        identity.employment_status == "contractor"
        and identity.off_hours_access_pct > 0.30
    ):
        return None
    return ContextSignal(
        signal_type="CONTRACTOR_NORM",
        explanation=(
            "Contractor account. Off-hours access is common for remote "
            "contractors but should be cross-referenced with contract scope."
        ),
        confidence=0.70,
        score_adjustment=0,
        rules_suppressed=[],
        requires_followup=False,
    )


def resolve(
    identity: IdentityRecord,
    baseline: BaselineProfile | None,
) -> list[ContextSignal]:
    """Resolve ambiguous access scenarios and suppress false positives.

    Evaluates the identity record against known context patterns and the
    behavioral baseline to produce context signals that adjust risk scores
    and suppress inapplicable detection rules.

    Args:
        identity: The identity record to evaluate.
        baseline: The behavioral baseline profile, or None if unavailable.

    Returns:
        A list of ContextSignal objects representing resolved scenarios.
        Returns an empty list if no patterns match.
    """
    now = datetime.now(timezone.utc)
    account_age_days = _compute_account_age_days(identity, now)
    days_since_login = _compute_days_since_login(identity, now)
    current_day_of_month = now.day
    behavior_zscore = (
        _compute_behavior_zscore(identity, baseline) if baseline else 0.0
    )

    signals: list[ContextSignal] = []

    sig = _sabbatical_possible(identity, days_since_login)
    if sig is not None:
        signals.append(sig)

    sig = _temp_elevation(identity)
    if sig is not None:
        signals.append(sig)

    sig = _new_hire_ramp(identity, baseline, account_age_days)
    if sig is not None:
        signals.append(sig)

    sig = _batch_job_pattern(identity, baseline, behavior_zscore)
    if sig is not None:
        signals.append(sig)

    sig = _month_end_finance(
        identity, baseline, current_day_of_month, behavior_zscore
    )
    if sig is not None:
        signals.append(sig)

    sig = _impossible_travel_confirmed(identity)
    if sig is not None:
        signals.append(sig)

    sig = _contractor_norm(identity)
    if sig is not None:
        signals.append(sig)

    return signals
