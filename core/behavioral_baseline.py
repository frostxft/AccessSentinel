"""Behavioral baseline module for AccessSentinel.

Establishes per-role, per-department behavioral baselines from 12 months of
event history to reduce false positives in anomaly detection by accounting
for department-specific seasonal patterns (e.g. Finance month/quarter-end).
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
_FINANCE_DEPT: str = "Finance"
_BASELINES_DIR: str = os.path.join(
    os.path.dirname(__file__), "..", "data", "baselines"
)
_MONTH_END_DAYS: frozenset[int] = frozenset(range(28, 32))
_QUARTER_END_MONTHS: frozenset[int] = frozenset({3, 6, 9, 12})
_THIRTY_DAYS: pd.Timedelta = pd.Timedelta(days=30)
_TWELVE_MONTHS: pd.Timedelta = pd.Timedelta(days=365)
_P95_PERCENTILE: float = 95.0
_P99_PERCENTILE: float = 99.0
_MONTH_END_DAY_THRESHOLD: int = 28
_EPSILON: float = 1e-8
_TOP_N_HOURS: int = 8
_TOP_N_RESOURCES: int = 10


@dataclass(frozen=True)
class BaselineProfile:
    """Per-role, per-department behavioral baseline profile.

    Captures aggregate event patterns computed from 12 months of history
    for a specific department+job_title combination. Used by the scoring
    engine to contextualize anomaly thresholds and suppress false positives
    that arise from predictable surges (e.g. Finance month-end).

    Attributes:
        role: Job title / role name (e.g. "Financial Analyst").
        department: Department name (e.g. "Finance").
        avg_events_30d: Mean 30-day event count across users in the group.
        std_events_30d: Standard deviation of 30-day event counts.
        avg_events_per_session: Mean events per session (day) across the group.
        typical_access_hours: Hours of day (0-23) with above-average activity,
            sorted most-active first.
        typical_resources: Most frequently accessed resources, sorted by count
            descending.
        avg_unique_resources_30d: Mean unique resources accessed per user in
            the last 30 days.
        std_unique_resources_30d: Standard deviation of unique resources per
            user in the last 30 days.
        p95_events_single_session: 95th percentile of events in a single
            session (day) — threshold for EXCESSIVE_ACCESS rule.
        p99_download_count: 99th percentile of download action count per user —
            threshold for BULK_DOWNLOAD rule.
        month_end_multiplier: For Finance: ratio of events on days 28-31
            vs overall daily average. Used to suppress false positives during
            month-end close. 1.0 for non-Finance departments.
        quarter_end_multiplier: For Finance: ratio of events in quarter-end
            months (3, 6, 9, 12) vs other months. 1.0 for non-Finance.
        computed_at: ISO-8601 timestamp of when this baseline was generated.
    """

    role: str
    department: str
    avg_events_30d: float
    std_events_30d: float
    avg_events_per_session: float
    typical_access_hours: list[int]
    typical_resources: list[str]
    avg_unique_resources_30d: float
    std_unique_resources_30d: float
    p95_events_single_session: float
    p99_download_count: float
    month_end_multiplier: float
    quarter_end_multiplier: float
    computed_at: str
    avg_daily_unique_resources: float = 0.0


def _prepare_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps and return filtered 12-month event slice.

    Args:
        events_df: Raw event DataFrame with at least a "timestamp" column.

    Returns:
        DataFrame with parsed ``_ts`` datetime column, filtered to the
        trailing 12-month window.
    """
    df = events_df.copy()
    df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["_ts"])
    if df.empty:
        return df
    max_ts = df["_ts"].max()
    cutoff_12m = max_ts - _TWELVE_MONTHS
    return df[df["_ts"] >= cutoff_12m]


def _compute_30d_event_stats(
    group_df: pd.DataFrame, cutoff_30d: pd.Timestamp
) -> dict[str, float]:
    """Compute per-user 30-day event count mean and std for a group.

    Args:
        group_df: Events for a single (department, job_title) group (12-month
            window).
        cutoff_30d: Timestamp 30 days before the reference date.

    Returns:
        Dict with ``avg_events_30d`` and ``std_events_30d``.
    """
    recent = group_df[group_df["_ts"] >= cutoff_30d]
    if recent.empty:
        return {"avg_events_30d": 0.0, "std_events_30d": 0.0}
    per_user = recent.groupby("user_id").size()
    return {
        "avg_events_30d": float(per_user.mean()),
        "std_events_30d": float(per_user.std(ddof=0)) if len(per_user) > 1 else 0.0,
    }


def _compute_session_stats(group_df: pd.DataFrame) -> dict[str, float]:
    """Compute per-session and single-session event statistics.

    A session is approximated as a calendar day.

    Args:
        group_df: Events for a single (department, job_title) group.

    Returns:
        Dict with ``avg_events_per_session`` and ``p95_events_single_session``.
    """
    group_df = group_df.copy()
    group_df["_date"] = group_df["_ts"].dt.date
    per_session = group_df.groupby(["user_id", "_date"]).size()
    if per_session.empty:
        return {"avg_events_per_session": 0.0, "p95_events_single_session": 0.0}
    return {
        "avg_events_per_session": float(per_session.mean()),
        "p95_events_single_session": float(
            np.percentile(per_session.values, _P95_PERCENTILE)
        ),
    }


def _compute_access_hours(group_df: pd.DataFrame) -> list[int]:
    """Return the most active access hours for the group.

    Args:
        group_df: Events for a single (department, job_title) group.

    Returns:
        Hour-of-day integers (0-23) sorted by event count descending,
        limited to hours with above-average activity.
    """
    if group_df.empty:
        return []
    hour_counts = group_df["_ts"].dt.hour.value_counts().sort_index()
    if hour_counts.empty:
        return []
    mean_count = hour_counts.mean()
    above_avg = hour_counts[hour_counts >= mean_count]
    return above_avg.sort_values(ascending=False).index.tolist()[: _TOP_N_HOURS]


def _compute_typical_resources(group_df: pd.DataFrame) -> list[str]:
    """Return the most frequently accessed resources for the group.

    Args:
        group_df: Events for a single (department, job_title) group.

    Returns:
        Resource names sorted by access count descending.
    """
    if group_df.empty or "resource" not in group_df.columns:
        return []
    resource_counts = group_df["resource"].value_counts()
    if resource_counts.empty:
        return []
    return resource_counts.head(_TOP_N_RESOURCES).index.tolist()


def _compute_unique_resources_stats(
    group_df: pd.DataFrame, cutoff_30d: pd.Timestamp
) -> dict[str, float]:
    """Compute per-user unique resource access statistics over 30 days.

    Args:
        group_df: Events for a single (department, job_title) group.
        cutoff_30d: Timestamp 30 days before the reference date.

    Returns:
        Dict with ``avg_unique_resources_30d`` and
        ``std_unique_resources_30d``.
    """
    recent = group_df[group_df["_ts"] >= cutoff_30d]
    if recent.empty or "resource" not in recent.columns:
        return {"avg_unique_resources_30d": 0.0, "std_unique_resources_30d": 0.0}
    per_user = recent.groupby("user_id")["resource"].nunique()
    if per_user.empty:
        return {"avg_unique_resources_30d": 0.0, "std_unique_resources_30d": 0.0}
    return {
        "avg_unique_resources_30d": float(per_user.mean()),
        "std_unique_resources_30d": (
            float(per_user.std(ddof=0)) if len(per_user) > 1 else 0.0
        ),
    }


def _compute_p99_downloads(group_df: pd.DataFrame) -> float:
    """Compute 99th percentile of download action counts per user.

    Args:
        group_df: Events for a single (department, job_title) group.

    Returns:
        99th-percentile download count per user, or 0.0 if no download events.
    """
    if group_df.empty or "action" not in group_df.columns:
        return 0.0
    download_mask = group_df["action"].astype(str).str.lower().eq("download")
    downloads = group_df[download_mask]
    if downloads.empty:
        return 0.0
    per_user = downloads.groupby("user_id").size()
    if per_user.empty:
        return 0.0
    return float(np.percentile(per_user.values, _P99_PERCENTILE))


def _compute_finance_multipliers(group_df: pd.DataFrame) -> dict[str, float]:
    """Compute month-end and quarter-end volume multipliers for Finance.

    Args:
        group_df: Events for a Finance (department, job_title) group.

    Returns:
        Dict with ``month_end_multiplier`` and ``quarter_end_multiplier``.
    """
    if group_df.empty:
        return {"month_end_multiplier": 1.0, "quarter_end_multiplier": 1.0}

    # Month-end multiplier: ratio of events on days 28-31 vs daily average
    day_counts = group_df["_ts"].dt.day.value_counts()
    all_days = day_counts.index[day_counts.index.isin(range(1, 32))]
    if len(all_days) == 0:
        daily_avg = 0.0
    else:
        daily_avg = day_counts[all_days].mean()

    month_end_days_present = [
        d for d in _MONTH_END_DAYS if d in day_counts.index
    ]
    if daily_avg > 0 and month_end_days_present:
        month_end_avg = day_counts[month_end_days_present].mean()
        month_end_multiplier = month_end_avg / daily_avg
    else:
        month_end_multiplier = 1.0

    # Quarter-end multiplier: ratio of events in months 3,6,9,12 vs others
    month_counts = group_df["_ts"].dt.month.value_counts()
    qe_months_present = [
        m for m in _QUARTER_END_MONTHS if m in month_counts.index
    ]
    non_qe_months_present = [
        m for m in month_counts.index if m not in _QUARTER_END_MONTHS
    ]

    if qe_months_present and non_qe_months_present:
        qe_avg = month_counts[qe_months_present].mean()
        non_qe_avg = month_counts[non_qe_months_present].mean()
        if non_qe_avg > 0:
            quarter_end_multiplier = qe_avg / non_qe_avg
        else:
            quarter_end_multiplier = 1.0
    else:
        quarter_end_multiplier = 1.0

    return {
        "month_end_multiplier": month_end_multiplier,
        "quarter_end_multiplier": quarter_end_multiplier,
    }


def _save_baseline(profile: BaselineProfile) -> None:
    """Persist a single BaselineProfile to a JSON file.

    Args:
        profile: The baseline profile to write to disk.
    """
    os.makedirs(_BASELINES_DIR, exist_ok=True)
    safe_role = profile.role.replace("/", "_").replace("\\", "_")
    safe_dept = profile.department.replace("/", "_").replace("\\", "_")
    filename = f"{safe_dept}_{safe_role}.json"
    filepath = os.path.join(_BASELINES_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(_baseline_to_dict(profile), fh, indent=2, default=str)


def _baseline_to_dict(profile: BaselineProfile) -> dict[str, Any]:
    """Convert a BaselineProfile to a JSON-serializable dict.

    Args:
        profile: The baseline profile to convert.

    Returns:
        Dict with all fields ready for JSON serialization.
    """
    return {
        "role": profile.role,
        "department": profile.department,
        "avg_events_30d": profile.avg_events_30d,
        "std_events_30d": profile.std_events_30d,
        "avg_events_per_session": profile.avg_events_per_session,
        "typical_access_hours": profile.typical_access_hours,
        "typical_resources": profile.typical_resources,
        "avg_unique_resources_30d": profile.avg_unique_resources_30d,
        "std_unique_resources_30d": profile.std_unique_resources_30d,
        "p95_events_single_session": profile.p95_events_single_session,
        "p99_download_count": profile.p99_download_count,
        "month_end_multiplier": profile.month_end_multiplier,
        "quarter_end_multiplier": profile.quarter_end_multiplier,
        "computed_at": profile.computed_at,
    }


def _dict_to_baseline(data: dict[str, Any]) -> BaselineProfile:
    """Convert a JSON-loaded dict back to a BaselineProfile.

    Args:
        data: Dict loaded from a baseline JSON file.

    Returns:
        A fully constructed BaselineProfile instance.
    """
    return BaselineProfile(
        role=data["role"],
        department=data["department"],
        avg_events_30d=float(data["avg_events_30d"]),
        std_events_30d=float(data["std_events_30d"]),
        avg_events_per_session=float(data["avg_events_per_session"]),
        typical_access_hours=[int(h) for h in data["typical_access_hours"]],
        typical_resources=[str(r) for r in data["typical_resources"]],
        avg_unique_resources_30d=float(data["avg_unique_resources_30d"]),
        std_unique_resources_30d=float(data["std_unique_resources_30d"]),
        p95_events_single_session=float(data["p95_events_single_session"]),
        p99_download_count=float(data["p99_download_count"]),
        month_end_multiplier=float(data["month_end_multiplier"]),
        quarter_end_multiplier=float(data["quarter_end_multiplier"]),
        computed_at=data["computed_at"],
    )


def _make_key(department: str, role: str) -> str:
    """Build the canonical "department|role" lookup key."""
    return f"{department}|{role}"


def build_baselines(events_df: pd.DataFrame) -> dict[str, BaselineProfile]:
    """Build per-role, per-department behavioral baselines from event history.

    Groups events by (department, job_title), computes aggregate metrics over
    a trailing 12-month window, computes Finance-specific multipliers where
    applicable, persists each profile to ``data/baselines/``, and returns
    the full dictionary of profiles.

    Args:
        events_df: DataFrame of event records. Must contain at least the
            columns ``timestamp``, ``user_id``, ``department``, and
            ``job_title``. ``resource`` and ``action`` columns are used
            when available for resource and download metrics.

    Returns:
        Dict mapping ``"department|role"`` keys to BaselineProfile instances.

    Raises:
        ValueError: If ``events_df`` is empty or missing required columns.
    """
    if events_df is None or events_df.empty:
        raise ValueError("events_df is empty or None")

    required_cols = {"timestamp", "user_id", "department", "job_title"}
    missing = required_cols - set(events_df.columns)
    if missing:
        raise ValueError(f"events_df missing required columns: {missing}")

    df = _prepare_events(events_df)
    if df.empty:
        raise ValueError("No valid events remain after timestamp parsing")

    max_ts = df["_ts"].max()
    cutoff_30d = max_ts - _THIRTY_DAYS

    baselines: dict[str, BaselineProfile] = {}
    computed_at = datetime.now().isoformat()

    for (dept, role), group_df in df.groupby(["department", "job_title"]):
        dept_str = str(dept)
        role_str = str(role)

        stats_30d = _compute_30d_event_stats(group_df, cutoff_30d)
        session_stats = _compute_session_stats(group_df)
        access_hours = _compute_access_hours(group_df)
        resources = _compute_typical_resources(group_df)
        unique_stats = _compute_unique_resources_stats(group_df, cutoff_30d)
        p99_dl = _compute_p99_downloads(group_df)

        # Average daily distinct resources across the full 12-month span
        if "_ts" in group_df.columns and "resource" in group_df.columns:
            daily_res = group_df.copy()
            daily_res["_date"] = daily_res["_ts"].dt.date
            per_day = daily_res.groupby(["_date"])["resource"].nunique()
            avg_daily_res = float(per_day.mean()) if len(per_day) > 0 else 0.0
        else:
            avg_daily_res = 0.0

        if dept_str == _FINANCE_DEPT:
            multipliers = _compute_finance_multipliers(group_df)
        else:
            multipliers = {"month_end_multiplier": 1.0, "quarter_end_multiplier": 1.0}

        profile = BaselineProfile(
            role=role_str,
            department=dept_str,
            avg_events_30d=stats_30d["avg_events_30d"],
            std_events_30d=stats_30d["std_events_30d"],
            avg_events_per_session=session_stats["avg_events_per_session"],
            typical_access_hours=access_hours,
            typical_resources=resources,
            avg_unique_resources_30d=unique_stats["avg_unique_resources_30d"],
            std_unique_resources_30d=unique_stats["std_unique_resources_30d"],
            p95_events_single_session=session_stats["p95_events_single_session"],
            p99_download_count=p99_dl,
            month_end_multiplier=multipliers["month_end_multiplier"],
            quarter_end_multiplier=multipliers["quarter_end_multiplier"],
            computed_at=computed_at,
            avg_daily_unique_resources=avg_daily_res,
        )

        key = _make_key(dept_str, role_str)
        baselines[key] = profile
        _save_baseline(profile)

    return baselines


def compute_behavior_zscore(
    identity: Any,
    baseline: BaselineProfile,
) -> float:
    """Compute the z-score of a user's 30-day event count against the baseline.

    For Finance department users on day 28 or later in the month, the raw
    z-score is divided by the baseline's ``month_end_multiplier`` to account
    for the predictable volume surge during month-end close. This prevents
    false positives that would otherwise be triggered by legitimate elevated
    activity.

    Args:
        identity: An object with an ``event_count_30d`` attribute (int or
            float).
        baseline: The BaselineProfile for this identity's department and role.

    Returns:
        The z-score of the user's 30-day event count versus the baseline
        mean and standard deviation, optionally adjusted for month-end
        volume. Returns 0.0 if the baseline standard deviation is zero.
    """
    raw_z = 0.0
    if baseline.std_events_30d > _EPSILON:
        raw_z = (identity.event_count_30d - baseline.avg_events_30d) / baseline.std_events_30d

    current_day = datetime.now().day
    if (
        baseline.department == _FINANCE_DEPT
        and current_day >= _MONTH_END_DAY_THRESHOLD
        and baseline.month_end_multiplier > _EPSILON
    ):
        return raw_z / baseline.month_end_multiplier

    return raw_z


def load_baselines() -> dict[str, BaselineProfile]:
    """Load all cached baseline profiles from the ``data/baselines/`` directory.

    Returns:
        Dict mapping ``"department|role"`` keys to BaselineProfile instances.
        Returns an empty dict if no baselines have been persisted or the
        directory does not exist.
    """
    if not os.path.isdir(_BASELINES_DIR):
        return {}

    baselines: dict[str, BaselineProfile] = {}
    for entry in os.listdir(_BASELINES_DIR):
        if not entry.endswith(".json"):
            continue
        filepath = os.path.join(_BASELINES_DIR, entry)
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        profile = _dict_to_baseline(data)
        key = _make_key(profile.department, profile.role)
        baselines[key] = profile

    return baselines
