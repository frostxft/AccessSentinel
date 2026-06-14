"""Feature engineering module for AccessSentinel.

Produces a fixed-length numpy array per identity for use by the anomaly
scoring pipeline.  All thresholds are named module-level constants.
"""

import numpy as np
from datetime import datetime, timezone

from core.ingestion import DAYS_NULL_SENTINEL, IdentityRecord

# ── Module-level constants ────────────────────────────────────────────────────
DAYS_NULL_SENTINEL: float = DAYS_NULL_SENTINEL

_SENSITIVITY_MAP: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

FEATURE_NAMES: list[str] = [
    "days_since_login",
    "permission_utilization",
    "role_count",
    "system_count",
    "role_change_velocity",
    "is_privileged_flag",
    "has_owner_flag",
    "hr_mismatch_flag",
    "off_hours_access_pct",
    "geo_anomaly_flag",
    "sso_linked_flag",
    "mfa_enabled_flag",
    "sensitivity_score",
    "behavior_zscore",
    "account_age_days",
    "event_anomaly_rate",
    "failed_attempt_rate",
    "unique_resources_30d",
    "bulk_download_flag",
    "impossible_travel_flag",
    "after_hours_event_pct",
    "peer_deviation_score",
    "avg_time_between_events_h",
]


def _compute_user_features(identity: IdentityRecord) -> np.ndarray:
    """Compute user-level features (columns 0-12) for a single identity.

    Args:
        identity: An IdentityRecord instance to featurize.

    Returns:
        np.ndarray of shape ``(13,)`` and dtype float64.
    """
    now = datetime.now(timezone.utc)
    if identity.last_login is not None:
        days_since = (now - identity.last_login).days
    else:
        days_since = DAYS_NULL_SENTINEL

    total_perms = len(identity.permissions)
    if total_perms > 0:
        perm_util = min(1.0, identity.unique_resources_accessed / total_perms)
    else:
        perm_util = 0.0

    role_velocity = identity.role_changes_90d / 90.0

    hr_mismatch = 0
    if identity.employment_status.lower().startswith("terminat"):
        hr_mismatch = 1

    sensitivity = _SENSITIVITY_MAP.get(
        identity.resource_sensitivity.lower(), 1
    )

    return np.array([
        float(days_since),
        float(perm_util),
        float(len(identity.roles)),
        float(identity.systems_count),
        float(role_velocity),
        float(int(identity.is_privileged)),
        float(int(bool(identity.owner_id))),
        float(hr_mismatch),
        float(identity.off_hours_access_pct),
        float(int(identity.geo_anomaly)),
        float(int(identity.sso_linked)),
        float(int(identity.mfa_enabled)),
        float(sensitivity),
    ], dtype=np.float64)


def _compute_event_features(
    identity: IdentityRecord,
    baseline: dict,
    peers: dict | None,
) -> np.ndarray:
    """Compute event-derived features (columns 13-22) for a single identity.

    Args:
        identity: An IdentityRecord instance to featurize.
        baseline: Per-user stats keyed by user_id.  Each value must be a
            dict with ``mean`` and ``std`` for behavior z-score.
        peers: Optional per-department stats keyed by department name.
            Each value must be a dict with ``mean`` and ``std``.

    Returns:
        np.ndarray of shape ``(10,)`` and dtype float64.
    """
    now = datetime.now(timezone.utc)
    if identity.created_at is not None:
        account_age = (now - identity.created_at).days
    else:
        account_age = 0

    user_bl = baseline.get(identity.user_id, {})
    b_mean = float(user_bl.get("mean", 0.0))
    b_std = float(user_bl.get("std", 1.0))
    behavior_z = (identity.event_count_30d - b_mean) / b_std if b_std > 0 else 0.0

    if identity.event_count_90d > 0:
        anomaly_rate = identity.anomaly_event_count / identity.event_count_90d
        failed_rate = identity.failed_attempt_count / identity.event_count_90d
    else:
        anomaly_rate = 0.0
        failed_rate = 0.0

    peer_score = 0.0
    if peers is not None:
        dept_stats = peers.get(identity.department)
        if dept_stats is not None:
            d_mean = float(dept_stats.get("mean", 0.0))
            d_std = float(dept_stats.get("std", 1.0))
            if d_std > 0:
                peer_score = (identity.event_count_30d - d_mean) / d_std

    return np.array([
        float(behavior_z),
        float(account_age),
        float(anomaly_rate),
        float(failed_rate),
        float(identity.unique_resources_accessed),
        float(int(identity.bulk_download_detected)),
        float(int(identity.impossible_travel_detected)),
        float(identity.off_hours_event_pct),
        float(peer_score),
        float(identity.avg_time_between_events_hours),
    ], dtype=np.float64)


def extract_features(
    identities: list[IdentityRecord],
    baselines: dict,
    peers: dict | None = None,
) -> np.ndarray:
    """Extract a fixed-length feature vector for each identity.

    Produces an ``(n_identities, 23)`` array where columns 0-12 hold
    user-level features and columns 13-22 hold event-derived features.
    Column order matches :data:`FEATURE_NAMES`.

    Args:
        identities: IdentityRecord instances to featurize.
        baselines: Mapping of ``user_id`` -> ``{"mean": float, "std": float}``
            for computing the behavior z-score (column 13).
        peers: Optional mapping of ``department`` -> ``{"mean": float, "std": float}``
            for computing the peer-deviation score (column 21).

    Returns:
        np.ndarray of shape ``(len(identities), 23)`` and dtype float64.
    """
    if not identities:
        return np.empty((0, 23), dtype=np.float64)

    rows: list[np.ndarray] = []
    for identity in identities:
        user_feats = _compute_user_features(identity)
        event_feats = _compute_event_features(identity, baselines, peers)
        rows.append(np.concatenate([user_feats, event_feats]))

    return np.array(rows, dtype=np.float64)
