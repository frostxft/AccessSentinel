"""Rules engine for AccessSentinel.

Applies 11 named detection rules to identity records, evaluating privilege
staleness, account orphan status, over-privilege, shadow admin, privilege
creep, service account abuse, credential sprawl, impossible travel, excessive
access, bulk download, and segregation-of-duty violations. Each rule returns a
frozen RuleResult with severity and evidence.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.ingestion import IdentityRecord

# ── Threshold constants ────────────────────────────────────────────────────────
STALE_DAYS_THRESHOLD: int = 30
OVERPRIV_UTILIZATION_LIMIT: float = 0.20
CREEP_CHANGE_THRESHOLD: int = 3
SPRAWL_SYSTEM_THRESHOLD: int = 5
EXCESSIVE_ACCESS_RATE: float = 0.30
LATERAL_MOVEMENT_SYSTEM_THRESHOLD: int = 20
LATERAL_MOVEMENT_PEER_MULTIPLIER: float = 3.0

# ── Segregation-of-duty conflict pairs (loaded from config) ──────────────────
import json
import os

_SOD_CONFIG_PATH: str = os.path.join(
    os.path.dirname(__file__), "..", "data", "sod_conflicts.json"
)
_sod_conflicts_data: list[dict] = []
try:
    with open(_SOD_CONFIG_PATH, "r", encoding="utf-8") as fh:
        _sod_conflicts_data = json.load(fh).get("conflicts", [])
except (FileNotFoundError, json.JSONDecodeError):
    pass

SOD_CONFLICT_PAIRS: list[tuple[str, str]] = [
    (entry["role_a"].lower(), entry["role_b"].lower())
    for entry in _sod_conflicts_data
]

SOD_CONFLICT_DETAILS: dict[tuple[str, str], dict] = {
    (entry["role_a"].lower(), entry["role_b"].lower()): entry
    for entry in _sod_conflicts_data
}


@dataclass(frozen=True)
class RuleResult:
    """Immutable result of a single rule evaluation against an identity.

    Attributes:
        rule_id: Short canonical name of the rule (e.g. ``"STALE_PRIVILEGED"``).
        severity: One of ``"CRITICAL"``, ``"HIGH"``, or ``"MEDIUM"``.
        triggered: Whether the rule conditions were satisfied.
        evidence_text: Human-readable description of the finding.
        suppressed_by: Optional identifier of a suppression rule, or ``None``.
    """

    rule_id: str
    severity: str
    triggered: bool
    evidence_text: str
    suppressed_by: str | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────


def _compute_days_since_login(identity: IdentityRecord) -> float:
    """Return days since last login, or a large sentinel when ``last_login`` is ``None``."""
    if identity.last_login is None:
        return 999.0
    return (datetime.now(timezone.utc) - identity.last_login).days


def _permission_utilization(identity: IdentityRecord) -> float:
    """Return the permission-utilization ratio for the identity.

    Falls back to the ``permission_utilization`` attribute when present;
    otherwise computes a naïve ratio from event counts and total permissions.
    """
    raw = getattr(identity, "permission_utilization", None)
    if raw is not None:
        return float(raw)
    perm_count = len(identity.permissions)
    if perm_count == 0:
        return 0.0
    return identity.unique_resources_accessed / perm_count


def _event_anomaly_rate(identity: IdentityRecord) -> float:
    """Return the fraction of anomaly events among recent events."""
    total = identity.event_count_30d
    if total == 0:
        return 0.0
    return identity.anomaly_event_count / total


# ── Individual rule functions ─────────────────────────────────────────────────


def _rule_stale_privileged(
    identity: IdentityRecord,
    days_since_login: float,
) -> RuleResult:
    """STALE_PRIVILEGED: Privileged account with no login beyond threshold."""
    triggered = days_since_login > STALE_DAYS_THRESHOLD and identity.is_privileged
    return RuleResult(
        rule_id="STALE_PRIVILEGED",
        severity="HIGH",
        triggered=triggered,
        evidence_text=(
            f"Privileged account stale for {days_since_login:.0f} days"
            if triggered
            else ""
        ),
    )


def _rule_orphaned_account(identity: IdentityRecord) -> RuleResult:
    """ORPHANED_ACCOUNT: Missing owner or terminated employee with active account."""
    has_no_owner = not identity.owner_id
    terminated_active = identity.employment_status.lower() == "terminated"
    triggered = has_no_owner or terminated_active
    if has_no_owner:
        evidence = "Account has no owner"
    elif terminated_active:
        evidence = "Terminated employee with active account"
    else:
        evidence = ""
    return RuleResult(
        rule_id="ORPHANED_ACCOUNT",
        severity="CRITICAL",
        triggered=triggered,
        evidence_text=evidence,
    )


def _rule_over_privileged(identity: IdentityRecord) -> RuleResult:
    """OVER_PRIVILEGED: Permission utilization below the minimum threshold."""
    utilization = _permission_utilization(identity)
    triggered = utilization < OVERPRIV_UTILIZATION_LIMIT
    return RuleResult(
        rule_id="OVER_PRIVILEGED",
        severity="HIGH",
        triggered=triggered,
        evidence_text=(
            f"Permission utilization {utilization:.1%} below {OVERPRIV_UTILIZATION_LIMIT:.0%}"
            if triggered
            else ""
        ),
    )


def _rule_shadow_admin(identity: IdentityRecord) -> RuleResult:
    """SHADOW_ADMIN: Non-admin account type that holds admin permissions."""
    is_admin_account = identity.account_type.lower() == "admin"
    has_admin_perm = any("admin" in p.lower() for p in identity.permissions)
    triggered = (not is_admin_account) and has_admin_perm
    return RuleResult(
        rule_id="SHADOW_ADMIN",
        severity="CRITICAL",
        triggered=triggered,
        evidence_text="Non-admin account_type with admin permissions" if triggered else "",
    )


def _rule_privilege_creep(identity: IdentityRecord) -> RuleResult:
    """PRIVILEGE_CREEP: Role changes in past 90 days meet or exceed threshold."""
    n = identity.role_changes_90d
    triggered = n >= CREEP_CHANGE_THRESHOLD
    return RuleResult(
        rule_id="PRIVILEGE_CREEP",
        severity="MEDIUM",
        triggered=triggered,
        evidence_text=f"{n} role changes in 90 days" if triggered else "",
    )


def _rule_service_acct_abuse(identity: IdentityRecord) -> RuleResult:
    """SERVICE_ACCT_ABUSE: Service account with interactive login enabled."""
    triggered = (
        identity.account_type.lower() == "service" and identity.interactive_login
    )
    return RuleResult(
        rule_id="SERVICE_ACCT_ABUSE",
        severity="CRITICAL",
        triggered=triggered,
        evidence_text="Service account with interactive login capability" if triggered else "",
    )


def _rule_credential_sprawl(identity: IdentityRecord) -> RuleResult:
    """CREDENTIAL_SPRAWL: Many systems accessed without SSO linkage."""
    n = identity.systems_count
    triggered = n >= SPRAWL_SYSTEM_THRESHOLD and not identity.sso_linked
    return RuleResult(
        rule_id="CREDENTIAL_SPRAWL",
        severity="MEDIUM",
        triggered=triggered,
        evidence_text=f"{n} systems without SSO" if triggered else "",
    )


def _rule_cross_system_travel(identity: IdentityRecord) -> RuleResult:
    """CROSS_SYSTEM_TRAVEL: Impossible travel corroborated across different source systems."""
    triggered = identity.cross_system_impossible_travel
    return RuleResult(
        rule_id="CROSS_SYSTEM_TRAVEL",
        severity="CRITICAL",
        triggered=triggered,
        evidence_text=(
            "Cross-system impossible travel detected — authentications from "
            "different source systems in geographically distant regions"
            if triggered
            else ""
        ),
    )


def _rule_impossible_travel(identity: IdentityRecord) -> RuleResult:
    """IMPOSSIBLE_TRAVEL: Impossible-travel anomaly flag is set."""
    triggered = identity.impossible_travel_detected
    return RuleResult(
        rule_id="IMPOSSIBLE_TRAVEL",
        severity="CRITICAL",
        triggered=triggered,
        evidence_text="Impossible travel detected" if triggered else "",
    )


def _rule_excessive_access(
    identity: IdentityRecord,
    baseline: Any,
) -> RuleResult:
    """EXCESSIVE_ACCESS: Resource and anomaly metrics exceed baselines."""
    if baseline is None:
        return RuleResult(
            rule_id="EXCESSIVE_ACCESS",
            severity="HIGH",
            triggered=False,
            evidence_text="",
        )
    res_count = identity.unique_resources_accessed
    anomaly_rate = _event_anomaly_rate(identity)
    triggered = (
        res_count > baseline.p95_events_single_session
        and anomaly_rate > EXCESSIVE_ACCESS_RATE
    )
    return RuleResult(
        rule_id="EXCESSIVE_ACCESS",
        severity="HIGH",
        triggered=triggered,
        evidence_text=(
            f"{res_count} resources in 30d exceeds baseline p95"
            if triggered
            else ""
        ),
    )


def _rule_lateral_movement_spike(
    identity: IdentityRecord,
    baseline_profile,
) -> RuleResult:
    """LATERAL_MOVEMENT_SPIKE: Unusually high distinct resources in one day.

    Compares the identity's max_resources_in_single_day against a peer
    baseline (avg_daily_unique_resources) multiplied by a guard factor.
    """
    max_res = identity.max_resources_in_single_day
    if max_res < LATERAL_MOVEMENT_SYSTEM_THRESHOLD:
        return RuleResult(
            rule_id="LATERAL_MOVEMENT_SPIKE",
            severity="CRITICAL",
            triggered=False,
            evidence_text="",
        )
    if baseline_profile is None:
        return RuleResult(
            rule_id="LATERAL_MOVEMENT_SPIKE",
            severity="CRITICAL",
            triggered=False,
            evidence_text="",
        )
    peer_avg = getattr(baseline_profile, "avg_daily_unique_resources", 0.0)
    threshold = peer_avg * LATERAL_MOVEMENT_PEER_MULTIPLIER
    triggered = max_res >= threshold if threshold > 0 else False
    return RuleResult(
        rule_id="LATERAL_MOVEMENT_SPIKE",
        severity="CRITICAL",
        triggered=triggered,
        evidence_text=(
            f"Accessed {max_res} distinct resources in a single day. "
            f"Peer ({identity.department}) average: {peer_avg:.1f}."
            if triggered
            else ""
        ),
    )


def _rule_bulk_download(identity: IdentityRecord) -> RuleResult:
    """BULK_DOWNLOAD: Bulk download detected with majority after-hours activity."""
    triggered = (
        identity.bulk_download_detected
        and identity.off_hours_event_pct > 0.50
    )
    return RuleResult(
        rule_id="BULK_DOWNLOAD",
        severity="CRITICAL",
        triggered=triggered,
        evidence_text=(
            f"Bulk download detected with {identity.off_hours_event_pct:.0%} after-hours activity"
            if triggered
            else ""
        ),
    )


def _rule_sod_violation(identity: IdentityRecord) -> RuleResult:
    """SOD_VIOLATION: Role combination matches a known segregation-of-duty conflict.

    Checks identity.roles and identity.permissions against the loaded
    conflict-pair registry.  Reports the highest-severity match; if
    multiple matches exist all conflict_ids appear in the evidence text.
    """
    role_set = {r.lower() for r in identity.roles}
    perm_set = {p.lower() for p in identity.permissions}
    combined = role_set | perm_set
    matches: list[dict] = []
    for (role_a, role_b), detail in SOD_CONFLICT_DETAILS.items():
        if role_a in combined and role_b in combined:
            matches.append(detail)
    if not matches:
        return RuleResult(
            rule_id="SOD_VIOLATION",
            severity="CRITICAL",
            triggered=False,
            evidence_text="",
        )
    # Report highest severity, all conflict IDs
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    matches.sort(key=lambda m: sev_order.get(m.get("severity", "LOW"), 3))
    worst = matches[0]
    ids = ", ".join(m["conflict_id"] for m in matches)
    descs = "; ".join(
        f"{m['conflict_id']}: {m['description']} (Conflict: {m['role_a']} + {m['role_b']}, ref {m['compliance']})"
        for m in matches
    )
    return RuleResult(
        rule_id="SOD_VIOLATION",
        severity=worst.get("severity", "CRITICAL"),
        triggered=True,
        evidence_text=descs,
    )


# ── Public API ─────────────────────────────────────────────────────────────────


def evaluate_rules(
    identity: IdentityRecord,
    baseline: Any = None,
) -> list[RuleResult]:
    """Evaluate all detection rules against a single identity record.

    Args:
        identity: The :class:`IdentityRecord` to inspect.
        baseline: Optional baseline profile with per-user statistical norms.
            When ``None`` the ``EXCESSIVE_ACCESS`` rule is skipped
            (``triggered=False``).

    Returns:
        A list of 11 :class:`RuleResult` instances, one per rule.
    """
    days_since_login = _compute_days_since_login(identity)
    return [
        _rule_stale_privileged(identity, days_since_login),
        _rule_orphaned_account(identity),
        _rule_over_privileged(identity),
        _rule_shadow_admin(identity),
        _rule_privilege_creep(identity),
        _rule_service_acct_abuse(identity),
        _rule_credential_sprawl(identity),
        _rule_impossible_travel(identity),
        _rule_cross_system_travel(identity),
        _rule_excessive_access(identity, baseline),
        _rule_lateral_movement_spike(identity, baseline),
        _rule_bulk_download(identity),
        _rule_sod_violation(identity),
    ]
