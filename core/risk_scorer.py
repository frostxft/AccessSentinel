"""Risk scorer for AccessSentinel.

Computes final identity risk scores using a weighted multi-factor algorithm
that combines detection rule weights, peer deviation, context signals, and
blast-radius amplification.
"""

from dataclasses import dataclass, field
from typing import Any

from core.rules_engine import RuleResult
from core.role_exceptions import resolve_role_exceptions
from core.remediation import RULE_TO_COMPLIANCE

# ── Module-level weight constants ────────────────────────────────────────────
WEIGHT_ADMIN_ROLE = 30
WEIGHT_ORPHANED = 25
WEIGHT_STALE = 20
WEIGHT_SENSITIVE_RESOURCE = 15
WEIGHT_ANOMALOUS_ACCESS = 10
WEIGHT_ROLE_STACK = 5
BLAST_RADIUS_MULTIPLIER = 1.20
BLAST_RADIUS_THRESHOLD = 3
BLAST_RADIUS_SYSTEM_POINTS = 5
BLAST_RADIUS_CRITICAL_RESOURCE_POINTS = 15

# ── Rule-to-weight mapping ───────────────────────────────────────────────────
RULE_WEIGHTS: dict[str, int] = {
    "STALE_PRIVILEGED": WEIGHT_STALE + WEIGHT_ADMIN_ROLE,       # 50
    "ORPHANED_ACCOUNT": WEIGHT_ORPHANED,                         # 25
    "OVER_PRIVILEGED": WEIGHT_SENSITIVE_RESOURCE + WEIGHT_ANOMALOUS_ACCESS,  # 25
    "SHADOW_ADMIN": WEIGHT_ADMIN_ROLE + WEIGHT_ORPHANED,         # 55
    "PRIVILEGE_CREEP": WEIGHT_ROLE_STACK + WEIGHT_ADMIN_ROLE,    # 35
    "SERVICE_ACCT_ABUSE": WEIGHT_ADMIN_ROLE + WEIGHT_SENSITIVE_RESOURCE,  # 45
    "CREDENTIAL_SPRAWL": WEIGHT_SENSITIVE_RESOURCE + WEIGHT_ANOMALOUS_ACCESS,  # 25
    "IMPOSSIBLE_TRAVEL": WEIGHT_ADMIN_ROLE + WEIGHT_ORPHANED,    # 55
    "EXCESSIVE_ACCESS": WEIGHT_ANOMALOUS_ACCESS + WEIGHT_SENSITIVE_RESOURCE,  # 25
    "BULK_DOWNLOAD": WEIGHT_ADMIN_ROLE + WEIGHT_SENSITIVE_RESOURCE,  # 45
    "SOD_VIOLATION": WEIGHT_ADMIN_ROLE + WEIGHT_ORPHANED,        # 55
}

# ── Severity-to-remediation mapping ──────────────────────────────────────────
_REMEDIATION: dict[str, str] = {
    "CRITICAL": "disable the account within 4 hours and notify Security Operations",
    "HIGH": "review and restrict permissions within 24 hours",
    "MEDIUM": "schedule an access review within 7 days",
    "LOW": "continue routine monitoring",
}

_ESCALATION: dict[str, str] = {
    "CRITICAL": "Security manager review required — escalate to CISO within 1 hour",
    "HIGH": "Security team review required within SLA — notify team lead",
    "MEDIUM": "Identity governance team review — schedule within 5 business days",
    "LOW": "Routine monitoring — no escalation required",
}

# ── Dataclass ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RiskResult:
    """Final risk assessment for a single identity.

    Attributes:
        score: Integer risk score from 0 to 100.
        tier: Risk tier (``CRITICAL``, ``HIGH``, ``MEDIUM``, or ``LOW``).
        contributing_factors: Rule IDs that triggered and were not suppressed.
        suppressed_factors: Rule IDs that triggered but were suppressed.
        context_signals: The context signals applied during scoring.
        blast_radius_applied: Whether the blast-radius multiplier was applied.
        confidence: Confidence value between 0.0 and 1.0.
        behavior_zscore: The raw behavior z-score used in the calculation.
        sequence_risk: Placeholder for future sequence risk analysis.
        risk_narrative: Human-readable summary of the risk assessment.
        role_exceptions: Senior-role exceptions applied to soften the score.
    """

    score: int
    tier: str
    contributing_factors: list[str]
    suppressed_factors: list[str]
    context_signals: list[Any]
    blast_radius_applied: bool
    confidence: float
    behavior_zscore: float
    sequence_risk: Any | None
    risk_narrative: str
    role_exceptions: list[Any] = field(default_factory=list)
    compliance_gaps: list[Any] = field(default_factory=list)
    next_escalation: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _assign_tier(score: int) -> str:
    """Assign a risk tier based on the numeric score.

    Args:
        score: Integer risk score between 0 and 100.

    Returns:
        One of ``CRITICAL``, ``HIGH``, ``MEDIUM``, or ``LOW``.
    """
    if score >= 80:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def _is_suppressed(rule: RuleResult, suppressed_rule_ids: set[str]) -> bool:
    """Determine whether a triggered rule is suppressed.

    Args:
        rule: The rule result to check.
        suppressed_rule_ids: Set of rule IDs suppressed by context signals.

    Returns:
        True if the rule should be considered suppressed.
    """
    if rule.suppressed_by is not None:
        return True
    return rule.rule_id in suppressed_rule_ids


def _top_severity(rules: list[RuleResult]) -> str:
    """Return the highest severity among a list of rules.

    Args:
        rules: List of rule results to inspect.

    Returns:
        The highest severity string found, defaulting to ``"LOW"``.
    """
    severity_order: dict[str, int] = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    best = "LOW"
    best_val = 1
    for rule in rules:
        val = severity_order.get(rule.severity, 0)
        if val > best_val:
            best_val = val
            best = rule.severity
    return best


def _build_suppression_explanations(
    triggered_suppressed: list[RuleResult],
    context_signals: list[Any],
) -> dict[str, str]:
    """Build a mapping from suppressed rule ID to explanation text.

    Args:
        triggered_suppressed: Triggered rules that are suppressed.
        context_signals: Context signals that may provide explanations.

    Returns:
        Dictionary keyed by rule_id with explanation strings.
    """
    signal_suppression: dict[str, str] = {}
    for sig in context_signals:
        if hasattr(sig, "rules_suppressed") and hasattr(sig, "explanation"):
            explanation: str = sig.explanation
            for r_id in sig.rules_suppressed:
                signal_suppression[r_id] = explanation

    explanations: dict[str, str] = {}
    for rule in triggered_suppressed:
        if rule.rule_id in signal_suppression:
            explanations[rule.rule_id] = signal_suppression[rule.rule_id]
        elif rule.suppressed_by:
            explanations[rule.rule_id] = rule.suppressed_by
    return explanations


def _build_narrative(
    username: str,
    score: int,
    tier: str,
    triggered_non_suppressed: list[RuleResult],
    triggered_suppressed: list[RuleResult],
    context_signals: list[Any],
    confidence: float,
    role_exceptions: list[Any],
) -> str:
    """Build a deterministic risk narrative paragraph.

    Args:
        username: The account username.
        score: Final integer risk score.
        tier: Assigned risk tier.
        triggered_non_suppressed: Triggered rules that were not suppressed.
        triggered_suppressed: Triggered rules that were suppressed.
        context_signals: Context signals applied during scoring.
        confidence: Computed confidence value.
        role_exceptions: Senior-role exceptions applied during scoring.

    Returns:
        A single-paragraph risk narrative of 80-150 words.
    """
    if not triggered_non_suppressed:
        evidence_block = (
            "This account exhibits no anomalous behavior and aligns "
            "with its peer group baseline across all monitored "
            "dimensions including access patterns, login frequency, "
            "resource utilization, and geographic activity. No "
            "MITRE ATT&CK techniques were triggered during this "
            "evaluation window."
        )
    else:
        top_two = sorted(
            triggered_non_suppressed,
            key=lambda r: RULE_WEIGHTS.get(r.rule_id, 0),
            reverse=True,
        )[:2]
        details = []
        for r in top_two:
            severity_label = getattr(r, "severity", "UNKNOWN")
            evidence = r.evidence_text or f"Rule {r.rule_id} was triggered"
            details.append(
                f"The {r.rule_id} rule ({severity_label} severity) was "
                f"triggered because {evidence}. This finding indicates "
                f"potential risk to the organization's identity security "
                f"posture"
            )
        evidence_block = ". ".join(details) + "."

    suppression_block = ""
    if triggered_suppressed:
        explanations = _build_suppression_explanations(
            triggered_suppressed, context_signals
        )
        sentences: list[str] = []
        for rule in triggered_suppressed:
            reason = explanations.get(rule.rule_id, "context-based suppression")
            sentences.append(
                f"The {rule.rule_id} rule was also triggered but was "
                f"suppressed from scoring because {reason}. Suppressed "
                f"rules are preserved in the audit trail for review."
            )
        suppression_block = " " + " ".join(sentences)
    else:
        suppression_block = (
            " No rules were suppressed during this evaluation, meaning "
            "all detected signals contributed to the final risk score."
        )

    top_sev = (
        _top_severity(triggered_non_suppressed)
        if triggered_non_suppressed
        else "LOW"
    )
    remediation = _REMEDIATION.get(top_sev, _REMEDIATION["LOW"])
    confidence_pct = round(confidence * 100)

    narrative = (
        f"Account {username} has a risk score of {score} out of 100 "
        f"({tier}). {evidence_block}{suppression_block} Recommended "
        f"action: {remediation}. This assessment was generated using "
        f"AccessSentinel's multi-factor risk engine combining behavioral "
        f"baseline analysis, peer group comparison, and contextual "
        f"signal resolution. Confidence: {confidence_pct} percent."
    )

    if role_exceptions:
        for exc in role_exceptions:
            narrative += f" {exc.narrative_override}"

    return narrative


# ── Public API ───────────────────────────────────────────────────────────────


def compute_risk_score(
    identity: Any,
    rules: list[RuleResult],
    context_signals: list[Any],
    peer_deviation_score: float = 0.0,
    behavior_zscore: float = 0.0,
) -> RiskResult:
    """Compute the final identity risk score.

    Args:
        identity: The identity record being scored (must have a ``username``
            attribute).
        rules: List of :class:`RuleResult` instances from the rules engine.
        context_signals: List of context signal objects with
            ``score_adjustment`` and ``rules_suppressed`` attributes.
        peer_deviation_score: Peer-group deviation metric (default 0.0).
        behavior_zscore: Raw behavior z-score (default 0.0).

    Returns:
        A :class:`RiskResult` containing score, tier, and narrative.
    """
    # Collect suppressed rule IDs from context signals
    suppressed_ids: set[str] = set()
    for sig in context_signals:
        if hasattr(sig, "rules_suppressed"):
            for r_id in sig.rules_suppressed:
                suppressed_ids.add(r_id)

    # Partition triggered rules
    triggered = [r for r in rules if r.triggered]
    triggered_non_suppressed = [
        r for r in triggered if not _is_suppressed(r, suppressed_ids)
    ]
    triggered_suppressed = [
        r for r in triggered if _is_suppressed(r, suppressed_ids)
    ]

    # Step 1: base_score = sum of weights for triggered AND non-suppressed
    base_score = sum(
        RULE_WEIGHTS.get(r.rule_id, 0) for r in triggered_non_suppressed
    )

    # Step 2: peer_component = min(10, peer_deviation_score * 5.0)
    peer_component = min(10.0, peer_deviation_score * 5.0)

    # Step 3: adjusted_score = base_score + peer_component
    adjusted_score = base_score + peer_component

    # Step 4: context_delta = sum of ContextSignal.score_adjustment values
    context_delta = 0
    for sig in context_signals:
        if hasattr(sig, "score_adjustment"):
            context_delta += sig.score_adjustment

    # Step 5: adjusted_score = adjusted_score + context_delta
    adjusted_score = adjusted_score + context_delta

    # ── Role exception delta (between Steps 5 and 6) ──
    role_exceptions = resolve_role_exceptions(identity)
    role_exception_delta = sum(e.score_adjustment for e in role_exceptions)
    adjusted_score = adjusted_score + role_exception_delta

    # Step 6: blast radius multiplier
    non_suppressed_count = len(triggered_non_suppressed)
    blast_radius_applied = non_suppressed_count >= BLAST_RADIUS_THRESHOLD
    if blast_radius_applied:
        adjusted_score = adjusted_score * BLAST_RADIUS_MULTIPLIER

    # Step 7: final_score = max(0, min(100, round(adjusted_score)))
    final_score = max(0, min(100, round(adjusted_score)))

    # Tier assignment
    tier = _assign_tier(final_score)

    # Confidence calculation
    confidence = min(
        0.99, 0.50 + (non_suppressed_count * 0.08) + (behavior_zscore * 0.05)
    )

    # Contributing and suppressed factor lists
    contributing_factors = [r.rule_id for r in triggered_non_suppressed]
    suppressed_factors = [r.rule_id for r in triggered_suppressed]

    # Compliance gaps from triggered AND non-suppressed rules only
    compliance_gaps = [
        RULE_TO_COMPLIANCE[rule_id]
        for rule_id in contributing_factors
        if rule_id in RULE_TO_COMPLIANCE
    ]

    # Build narrative
    username = getattr(identity, "username", "unknown")
    risk_narrative = _build_narrative(
        username=username,
        score=final_score,
        tier=tier,
        triggered_non_suppressed=triggered_non_suppressed,
        triggered_suppressed=triggered_suppressed,
        context_signals=context_signals,
        confidence=confidence,
        role_exceptions=role_exceptions,
    )

    return RiskResult(
        score=final_score,
        tier=tier,
        contributing_factors=contributing_factors,
        suppressed_factors=suppressed_factors,
        context_signals=context_signals,
        blast_radius_applied=blast_radius_applied,
        confidence=confidence,
        behavior_zscore=behavior_zscore,
        sequence_risk=None,
        risk_narrative=risk_narrative,
        role_exceptions=role_exceptions,
        compliance_gaps=compliance_gaps,
    )


def compute_blast_radius(
    identity: Any,
    all_identities: list[Any],
    privilege_graph: Any,
) -> dict:
    """Estimate the blast radius if this identity's credentials are compromised.

    Args:
        identity: The target identity.
        all_identities: All identity records.
        privilege_graph: A networkx DiGraph with user->role->system->resource edges.

    Returns:
        Dict with systems_at_risk, downstream_users, sensitive_resources,
        estimated_impact_score, and narrative.
    """
    import networkx as nx
    identity_id = getattr(identity, "user_id", "")
    user_node = f"user:{identity_id}"

    # Reachable nodes from this user
    reachable: set[str] = set()
    if isinstance(privilege_graph, nx.DiGraph) and user_node in privilege_graph:
        reachable = nx.descendants(privilege_graph, user_node) | {user_node}

    systems = {n for n in reachable if n.startswith("system:")}
    resources = {n for n in reachable if n.startswith("resource:")}
    # Sensitive/critical resources
    sensitive = {
        n for n in resources
        if getattr(identity, "resource_sensitivity", "") == "critical"
    } or resources

    # Downstream users: other users sharing any system
    downstream: set[str] = set()
    for other in all_identities:
        other_id = getattr(other, "user_id", "")
        if other_id == identity_id:
            continue
        other_node = f"user:{other_id}"
        if isinstance(privilege_graph, nx.DiGraph) and other_node in privilege_graph:
            other_reachable = nx.descendants(privilege_graph, other_node)
            if systems & {n for n in other_reachable if n.startswith("system:")}:
                downstream.add(other_id)

    score = min(100,
        len(systems) * BLAST_RADIUS_SYSTEM_POINTS
        + len(sensitive) * BLAST_RADIUS_CRITICAL_RESOURCE_POINTS
    )
    top_systems = sorted(systems)[:2]
    narrative = (
        f"If this account were compromised, an attacker would gain access to "
        f"{len(systems)} systems including {', '.join(top_systems) or 'none'}, "
        f"with {len(downstream)} downstream users sharing overlapping access. "
        f"{len(sensitive)} critical resources are reachable."
    )
    return {
        "identity_id": identity_id,
        "systems_at_risk": sorted(systems),
        "downstream_users": sorted(downstream),
        "sensitive_resources": sorted(sensitive),
        "estimated_impact_score": score,
        "narrative": narrative,
    }
