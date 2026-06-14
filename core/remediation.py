"""Remediation engine for AccessSentinel.

Generates prioritized remediation actions based on triggered detection rules
and identity risk scores, producing human-readable guidance and source-system-
aware CLI commands.
"""

from dataclasses import dataclass, replace
from typing import Any

from core.ingestion import IdentityRecord
from core.rules_engine import RuleResult

# ── SLA constants (hours) ─────────────────────────────────────────────────────
DISABLE_ACCOUNT_HOURS: int = 4
REVOKE_ROLE_HOURS: int = 8
ROTATE_CREDENTIALS_HOURS: int = 4
RESTRICT_SCOPE_HOURS: int = 16
FLAG_HR_REVIEW_HOURS: int = 48
ENABLE_MFA_HOURS: int = 24
LINK_SSO_HOURS: int = 72
VERIFY_WITH_MANAGER_HOURS: int = 24
BLOCK_INTERACTIVE_LOGIN_HOURS: int = 8
SCOPE_TO_RESOURCES_HOURS: int = 24

# ── Action-type → SLA mapping ─────────────────────────────────────────────────
_ACTION_SLA: dict[str, int] = {
    "DISABLE_ACCOUNT": DISABLE_ACCOUNT_HOURS,
    "REVOKE_ROLE": REVOKE_ROLE_HOURS,
    "ROTATE_CREDENTIALS": ROTATE_CREDENTIALS_HOURS,
    "RESTRICT_SCOPE": RESTRICT_SCOPE_HOURS,
    "FLAG_HR_REVIEW": FLAG_HR_REVIEW_HOURS,
    "ENABLE_MFA": ENABLE_MFA_HOURS,
    "LINK_SSO": LINK_SSO_HOURS,
    "VERIFY_WITH_MANAGER": VERIFY_WITH_MANAGER_HOURS,
    "BLOCK_INTERACTIVE_LOGIN": BLOCK_INTERACTIVE_LOGIN_HOURS,
    "SCOPE_TO_RESOURCES": SCOPE_TO_RESOURCES_HOURS,
}

# ── Rule → action-type mapping ───────────────────────────────────────────────
_RULE_ACTION_MAP: dict[str, str] = {
    "STALE_PRIVILEGED": "DISABLE_ACCOUNT",
    "ORPHANED_ACCOUNT": "DISABLE_ACCOUNT",
    "OVER_PRIVILEGED": "RESTRICT_SCOPE",
    "SHADOW_ADMIN": "REVOKE_ROLE",
    "PRIVILEGE_CREEP": "REVOKE_ROLE",
    "SERVICE_ACCT_ABUSE": "BLOCK_INTERACTIVE_LOGIN",
    "CREDENTIAL_SPRAWL": "LINK_SSO",
    "IMPOSSIBLE_TRAVEL": "ROTATE_CREDENTIALS",
    "EXCESSIVE_ACCESS": "SCOPE_TO_RESOURCES",
    "BULK_DOWNLOAD": "DISABLE_ACCOUNT",
    "SOD_VIOLATION": "REVOKE_ROLE",
}

# ── Severity ordering and risk reduction ─────────────────────────────────────
_SEVERITY_ORDER: dict[str, int] = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
}

_SEVERITY_RISK_REDUCTION: dict[str, int] = {
    "CRITICAL": 30,
    "HIGH": 20,
    "MEDIUM": 10,
    "LOW": 5,
}

# ── Source-system-specific disable commands ──────────────────────────────────
_SOURCE_DISABLE_COMMANDS: dict[str, str] = {
    "AD": "Disable-ADAccount -Identity {username}",
    "AzureAD": "az ad user update --id {email} --account-enabled false",
    "AWS_IAM": "aws iam delete-login-profile --user-name {username}",
    "GCP": "gcloud iam service-accounts disable {email}",
    "Okta": "okta users deactivate --login {email}",
}

# ── Generic (any source system) commands ─────────────────────────────────────
_GENERIC_COMMANDS: dict[str, str] = {
    "REVOKE_ROLE": "Remove role assignments for {username} on {source_system}",
    "ROTATE_CREDENTIALS": (
        "Force password reset: Set-ADUser -Identity {username}"
        " -ChangePasswordAtLogon $true"
    ),
    "RESTRICT_SCOPE": (
        "Limit {username} to job-specific resources on {source_system}"
    ),
    "ENABLE_MFA": "Enforce MFA enrollment for {username} on {source_system}",
    "LINK_SSO": "Link {username} to SSO provider on {source_system}",
    "BLOCK_INTERACTIVE_LOGIN": (
        "Block interactive login for service account {username}"
        " on {source_system}"
    ),
    "VERIFY_WITH_MANAGER": (
        "Notify manager for {username} on {source_system} to verify access"
    ),
    "FLAG_HR_REVIEW": (
        "Flag {username} for HR access review on {source_system}"
    ),
    "SCOPE_TO_RESOURCES": (
        "Scope {username} to job-specific resources on {source_system}"
    ),
}

# ── Action types requiring approval ──────────────────────────────────────────
_APPROVAL_REQUIRED: frozenset[str] = frozenset({"DISABLE_ACCOUNT", "REVOKE_ROLE"})


@dataclass(frozen=True)
class RemediationAction:
    """A single prioritized remediation action for an identity.

    Attributes:
        priority: Sort order, with 1 being the most urgent.
        action_type: Canonical action category (e.g. ``"DISABLE_ACCOUNT"``).
        target: The identity subject of the action (username or email).
        human_readable_description: Plain-language explanation of the action.
        machine_actionable_command: Source-system-aware CLI command string.
        estimated_risk_reduction: Risk-score points removed if action is taken.
        expected_resolution_hours: SLA target in hours.
        requires_approval: Whether the action needs manual approval.
    """

    priority: int
    action_type: str
    target: str
    human_readable_description: str
    machine_actionable_command: str
    estimated_risk_reduction: int
    expected_resolution_hours: int
    requires_approval: bool


def _format_command(template: str, identity: IdentityRecord) -> str:
    """Substitute identity fields into a command template.

    Args:
        template: Command string with ``{username}``, ``{email}``,
            and ``{source_system}`` placeholders.
        identity: The identity record supplying substitution values.

    Returns:
        Formatted command string.
    """
    return (
        template.replace("{username}", identity.username)
        .replace("{email}", identity.email)
        .replace("{source_system}", identity.source_system)
    )


def _build_command(action_type: str, identity: IdentityRecord) -> str:
    """Build a source-system-aware CLI command for an action type.

    Args:
        action_type: Canonical action category.
        identity: The identity record to operate on.

    Returns:
        A formatted command string specific to the identity's source
        system.
    """
    if action_type == "DISABLE_ACCOUNT":
        source = identity.source_system
        template = _SOURCE_DISABLE_COMMANDS.get(
            source,
            "Disable account {username} on {source_system}",
        )
        return _format_command(template, identity)
    template = _GENERIC_COMMANDS.get(
        action_type,
        "{action_type} applied to {username} on {source_system}",
    )
    return _format_command(template, identity)


def _gather_suppressed_rule_ids(context_signals: list[Any]) -> set[str]:
    """Collect rule IDs suppressed by any context signal.

    Args:
        context_signals: List of context signal objects with a
            ``rules_suppressed`` attribute.

    Returns:
        Set of suppressed rule IDs, or empty set if no signals.
    """
    suppressed: set[str] = set()
    for sig in context_signals:
        for rule_id in getattr(sig, "rules_suppressed", []):
            suppressed.add(rule_id)
    return suppressed


def _any_requires_followup(context_signals: list[Any]) -> bool:
    """Check whether any context signal requires manager followup.

    Args:
        context_signals: List of context signal objects.

    Returns:
        True if at least one signal has ``requires_followup`` set to True.
    """
    for sig in context_signals:
        if getattr(sig, "requires_followup", False):
            return True
    return False


def _severity_from_reduction(reduction: int) -> str:
    """Infer rule severity from the estimated risk reduction value.

    Args:
        reduction: Estimated risk reduction score points.

    Returns:
        Severity string (``"CRITICAL"``, ``"HIGH"``, ``"MEDIUM"``,
        or ``"LOW"``).
    """
    if reduction >= 30:
        return "CRITICAL"
    if reduction >= 20:
        return "HIGH"
    if reduction >= 10:
        return "MEDIUM"
    return "LOW"


def _action_sort_key(
    action: RemediationAction,
    risk_score: int,
) -> tuple[int, int, int]:
    """Compute a sort key for prioritising remediation actions.

    Args:
        action: The remediation action to sort.
        risk_score: The overall identity risk score.

    Returns:
        A tuple of (severity_order, negated_risk_score,
        negated_risk_reduction) so that sorting ascending yields
        highest priority first.
    """
    sev = _SEVERITY_ORDER[_severity_from_reduction(action.estimated_risk_reduction)]
    return (sev, -risk_score, -action.estimated_risk_reduction)


def _build_remediation_action(
    rule: RuleResult,
    identity: IdentityRecord,
    action_type: str,
) -> RemediationAction:
    """Build a RemediationAction from a triggered rule result.

    Args:
        rule: The triggered rule result.
        identity: The identity record.
        action_type: The mapped action type for the rule.

    Returns:
        A fully-populated RemediationAction with placeholder priority.
    """
    return RemediationAction(
        priority=0,
        action_type=action_type,
        target=identity.username or identity.email,
        human_readable_description=rule.evidence_text,
        machine_actionable_command=_build_command(action_type, identity),
        estimated_risk_reduction=_SEVERITY_RISK_REDUCTION.get(rule.severity, 0),
        expected_resolution_hours=_ACTION_SLA.get(action_type, 24),
        requires_approval=action_type in _APPROVAL_REQUIRED,
    )


def _make_context_action(
    action_type: str,
    description: str,
    risk_reduction: int,
    identity: IdentityRecord,
) -> RemediationAction:
    """Create a context-driven remediation action.

    Args:
        action_type: Canonical action category.
        description: Human-readable description of the action.
        risk_reduction: Risk score points removed if action is taken.
        identity: The identity record.

    Returns:
        A RemediationAction with placeholder priority.
    """
    return RemediationAction(
        priority=0,
        action_type=action_type,
        target=identity.username or identity.email,
        human_readable_description=description,
        machine_actionable_command=_build_command(action_type, identity),
        estimated_risk_reduction=risk_reduction,
        expected_resolution_hours=_ACTION_SLA.get(action_type, 24),
        requires_approval=action_type in _APPROVAL_REQUIRED,
    )


def generate_remediation_plan(
    rules: list[RuleResult],
    identity: IdentityRecord,
    risk_score: int,
    context_signals: list[Any],
) -> list[RemediationAction]:
    """Generate prioritized remediation actions for an identity.

    Args:
        rules: RuleResult objects from the rules engine.
        identity: The identity record being evaluated.
        risk_score: The overall computed risk score.
        context_signals: Context signals for rule suppression and follow-up.

    Returns:
        Sorted list of RemediationAction objects (priority 1 = highest).
    """
    suppressed = _gather_suppressed_rule_ids(context_signals)
    actions: list[RemediationAction] = [
        _build_remediation_action(rule, identity, _RULE_ACTION_MAP[rule.rule_id])
        for rule in rules
        if rule.triggered
        and rule.rule_id not in suppressed
        and rule.rule_id in _RULE_ACTION_MAP
    ]

    if _any_requires_followup(context_signals):
        actions.append(_make_context_action(
            "VERIFY_WITH_MANAGER", "Context signals require manager verification",
            5, identity))

    if not identity.mfa_enabled:
        actions.append(_make_context_action(
            "ENABLE_MFA", "MFA is not enabled for this account", 10, identity))

    actions.sort(key=lambda a: _action_sort_key(a, risk_score))
    return [
        replace(action, priority=i) for i, action in enumerate(actions, start=1)
    ]


# ── Compliance Gap Analysis ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ComplianceGap:
    """A compliance framework gap triggered by a detection rule.

    Attributes:
        framework: The compliance framework (e.g. ``"NIST SP 800-53"``).
        control_id: The specific control identifier.
        control_name: Human-readable control name.
        gap_description: What the gap means in plain English.
        triggered_by_rule: The rule_id that produced this gap.
        remediation_required: Required action to close the gap.
    """

    framework: str
    control_id: str
    control_name: str
    gap_description: str
    triggered_by_rule: str
    remediation_required: str


RULE_TO_COMPLIANCE: dict[str, ComplianceGap] = {
    "STALE_PRIVILEGED": ComplianceGap(
        framework="NIST SP 800-53",
        control_id="AC-2",
        control_name="Account Management",
        gap_description="Inactive privileged account not disabled within review cycle.",
        triggered_by_rule="STALE_PRIVILEGED",
        remediation_required="Disable or suspend account. Document exception if active.",
    ),
    "ORPHANED_ACCOUNT": ComplianceGap(
        framework="NIST SP 800-53",
        control_id="AC-2(3)",
        control_name="Disable Inactive Accounts",
        gap_description="Account has no owner and cannot be reviewed or recertified.",
        triggered_by_rule="ORPHANED_ACCOUNT",
        remediation_required="Assign owner or disable immediately.",
    ),
    "OVER_PRIVILEGED": ComplianceGap(
        framework="GDPR",
        control_id="Article 32",
        control_name="Security of Processing",
        gap_description="User has access to sensitive data beyond role requirements.",
        triggered_by_rule="OVER_PRIVILEGED",
        remediation_required="Apply least-privilege: revoke unused permissions.",
    ),
    "SOD_VIOLATION": ComplianceGap(
        framework="SOX",
        control_id="Section 302",
        control_name="Corporate Responsibility",
        gap_description="Role combination creates financial control bypass risk.",
        triggered_by_rule="SOD_VIOLATION",
        remediation_required="Separate conflicting roles. Require dual approval.",
    ),
    "SHADOW_ADMIN": ComplianceGap(
        framework="NIST SP 800-53",
        control_id="AC-6",
        control_name="Least Privilege",
        gap_description="Admin permissions granted outside formal role assignment.",
        triggered_by_rule="SHADOW_ADMIN",
        remediation_required="Remove direct permission grant. Use role-based assignment.",
    ),
}
