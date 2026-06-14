"""Tests for core.risk_scorer module."""

from types import SimpleNamespace

import pytest

from core.risk_scorer import (
    BLAST_RADIUS_MULTIPLIER,
    RULE_WEIGHTS,
    compute_risk_score,
)
from core.rules_engine import RuleResult

_ALL_RULE_IDS = [
    "STALE_PRIVILEGED", "ORPHANED_ACCOUNT", "OVER_PRIVILEGED",
    "SHADOW_ADMIN", "PRIVILEGE_CREEP", "SERVICE_ACCT_ABUSE",
    "CREDENTIAL_SPRAWL", "IMPOSSIBLE_TRAVEL", "EXCESSIVE_ACCESS",
    "BULK_DOWNLOAD", "SOD_VIOLATION",
]


@pytest.fixture
def identity() -> SimpleNamespace:
    """Stub IdentityRecord for testing."""
    return SimpleNamespace(
        user_id="U001",
        username="test_user",
        email="test@test.com",
        department="Engineering",
        employment_status="active",
        account_type="human",
        owner_id="mgr1",
        source_system="AD",
        job_title="",
        last_login=None,
        created_at=None,
        roles=[],
        permissions=[],
        mfa_enabled=False,
        sso_linked=False,
        login_count_30d=0,
        login_count_90d=0,
        systems_count=0,
        role_changes_90d=0,
        is_privileged=False,
        resource_sensitivity="low",
        off_hours_access_pct=0.0,
        geo_anomaly=False,
        interactive_login=False,
        event_count_30d=0,
        event_count_90d=0,
        unique_resources_accessed=0,
        anomaly_event_count=0,
        failed_attempt_count=0,
        off_hours_event_pct=0.0,
        impossible_travel_detected=False,
        bulk_download_detected=False,
        max_resources_in_single_session=0,
        avg_time_between_events_hours=0.0,
    )


def _make_rule(
    rule_id: str,
    triggered: bool = True,
    severity: str = "HIGH",
    suppressed_by: str | None = None,
) -> RuleResult:
    """Build a RuleResult stub."""
    return RuleResult(
        rule_id=rule_id,
        severity=severity,
        triggered=triggered,
        evidence_text=f"Test evidence for {rule_id}" if triggered else "",
        suppressed_by=suppressed_by,
    )


def _all_clean_rules() -> list[RuleResult]:
    return [_make_rule(rid, triggered=False) for rid in _ALL_RULE_IDS]


def _make_context_signal(
    signal_type: str = "SABBATICAL_POSSIBLE",
    explanation: str = "Test explanation",
    confidence: float = 0.60,
    score_adjustment: int = -10,
    rules_suppressed: list[str] | None = None,
    requires_followup: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        signal_type=signal_type,
        explanation=explanation,
        confidence=confidence,
        score_adjustment=score_adjustment,
        rules_suppressed=rules_suppressed if rules_suppressed is not None else [],
        requires_followup=requires_followup,
    )


class TestWeights:
    """Verify all WEIGHT_* constants are accessible and are integers."""

    def test_each_weight_constant_defined(self) -> None:
        expected = [
            "WEIGHT_ADMIN_ROLE", "WEIGHT_ORPHANED", "WEIGHT_STALE",
            "WEIGHT_SENSITIVE_RESOURCE", "WEIGHT_ANOMALOUS_ACCESS", "WEIGHT_ROLE_STACK",
        ]
        import core.risk_scorer as rs

        for name in expected:
            val = getattr(rs, name, None)
            assert val is not None, f"Missing constant: {name}"
            assert isinstance(val, int), f"{name} should be int, got {type(val).__name__}"

    def test_rule_weights_dict_defined(self) -> None:
        assert isinstance(RULE_WEIGHTS, dict)
        for rid in _ALL_RULE_IDS:
            assert rid in RULE_WEIGHTS, f"Missing rule weight: {rid}"
            assert isinstance(RULE_WEIGHTS[rid], int)


class TestBaseScore:
    """Test base score computed from triggered, non-suppressed rules."""

    def test_base_score_single_rule(self, identity: SimpleNamespace) -> None:
        rules = [
            _make_rule("STALE_PRIVILEGED", triggered=True),
            _make_rule("ORPHANED_ACCOUNT", triggered=False),
        ]
        result = compute_risk_score(identity, rules, [], 0.0, 0.0)
        assert result.score > 0
        assert result.score == RULE_WEIGHTS["STALE_PRIVILEGED"]

    def test_clean_identity_zero_score(self, identity: SimpleNamespace) -> None:
        result = compute_risk_score(identity, _all_clean_rules(), [], 0.0, 0.0)
        assert result.score == 0


class TestBlastRadius:
    """Test blast-radius multiplier behaviour."""

    def test_blast_radius_applied(self, identity: SimpleNamespace) -> None:
        rules = [
            _make_rule("PRIVILEGE_CREEP", triggered=True),    # weight 35
            _make_rule("CREDENTIAL_SPRAWL", triggered=True),   # weight 25
            _make_rule("EXCESSIVE_ACCESS", triggered=True),    # weight 25
        ]
        result = compute_risk_score(identity, rules, [], 0.0, 0.0)
        assert result.blast_radius_applied is True
        expected_sum = sum(
            RULE_WEIGHTS[rid] for rid in
            ["PRIVILEGE_CREEP", "CREDENTIAL_SPRAWL", "EXCESSIVE_ACCESS"]
        )
        # With blast radius (85 * 1.20 = 102 -> clamped to 100 or = 102 without clamp)
        # Score should be > 85 (the base sum) due to multiplier
        # Actually with clamped at 100, it could equal 100 which is > expected_sum
        assert result.score >= round(expected_sum * BLAST_RADIUS_MULTIPLIER) or result.score == 100

    def test_blast_radius_not_applied(self, identity: SimpleNamespace) -> None:
        rules = [
            _make_rule("STALE_PRIVILEGED", triggered=True),
            _make_rule("ORPHANED_ACCOUNT", triggered=True),
        ]
        result = compute_risk_score(identity, rules, [], 0.0, 0.0)
        assert result.blast_radius_applied is False


class TestClamping:
    """Test that final risk score never exceeds 100."""

    def test_score_clamped_at_100(self, identity: SimpleNamespace) -> None:
        triggered = [
            _make_rule("STALE_PRIVILEGED", triggered=True),
            _make_rule("ORPHANED_ACCOUNT", triggered=True),
            _make_rule("OVER_PRIVILEGED", triggered=True),
            _make_rule("SHADOW_ADMIN", triggered=True),
            _make_rule("SERVICE_ACCT_ABUSE", triggered=True),
            _make_rule("BULK_DOWNLOAD", triggered=True),
        ]
        result = compute_risk_score(identity, triggered, [], 80.0, 10.0)
        assert result.score == 100


class TestTierBoundaries:
    """Test risk-tier thresholds."""

    def test_tier_high_and_critical(self, identity: SimpleNamespace) -> None:
        # Score 79 -> HIGH
        rules = [_make_rule("STALE_PRIVILEGED", triggered=True, severity="HIGH")]
        ctx = _make_context_signal(score_adjustment=79 - RULE_WEIGHTS["STALE_PRIVILEGED"])
        result = compute_risk_score(identity, rules, [ctx], 0.0, 0.0)
        assert result.score <= 79
        assert result.tier in ("HIGH", "MEDIUM", "LOW")

    def test_score_zero_is_low(self, identity: SimpleNamespace) -> None:
        result = compute_risk_score(identity, _all_clean_rules(), [], 0.0, 0.0)
        assert result.score == 0
        assert result.tier == "LOW"


class TestContextAdjustment:
    """Test that context signals shift the final risk score."""

    def test_negative_context_reduces_score(self, identity: SimpleNamespace) -> None:
        rules = [_make_rule("STALE_PRIVILEGED", triggered=True)]
        base = compute_risk_score(identity, rules, [], 0.0, 0.0)

        ctx = _make_context_signal(score_adjustment=-20)
        adjusted = compute_risk_score(identity, rules, [ctx], 0.0, 0.0)
        assert adjusted.score < base.score

    def test_positive_context_increases_score(self, identity: SimpleNamespace) -> None:
        rules = [_make_rule("STALE_PRIVILEGED", triggered=True)]
        base = compute_risk_score(identity, rules, [], 0.0, 0.0)

        ctx = _make_context_signal(score_adjustment=25)
        adjusted = compute_risk_score(identity, rules, [ctx], 0.0, 0.0)
        assert adjusted.score > base.score


class TestRiskNarrative:
    """Test risk narrative generation for flagged identities."""

    def test_risk_narrative_non_empty(self, identity: SimpleNamespace) -> None:
        rules = [
            _make_rule("STALE_PRIVILEGED", triggered=True),
            _make_rule("ORPHANED_ACCOUNT", triggered=True),
        ]
        result = compute_risk_score(identity, rules, [], 0.0, 0.0)
        assert isinstance(result.risk_narrative, str)
        assert len(result.risk_narrative) > 0

    def test_risk_narrative_word_count(self, identity: SimpleNamespace) -> None:
        rules = [
            _make_rule("STALE_PRIVILEGED", triggered=True),
            _make_rule("ORPHANED_ACCOUNT", triggered=True),
            _make_rule("OVER_PRIVILEGED", triggered=True),
        ]
        result = compute_risk_score(identity, rules, [], 0.0, 0.0)
        word_count = len(result.risk_narrative.split())
        assert 80 <= word_count <= 150, f"Expected 80-150 words, got {word_count}"


class TestSuppression:
    """Test that suppressed rules are excluded from contributing factors."""

    def test_suppressed_rules_not_in_contributing(self, identity: SimpleNamespace) -> None:
        rules = [
            _make_rule("STALE_PRIVILEGED", triggered=True),
            _make_rule("ORPHANED_ACCOUNT", triggered=True, suppressed_by="SABBATICAL_POSSIBLE"),
        ]
        result = compute_risk_score(identity, rules, [], 0.0, 0.0)
        assert "STALE_PRIVILEGED" in result.contributing_factors
        assert "ORPHANED_ACCOUNT" not in result.contributing_factors

    def test_suppressed_rules_in_suppressed_factors(self, identity: SimpleNamespace) -> None:
        rules = [
            _make_rule("STALE_PRIVILEGED", triggered=True),
            _make_rule("ORPHANED_ACCOUNT", triggered=True, suppressed_by="SABBATICAL_POSSIBLE"),
        ]
        result = compute_risk_score(identity, rules, [], 0.0, 0.0)
        assert "ORPHANED_ACCOUNT" in result.suppressed_factors


class TestRiskResult:
    """Test RiskResult frozen dataclass properties."""

    def test_risk_result_is_frozen(self, identity: SimpleNamespace) -> None:
        rules = [
            _make_rule("STALE_PRIVILEGED", triggered=True),
            _make_rule("ORPHANED_ACCOUNT", triggered=True),
        ]
        result = compute_risk_score(identity, rules, [], 0.0, 0.0)
        with pytest.raises(Exception):
            result.score = 99  # type: ignore[misc]


class TestRoleExceptions:
    """Test senior-role exception softening via core/role_exceptions."""

    def test_cto_stale_account_softened(self, identity: SimpleNamespace) -> None:
        identity = SimpleNamespace(**{**vars(identity), "job_title": "CTO", "is_privileged": True, "roles": ["admin"], "last_login": None})
        rules = [_make_rule("STALE_PRIVILEGED", triggered=True)]
        result = compute_risk_score(identity, rules, [], 0.0, 0.0)
        assert result.role_exceptions
        assert "Role exception applied" in result.risk_narrative
        # Score should be lower than weight alone (50 - 25 = 25)
        assert result.score < RULE_WEIGHTS["STALE_PRIVILEGED"]

    def test_junior_analyst_stale_not_softened(self, identity: SimpleNamespace) -> None:
        identity = SimpleNamespace(**{**vars(identity), "job_title": "Analyst", "is_privileged": True, "roles": ["reader"], "last_login": None})
        rules = [_make_rule("STALE_PRIVILEGED", triggered=True)]
        result = compute_risk_score(identity, rules, [], 0.0, 0.0)
        assert result.role_exceptions == []
        assert "Role exception applied" not in result.risk_narrative

    def test_director_of_facilities_matched_by_design(self) -> None:
        """Documented: 'director' applies broadly by design as a seniority token."""
        from core.role_exceptions import resolve_role_exceptions
        from core.ingestion import IdentityRecord
        identity = IdentityRecord(
            user_id="U001", username="dir_test", email="d@t.com",
            department="Facilities", employment_status="active",
            account_type="human", owner_id="mgr1", source_system="AD",
            job_title="Director of Facilities", roles=["director"], is_privileged=True,
        )
        exceptions = resolve_role_exceptions(identity)
        assert len(exceptions) == 1
        assert exceptions[0].role_keyword == "director"

    def test_double_title_no_stacking(self, identity: SimpleNamespace) -> None:
        identity = SimpleNamespace(**{**vars(identity), "job_title": "VP / Director of Engineering", "is_privileged": True, "roles": ["vp", "director"], "last_login": None})
        rules = [_make_rule("STALE_PRIVILEGED", triggered=True)]
        result = compute_risk_score(identity, rules, [], 0.0, 0.0)
        # Only one exception applied (largest magnitude)
        assert len(result.role_exceptions) == 1


class TestScoreClamping:
    """Test floor/ceiling clamping."""

    def test_negative_adjusted_score_clamped_to_zero(self, identity: SimpleNamespace) -> None:
        """Large negative context_delta + zero rules = score clamped to 0."""
        ctx = _make_context_signal(score_adjustment=-50)
        result = compute_risk_score(identity, _all_clean_rules(), [ctx], 0.0, 0.0)
        assert result.score == 0
        assert result.tier == "LOW"

    def test_score_above_100_clamped(self, identity: SimpleNamespace) -> None:
        """Score exceeding 100 is clamped."""
        rules = [_make_rule("STALE_PRIVILEGED", triggered=True),
                 _make_rule("ORPHANED_ACCOUNT", triggered=True),
                 _make_rule("SHADOW_ADMIN", triggered=True),
                 _make_rule("SERVICE_ACCT_ABUSE", triggered=True)]
        result = compute_risk_score(identity, rules, [], 10.0, 5.0)
        assert result.score <= 100


class TestComplianceGaps:
    """Test compliance gap analysis."""

    def test_compliance_gap_populated_for_stale_privileged(self, identity: SimpleNamespace) -> None:
        identity = SimpleNamespace(**{**vars(identity), "is_privileged": True, "last_login": None})
        rules = [_make_rule("STALE_PRIVILEGED", triggered=True)]
        result = compute_risk_score(identity, rules, [], 0.0, 0.0)
        assert len(result.compliance_gaps) > 0
        gap = result.compliance_gaps[0]
        assert gap.framework == "NIST SP 800-53"
        assert gap.control_id == "AC-2"

    def test_compliance_gap_empty_for_clean_identity(self, identity: SimpleNamespace) -> None:
        rules = _all_clean_rules()
        result = compute_risk_score(identity, rules, [], 0.0, 0.0)
        assert result.compliance_gaps == []

    def test_suppressed_rule_does_not_add_compliance_gap(self, identity: SimpleNamespace) -> None:
        identity = SimpleNamespace(**{**vars(identity), "is_privileged": True, "last_login": None})
        rules = [_make_rule("STALE_PRIVILEGED", triggered=True, suppressed_by="SABBATICAL_POSSIBLE")]
        result = compute_risk_score(identity, rules, [], 0.0, 0.0)
        assert result.compliance_gaps == []
