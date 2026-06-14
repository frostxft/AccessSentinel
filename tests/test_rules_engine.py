"""Tests for core.rules_engine module."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from core.rules_engine import RuleResult, evaluate_rules
from core.ingestion import IdentityRecord


def _make_identity(**overrides):
    defaults = dict(
        user_id="U00001",
        username="test_user",
        email="test@example.com",
        department="Engineering",
        employment_status="active",
        account_type="human",
        owner_id="mgr001",
        source_system="AD",
        last_login=datetime.now(timezone.utc) - timedelta(days=1),
        created_at=None,
        roles=[],
        permissions=[],
        mfa_enabled=True,
        sso_linked=True,
        login_count_30d=50,
        login_count_90d=150,
        systems_count=1,
        role_changes_90d=0,
        is_privileged=False,
        resource_sensitivity="low",
        off_hours_access_pct=0.0,
        geo_anomaly=False,
        interactive_login=False,
        event_count_30d=100,
        event_count_90d=300,
        unique_resources_accessed=5,
        anomaly_event_count=0,
        failed_attempt_count=0,
        off_hours_event_pct=0.0,
        impossible_travel_detected=False,
        cross_system_impossible_travel=False,
        bulk_download_detected=False,
        max_resources_in_single_session=3,
        max_resources_in_single_day=0,
        avg_time_between_events_hours=2.0,
    )
    defaults.update(overrides)
    return IdentityRecord(**defaults)


def _make_baseline(**overrides):
    defaults = dict(
        p95_events_single_session=10.0,
        p99_download_count=100.0,
        department="Engineering",
        role="Developer",
        avg_events_30d=20.0,
        std_events_30d=5.0,
        avg_unique_resources_30d=8.0,
        typical_access_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
        typical_resources=[],
        month_end_multiplier=1.5,
        quarter_end_multiplier=1.3,
        computed_at="2026-01-01",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _find_result(results, rule_id):
    for r in results:
        if r.rule_id == rule_id:
            return r
    return None


class TestStalePrivileged:
    def test_positive_last_login_31_days_privileged(self) -> None:
        identity = _make_identity(
            last_login=datetime.now(timezone.utc) - timedelta(days=31),
            is_privileged=True,
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "STALE_PRIVILEGED")
        assert r is not None
        assert r.triggered is True
        assert r.severity == "HIGH"

    def test_negative_last_login_29_days(self) -> None:
        identity = _make_identity(
            last_login=datetime.now(timezone.utc) - timedelta(days=29),
            is_privileged=True,
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "STALE_PRIVILEGED")
        assert r is not None
        assert r.triggered is False

    def test_boundary_exactly_30_days_not_triggered(self) -> None:
        identity = _make_identity(
            last_login=datetime.now(timezone.utc) - timedelta(days=30),
            is_privileged=True,
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "STALE_PRIVILEGED")
        assert r is not None
        assert r.triggered is False


class TestOrphanedAccount:
    def test_positive_no_owner(self) -> None:
        identity = _make_identity(owner_id="")
        results = evaluate_rules(identity)
        r = _find_result(results, "ORPHANED_ACCOUNT")
        assert r is not None
        assert r.triggered is True
        assert r.severity == "CRITICAL"
        assert "no owner" in r.evidence_text.lower()

    def test_positive_terminated_status(self) -> None:
        identity = _make_identity(
            owner_id="mgr001",
            employment_status="terminated",
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "ORPHANED_ACCOUNT")
        assert r is not None
        assert r.triggered is True
        assert "terminated" in r.evidence_text.lower()

    def test_negative_normal_with_owner(self) -> None:
        identity = _make_identity(
            owner_id="mgr001",
            employment_status="active",
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "ORPHANED_ACCOUNT")
        assert r is not None
        assert r.triggered is False


class TestOverPrivileged:
    def test_positive_low_utilization(self) -> None:
        identity = _make_identity(
            permissions=["read", "write", "admin", "delete"],
            unique_resources_accessed=0,
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "OVER_PRIVILEGED")
        assert r is not None
        assert r.triggered is True
        assert r.severity == "HIGH"

    def test_negative_normal_utilization(self) -> None:
        identity = _make_identity(
            permissions=["read", "write", "admin"],
            unique_resources_accessed=10,
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "OVER_PRIVILEGED")
        assert r is not None
        assert r.triggered is False

    def test_rule_is_present_in_output(self) -> None:
        identity = _make_identity()
        results = evaluate_rules(identity)
        r = _find_result(results, "OVER_PRIVILEGED")
        assert r is not None
        assert isinstance(r, RuleResult)


class TestShadowAdmin:
    def test_positive_human_with_admin_permission(self) -> None:
        identity = _make_identity(
            account_type="human",
            permissions=["read", "admin"],
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "SHADOW_ADMIN")
        assert r is not None
        assert r.triggered is True
        assert r.severity == "CRITICAL"

    def test_negative_admin_account_with_admin_permission(self) -> None:
        identity = _make_identity(
            account_type="admin",
            permissions=["admin"],
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "SHADOW_ADMIN")
        assert r is not None
        assert r.triggered is False


class TestPrivilegeCreep:
    def test_positive_five_role_changes(self) -> None:
        identity = _make_identity(role_changes_90d=5)
        results = evaluate_rules(identity)
        r = _find_result(results, "PRIVILEGE_CREEP")
        assert r is not None
        assert r.triggered is True
        assert r.severity == "MEDIUM"

    def test_negative_two_role_changes(self) -> None:
        identity = _make_identity(role_changes_90d=2)
        results = evaluate_rules(identity)
        r = _find_result(results, "PRIVILEGE_CREEP")
        assert r is not None
        assert r.triggered is False


class TestServiceAcctAbuse:
    def test_positive_service_with_interactive_login(self) -> None:
        identity = _make_identity(
            account_type="service",
            interactive_login=True,
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "SERVICE_ACCT_ABUSE")
        assert r is not None
        assert r.triggered is True
        assert r.severity == "CRITICAL"

    def test_negative_service_without_interactive_login(self) -> None:
        identity = _make_identity(
            account_type="service",
            interactive_login=False,
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "SERVICE_ACCT_ABUSE")
        assert r is not None
        assert r.triggered is False


class TestCredentialSprawl:
    def test_positive_seven_systems_no_sso(self) -> None:
        identity = _make_identity(
            systems_count=7,
            sso_linked=False,
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "CREDENTIAL_SPRAWL")
        assert r is not None
        assert r.triggered is True
        assert r.severity == "MEDIUM"

    def test_negative_five_systems_with_sso(self) -> None:
        identity = _make_identity(
            systems_count=5,
            sso_linked=True,
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "CREDENTIAL_SPRAWL")
        assert r is not None
        assert r.triggered is False


class TestImpossibleTravel:
    def test_positive_travel_detected(self) -> None:
        identity = _make_identity(impossible_travel_detected=True)
        results = evaluate_rules(identity)
        r = _find_result(results, "IMPOSSIBLE_TRAVEL")
        assert r is not None
        assert r.triggered is True
        assert r.severity == "CRITICAL"

    def test_negative_travel_not_detected(self) -> None:
        identity = _make_identity(impossible_travel_detected=False)
        results = evaluate_rules(identity)
        r = _find_result(results, "IMPOSSIBLE_TRAVEL")
        assert r is not None
        assert r.triggered is False


class TestExcessiveAccess:
    def test_positive_exceeds_baseline_and_high_anomaly(self) -> None:
        baseline = _make_baseline(p95_events_single_session=10.0)
        identity = _make_identity(
            unique_resources_accessed=20,
            event_count_30d=40,
            anomaly_event_count=20,
        )
        results = evaluate_rules(identity, baseline)
        r = _find_result(results, "EXCESSIVE_ACCESS")
        assert r is not None
        assert r.triggered is True
        assert r.severity == "HIGH"

    def test_negative_within_baseline(self) -> None:
        baseline = _make_baseline(p95_events_single_session=10.0)
        identity = _make_identity(
            unique_resources_accessed=5,
        )
        results = evaluate_rules(identity, baseline)
        r = _find_result(results, "EXCESSIVE_ACCESS")
        assert r is not None
        assert r.triggered is False

    def test_negative_baseline_none_skips_rule(self) -> None:
        identity = _make_identity(
            unique_resources_accessed=999,
            event_count_30d=1000,
            anomaly_event_count=1000,
        )
        results = evaluate_rules(identity, baseline=None)
        r = _find_result(results, "EXCESSIVE_ACCESS")
        assert r is not None
        assert r.triggered is False


class TestBulkDownload:
    def test_positive_download_with_high_off_hours(self) -> None:
        identity = _make_identity(
            bulk_download_detected=True,
            off_hours_event_pct=0.80,
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "BULK_DOWNLOAD")
        assert r is not None
        assert r.triggered is True
        assert r.severity == "CRITICAL"

    def test_negative_download_with_low_off_hours(self) -> None:
        identity = _make_identity(
            bulk_download_detected=True,
            off_hours_event_pct=0.30,
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "BULK_DOWNLOAD")
        assert r is not None
        assert r.triggered is False


class TestSodViolation:
    def test_positive_finance_approver_and_payment_executor(self) -> None:
        identity = _make_identity(
            roles=["finance_approver", "payment_executor"],
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "SOD_VIOLATION")
        assert r is not None
        assert r.triggered is True
        assert r.severity == "CRITICAL"

    def test_negative_reader_and_writer(self) -> None:
        identity = _make_identity(
            roles=["reader", "writer"],
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "SOD_VIOLATION")
        assert r is not None
        assert r.triggered is False


class TestCleanIdentity:
    def test_clean_identity_zero_rules_triggered(self) -> None:
        identity = _make_identity(
            user_id="clean01",
            username="cleanuser",
            email="clean@example.com",
            department="Engineering",
            employment_status="active",
            account_type="human",
            owner_id="mgr001",
            source_system="AD",
            last_login=datetime.now(timezone.utc) - timedelta(days=1),
            roles=["reader"],
            permissions=["read", "write"],
            mfa_enabled=True,
            sso_linked=True,
            login_count_30d=10,
            login_count_90d=30,
            systems_count=2,
            role_changes_90d=0,
            is_privileged=False,
            resource_sensitivity="low",
            off_hours_access_pct=0.1,
            geo_anomaly=False,
            interactive_login=False,
            event_count_30d=100,
            event_count_90d=300,
            unique_resources_accessed=2,
            anomaly_event_count=0,
            failed_attempt_count=0,
            off_hours_event_pct=0.0,
            impossible_travel_detected=False,
            bulk_download_detected=False,
            max_resources_in_single_session=3,
            avg_time_between_events_hours=2.0,
        )
        results = evaluate_rules(identity)
        assert len(results) == 13  # 11 original + LATERAL_MOVEMENT_SPIKE
        triggered = [r for r in results if r.triggered]
        assert len(triggered) == 0, (
            f"Expected 0 triggered rules, got {len(triggered)}: "
            f"{[(r.rule_id, r.evidence_text) for r in triggered]}"
        )


class TestRuleResult:
    def test_rule_result_is_frozen(self) -> None:
        r = RuleResult(
            rule_id="TEST",
            severity="HIGH",
            triggered=True,
            evidence_text="test evidence",
        )
        with pytest.raises(Exception):
            r.rule_id = "modified"  # type: ignore[misc]

    def test_suppressed_rule_still_in_output(self) -> None:
        identity = _make_identity(is_privileged=True, last_login=None)
        results = evaluate_rules(identity)
        assert len(results) == 13  # 11 original + LATERAL_MOVEMENT_SPIKE
        for r in results:
            assert isinstance(r, RuleResult)
            assert isinstance(r.rule_id, str)
            assert isinstance(r.severity, str)
            assert isinstance(r.triggered, bool)
            assert isinstance(r.evidence_text, str)
            assert r.suppressed_by is None
        stale = _find_result(results, "STALE_PRIVILEGED")
        assert stale is not None
        assert stale.triggered is True
        assert stale.evidence_text != ""


class TestLateralMovementSpike:
    """Tests for the LATERAL_MOVEMENT_SPIKE rule."""

    def test_lateral_movement_spike_triggers(self) -> None:
        baseline = _make_baseline(avg_daily_unique_resources=10.0)
        identity = _make_identity(
            max_resources_in_single_day=50  # 50 > 20 threshold AND 50 > 10*3
        )
        results = evaluate_rules(identity, baseline)
        r = _find_result(results, "LATERAL_MOVEMENT_SPIKE")
        assert r is not None
        assert r.triggered is True
        assert r.severity == "CRITICAL"

    def test_power_user_not_falsely_flagged(self) -> None:
        baseline = _make_baseline(avg_daily_unique_resources=20.0)
        identity = _make_identity(
            max_resources_in_single_day=25  # 25 < 20*3 = 60
        )
        results = evaluate_rules(identity, baseline)
        r = _find_result(results, "LATERAL_MOVEMENT_SPIKE")
        assert r is not None
        assert r.triggered is False

    def test_boundary_exactly_threshold(self) -> None:
        baseline = _make_baseline(avg_daily_unique_resources=5.0)
        identity = _make_identity(
            max_resources_in_single_day=20  # exactly at threshold, 20 >= 5*3
        )
        results = evaluate_rules(identity, baseline)
        r = _find_result(results, "LATERAL_MOVEMENT_SPIKE")
        assert r is not None
        assert r.triggered is True

    def test_below_threshold_not_triggered(self) -> None:
        baseline = _make_baseline(avg_daily_unique_resources=5.0)
        identity = _make_identity(
            max_resources_in_single_day=19  # below system threshold of 20
        )
        results = evaluate_rules(identity, baseline)
        r = _find_result(results, "LATERAL_MOVEMENT_SPIKE")
        assert r is not None
        assert r.triggered is False


class TestCrossSystemTravel:
    """Tests for CROSS_SYSTEM_TRAVEL rule (Fix 3)."""

    def test_cross_system_travel_detected(self) -> None:
        identity = _make_identity(cross_system_impossible_travel=True)
        results = evaluate_rules(identity)
        r = _find_result(results, "CROSS_SYSTEM_TRAVEL")
        assert r is not None
        assert r.triggered is True
        assert r.severity == "CRITICAL"

    def test_same_system_not_cross_system(self) -> None:
        identity = _make_identity(
            cross_system_impossible_travel=False,
            impossible_travel_detected=False,
        )
        results = evaluate_rules(identity)
        r = _find_result(results, "CROSS_SYSTEM_TRAVEL")
        assert r is not None
        assert r.triggered is False
