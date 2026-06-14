"""Tests for core.context_resolver module."""

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.context_resolver import ContextSignal, resolve
from core.ingestion import IdentityRecord


def _make_identity(**overrides):
    """Create an IdentityRecord with sensible defaults for testing."""
    defaults = dict(
        user_id="U00001",
        username="testuser",
        email="test@example.com",
        department="Engineering",
        employment_status="active",
        account_type="human",
        owner_id="",
        source_system="AD",
        last_login=None,
        created_at=None,
        roles=[],
        permissions=[],
        mfa_enabled=False,
        sso_linked=False,
        login_count_30d=0,
        login_count_90d=0,
        systems_count=5,
        role_changes_90d=0,
        is_privileged=False,
        resource_sensitivity="low",
        off_hours_access_pct=0.0,
        geo_anomaly=False,
        interactive_login=False,
        event_count_30d=10,
        event_count_90d=30,
        unique_resources_accessed=3,
        anomaly_event_count=0,
        failed_attempt_count=0,
        off_hours_event_pct=0.0,
        impossible_travel_detected=False,
        bulk_download_detected=False,
        max_resources_in_single_session=5,
        avg_time_between_events_hours=2.0,
    )
    defaults.update(overrides)
    return IdentityRecord(**defaults)


def _make_baseline(**overrides):
    """Create a stub baseline with minimal attributes for context resolution."""
    defaults = dict(
        role="Developer",
        department="Engineering",
        avg_events_30d=20.0,
        std_events_30d=5.0,
        avg_unique_resources_30d=5.0,
        std_unique_resources_30d=2.0,
        typical_access_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
        typical_resources=["app_server"],
        p95_events_single_session=15.0,
        p99_download_count=100.0,
        month_end_multiplier=1.5,
        quarter_end_multiplier=1.3,
        computed_at="2026-01-01",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _patch_now(now):
    """Return a patcher that sets datetime.now() to the given datetime."""
    mock_dt = MagicMock()
    mock_dt.now.return_value = now
    return patch("core.context_resolver.datetime", mock_dt)


class TestContextResolver:
    """Test suite for the core.context_resolver module."""

    def test_sabbatical_possible_triggered(self):
        now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        last_login = now - timedelta(days=60)
        identity = _make_identity(
            is_privileged=True,
            employment_status="active",
            owner_id="mgr1",
            last_login=last_login,
        )

        with _patch_now(now):
            signals = resolve(identity, None)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.signal_type == "SABBATICAL_POSSIBLE"
        assert sig.score_adjustment == -10
        assert sig.requires_followup is True

    def test_sabbatical_not_triggered_if_not_privileged(self):
        now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        last_login = now - timedelta(days=60)
        identity = _make_identity(
            is_privileged=False,
            employment_status="active",
            owner_id="mgr1",
            last_login=last_login,
        )

        with _patch_now(now):
            signals = resolve(identity, None)

        signal_types = {s.signal_type for s in signals}
        assert "SABBATICAL_POSSIBLE" not in signal_types

    def test_temp_elevation_triggered(self):
        identity = _make_identity(role_changes_90d=1)
        signals = resolve(identity, None)

        temp_signals = [s for s in signals if s.signal_type == "TEMP_ELEVATION"]
        assert len(temp_signals) == 1
        sig = temp_signals[0]
        assert "PRIVILEGE_CREEP" in sig.rules_suppressed

    def test_new_hire_ramp_triggered(self):
        now = datetime(2026, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        created_at = now - timedelta(days=20)
        identity = _make_identity(
            created_at=created_at,
            systems_count=15,
        )
        baseline = _make_baseline(avg_unique_resources_30d=5.0)

        with _patch_now(now):
            signals = resolve(identity, baseline)

        ramp_signals = [s for s in signals if s.signal_type == "NEW_HIRE_RAMP"]
        assert len(ramp_signals) == 1
        sig = ramp_signals[0]
        assert "OVER_PRIVILEGED" in sig.rules_suppressed
        assert sig.score_adjustment == -10

    def test_batch_job_pattern_triggered(self):
        identity = _make_identity(
            account_type="service",
            off_hours_access_pct=0.85,
            interactive_login=False,
            event_count_30d=50,
        )
        baseline = _make_baseline(
            avg_events_30d=45.0,
            std_events_30d=5.0,
        )

        signals = resolve(identity, baseline)

        batch_signals = [s for s in signals if s.signal_type == "BATCH_JOB_PATTERN"]
        assert len(batch_signals) == 1
        sig = batch_signals[0]
        assert "SERVICE_ACCT_ABUSE" in sig.rules_suppressed
        assert sig.score_adjustment == -15

    def test_month_end_finance_triggered(self):
        now = datetime(2026, 6, 28, 12, 0, 0, tzinfo=timezone.utc)
        identity = _make_identity(
            department="Finance",
            event_count_30d=40,
        )
        baseline = _make_baseline(
            department="Finance",
            avg_events_30d=20.0,
            std_events_30d=5.0,
            month_end_multiplier=1.5,
        )

        with _patch_now(now):
            signals = resolve(identity, baseline)

        me_signals = [s for s in signals if s.signal_type == "MONTH_END_FINANCE"]
        assert len(me_signals) == 1
        sig = me_signals[0]
        assert "EXCESSIVE_ACCESS" in sig.rules_suppressed
        assert sig.score_adjustment == -20

    def test_impossible_travel_confirmed(self):
        identity = _make_identity(impossible_travel_detected=True)

        signals = resolve(identity, None)

        travel_signals = [s for s in signals if s.signal_type == "IMPOSSIBLE_TRAVEL_CONFIRMED"]
        assert len(travel_signals) == 1
        sig = travel_signals[0]
        assert sig.score_adjustment == 25

    def test_contractor_norm(self):
        identity = _make_identity(
            employment_status="contractor",
            off_hours_access_pct=0.5,
        )

        signals = resolve(identity, None)

        contractor_signals = [s for s in signals if s.signal_type == "CONTRACTOR_NORM"]
        assert len(contractor_signals) == 1
        sig = contractor_signals[0]
        assert sig.score_adjustment == 0
        assert sig.rules_suppressed == []

    def test_clean_identity_zero_signals(self):
        identity = _make_identity()
        signals = resolve(identity, None)
        assert signals == []

    def test_context_signal_is_frozen(self):
        sig = ContextSignal(
            signal_type="TEST",
            explanation="test",
            confidence=0.5,
            score_adjustment=0,
            rules_suppressed=[],
            requires_followup=False,
        )
        with pytest.raises(Exception):
            sig.signal_type = "MUTATED"  # type: ignore[misc]
