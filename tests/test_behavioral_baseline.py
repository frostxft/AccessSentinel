"""Tests for core.behavioral_baseline module."""

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from core.behavioral_baseline import (
    BaselineProfile,
    build_baselines,
    compute_behavior_zscore,
    load_baselines,
    _BASELINES_DIR,
)
from core.ingestion import IdentityRecord


_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "sample_data", "identity_events.csv"
)


def _make_finance_events_df(days_1_27_count: int = 2, days_28_31_count: int = 20) -> pd.DataFrame:
    """Build a small events DataFrame with Finance users and month-end concentration."""
    records = []
    event_idx = 0
    for day in range(1, 32):
        count = days_28_31_count if day >= 28 else days_1_27_count
        for _ in range(count):
            records.append({
                "event_id": f"EVT{event_idx:06d}",
                "user_id": "U00001",
                "username": "fin_user",
                "department": "Finance",
                "job_title": "Controller",
                "resource": "financial_ledger",
                "action": "read",
                "timestamp": f"2025-12-{day:02d}T12:00:00Z",
                "source_ip": "10.0.0.1",
                "location": "New York, US",
                "success": "true",
            })
            event_idx += 1
    return pd.DataFrame(records)


def _make_non_finance_events_df() -> pd.DataFrame:
    """Build a small events DataFrame with non-Finance users."""
    records = []
    for i in range(31):
        records.append({
            "event_id": f"EVT{i:06d}",
            "user_id": "U00001",
            "username": "eng_user",
            "department": "Engineering",
            "job_title": "Software Engineer",
            "resource": "source_code_repo",
            "action": "read",
            "timestamp": f"2025-06-{min(i + 1, 28):02d}T10:00:00Z",
            "source_ip": "10.0.0.1",
            "location": "San Francisco, US",
            "success": "true",
        })
    return pd.DataFrame(records)


# ── build_baselines tests ───────────────────────────────────────────────────

class TestBuildBaselines:

    def test_build_baselines_from_sample_data(self) -> None:
        df = pd.read_csv(_CSV_PATH)
        baselines = build_baselines(df)
        assert isinstance(baselines, dict)
        assert len(baselines) > 0
        for key, profile in baselines.items():
            assert isinstance(profile, BaselineProfile)
            assert "|" in key

    def test_finance_month_end_multiplier(self) -> None:
        df = _make_finance_events_df(days_1_27_count=2, days_28_31_count=20)
        baselines = build_baselines(df)
        assert len(baselines) > 0
        for profile in baselines.values():
            assert profile.month_end_multiplier > 1.0

    def test_baseline_persists_to_disk(self) -> None:
        df = pd.read_csv(_CSV_PATH)
        build_baselines(df)
        assert os.path.isdir(_BASELINES_DIR)
        json_files = [f for f in os.listdir(_BASELINES_DIR) if f.endswith(".json")]
        assert len(json_files) > 0

    def test_baseline_loads_from_cache(self) -> None:
        df = pd.read_csv(_CSV_PATH)
        built = build_baselines(df)
        loaded = load_baselines()
        assert len(loaded) == len(built)

    def test_empty_event_log_does_not_raise(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_baselines(pd.DataFrame())


# ── BaselineProfile tests ───────────────────────────────────────────────────

class TestBaselineProfile:

    def test_baseline_profile_is_frozen(self) -> None:
        profile = BaselineProfile(
            role="Financial Analyst",
            department="Finance",
            avg_events_30d=100.0,
            std_events_30d=10.0,
            avg_events_per_session=5.0,
            typical_access_hours=[9, 10, 11],
            typical_resources=["financial_ledger"],
            avg_unique_resources_30d=5.0,
            std_unique_resources_30d=2.0,
            p95_events_single_session=20.0,
            p99_download_count=5.0,
            month_end_multiplier=1.0,
            quarter_end_multiplier=1.0,
            computed_at="2026-06-13T00:00:00",
        )
        with pytest.raises(Exception):
            profile.avg_events_30d = 999.0  # type: ignore[misc]


# ── compute_behavior_zscore tests ───────────────────────────────────────────

class TestComputeBehaviorZscore:

    @staticmethod
    def _make_identity(event_count: int) -> IdentityRecord:
        return IdentityRecord(
            user_id="U00001",
            username="test_user",
            email="test@example.com",
            department="Engineering",
            employment_status="active",
            account_type="human",
            owner_id="",
            source_system="AD",
            event_count_30d=event_count,
        )

    def test_compute_zscore_normal(self) -> None:
        identity = self._make_identity(event_count=100)
        profile = BaselineProfile(
            role="Engineer",
            department="Engineering",
            avg_events_30d=100.0,
            std_events_30d=10.0,
            avg_events_per_session=5.0,
            typical_access_hours=[9, 10],
            typical_resources=[],
            avg_unique_resources_30d=5.0,
            std_unique_resources_30d=2.0,
            p95_events_single_session=20.0,
            p99_download_count=0.0,
            month_end_multiplier=1.0,
            quarter_end_multiplier=1.0,
            computed_at="2026-06-13",
        )
        zscore = compute_behavior_zscore(identity, profile)
        assert zscore == pytest.approx(0.0, abs=0.1)

    def test_compute_zscore_elevated(self) -> None:
        identity = self._make_identity(event_count=130)
        profile = BaselineProfile(
            role="Engineer",
            department="Engineering",
            avg_events_30d=100.0,
            std_events_30d=10.0,
            avg_events_per_session=5.0,
            typical_access_hours=[9, 10],
            typical_resources=[],
            avg_unique_resources_30d=5.0,
            std_unique_resources_30d=2.0,
            p95_events_single_session=20.0,
            p99_download_count=0.0,
            month_end_multiplier=1.0,
            quarter_end_multiplier=1.0,
            computed_at="2026-06-13",
        )
        zscore = compute_behavior_zscore(identity, profile)
        assert zscore > 2.5

    def test_finance_user_day_30_not_anomalous(self) -> None:
        identity = IdentityRecord(
            user_id="U00001",
            username="fin_user",
            email="fin@example.com",
            department="Finance",
            employment_status="active",
            account_type="human",
            owner_id="",
            source_system="AD",
            event_count_30d=130,
        )
        profile = BaselineProfile(
            role="Controller",
            department="Finance",
            avg_events_30d=100.0,
            std_events_30d=10.0,
            avg_events_per_session=5.0,
            typical_access_hours=[9, 10, 11],
            typical_resources=["financial_ledger"],
            avg_unique_resources_30d=5.0,
            std_unique_resources_30d=2.0,
            p95_events_single_session=20.0,
            p99_download_count=5.0,
            month_end_multiplier=3.0,
            quarter_end_multiplier=1.5,
            computed_at="2026-06-13T00:00:00",
        )
        with patch("core.behavioral_baseline.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 30, 12, 0, 0, tzinfo=timezone.utc)
            zscore = compute_behavior_zscore(identity, profile)
        assert zscore == pytest.approx(1.0, abs=0.1)

    def test_zscore_zero_when_std_zero(self) -> None:
        identity = self._make_identity(event_count=50)
        profile = BaselineProfile(
            role="Unique",
            department="Engineering",
            avg_events_30d=50.0,
            std_events_30d=0.0,
            avg_events_per_session=1.0,
            typical_access_hours=[12],
            typical_resources=[],
            avg_unique_resources_30d=0.0,
            std_unique_resources_30d=0.0,
            p95_events_single_session=0.0,
            p99_download_count=0.0,
            month_end_multiplier=1.0,
            quarter_end_multiplier=1.0,
            computed_at="2026-06-13",
        )
        zscore = compute_behavior_zscore(identity, profile)
        assert zscore == 0.0


# ── load_baselines tests ────────────────────────────────────────────────────

class TestLoadBaselines:

    def test_load_baselines_empty_dir(self, monkeypatch) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr("core.behavioral_baseline._BASELINES_DIR", tmpdir)
            loaded = load_baselines()
            assert loaded == {}
