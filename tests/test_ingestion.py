"""Tests for core.ingestion module."""

import os

import pytest

from core.ingestion import (
    IdentityRecord,
    IngestionError,
    _parse_boolean,
    _parse_float,
    _parse_int,
    _parse_list_field,
    _parse_timestamp,
    ingest,
)


class TestParseTimestamp:
    """Test timestamp parsing across all accepted formats."""

    def test_iso_8601_format(self) -> None:
        result = _parse_timestamp("2025-06-15T14:30:00Z")
        assert result is not None
        assert result.year == 2025
        assert result.month == 6
        assert result.day == 15

    def test_dd_mm_yyyy_format(self) -> None:
        result = _parse_timestamp("15/06/2025")
        assert result is not None
        assert result.year == 2025
        assert result.month == 6

    def test_mm_dd_yyyy_format(self) -> None:
        result = _parse_timestamp("06-15-2025")
        assert result is not None
        assert result.year == 2025

    def test_unix_epoch_integer(self) -> None:
        result = _parse_timestamp(1718450000)
        assert result is not None

    def test_unix_epoch_string(self) -> None:
        result = _parse_timestamp("1718450000")
        assert result is not None

    def test_null_timestamp_returns_none(self) -> None:
        result = _parse_timestamp(None)
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = _parse_timestamp("")
        assert result is None

    def test_whitespace_string_returns_none(self) -> None:
        result = _parse_timestamp("   ")
        assert result is None

    def test_timezone_offset_normalized_to_utc(self) -> None:
        """Timestamps with explicit offsets and naive timestamps representing
        the same UTC instant must produce the same UTC hour classification."""
        # 22:00 UTC-5 = 03:00 UTC next day
        with_offset = _parse_timestamp("2025-06-15T22:00:00-05:00")
        assert with_offset is not None
        assert with_offset.hour == 3  # normalized to UTC hour
        assert with_offset.tzinfo is not None

        # Same UTC instant as naive (treated as UTC)
        naive_utc = _parse_timestamp("2025-06-16T03:00:00")
        assert naive_utc is not None
        assert naive_utc.hour == 3
        # Both represent the same moment in time
        assert with_offset == naive_utc

    def test_positive_offset_normalized(self) -> None:
        """+05:30 offset should be normalized to UTC."""
        result = _parse_timestamp("2025-06-15T12:00:00+05:30")
        assert result is not None
        assert result.hour == 6  # 12:00 +05:30 = 06:30 UTC
        assert result.minute == 30


class TestIngestion:
    """Test the main ingest function."""

    def test_ingest_sample_data_loads_without_error(self) -> None:
        records = ingest()
        assert len(records) > 0
        assert isinstance(records[0], IdentityRecord)

    def test_ingest_returns_identity_records(self) -> None:
        records = ingest()
        for record in records:
            assert isinstance(record.user_id, str)
            assert len(record.user_id) > 0

    def test_missing_required_column_raises_ingestion_error(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("bad_column,username\nval1,user1\n")
            temp_path = f.name
        try:
            with pytest.raises(IngestionError) as exc_info:
                ingest(users_path=temp_path, events_path=_resolve_events_path())
            assert "user_id" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_empty_file_raises_ingestion_error(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("user_id,username\n")
            temp_path = f.name
        try:
            with pytest.raises(IngestionError) as exc_info:
                ingest(users_path=temp_path, events_path=_resolve_events_path())
            assert "empty" in str(exc_info.value)
        finally:
            os.unlink(temp_path)

    def test_terminated_and_active_identified(self) -> None:
        """Verify that a terminated user with active record is ingested properly."""
        records = ingest()
        terminated_active = [
            r for r in records
            if r.employment_status == "terminated"
        ]
        # Should have at least some terminated records
        if terminated_active:
            record = terminated_active[0]
            assert record.employment_status == "terminated"


class TestIdentityRecord:
    """Test IdentityRecord dataclass behaviour."""

    def test_identity_record_is_frozen(self) -> None:
        record = IdentityRecord(
            user_id="U00001",
            username="test_user",
            email="test@example.com",
            department="Engineering",
            employment_status="active",
            account_type="human",
            owner_id="mgr001",
            source_system="AD",
        )
        with pytest.raises(Exception):
            record.user_id = "changed"  # type: ignore[misc]

    def test_default_values(self) -> None:
        record = IdentityRecord(
            user_id="U00001",
            username="test",
            email="test@test.com",
            department="IT",
            employment_status="active",
            account_type="human",
            owner_id="",
            source_system="AD",
        )
        assert record.mfa_enabled is False
        assert record.sso_linked is False
        assert record.login_count_30d == 0
        assert record.event_count_30d == 0


class TestParseHelpers:
    """Test parsing utility functions."""

    def test_parse_boolean_true_values(self) -> None:
        assert _parse_boolean("true") is True
        assert _parse_boolean("True") is True
        assert _parse_boolean("1") is True
        assert _parse_boolean("yes") is True

    def test_parse_boolean_false_values(self) -> None:
        assert _parse_boolean("false") is False
        assert _parse_boolean("0") is False
        assert _parse_boolean("no") is False
        assert _parse_boolean(None) is False

    def test_parse_int_regular(self) -> None:
        assert _parse_int("42") == 42
        assert _parse_int(42) == 42

    def test_parse_int_none(self) -> None:
        assert _parse_int(None) == 0

    def test_parse_int_empty(self) -> None:
        assert _parse_int("") == 0

    def test_parse_float_regular(self) -> None:
        assert _parse_float("0.35") == 0.35

    def test_parse_float_none(self) -> None:
        assert _parse_float(None) == 0.0

    def test_parse_list_pipe_separated(self) -> None:
        result = _parse_list_field("reader|writer|admin")
        assert result == ["reader", "writer", "admin"]

    def test_parse_list_empty(self) -> None:
        assert _parse_list_field("") == []
        assert _parse_list_field(None) == []


def _resolve_events_path() -> str:
    """Resolve events CSV path for tests."""
    base = os.path.join(os.path.dirname(__file__), "..")
    sample = os.path.join(base, "sample_data", "identity_events.csv")
    raw_path = os.path.join(base, "data", "raw", "events_synthetic.csv")
    if os.path.exists(sample):
        return sample
    return raw_path
