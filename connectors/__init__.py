"""Shared normalized identity schema for all IdP connectors.

All connectors must map their provider-specific responses into this schema
so downstream consumers can work with a consistent data shape regardless
of the source provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class NormalizedIdentity:
    """Normalized identity record from any identity provider."""

    provider: str  # "okta", "azuread", "aws_iam"
    source_type: str  # "live" or "mock"
    external_id: str
    username: str = ""
    email: str = ""
    display_name: str = ""
    account_type: str = "human"  # human, service, bot
    privilege_level: str = "user"  # user, admin, power-user, service-account
    status: str = "active"  # active, inactive, suspended, terminated
    last_login: str | None = None
    groups: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    raw_attributes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedEvent:
    """Normalized event/activity record from any identity provider."""

    provider: str
    source_type: str
    event_id: str
    actor: str  # external_id or username
    target: str = ""
    event_type: str = "unknown"
    timestamp: str = ""
    source_system: str = ""
    ip_address: str = ""
    resource: str = ""
    sensitivity: str = "low"
    raw_attributes: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectorResult:
    """Result from a connector fetch operation."""

    provider: str
    source_type: str  # "live" or "mock"
    fetch_status: str  # "ok", "mock_fallback", "error"
    identities: list[NormalizedIdentity]
    events: list[NormalizedEvent] = field(default_factory=list)
    error: str | None = None
    metadata: dict = field(default_factory=dict)
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary(self) -> str:
        label = "LIVE" if self.source_type == "live" else "MOCK"
        return f"{self.provider}: {len(self.identities)} users ({label})"


def make_mock_result(
    provider: str,
    identities: list[NormalizedIdentity],
    reason: str = "credentials not configured",
) -> ConnectorResult:
    """Build a ConnectorResult for mock fallback mode."""
    return ConnectorResult(
        provider=provider,
        source_type="mock",
        fetch_status="mock_fallback",
        identities=identities,
        error=None,
        metadata={"reason": reason},
    )


def make_error_result(
    provider: str,
    identities: list[NormalizedIdentity],
    error: str,
) -> ConnectorResult:
    """Build a ConnectorResult for a failed live fetch with mock fallback."""
    return ConnectorResult(
        provider=provider,
        source_type="mock",
        fetch_status="error",
        identities=identities,
        error=error,
        metadata={"reason": f"Live fetch failed: {error}"},
    )


def make_live_result(
    provider: str,
    identities: list[NormalizedIdentity],
    metadata: dict | None = None,
) -> ConnectorResult:
    """Build a ConnectorResult for a successful live fetch."""
    return ConnectorResult(
        provider=provider,
        source_type="live",
        fetch_status="ok",
        identities=identities,
        metadata=metadata or {},
    )
