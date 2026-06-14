"""Okta identity provider connector.

Fetches users from Okta REST API (GET /api/v1/users) using an SSWS token.
Falls back to mock data when OKTA_ORG_URL or OKTA_API_TOKEN is not configured.
Handles pagination via Link headers.
"""

from __future__ import annotations

import os
import requests as _requests

from connectors import (
    ConnectorResult,
    NormalizedIdentity,
    make_live_result,
    make_mock_result,
    make_error_result,
)

_OKTA_MOCK_USERS = [
    NormalizedIdentity(
        provider="okta", source_type="mock", external_id="okta-001",
        username="alice.chen", email="alice.chen@company.com",
        status="active", privilege_level="admin", groups=["Administrators"],
    ),
    NormalizedIdentity(
        provider="okta", source_type="mock", external_id="okta-002",
        username="bob.martin", email="bob.martin@company.com",
        status="active", privilege_level="user", groups=["Engineering"],
    ),
    NormalizedIdentity(
        provider="okta", source_type="mock", external_id="okta-003",
        username="carla.diaz", email="carla.diaz@company.com",
        status="inactive", privilege_level="user", groups=["Finance"],
    ),
    NormalizedIdentity(
        provider="okta", source_type="mock", external_id="okta-004",
        username="david.kim", email="david.kim@company.com",
        status="suspended", privilege_level="user", groups=["Engineering"],
    ),
]


def is_configured() -> bool:
    """Check whether Okta credentials are configured in the environment."""
    return bool(os.environ.get("OKTA_ORG_URL") and os.environ.get("OKTA_API_TOKEN"))


def fetch_identities() -> ConnectorResult:
    """Fetch Okta users via REST API or return mock data.

    Expects OKTA_ORG_URL (e.g. https://dev-123456.okta.com) and
    OKTA_API_TOKEN (SSWS token) in the environment.
    Handles pagination via the Link header.
    """
    org_url = os.environ.get("OKTA_ORG_URL", "").rstrip("/")
    token = os.environ.get("OKTA_API_TOKEN", "")

    if not org_url or not token:
        return make_mock_result("okta", _OKTA_MOCK_USERS)

    try:
        all_users: list[dict] = []
        url = f"{org_url}/api/v1/users?limit=200"

        while url:
            resp = _requests.get(
                url,
                headers={
                    "Authorization": f"SSWS {token}",
                    "Accept": "application/json",
                },
                timeout=30,
            )
            if resp.status_code != 200:
                return make_error_result(
                    "okta", _OKTA_MOCK_USERS,
                    f"HTTP {resp.status_code}: {resp.text[:200]}",
                )

            users = resp.json()
            all_users.extend(users)

            # Pagination via Link header (RFC 5988)
            link = resp.headers.get("Link", "")
            url = ""
            if 'rel="next"' in link:
                for part in link.split(","):
                    if 'rel="next"' in part:
                        url = part.split(";")[0].strip(" <>")
                        break

        normalized = [_normalize_okta_user(u) for u in all_users]
        return make_live_result("okta", normalized, {"total_fetched": len(normalized)})

    except Exception as exc:
        return make_error_result("okta", _OKTA_MOCK_USERS, str(exc))


def _normalize_okta_user(user: dict) -> NormalizedIdentity:
    """Map an Okta user dict to NormalizedIdentity."""
    profile = user.get("profile", {})
    status = user.get("status", "").lower()
    return NormalizedIdentity(
        provider="okta",
        source_type="live",
        external_id=user.get("id", ""),
        username=profile.get("login", "").split("@")[0] if "@" in profile.get("login", "") else profile.get("login", ""),
        email=profile.get("email", profile.get("login", "")),
        display_name=f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip(),
        status="active" if status == "active" else "inactive" if status in ("inactive", "deprovisioned") else "suspended",
        last_login=user.get("lastLogin"),
        groups=[],
        raw_attributes={"status": user.get("status"), "created": user.get("created")},
    )


def healthcheck() -> dict:
    """Return Okta connector health status.
    
    Status: not_configured | live | error
    """
    if not is_configured():
        return {"provider": "okta", "configured": False, "status": "not_configured"}
    try:
        resp = _requests.get(
            f"{os.environ['OKTA_ORG_URL'].rstrip('/')}/api/v1/users?limit=1",
            headers={"Authorization": f"SSWS {os.environ['OKTA_API_TOKEN']}",
                     "Accept": "application/json"},
            timeout=10,
        )
        return {
            "provider": "okta", "configured": True,
            "status": "live" if resp.status_code == 200 else "error",
            "http_status": resp.status_code,
        }
    except Exception as exc:
        return {"provider": "okta", "configured": True, "status": "error", "error": str(exc)}
