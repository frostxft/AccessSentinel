"""Azure AD / Microsoft Graph identity provider connector.

Fetches users from Microsoft Graph API (GET /v1.0/users) using OAuth2
client credentials grant. Falls back to mock data when credentials are
not configured. Handles pagination via @odata.nextLink.
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

_AZUREAD_MOCK_USERS = [
    NormalizedIdentity(
        provider="azuread", source_type="mock", external_id="az-001",
        username="victor.ng", email="victor.ng@company.com",
        status="active", privilege_level="admin",
    ),
    NormalizedIdentity(
        provider="azuread", source_type="mock", external_id="az-002",
        username="gina.alvarez", email="gina.alvarez@company.com",
        status="active", privilege_level="user",
    ),
    NormalizedIdentity(
        provider="azuread", source_type="mock", external_id="az-003",
        username="wendy.cole", email="wendy.cole@company.com",
        status="inactive", privilege_level="user",
    ),
]


def is_configured() -> bool:
    """Check whether Azure AD credentials are configured."""
    return bool(
        os.environ.get("AZURE_TENANT_ID")
        and os.environ.get("AZURE_CLIENT_ID")
        and os.environ.get("AZURE_CLIENT_SECRET")
    )


def _acquire_token(tenant_id: str, client_id: str, client_secret: str) -> str | None:
    """Acquire an OAuth2 access token via client credentials grant."""
    try:
        resp = _requests.post(
            f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": "https://graph.microsoft.com/.default",
                "grant_type": "client_credentials",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("access_token")
        return None
    except Exception:
        return None


def fetch_identities() -> ConnectorResult:
    """Fetch Azure AD users via Microsoft Graph or return mock data.

    Expects AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET
    in the environment. Handles pagination via @odata.nextLink.
    """
    tenant_id = os.environ.get("AZURE_TENANT_ID", "")
    client_id = os.environ.get("AZURE_CLIENT_ID", "")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET", "")

    if not tenant_id or not client_id or not client_secret:
        return make_mock_result("azuread", _AZUREAD_MOCK_USERS)

    token = _acquire_token(tenant_id, client_id, client_secret)
    if not token:
        return make_error_result(
            "azuread", _AZUREAD_MOCK_USERS,
            "Failed to acquire Microsoft Graph access token",
        )

    try:
        all_users: list[dict] = []
        url = "https://graph.microsoft.com/v1.0/users?$top=200"

        while url:
            resp = _requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            if resp.status_code != 200:
                return make_error_result(
                    "azuread", _AZUREAD_MOCK_USERS,
                    f"Graph API HTTP {resp.status_code}: {resp.text[:200]}",
                )

            data = resp.json()
            all_users.extend(data.get("value", []))

            # Pagination via @odata.nextLink
            url = data.get("@odata.nextLink", "")

        normalized = [_normalize_azuread_user(u) for u in all_users]
        return make_live_result("azuread", normalized, {"total_fetched": len(normalized)})

    except Exception as exc:
        return make_error_result("azuread", _AZUREAD_MOCK_USERS, str(exc))


def _normalize_azuread_user(user: dict) -> NormalizedIdentity:
    """Map a Microsoft Graph user dict to NormalizedIdentity."""
    upn = user.get("userPrincipalName", "")
    return NormalizedIdentity(
        provider="azuread",
        source_type="live",
        external_id=user.get("id", ""),
        username=upn.split("@")[0] if "@" in upn else upn,
        email=user.get("mail", upn),
        display_name=user.get("displayName", ""),
        status="active" if user.get("accountEnabled", True) else "inactive",
        last_login=None,  # Requires signInActivity with AuditLog.Read.All
        groups=[],
        roles=[],
        raw_attributes={
            "jobTitle": user.get("jobTitle", ""),
            "department": user.get("department", ""),
        },
    )


def healthcheck() -> dict:
    """Return Azure AD connector health status.
    
    Status: not_configured | live | error
    """
    if not is_configured():
        return {"provider": "azuread", "configured": False, "status": "not_configured"}
    try:
        token = _acquire_token(
            os.environ["AZURE_TENANT_ID"],
            os.environ["AZURE_CLIENT_ID"],
            os.environ["AZURE_CLIENT_SECRET"],
        )
        if not token:
            return {"provider": "azuread", "configured": True, "status": "error", "error": "Token acquisition failed"}
        resp = _requests.get(
            "https://graph.microsoft.com/v1.0/users?$top=1",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        return {
            "provider": "azuread", "configured": True,
            "status": "live" if resp.status_code == 200 else "error",
            "http_status": resp.status_code,
        }
    except Exception as exc:
        return {"provider": "azuread", "configured": True, "status": "error", "error": str(exc)}
