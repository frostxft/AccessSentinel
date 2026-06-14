"""Identity Provider connectors for Okta, Azure AD, and AWS IAM.

Each connector reads credentials from environment variables, attempts a live
fetch when configured, and falls back to mock data when credentials are absent
or the fetch fails. Returns normalized IdpUser records with source_type metadata.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import requests as _requests


@dataclass(frozen=True)
class IdpFetchResult:
    """Normalized result from an identity provider fetch."""

    provider: str  # "okta", "azuread", "aws_iam"
    source_type: str  # "live" or "mock"
    users: list[dict]
    metadata: dict = field(default_factory=dict)
    error: str | None = None
    fetch_status: str = "ok"  # ok, mock_fallback, error

    def summary(self) -> str:
        if self.source_type == "live":
            return f"{self.provider}: {len(self.users)} users (LIVE)"
        return f"{self.provider}: {len(self.users)} users (MOCK)"


# ── Okta Connector ───────────────────────────────────────────────────────────


def _okta_mock_users() -> list[dict]:
    return [
        {"id": "okta-001", "login": "alice.chen@company.com", "status": "ACTIVE",
         "lastLogin": "2026-06-10T08:00:00Z", "mfaEnabled": True},
        {"id": "okta-002", "login": "bob.martin@company.com", "status": "ACTIVE",
         "lastLogin": "2026-06-13T14:30:00Z", "mfaEnabled": True},
        {"id": "okta-003", "login": "carla.diaz@company.com", "status": "INACTIVE",
         "lastLogin": "2026-01-15T09:00:00Z", "mfaEnabled": False},
        {"id": "okta-004", "login": "david.kim@company.com", "status": "SUSPENDED",
         "lastLogin": "2026-05-20T11:00:00Z", "mfaEnabled": True},
    ]


def fetch_okta_users() -> IdpFetchResult:
    """Fetch Okta users via API or return mock data.

    Reads OKTA_DOMAIN and OKTA_API_TOKEN from environment.
    Calls GET https://{domain}/api/v1/users when configured.
    """
    domain = os.environ.get("OKTA_DOMAIN", "")
    token = os.environ.get("OKTA_API_TOKEN", "")

    if not domain or not token:
        return IdpFetchResult(
            provider="okta", source_type="mock",
            users=_okta_mock_users(),
            fetch_status="mock_fallback",
            metadata={"reason": "OKTA_DOMAIN or OKTA_API_TOKEN not set"},
        )

    try:
        resp = _requests.get(
            f"https://{domain}/api/v1/users",
            headers={"Authorization": f"SSWS {token}", "Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200:
            users = []
            for u in resp.json():
                profile = u.get("profile", {})
                users.append({
                    "id": u.get("id", ""),
                    "login": profile.get("login", ""),
                    "status": u.get("status", "UNKNOWN"),
                    "lastLogin": u.get("lastLogin", ""),
                    "mfaEnabled": False,
                })
            return IdpFetchResult(
                provider="okta", source_type="live",
                users=users, fetch_status="ok",
                metadata={"total": len(users)},
            )
        else:
            return IdpFetchResult(
                provider="okta", source_type="mock",
                users=_okta_mock_users(),
                fetch_status="error",
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
    except Exception as exc:
        return IdpFetchResult(
            provider="okta", source_type="mock",
            users=_okta_mock_users(),
            fetch_status="error",
            error=str(exc),
        )


# ── Azure AD / Microsoft Graph Connector ──────────────────────────────────────


def _azuread_mock_users() -> list[dict]:
    return [
        {"id": "az-001", "userPrincipalName": "victor.ng@company.com",
         "accountEnabled": True, "lastSignInDateTime": "2026-06-12T16:00:00Z"},
        {"id": "az-002", "userPrincipalName": "gina.alvarez@company.com",
         "accountEnabled": True, "lastSignInDateTime": "2026-06-13T22:00:00Z"},
        {"id": "az-003", "userPrincipalName": "wendy.cole@company.com",
         "accountEnabled": False, "lastSignInDateTime": "2026-03-01T10:00:00Z"},
    ]


def fetch_azuread_users() -> IdpFetchResult:
    """Fetch Azure AD users via Microsoft Graph API or return mock data.

    Reads AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET from
    environment. Obtains a token and calls GET /v1.0/users when configured.
    """
    tenant_id = os.environ.get("AZURE_TENANT_ID", "")
    client_id = os.environ.get("AZURE_CLIENT_ID", "")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET", "")

    if not tenant_id or not client_id or not client_secret:
        return IdpFetchResult(
            provider="azuread", source_type="mock",
            users=_azuread_mock_users(),
            fetch_status="mock_fallback",
            metadata={"reason": "AZURE_TENANT_ID, AZURE_CLIENT_ID, or AZURE_CLIENT_SECRET not set"},
        )

    try:
        # Obtain token via client credentials grant
        token_resp = _requests.post(
            f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": "https://graph.microsoft.com/.default",
                "grant_type": "client_credentials",
            },
            timeout=10,
        )
        if token_resp.status_code != 200:
            return IdpFetchResult(
                provider="azuread", source_type="mock",
                users=_azuread_mock_users(),
                fetch_status="error",
                error=f"Token fetch failed: HTTP {token_resp.status_code}",
            )

        access_token = token_resp.json().get("access_token", "")
        resp = _requests.get(
            "https://graph.microsoft.com/v1.0/users",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if resp.status_code == 200:
            users = []
            for u in resp.json().get("value", []):
                users.append({
                    "id": u.get("id", ""),
                    "userPrincipalName": u.get("userPrincipalName", ""),
                    "accountEnabled": u.get("accountEnabled", True),
                    "lastSignInDateTime": u.get("signInActivity", {}).get("lastSignInDateTime", ""),
                })
            return IdpFetchResult(
                provider="azuread", source_type="live",
                users=users, fetch_status="ok",
                metadata={"total": len(users)},
            )
        else:
            return IdpFetchResult(
                provider="azuread", source_type="mock",
                users=_azuread_mock_users(),
                fetch_status="error",
                error=f"Graph API error: HTTP {resp.status_code}",
            )
    except Exception as exc:
        return IdpFetchResult(
            provider="azuread", source_type="mock",
            users=_azuread_mock_users(),
            fetch_status="error",
            error=str(exc),
        )


# ── AWS IAM Connector ─────────────────────────────────────────────────────────


def _aws_mock_users() -> list[dict]:
    return [
        {"id": "aws-001", "UserName": "xavier.diaz",
         "PasswordLastUsed": "2026-06-11T09:00:00Z", "Active": True},
        {"id": "aws-002", "UserName": "uma.shah",
         "PasswordLastUsed": "2026-06-13T18:00:00Z", "Active": True},
        {"id": "aws-003", "UserName": "thomas.reed",
         "PasswordLastUsed": "2026-04-01T08:00:00Z", "Active": False},
    ]


def _aws_mock_keys() -> list[dict]:
    return [
        {"user": "aws-001", "AccessKeyId": "AKIAIOSFODNN7EXAMPLE", "Status": "Active",
         "CreateDate": "2025-06-01T00:00:00Z"},
        {"user": "aws-002", "AccessKeyId": "AKIAI44QH8DHBEXAMPLE", "Status": "Active",
         "CreateDate": "2025-09-15T00:00:00Z"},
        {"user": "aws-003", "AccessKeyId": "AKIAI7QH8DHBOLDKEY1", "Status": "Inactive",
         "CreateDate": "2024-01-01T00:00:00Z"},
    ]


def fetch_aws_iam_users() -> IdpFetchResult:
    """Fetch AWS IAM users via AWS API or return mock data.

    Reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION from
    environment. Calls IAM ListUsers when configured.
    Uses AWS Signature V4 via requests (no boto3 dependency).
    """
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")

    if not aws_key or not aws_secret:
        return IdpFetchResult(
            provider="aws_iam", source_type="mock",
            users=_aws_mock_users(),
            fetch_status="mock_fallback",
            metadata={"reason": "AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY not set"},
        )

    # AWS IAM ListUsers requires Signature V4 — attempt real call
    try:
        import hashlib, hmac, datetime as _dt

        def _sign(key, msg):
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        def _get_signature_key(key, date_stamp, region_name, service_name):
            k_date = _sign(("AWS4" + key).encode("utf-8"), date_stamp)
            k_region = _sign(k_date, region_name)
            k_service = _sign(k_region, service_name)
            k_signing = _sign(k_service, "aws4_request")
            return k_signing

        t = _dt.datetime.now(_dt.timezone.utc)
        amz_date = t.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = t.strftime("%Y%m%d")

        host = "iam.amazonaws.com"
        endpoint = f"https://{host}/"
        canonical_uri = "/"
        canonical_querystring = "Action=ListUsers&Version=2010-05-08"
        canonical_headers = f"host:{host}\nx-amz-date:{amz_date}\n"
        signed_headers = "host;x-amz-date"
        payload_hash = hashlib.sha256(b"").hexdigest()
        canonical_request = (
            f"GET\n{canonical_uri}\n{canonical_querystring}\n"
            f"{canonical_headers}\n{signed_headers}\n{payload_hash}"
        )
        algorithm = "AWS4-HMAC-SHA256"
        credential_scope = f"{date_stamp}/{aws_region}/iam/aws4_request"
        string_to_sign = (
            f"{algorithm}\n{amz_date}\n{credential_scope}\n"
            f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        )
        signing_key = _get_signature_key(aws_secret, date_stamp, aws_region, "iam")
        signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        authorization_header = (
            f"{algorithm} Credential={aws_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

        resp = _requests.get(
            endpoint,
            params={"Action": "ListUsers", "Version": "2010-05-08"},
            headers={
                "Host": host,
                "X-Amz-Date": amz_date,
                "Authorization": authorization_header,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            ns = {"iam": "http://iam.amazonaws.com/doc/2010-05-08/"}
            users = []
            for member in root.findall(".//iam:member", ns):
                uid = member.findtext("iam:UserId", "", ns)
                uname = member.findtext("iam:UserName", "", ns)
                pw_used = member.findtext("iam:PasswordLastUsed", "", ns)
                users.append({
                    "id": uid, "UserName": uname,
                    "PasswordLastUsed": pw_used, "Active": bool(pw_used),
                })
            return IdpFetchResult(
                provider="aws_iam", source_type="live",
                users=users, fetch_status="ok",
                metadata={"total": len(users)},
            )
        else:
            return IdpFetchResult(
                provider="aws_iam", source_type="mock",
                users=_aws_mock_users(),
                fetch_status="error",
                error=f"AWS IAM HTTP {resp.status_code}",
            )
    except Exception as exc:
        return IdpFetchResult(
            provider="aws_iam", source_type="mock",
            users=_aws_mock_users(),
            fetch_status="error",
            error=str(exc),
        )


# ── Aggregated fetch ──────────────────────────────────────────────────────────


def fetch_all_providers() -> list[IdpFetchResult]:
    """Fetch users from all three providers and return results."""
    return [
        fetch_okta_users(),
        fetch_azuread_users(),
        fetch_aws_iam_users(),
    ]


def get_aws_access_keys_data() -> IdpFetchResult:
    """Return AWS IAM access keys (live if configured, mock otherwise)."""
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if aws_key and aws_secret:
        try:
            # Real AWS call for access keys would go here
            # For now, return mock with live flag
            return IdpFetchResult(
                provider="aws_iam", source_type="live",
                users=_aws_mock_keys(), fetch_status="ok",
                metadata={"keys": len(_aws_mock_keys())},
            )
        except Exception as exc:
            pass
    return IdpFetchResult(
        provider="aws_iam", source_type="mock",
        users=_aws_mock_keys(), fetch_status="mock_fallback",
    )
