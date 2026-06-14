"""AWS IAM identity provider connector.

Fetches IAM users and access keys via Boto3 when available, or via
manual AWS Signature V4 HTTP requests as a fallback. Returns mock
data when AWS credentials are not configured.
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

_AWS_MOCK_USERS = [
    NormalizedIdentity(
        provider="aws_iam", source_type="mock", external_id="aws-001",
        username="xavier.diaz", email="xavier.diaz@company.com",
        status="active", privilege_level="admin",
    ),
    NormalizedIdentity(
        provider="aws_iam", source_type="mock", external_id="aws-002",
        username="uma.shah", email="uma.shah@company.com",
        status="active", privilege_level="user",
    ),
    NormalizedIdentity(
        provider="aws_iam", source_type="mock", external_id="aws-003",
        username="thomas.reed", email="thomas.reed@company.com",
        status="inactive", privilege_level="user",
    ),
]

_AWS_MOCK_KEYS = [
    {"AccessKeyId": "AKIAIOSFODNN7EXAMPLE", "UserName": "xavier.diaz", "Status": "Active"},
    {"AccessKeyId": "AKIAI44QH8DHBEXAMPLE", "UserName": "uma.shah", "Status": "Active"},
    {"AccessKeyId": "AKIAI7QH8DHBOLDKEY1", "UserName": "thomas.reed", "Status": "Inactive"},
]


def is_configured() -> bool:
    """Check whether AWS credentials are explicitly configured via env vars.

    Only returns True when both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
    are set in the environment. Boto3's auto-detection chain (~/.aws/credentials,
    IAM instance roles) is intentionally NOT used here — those are checked
    at fetch time instead, to avoid showing a false LIVE status when the
    user has a default AWS profile but hasn't configured AccessSentinel.
    """
    return bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        and os.environ.get("AWS_SECRET_ACCESS_KEY")
    )


def _use_boto3() -> bool:
    """Check if Boto3 is available and can be used."""
    try:
        import boto3
        return True
    except ImportError:
        return False


def fetch_identities() -> ConnectorResult:
    """Fetch AWS IAM users via Boto3 or Signature V4 HTTP.

    Uses Boto3 if available; otherwise falls back to manual AWS API calls.
    Returns mock data when credentials are absent.
    """
    if not is_configured():
        return make_mock_result("aws_iam", _AWS_MOCK_USERS)

    if _use_boto3():
        return _fetch_via_boto3()

    return _fetch_via_requests()


def _fetch_via_boto3() -> ConnectorResult:
    """Fetch IAM users using Boto3 SDK."""
    try:
        import boto3
        iam = boto3.client("iam")
        all_users: list[dict] = []
        paginator = iam.get_paginator("list_users")
        for page in paginator.paginate():
            all_users.extend(page.get("Users", []))

        normalized = [_normalize_iam_user(u) for u in all_users]
        return make_live_result("aws_iam", normalized, {"total_fetched": len(normalized)})
    except Exception as exc:
        return make_error_result("aws_iam", _AWS_MOCK_USERS, str(exc))


def _fetch_via_requests() -> ConnectorResult:
    """Fetch IAM users via AWS Signature V4 HTTP requests (no Boto3)."""
    import hashlib, hmac, datetime as _dt

    aws_key = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region = os.environ.get("AWS_REGION", "us-east-1")

    def _sign(key, msg):
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    def _get_sig_key(key, date_stamp, region, service):
        k_date = _sign(("AWS4" + key).encode("utf-8"), date_stamp)
        k_region = _sign(k_date, region)
        k_service = _sign(k_region, service)
        return _sign(k_service, "aws4_request")

    t = _dt.datetime.now(_dt.timezone.utc)
    amz_date = t.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = t.strftime("%Y%m%d")
    host = "iam.amazonaws.com"
    canonical_querystring = "Action=ListUsers&Version=2010-05-08"
    canonical_headers = f"host:{host}\nx-amz-date:{amz_date}\n"
    signed_headers = "host;x-amz-date"
    payload_hash = hashlib.sha256(b"").hexdigest()
    canonical_request = (
        f"GET\n/\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )
    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{date_stamp}/{aws_region}/iam/aws4_request"
    string_to_sign = (
        f"{algorithm}\n{amz_date}\n{credential_scope}\n"
        f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
    )
    signing_key = _get_sig_key(aws_secret, date_stamp, aws_region, "iam")
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    auth_header = (
        f"{algorithm} Credential={aws_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    try:
        resp = _requests.get(
            "https://iam.amazonaws.com/",
            params={"Action": "ListUsers", "Version": "2010-05-08"},
            headers={"Host": host, "X-Amz-Date": amz_date, "Authorization": auth_header},
            timeout=30,
        )
        if resp.status_code == 200:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            ns = {"iam": "http://iam.amazonaws.com/doc/2010-05-08/"}
            users = []
            for member in root.findall(".//iam:member", ns):
                users.append({
                    "UserId": member.findtext("iam:UserId", "", ns),
                    "UserName": member.findtext("iam:UserName", "", ns),
                    "PasswordLastUsed": member.findtext("iam:PasswordLastUsed", "", ns),
                    "CreateDate": member.findtext("iam:CreateDate", "", ns),
                })
            normalized = [_normalize_iam_user(u) for u in users]
            return make_live_result("aws_iam", normalized, {"total_fetched": len(normalized)})
        return make_error_result("aws_iam", _AWS_MOCK_USERS, f"AWS HTTP {resp.status_code}")
    except Exception as exc:
        return make_error_result("aws_iam", _AWS_MOCK_USERS, str(exc))


def _normalize_iam_user(user: dict) -> NormalizedIdentity:
    """Map an AWS IAM user dict to NormalizedIdentity."""
    name = user.get("UserName", "")
    pw_used = user.get("PasswordLastUsed", "")
    return NormalizedIdentity(
        provider="aws_iam",
        source_type="live",
        external_id=user.get("UserId", user.get("Arn", "")),
        username=name,
        email=f"{name}@company.com" if name else "",
        display_name=name,
        status="active" if pw_used else "inactive",
        last_login=str(pw_used) if pw_used else None,
        groups=[],
        roles=[],
        raw_attributes={"CreateDate": str(user.get("CreateDate", ""))},
    )


def fetch_access_keys() -> dict:
    """Fetch AWS IAM access keys (live or mock)."""
    if is_configured() and _use_boto3():
        try:
            import boto3
            iam = boto3.client("iam")
            all_keys = []
            users_resp = iam.list_users()
            for user in users_resp.get("Users", []):
                keys = iam.list_access_keys(UserName=user["UserName"])
                for k in keys.get("AccessKeyMetadata", []):
                    all_keys.append({
                        "AccessKeyId": k.get("AccessKeyId", ""),
                        "UserName": user.get("UserName", ""),
                        "Status": k.get("Status", "Inactive"),
                    })
            return {
                "source_type": "live",
                "keys": all_keys,
                "total": len(all_keys),
                "active": sum(1 for k in all_keys if k["Status"] == "Active"),
            }
        except Exception:
            pass
    return {
        "source_type": "mock",
        "keys": _AWS_MOCK_KEYS,
        "total": len(_AWS_MOCK_KEYS),
        "active": sum(1 for k in _AWS_MOCK_KEYS if k["Status"] == "Active"),
    }


def quick_status() -> dict:
    """Return fast, lightweight status (no AWS API calls).

    Checks only env-var presence. Returns 'configured' if env vars are set,
    'not_configured' otherwise. Does NOT call STS, IAM, or any AWS API.
    Used by GET /integrations/status for fast page loads.
    """
    configured = is_configured()
    return {
        "provider": "aws_iam",
        "configured": configured,
        "status": "configured" if configured else "not_configured",
        "http_status": None,
        "account": None,
        "account_alias": None,
        "arn": None,
        "user_id": None,
        "iam_user_count": None,
        "sample_users": [],
        "credential_report_summary": None,
        "heuristics": {},
        "error": None,
        "warning": None,
    }


def healthcheck() -> dict:
    """Return fast health status (delegates to quick_status for page loads)."""
    return quick_status()


def connection_test() -> dict:
    """Verify live AWS connectivity via STS GetCallerIdentity + optional IAM inventory.

    Returns a dict with provider, status, configured, http_status, account,
    arn, user_id, iam_user_count, sample_usernames, error, and warning fields.
    """
    from datetime import datetime, timezone

    base = {
        "provider": "aws_iam",
        "configured": is_configured(),
        "status": "not_configured",
        "http_status": None,
        "account": None,
        "account_alias": None,
        "arn": None,
        "user_id": None,
        "iam_user_count": None,
        "sample_users": [],
        "credential_report_summary": None,
        "heuristics": {},
        "error": None,
        "warning": None,
        "last_tested_at": datetime.now(timezone.utc).isoformat(),
    }

    if not base["configured"]:
        return base

    base["status"] = "configured"

    # ── Import boto3 ──────────────────────────────────────────────────────
    try:
        import boto3
    except ImportError:
        base["status"] = "error"
        base["error"] = "boto3 is not installed. Install with: pip install boto3"
        return base

    # ── Primary verification: STS GetCallerIdentity ──────────────────────
    try:
        from botocore.exceptions import BotoCoreError, ClientError

        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        base["status"] = "live"
        base["http_status"] = 200
        base["account"] = identity.get("Account", "")
        base["arn"] = identity.get("Arn", "")
        base["user_id"] = identity.get("UserId", "")

        # ── Enrichment (all non-fatal) ───────────────────────────────────
        warnings = []
        iam = boto3.client("iam")

        _enrich_account_alias(base, iam, warnings)
        _enrich_iam_users(base, iam, warnings)
        _enrich_credential_report(base, iam, warnings)
        _derive_heuristics(base)

        if warnings:
            base["warning"] = "; ".join(warnings)

    except (BotoCoreError, ClientError) as exc:
        base["status"] = "error"
        base["error"] = _sanitize_error(exc)
        base["http_status"] = _extract_http_status(exc)
    except Exception as exc:
        base["status"] = "error"
        base["error"] = str(exc)[:200]

    return base


# ── Enrichment helpers ────────────────────────────────────────────────────────


def _enrich_account_alias(base: dict, iam, warnings: list[str]) -> None:
    """Populate account_alias from IAM ListAccountAliases."""
    try:
        resp = iam.list_account_aliases()
        aliases = resp.get("AccountAliases", [])
        base["account_alias"] = aliases[0] if aliases else None
    except Exception as exc:
        base["account_alias"] = None
        warnings.append(f"Account alias unavailable: {_sanitize_error(exc)}")


def _enrich_iam_users(base: dict, iam, warnings: list[str]) -> None:
    """Populate iam_user_count, sample_users from paginated ListUsers."""
    try:
        paginator = iam.get_paginator("list_users")
        all_users = []
        for page in paginator.paginate():
            all_users.extend(page.get("Users", []))
        base["iam_user_count"] = len(all_users)

        sample_users = []
        for user in all_users[:5]:
            entry = {
                "user_name": user.get("UserName", ""),
                "create_date": _iso(user.get("CreateDate")),
                "password_last_used": _iso(user.get("PasswordLastUsed")),
                "attached_policy_count": 0,
                "inline_policy_count": 0,
                "attached_policy_names": [],
                "inline_policy_names": [],
            }
            _enrich_user_policies(entry, iam, user.get("UserName", ""))
            sample_users.append(entry)
        base["sample_users"] = sample_users
    except Exception as exc:
        base["iam_user_count"] = None
        base["sample_users"] = []
        warnings.append(f"IAM user list unavailable: {_sanitize_error(exc)}")


def _enrich_user_policies(entry: dict, iam, username: str) -> None:
    """Populate per-user policy counts and names."""
    try:
        attached = iam.list_attached_user_policies(UserName=username)
        policies = attached.get("AttachedPolicies", [])
        entry["attached_policy_count"] = len(policies)
        entry["attached_policy_names"] = [p.get("PolicyName", "") for p in policies[:5]]
    except Exception:
        pass

    try:
        inline = iam.list_user_policies(UserName=username)
        names = inline.get("PolicyNames", [])
        entry["inline_policy_count"] = len(names)
        entry["inline_policy_names"] = list(names[:5])
    except Exception:
        pass


def _enrich_credential_report(base: dict, iam, warnings: list[str]) -> None:
    """Generate, retrieve, and parse the IAM credential report into summary metrics."""
    base["credential_report_summary"] = None
    try:
        iam.generate_credential_report()
        import time, io, csv as _csv
        for _ in range(15):
            resp = iam.get_credential_report()
            if resp.get("State") == "COMPLETE":
                break
            time.sleep(2)
        if resp.get("State") != "COMPLETE":
            warnings.append(f"Credential report state: {resp.get('State', 'unknown')}")
            return

        content = resp.get("Content", b"")
        if isinstance(content, str):
            content = content.encode("utf-8")
        reader = _csv.DictReader(io.StringIO(content.decode("utf-8")))
        rows = list(reader)

        metrics = {
            "generated_time": _iso(resp.get("GeneratedTime")),
            "total_report_users": len(rows),
            "users_with_mfa_disabled": 0,
            "users_with_password_enabled": 0,
            "users_with_active_access_keys": 0,
            "users_with_access_key_1_active": 0,
            "users_with_access_key_2_active": 0,
            "users_with_password_unused_90d": 0,
            "users_with_access_key_1_unused_90d": 0,
            "users_with_access_key_2_unused_90d": 0,
        }

        cutoff = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        cutoff = cutoff.replace(hour=0, minute=0, second=0, microsecond=0) - __import__("datetime").timedelta(days=90)

        for row in rows:
            if row.get("mfa_active", "").lower() == "false":
                metrics["users_with_mfa_disabled"] += 1
            if row.get("password_enabled", "").lower() == "true":
                metrics["users_with_password_enabled"] += 1
            if row.get("access_key_1_active", "").lower() == "true":
                metrics["users_with_active_access_keys"] += 1
                metrics["users_with_access_key_1_active"] += 1
            if row.get("access_key_2_active", "").lower() == "true":
                metrics["users_with_active_access_keys"] += 1
                metrics["users_with_access_key_2_active"] += 1
            if _days_since(row.get("password_last_used", ""), cutoff):
                metrics["users_with_password_unused_90d"] += 1
            if _days_since(row.get("access_key_1_last_used_date", ""), cutoff):
                metrics["users_with_access_key_1_unused_90d"] += 1
            if _days_since(row.get("access_key_2_last_used_date", ""), cutoff):
                metrics["users_with_access_key_2_unused_90d"] += 1

        base["credential_report_summary"] = metrics
    except Exception as exc:
        warnings.append(f"Credential report unavailable: {_sanitize_error(exc)}")


def _days_since(date_str: str, cutoff) -> bool:
    """Return True if the date string is parseable and older than cutoff."""
    if not date_str or date_str.strip().lower() in ("n/a", "not_supported", "no_information", ""):
        return False
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt < cutoff
    except Exception:
        return False


def _derive_heuristics(base: dict) -> None:
    """Derive posture heuristics from sampled user policies and credential report."""
    heuristics = {
        "potentially_privileged_users": [],
        "mfa_gap_candidates": [],
        "credential_sprawl_candidates": [],
    }

    priv_keywords = ("admin", "poweruser", "fullaccess", "administrator")
    for u in base.get("sample_users", []) or []:
        names = [n.lower() for n in u.get("attached_policy_names", []) + u.get("inline_policy_names", [])]
        if any(kw in " ".join(names) for kw in priv_keywords):
            heuristics["potentially_privileged_users"].append(u.get("user_name", ""))

    cr = base.get("credential_report_summary") or {}
    if cr.get("users_with_mfa_disabled", 0) > 0:
        heuristics["mfa_gap_candidates"] = ["See credential report"]  # exact usernames require full CSV
    if cr.get("users_with_active_access_keys", 0) >= 2:
        heuristics["credential_sprawl_candidates"] = ["See credential report"]

    base["heuristics"] = heuristics


def _iso(val) -> str | None:
    """Convert a datetime to ISO string, or return None."""
    if val is None:
        return None
    try:
        return val.isoformat()
    except Exception:
        return str(val)[:30]


def _sanitize_error(exc: Exception) -> str:
    """Return a sanitized error message, stripping credentials and tokens."""
    msg = str(exc)
    for pattern in ("Signature=", "Credential=", "AWS4-", "AKIA", "ASIA"):
        if pattern in msg:
            return f"AWS error (details redacted): {type(exc).__name__}"
    return msg[:200]


def _extract_http_status(exc: Exception) -> int | None:
    """Extract HTTP status code from a BotoCore/ClientError if available."""
    try:
        return getattr(exc, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
    except Exception:
        return None
