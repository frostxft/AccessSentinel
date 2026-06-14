"""Mock Okta / Azure AD integration for AccessSentinel.

Provides mock-safe API hooks that simulate identity provider responses
for demo purposes. Replace with real API calls for production use.
"""

from dataclasses import dataclass

_MOCK_OKTA_USERS: list[dict] = [
    {"id": "okta-001", "login": "alice.chen@company.com", "status": "ACTIVE",
     "lastLogin": "2026-06-10T08:00:00Z", "mfaEnabled": True},
    {"id": "okta-002", "login": "bob.martin@company.com", "status": "ACTIVE",
     "lastLogin": "2026-06-13T14:30:00Z", "mfaEnabled": True},
    {"id": "okta-003", "login": "carla.diaz@company.com", "status": "INACTIVE",
     "lastLogin": "2026-01-15T09:00:00Z", "mfaEnabled": False},
    {"id": "okta-004", "login": "david.kim@company.com", "status": "SUSPENDED",
     "lastLogin": "2026-05-20T11:00:00Z", "mfaEnabled": True},
]

_MOCK_AZURE_USERS: list[dict] = [
    {"id": "az-001", "userPrincipalName": "victor.ng@company.com",
     "accountEnabled": True, "lastSignInDateTime": "2026-06-12T16:00:00Z"},
    {"id": "az-002", "userPrincipalName": "gina.alvarez@company.com",
     "accountEnabled": True, "lastSignInDateTime": "2026-06-13T22:00:00Z"},
    {"id": "az-003", "userPrincipalName": "wendy.cole@company.com",
     "accountEnabled": False, "lastSignInDateTime": "2026-03-01T10:00:00Z"},
]

_MOCK_OKTA_GROUPS: list[dict] = [
    {"id": "grp-admin", "name": "Administrators", "members": ["okta-001"]},
    {"id": "grp-eng", "name": "Engineering", "members": ["okta-002", "okta-004"]},
    {"id": "grp-finance", "name": "Finance", "members": ["okta-003"]},
]

_MOCK_AWS_USERS: list[dict] = [
    {"id": "aws-001", "UserName": "xavier.diaz", "ARN": "arn:aws:iam::123456789012:user/xavier.diaz",
     "PasswordLastUsed": "2026-06-11T09:00:00Z", "Active": True},
    {"id": "aws-002", "UserName": "uma.shah", "ARN": "arn:aws:iam::123456789012:user/uma.shah",
     "PasswordLastUsed": "2026-06-13T18:00:00Z", "Active": True},
    {"id": "aws-003", "UserName": "thomas.reed", "ARN": "arn:aws:iam::123456789012:user/thomas.reed",
     "PasswordLastUsed": "2026-04-01T08:00:00Z", "Active": False},
]

_MOCK_AWS_ACCESS_KEYS: list[dict] = [
    {"user": "aws-001", "AccessKeyId": "AKIAIOSFODNN7EXAMPLE", "Status": "Active",
     "CreateDate": "2025-06-01T00:00:00Z"},
    {"user": "aws-002", "AccessKeyId": "AKIAI44QH8DHBEXAMPLE", "Status": "Active",
     "CreateDate": "2025-09-15T00:00:00Z"},
    {"user": "aws-003", "AccessKeyId": "AKIAI7QH8DHBOLDKEY1", "Status": "Inactive",
     "CreateDate": "2024-01-01T00:00:00Z"},
]

# CLI command templates per provider
PROVIDER_CLI_COMMANDS: dict[str, dict[str, str]] = {
    "okta": {
        "disable_user": "okta users deactivate --login {email}",
        "reset_password": "okta users reset-password --login {email}",
        "remove_group": "okta groups remove-user --group {group} --login {email}",
    },
    "azuread": {
        "disable_user": "az ad user update --id {email} --account-enabled false",
        "reset_password": "az ad user update --id {email} --force-change-password-next-sign-in true",
        "remove_role": "az role assignment delete --assignee {email} --role {role}",
    },
    "aws_iam": {
        "disable_user": "aws iam delete-login-profile --user-name {username}",
        "deactivate_keys": "aws iam update-access-key --access-key-id {key_id} --status Inactive --user-name {username}",
        "remove_policy": "aws iam detach-user-policy --user-name {username} --policy-arn {policy_arn}",
    },
}


@dataclass(frozen=True)
class IdpUser:
    """A user entity from an identity provider."""

    provider: str  # "okta" or "azuread"
    external_id: str
    login: str
    active: bool
    last_login: str | None
    mfa_enabled: bool = False


def get_okta_users() -> list[IdpUser]:
    """Return mock Okta users. Replace with real Okta API call."""
    return [
        IdpUser(
            provider="okta", external_id=u["id"], login=u["login"],
            active=u["status"] == "ACTIVE", last_login=u.get("lastLogin"),
            mfa_enabled=u.get("mfaEnabled", False),
        )
        for u in _MOCK_OKTA_USERS
    ]


def get_azuread_users() -> list[IdpUser]:
    """Return mock Azure AD users. Replace with real Microsoft Graph API call."""
    return [
        IdpUser(
            provider="azuread", external_id=u["id"],
            login=u["userPrincipalName"],
            active=u["accountEnabled"],
            last_login=u.get("lastSignInDateTime"),
            mfa_enabled=True,  # Azure AD defaults
        )
        for u in _MOCK_AZURE_USERS
    ]


def get_okta_groups() -> list[dict]:
    """Return mock Okta groups with memberships."""
    return _MOCK_OKTA_GROUPS


def get_aws_iam_users() -> list[IdpUser]:
    """Return mock AWS IAM users. Replace with real AWS IAM API call."""
    return [
        IdpUser(
            provider="aws_iam", external_id=u["id"], login=u["UserName"],
            active=u["Active"], last_login=u.get("PasswordLastUsed"),
            mfa_enabled=True,  # assume IAM users have MFA
        )
        for u in _MOCK_AWS_USERS
    ]


def get_aws_access_keys() -> list[dict]:
    """Return mock AWS IAM access keys for credential sprawl detection."""
    return _MOCK_AWS_ACCESS_KEYS


def get_all_idp_users() -> list[IdpUser]:
    """Return all mock identity provider users across Okta, Azure AD, and AWS IAM."""
    return get_okta_users() + get_azuread_users() + get_aws_iam_users()


def cross_reference_idp(
    identity_users: list[dict],
) -> dict[str, list[dict]]:
    """Cross-reference internal identities with IDP records.

    Args:
        identity_users: List of identity dicts with email and source_system fields.

    Returns:
        Dict with "matched", "orphaned_idp", "orphaned_internal" lists.
    """
    okta_emails = {u["login"]: u for u in _MOCK_OKTA_USERS}
    azure_emails = {u["userPrincipalName"]: u for u in _MOCK_AZURE_USERS}
    aws_emails = {
        u["UserName"] + "@company.com": u for u in _MOCK_AWS_USERS
    }  # AWS users matched by constructed email

    matched = []
    orphaned_idp = []
    for user in identity_users:
        email = user.get("email", "")
        if email in okta_emails:
            matched.append({"internal": user, "idp": okta_emails[email], "provider": "okta"})
        elif email in azure_emails:
            matched.append({"internal": user, "idp": azure_emails[email], "provider": "azuread"})
        elif email in aws_emails:
            matched.append({"internal": user, "idp": aws_emails[email], "provider": "aws_iam"})
        else:
            orphaned_idp.append(user)

    return {
        "matched": matched,
        "orphaned_idp": orphaned_idp,
        "summary": f"{len(matched)} identities matched to IDP records, "
                   f"{len(orphaned_idp)} internal identities not found in IDP.",
    }
