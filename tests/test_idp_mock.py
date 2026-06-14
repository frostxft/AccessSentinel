"""Tests for IdP mock integration (Okta, Azure AD, AWS IAM)."""
import pytest
from core.idp_mock import (
    get_okta_users, get_azuread_users, get_aws_iam_users,
    get_all_idp_users, cross_reference_idp, PROVIDER_CLI_COMMANDS,
)


def test_okta_users_returned():
    users = get_okta_users()
    assert len(users) >= 1
    assert all(u.provider == "okta" for u in users)


def test_azuread_users_returned():
    users = get_azuread_users()
    assert len(users) >= 1
    assert all(u.provider == "azuread" for u in users)


def test_aws_iam_users_returned():
    users = get_aws_iam_users()
    assert len(users) >= 1
    assert all(u.provider == "aws_iam" for u in users)


def test_all_idp_users_includes_all_three():
    users = get_all_idp_users()
    providers = {u.provider for u in users}
    assert "okta" in providers
    assert "azuread" in providers
    assert "aws_iam" in providers


def test_cross_reference_matches_correctly():
    users = [
        {"email": "alice.chen@company.com"},
        {"email": "victor.ng@company.com"},
        {"email": "xavier.diaz@company.com"},
        {"email": "unknown@company.com"},
    ]
    result = cross_reference_idp(users)
    assert len(result["matched"]) == 3
    assert len(result["orphaned_idp"]) == 1
    providers = {m["provider"] for m in result["matched"]}
    assert providers == {"okta", "azuread", "aws_iam"}


def test_provider_cli_commands_present():
    assert "okta" in PROVIDER_CLI_COMMANDS
    assert "azuread" in PROVIDER_CLI_COMMANDS
    assert "aws_iam" in PROVIDER_CLI_COMMANDS
    assert "disable_user" in PROVIDER_CLI_COMMANDS["aws_iam"]
    assert "deactivate_keys" in PROVIDER_CLI_COMMANDS["aws_iam"]


def test_aws_access_keys_returned():
    from core.idp_mock import get_aws_access_keys
    keys = get_aws_access_keys()
    assert len(keys) >= 1
    assert all("AccessKeyId" in k for k in keys)
