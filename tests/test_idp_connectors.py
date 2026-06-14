"""Tests for IdP connectors (Okta, Azure AD, AWS IAM)."""
import os
import pytest
from core.idp_connectors import (
    fetch_okta_users, fetch_azuread_users, fetch_aws_iam_users,
    fetch_all_providers, get_aws_access_keys_data,
)


def test_okta_mock_fallback_when_no_credentials():
    if "OKTA_DOMAIN" in os.environ:
        del os.environ["OKTA_DOMAIN"]
    if "OKTA_API_TOKEN" in os.environ:
        del os.environ["OKTA_API_TOKEN"]
    result = fetch_okta_users()
    assert result.source_type == "mock"
    assert result.fetch_status == "mock_fallback"
    assert len(result.users) == 4
    assert result.users[0]["login"] == "alice.chen@company.com"


def test_azuread_mock_fallback_when_no_credentials():
    for k in ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"):
        if k in os.environ:
            del os.environ[k]
    result = fetch_azuread_users()
    assert result.source_type == "mock"
    assert len(result.users) == 3


def test_aws_iam_mock_fallback_when_no_credentials():
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if k in os.environ:
            del os.environ[k]
    result = fetch_aws_iam_users()
    assert result.source_type == "mock"
    assert len(result.users) == 3


def test_fetch_all_providers_returns_three():
    results = fetch_all_providers()
    assert len(results) == 3
    providers = {r.provider for r in results}
    assert providers == {"okta", "azuread", "aws_iam"}


def test_aws_access_keys_data():
    result = get_aws_access_keys_data()
    assert result.provider == "aws_iam"
    assert len(result.users) == 3


def test_live_connector_with_invalid_credentials_gracefully_falls_back():
    os.environ["OKTA_DOMAIN"] = "fake.okta.com"
    os.environ["OKTA_API_TOKEN"] = "invalid-token"
    result = fetch_okta_users()
    del os.environ["OKTA_DOMAIN"]
    del os.environ["OKTA_API_TOKEN"]
    # Should fall back to mock when API call fails
    assert result.source_type == "mock"
    assert len(result.users) == 4


def test_normalized_output_shape_consistent():
    for fetch_fn in [fetch_okta_users, fetch_azuread_users, fetch_aws_iam_users]:
        result = fetch_fn()
        assert result.provider in ("okta", "azuread", "aws_iam")
        assert result.source_type in ("live", "mock")
        assert isinstance(result.users, list)
        assert len(result.users) > 0
        assert result.fetch_status in ("ok", "mock_fallback", "error")
        # First user should have an id field
        assert "id" in result.users[0] or "UserName" in result.users[0] or "userPrincipalName" in result.users[0]
