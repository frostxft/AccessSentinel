"""Tests for IdP connector architecture."""
import os
import pytest
from connectors import NormalizedIdentity, ConnectorResult, make_mock_result
from connectors.okta_connector import is_configured as okta_configured, fetch_identities as okta_fetch, _OKTA_MOCK_USERS
from connectors.azuread_connector import is_configured as azure_configured, fetch_identities as azure_fetch, _AZUREAD_MOCK_USERS
from connectors.aws_connector import is_configured as aws_configured, fetch_identities as aws_fetch, _AWS_MOCK_USERS


def _clear_env():
    for k in ("OKTA_ORG_URL", "OKTA_API_TOKEN", "AZURE_TENANT_ID", "AZURE_CLIENT_ID",
              "AZURE_CLIENT_SECRET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if k in os.environ:
            del os.environ[k]


class TestOktaConnector:
    def test_not_configured_falls_back_to_mock(self):
        _clear_env()
        assert not okta_configured()
        result = okta_fetch()
        assert result.source_type == "mock"
        assert result.fetch_status == "mock_fallback"
        assert len(result.identities) == 4

    def test_normalized_schema_consistent(self):
        _clear_env()
        result = okta_fetch()
        for ident in result.identities:
            assert ident.provider == "okta"
            assert ident.external_id
            assert ident.username


class TestAzureAdConnector:
    def test_not_configured_falls_back_to_mock(self):
        _clear_env()
        assert not azure_configured()
        result = azure_fetch()
        assert result.source_type == "mock"
        assert len(result.identities) == 3

    def test_normalized_schema_consistent(self):
        _clear_env()
        result = azure_fetch()
        for ident in result.identities:
            assert ident.provider == "azuread"
            assert ident.external_id


class TestAwsConnector:
    def test_not_configured_falls_back_to_mock(self):
        _clear_env()
        result = aws_fetch()
        # AWS Boto3 may auto-detect credentials from ~/.aws or IAM role
        # So we only assert on the result shape
        assert result.provider == "aws_iam"
        assert len(result.identities) >= 3

    def test_normalized_schema_consistent(self):
        _clear_env()
        result = aws_fetch()
        for ident in result.identities:
            assert ident.provider == "aws_iam"


class TestIntegrationStatus:
    def test_status_endpoint_returns_three_providers(self):
        import sys; sys.path.insert(0, '.')
        import asyncio
        from api.routes.risk import integration_status
        result = asyncio.run(integration_status())
        assert "providers" in result
        assert "okta" in result["providers"]
        assert "azuread" in result["providers"]
        assert "aws_iam" in result["providers"]
        assert result["overall_mode"] in ("live", "configured", "not_configured")

    def test_aws_not_configured_without_env_vars(self):
        _clear_env()
        from connectors.aws_connector import is_configured, healthcheck
        assert not is_configured()
        hc = healthcheck()
        assert hc["status"] == "not_configured"
        assert hc["configured"] == False

    def test_aws_configured_flag_requires_both_keys(self):
        _clear_env()
        os.environ["AWS_ACCESS_KEY_ID"] = "test"
        from connectors.aws_connector import is_configured
        # Missing secret => not configured
        assert not is_configured()
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
        assert is_configured()
        _clear_env()


class TestConnectionTest:
    def test_not_configured_returns_not_configured(self):
        _clear_env()
        from connectors.aws_connector import connection_test
        result = connection_test()
        assert result["status"] == "not_configured"
        assert result["configured"] == False
        assert result["last_tested_at"] is not None

    def test_configured_without_boto3_returns_error(self):
        _clear_env()
        os.environ["AWS_ACCESS_KEY_ID"] = "test"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
        from connectors.aws_connector import connection_test
        result = connection_test()
        # When env vars are set: status=error (either boto3 missing or invalid credentials)
        assert result["configured"] == True
        assert result["status"] == "error"
        assert result["error"] is not None
        _clear_env()

    def test_connection_test_has_required_fields(self):
        _clear_env()
        from connectors.aws_connector import connection_test
        result = connection_test()
        for field in ("provider", "configured", "status", "last_tested_at",
                      "account", "account_alias", "arn", "user_id", "iam_user_count",
                      "sample_users", "credential_report_summary", "heuristics",
                      "error", "warning", "http_status"):
            assert field in result, f"Missing field: {field}"


class TestAwsHealthcheck:
    def test_healthcheck_delegates_to_connection_test(self):
        _clear_env()
        from connectors.aws_connector import healthcheck
        h = healthcheck()
        assert h["status"] == "not_configured"
        assert h["configured"] == False
        assert h["account"] is None
        assert h["arn"] is None
        assert h["user_id"] is None

    def test_healthcheck_has_full_payload_fields(self):
        _clear_env()
        from connectors.aws_connector import healthcheck
        h = healthcheck()
        for field in ("provider", "configured", "status", "account", "arn",
                      "user_id", "iam_user_count", "sample_users",
                      "credential_report_summary", "heuristics",
                      "http_status", "error", "warning"):
            assert field in h, f"Missing field: {field}"


class TestConnectorResult:
    def test_make_mock_result(self):
        identities = [NormalizedIdentity(provider="test", source_type="mock", external_id="t1")]
        r = make_mock_result("test", identities)
        assert r.source_type == "mock"
        assert r.summary() == "test: 1 users (MOCK)"
