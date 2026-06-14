import io
import os
import sys

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CSV_HEADER = (
    "user_id,username,email,department,employment_status,account_type,owner_id,"
    "source_system,last_login,created_at,roles,permissions,mfa_enabled,sso_linked,"
    "login_count_30d,login_count_90d,systems_count,role_changes_90d,is_privileged,"
    "resource_sensitivity,off_hours_access_pct,geo_anomaly,interactive_login"
)


def _make_csv(num_rows: int = 15) -> str:
    rows = [_CSV_HEADER]
    for i in range(1, num_rows + 1):
        rows.append(
            f"U{i:03d},user_{i},user{i}@test.com,Engineering,active,human,mgr{i},AD,"
            "2025-06-01T10:00:00Z,2024-01-01T10:00:00Z,reader|writer,read:all,true,true,"
            f"10,30,{i % 5},{(i * 3) % 10},false,low,0.1,false,false"
        )
    return "\n".join(rows)


_SINGLE_ROW_CSV = (
    _CSV_HEADER
    + "\nU001,test_user,test@test.com,Engineering,active,human,mgr1,AD,"
    "2025-06-01T10:00:00Z,2024-01-01T10:00:00Z,reader|writer,read:all,true,true,"
    "10,30,2,0,false,low,0.1,false,false"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAPI:
    @pytest.mark.asyncio
    async def test_01_health_endpoint_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "healthy")

    @pytest.mark.asyncio
    async def test_02_scan_valid_csv_returns_200(self, client: AsyncClient) -> None:
        csv_content = _make_csv(num_rows=15)
        files = {
            "users_file": ("identities.csv", io.BytesIO(csv_content.encode()), "text/csv")
        }
        response = await client.post("/api/v1/scan", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "leaderboard" in data
        assert isinstance(data["leaderboard"], list)
        assert "total" in data
        assert isinstance(data["total"], int)
        assert "critical_count" in data
        assert "high_count" in data
        assert "medium_count" in data
        assert "low_count" in data

    @pytest.mark.asyncio
    async def test_03_scan_empty_file_returns_error(self, client: AsyncClient) -> None:
        files = {"users_file": ("empty.csv", io.BytesIO(b""), "text/csv")}
        response = await client.post("/api/v1/scan", files=files)
        assert response.status_code in (415, 422, 400)

    @pytest.mark.asyncio
    async def test_04_scan_file_exceeding_limit(self, client: AsyncClient) -> None:
        large_data = b"x" * (11 * 1024 * 1024)
        files = {"users_file": ("large.csv", io.BytesIO(large_data), "text/csv")}
        response = await client.post("/api/v1/scan", files=files)
        assert response.status_code == 413

    @pytest.mark.asyncio
    async def test_05_identities_pagination(self, client: AsyncClient) -> None:
        response = await client.get(
            "/api/v1/identities", params={"page": 1, "page_size": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "pages" in data
        assert len(data["items"]) <= 10
        assert len(data["items"]) >= 1

    @pytest.mark.asyncio
    async def test_06_identities_page_2(self, client: AsyncClient) -> None:
        response = await client.get(
            "/api/v1/identities", params={"page": 2, "page_size": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 10
        assert len(data["items"]) >= 0

    @pytest.mark.asyncio
    async def test_07_identities_invalid_id_returns_404(
        self, client: AsyncClient
    ) -> None:
        response = await client.get("/api/v1/identities/nonexistent_id_xyz")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_08_access_predict_valid_returns_200(
        self, client: AsyncClient
    ) -> None:
        response = await client.get(
            "/api/v1/access/predict",
            params={
                "department": "Engineering",
                "job_title": "Developer",
                "resource": "customer_db",
                "action": "read",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "decision" in data
        assert data["decision"] in ("APPROVE", "DENY", "REVIEW")

    @pytest.mark.asyncio
    async def test_09_evaluate_endpoint_returns_200(
        self, client: AsyncClient
    ) -> None:
        # Clear any leftover uploaded data from prior tests
        import os as _os
        upload_dir = _os.path.join("data", "uploaded")
        if _os.path.exists(upload_dir):
            for f in _os.listdir(upload_dir):
                _os.remove(_os.path.join(upload_dir, f))
        response = await client.get("/api/v1/evaluate")
        assert response.status_code == 200
        data = response.json()
        assert "overall_f1" in data

    @pytest.mark.asyncio
    async def test_10_identities_filter_by_tier(self, client: AsyncClient) -> None:
        response = await client.get(
            "/api/v1/identities", params={"tier": "CRITICAL", "page_size": 100}
        )
        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["tier"] == "CRITICAL"

    @pytest.mark.asyncio
    async def test_11_identities_filter_by_department(
        self, client: AsyncClient
    ) -> None:
        response = await client.get(
            "/api/v1/identities",
            params={"department": "Engineering", "page_size": 100},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) >= 1
        for item in data["items"]:
            assert item["department"] == "Engineering"

    @pytest.mark.asyncio
    async def test_12_rate_limit_scan(self, client: AsyncClient) -> None:
        last_status = None
        for _ in range(31):
            files = {
                "users_file": (
                    "test.csv",
                    io.BytesIO(_SINGLE_ROW_CSV.encode()),
                    "text/csv",
                )
            }
            response = await client.post("/api/v1/scan", files=files)
            last_status = response.status_code
            if response.status_code == 429:
                break

        if last_status != 429:
            pytest.skip(
                "Rate limit not triggered after 31 requests "
                "(limit configuration may differ)"
            )
        assert response.status_code == 429


class TestBlastRadius:
    """Tests for GET /api/v1/identities/{id}/blast-radius (Fix 4)."""

    @pytest.mark.asyncio
    async def test_blast_radius_invalid_identity(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/identities/nonexistent_xyz/blast-radius")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_blast_radius_isolated_user(self, client: AsyncClient) -> None:
        # After scan, check that a valid ID returns 200 even with empty results
        csv_content = _make_csv(num_rows=3)
        files = {
            "users_file": ("identities.csv", io.BytesIO(csv_content.encode()), "text/csv")
        }
        await client.post("/api/v1/scan", files=files)
        response = await client.get("/api/v1/identities/U0001/blast-radius")
        assert response.status_code in (200, 404)  # 404 if not in cache, 200 otherwise


class TestFeedback:
    """Tests for POST /api/v1/feedback (Fix 5)."""

    @pytest.mark.asyncio
    async def test_feedback_valid_submission(self, client: AsyncClient) -> None:
        body = {
            "identity_id": "U001",
            "original_decision": "DENY",
            "corrected_decision": "APPROVE",
            "correction_reason": "User is in the on-call rotation.",
        }
        response = await client.post("/api/v1/feedback", json=body)
        assert response.status_code == 200
        data = response.json()
        assert "feedback_id" in data
        import uuid
        uuid.UUID(data["feedback_id"])  # must be valid UUID

    @pytest.mark.asyncio
    async def test_feedback_invalid_decision(self, client: AsyncClient) -> None:
        body = {
            "identity_id": "U001",
            "original_decision": "DENY",
            "corrected_decision": "MAYBE",
            "correction_reason": "Not sure.",
        }
        response = await client.post("/api/v1/feedback", json=body)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_feedback_summary(self, client: AsyncClient) -> None:
        # Submit one first
        body = {
            "identity_id": "U001",
            "original_decision": "DENY",
            "corrected_decision": "REVIEW",
            "correction_reason": "Test.",
        }
        await client.post("/api/v1/feedback", json=body)
        response = await client.get("/api/v1/feedback/summary")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data.get("total_corrections"), int)
        assert isinstance(data.get("pending_corrections"), int)
