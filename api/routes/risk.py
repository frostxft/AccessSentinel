"""Risk assessment, clustering, and privilege graph routes."""

from __future__ import annotations

import os

import pandas as pd
from fastapi import APIRouter, Query

from api.schemas import (
    AccessDecisionResponse,
    BlastRadiusReport,
    ClusterSummary,
    FeedbackRequest,
    FeedbackResponse,
)
from core.ingestion import _normalize_column_names
from core.models.kmeans_clustering import RoleMiner
from core.models.random_forest import compute_peer_access_rate
from api.routes.identity import _scan_cache

router = APIRouter(prefix="/api/v1", tags=["risk"])

_event_cache: pd.DataFrame | None = None
_cluster_cache: list[dict] = []


def _get_event_cache() -> pd.DataFrame:
    global _event_cache
    if _event_cache is not None:
        return _event_cache
    events_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "sample_data", "identity_events.csv"
    )
    if os.path.exists(events_path):
        _event_cache = pd.read_csv(events_path)
        _event_cache = _normalize_column_names(_event_cache)
    else:
        _event_cache = pd.DataFrame()
    return _event_cache


def _get_cluster_profiles() -> list[dict]:
    global _cluster_cache
    if _cluster_cache:
        return _cluster_cache
    events_df = _get_event_cache()
    if events_df.empty:
        return []
    try:
        miner = RoleMiner()
        miner.fit(events_df)
        _labels, profiles = miner.predict(events_df)
        _cluster_cache = [
            {
                "cluster_id": str(p.cluster_id),
                "label": p.label,
                "user_count": p.user_count,
                "avg_risk_score": p.avg_risk_score,
                "dominant_resources": p.dominant_resources,
                "dominant_actions": p.dominant_actions,
                "outlier_count": p.outlier_count,
            }
            for p in profiles
        ]
    except Exception:
        _cluster_cache = []
    return _cluster_cache


@router.get("/access/predict", response_model=AccessDecisionResponse)
async def predict_access(
    department: str = Query(...),
    job_title: str = Query(...),
    resource: str = Query(...),
    action: str = Query(...),
):
    events_df = _get_event_cache()
    peer_rate = compute_peer_access_rate(resource, action, department, events_df)

    if peer_rate > 0.80:
        decision = "APPROVE"
        confidence = 0.95
        note = (
            f"Peer access rate for {resource}/{action} in {department} "
            f"is {peer_rate:.0%}. High peer prevalence indicates legitimate access."
        )
    elif peer_rate > 0.40:
        decision = "REVIEW"
        confidence = 0.65
        note = (
            f"Peer access rate for {resource}/{action} in {department} "
            f"is {peer_rate:.0%}. Moderate peer prevalence. Manual review recommended."
        )
    elif peer_rate > 0.0:
        decision = "DENY"
        confidence = 0.80
        note = (
            f"Peer access rate for {resource}/{action} in {department} "
            f"is {peer_rate:.0%}. Low peer prevalence indicates unusual access."
        )
    else:
        decision = "DENY"
        confidence = 0.90
        note = (
            f"No peers in {department} have accessed {resource}/{action}. "
            f"This is a highly unusual access pattern."
        )

    return AccessDecisionResponse(
        decision=decision,
        confidence=confidence,
        peer_comparison_note=note,
    )


@router.get("/clusters", response_model=list[ClusterSummary])
async def list_clusters():
    profiles = _get_cluster_profiles()
    if not profiles:
        return []
    return [
        ClusterSummary(
            cluster_id=p["cluster_id"],
            label=p["label"],
            user_count=p["user_count"],
            avg_risk_score=p["avg_risk_score"],
            dominant_resources=p["dominant_resources"],
            dominant_actions=p["dominant_actions"],
            outlier_count=p["outlier_count"],
        )
        for p in profiles
    ]


@router.get("/graph")
async def privilege_graph():
    from api.routes.identity import _identity_cache, _last_scan_id, _scan_cache

    identities: list[dict] = list(_identity_cache.values())
    if _last_scan_id and _scan_cache.get(_last_scan_id):
        identities = _scan_cache[_last_scan_id]

    if not identities:
        return {"nodes": [], "edges": []}

    nodes: list[dict] = []
    edges: list[dict] = []
    seen_user_ids: set[str] = set()
    seen_role_ids: set[str] = set()
    seen_system_ids: set[str] = set()
    seen_resource_ids: set[str] = set()
    connected_user_ids: set[str] = set()

    for identity in identities:
        uid = identity["user_id"]
        username = identity["username"]
        if uid not in seen_user_ids:
            is_admin = identity.get("is_privileged", False)
            account_type = identity.get("account_type", "")
            is_shadow_admin = (
                not is_admin
                and account_type != "admin"
                and any(
                    "admin" in p.lower() for p in identity.get("permissions", [])
                )
            )
            nodes.append(
                {
                    "id": uid,
                    "label": username,
                    "type": "user",
                    "tier": identity.get("tier", "LOW"),
                    "score": identity.get("score", 0),
                    "shadow_admin": is_shadow_admin,
                }
            )
            seen_user_ids.add(uid)

        for role in identity.get("roles", []):
            role_id = f"role:{role}"
            if role_id not in seen_role_ids:
                nodes.append(
                    {"id": role_id, "label": role, "type": "role"}
                )
                seen_role_ids.add(role_id)
            edges.append(
                {
                    "source": uid,
                    "target": role_id,
                    "type": "user_role",
                }
            )
            connected_user_ids.add(uid)

        if identity.get("source_system"):
            sys = identity["source_system"]
            sys_id = f"system:{sys}"
            if sys_id not in seen_system_ids:
                nodes.append(
                    {"id": sys_id, "label": sys, "type": "system"}
                )
                seen_system_ids.add(sys_id)
            edges.append(
                {
                    "source": uid,
                    "target": sys_id,
                    "type": "user_system",
                }
            )
            connected_user_ids.add(uid)

        for perm in identity.get("permissions", []):
            res_id = f"resource:{perm}"
            if res_id not in seen_resource_ids:
                nodes.append(
                    {"id": res_id, "label": perm, "type": "resource"}
                )
                seen_resource_ids.add(res_id)
            edges.append(
                {
                    "source": uid,
                    "target": res_id,
                    "type": "user_resource",
                }
            )
            connected_user_ids.add(uid)

    nodes = [n for n in nodes if n["type"] == "user" or n["id"] in connected_user_ids or any(
        e["target"] == n["id"] or e["source"] == n["id"]
        for e in edges
    )]

    return {"nodes": nodes, "edges": edges}


# ── Blast Radius ─────────────────────────────────────────────────────────────


@router.get("/identities/{identity_id}/blast-radius")
async def blast_radius(identity_id: str):
    """Return the blast radius simulation for an identity.

    Returns 404 if the identity is not found in the current scan cache.
    Returns 200 with an empty report (systems_at_risk=[]) for an identity
    with no reachable privilege-graph nodes.
    """
    from api.schemas import BlastRadiusReport
    from fastapi import HTTPException
    import networkx as nx
    from core.risk_scorer import compute_blast_radius as _compute_blast

    # Look up identity in scan cache
    identity_dict = None
    all_identities = []
    for sid, scan_records in _scan_cache.items():
        for rec in scan_records:
            all_identities.append(rec)
            if rec.get("user_id") == identity_id:
                identity_dict = rec

    if identity_dict is None:
        raise HTTPException(status_code=404, detail=f"Identity {identity_id} not found.")

    # Build a simple privilege graph from identities
    graph = nx.DiGraph()
    graph.add_node(f"user:{identity_id}", type="user")
    for role_val in identity_dict.get("roles", []):
        role_node = f"role:{role_val}"
        graph.add_node(role_node, type="role")
        graph.add_edge(f"user:{identity_id}", role_node)
        sys_node = f"system:{identity_dict.get('source_system', '')}"
        graph.add_node(sys_node, type="system")
        graph.add_edge(role_node, sys_node)
        for perm_val in identity_dict.get("permissions", []):
            res_node = f"resource:{perm_val}"
            graph.add_node(res_node, type="resource")
            graph.add_edge(sys_node, res_node)

    # Translate dicts to stubs for compute_blast_radius
    from types import SimpleNamespace
    identity_stub = SimpleNamespace(
        user_id=identity_id,
        resource_sensitivity=identity_dict.get("resource_sensitivity", "low"),
    )
    all_stubs = [
        SimpleNamespace(user_id=r.get("user_id", ""))
        for r in all_identities
    ]

    report = _compute_blast(identity_stub, all_stubs, graph)
    return BlastRadiusReport(**report)


# ── Feedback ──────────────────────────────────────────────────────────────────


@router.post("/feedback")
async def submit_feedback(request: "FeedbackRequest"):  # type: ignore[name-defined]
    """Record a human correction to an access decision."""
    import uuid
    from datetime import datetime, timezone
    from fastapi import Request as FastAPIRequest
    from api.schemas import FeedbackRequest, FeedbackResponse

    feedback_id = str(uuid.uuid4())
    # In-memory store (no ORM for hackathon simplicity)
    _feedback_store: list[dict] = getattr(submit_feedback, "_store", [])
    _feedback_store.append({
        "feedback_id": feedback_id,
        "identity_id": request.identity_id,
        "original_decision": request.original_decision,
        "corrected_decision": request.corrected_decision,
        "correction_reason": request.correction_reason,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "applied_to_retraining": False,
    })
    submit_feedback._store = _feedback_store  # type: ignore[attr-defined]
    return FeedbackResponse(
        feedback_id=feedback_id,
        message="Correction recorded. Will be applied at next training cycle.",
    )


@router.get("/feedback/summary")
async def feedback_summary():
    """Return counts of total and pending feedback corrections."""
    _feedback_store: list[dict] = getattr(submit_feedback, "_store", [])
    total = len(_feedback_store)
    pending = sum(1 for r in _feedback_store if not r.get("applied_to_retraining", False))
    return {"total_corrections": total, "pending_corrections": pending}


# ── Organizational Anomaly Detection ──────────────────────────────────────────


@router.get("/org-anomalies")
async def org_anomalies():
    """Return department-level risk aggregation for organizational anomaly detection.

    Aggregates identity risk scores by department to surface which departments
    have elevated risk profiles relative to the organization baseline.
    """
    from api.routes.identity import _scan_cache, _last_scan_id, _identity_cache

    scan_results = list(_identity_cache.values())
    if _last_scan_id and _scan_cache.get(_last_scan_id):
        scan_results = _scan_cache[_last_scan_id]

    if not scan_results:
        return {"departments": [], "organization_risk": 0.0, "summary": "No scan data available."}

    dept_stats = {}
    for r in scan_results:
        dept = r.get("department", "Unknown")
        if dept not in dept_stats:
            dept_stats[dept] = {"total": 0, "critical": 0, "high": 0, "score_sum": 0, "users": []}
        d = dept_stats[dept]
        d["total"] += 1
        tier = r.get("tier", "LOW")
        if tier == "CRITICAL": d["critical"] += 1
        elif tier == "HIGH": d["high"] += 1
        d["score_sum"] += r.get("score", 0)
        d["users"].append(r.get("username", ""))

    dept_list = []
    for dept, stats in dept_stats.items():
        avg_score = stats["score_sum"] / stats["total"] if stats["total"] > 0 else 0
        risk_pct = (stats["critical"] * 2 + stats["high"]) / stats["total"] * 100 if stats["total"] > 0 else 0
        dept_list.append({
            "department": dept,
            "user_count": stats["total"],
            "critical_count": stats["critical"],
            "high_count": stats["high"],
            "avg_score": round(avg_score, 1),
            "risk_percentage": round(risk_pct, 1),
        })

    dept_list.sort(key=lambda d: d["avg_score"], reverse=True)
    org_risk = sum(d["score_sum"] for d in dept_stats.values()) / sum(d["total"] for d in dept_stats.values()) if dept_stats else 0

    return {
        "departments": dept_list,
        "organization_risk": round(org_risk, 1),
        "summary": f"{len(dept_list)} departments analyzed. "
                   f"Org avg risk: {org_risk:.1f}. "
                   f"Highest risk dept: {dept_list[0]['department'] if dept_list else 'N/A'}.",
    }


# ── Identity Provider (IdP) Integration Summary ──────────────────────────────


@router.get("/idp-summary")
async def idp_summary():
    """Return multi-cloud identity provider summary with source-type labels.

    Uses real connector calls (Okta API, Microsoft Graph, AWS IAM) when
    credentials are configured; falls back to mock data otherwise.
    Source type (live/mock) is reported per provider.
    """
    from api.routes.identity import _scan_cache, _last_scan_id, _identity_cache
    from core.idp_connectors import (
        fetch_all_providers, fetch_aws_iam_users, get_aws_access_keys_data,
    )

    scan_results = list(_identity_cache.values())
    if _last_scan_id and _scan_cache.get(_last_scan_id):
        scan_results = _scan_cache[_last_scan_id]

    fetches = fetch_all_providers()

    providers = {}
    all_users = []
    for f in fetches:
        key = f.provider
        active = sum(1 for u in f.users if u.get("status") == "ACTIVE" or
                     u.get("accountEnabled") or u.get("Active"))
        inactive = len(f.users) - active
        providers[key] = {
            "total": len(f.users), "active": active, "inactive": inactive,
            "source_type": f.source_type, "fetch_status": f.fetch_status,
            "error": f.error,
        }
        all_users.extend(f.users)

    # AWS access keys
    aws_keys_result = get_aws_access_keys_data()
    active_keys = [k for k in aws_keys_result.users if k.get("Status") == "Active"]
    aws_risks = {
        "total_keys": len(aws_keys_result.users),
        "active_keys": len(active_keys),
        "inactive_keys": len(aws_keys_result.users) - len(active_keys),
        "key_sprawl_detected": len(active_keys) >= 2,
        "source_type": aws_keys_result.source_type,
    }

    cross_ref_count = 0
    if scan_results:
        internal_emails = {r.get("email", "") for r in scan_results}
        idp_logins = {u.get("login", u.get("userPrincipalName", u.get("UserName", ""))) for u in all_users}
        cross_ref_count = len(internal_emails & idp_logins)

    source_types = {p["source_type"] for p in providers.values()}
    overall_mode = "live" if "live" in source_types else "mock"

    return {
        "providers": {
            "okta": providers.get("okta", {}),
            "azuread": providers.get("azuread", {}),
            "aws_iam": {**providers.get("aws_iam", {}), "access_keys": aws_risks},
        },
        "total_idp_users": len(all_users),
        "cross_referenced": cross_ref_count,
        "overall_mode": overall_mode,
        "aws_iam_key_sprawl": aws_risks["key_sprawl_detected"],
        "summary": (
            f"{len(all_users)} IdP users across Okta ({providers.get('okta',{}).get('total',0)}), "
            f"Azure AD ({providers.get('azuread',{}).get('total',0)}), AWS IAM ({providers.get('aws_iam',{}).get('total',0)}). "
            f"Mode: {overall_mode.upper()}. "
            f"{cross_ref_count} matched to internal identities."
        ),
    }


# ── Integration Status ───────────────────────────────────────────────────────


@router.get("/integrations/status")
async def integration_status():
    """Return health and configuration status for all identity provider connectors."""
    statuses = {}
    for name, module_path in [
        ("okta", "connectors.okta_connector"),
        ("azuread", "connectors.azuread_connector"),
        ("aws_iam", "connectors.aws_connector"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            statuses[name] = mod.healthcheck()
        except Exception as exc:
            statuses[name] = {"provider": name, "configured": False, "status": "error", "error": str(exc)}

    # Determine overall mode
    statuses_list = list(statuses.values())
    if any(s.get("status") == "live" for s in statuses_list):
        overall = "live"
    elif any(s.get("status") == "configured" for s in statuses_list):
        overall = "configured"
    else:
        overall = "not_configured"

    return {
        "providers": statuses,
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "overall_mode": overall,
    }


@router.get("/integrations/test/{provider}")
async def test_provider_connection(provider: str):
    """Run deep live verification for a specific provider (STS + enrichment).

    This endpoint performs real AWS API calls and may take 10-45 seconds.
    Use for explicit Test Connection button clicks, not for page-load status.
    """
    if provider not in ("okta", "azuread", "aws_iam"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")

    module_paths = {
        "okta": "connectors.okta_connector",
        "azuread": "connectors.azuread_connector",
        "aws_iam": "connectors.aws_connector",
    }
    module_path = module_paths.get(provider)
    if not module_path:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Unknown provider: {provider}")
    try:
        import importlib
        mod = importlib.import_module(module_path)
        # Use connection_test if available (deep), fall back to healthcheck
        if hasattr(mod, "connection_test"):
            result = mod.connection_test()
        else:
            result = mod.healthcheck()
        return result
    except Exception as exc:
        return {"provider": provider, "configured": False, "status": "error", "error": str(exc)}
