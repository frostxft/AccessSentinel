"""Identity scan, listing, and detail routes."""

from __future__ import annotations

import io
import os
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from api.schemas import (
    ContextSignalSchema,
    IdentityDetail,
    IdentitySummary,
    MitreTechniqueSchema,
    PaginatedResponse,
    RemediationActionSchema,
    RuleResultSchema,
    ScanResponse,
)
from core.behavioral_baseline import (
    BaselineProfile,
    build_baselines,
    compute_behavior_zscore,
    load_baselines,
)
from core.context_resolver import resolve
from core.features import extract_features
from core.ingestion import (
    IdentityRecord,
    _compute_event_derived_features,
    _normalize_column_names,
    _parse_boolean,
    _parse_float,
    _parse_int,
    _parse_list_field,
    _parse_timestamp,
)
from core.mitre_mapper import map_rules_to_mitre
from core.models.ensemble_detector import EnsembleAnomalyDetector
from core.models.kmeans_clustering import RoleMiner
from core.models.sequential_detector import detect_sequences, SequenceRisk
from core.remediation import generate_remediation_plan
from core.rules_engine import evaluate_rules
from core.risk_scorer import compute_risk_score

router = APIRouter(prefix="/api/v1", tags=["identity"])

_identity_cache: dict[str, dict] = {}
_scan_cache: dict[str, list] = {}
_last_scan_id: str | None = None
_event_cache: pd.DataFrame | None = None

MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB


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


def _validate_magic_bytes(content: bytes) -> str:
    text = content.decode("utf-8-sig", errors="ignore").lstrip()
    if text.startswith("{") or text.startswith("["):
        return "application/json"
    if text and text[0].isalnum():
        return "text/csv"
    raise HTTPException(
        status_code=415, detail="Unsupported file format. Expected CSV or JSON."
    )


def _build_identity_records(df: pd.DataFrame) -> list[IdentityRecord]:
    df = _normalize_column_names(df)
    # Apply competition-schema adapter if needed (same as core/ingestion.py)
    if "privilege_level" in df.columns:
        from core.ingestion import _adapt_competition_users
        df = _adapt_competition_users(df)
    records: list[IdentityRecord] = []
    for _, row in df.iterrows():
        uid = str(row.get("user_id", ""))
        if not uid or uid.lower() == "nan":
            continue
        last_login = None
        try:
            last_login = _parse_timestamp(row.get("last_login"))
        except Exception:
            last_login = None
        created_at = None
        try:
            created_at = _parse_timestamp(
                row.get("created_at", row.get("last_login"))
            )
        except Exception:
            created_at = None
        owner = str(row.get("owner_id", ""))
        if owner in ("", "nan", "None"):
            owner = ""
        record = IdentityRecord(
            user_id=uid,
            username=str(row.get("username", "")),
            email=str(row.get("email", "")),
            department=str(row.get("department", "")),
            employment_status=str(row.get("employment_status", "active")),
            account_type=str(row.get("account_type", "human")),
            owner_id=owner,
            source_system=str(row.get("source_system", "")),
            last_login=last_login,
            created_at=created_at,
            roles=_parse_list_field(row.get("roles", "")),
            permissions=_parse_list_field(row.get("permissions", "")),
            mfa_enabled=_parse_boolean(row.get("mfa_enabled")),
            sso_linked=_parse_boolean(row.get("sso_linked")),
            login_count_30d=_parse_int(row.get("login_count_30d")),
            login_count_90d=_parse_int(row.get("login_count_90d")),
            systems_count=_parse_int(row.get("systems_count")),
            role_changes_90d=_parse_int(row.get("role_changes_90d")),
            is_privileged=_parse_boolean(row.get("is_privileged")),
            resource_sensitivity=str(row.get("resource_sensitivity", "low")),
            off_hours_access_pct=_parse_float(row.get("off_hours_access_pct")),
            geo_anomaly=_parse_boolean(row.get("geo_anomaly")),
            interactive_login=_parse_boolean(row.get("interactive_login")),
        )
        records.append(record)
    return records


def _enrich_with_events(records: list[IdentityRecord]) -> list[IdentityRecord]:
    events_df = _get_event_cache()
    if events_df.empty:
        return records
    event_features = _compute_event_derived_features(events_df)
    enriched: list[IdentityRecord] = []
    for r in records:
        feats = event_features.get(r.user_id, {})
        enriched.append(
            IdentityRecord(
                user_id=r.user_id,
                username=r.username,
                email=r.email,
                department=r.department,
                employment_status=r.employment_status,
                account_type=r.account_type,
                owner_id=r.owner_id,
                source_system=r.source_system,
                last_login=r.last_login,
                created_at=r.created_at,
                roles=r.roles,
                permissions=r.permissions,
                mfa_enabled=r.mfa_enabled,
                sso_linked=r.sso_linked,
                login_count_30d=r.login_count_30d,
                login_count_90d=r.login_count_90d,
                systems_count=r.systems_count,
                role_changes_90d=r.role_changes_90d,
                is_privileged=r.is_privileged,
                resource_sensitivity=r.resource_sensitivity,
                off_hours_access_pct=r.off_hours_access_pct,
                geo_anomaly=r.geo_anomaly,
                interactive_login=r.interactive_login,
                event_count_30d=feats.get("event_count_30d", 0),
                event_count_90d=feats.get("event_count_90d", 0),
                unique_resources_accessed=feats.get("unique_resources_accessed", 0),
                anomaly_event_count=feats.get("anomaly_event_count", 0),
                failed_attempt_count=feats.get("failed_attempt_count", 0),
                off_hours_event_pct=feats.get("off_hours_event_pct", 0.0),
                impossible_travel_detected=feats.get(
                    "impossible_travel_detected", False
                ),
                bulk_download_detected=feats.get("bulk_download_detected", False),
                max_resources_in_single_session=feats.get(
                    "max_resources_in_single_session", 0
                ),
                avg_time_between_events_hours=feats.get(
                    "avg_time_between_events_hours", 0.0
                ),
            )
        )
    return enriched


def run_full_pipeline(df: pd.DataFrame) -> tuple[str, list[dict]]:
    records = _build_identity_records(df)
    records = _enrich_with_events(records)
    if not records:
        return str(uuid.uuid4()), []

    events_df = _get_event_cache()

    baselines: dict[str, BaselineProfile] = load_baselines()
    if not baselines and not events_df.empty:
        try:
            baselines = build_baselines(events_df)
        except Exception:
            baselines = {}

    baseline_for_features: dict[str, dict[str, float]] = {
        r.user_id: {"mean": 0.0, "std": 1.0} for r in records
    }
    peer_stats: dict[str, dict[str, float]] = {}
    for dept in {r.department for r in records if r.department}:
        dept_counts = [r.event_count_30d for r in records if r.department == dept]
        if dept_counts:
            mean_val = sum(dept_counts) / len(dept_counts)
            variance = (
                sum((x - mean_val) ** 2 for x in dept_counts) / len(dept_counts)
            )
            peer_stats[dept] = {
                "mean": mean_val,
                "std": variance ** 0.5 if variance > 0 else 1.0,
            }

    feature_matrix = extract_features(records, baseline_for_features, peer_stats)

    ensemble = EnsembleAnomalyDetector(name="scan_ensemble")
    try:
        ensemble.fit(feature_matrix)
        _is_anomaly, _anomaly_scores = ensemble.predict(feature_matrix)
    except Exception:
        _is_anomaly = [False] * len(records)
        _anomaly_scores = [0.0] * len(records)

    cluster_labels: dict[str, str] = {}
    if not events_df.empty and len(events_df) > 1:
        try:
            miner = RoleMiner()
            miner.fit(events_df)
            labels, _profiles = miner.predict(events_df)
            user_ids = events_df["user_id"].unique()
            for uid, lbl in zip(user_ids, labels):
                cluster_labels[str(uid)] = str(lbl)
        except Exception:
            pass

    sequence_risks: dict[str, Any] = {}
    if not events_df.empty and "anomaly_type" in events_df.columns:
        try:
            seq_results: list[SequenceRisk] = detect_sequences(events_df)
            for sr in seq_results:
                sequence_risks[sr.user_id] = {
                    "pattern_detected": sr.pattern_detected,
                    "pattern_type": sr.pattern_type,
                    "confidence": sr.confidence,
                }
        except Exception:
            pass

    results: list[dict] = []
    for i, record in enumerate(records):
        key_role = record.roles[0] if record.roles else ""
        baseline_key = f"{record.department}|{key_role}"
        baseline = baselines.get(baseline_key)

        rules = evaluate_rules(record, baseline)

        context_signals = resolve(record, baseline)

        behavior_z = 0.0
        if baseline:
            behavior_z = compute_behavior_zscore(record, baseline)
        peer_dev = (
            float(feature_matrix[i, 21])
            if feature_matrix.shape[1] > 21
            else 0.0
        )

        risk = compute_risk_score(
            record,
            rules,
            context_signals,
            peer_deviation_score=peer_dev,
            behavior_zscore=behavior_z,
        )

        mitre_techniques = map_rules_to_mitre(risk.contributing_factors)

        remediation_actions = generate_remediation_plan(
            rules, record, risk.score, context_signals
        )

        anomaly_types: list[str] = []
        for rule in rules:
            if rule.triggered:
                anomaly_types.append(rule.rule_id)

        suppressed_ids: set[str] = set()
        for sig in context_signals:
            for rid in getattr(sig, "rules_suppressed", []):
                suppressed_ids.add(rid)

        triggered_rules = [r for r in rules if r.triggered]

        triggered_rule_schemas = [
            RuleResultSchema(
                rule_id=r.rule_id,
                severity=r.severity,
                triggered=r.triggered,
                evidence_text=r.evidence_text,
                suppressed_by=r.suppressed_by,
            )
            for r in triggered_rules
        ]

        suppressed_rule_schemas = [
            rs for rs in triggered_rule_schemas if rs.rule_id in suppressed_ids
        ]
        active_rule_schemas = [
            rs
            for rs in triggered_rule_schemas
            if rs.rule_id not in suppressed_ids
        ]

        context_signal_schemas = [
            ContextSignalSchema(
                signal_type=getattr(sig, "signal_type", ""),
                explanation=getattr(sig, "explanation", ""),
                confidence=getattr(sig, "confidence", 0.0),
                score_adjustment=getattr(sig, "score_adjustment", 0),
                rules_suppressed=getattr(sig, "rules_suppressed", []),
                requires_followup=getattr(sig, "requires_followup", False),
            )
            for sig in context_signals
        ]

        mitre_schemas = [
            MitreTechniqueSchema(
                technique_id=mt.technique_id,
                name=mt.name,
                tactic=mt.tactic,
                url=mt.url,
                triggered_by_rule=mt.triggered_by_rule,
            )
            for mt in mitre_techniques
        ]

        remediation_schemas = [
            RemediationActionSchema(
                priority=ra.priority,
                action_type=ra.action_type,
                target=ra.target,
                human_readable_description=ra.human_readable_description,
                machine_actionable_command=ra.machine_actionable_command,
                estimated_risk_reduction=ra.estimated_risk_reduction,
                expected_resolution_hours=ra.expected_resolution_hours,
                requires_approval=ra.requires_approval,
            )
            for ra in remediation_actions
        ]

        primary_rule = (
            risk.contributing_factors[0] if risk.contributing_factors else ""
        )
        mitre_tech_str = (
            mitre_techniques[0].technique_id if mitre_techniques else ""
        )

        result = {
            "user_id": record.user_id,
            "username": record.username,
            "email": record.email,
            "department": record.department,
            "source_system": record.source_system,
            "score": risk.score,
            "tier": risk.tier,
            "anomaly_types": anomaly_types,
            "mitre_technique": mitre_tech_str,
            "primary_rule": primary_rule,
            "suppressed_rule_count": len(suppressed_rule_schemas),
            "blast_radius_applied": risk.blast_radius_applied,
            "account_type": record.account_type,
            "employment_status": record.employment_status,
            "mfa_enabled": record.mfa_enabled,
            "sso_linked": record.sso_linked,
            "last_login": record.last_login.isoformat() if record.last_login else None,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "roles": record.roles,
            "permissions": record.permissions,
            "systems_count": record.systems_count,
            "off_hours_access_pct": record.off_hours_access_pct,
            "is_privileged": record.is_privileged,
            "risk_narrative": risk.risk_narrative,
            "contributing_factors": risk.contributing_factors,
            "suppressed_factors": risk.suppressed_factors,
            "context_signals": [cs.model_dump() for cs in context_signal_schemas],
            "mitre_techniques": [mt.model_dump() for mt in mitre_schemas],
            "remediation_actions": [ra.model_dump() for ra in remediation_schemas],
            "behavior_zscore": risk.behavior_zscore,
            "confidence": risk.confidence,
            "cluster_assignment": cluster_labels.get(record.user_id, ""),
            "peer_deviation_score": peer_dev,
            "sequence_risk": sequence_risks.get(record.user_id),
            "triggered_rules": [rs.model_dump() for rs in active_rule_schemas],
            "suppressed_rules": [rs.model_dump() for rs in suppressed_rule_schemas],
        }
        results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    return str(uuid.uuid4()), results


@router.post("/scan", response_model=ScanResponse)
async def scan(
    users_file: UploadFile | None = File(None),
    events_file: UploadFile | None = File(None),
    users_labels_file: UploadFile | None = File(None),
    events_labels_file: UploadFile | None = File(None),
):
    if users_file is None:
        raise HTTPException(
            status_code=400,
            detail="A users CSV is required. Upload the identity_users.csv file.",
        )

    async def _read_upload(upload: UploadFile) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await upload.read(8192)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail="File size exceeds maximum allowed size of 10 MB.",
                )
            chunks.append(chunk)
        return b"".join(chunks)

    users_content = await _read_upload(users_file)
    _validate_magic_bytes(users_content)
    if users_file.content_type and users_file.content_type not in (
        "text/csv", "application/json",
    ):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported Content-Type: {users_file.content_type}.",
        )

    try:
        users_df = pd.read_csv(io.BytesIO(users_content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid users CSV file.")

    # Save uploaded users data for evaluation
    try:
        upload_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "uploaded")
        os.makedirs(upload_dir, exist_ok=True)
        with open(os.path.join(upload_dir, "users.csv"), "wb") as f:
            f.write(users_content)
    except Exception:
        pass

    if events_file is not None:
        events_content = await _read_upload(events_file)
        _validate_magic_bytes(events_content)
        if events_file.content_type and events_file.content_type not in (
            "text/csv", "application/json",
        ):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Content-Type: {events_file.content_type}.",
            )
        try:
            global _event_cache
            _event_cache = pd.read_csv(io.BytesIO(events_content))
            _event_cache = _normalize_column_names(_event_cache)
            # Apply competition-schema adapter for events if needed
            if "time_classification" in _event_cache.columns and "department" not in _event_cache.columns:
                from core.ingestion import _adapt_competition_events
                _event_cache = _adapt_competition_events(_event_cache)
                # Join department + job_title from users
                if users_df is not None:
                    ud = users_df
                    ud = _normalize_column_names(ud)
                    if "privilege_level" in ud.columns:
                        from core.ingestion import _adapt_competition_users
                        ud = _adapt_competition_users(ud)
                    user_dept = dict(zip(ud["user_id"].astype(str), ud["department"].astype(str)))
                    user_jt = dict(zip(ud["user_id"].astype(str), ud["job_title"].astype(str)))
                    _event_cache["department"] = _event_cache["user_id"].astype(str).map(user_dept).fillna("")
                    _event_cache["job_title"] = _event_cache["user_id"].astype(str).map(user_jt).fillna("")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid events CSV file.")
        # Save uploaded events data
        try:
            with open(os.path.join(upload_dir, "events.csv"), "wb") as f:
                f.write(events_content)
        except Exception:
            pass

    # Save uploaded label files if provided
    if users_labels_file is not None:
        try:
            labels_content = await _read_upload(users_labels_file)
            with open(os.path.join(upload_dir, "users_labels.csv"), "wb") as f:
                f.write(labels_content)
        except Exception:
            pass
    if events_labels_file is not None:
        try:
            labels_content = await _read_upload(events_labels_file)
            with open(os.path.join(upload_dir, "events_labels.csv"), "wb") as f:
                f.write(labels_content)
        except Exception:
            pass

    scan_id, results = run_full_pipeline(users_df)

    global _identity_cache, _scan_cache, _last_scan_id
    _scan_cache[scan_id] = results
    _last_scan_id = scan_id
    for r in results:
        _identity_cache[r["user_id"]] = r

    leaderboard = results[:50]
    leaderboard_summaries = [
        IdentitySummary(
            user_id=r["user_id"],
            username=r["username"],
            email=r["email"],
            department=r["department"],
            source_system=r["source_system"],
            score=r["score"],
            tier=r["tier"],
            anomaly_types=r["anomaly_types"],
            mitre_technique=r["mitre_technique"],
            primary_rule=r["primary_rule"],
            suppressed_rule_count=r["suppressed_rule_count"],
            blast_radius_applied=r["blast_radius_applied"],
            cluster_assignment=r.get("cluster_assignment", ""),
        )
        for r in leaderboard
    ]

    critical = sum(1 for r in results if r["tier"] == "CRITICAL")
    high = sum(1 for r in results if r["tier"] == "HIGH")
    medium = sum(1 for r in results if r["tier"] == "MEDIUM")
    low = sum(1 for r in results if r["tier"] == "LOW")

    return ScanResponse(
        message=f"Scan complete. {len(results)} identities analyzed.",
        leaderboard=leaderboard_summaries,
        total=len(results),
        critical_count=critical,
        high_count=high,
        medium_count=medium,
        low_count=low,
        f1_score=None,
    )


@router.get("/identities", response_model=PaginatedResponse)
async def list_identities(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    tier: str | None = Query(None),
    source_system: str | None = Query(None),
    department: str | None = Query(None),
    anomaly_type: str | None = Query(None),
):
    all_identities: list[dict] = list(_identity_cache.values())
    if _last_scan_id and _scan_cache.get(_last_scan_id):
        all_identities = _scan_cache[_last_scan_id]

    if tier:
        all_identities = [
            r for r in all_identities if r["tier"] == tier.upper()
        ]
    if source_system:
        all_identities = [
            r
            for r in all_identities
            if r["source_system"].lower() == source_system.lower()
        ]
    if department:
        all_identities = [
            r
            for r in all_identities
            if r["department"].lower() == department.lower()
        ]
    if anomaly_type:
        all_identities = [
            r
            for r in all_identities
            if anomaly_type.upper() in [a.upper() for a in r["anomaly_types"]]
        ]

    all_identities.sort(key=lambda x: x["score"], reverse=True)
    total = len(all_identities)
    pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = all_identities[start:end]

    summaries = [
        IdentitySummary(
            user_id=r["user_id"],
            username=r["username"],
            email=r["email"],
            department=r["department"],
            source_system=r["source_system"],
            score=r["score"],
            tier=r["tier"],
            anomaly_types=r["anomaly_types"],
            mitre_technique=r["mitre_technique"],
            primary_rule=r["primary_rule"],
            suppressed_rule_count=r["suppressed_rule_count"],
            blast_radius_applied=r["blast_radius_applied"],
            cluster_assignment=r.get("cluster_assignment", ""),
        )
        for r in page_items
    ]

    return PaginatedResponse(
        items=[s.model_dump() for s in summaries],
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@router.get("/identities/{identity_id}", response_model=IdentityDetail)
async def get_identity(identity_id: str):
    record = _identity_cache.get(identity_id)
    if not record:
        if _scan_cache.get(_last_scan_id):
            for r in _scan_cache[_last_scan_id]:
                if r["user_id"] == identity_id:
                    record = r
                    break
    if not record:
        raise HTTPException(
            status_code=404, detail=f"Identity {identity_id} not found."
        )

    return IdentityDetail(
        user_id=record["user_id"],
        username=record["username"],
        email=record["email"],
        department=record["department"],
        source_system=record["source_system"],
        score=record["score"],
        tier=record["tier"],
        anomaly_types=record["anomaly_types"],
        mitre_technique=record["mitre_technique"],
        primary_rule=record["primary_rule"],
        suppressed_rule_count=record["suppressed_rule_count"],
        blast_radius_applied=record["blast_radius_applied"],
        account_type=record["account_type"],
        employment_status=record["employment_status"],
        mfa_enabled=record["mfa_enabled"],
        sso_linked=record["sso_linked"],
        last_login=record["last_login"],
        created_at=record["created_at"],
        roles=record["roles"],
        permissions=record["permissions"],
        systems_count=record["systems_count"],
        off_hours_access_pct=record["off_hours_access_pct"],
        is_privileged=record["is_privileged"],
        risk_narrative=record["risk_narrative"],
        contributing_factors=record["contributing_factors"],
        suppressed_factors=record["suppressed_factors"],
        context_signals=record["context_signals"],
        mitre_techniques=record["mitre_techniques"],
        remediation_actions=record["remediation_actions"],
        behavior_zscore=record["behavior_zscore"],
        confidence=record["confidence"],
        cluster_assignment=record["cluster_assignment"],
        peer_deviation_score=record["peer_deviation_score"],
        sequence_risk=record["sequence_risk"],
    )
