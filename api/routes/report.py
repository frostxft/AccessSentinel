"""Report generation, evaluation, and health routes."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.schemas import (
    EvaluationReportResponse,
    HealthResponse,
    PaginatedResponse,
)

router = APIRouter(prefix="/api/v1", tags=["report"])


@router.get("/evaluate", response_model=EvaluationReportResponse)
async def evaluate():
    """Run the unified holdout evaluation pipeline.

    Uses uploaded data from ``data/uploaded/`` if a manual scan was performed.
    If uploaded data exists but evaluation fails, returns an error — never
    silently substitutes sample_data metrics for uploaded-data requests.
    """
    from core.evaluator import run_holdout_evaluation

    upload_users = os.path.join("data", "uploaded", "users.csv")
    upload_events = os.path.join("data", "uploaded", "events.csv")
    upload_labels = os.path.join("data", "uploaded", "users_labels.csv")
    upload_event_labels = os.path.join("data", "uploaded", "events_labels.csv")
    source = "sample_data"
    labels_source = "sample_data"

    if os.path.exists(upload_users):
        source = "uploaded"
        labels_source = "uploaded" if os.path.exists(upload_labels) else "none"
        events_path: str | None = upload_events if os.path.exists(upload_events) else None
        labels_path: str | None = upload_labels if os.path.exists(upload_labels) else None
        try:
            report = run_holdout_evaluation(
                users_path=upload_users, events_path=events_path,
                labels_path=labels_path,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Uploaded-data evaluation failed: {exc}",
            )
        # Override status for missing-labels case
        if labels_source == "none":
            report.evaluation_status = "missing_labels"
    else:
        report = run_holdout_evaluation()
        labels_source = "sample_data"

    return EvaluationReportResponse(
        overall_f1=report.overall_f1,
        macro_f1=report.macro_f1,
        weighted_f1=report.weighted_f1,
        precision_by_class=report.precision_by_class,
        recall_by_class=report.recall_by_class,
        f1_by_class=report.f1_by_class,
        false_positive_rate=report.false_positive_rate,
        false_positive_rate_by_dept=report.false_positive_rate_by_dept,
        confusion_matrix_data=report.confusion_matrix_data,
        source=source,
        labels_source=labels_source,
        evaluation_status=report.evaluation_status,
        label_match_count=report.label_match_count,
    )


@router.get("/report")
async def generate_report():
    from api.routes.identity import _identity_cache, _last_scan_id, _scan_cache

    scan_results: list[dict] = list(_identity_cache.values())
    if _last_scan_id and _scan_cache.get(_last_scan_id):
        scan_results = _scan_cache[_last_scan_id]

    html_content = _generate_ciso_report_html(scan_results)

    tmpdir = tempfile.gettempdir()
    report_path = os.path.join(tmpdir, "accesssentinel_ciso_report.html")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html_content)
    return FileResponse(report_path, media_type="text/html", filename="accesssentinel_ciso_report.html")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    baseline_dir = os.path.join("data", "baselines")
    models_dir = os.path.join("models")
    baseline_loaded = os.path.isdir(baseline_dir) and any(os.scandir(baseline_dir))
    models_loaded = os.path.isdir(models_dir) and any(
        f.name.endswith(".pkl") for f in os.scandir(models_dir)
    )
    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded,
        baseline_loaded=baseline_loaded,
        version="1.0.0",
    )


def _report_logo_img() -> str:
    """Return an HTML <img> tag with the embedded base64 logo, or empty string."""
    import base64

    # Navigate from api/routes/report.py → api/ → project root
    logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logo.png")
    try:
        with open(logo_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" alt="AccessSentinel" class="report-header-logo">'
    except Exception:
        return ""


def _build_llm_section(
    scan_results: list[dict],
    critical: int, high: int, medium: int, low: int,
    top10: list[dict],
    triggered_rule_ids: set[str],
    mitre_matches: list,
    remediation_actions: list[dict],
) -> str:
    """Build the optional LLM-assisted advisory narrative HTML section.

    Returns an HTML string.  If the LLM is disabled, unavailable, or fails,
    returns a small fallback note.
    """
    from core.llm_narrative import generate_narrative

    # Build grounded facts dict from already-computed data
    top_rules_list: list[str] = []
    rule_counts: dict[str, int] = {}
    for r in scan_results:
        pr = r.get("primary_rule", "")
        if pr:
            rule_counts[pr] = rule_counts.get(pr, 0) + 1
    top_rules_list = sorted(rule_counts, key=rule_counts.get, reverse=True)[:8]

    mitre_dicts = []
    for m in mitre_matches:
        mitre_dicts.append({
            "technique_id": getattr(m, "technique_id", ""),
            "name": getattr(m, "name", ""),
            "tactic": getattr(m, "tactic", ""),
            "triggered_by_rule": getattr(m, "triggered_by_rule", ""),
        })
    # Deduplicate by technique_id
    seen = set()
    mitre_deduped = []
    for mt in mitre_dicts:
        tid = mt["technique_id"]
        if tid and tid not in seen:
            seen.add(tid)
            mitre_deduped.append(mt)

    top_identity_dicts = []
    for r in top10:
        top_identity_dicts.append({
            "username": r.get("username", ""),
            "department": r.get("department", ""),
            "score": r.get("score", 0),
            "tier": r.get("tier", ""),
            "primary_rule": r.get("primary_rule", ""),
        })

    rem_dicts = []
    for a in remediation_actions[:6]:
        rem_dicts.append({
            "action_type": a.get("action_type", ""),
            "human_readable_description": a.get("human_readable_description", "")[:100],
        })

    # Try to get evaluation metrics if available
    eval_text = ""
    try:
        import json as _json, os as _os
        eval_path = _os.path.join("data", "evaluation_report.json")
        if _os.path.exists(eval_path):
            with open(eval_path, "r") as fh:
                eval_data = _json.load(fh)
            f1 = eval_data.get("overall_f1", "")
            fpr = eval_data.get("false_positive_rate", "")
            eval_text = f"Overall F1: {f1}, FPR: {fpr}" if f1 else ""
    except Exception:
        pass

    context_signal_types = [
        "SABBATICAL_POSSIBLE", "TEMP_ELEVATION", "NEW_HIRE_RAMP",
        "BATCH_JOB_PATTERN", "MONTH_END_FINANCE",
        "IMPOSSIBLE_TRAVEL_CONFIRMED", "CONTRACTOR_NORM",
    ]

    report_facts = {
        "total_identities": len(scan_results),
        "critical_count": critical,
        "high_count": high,
        "medium_count": medium,
        "low_count": low,
        "top_identities": top_identity_dicts,
        "top_rules": top_rules_list,
        "mitre_techniques": mitre_deduped,
        "remediation_actions": rem_dicts,
        "context_signals": context_signal_types,
        "evaluation_metrics": eval_text,
    }

    narrative = generate_narrative(report_facts)

    if narrative.api_call_succeeded:
        findings_html = ""
        for f in narrative.key_findings:
            findings_html += f"<li>{f}</li>"
        if not findings_html:
            findings_html = "<li>No specific findings generated.</li>"

        priorities_html = ""
        for p in narrative.remediation_priorities:
            priorities_html += f"<li>{p}</li>"
        if not priorities_html:
            priorities_html = "<li>No specific priorities generated.</li>"

        model_note = f" via {narrative.model_used}" if narrative.model_used else ""
        latency_note = f" (generated in {narrative.generation_latency_ms}ms)" if narrative.generation_latency_ms else ""

        return f"""
<div class="llm-advisory">
<h3>AI-Assisted Executive Summary (Advisory{model_note}{latency_note})</h3>
<p>{narrative.executive_summary}</p>
<br><strong>Key Findings:</strong>
<ul>{findings_html}</ul>
<strong>Remediation Priorities:</strong>
<ul>{priorities_html}</ul>
{narrative.business_impact if narrative.business_impact else ''}
{narrative.confidence_note if narrative.confidence_note else ''}
</div>"""
    else:
        reason = narrative.fallback_reason or "unknown"
        return f'<div class="llm-fallback">AI-assisted narrative unavailable: {reason}. Report uses deterministic content only.</div>'


def _generate_ciso_report_html(scan_results: list[dict]) -> str:
    critical = sum(1 for r in scan_results if r.get("tier") == "CRITICAL")
    high = sum(1 for r in scan_results if r.get("tier") == "HIGH")
    medium = sum(1 for r in scan_results if r.get("tier") == "MEDIUM")
    low = sum(1 for r in scan_results if r.get("tier") == "LOW")

    top10 = sorted(scan_results, key=lambda r: r.get("score", 0), reverse=True)[:10]
    top_rows = ""
    for r in top10:
        top_rows += (
            f'<tr><td>{r.get("username","")}</td><td>{r.get("department","")}</td>'
            f'<td>{r.get("score",0)}</td><td>{r.get("tier","")}</td>'
            f'<td>{r.get("primary_rule","")}</td></tr>'
        )

    # Risk Tier Distribution data for chart
    import base64, io as _io
    tier_data = {"CRITICAL": critical, "HIGH": high, "MEDIUM": medium, "LOW": low}
    tier_pct = {t: (c / max(len(scan_results), 1) * 100) for t, c in tier_data.items()}
    tier_rows = ""
    tier_colors = {"CRITICAL": "#DA3633", "HIGH": "#D29922", "MEDIUM": "#1F6FEB", "LOW": "#3FB950"}
    for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        c = tier_data[tier]
        pct = tier_pct[tier]
        color = tier_colors[tier]
        tier_rows += (
            f'<tr><td><span style="color:{color};font-weight:700">{tier}</span></td>'
            f'<td>{c}</td><td>{pct:.1f}%</td>'
            f'<td><div style="background:#21262D;border-radius:4px;width:200px">'
            f'<div style="background:{color};height:16px;width:{pct:.0f}%;border-radius:4px"></div>'
            f'</div></td></tr>'
        )

    # MITRE ATT&CK summary
    from core.mitre_mapper import get_all_mitre_techniques, map_rules_to_mitre
    all_techniques = get_all_mitre_techniques()
    triggered_rule_ids = set()
    for r in scan_results:
        if r.get("primary_rule"):
            triggered_rule_ids.add(r["primary_rule"])
    mitre_matches = map_rules_to_mitre(list(triggered_rule_ids))
    mitre_rows = ""
    for m in mitre_matches:
        mitre_rows += (
            f'<tr><td>{m.technique_id}</td><td>{m.name}</td>'
            f'<td>{m.tactic}</td><td>{m.triggered_by_rule}</td></tr>'
        )

    # Remediation summary
    remediation_actions = []
    for r in scan_results[:20]:
        ra = r.get("remediation_actions", [])
        for a in (ra[:2] if ra else []):
            remediation_actions.append(a)
    rem_rows = ""
    for a in remediation_actions[:10]:
        rem_rows += (
            f'<tr><td>{a.get("priority","")}</td><td>{a.get("action_type","")}</td>'
            f'<td>{a.get("human_readable_description","")[:100]}</td>'
            f'<td>{a.get("expected_resolution_hours","")}h</td></tr>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>AccessSentinel CISO Report</title>
<style>
body{{font-family:-apple-system,'Segoe UI',sans-serif;background:#0D1117;color:#E6EDF3;padding:24px;max-width:960px;margin:auto}}
h1{{color:#388BFD}}h2{{color:#E6EDF3;border-bottom:1px solid #30363D;padding-bottom:8px}}
h3{{color:#8B949E;font-size:14px;text-transform:uppercase;letter-spacing:0.5px}}
.stat-label{{font-size:12px;color:#8B949E;text-transform:uppercase;letter-spacing:1px}}
.stat-value{{font-size:28px;font-weight:700;color:#E6EDF3;font-family:'JetBrains Mono',monospace}}
.stats{{display:flex;gap:16px;margin:16px 0}}
.stat{{background:#161B22;border:1px solid #30363D;border-radius:6px;padding:16px;flex:1;text-align:center}}
table{{width:100%;border-collapse:collapse;margin:16px 0}}
th,td{{border:1px solid #30363D;padding:8px 12px;text-align:left;font-size:14px}}
th{{background:#161B22;color:#8B949E;text-transform:uppercase;font-size:12px}}
.section{{margin:24px 0}}
.report-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}}
.report-header-logo{{max-height:40px;width:auto;object-fit:contain}}
.llm-advisory{{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);border:1px solid #30363D;border-left:3px solid #7C3AED;border-radius:0 8px 8px 0;padding:16px 20px;margin:20px 0}}
.llm-advisory h3{{color:#7C3AED;font-size:13px;text-transform:uppercase;letter-spacing:0.5px;margin:0 0 8px 0;border:none}}
.llm-advisory p,.llm-advisory li{{color:#C9D1D9;font-size:14px;line-height:1.6;margin:4px 0}}
.llm-fallback{{background:#161B22;border:1px solid #30363D;border-radius:6px;padding:12px 16px;color:#8B949E;font-size:12px;font-style:italic;margin:16px 0}}
</style></head>
<body>
<div class="report-header">
<h1>AccessSentinel CISO Report</h1>
{_report_logo_img()}
</div>
<p>Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</p>

<div class="section"><h2>1. Executive Summary</h2>
<p>This report presents the results of an identity risk scan across {len(scan_results)} identities.
{critical} critical-risk, {high} high-risk, {medium} medium-risk, and {low} low-risk identities were identified.</p>
<div class="stats">
<div class="stat"><div class="stat-label">Critical</div><div class="stat-value">{critical}</div></div>
<div class="stat"><div class="stat-label">High</div><div class="stat-value">{high}</div></div>
<div class="stat"><div class="stat-label">Medium</div><div class="stat-value">{medium}</div></div>
<div class="stat"><div class="stat-label">Low</div><div class="stat-value">{low}</div></div>
</div></div>

{_build_llm_section(scan_results, critical, high, medium, low, top10, triggered_rule_ids, mitre_matches, remediation_actions)}

<div class="section"><h2>2. Risk Tier Distribution</h2>
<table><tr><th>Tier</th><th>Count</th><th>Percentage</th><th>Distribution</th></tr>
{tier_rows}</table></div>

<div class="section"><h2>3. Top 10 Risk Identities</h2>
<table><tr><th>Username</th><th>Dept</th><th>Score</th><th>Tier</th><th>Primary Rule</th></tr>
{top_rows}</table></div>

<div class="section"><h2>4. MITRE ATT&CK Summary</h2>
<p>{len(mitre_matches)} MITRE ATT&CK techniques triggered across {len(triggered_rule_ids)} rule categories.</p>
<table><tr><th>Technique ID</th><th>Name</th><th>Tactic</th><th>Triggered By Rule</th></tr>
{mitre_rows}</table></div>

<div class="section"><h2>5. Remediation Action Plan</h2>
<table><tr><th>Priority</th><th>Action Type</th><th>Description</th><th>SLA (hours)</th></tr>
{rem_rows if rem_rows else '<tr><td colspan="4">No automated remediation actions generated.</td></tr>'}</table></div>

<div class="section"><h2>6. Model Evaluation Metrics</h2>
<p>Evaluation metrics are available in the AccessSentinel dashboard under the Model Evaluation page.
Run a scan with matching label files to populate this section with F1, precision, recall, and confusion matrix data.</p></div>

<div class="section"><h2>7. Context Signals Summary</h2>
<p>AccessSentinel applies context-aware signal suppression to reduce false positives.
Signals include: SABBATICAL_POSSIBLE, TEMP_ELEVATION, NEW_HIRE_RAMP, BATCH_JOB_PATTERN,
MONTH_END_FINANCE, IMPOSSIBLE_TRAVEL_CONFIRMED, and CONTRACTOR_NORM.
Suppressed rules are preserved in the audit trail for review.</p></div>

<div class="section"><h2>8. Methodology</h2>
<p>AccessSentinel evaluates identity risk using a multi-layered ITDR pipeline:
Ingestion, Behavioral Baseline, Feature Extraction, Ensemble Detection (IF + OCSVM + LOF),
Rules Engine (13 detection rules), Context Resolver (7 signal types), Risk Scorer,
MITRE ATT&CK Mapper, and Remediation Engine.</p></div>
</body></html>"""
    return html
