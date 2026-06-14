import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import base64
from datetime import datetime, timedelta
import random

from dotenv import load_dotenv
load_dotenv()

from streamlit_design import apply_theme, apply_plotly_theme

st.set_page_config(layout="wide", page_title="AccessSentinel", page_icon="logo.png")

apply_theme()

# ── Custom component CSS classes (kept for app-specific HTML helpers) ──────────
st.markdown("""
<style>
.tier-critical, .tier-high, .tier-medium, .tier-low {
  display: inline-block;
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.05em;
}
.tier-critical { background-color: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.4); }
.tier-high     { background-color: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.4); }
.tier-medium   { background-color: rgba(6, 182, 212, 0.15); color: #06B6D4; border: 1px solid rgba(6, 182, 212, 0.4); }
.tier-low      { background-color: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.4); }

.score-display {
  font-size: 3rem;
  font-weight: 700;
  color: #F8FAFC;
  text-align: center;
  line-height: 1;
}
.score-label {
  font-size: 0.75rem;
  color: #94A3B8;
  text-align: center;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 0.25rem;
}

.panel {
  background: linear-gradient(145deg, var(--surface) 0%, rgba(15,23,42,0.6) 100%);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 8px;
  padding: 1rem 1.25rem;
  margin-bottom: 0.75rem;
}
.panel-header {
  font-size: 0.8rem;
  font-weight: 600;
  color: #94A3B8;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.75rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid rgba(255,255,255,0.07);
}

.narrative-block {
  background-color: var(--surface);
  border-left: 3px solid var(--cyan);
  padding: 0.75rem 1rem;
  border-radius: 0 8px 8px 0;
  font-size: 0.9rem;
  line-height: 1.6;
  color: #C9D1D9;
}

.cli-block {
  background-color: var(--bg);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 6px;
  padding: 0.75rem 1rem;
  font-size: 0.8rem;
  color: var(--green);
  white-space: pre;
  overflow-x: auto;
}

.suppressed-rule {
  background-color: var(--surface);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 4px;
  padding: 0.5rem 0.75rem;
  margin: 0.25rem 0;
  opacity: 0.65;
  font-size: 0.85rem;
}
.suppressed-label {
  font-size: 0.7rem;
  color: #94A3B8;
  text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

API_BASE = "http://localhost:8000/api/v1"

# ── Dark-theme tier colors ────────────────────────────────────────────────────

TIER_COLORS = {
    "CRITICAL": "#EF4444",
    "HIGH": "#F59E0B",
    "MEDIUM": "#06B6D4",
    "LOW": "#10B981",
}

TIER_BG = {
    "CRITICAL": "background-color: rgba(239,68,68,0.15); color: #EF4444; font-weight: 700; padding: 4px 10px; border-radius: 4px; border: 1px solid rgba(239,68,68,0.4);",
    "HIGH": "background-color: rgba(245,158,11,0.15); color: #F59E0B; font-weight: 700; padding: 4px 10px; border-radius: 4px; border: 1px solid rgba(245,158,11,0.4);",
    "MEDIUM": "background-color: rgba(6,182,212,0.15); color: #06B6D4; font-weight: 700; padding: 4px 10px; border-radius: 4px; border: 1px solid rgba(6,182,212,0.4);",
    "LOW": "background-color: rgba(16,185,129,0.15); color: #10B981; font-weight: 700; padding: 4px 10px; border-radius: 4px; border: 1px solid rgba(16,185,129,0.4);",
}


# ── Plotly dark theme helper (delegates to design module) ────────────────────

def get_dark_chart_layout(title: str | None = None) -> dict:
    """Return a Plotly layout dict applying the AccessSentinel dark theme."""
    layout: dict = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.5)",
        font=dict(color="#94A3B8", family="Inter, sans-serif", size=11),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)",
                   linecolor="rgba(255,255,255,0.08)", tickfont=dict(color="#94A3B8")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)",
                   linecolor="rgba(255,255,255,0.08)", tickfont=dict(color="#94A3B8")),
        margin=dict(l=16, r=16, t=36, b=16),
        showlegend=True,
        legend=dict(bgcolor="rgba(15,23,42,0.7)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1, font=dict(color="#94A3B8")),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(color="#F8FAFC", size=14))
    return layout

DEPARTMENTS = ["Finance", "Engineering", "IT", "HR", "Legal", "Marketing", "Operations", "Customer Support", "Product"]
SOURCE_SYSTEMS = ["Active Directory", "Azure AD", "Okta", "AWS IAM", "Google Workspace"]
ACCOUNT_TYPES = ["human", "service", "admin", "vendor"]
JOB_TITLES = {
    "Finance": ["CFO", "Controller", "Finance Manager", "Financial Analyst", "Senior Accountant"],
    "Engineering": ["Engineering Manager", "Senior Engineer", "Staff Engineer", "Software Engineer", "DevOps Engineer"],
    "IT": ["CISO", "Security Analyst", "Systems Administrator", "Network Engineer", "IT Support"],
    "HR": ["CHRO", "HR Manager", "HR Coordinator", "Recruiter", "Benefits Specialist"],
    "Legal": ["CLO", "General Counsel", "Corporate Counsel", "Compliance Officer", "Paralegal"],
    "Marketing": ["CMO", "Growth Lead", "Brand Manager", "Marketing Coordinator", "SEO Specialist"],
    "Operations": ["COO", "Ops Manager", "Logistics Coordinator", "Supply Chain Lead", "Ops Analyst"],
    "Customer Support": ["CSO", "Support Manager", "Support Team Lead", "Senior Support Agent", "Support Agent"],
    "Product": ["CPO", "Product Manager", "Associate PM", "Product Designer", "Product Analyst"],
}
MITRE_TECHNIQUES = [
    "T1078", "T1078.001", "T1078.002", "T1078.003", "T1078.004",
    "T1098", "T1098.001", "T1098.002", "T1098.003", "T1098.004",
    "T1136", "T1136.001", "T1136.002", "T1136.003",
    "T1525", "T1526", "T1530", "T1552", "T1552.004", "T1606",
]
RULES = [
    "STALE_PRIVILEGED", "ORPHANED_ACCOUNT", "OVER_PRIVILEGED",
    "SHADOW_ADMIN", "PRIVILEGE_CREEP", "SERVICE_ACCT_ABUSE",
    "CREDENTIAL_SPRAWL", "IMPOSSIBLE_TRAVEL", "EXCESSIVE_ACCESS",
    "BULK_DOWNLOAD", "SOD_VIOLATION",
]

random.seed(42)


def _mock_scan_results(count=500):
    usernames = [
        f"{random.choice(['john','jane','alex','sam','taylor','morgan','casey','riley','jordan','quinn','blake','avery','dakota','harper','reese','skyler','finley','sasha','devon','logan'])}{random.choice(['.','_',''])}{random.choice(['smith','johnson','williams','brown','jones','davis','miller','wilson','moore','taylor'])}{random.randint(1,999)}"
        for _ in range(count)
    ]
    results = []
    for i in range(count):
        dept = random.choice(DEPARTMENTS)
        tier_roll = random.random()
        if tier_roll < 0.05:
            tier = "CRITICAL"
            score = random.randint(75, 99)
        elif tier_roll < 0.18:
            tier = "HIGH"
            score = random.randint(55, 74)
        elif tier_roll < 0.45:
            tier = "MEDIUM"
            score = random.randint(30, 54)
        else:
            tier = "LOW"
            score = random.randint(1, 29)

        num_anomalies = random.randint(0, 4)
        anomaly_types = random.sample(RULES, min(num_anomalies, len(RULES)))
        if tier in ("CRITICAL", "HIGH") and not anomaly_types:
            anomaly_types = random.sample(RULES, 2)

        job = random.choice(JOB_TITLES.get(dept, ["Analyst"]))
        systems_count = random.randint(1, 8)
        mfa = random.random() > 0.25
        sso = random.random() > 0.3
        privileged = tier in ("CRITICAL", "HIGH") or random.random() < 0.1

        results.append({
            "user_id": f"user_{i:04d}",
            "username": usernames[i],
            "email": f"{usernames[i]}@{random.choice(['corp.com','company.org','enterprise.io'])}",
            "department": dept,
            "source_system": random.choice(SOURCE_SYSTEMS),
            "score": score,
            "tier": tier,
            "anomaly_types": anomaly_types,
            "mitre_technique": random.choice(MITRE_TECHNIQUES) if anomaly_types else "",
            "primary_rule": anomaly_types[0] if anomaly_types else "",
            "suppressed_rule_count": random.randint(0, 3),
            "blast_radius_applied": tier in ("CRITICAL", "HIGH") and random.random() > 0.5,
            "account_type": "service" if random.random() < 0.08 else ("admin" if random.random() < 0.12 else "human"),
            "employment_status": random.choice(["active", "active", "active", "departed", "suspended"]),
            "mfa_enabled": mfa,
            "sso_linked": sso,
            "last_login": (datetime.now() - timedelta(days=random.randint(0, 180))).isoformat(),
            "created_at": (datetime.now() - timedelta(days=random.randint(500, 1500))).isoformat(),
            "roles": [f"{dept}_{job}"],
            "permissions": random.sample(
                ["read:reports", "write:reports", "admin:console", "read:logs", "write:config", "read:users", "write:users", "s3:admin", "db:readonly"],
                random.randint(1, 5)
            ),
            "systems_count": systems_count,
            "off_hours_access_pct": round(random.uniform(0, 0.45), 3),
            "is_privileged": privileged,
            "risk_narrative": f"Identity {usernames[i]} in {dept} shows {tier.lower()} risk profile. " + (
                "Multiple anomaly detection rules triggered, including " + ", ".join(anomaly_types[:2]) + ". " if anomaly_types else ""
            ) + "Recommend immediate review of access patterns and privilege assignments.",
            "contributing_factors": anomaly_types,
            "suppressed_factors": random.sample(RULES, min(random.randint(0, 2), len(RULES))),
            "context_signals": [
                {
                    "signal_type": random.choice(["ACCESS_REVIEW_PENDING", "SABBATICAL", "NEW_HIRE_RAMP", "MONTH_END_SURGE", "BATCH_JOB"]),
                    "explanation": f"Context signal detected for {usernames[i]} based on employment patterns.",
                    "confidence": round(random.uniform(0.5, 0.95), 2),
                    "score_adjustment": random.randint(-15, -5),
                    "rules_suppressed": random.sample(RULES, min(1, len(RULES))),
                    "requires_followup": random.random() > 0.7,
                }
            ] if anomaly_types else [],
            "mitre_techniques": [
                {
                    "technique_id": random.choice(MITRE_TECHNIQUES),
                    "name": random.choice(["Valid Accounts", "Account Manipulation", "Create Account", "Cloud Accounts", "Unsecured Credentials"]),
                    "tactic": random.choice(["Persistence", "Privilege Escalation", "Defense Evasion", "Credential Access"]),
                    "url": f"https://attack.mitre.org/techniques/{random.choice(MITRE_TECHNIQUES).replace('.','/')}",
                    "triggered_by_rule": random.choice(RULES),
                }
            ] if anomaly_types else [],
            "remediation_actions": [
                {
                    "priority": n + 1,
                    "action_type": random.choice(["REVOKE_ACCESS", "DISABLE_ACCOUNT", "REVIEW_PERMISSIONS", "ENABLE_MFA", "ROTATE_CREDENTIALS"]),
                    "target": usernames[i],
                    "human_readable_description": f"Review and remediate access for {usernames[i]} in {dept}.",
                    "machine_actionable_command": f"Disable-ADAccount -Identity '{usernames[i]}'",
                    "estimated_risk_reduction": random.randint(10, 40),
                    "expected_resolution_hours": random.choice([2, 4, 8, 24, 48, 72]),
                    "requires_approval": tier in ("CRITICAL", "HIGH"),
                }
                for n in range(min(random.randint(1, 4), 3))
            ] if anomaly_types else [],
            "behavior_zscore": round(random.uniform(-2.5, 4.5), 2),
            "confidence": round(random.uniform(0.55, 0.98), 2),
            "cluster_assignment": str(random.randint(0, 7)),
            "peer_deviation_score": round(random.uniform(0, 3.5), 2),
            "sequence_risk": {
                "pattern_detected": random.random() > 0.7,
                "pattern_type": random.choice(["ESCALATION", "LATERAL_MOVEMENT", "DATA_EXFIL"]),
                "confidence": round(random.uniform(0.5, 0.9), 2),
            } if random.random() > 0.6 else None,
            "triggered_rules": [{"rule_id": r, "severity": random.choice(["CRITICAL", "HIGH", "MEDIUM", "LOW"]), "triggered": True, "evidence_text": f"Rule {r} triggered for {usernames[i]}", "suppressed_by": None} for r in anomaly_types[:2]],
            "suppressed_rules": [],
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _mock_scan_response(results):
    critical = sum(1 for r in results if r["tier"] == "CRITICAL")
    high = sum(1 for r in results if r["tier"] == "HIGH")
    medium = sum(1 for r in results if r["tier"] == "MEDIUM")
    low = sum(1 for r in results if r["tier"] == "LOW")
    return {
        "message": f"Scan complete. {len(results)} identities analyzed.",
        "leaderboard": [
            {
                "user_id": r["user_id"], "username": r["username"], "email": r["email"],
                "department": r["department"], "source_system": r["source_system"],
                "score": r["score"], "tier": r["tier"], "anomaly_types": r["anomaly_types"],
                "mitre_technique": r["mitre_technique"], "primary_rule": r["primary_rule"],
                "suppressed_rule_count": r["suppressed_rule_count"], "blast_radius_applied": r["blast_radius_applied"],
            }
            for r in results
        ],
        "total": len(results),
        "critical_count": critical, "high_count": high,
        "medium_count": medium, "low_count": low, "f1_score": None,
    }


def _mock_clusters():
    return [
        {"cluster_id": "0", "label": "Standard Users", "user_count": 62, "avg_risk_score": 18.3, "dominant_resources": ["email", "drive", "chat"], "dominant_actions": ["read", "send"], "outlier_count": 3},
        {"cluster_id": "1", "label": "Engineering Power Users", "user_count": 28, "avg_risk_score": 42.1, "dominant_resources": ["github", "jenkins", "aws-console"], "dominant_actions": ["deploy", "commit"], "outlier_count": 5},
        {"cluster_id": "2", "label": "Finance Analysts", "user_count": 19, "avg_risk_score": 35.7, "dominant_resources": ["erp-finance", "excel", "powerbi"], "dominant_actions": ["export", "approve"], "outlier_count": 2},
        {"cluster_id": "3", "label": "IT Admins", "user_count": 14, "avg_risk_score": 67.8, "dominant_resources": ["admin-console", "active-directory", "servicenow"], "dominant_actions": ["grant", "revoke", "reset"], "outlier_count": 6},
        {"cluster_id": "4", "label": "HR Personnel", "user_count": 12, "avg_risk_score": 22.4, "dominant_resources": ["workday", "bamboo-hr", "slack"], "dominant_actions": ["read", "update"], "outlier_count": 1},
        {"cluster_id": "5", "label": "Shadow Admins", "user_count": 7, "avg_risk_score": 85.2, "dominant_resources": ["iam", "s3", "db-admin"], "dominant_actions": ["admin", "delete", "modify"], "outlier_count": 7},
        {"cluster_id": "6", "label": "Legal & Compliance", "user_count": 8, "avg_risk_score": 19.9, "dominant_resources": ["docusign", "sharepoint", "compliance-db"], "dominant_actions": ["sign", "review"], "outlier_count": 0},
        {"cluster_id": "7", "label": "Marketing Ops", "user_count": 11, "avg_risk_score": 26.5, "dominant_resources": ["hubspot", "salesforce", "tableau"], "dominant_actions": ["campaign", "export"], "outlier_count": 2},
    ]


def _mock_graph():
    nodes = []
    edges = []
    departments_sample = DEPARTMENTS[:6]
    for i in range(40):
        dept = random.choice(departments_sample)
        tier = random.choice(["CRITICAL", "HIGH", "MEDIUM", "LOW"])
        uid = f"user_{i:04d}"
        nodes.append({
            "id": uid, "label": f"user_{random.choice(['alice','bob','charlie','diana','eve','frank','grace'])}{i}",
            "type": "user", "tier": tier, "score": random.randint(1, 99),
            "shadow_admin": tier == "HIGH" and random.random() > 0.6,
        })
        for j in range(random.randint(1, 3)):
            role_id = f"role:{dept}_{random.choice(['admin','readonly','contributor','viewer'])}"
            if not any(n["id"] == role_id for n in nodes):
                nodes.append({"id": role_id, "label": role_id.replace("role:", ""), "type": "role"})
            edges.append({"source": uid, "target": role_id, "type": "user_role"})
        sys_id = f"system:{random.choice(SOURCE_SYSTEMS)}"
        if not any(n["id"] == sys_id for n in nodes):
            nodes.append({"id": sys_id, "label": sys_id.replace("system:", ""), "type": "system"})
        edges.append({"source": uid, "target": sys_id, "type": "user_system"})
    orphaned = ["user_orphan_1", "user_orphan_2"]
    for oid in orphaned:
        nodes.append({"id": oid, "label": oid, "type": "user", "tier": "LOW", "score": 5, "shadow_admin": False})
    return {"nodes": nodes, "edges": edges}


def _mock_evaluation():
    return {
        "overall_f1": 0.800,
        "macro_f1": 0.857,
        "weighted_f1": 0.910,
        "precision_by_class": {"0": 0.95, "1": 0.86},
        "recall_by_class": {"0": 0.98, "1": 0.75},
        "f1_by_class": {"0": 0.96, "1": 0.80},
        "false_positive_rate": 0.024,
        "false_positive_rate_by_dept": {
            "Finance": 0.06, "Engineering": 0.03, "IT": 0.00,
            "HR": 0.00, "Legal": 0.00, "Marketing": 0.00,
            "Operations": 0.00, "Customer Support": 0.00, "Product": 0.00,
        },
        "confusion_matrix_data": [[82, 2], [4, 12]],
    }


def _mock_behavioral_timeline(username):
    months = []
    now = datetime.now()
    for i in range(11, -1, -1):
        months.append((now - timedelta(days=30 * i)).strftime("%Y-%m"))
    baseline_mean = [random.uniform(20, 60) for _ in range(12)]
    baseline_std = [m * 0.25 for m in baseline_mean]
    user_events = [m + random.uniform(-1.5, 2.5) * s for m, s in zip(baseline_mean, baseline_std)]
    user_events[9] = baseline_mean[9] + 4.2 * baseline_std[9]
    user_events[10] = baseline_mean[10] + 3.8 * baseline_std[10]
    user_events[11] = baseline_mean[11] + 5.1 * baseline_std[11]
    return months, baseline_mean, baseline_std, user_events


def _mock_permission_gauge():
    return random.randint(15, 65)


def _to_csv_download_link(df, filename="export.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="csv-download-btn">Export to CSV</a>'
    return href


def _styled_tier_cell(val):
    color_map = {
        "CRITICAL": "color: #EF4444; font-weight: 700;",
        "HIGH": "color: #F59E0B; font-weight: 700;",
        "MEDIUM": "color: #06B6D4; font-weight: 700;",
        "LOW": "color: #10B981; font-weight: 700;",
    }
    return color_map.get(val, "")


def _score_bar(val):
    pct = max(0, min(100, val))
    bar = "█" * int(pct / 4)
    return f"{val:3d} {bar}"


def _tier_html(val):
    bg = {
        "CRITICAL": "rgba(239,68,68,0.15)", "HIGH": "rgba(245,158,11,0.15)",
        "MEDIUM": "rgba(6,182,212,0.15)", "LOW": "rgba(16,185,129,0.15)",
    }
    tc = {
        "CRITICAL": "#EF4444", "HIGH": "#F59E0B",
        "MEDIUM": "#06B6D4", "LOW": "#10B981",
    }
    return f'<span style="background:{bg.get(val,"#0F172A")};color:{tc.get(val,"#94A3B8")};padding:3px 10px;border-radius:4px;font-weight:700;font-size:12px;border:1px solid {tc.get(val,"rgba(255,255,255,0.07)")};">{val}</span>'


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE-SPECIFIC CSS OVERRIDES
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main > div { max-width: 100%; padding: 0 24px; }
    .csv-download-btn {
        display: inline-block; padding: 8px 16px;
        background: var(--surface-2); color: var(--cyan); border-radius: 4px;
        text-decoration: none; font-weight: 600; font-size: 14px;
        border: 1px solid rgba(6, 182, 212, 0.4); margin: 8px 0;
    }
    .csv-download-btn:hover { background: var(--cyan); color: var(--bg); text-decoration: none; }
    .narrative-box {
        border: 1px solid rgba(255,255,255,0.07); border-left: 3px solid var(--cyan);
        padding: 0.75rem 1rem; border-radius: 0 8px 8px 0;
        background: var(--surface); margin: 12px 0; font-size: 0.9rem;
        line-height: 1.6; color: #C9D1D9;
    }
    .profile-stat {
        background: var(--surface); border: 1px solid rgba(255,255,255,0.07); border-radius: 8px;
        padding: 12px 16px; text-align: center;
    }
    .profile-stat .label { font-size: 11px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px; }
    .profile-stat .value { font-size: 16px; font-weight: 700; color: #F8FAFC; }
    .section-heading {
        font-size: 16px; font-weight: 700; color: #F8FAFC;
        border-bottom: 2px solid var(--cyan); padding-bottom: 6px; margin: 20px 0 12px 0;
    }
    .methodology-box {
        background: var(--surface); border-left: 3px solid var(--cyan);
        padding: 12px 16px; border-radius: 6px; font-size: 13px;
        margin: 12px 0; color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

# Load logo for sidebar branding
_logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logo.png")
_logo_b64 = ""
try:
    with open(_logo_path, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()
except Exception:
    pass

with st.sidebar:
    if _logo_b64:
        st.markdown(
            f'<img src="data:image/png;base64,{_logo_b64}" alt="AccessSentinel" '
            f'style="max-height:28px;width:auto;display:block;margin-bottom:6px">',
            unsafe_allow_html=True,
        )
    st.markdown(
        "<h2 style='color: #F8FAFC; font-weight: 700; margin-bottom: 0;'>AccessSentinel</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.75rem; margin-top: 0;'>Identity Risk Intelligence</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio("Navigation", [
        "Scan",
        "Leaderboard",
        "Drill-Down",
        "Role Clusters",
        "Integrations",
        "Evaluation",
        "Export Report",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 0.75rem;'>v1.0.0</p>",
        unsafe_allow_html=True,
    )

# ── Page routing map ───────────────────────────────────────────────────────────

PAGE_MAP = {
    "Scan": "Upload & Scan",
    "Leaderboard": "Risk Leaderboard",
    "Drill-Down": "Identity Drill-Down",
    "Role Clusters": "Role Clusters & Graph",
    "Integrations": "Integrations",
    "Evaluation": "Model Evaluation",
    "Export Report": "CISO Report",
}
page = PAGE_MAP.get(page, page)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 - Upload & Scan
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Upload & Scan":
    st.markdown("<h2 style='color: #F8FAFC;'>Identity Risk Scan</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        users_file = st.file_uploader(
            "Upload Identity Users CSV",
            type=["csv"],
            accept_multiple_files=False,
            key="scan_users_upload",
        )
        events_file = st.file_uploader(
            "Upload Identity Events CSV",
            type=["csv"],
            accept_multiple_files=False,
            key="scan_events_upload",
        )
        with st.expander("Optional: Upload evaluation labels"):
            st.caption(
                "Label files are only needed for evaluation metrics "
                "(F1, precision, recall, FPR by department). "
                "Scanning and risk scoring work without them."
            )
            users_labels_file = st.file_uploader(
                "Upload Users Labels CSV (optional)",
                type=["csv"],
                accept_multiple_files=False,
                key="scan_users_labels_upload",
            )
            events_labels_file = st.file_uploader(
                "Upload Events Labels CSV (optional)",
                type=["csv"],
                accept_multiple_files=False,
                key="scan_events_labels_upload",
            )
        st.caption(
            "Users and events files are enough for scanning and risk scoring. "
            "Both label files are optional and only needed for evaluation metrics."
        )
        has_both = users_file is not None and events_file is not None
        submitted = st.button("Run Scan", disabled=not has_both, width="stretch")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        load_demo = st.button("Load Demo Dataset", width="stretch")

    if "scan_results" not in st.session_state:
        st.session_state.scan_results = None
    if "scan_summary" not in st.session_state:
        st.session_state.scan_summary = None
    if "last_scan_time" not in st.session_state:
        st.session_state.last_scan_time = None
    if "scan_identities" not in st.session_state:
        st.session_state.scan_identities = None
    if "scan_clusters" not in st.session_state:
        st.session_state.scan_clusters = None
    if "scan_graph" not in st.session_state:
        st.session_state.scan_graph = None
    if "scan_in_progress" not in st.session_state:
        st.session_state.scan_in_progress = False

    need_scan = False
    scan_users_bytes = None
    scan_events_bytes = None
    scan_labels_bytes = None
    scan_events_labels_bytes = None
    scan_url = f"{API_BASE}/scan"

    if load_demo and not st.session_state.scan_in_progress:
        need_scan = True
        st.session_state.scan_in_progress = True
        try:
            with open("sample_data/identity_users.csv", "rb") as f:
                scan_users_bytes = f.read()
        except Exception:
            scan_users_bytes = None
        try:
            with open("sample_data/identity_events.csv", "rb") as f:
                scan_events_bytes = f.read()
        except Exception:
            scan_events_bytes = None
        try:
            with open("sample_data/identity_users_labels.csv", "rb") as f:
                scan_labels_bytes = f.read()
        except Exception:
            scan_labels_bytes = None
        try:
            with open("sample_data/identity_events_labels.csv", "rb") as f:
                scan_events_labels_bytes = f.read()
        except Exception:
            scan_events_labels_bytes = None

    if submitted and not st.session_state.scan_in_progress:
        need_scan = True
        st.session_state.scan_in_progress = True
        scan_users_bytes = users_file.getvalue()
        scan_events_bytes = events_file.getvalue()
        if users_labels_file is not None:
            scan_labels_bytes = users_labels_file.getvalue()
        if events_labels_file is not None:
            scan_events_labels_bytes = events_labels_file.getvalue()

    # Show warnings only when the user clicks Run Scan with incomplete files
    if submitted and (users_file is None or events_file is None):
        if users_file is None:
            st.warning("A users CSV is required. Please upload both files.")
        if events_file is None:
            st.warning("An events CSV is required. Please upload both files.")

    if need_scan and scan_users_bytes:
        progress_bar = st.progress(0, text="Initializing...")
        stages = [
            (0.2, "Ingesting identity records"),
            (0.4, "Building behavioral baselines"),
            (0.6, "Running ensemble anomaly detection"),
            (0.75, "Applying context resolution"),
            (0.9, "Scoring and mapping MITRE techniques"),
        ]
        response_data = None
        api_success = False

        for pct, label in stages:
            progress_bar.progress(pct, text=f"Stage {stages.index((pct, label)) + 1}: {label}")
            if not api_success:
                try:
                    files_payload: dict = {
                        "users_file": ("users.csv", scan_users_bytes, "text/csv"),
                    }
                    if scan_events_bytes:
                        files_payload["events_file"] = ("events.csv", scan_events_bytes, "text/csv")
                    if scan_labels_bytes:
                        files_payload["users_labels_file"] = ("users_labels.csv", scan_labels_bytes, "text/csv")
                    if scan_events_labels_bytes:
                        files_payload["events_labels_file"] = ("events_labels.csv", scan_events_labels_bytes, "text/csv")
                    resp = requests.post(
                        scan_url,
                        files=files_payload,  # type: ignore[arg-type]
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        response_data = resp.json()
                        api_success = True
                        progress_bar.progress(pct, text=f"Stage {stages.index((pct, label)) + 1}: {label}")
                        time.sleep(0.3)
                        continue
                except Exception:
                    pass
            if not api_success:
                time.sleep(0.2)

        progress_bar.progress(1.0, text="Scan complete")

        if api_success and response_data:
            # Fetch verified evaluation metrics to populate the F1 card
            try:
                eval_resp = requests.get(f"{API_BASE}/evaluate", timeout=30)
                if eval_resp.status_code == 200:
                    response_data["f1_score"] = eval_resp.json().get("overall_f1")
            except Exception:
                pass
            st.session_state.scan_summary = response_data
            st.session_state.scan_results = response_data.get("leaderboard", [])

            # Store identities for downstream pages (Role Clusters, Drill-Down)
            try:
                id_resp = requests.get(f"{API_BASE}/identities?page_size=200", timeout=10)
                if id_resp.status_code == 200:
                    st.session_state.scan_identities = id_resp.json().get("items", [])
                else:
                    st.session_state.scan_identities = _mock_scan_results(500)
            except Exception:
                st.session_state.scan_identities = _mock_scan_results(500)

            # Store clusters
            try:
                c_resp = requests.get(f"{API_BASE}/clusters", timeout=5)
                if c_resp.status_code == 200:
                    st.session_state.scan_clusters = c_resp.json()
                else:
                    st.session_state.scan_clusters = _mock_clusters()
            except Exception:
                st.session_state.scan_clusters = _mock_clusters()

            # Store graph data
            try:
                g_resp = requests.get(f"{API_BASE}/graph", timeout=5)
                if g_resp.status_code == 200:
                    st.session_state.scan_graph = g_resp.json()
                else:
                    st.session_state.scan_graph = _mock_graph()
            except Exception:
                st.session_state.scan_graph = _mock_graph()
        else:
            mock_results = _mock_scan_results(500)
            st.session_state.scan_summary = _mock_scan_response(mock_results)
            st.session_state.scan_results = mock_results
            st.session_state.scan_identities = mock_results
            st.session_state.scan_clusters = _mock_clusters()
            st.session_state.scan_graph = _mock_graph()
            st.info("Using mock data (API unavailable). Upload a CSV and ensure the AccessSentinel API is running at localhost:8000 for live analysis.")

        st.session_state.last_scan_time = datetime.now()
        st.session_state.scan_in_progress = False
        time.sleep(0.5)
        st.rerun()

    if st.session_state.scan_summary:
        summary = st.session_state.scan_summary
        st.markdown("---")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total", summary.get("total", 0))
        c2.metric("CRITICAL", summary.get("critical_count", 0), delta_color="inverse")
        c3.metric("HIGH", summary.get("high_count", 0), delta_color="inverse")
        c4.metric("MEDIUM", summary.get("medium_count", 0), delta_color="off")
        c5.metric("LOW", summary.get("low_count", 0), delta_color="normal")
        f1_val = summary.get("f1_score")
        c6.metric("F1 Score", f"{f1_val:.3f}" if f1_val else "N/A")

        st.markdown("---")

        if st.session_state.scan_results:
            df_results = pd.DataFrame(st.session_state.scan_results)
            anomaly_counts = {}
            for r in st.session_state.scan_results:
                for at in r.get("anomaly_types", []):
                    anomaly_counts[at] = anomaly_counts.get(at, 0) + 1
            if anomaly_counts:
                adf = pd.DataFrame(
                    {"Anomaly Type": list(anomaly_counts.keys()), "Count": list(anomaly_counts.values())}
                ).sort_values("Count", ascending=True)
                fig = px.bar(
                    adf, x="Count", y="Anomaly Type", orientation="h",
                    title="Anomaly Type Distribution",
                    color_discrete_sequence=["#06B6D4"],
                )
                fig.update_layout(**get_dark_chart_layout())
                st.plotly_chart(fig, width="stretch")

        placeholder = st.empty()
        current_time = datetime.now()
        if st.session_state.last_scan_time:
            elapsed = (current_time - st.session_state.last_scan_time).seconds
            if elapsed < 30:
                with placeholder:
                    st.caption(f"Scan completed {elapsed}s ago. Auto-refreshing in {30 - elapsed}s...")
                    time.sleep(1)
                    st.rerun()

    elif not need_scan:
        st.info("Upload a CSV file or click 'Load Demo Dataset' to begin the identity risk scan.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 - Risk Leaderboard
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Risk Leaderboard":
    st.title("Risk Leaderboard")

    with st.sidebar:
        st.markdown("### Filters")
        tier_filter = st.multiselect("Tier", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH", "MEDIUM", "LOW"])
        PAGE_SIZE = 50
        if "leaderboard_page" not in st.session_state:
            st.session_state.leaderboard_page = 1

        # Fetch current page from API
        api_available = False
        all_results = []
        total_count = 0
        try:
            resp = requests.get(
                f"{API_BASE}/identities?page_size={PAGE_SIZE}&page={st.session_state.leaderboard_page}",
                timeout=10,
            )
            if resp.status_code == 200:
                all_data = resp.json()
                all_results = all_data.get("items", [])
                total_count = all_data.get("total", len(all_results))
                api_available = True
        except Exception:
            pass

        if not all_results:
            all_results = st.session_state.get("scan_results") or []

        if isinstance(all_results, list) and len(all_results) > 0:
            pass
        else:
            all_results = _mock_scan_results(500)

        if isinstance(all_results, list) and len(all_results) > 0 and isinstance(all_results[0], dict):
            depts = sorted(set(r.get("department", "Unknown") for r in all_results))
            systems = sorted(set(r.get("source_system", "Unknown") for r in all_results))
        else:
            depts = DEPARTMENTS
            systems = SOURCE_SYSTEMS

        dept_filter = st.selectbox("Department", ["All"] + depts)
        system_filter = st.selectbox("Source System", ["All"] + systems)
        anomaly_types_all = sorted(set(
            at for r in (all_results if isinstance(all_results, list) else [])
            for at in r.get("anomaly_types", [])
        ))
        anomaly_filter = st.selectbox("Anomaly Type", ["All"] + anomaly_types_all)
        search_term = st.text_input("Search Username/Email", "")

        # Page navigation
        total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE) if total_count else 1
        st.markdown("---")
        st.caption(f"Page {st.session_state.leaderboard_page} of {total_pages} ({total_count} total)")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Previous Page", disabled=st.session_state.leaderboard_page <= 1):
                st.session_state.leaderboard_page -= 1
                st.rerun()
        with c2:
            if st.button("Next Page", disabled=st.session_state.leaderboard_page >= total_pages):
                st.session_state.leaderboard_page += 1
                st.rerun()

    results = all_results if isinstance(all_results, list) else []

    # Show data source indicator after API check
    if st.session_state.get("last_scan_time"):
        elapsed = (datetime.now() - st.session_state.last_scan_time).seconds
        source = "live API" if api_available else "cached"
        st.caption(f"Data from last scan ({elapsed}s ago) — {source}")

    if not results:
        st.info("No scan results available. Run a scan from the Scan page first, then return here.")
        st.stop()
    if tier_filter:
        results = [r for r in results if r.get("tier") in tier_filter]
    if dept_filter and dept_filter != "All":
        results = [r for r in results if r.get("department") == dept_filter]
    if system_filter and system_filter != "All":
        results = [r for r in results if r.get("source_system") == system_filter]
    if anomaly_filter and anomaly_filter != "All":
        results = [r for r in results if anomaly_filter in r.get("anomaly_types", [])]
    if search_term:
        st_lower = search_term.lower()
        results = [r for r in results if st_lower in r.get("username", "").lower() or st_lower in r.get("email", "").lower()]

    rows = []
    for rank, r in enumerate(results, 1):
        rows.append({
            "Rank": rank,
            "Username": r.get("username", ""),
            "Source System": r.get("source_system", ""),
            "Score": r.get("score", 0),
            "Tier": r.get("tier", ""),
            "MITRE Technique": r.get("mitre_technique", ""),
            "Primary Rule": r.get("primary_rule", ""),
            "Suppressed Rules": r.get("suppressed_rule_count", 0),
            "Blast Radius Applied": "Yes" if r.get("blast_radius_applied") else "No",
            "_tier_html": _tier_html(r.get("tier", "")),
            "_score_bar": _score_bar(r.get("score", 0)),
        })

    df = pd.DataFrame(rows)

    critical_in_results = sum(1 for r in results if r.get("tier") == "CRITICAL")
    show_start = (st.session_state.leaderboard_page - 1) * PAGE_SIZE + 1 if results else 0
    show_end = show_start + len(results) - 1 if results else 0
    st.markdown(
        f"<span style='color: #94A3B8; font-size: 0.85rem;'>Showing {show_start}–{show_end} of {total_count} identities"
        f" &nbsp;|&nbsp; {critical_in_results} CRITICAL (this page)</span>",
        unsafe_allow_html=True,
    )

    # CSV Export: fetch ALL filtered pages for complete export
    # The visible table shows first 200 rows for performance;
    # export fetches all pages and applies the same filters.
    col_tbl, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("Export to CSV", key="export_csv"):
            with st.spinner("Fetching all filtered identities..."):
                all_pages = list(all_results)  # start with page 1
                page = 2
                max_pages = 10  # safety limit
                while len(all_pages) < 2000 and page <= max_pages:
                    try:
                        resp = requests.get(
                            f"{API_BASE}/identities?page_size=200&page={page}", timeout=10
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            items = data.get("items", [])
                            if not items:
                                break
                            all_pages.extend(items)
                            page += 1
                        else:
                            break
                    except Exception:
                        break
                # Apply same filters as visible table
                export_filtered = all_pages
                if tier_filter:
                    export_filtered = [r for r in export_filtered if r.get("tier") in tier_filter]
                if dept_filter and dept_filter != "All":
                    export_filtered = [r for r in export_filtered if r.get("department") == dept_filter]
                if system_filter and system_filter != "All":
                    export_filtered = [r for r in export_filtered if r.get("source_system") == system_filter]
                if anomaly_filter and anomaly_filter != "All":
                    export_filtered = [r for r in export_filtered if anomaly_filter in r.get("anomaly_types", [])]
                if search_term:
                    st_lower = search_term.lower()
                    export_filtered = [r for r in export_filtered if st_lower in r.get("username", "").lower() or st_lower in r.get("email", "").lower()]

                export_rows = []
                for rank, r in enumerate(export_filtered, 1):
                    export_rows.append({
                        "Rank": rank, "Username": r.get("username", ""),
                        "Source System": r.get("source_system", ""),
                        "Score": r.get("score", 0), "Tier": r.get("tier", ""),
                        "MITRE Technique": r.get("mitre_technique", ""),
                        "Primary Rule": r.get("primary_rule", ""),
                        "Suppressed Rules": r.get("suppressed_rule_count", 0),
                        "Blast Radius Applied": "Yes" if r.get("blast_radius_applied") else "No",
                    })
                export_df = pd.DataFrame(export_rows)
                csv_data = export_df.to_csv(index=False)
                b64 = base64.b64encode(csv_data.encode()).decode()
                st.markdown(
                    f'<a href="data:file/csv;base64,{b64}" download="risk_leaderboard.csv" class="csv-download-btn">Download CSV ({len(export_rows)} rows)</a>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Exported {len(export_rows)} rows from {len(all_pages)} total identities across API.")

    if not df.empty:
        display_cols = ["Rank", "Username", "Source System", "Score", "Tier", "MITRE Technique", "Primary Rule", "Suppressed Rules", "Blast Radius Applied"]
        display_df = df[display_cols].copy()
        display_df["Score"] = df["_score_bar"]
        display_df["Tier"] = df["_tier_html"]

        st.markdown(
            display_df.to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )
    else:
        st.info("No identities match the selected filters.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 - Identity Drill-Down
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Identity Drill-Down":
    st.title("Identity Profile")

    try:
        resp = requests.get(f"{API_BASE}/identities?page_size=200", timeout=5)
        if resp.status_code == 200:
            all_ids = resp.json().get("items", [])
        else:
            all_ids = st.session_state.get("scan_results") or _mock_scan_results(500)
    except Exception:
        all_ids = st.session_state.get("scan_results") or _mock_scan_results(500)

    if isinstance(all_ids, list) and len(all_ids) > 0 and isinstance(all_ids[0], dict):
        usernames = [r.get("username", r.get("user_id", "")) for r in all_ids]
    else:
        all_ids = _mock_scan_results(500)
        usernames = [r["username"] for r in all_ids]

    selected_username = st.selectbox("Select Identity", usernames, key="drilldown_select")

    identity = None
    for r in all_ids:
        if isinstance(r, dict) and r.get("username") == selected_username:
            identity = r
            break

    if not identity:
        mock_ids = _mock_scan_results(500)
        for r in mock_ids:
            if r["username"] == selected_username:
                identity = r
                break

    if identity:
        uid = identity.get("user_id", "")
        try:
            detail_resp = requests.get(f"{API_BASE}/identities/{uid}", timeout=5)
            if detail_resp.status_code == 200:
                detail = detail_resp.json()
            else:
                detail = identity
        except Exception:
            detail = identity

        st.markdown('<div class="section-heading">Profile Summary</div>', unsafe_allow_html=True)
        cols = st.columns(6)
        profiles = [
            ("Account Type", detail.get("account_type", "N/A")),
            ("Employment Status", detail.get("employment_status", "N/A")),
            ("MFA Enabled", "Yes" if detail.get("mfa_enabled") else "No"),
            ("SSO Linked", "Yes" if detail.get("sso_linked") else "No"),
            ("Account Age", "N/A"),
            ("Cluster", detail.get("cluster_assignment", "N/A")),
        ]
        if detail.get("created_at"):
            try:
                created = datetime.fromisoformat(detail["created_at"].replace("Z", "+00:00"))
                days = (datetime.now() - created.replace(tzinfo=None)).days
                profiles[4] = ("Account Age", f"{days} days")
            except Exception:
                pass

        for col, (label, val) in zip(cols, profiles):
            with col:
                st.markdown(
                    f'<div class="profile-stat"><div class="label">{label}</div><div class="value">{val}</div></div>',
                    unsafe_allow_html=True,
                )

        narrative = detail.get("risk_narrative", "No risk narrative available.")
        st.markdown(
            f'<div class="narrative-box"><strong>Risk Narrative:</strong> {narrative}</div>',
            unsafe_allow_html=True,
        )

        context_signals = detail.get("context_signals", [])
        if context_signals:
            st.markdown('<div class="section-heading">Context Signals</div>', unsafe_allow_html=True)
            for sig in context_signals:
                with st.expander(f"{sig.get('signal_type', 'Signal')} (Confidence: {sig.get('confidence', 0):.0%})"):
                    st.write(sig.get("explanation", ""))
                    st.caption(f"Score Adjustment: {sig.get('score_adjustment', 0)}")
                    suppressed = sig.get("rules_suppressed", [])
                    if suppressed:
                        st.caption(f"Rules Suppressed: {', '.join(suppressed)}")

        triggered = detail.get("triggered_rules", [])
        if triggered:
            st.markdown('<div class="section-heading">Rules Triggered</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(triggered), width="stretch", hide_index=True)

        suppressed = detail.get("suppressed_rules", [])
        if suppressed:
            st.markdown('<div class="section-heading">Rules Suppressed</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(suppressed), width="stretch", hide_index=True)

        st.markdown('<div class="section-heading">Behavioral Timeline</div>', unsafe_allow_html=True)
        months, b_mean, b_std, u_events = _mock_behavioral_timeline(selected_username)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=[m + 2 * s for m, s in zip(b_mean, b_std)],
            fill=None, mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=months, y=[max(0, m - 2 * s) for m, s in zip(b_mean, b_std)],
            fill="tonexty", fillcolor="rgba(56, 139, 253, 0.08)",
            mode="lines", line=dict(width=0), name="Baseline +/- 2 std",
        ))
        fig.add_trace(go.Scatter(
            x=months, y=b_mean, mode="lines",
            line=dict(color="#94A3B8", dash="dot", width=1.5), name="Baseline Mean",
        ))
        fig.add_trace(go.Scatter(
            x=months, y=u_events, mode="lines+markers",
            line=dict(color="#06B6D4", width=2), marker=dict(size=6),
            name=f"{selected_username} Events",
        ))
        fig.update_layout(**get_dark_chart_layout(title="User Events vs Baseline (12 Months)"))
        st.plotly_chart(fig, width="stretch")

        col_g1, col_g2 = st.columns([1, 2])
        with col_g1:
            st.markdown('<div class="section-heading">Permission Utilization</div>', unsafe_allow_html=True)
            util_pct = _mock_permission_gauge()
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=util_pct,
                title={"text": "Permissions Used (%)"},
                delta={"reference": 100, "decreasing": {"color": "#27ae60"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#06B6D4"},
                    "steps": [
                        {"range": [0, 40], "color": "rgba(35,134,54,0.15)"},
                        {"range": [40, 70], "color": "rgba(210,153,34,0.15)"},
                        {"range": [70, 100], "color": "rgba(218,54,51,0.15)"},
                    ],
                    "threshold": {"line": {"color": "#F8FAFC", "width": 3}, "thickness": 0.8, "value": 70},
                },
            ))
            fig_g.update_layout(**get_dark_chart_layout())
            st.plotly_chart(fig_g, width="stretch")

        mitre_list = detail.get("mitre_techniques", [])
        if mitre_list:
            with col_g2:
                st.markdown('<div class="section-heading">MITRE ATT&CK Techniques</div>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(mitre_list), width="stretch", hide_index=True)

        remediation = detail.get("remediation_actions", [])
        if remediation:
            st.markdown('<div class="section-heading">Remediation Plan</div>', unsafe_allow_html=True)
            for action in sorted(remediation, key=lambda x: x.get("priority", 99)):
                with st.expander(
                    f"Priority {action.get('priority', '?')} - {action.get('action_type', '')} - "
                    f"{action.get('human_readable_description', '')[:80]}..."
                ):
                    st.write(f"**Action:** {action.get('human_readable_description', '')}")
                    cmd = action.get("machine_actionable_command", "")
                    st.markdown(f'<div class="cli-block">{cmd}</div>', unsafe_allow_html=True)
                    st.caption(
                        f"Risk Reduction: {action.get('estimated_risk_reduction', 0)}% | "
                        f"SLA: {action.get('expected_resolution_hours', 0)}h | "
                        f"Approval Required: {'Yes' if action.get('requires_approval') else 'No'}"
                    )

        blast = detail.get("blast_radius_applied", False)
        if blast:
            st.warning(
                "Blast Radius Applied: This identity is part of a broader risk cluster. "
                "Remediation actions may impact multiple dependent identities and systems."
            )
    else:
        st.info("No identity data available. Run a scan first from the Upload & Scan page.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 - Role Clusters & Privilege Graph
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Role Clusters & Graph":
    st.title("Role Clusters & Privilege Graph")

    view_mode = st.radio("View", ["Cluster Scatter", "Privilege Graph", "Multi-Cloud IdP"], horizontal=True, key="cluster_view")

    if view_mode == "Cluster Scatter":
        st.markdown('<div class="section-heading">Cluster Scatter Plot</div>', unsafe_allow_html=True)

        identities = st.session_state.get("scan_identities")
        if not identities:
            try:
                resp = requests.get(f"{API_BASE}/identities?page_size=200", timeout=5)
                if resp.status_code == 200:
                    identities = resp.json().get("items", [])
            except Exception:
                pass
        if not identities:
            identities = _mock_scan_results(500)

        scatter_data = []
        for r in identities:
            days_since = r.get("days_inactive", None)
            if days_since is None:
                try:
                    last_login_str = r.get("last_login", "")
                    if last_login_str:
                        last_login = datetime.fromisoformat(last_login_str.replace("Z", "+00:00"))
                        days_since = (datetime.now() - last_login.replace(tzinfo=None)).days
                    else:
                        days_since = random.randint(0, 180)
                except Exception:
                    days_since = random.randint(0, 180)
            else:
                try:
                    days_since = int(float(str(days_since)))
                except (ValueError, TypeError):
                    days_since = random.randint(0, 180)
            scatter_data.append({
                "Username": r.get("username", ""),
                "Days Since Login": int(days_since),
                "Risk Score": r.get("score", 0),
                "Cluster": r.get("cluster_assignment", "N/A"),
                "Tier": r.get("tier", "LOW"),
                "Department": r.get("department", ""),
            })

        if not scatter_data:
            st.warning("No identity data available for cluster scatter plot. Run a scan first.")
        else:
            sdf = pd.DataFrame(scatter_data)
            is_outlier = sdf["Days Since Login"] > 120
            sdf["Marker"] = sdf["Tier"].apply(lambda t: "x" if t in ("CRITICAL", "HIGH") else "circle")

            fig_s = px.scatter(
                sdf, x="Days Since Login", y="Risk Score",
                color="Cluster", symbol="Tier",
                hover_data=["Username", "Tier", "Department"],
                title="Identity Risk by Days Since Login",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_s.update_layout(**get_dark_chart_layout(title="Identity Risk by Days Since Login"))
            fig_s.update_layout(height=400)
            fig_s.update_traces(marker=dict(size=9, line=dict(width=1, color="rgba(255,255,255,0.07)")))
            st.plotly_chart(fig_s, width="stretch")

        st.markdown('<div class="section-heading">Cluster Summary</div>', unsafe_allow_html=True)
        clusters = st.session_state.get("scan_clusters") or []
        if not clusters:
            try:
                c_resp = requests.get(f"{API_BASE}/clusters", timeout=5)
                if c_resp.status_code == 200:
                    clusters = c_resp.json()
            except Exception:
                pass
        if not clusters:
            clusters = _mock_clusters()

        cdf = pd.DataFrame(clusters)
        st.dataframe(cdf, width="stretch", hide_index=True)
        csv_b64 = base64.b64encode(cdf.to_csv(index=False).encode()).decode()
        st.markdown(
            f'<a href="data:file/csv;base64,{csv_b64}" download="clusters.csv" class="csv-download-btn">Export to CSV</a>',
            unsafe_allow_html=True,
        )

    elif view_mode == "Privilege Graph":
        st.markdown('<div class="section-heading">Privilege Graph</div>', unsafe_allow_html=True)

        graph_data = st.session_state.get("scan_graph")
        if not graph_data:
            try:
                g_resp = requests.get(f"{API_BASE}/graph", timeout=5)
                if g_resp.status_code == 200:
                    graph_data = g_resp.json()
            except Exception:
                pass
        if not graph_data:
            graph_data = _mock_graph()

        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if nodes:
            import networkx as nx
            G = nx.Graph()
            nids = set()
            for n in nodes:
                G.add_node(n["id"], label=n.get("label", n["id"]), type=n.get("type", "user"),
                           tier=n.get("tier", "LOW"), score=n.get("score", 0),
                           shadow_admin=n.get("shadow_admin", False))
                nids.add(n["id"])
            for e in edges:
                if e["source"] in nids and e["target"] in nids:
                    G.add_edge(e["source"], e["target"], type=e.get("type", ""))

            pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=0.5, color="#aaa"),
                hoverinfo="none",
            )

            tier_cmap = {"CRITICAL": "#e94560", "HIGH": "#e67e22", "MEDIUM": "#f1c40f", "LOW": "#27ae60"}
            user_node_x, user_node_y, user_colors, user_text, user_sizes = [], [], [], [], []
            other_node_x, other_node_y, other_text = [], [], []

            for n in G.nodes():
                x, y = pos[n]
                nd = G.nodes[n]
                if nd.get("type") == "user":
                    user_node_x.append(x)
                    user_node_y.append(y)
                    user_colors.append(tier_cmap.get(nd.get("tier", "LOW"), "#27ae60"))
                    user_text.append(f"{nd.get('label','')}<br>Tier: {nd.get('tier','')}<br>Score: {nd.get('score',0)}<br>Shadow: {nd.get('shadow_admin',False)}")
                    user_sizes.append(18 if nd.get("shadow_admin") else 12)
                else:
                    other_node_x.append(x)
                    other_node_y.append(y)
                    other_text.append(nd.get("label", ""))

            user_trace = go.Scatter(
                x=user_node_x, y=user_node_y, mode="markers",
                marker=dict(size=user_sizes, color=user_colors, line=dict(width=1, color="#333")),
                text=user_text, hoverinfo="text",
                name="Identities",
            )
            other_trace = go.Scatter(
                x=other_node_x, y=other_node_y, mode="markers",
                marker=dict(size=8, color="#888", symbol="square", line=dict(width=1, color="#555")),
                text=other_text, hoverinfo="text",
                name="Roles/Systems",
            )

            fig_g = go.Figure(data=[edge_trace, user_trace, other_trace])
            fig_g.update_layout(**get_dark_chart_layout(title="Privilege & Access Graph"))
            fig_g.update_layout(
                height=600,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig_g, width="stretch")

            st.markdown('<div class="section-heading">Identity Details</div>', unsafe_allow_html=True)
            selected_node = st.selectbox("Click a user node label to view profile", [""] + [
                nd.get("label", "") for nd in nodes if nd.get("type") == "user"
            ], key="graph_user_select")
            if selected_node:
                selected_data = next((nd for nd in nodes if nd.get("label") == selected_node and nd.get("type") == "user"), None)
                if selected_data:
                    st.markdown(
                        f"**{selected_data.get('label', '')}** | Tier: {selected_data.get('tier', '')} | "
                        f"Score: {selected_data.get('score', 0)} | Shadow Admin: {selected_data.get('shadow_admin', False)}"
                    )
        else:
            st.info("No graph data available. Run a scan first.")

    elif view_mode == "Multi-Cloud IdP":
        st.markdown('<div class="section-heading">Multi-Cloud Identity Providers</div>', unsafe_allow_html=True)
        try:
            idp_resp = requests.get(f"{API_BASE}/idp-summary", timeout=10)
            if idp_resp.status_code == 200:
                idp_data = idp_resp.json()
            else:
                idp_data = None
        except Exception:
            idp_data = None

        if idp_data:
            providers = idp_data.get("providers", {})
            c1, c2, c3 = st.columns(3)
            for col, (name, key) in zip(
                [c1, c2, c3],
                [("Okta", "okta"), ("Azure AD", "azuread"), ("AWS IAM", "aws_iam")],
            ):
                p = providers.get(key, {})
                src = p.get("source_type", "mock")
                src_badge = f'<span style="font-size:0.65rem;color:{"#10B981" if src=="live" else "#F59E0B"};font-family:monospace">{"LIVE" if src=="live" else "MOCK"}</span>'
                err = p.get("error", "")
                with col:
                    st.markdown(
                        f'<div class="panel"><div class="panel-header">{name} {src_badge}</div>'
                        f'<span style="color:#F8FAFC;font-size:1.5rem;font-weight:700">{p.get("total",0)}</span>'
                        f'<span style="color:#94A3B8;font-size:0.8rem"> users</span><br>'
                        f'<span style="color:#10B981">{p.get("active",0)} active</span> | '
                        f'<span style="color:#EF4444">{p.get("inactive",0)} inactive</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if err:
                        st.caption(f"Error: {err[:80]}")

            st.markdown(
                f'<span style="font-size:0.75rem;color:#94A3B8">Overall mode: '
                f'{"LIVE" if idp_data.get("overall_mode")=="live" else "MOCK (credentials not configured)"}</span>',
                unsafe_allow_html=True,
            )

            # AWS-specific risks
            aws = providers.get("aws_iam", {})
            aws_keys = aws.get("access_keys", {})
            if aws_keys:
                st.markdown('<div class="section-heading">AWS IAM Access Keys</div>', unsafe_allow_html=True)
                ck1, ck2 = st.columns(2)
                ck1.metric("Total Keys", aws_keys.get("total_keys", 0))
                ck2.metric("Active Keys", aws_keys.get("active_keys", 0),
                           delta=f"{aws_keys.get('inactive_keys', 0)} inactive",
                           delta_color="off")
                if aws_keys.get("key_sprawl_detected"):
                    st.warning("AWS IAM key sprawl detected — multiple active access keys. Rotate inactive keys.")

            st.markdown(f'<div class="panel">{idp_data.get("summary", "")}</div>', unsafe_allow_html=True)
        else:
            st.info("IdP integration data unavailable. Run a scan first.")

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 - Integrations
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Integrations":
    st.markdown("<h2 style='color: #F8FAFC;'>API Integrations</h2>", unsafe_allow_html=True)
    st.caption("Identity provider connectors for Okta, Azure AD, and AWS IAM.")
    st.info(
        "To enable LIVE mode, configure provider credentials in the environment "
        "and restart the application. See .env.example and README for setup instructions. "
        "All providers default to MOCK mode when credentials are not configured. "
        "Note: Provider connectors provide identity inventory visibility. "
        "Risk scoring requires event data from uploaded CSV files or the demo dataset."
    )

    # Page-level refresh button
    if st.button("Refresh Integration Status", width="stretch"):
        st.rerun()

    st.markdown("---")

    api_error = None
    try:
        resp = requests.get(f"{API_BASE}/integrations/status", timeout=(3, 5))
        if resp.status_code == 200:
            status_data = resp.json()
        else:
            api_error = f"HTTP {resp.status_code} from {API_BASE}/integrations/status"
            status_data = None
    except requests.exceptions.ConnectionError:
        api_error = f"Backend not reachable at {API_BASE}. Is the API server running?"
        status_data = None
    except requests.exceptions.Timeout:
        api_error = f"Status request timed out. Backend may be overloaded."
        status_data = None
    except Exception as e:
        api_error = f"Request failed: {str(e)[:500]}"
        status_data = None

    if not status_data:
        st.warning(f"Integration status API unavailable. {api_error or 'Check that the API server is running on port 8000.'}")
    else:
        providers_status = status_data.get("providers", {})

        provider_info = {
            "okta": {
                "name": "Okta",
                "env_vars": ["OKTA_ORG_URL", "OKTA_API_TOKEN"],
                "description": "Fetches users from Okta REST API using SSWS token authentication.",
            },
            "azuread": {
                "name": "Azure AD / Microsoft Graph",
                "env_vars": ["AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET"],
                "description": "Fetches users from Microsoft Graph using OAuth2 client credentials grant.",
            },
            "aws_iam": {
                "name": "AWS IAM",
                "env_vars": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
                "description": "Fetches IAM users using Boto3 SDK (preferred) or AWS Signature V4 HTTP requests.",
            },
        }

        for key, info in provider_info.items():
            ps = providers_status.get(key, {})
            configured = ps.get("configured", False)
            status = ps.get("status", "not_configured")

            # Reconcile: if env vars are missing, override status to not_configured
            # regardless of what the API returned (prevents display contradictions)
            all_vars_present = all(bool(os.environ.get(v)) for v in info["env_vars"])
            if not all_vars_present:
                configured = False
                status = "not_configured"

            # Status badge: NOT CONFIGURED / CONFIGURED / LIVE / ERROR
            status_badges = {
                "not_configured": ("#94A3B8", "NOT CONFIGURED"),
                "configured": ("#F59E0B", "CONFIGURED"),
                "live": ("#10B981", "LIVE"),
                "error": ("#EF4444", "ERROR"),
            }
            badge_color, badge_text = status_badges.get(status, ("#94A3B8", status.upper()))
            is_live = status == "live"

            with st.expander(
                f"{info['name']}  [{badge_text}]",
                expanded=is_live,
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Status:** {status.upper()}")
                    st.markdown(f"**Configured:** {'Yes' if configured else 'No'}")
                    st.markdown(f"**HTTP Status:** {ps.get('http_status', 'N/A')}")
                with col2:
                    st.markdown(f"**Error:** {ps.get('error', 'None')}")
                    if status == "live":
                        st.markdown(f"**Account:** {ps.get('account', 'N/A')}")
                        st.markdown(f"**ARN:** {ps.get('arn', 'N/A')}")
                        st.markdown(f"**User ID:** {ps.get('user_id', 'N/A')}")
                    elif status == "configured":
                        st.markdown(f"**Account:** N/A (not yet verified)")
                    else:
                        st.markdown(f"**Account:** N/A")

                if status == "live":
                    st.success("Verified live against AWS STS")
                    st.markdown(f"**Account:** {ps.get('account', 'N/A')}")
                    st.markdown(f"**Account Alias:** {ps.get('account_alias') or 'N/A'}")
                    st.markdown(f"**ARN:** {ps.get('arn', 'N/A')}")
                    st.markdown(f"**User ID:** {ps.get('user_id', 'N/A')}")

                    # IAM Posture Snapshot
                    if ps.get("iam_user_count") is not None or ps.get("sample_users"):
                        st.markdown("---")
                        st.markdown("#### AWS IAM Posture Snapshot")
                        st.caption("Connector-derived AWS IAM posture indicators (fetched live from AWS; not part of core event risk scoring)")

                        if ps.get("iam_user_count") is not None:
                            st.markdown(f"**IAM Users:** {ps['iam_user_count']}")

                        sample_users = ps.get("sample_users", [])
                        if sample_users:
                            st.markdown("**Sampled Users:**")
                            su_data = []
                            for su in sample_users:
                                su_data.append({
                                    "User Name": su.get("user_name", ""),
                                    "Created": su.get("create_date", "N/A")[:10] if su.get("create_date") else "N/A",
                                    "Pwd Last Used": su.get("password_last_used", "N/A")[:10] if su.get("password_last_used") else "N/A",
                                    "Attached Policies": su.get("attached_policy_count", 0),
                                    "Inline Policies": su.get("inline_policy_count", 0),
                                })
                            st.dataframe(pd.DataFrame(su_data), use_container_width=True, hide_index=True)

                        # Credential Report Summary
                        cr = ps.get("credential_report_summary")
                        if cr:
                            st.markdown("**Credential Hygiene:**")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Total Users", cr.get("total_report_users", 0))
                            c2.metric("MFA Disabled", cr.get("users_with_mfa_disabled", 0))
                            c3.metric("Active Keys", cr.get("users_with_active_access_keys", 0))
                            c4.metric("Pwd Unused >90d", cr.get("users_with_password_unused_90d", 0))

                        # Heuristics
                        heuristics = ps.get("heuristics", {})
                        if heuristics:
                            st.markdown("**Derived Heuristics:**")
                            if heuristics.get("potentially_privileged_users"):
                                st.warning(f"Potentially privileged: {', '.join(heuristics['potentially_privileged_users'])}")
                            if heuristics.get("mfa_gap_candidates"):
                                st.warning("MFA gap candidates detected — see credential report")
                            if heuristics.get("credential_sprawl_candidates"):
                                st.warning("Credential sprawl candidates detected — see credential report")

                        st.caption("These AWS IAM indicators support identity-sprawl, privileged-access, MFA-gap, and credential-hygiene visibility. They are connector-derived visibility signals only and do not affect the main risk leaderboard.")

                    if ps.get("warning"):
                        st.caption(f"Note: {ps['warning']}")

                elif status == "configured":
                    st.info("Credentials configured but not yet verified. Click Test Connection.")
                elif status == "error":
                    st.error(f"Connection failed: {ps.get('error', 'Unknown error')}")

                st.markdown("---")
                st.markdown("#### Required Environment Variables")
                import os as _os
                for var in info["env_vars"]:
                    present = bool(_os.environ.get(var))
                    st.markdown(
                        f"- `{var}`: "
                        f"<span style='color:{'#10B981' if present else '#94A3B8'}'>"
                        f"{'PRESENT' if present else 'MISSING'}</span>",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                st.caption(info["description"])
                st.caption(
                    "When required credentials are configured via environment variables, "
                    "this provider attempts live API fetches. Otherwise it runs in MOCK mode."
                )

                if st.button(f"Test {info['name']} Connection", key=f"test_{key}"):
                    with st.spinner(f"Testing {info['name']} connection (may take up to 60s)..."):
                        try:
                            test_resp = requests.get(
                                f"{API_BASE}/integrations/test/{key}",
                                timeout=(5, 60),
                            )
                            if test_resp.status_code == 200:
                                test_data = test_resp.json()
                                p = test_data  # deep endpoint returns result directly
                                s = p.get("status", "not_configured")
                                if s == "live":
                                    st.success(f"{info['name']}: LIVE — verified against AWS STS")
                                    st.caption(f"Account: {p.get('account')} | ARN: {p.get('arn')} | User: {p.get('user_id')}")
                                    if p.get("iam_user_count") is not None:
                                        st.caption(f"IAM Users: {p['iam_user_count']}")
                                elif s == "configured":
                                    st.warning(f"{info['name']}: CONFIGURED — credentials set but not verified live. Click Test Connection again.")
                                elif s == "error":
                                    st.error(f"{info['name']}: ERROR — {p.get('error', 'Unknown')}")
                                else:
                                    st.info(f"{info['name']}: NOT CONFIGURED — set environment variables to enable")
                            else:
                                st.error(f"Test failed: HTTP {test_resp.status_code}")
                        except requests.exceptions.Timeout:
                            st.error(f"{info['name']}: Live enrichment timed out. AWS API took too long to respond.")
                        except Exception as e:
                            st.error(f"Test failed: {e}")

    st.markdown("---")
    if not status_data:
        overall = "api_unavailable"
    else:
        overall = status_data.get("overall_mode", "not_configured")
    mode_labels = {
        "live": "LIVE — at least one provider successfully connected",
        "not_configured": "NOT CONFIGURED — no provider credentials set",
        "configured": "CONFIGURED — credentials set but not verified live",
        "api_unavailable": "API UNAVAILABLE — backend not reachable",
    }
    st.caption(f"Overall: {mode_labels.get(overall, overall.upper())}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 - Model Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Evaluation":
    st.title("Model Evaluation")

    if st.button("Run Evaluation", type="primary", width="stretch"):
        eval_data = None
        api_success = False
        try:
            resp = requests.get(f"{API_BASE}/evaluate", timeout=30)
            if resp.status_code == 200:
                eval_data = resp.json()
                api_success = True
        except Exception:
            pass

        if not api_success or not eval_data:
            eval_data = _mock_evaluation()
            st.info("Using mock evaluation data (API unavailable).")

        # Show evaluation dataset source
        source = eval_data.get("source", "unknown")
        source_label = "Uploaded dataset" if source == "uploaded" else "Sample dataset" if source == "sample_data" else source

        labels_src = eval_data.get("labels_source", "unknown")
        eval_status = eval_data.get("evaluation_status", "unknown")
        match_count = eval_data.get("label_match_count", -1)
        labels_unavailable = eval_status in ("missing_labels", "label_mismatch")

        # Status header
        if eval_status == "missing_labels":
            st.warning(
                "**Evaluation metrics are not available for this uploaded dataset.**\n\n"
                "This uploaded dataset is valid for scanning and risk scoring. "
                "However, metrics like F1, precision, and recall require matching "
                "ground truth labels, which are not part of the standard two-file "
                "upload (users CSV + events CSV).\n\n"
                "To see evaluation metrics, use the demo dataset (which includes "
                "built-in labels) or provide a matching users label CSV."
            )
        elif eval_status == "label_mismatch":
            st.warning(
                f"**Uploaded labels do not match this dataset.**\n\n"
                f"No uploaded user IDs matched the provided labels "
                f"({match_count} matches found). The label file must use "
                f"the same user_id values as the uploaded users CSV."
            )
        elif labels_src == "uploaded" and eval_status == "ok":
            st.info(f"Evaluation complete using uploaded labels ({match_count} matching IDs)")
        else:
            st.caption(f"Dataset: {source_label}  |  Labels: {labels_src}")

        st.markdown("---")

        # Metric cards — show N/A when labels unavailable
        c1, c2, c3 = st.columns(3)
        if labels_unavailable:
            c1.metric("Overall F1 Score", "N/A")
            c2.metric("Macro F1", "N/A")
            c3.metric("Weighted F1", "N/A")
        else:
            c1.metric("Overall F1 Score", f"{eval_data.get('overall_f1', 0):.3f}")
            c2.metric("Macro F1", f"{eval_data.get('macro_f1', 0):.3f}")
            c3.metric("Weighted F1", f"{eval_data.get('weighted_f1', 0):.3f}")

        st.markdown("---")

        if not labels_unavailable:
            st.markdown('<div class="section-heading">Precision / Recall / F1 by Class</div>', unsafe_allow_html=True)
            prf_data = {
                "Class": ["Normal (0)", "Anomaly (1)"],
                "Precision": [
                    eval_data.get("precision_by_class", {}).get("0", 0),
                    eval_data.get("precision_by_class", {}).get("1", 0),
                ],
                "Recall": [
                    eval_data.get("recall_by_class", {}).get("0", 0),
                    eval_data.get("recall_by_class", {}).get("1", 0),
                ],
                "F1 Score": [
                    eval_data.get("f1_by_class", {}).get("0", 0),
                    eval_data.get("f1_by_class", {}).get("1", 0),
                ],
            }
            st.dataframe(pd.DataFrame(prf_data), width="stretch", hide_index=True)

            st.markdown('<div class="section-heading">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = eval_data.get("confusion_matrix_data", [[0, 0], [0, 0]])
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Normal", "Anomaly"], y=["Normal", "Anomaly"],
                color_continuous_scale="Blues",
                title="Confusion Matrix",
            )
            fig_cm.update_layout(**get_dark_chart_layout())
            st.plotly_chart(fig_cm, width="stretch")

            fpr_dept = eval_data.get("false_positive_rate_by_dept", {})
            if fpr_dept:
                st.markdown('<div class="section-heading">False Positive Rate by Department</div>', unsafe_allow_html=True)
                fpr_df = pd.DataFrame(
                    {"Department": list(fpr_dept.keys()), "FPR": list(fpr_dept.values())}
                ).sort_values("FPR", ascending=True)
                bar_colors = ["#F59E0B" if v > 0.20 else "#06B6D4" for v in fpr_df["FPR"]]
                fig_fpr = px.bar(
                    fpr_df, x="FPR", y="Department", orientation="h",
                    title="FPR by Department",
                )
                fig_fpr.update_traces(marker_color=bar_colors)
                fig_fpr.update_layout(**get_dark_chart_layout())
                fig_fpr.update_xaxes(range=[0, 1], title="False Positive Rate")
                st.plotly_chart(fig_fpr, width="stretch")

            st.markdown(
                '<div class="methodology-box">Evaluated against ground truth labels from the hackathon dataset.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Precision/recall breakdown and confusion matrix are not available without matching ground truth labels. The scan and risk scoring pipeline works correctly on the uploaded dataset — only the accuracy evaluation step requires labels.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 - CISO Report Export
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "CISO Report":
    st.title("Export")

    if st.button("Generate CISO Report", type="primary", width="stretch"):
        report_html = None
        api_success = False
        try:
            resp = requests.get(f"{API_BASE}/report", timeout=30)
            if resp.status_code == 200:
                report_html = resp.text
                api_success = True
        except Exception:
            pass

        if not api_success or not report_html:
            st.info("Generating mock report (API unavailable).")
            mock_results = st.session_state.get("scan_results") or _mock_scan_results(500)
            total = len(mock_results)
            critical = sum(1 for r in mock_results if r["tier"] == "CRITICAL")
            high = sum(1 for r in mock_results if r["tier"] == "HIGH")
            medium = sum(1 for r in mock_results if r["tier"] == "MEDIUM")
            low = sum(1 for r in mock_results if r["tier"] == "LOW")
            avg_score = sum(r["score"] for r in mock_results) / total if total else 0
            avg_conf = sum(r.get("confidence", 0) for r in mock_results) / total if total else 0

            top10_rows = ""
            for i, r in enumerate(mock_results[:10]):
                top10_rows += (
                    f"<tr><td>{i+1}</td><td>{r['username']}</td><td>{r['department']}</td>"
                    f"<td><span class='badge badge-{r['tier'].lower()}'>{r['tier']}</span></td>"
                    f"<td>{r['score']}</td><td>{r.get('primary_rule','')}</td>"
                    f"<td>{', '.join(r.get('anomaly_types',[])[:2])}</td></tr>"
                )

            report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AccessSentinel CISO Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 40px; color: #1a1a2e; background: #f5f6fa; line-height: 1.6; }}
    .container {{ max-width: 960px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
    h1 {{ color: #16213e; margin-top: 0; font-size: 28px; }}
    h2 {{ color: #0f3460; border-bottom: 2px solid #e94560; padding-bottom: 8px; margin-top: 32px; font-size: 20px; }}
    .meta {{ color: #666; font-size: 14px; margin-bottom: 24px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }}
    th, td {{ padding: 10px 12px; text-align: left; border: 1px solid #ddd; }}
    th {{ background: #16213e; color: #fff; font-weight: 600; }}
    tr:nth-child(even) {{ background: #f8f9fa; }}
    .stat-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 16px 0; }}
    .stat-card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 16px; text-align: center; }}
    .stat-value {{ font-size: 32px; font-weight: 700; }}
    .stat-label {{ font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
    .critical {{ color: #e94560; }} .high {{ color: #e67e22; }} .medium {{ color: #b7950b; }} .low {{ color: #27ae60; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
    .badge-critical {{ background: #fde8ec; color: #e94560; }}
    .badge-high {{ background: #fef3e2; color: #e67e22; }}
    .badge-medium {{ background: #fef9e7; color: #b7950b; }}
    .badge-low {{ background: #e8f8ef; color: #27ae60; }}
    .methodology {{ background: #f8f9fa; padding: 16px; border-radius: 6px; border-left: 4px solid #16213e; font-size: 14px; }}
    .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #ddd; font-size: 12px; color: #999; text-align: center; }}
</style>
</head>
<body>
<div class="container">
<h1>AccessSentinel Identity Risk Assessment Report</h1>
<p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC | AccessSentinel v1.0.0</p>
<h2>Executive Summary</h2>
<p>AccessSentinel analyzed <strong>{total}</strong> identity records. Average risk score: <strong>{avg_score:.1f}</strong> / 100. Average confidence: <strong>{avg_conf:.1%}</strong>.</p>
<div class="stat-grid">
    <div class="stat-card"><div class="stat-value critical">{critical}</div><div class="stat-label">Critical</div></div>
    <div class="stat-card"><div class="stat-value high">{high}</div><div class="stat-label">High</div></div>
    <div class="stat-card"><div class="stat-value medium">{medium}</div><div class="stat-label">Medium</div></div>
    <div class="stat-card"><div class="stat-value low">{low}</div><div class="stat-label">Low</div></div>
</div>
<h2>Top 10 Highest-Risk Identities</h2>
<table><tr><th>#</th><th>Username</th><th>Department</th><th>Tier</th><th>Score</th><th>Primary Rule</th><th>Anomaly Types</th></tr>{top10_rows}</table>
<h2>Methodology</h2>
<div class="methodology">
<p>AccessSentinel evaluates identity risk using a multi-layered pipeline: Ingestion, Behavioral Baseline, Feature Extraction, Ensemble Detection (IF + OCSVM + LOF), Rules Engine (11 detection rules), Context Resolver, Risk Scorer, MITRE ATT&CK Mapper, and Remediation Engine.</p>
</div>
<div class="footer"><p>AccessSentinel -- Identity Security Posture Management</p><p>This report is auto-generated and contains sensitive security information.</p></div>
</div>
</body>
</html>"""

        b64 = base64.b64encode(report_html.encode()).decode()
        st.markdown(
            f'<a href="data:text/html;base64,{b64}" download="accesssentinel_ciso_report.html" class="csv-download-btn">Download CISO Report (HTML)</a>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown('<div class="section-heading">Report Preview</div>', unsafe_allow_html=True)

        with st.expander("Executive Summary"):
            st.write("Summary of identity risk distribution across the organization.")
            results_for_preview = st.session_state.get("scan_results") or _mock_scan_results(500)
            total = len(results_for_preview)
            crit = sum(1 for r in results_for_preview if r["tier"] == "CRITICAL")
            hi = sum(1 for r in results_for_preview if r["tier"] == "HIGH")
            med = sum(1 for r in results_for_preview if r["tier"] == "MEDIUM")
            lo = sum(1 for r in results_for_preview if r["tier"] == "LOW")
            st.write(f"Total: {total} | Critical: {crit} | High: {hi} | Medium: {med} | Low: {lo}")

        with st.expander("Risk Tier Distribution Chart"):
            results_for_preview = st.session_state.get("scan_results") or _mock_scan_results(500)
            tier_counts = {}
            for r in results_for_preview:
                t = r["tier"]
                tier_counts[t] = tier_counts.get(t, 0) + 1
            tdf = pd.DataFrame({"Tier": list(tier_counts.keys()), "Count": list(tier_counts.values())})
            fig_d = px.pie(tdf, names="Tier", values="Count", title="Risk Tier Distribution",
                           color="Tier", color_discrete_map=TIER_COLORS)
            fig_d.update_layout(**get_dark_chart_layout())
            st.plotly_chart(fig_d, width="stretch")

        with st.expander("Top Risk Identities Table"):
            preview = st.session_state.get("scan_results") or _mock_scan_results(500)
            pvdf = pd.DataFrame([
                {"Username": r["username"], "Department": r["department"], "Score": r["score"], "Tier": r["tier"]}
                for r in preview[:10]
            ])
            st.dataframe(pvdf, width="stretch", hide_index=True)

        with st.expander("MITRE ATT&CK Summary"):
            mt_counts = {}
            for r in (st.session_state.get("scan_results") or _mock_scan_results(500)):
                mt = r.get("mitre_technique", "")
                if mt:
                    mt_counts[mt] = mt_counts.get(mt, 0) + 1
            mtdf = pd.DataFrame({"Technique": list(mt_counts.keys()), "Count": list(mt_counts.values())}).sort_values("Count", ascending=False)
            st.dataframe(mtdf, width="stretch", hide_index=True)

        with st.expander("Methodology"):
            st.markdown(
                "1. **Ingestion** - Parses and normalizes identity and event data from CSV/JSON sources.\n"
                "2. **Behavioral Baseline** - Establishes per-department, per-role behavioral profiles using z-score analysis.\n"
                "3. **Feature Extraction** - Computes 23 fixed-length features per identity.\n"
                "4. **Ensemble Detection** - Combines IsolationForest, OneClassSVM, and LocalOutlierFactor.\n"
                "5. **Rules Engine** - Applies 11 named detection rules.\n"
                "6. **Context Resolver** - Disambiguates ambiguous scenarios and suppresses false positives.\n"
                "7. **Risk Scorer** - Computes final scores (0-100) with blast-radius amplification.\n"
                "8. **MITRE ATT&CK Mapper** - Maps triggered rules to MITRE techniques.\n"
                "9. **Remediation Engine** - Generates prioritized CLI commands with SLA targets."
            )
    else:
        st.info("Click 'Generate CISO Report' to create a self-contained HTML report of the current scan results.")
