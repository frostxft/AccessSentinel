# AccessSentinel — Identity Risk Intelligence Platform

**Societe Generale x MASAI Hackathon 2026 (PS1: Identity Sprawl & Privilege Abuse Detection)**

AccessSentinel is a production-grade identity risk intelligence platform that ingests user and event data, builds behavioral baselines, detects anomalies using ensemble ML + rule-based scoring, maps risks to MITRE ATT&CK techniques, and generates prioritized remediation plans with source-system-aware CLI commands.

---

## Why This Matters

Modern organizations manage thousands of identities across multiple identity providers, cloud platforms, and on-premises systems. Attackers increasingly exploit **valid accounts** rather than breaking in — using legitimate credentials to move laterally, escalate privileges, and exfiltrate data without triggering traditional perimeter alerts.

AccessSentinel addresses three critical gaps in identity security:

- **Identity sprawl**: Orphaned accounts, stale privileged credentials, and unmanaged service identities accumulate over time without detection
- **Privilege abuse**: Users accumulate excessive permissions through role changes, creating shadow admins and privilege creep that IAM tools miss
- **Valid-account misuse**: Malicious insiders and compromised accounts blend into normal behavior patterns, evading signature-based detection

By combining unsupervised anomaly detection, 13 deterministic detection rules, and context-aware signal suppression, AccessSentinel surfaces the risky identities that traditional IAM audits overlook — and produces CISO-ready remediation plans with actionable CLI commands.

---

## Key Differentiators

| Capability | Why It Matters |
|-----------|----------------|
| **Ensemble ML + rules** | Three independent anomaly detectors (IF + OneClassSVM + LOF) vote, augmented by 13 deterministic rules — reducing false positives through convergent signals |
| **MITRE ATT&CK mapping** | Every triggered rule maps to a MITRE technique (T1078, T1098, etc.), giving analysts immediate threat-context understanding |
| **Deterministic remediation** | Each risky identity gets a prioritized action plan with source-system-specific CLI commands (Active Directory, Azure AD, AWS IAM, GCP, Okta), SLA estimates, and approval requirements |
| **Blast radius analysis** | NetworkX-based reachability graph calculates downstream access impact when a privileged identity is compromised |
| **Context-aware suppression** | 7 context signal detectors (sabbatical leave, new-hire ramp, finance month-end, etc.) automatically reduce false positives without losing audit trail |
| **Multi-provider visibility** | Okta, Azure AD, and AWS IAM connectors provide inventory visibility; AWS IAM goes deeper with credential report analysis and heuristic risk detection |
| **CISO-ready reporting** | Self-contained HTML report with 8 sections: executive summary, tier distribution, top identities, MITRE table, remediation plan, evaluation metrics, context signals, methodology |
| **Optional AI advisory narrative** | DeepSeek-powered executive summary adds business-context analysis to the CISO report — advisory-only, never overrides deterministic findings |
| **Real-time evaluation** | 80/20 stratified holdout with LogReg stacking across 4 base models + rule features + discriminators; FPR broken down by department |

---

## Architecture

```
CSV Upload → Ingestion → Behavioral Baseline → Feature Extraction
    → Ensemble Detection (IF + SVM + LOF) → Rules Engine (13 rules)
    → Context Resolver (7 signals) → Risk Scorer + Blast Radius
    → MITRE Mapper → Remediation Engine → API → Dashboard
```

### Technology Stack
- **Backend**: FastAPI (Python 3.10+), Pydantic v2, SQLAlchemy async, SlowAPI rate limiting
- **ML**: scikit-learn (IsolationForest, OneClassSVM, LocalOutlierFactor, RandomForest, ExtraTrees, KMeans, LogisticRegression), imbalanced-learn (SMOTE)
- **Dashboard**: Streamlit, Plotly, custom JARVIS-style dark-cyber theme
- **Evaluation**: 80/20 stratified holdout, LogReg stacking with rule features + raw features + discriminators
- **Graph analysis**: NetworkX for privilege graph, blast radius reachability
- **Optional**: TensorFlow (LSTM/Transformer sequential detection), DeepSeek API (advisory CISO narratives), Boto3 (AWS deep posture analysis)

---

## Quick Start

### Prerequisites
- Python 3.10 or later
- pip

### Installation

```bash
# Clone and install
git clone <repo-url> && cd accesssentinel
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with any API keys needed (IdP connectors, DeepSeek — all optional)
```

### Launch

```bash
# Terminal 1 — API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Dashboard
streamlit run dashboard/app.py

# Open http://localhost:8501
```

### Docker (alternative)

```bash
docker-compose -f docker/docker-compose.yml up --build
```

### Environment Variables

All variables are optional unless explicitly marked. The platform runs fully offline with sample or uploaded CSV data.

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATABASE_URL` | SQLite connection string | `sqlite+aiosqlite:///./data/accesssentinel.db` |
| `API_HOST`, `API_PORT` | Server bind address | `0.0.0.0` / `8000` |
| `CORS_ORIGINS` | Allowed dashboard origin | `http://localhost:8501` |
| `SCAN_RATE_LIMIT`, `GENERAL_RATE_LIMIT` | API rate limits | `30/min`, `100/min` |
| `RANDOM_SEED` | Reproducibility seed | `42` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `OKTA_ORG_URL`, `OKTA_API_TOKEN` | Okta connector (optional) | — |
| `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` | Azure AD connector (optional) | — |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | AWS IAM connector (optional) | — |
| `ENABLE_LLM_CISO_SUMMARY` | Enable DeepSeek advisory narrative | `0` (disabled) |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `DEEPSEEK_TIMEOUT_SECONDS` | DeepSeek config (optional) | `deepseek-chat` / `30` |

---

## Dashboard Pages

### 1. Scan (Upload & Scan)
- Upload users CSV + events CSV (both required)
- Optional: upload label CSVs for evaluation metrics
- "Load Demo Dataset" — loads bundled `sample_data/` with labels
- Scan pipeline: ingestion → baselines → ensemble → rules → context → scoring → MITRE mapping
- Post-scan summary: Total, CRITICAL, HIGH, MEDIUM, LOW counts, F1 Score

### 2. Risk Leaderboard
- Full identity list sorted by risk score
- Filters: Tier, Department, Source System, Anomaly Type, text search
- Score bars, tier badges, MITRE technique column, suppressed rule counts
- CSV export of full filtered dataset
- Data source indicator showing last scan time and live/cached status

### 3. Identity Drill-Down
- Select any identity for full risk profile
- Deterministic risk narrative (context-aware, generated from triggered rules and context signals)
- Context signals: 7 types with confidence scores and score adjustments
- Rules triggered/suppressed with evidence text
- Behavioral timeline chart (12 months vs baseline band)
- Permission utilization gauge
- MITRE ATT&CK techniques table
- Remediation plan with source-system-specific CLI commands and SLA estimates
- Blast radius assessment

### 4. Role Clusters & Privilege Graph
- **Cluster Scatter**: Identity risk by days since login, colored by KMeans cluster, outliers highlighted
- **Cluster Summary**: Per-cluster profiles with user counts, avg scores, dominant resources
- **Privilege Graph**: NetworkX graph of user → role → system → resource paths
- **Multi-Cloud IdP**: Okta, Azure AD, AWS IAM provider status cards with LIVE/MOCK badges

### 5. Integrations (API Providers)
- Okta, Azure AD (Microsoft Graph), AWS IAM connectors
- LIVE / MOCK / CONFIGURED / NOT CONFIGURED status badges per provider
- **AWS IAM** (deep): STS GetCallerIdentity verification, paginated IAM user inventory with per-user policy inspection, credential report (MFA, access key, password hygiene), derived heuristics (privileged users, MFA gaps, credential sprawl)
- Provider connectors are inventory visibility only — risk scoring requires CSV event data

### 6. Model Evaluation
- "Run Evaluation" triggers 80/20 stratified holdout
- Metrics: Overall F1, Macro F1, Weighted F1, Precision/Recall/F1 per class
- Confusion matrix heatmap, FPR by department bar chart
- Ground-truth label alignment; handles missing-label and label-mismatch scenarios

### 7. CISO Report Export
- Self-contained HTML report with 8 sections:
  1. Executive Summary, 2. Risk Tier Distribution, 3. Top 10 Identities,
  4. MITRE ATT&CK Summary, 5. Remediation Action Plan, 6. Model Evaluation Metrics,
  7. Context Signals Summary, 8. Methodology
- Optional AI-assisted advisory narrative (DeepSeek-powered, advisory-only)
- Download as standalone HTML file

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/scan` | Upload users/events CSVs; accepts optional label files. Returns leaderboard + summary. |
| GET | `/api/v1/identities` | Paginated identity list with tier/department/source/anomaly filtering. Default 50, max 200 per page. |
| GET | `/api/v1/identities/{id}` | Full identity profile with score, narrative, rules, signals, MITRE, remediation. |
| GET | `/api/v1/identities/{id}/blast-radius` | Blast radius simulation for an identity. |
| GET | `/api/v1/access/predict` | AccessPredictor decision (APPROVE/DENY/REVIEW). |
| GET | `/api/v1/clusters` | KMeans role cluster profiles. |
| GET | `/api/v1/graph` | Privilege graph nodes and edges. |
| GET | `/api/v1/evaluate` | Run holdout evaluation (80/20 split, LogReg stacking). |
| GET | `/api/v1/report` | Download CISO HTML report. |
| GET | `/api/v1/health` | Health check with model/baseline load status. |
| POST | `/api/v1/feedback` | Record access decision correction feedback. |
| GET | `/api/v1/feedback/summary` | Feedback correction counts. |
| GET | `/api/v1/idp-summary` | Multi-cloud IdP provider summary with user counts. |
| GET | `/api/v1/integrations/status` | Per-provider connector health (configured, status, error). |
| GET | `/api/v1/integrations/test/{provider}` | Deep live verification for a specific provider. |
| GET | `/api/v1/org-anomalies` | Department-level risk aggregation. |

---

## Detection Pipeline

### Feature Engineering
- **23 user-level features**: days_since_login, permission_utilization, role_count, system_count, is_privileged_flag, has_owner_flag, hr_mismatch, off_hours_pct, geo_anomaly, sso/mfa flags, sensitivity_score, behavior_zscore, account_age, event anomaly/fail rates, unique resources, after_hours_pct, peer deviation, avg_time_between_events
- **13 rule trigger flags**: STALE_PRIVILEGED, ORPHANED_ACCOUNT, OVER_PRIVILEGED, SHADOW_ADMIN, PRIVILEGE_CREEP, SERVICE_ACCT_ABUSE, CREDENTIAL_SPRAWL, IMPOSSIBLE_TRAVEL, CROSS_SYSTEM_TRAVEL, EXCESSIVE_ACCESS, LATERAL_MOVEMENT_SPIKE, BULK_DOWNLOAD, SOD_VIOLATION
- **10 discriminators**: privilege × inactivity interactions, role change × privilege, system count × SSO
- **Event aggregation**: per-user anomaly type distribution from event logs

### Model Architecture (LogReg Stacking)
1. **Base Models**: SMOTE RF (RiskClassifier), balanced RF (class_weight), ExtraTrees, ensemble (IF+SVM+LOF)
2. **Features**: 23 scaled raw + 13 rule flags + 10 discriminators + event aggregation
3. **Meta-Learner**: LogisticRegression (C=0.5, decision threshold=0.22)
4. **Training**: 80/20 stratified holdout, SMOTE for class imbalance (<20% positive)

### Evaluation Metrics
- **User-level**: Overall F1, precision, recall, FPR, confusion matrix on held-out 20%
- **Event-level**: Per-anomaly-type F1 (impossible_travel, unusual_location, privilege_escalation, unusual_resource, excessive_access, unusual_time)
- **FPR by Department**: False positive rate broken down by organizational unit

---

## Live vs Mock Integrations

| Provider | Inventory Visibility | Deep Posture Analysis | Risk Scoring Input |
|----------|---------------------|----------------------|-------------------|
| **AWS IAM** | Live (via Boto3 when credentials configured) | Live — STS, IAM ListUsers, credential report, per-user policy inspection, heuristics | Requires CSV event data |
| **Azure AD** | Live (OAuth2 to Microsoft Graph when credentials configured) | Health check only | Requires CSV event data |
| **Okta** | Live (SSWS REST API when credentials configured) | Health check only | Requires CSV event data |

**Important**: The IdP connectors provide inventory visibility and health status. Risk scoring, anomaly detection, and remediation recommendations require CSV event-log data uploaded via the Scan page or bundled sample data.

---

## Compliance Alignment

AccessSentinel supports identity governance practices aligned with common frameworks. It does **not** provide compliance certification, but surfaces evidence and risks relevant to these controls:

| Framework / Control | How AccessSentinel Supports It |
|---------------------|-------------------------------|
| **NIST AC-2** (Account Management) | Detects stale privileged accounts, orphaned identities, and inactive users. Surfaces excessive access and privilege creep for periodic access review workflows. |
| **NIST AC-6** (Least Privilege) | Identifies over-privileged users, shadow admins, and service account abuse. Generates per-identity remediation plans with specific access reduction recommendations. |
| **GDPR Article 32** (Security of Processing) | Provides risk scoring, anomaly detection, context signal documentation, and auditable CISO reports. Supports evidence-based risk review with deterministic findings. |
| **SOX** (Segregation of Duties) | Enforces SOD conflict rules from `data/sod_conflicts.json`. Flags identities that simultaneously hold incompatible role combinations. |

---

## Optional: DeepSeek AI-Assisted CISO Report Narrative

The CISO HTML report can include an optional AI-generated advisory narrative powered by the DeepSeek API. This narrative is **advisory-only** — it never overrides computed risk scores, tiers, MITRE mappings, or remediation actions. Deterministic findings remain authoritative.

### Enablement

```bash
# In .env:
ENABLE_LLM_CISO_SUMMARY=1
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TIMEOUT_SECONDS=30
```

### Safety Guarantees

- The LLM receives only pre-computed, sanitized report facts (tier counts, top rules, MITRE techniques, remediation actions, evaluation metrics)
- Raw logs, secrets, PII, or system internals are never passed to the model
- Prompt-injection defenses treat all source data as untrusted content
- Output is validated against a strict JSON schema with per-field length caps
- HTML/script content is stripped before embedding in the report
- On any failure (disabled, missing key, timeout, invalid JSON) the report falls back to deterministic content
- The LLM section is clearly labeled "AI-Assisted Executive Summary (Advisory)"

---

## 90-Second Demo Flow

```bash
# 1. Launch
streamlit run dashboard/app.py

# 2. Scan page → Click "Load Demo Dataset"
#    500 identities, 7500 events ingested in 5-stage pipeline

# 3. Leaderboard → Show tier distribution, score bars, MITRE T1078 mapping
#    Filter by CRITICAL tier to highlight top risks

# 4. Drill-Down → Click any CRITICAL identity
#    Show risk narrative, context signals, behavioral timeline, CLI remediation commands

# 5. Role Clusters → Show KMeans scatter plot and privilege graph
#    Highlight "Shadow Admins" cluster and outlier identities

# 6. Evaluation → Run evaluation
#    Show F1 scores, confusion matrix, FPR by department

# 7. CISO Report → Generate and download the self-contained HTML report
```

**Sample demo output** (using bundled sample_data, seeded with random_state=42):
- ~300 CRITICAL, ~75 HIGH, ~69 MEDIUM, ~56 LOW risk identities
- Overall F1 ~0.78, FPR ~0.07
- Top triggered rule: `T1078` (Valid Accounts) across 12 MITRE technique categories

---

## Testing

```bash
# Full test suite with coverage
pytest tests/ --cov=core --cov=api --cov-report=term-missing -v

# LLM narrative tests only
pytest tests/test_llm_narrative.py -v
```

- **174 tests**, 1 skipped
- Coverage: models, ingestion, features, rules engine, context resolver, risk scorer, API routes, connectors, LLM narrative

---

## Known Limitations

| Limitation | Detail |
|-----------|--------|
| **CSV-dependent risk scoring** | The scan pipeline ingests CSV files. IdP connectors provide inventory visibility but do not feed the scoring pipeline directly — event telemetry from providers is needed for end-to-end live scoring. |
| **F1 ceiling** | Current model architecture achieves ~0.78 F1 on sample data. Further gains require additional labeled data or anomaly-type-specific feature engineering. |
| **Sequential detection** | LSTM and Transformer models are defined but trained on-demand; the public API currently returns placeholder results. Sliding-window rule-based detection is active. |
| **Drill-down charts** | The behavioral timeline and permission utilization gauge on the drill-down page use generated demo data. Live API integration for these charts is pending. |
| **AWS access keys** | The core IdP summary layer returns mock access key data even when AWS is configured. The deep posture analysis (Integrations page > Test Connection) provides live data. |
| **Pagination** | API limits 200 identities per page; full dataset export is handled client-side via the CSV export button. |
| **No dark/light mode toggle** | Theme is hardcoded dark. |
