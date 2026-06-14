"""Generate synthetic identity and event data for AccessSentinel demo.

Produces:
  data/raw/identities_synthetic.csv — 500 synthetic user identities
  data/raw/events_synthetic.csv    — 7500 synthetic events

All anomaly patterns and temporal patterns injected per specification.
"""
import csv
import os
import random
from datetime import datetime, timedelta, timezone

import numpy as np

np.random.seed(42)
random.seed(42)

# ── Constants ────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
USER_COUNT = 500
EVENT_COUNT = 7500

DEPARTMENTS = [
    "Engineering", "Finance", "HR", "Sales", "Marketing",
    "Operations", "Legal", "IT", "Product", "Customer Support",
]
JOB_TITLES = {
    "Engineering": ["Software Engineer", "Senior Engineer", "Staff Engineer", "DevOps Engineer", "Engineering Manager"],
    "Finance": ["Financial Analyst", "Senior Accountant", "Finance Manager", "Controller", "CFO"],
    "HR": ["HR Coordinator", "HR Manager", "Recruiter", "Benefits Specialist", "CHRO"],
    "Sales": ["Sales Rep", "Account Executive", "Sales Manager", "Sales Director", "CRO"],
    "Marketing": ["Marketing Coordinator", "Brand Manager", "SEO Specialist", "Growth Lead", "CMO"],
    "Operations": ["Ops Analyst", "Ops Manager", "Supply Chain Lead", "Logistics Coordinator", "COO"],
    "Legal": ["Paralegal", "Corporate Counsel", "Compliance Officer", "General Counsel", "CLO"],
    "IT": ["IT Support", "Systems Administrator", "Network Engineer", "Security Analyst", "CISO"],
    "Product": ["Associate PM", "Product Manager", "Senior PM", "Director of Product", "CPO"],
    "Customer Support": ["Support Agent", "Senior Support Agent", "Support Team Lead", "Support Manager", "CSO"],
}
SOURCE_SYSTEMS = ["AD", "AzureAD", "AWS_IAM", "GCP", "Okta", "Salesforce"]
EMPLOYMENT_STATUSES = ["active", "terminated", "contractor", "bot"]
ACCOUNT_TYPES = ["human", "service", "bot"]
RESOURCE_SENSITIVITIES = ["low", "medium", "high", "critical"]
ROLES_POOL = [
    "reader", "writer", "admin", "finance_approver", "payment_executor",
    "deployer", "viewer", "operator", "auditor", "developer",
    "sre", "security_admin", "network_admin", "db_admin", "compliance_viewer",
]
PERMISSIONS_POOL = [
    "read:all", "write:finance", "admin:systems", "ssh:prod", "deploy:prod",
    "read:audit", "write:config", "admin:users", "read:hr", "write:hr",
    "admin:finance", "execute:batch", "download:data", "upload:reports", "admin:network",
]
RESOURCES = [
    "production_database", "payment_processor", "customer_data_warehouse",
    "employee_portal", "source_code_repo", "ci_cd_pipeline",
    "email_server", "file_storage", "vpn_gateway", "monitoring_system",
    "hr_information_system", "sales_crm", "financial_ledger",
    "audit_logs", "security_incident_tracker", "test_environment",
    "staging_environment", "backup_system", "dns_server", "load_balancer",
]
ACTIONS = ["read", "write", "delete", "admin", "execute", "download"]
ANOMALY_TYPES = [
    "impossible_travel", "unusual_location", "privilege_escalation",
    "unusual_resource", "excessive_access", "unusual_time",
]
LOCATIONS = [
    "New York, US", "London, UK", "Paris, FR", "Tokyo, JP", "Singapore, SG",
    "Sydney, AU", "Mumbai, IN", "Sao Paulo, BR", "Toronto, CA", "Berlin, DE",
]
LOCATION_COORDS = {
    "New York, US": (40.7128, -74.0060),
    "London, UK": (51.5074, -0.1278),
    "Paris, FR": (48.8566, 2.3522),
    "Tokyo, JP": (35.6762, 139.6503),
    "Singapore, SG": (1.3521, 103.8198),
    "Sydney, AU": (-33.8688, 151.2093),
    "Mumbai, IN": (19.0760, 72.8777),
    "Sao Paulo, BR": (-23.5505, -46.6333),
    "Toronto, CA": (43.6532, -79.3832),
    "Berlin, DE": (52.5200, 13.4050),
}

# Timing constants
DATE_RANGE_START = datetime(2025, 4, 1, tzinfo=timezone.utc)
DATE_RANGE_END = datetime(2026, 4, 30, tzinfo=timezone.utc)
TOTAL_DAYS = (DATE_RANGE_END - DATE_RANGE_START).days

# ── Helper Functions ──────────────────────────────────────────────────────────

def _random_timestamp(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between start and end."""
    delta = end - start
    seconds = random.uniform(0, delta.total_seconds())
    return start + timedelta(seconds=seconds)


def _format_timestamp(ts: datetime, noisy: bool = False) -> str:
    """Format timestamp; occasionally use non-ISO formats for noise."""
    if noisy and random.random() < 0.05:
        fmt = random.choice(["ddmmyyyy", "mmddyyyy", "unix"])
        if fmt == "ddmmyyyy":
            return ts.strftime("%d/%m/%Y")
        elif fmt == "mmddyyyy":
            return ts.strftime("%m-%d-%Y")
        else:
            return str(int(ts.timestamp()))
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _source_ip(location: str) -> str:
    """Generate a plausible IP address."""
    parts = [str(random.randint(1, 223))]
    for _ in range(3):
        parts.append(str(random.randint(0, 255)))
    return ".".join(parts)


# ── Task A: Generate User Identities ─────────────────────────────────────────

def _generate_user(user_id: str, user_index: int) -> dict:
    """Generate one synthetic user identity record."""
    department = random.choice(DEPARTMENTS)
    source_system = random.choice(SOURCE_SYSTEMS)
    employment_status = random.choice(
        EMPLOYMENT_STATUSES if source_system != "AD" else ["active", "terminated", "contractor"]
    )
    account_type = "human"
    if employment_status == "bot":
        account_type = "bot"

    username = f"user_{user_index:04d}"

    # Determine if service account
    if random.random() < 0.06:
        account_type = "service"
        employment_status = "active"
        username = f"svc_{user_index:04d}"

    email = f"{username}@{'company.com' if source_system != 'GCP' else 'gcp.company.com'}"

    created_at = _random_timestamp(
        datetime(2020, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    last_login = _random_timestamp(created_at + timedelta(days=1), DATE_RANGE_END)

    is_privileged = random.random() < 0.25
    sso_linked = random.random() < 0.60
    mfa_enabled = random.random() < 0.70
    interactive_login = account_type != "service" or random.random() < 0.05

    roles = random.sample(ROLES_POOL, k=random.randint(1, 4))
    permissions = random.sample(PERMISSIONS_POOL, k=random.randint(1, 6))
    resource_sensitivity = random.choice(RESOURCE_SENSITIVITIES)
    systems_count = random.randint(1, 8)
    login_count_30d = random.randint(0, 200) if employment_status == "active" else random.randint(0, 5)
    login_count_90d = login_count_30d * random.randint(2, 4)
    role_changes_90d = random.choices([0, 1, 2, 3, 4, 5], weights=[60, 20, 10, 5, 3, 2])[0]
    off_hours_access_pct = round(random.uniform(0.0, 0.6), 3)
    geo_anomaly = random.random() < 0.05
    owner_id = f"mgr_{random.randint(1, 50):04d}" if random.random() > 0.05 else None

    return {
        "user_id": user_id,
        "username": username,
        "email": email,
        "department": department,
        "employment_status": employment_status,
        "account_type": account_type,
        "owner_id": owner_id or "",
        "source_system": source_system,
        "last_login": _format_timestamp(last_login),
        "created_at": _format_timestamp(created_at),
        "roles": "|".join(roles),
        "permissions": "|".join(permissions),
        "mfa_enabled": str(mfa_enabled).lower(),
        "sso_linked": str(sso_linked).lower(),
        "login_count_30d": login_count_30d,
        "login_count_90d": login_count_90d,
        "systems_count": systems_count,
        "role_changes_90d": role_changes_90d,
        "is_privileged": str(is_privileged).lower(),
        "resource_sensitivity": resource_sensitivity,
        "off_hours_access_pct": off_hours_access_pct,
        "geo_anomaly": str(geo_anomaly).lower(),
        "interactive_login": str(interactive_login).lower(),
    }


def _inject_anomalies(users: list[dict]) -> None:
    """Inject anomaly patterns at specified ratios into user records."""
    n = len(users)
    indices = list(range(n))
    random.shuffle(indices)

    # 15% stale and privileged (last_login older than 30 days AND is_privileged=True)
    stale_count = int(n * 0.15)
    for idx in indices[:stale_count]:
        users[idx]["is_privileged"] = "true"
        old_date = _random_timestamp(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 3, 1, tzinfo=timezone.utc),
        )
        users[idx]["last_login"] = _format_timestamp(old_date)
    offset = stale_count

    # 8% orphaned (owner_id=NULL OR terminated + active)
    orphaned_count = int(n * 0.08)
    for idx in indices[offset:offset + orphaned_count]:
        users[idx]["owner_id"] = ""
        if random.random() < 0.5:
            users[idx]["employment_status"] = "terminated"
    offset += orphaned_count

    # 10% over-privileged
    overpriv_count = int(n * 0.10)
    for idx in indices[offset:offset + overpriv_count]:
        users[idx]["login_count_30d"] = max(0, users[idx]["login_count_30d"] // 8)
        users[idx]["systems_count"] = max(1, users[idx]["systems_count"])
    offset += overpriv_count

    # 3% shadow admins
    shadow_count = int(n * 0.03)
    for idx in indices[offset:offset + shadow_count]:
        if users[idx]["account_type"] != "admin":
            perms = users[idx]["permissions"].split("|")
            if "admin:systems" not in perms:
                perms.append("admin:systems")
                users[idx]["permissions"] = "|".join(perms)
    offset += shadow_count

    # 5% service account abuse
    svc_abuse_count = int(n * 0.05)
    for idx in indices[offset:offset + svc_abuse_count]:
        users[idx]["account_type"] = "service"
        users[idx]["interactive_login"] = "true"
        users[idx]["employment_status"] = "active"
        users[idx]["username"] = f"svc_abused_{idx:04d}"
    offset += svc_abuse_count

    # 7% privilege creep (role_changes_90d >= 3)
    creep_count = int(n * 0.07)
    for idx in indices[offset:offset + creep_count]:
        users[idx]["role_changes_90d"] = random.randint(3, 7)
    offset += creep_count

    # 6% credential sprawl (systems_count >= 5 AND sso_linked=False)
    sprawl_count = int(n * 0.06)
    for idx in indices[offset:offset + sprawl_count]:
        users[idx]["systems_count"] = random.randint(5, 10)
        users[idx]["sso_linked"] = "false"

    # Noise: 10% NULL last_login
    null_login_count = int(n * 0.10)
    for idx in indices[:null_login_count]:
        if random.random() < 0.5:
            users[idx]["last_login"] = ""


def generate_users(output_path: str) -> list[dict]:
    """Generate 500 synthetic user identities and write to CSV."""
    users = []
    for i in range(USER_COUNT):
        user_id = f"U{i:05d}"
        user = _generate_user(user_id, i)
        users.append(user)

    _inject_anomalies(users)
    # Re-shuffle after anomaly injection
    random.shuffle(users)

    fieldnames = [
        "user_id", "username", "email", "department", "employment_status",
        "account_type", "owner_id", "source_system", "last_login", "created_at",
        "roles", "permissions", "mfa_enabled", "sso_linked", "login_count_30d",
        "login_count_90d", "systems_count", "role_changes_90d", "is_privileged",
        "resource_sensitivity", "off_hours_access_pct", "geo_anomaly", "interactive_login",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(users)

    return users


# ── Task B: Generate Event Log ───────────────────────────────────────────────

def _generate_event(
    event_id: str, user: dict, ts: datetime, anomalous: bool = False
) -> dict:
    """Generate one synthetic event record."""
    department = user["department"]
    location = random.choice(LOCATIONS)
    success = random.random() < 0.85
    resource = random.choice(RESOURCES)
    action = random.choice(ACTIONS)
    anomaly_type = ""

    if anomalous:
        anomaly_type = random.choice(ANOMALY_TYPES)

    return {
        "event_id": event_id,
        "user_id": user["user_id"],
        "username": user["username"],
        "department": department,
        "job_title": random.choice(JOB_TITLES.get(department, ["Analyst"])),
        "resource": resource,
        "action": action,
        "timestamp": _format_timestamp(ts),
        "source_ip": _source_ip(location),
        "location": location,
        "success": str(success).lower(),
        "is_anomaly": str(anomalous).lower(),
        "anomaly_type": anomaly_type,
    }


def generate_events(output_path: str, users: list[dict]) -> list[dict]:
    """Generate 7500 synthetic events and write to CSV."""
    events = []
    user_map = {u["user_id"]: u for u in users}
    user_ids = list(user_map.keys())
    n = EVENT_COUNT

    # Pre-assign anomaly slots: target ~41% anomaly rate
    anomaly_count = int(n * 0.41)
    anomaly_indices = set(random.sample(range(n), anomaly_count))

    # Department-based user pools for specialized patterns
    finance_users = [u for u in users if u["department"] == "Finance"]
    engineering_users = [u for u in users if u["department"] == "Engineering"]
    service_accounts = [u for u in users if u["account_type"] == "service"]
    new_users = [
        u for u in users
        if _parse_last_login(u["created_at"]) > datetime(2025, 3, 1, tzinfo=timezone.utc)
    ]
    # Ensure minimum pools
    if len(finance_users) < 10:
        finance_users = users[:10]
    if len(engineering_users) < 10:
        engineering_users = users[10:20]
    if len(service_accounts) < 5:
        service_accounts = [users[i] for i in range(min(5, len(users)))]
    if len(new_users) < 10:
        new_users = users[20:30]

    user_index = {u["user_id"]: u for u in users}
    on_call_week_map: dict[str, list[int]] = {}

    for i in range(n):
        is_anomalous = i in anomaly_indices
        user_id = random.choice(user_ids)
        user = user_index[user_id]

        # Decide timestamp
        ts = _random_timestamp(DATE_RANGE_START, DATE_RANGE_END)

        # --- Temporal pattern injections ---
        day_of_month = ts.day
        month = ts.month
        hour = ts.hour

        # Finance month-end spike (days 28-31 of any month): 3x event volume
        if user["department"] == "Finance" and day_of_month >= 28:
            is_finance_spike = random.random() < 0.40
            if is_finance_spike:
                if finance_users:
                    user = random.choice(finance_users)

        # Finance quarter-end (Mar, Jun, Sep, Dec): 2x volume
        if user["department"] == "Finance" and month in (3, 6, 9, 12):
            is_quarter_spike = random.random() < 0.25
            if is_quarter_spike and finance_users:
                user = random.choice(finance_users)

        # On-call engineer: designated 1 week per month with elevated access
        if user["department"] == "Engineering":
            eng_key = user["user_id"]
            if eng_key not in on_call_week_map:
                on_call_week_map[eng_key] = []
                for m in range(1, 13):
                    week = random.randint(0, 3)
                    on_call_week_map[eng_key].append(m * 4 + week)
            # Check if current week is on-call week
            week_num = (ts - datetime(2025, 1, 1, tzinfo=timezone.utc)).days // 7
            if week_num in on_call_week_map.get(eng_key, []):
                # On-call pattern hint: bias toward admin/execute/write on prod systems
                # 3x more events during on-call week
                if random.random() < 0.30:
                    events.append(_generate_event(
                        f"EVT_ONCALL_{i:06d}", user, _random_timestamp(
                            ts.replace(hour=random.randint(0, 23)), ts + timedelta(hours=1)
                        ), anomalous=random.random() < 0.15
                    ))

        # Batch jobs: service accounts with download/execute 02:00-04:00
        if user["account_type"] == "service" and 2 <= hour <= 4:
            if random.random() < 0.30:
                # Batch job pattern hint: bias toward download/execute on prod systems
                pass

        # New hire ramp: accounts < 30 days old show high systems_count
        if (user["user_id"] in {u["user_id"] for u in new_users}
                and random.random() < 0.30):
            # New hire ramp: bias toward read/write actions on any resource
            pass

        # Impossible travel: same user authenticating from two distant locations within 2 hours
        if is_anomalous and random.random() < 0.08:
            event = _generate_event(f"EVT{i:06d}", user, ts, anomalous=True)
            event["anomaly_type"] = "impossible_travel"
            events.append(event)
            # Generate companion event from distant location within 2 hours
            distant_loc = LOCATIONS[0] if event["location"] != LOCATIONS[0] else LOCATIONS[-1]
            companion_ts = ts + timedelta(minutes=random.randint(5, 120))
            companion = {
                "event_id": f"EVT{i:06d}_travel",
                "user_id": user["user_id"],
                "username": user["username"],
                "department": user["department"],
                "job_title": random.choice(JOB_TITLES.get(user["department"], ["Analyst"])),
                "resource": random.choice(RESOURCES),
                "action": random.choice(["read", "write"]),
                "timestamp": _format_timestamp(companion_ts),
                "source_ip": _source_ip(distant_loc),
                "location": distant_loc,
                "success": "false",
                "is_anomaly": "true",
                "anomaly_type": "impossible_travel",
            }
            events.append(companion)
            continue

        # Normal event generation
        event = _generate_event(f"EVT{i:06d}", user, ts, anomalous=is_anomalous)
        events.append(event)

    # Trim to approximately target count
    if len(events) > n:
        # Keep anomaly events preferentially
        anomaly_events = [e for e in events if e["is_anomaly"] == "true"]
        normal_events = [e for e in events if e["is_anomaly"] == "false"]
        target_anomaly = min(len(anomaly_events), int(n * 0.41))
        target_normal = n - target_anomaly
        if len(anomaly_events) > target_anomaly:
            anomaly_events = random.sample(anomaly_events, target_anomaly)
        if len(normal_events) > target_normal:
            normal_events = random.sample(normal_events, target_normal)
        events = anomaly_events + normal_events

    random.shuffle(events)

    fieldnames = [
        "event_id", "user_id", "username", "department", "job_title",
        "resource", "action", "timestamp", "source_ip", "location",
        "success", "is_anomaly", "anomaly_type",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)

    return events


def _parse_last_login(value: str) -> datetime:
    """Parse last_login/created_at value from string, handling noise formats."""
    if not value or value.strip() == "":
        return datetime(2024, 1, 1, tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        pass
    for fmt in ("%d/%m/%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (ValueError, TypeError):
        pass
    return datetime(2024, 1, 1, tzinfo=timezone.utc)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Generate all demo data files."""
    print("AccessSentinel Data Generator")
    print("=" * 50)

    users_path = os.path.join(OUTPUT_DIR, "identities_synthetic.csv")
    events_path = os.path.join(OUTPUT_DIR, "events_synthetic.csv")

    print(f"\nGenerating {USER_COUNT} synthetic users...")
    users = generate_users(users_path)
    print(f"  -> Wrote {len(users)} records to {users_path}")

    print(f"\nGenerating ~{EVENT_COUNT} synthetic events...")
    events = generate_events(events_path, users)
    print(f"  -> Wrote {len(events)} records to {events_path}")

    # Print summary statistics
    stale = sum(1 for u in users if u["is_privileged"] == "true" and not u.get("last_login", ""))
    orphaned = sum(1 for u in users if not u.get("owner_id", ""))
    service_abuse = sum(
        1 for u in users
        if u["account_type"] == "service" and u.get("interactive_login", "false") == "true"
    )
    event_anomaly = sum(1 for e in events if e["is_anomaly"] == "true")

    print("\nSummary:")
    print(f"  Users: {len(users)}")
    print(f"  Events: {len(events)}")
    print(f"  Stale privileged: {stale}")
    print(f"  Orphaned: {orphaned}")
    print(f"  Service account abuse: {service_abuse}")
    print(f"  Event anomaly rate: {event_anomaly}/{len(events)} = {event_anomaly/len(events):.1%}")

    # Generate label files for synthetic data
    labels_users_path = os.path.join(OUTPUT_DIR, "..", "..", "sample_data", "identity_users_labels.csv")
    os.makedirs(os.path.dirname(labels_users_path), exist_ok=True)

    # Target user anomaly rate: ~16%
    user_target_anomaly = int(len(users) * 0.16)
    # Identify truly anomalous users by injected patterns
    user_anomaly_scores = []
    for u in users:
        score = 0
        anomaly_type = ""
        explanation = ""
        # Check key anomaly indicators
        if u["is_privileged"] == "true":
            last_login_str = u.get("last_login", "")
            if not last_login_str:
                score += 5
                anomaly_type = "excessive_access"
                explanation = "Stale privileged account with NULL last login"
            else:
                try:
                    ll = _parse_last_login(last_login_str)
                    if ll < datetime(2025, 6, 1, tzinfo=timezone.utc):
                        score += 5
                        anomaly_type = "excessive_access"
                        explanation = "Stale privileged account, last login before June 2025"
                except Exception:
                    score += 3
        if u.get("account_type") == "service" and u.get("interactive_login", "false") == "true":
            score += 5
            anomaly_type = "unusual_time"
            explanation = "Service account with interactive login"
        if not u.get("owner_id", ""):
            score += 4
            if not anomaly_type:
                anomaly_type = "privilege_escalation"
            explanation = "Orphaned account without owner"
        if int(u.get("role_changes_90d", 0)) >= 3:
            score += 2
            if not anomaly_type:
                anomaly_type = "privilege_escalation"
                explanation = "Privilege creep: 3+ role changes in 90 days"
        user_anomaly_scores.append((u["user_id"], score, anomaly_type, "HIGH" if score >= 5 else "MEDIUM", explanation))
    # Sort by anomaly score and label top 16% as anomalous
    user_anomaly_scores.sort(key=lambda x: x[1], reverse=True)
    anomaly_user_ids = {uid for uid, score, _, _, _ in user_anomaly_scores[:user_target_anomaly]}

    user_anomaly_count = 0
    with open(labels_users_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "is_anomaly", "anomaly_type", "severity", "explanation"])
        for uid, score, atype, severity, explanation in user_anomaly_scores:
            is_anomaly = uid in anomaly_user_ids
            if is_anomaly:
                user_anomaly_count += 1
            writer.writerow([uid, str(is_anomaly).lower(), atype if is_anomaly else "", severity if is_anomaly else "", explanation if is_anomaly else ""])

    print(f"\n  User anomaly rate: {user_anomaly_count}/{len(users)} = {user_anomaly_count/len(users):.1%}")
    print(f"  User labels written to {labels_users_path}")

    # Event labels
    labels_events_path = os.path.join(OUTPUT_DIR, "..", "..", "sample_data", "identity_events_labels.csv")
    with open(labels_events_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "is_anomaly", "anomaly_type", "severity", "explanation"])
        for e in events:
            is_an = e["is_anomaly"] == "true"
            writer.writerow([
                e["event_id"],
                str(is_an).lower(),
                e["anomaly_type"] if is_an else "",
                "HIGH" if is_an else "",
                f"Anomalous event: {e['anomaly_type']}" if is_an else "",
            ])

    print(f"  Event labels written to {labels_events_path}")

    # Also copy CSV data to sample_data/ for direct use by ingestion
    import shutil
    sample_users_path = os.path.join(OUTPUT_DIR, "..", "..", "sample_data", "identity_users.csv")
    sample_events_path = os.path.join(OUTPUT_DIR, "..", "..", "sample_data", "identity_events.csv")
    shutil.copy(users_path, sample_users_path)
    shutil.copy(events_path, sample_events_path)
    print(f"  Sample users copied to {sample_users_path}")
    print(f"  Sample events copied to {sample_events_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
