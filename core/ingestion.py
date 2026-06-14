"""Ingestion module for AccessSentinel.

Parses and normalizes identity and event data from CSV files. Produces
IdentityRecord dataclasses with all user fields and event-derived features.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
DAYS_NULL_SENTINEL: float = 999.0
_SAMPLE_DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "sample_data")
_RAW_DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Column name mappings for source system variants
_COLUMN_ALIASES: dict[str, str] = {
    "uid": "user_id",
    "user_name": "username",
    "user_email": "email",
    "dept": "department",
    "emp_status": "employment_status",
    "acct_type": "account_type",
    "mgr_id": "owner_id",
    "source": "source_system",
    "last_auth": "last_login",
    "create_date": "created_at",
    "role_list": "roles",
    "perm_list": "permissions",
    "mfa": "mfa_enabled",
    "sso": "sso_linked",
    "login_count_30": "login_count_30d",
    "login_count_90": "login_count_90d",
    "sys_count": "systems_count",
    "role_changes": "role_changes_90d",
    "privileged": "is_privileged",
    "sensitivity": "resource_sensitivity",
    "off_hours": "off_hours_access_pct",
    "geo_flag": "geo_anomaly",
    "interactive": "interactive_login",
}


class IngestionError(Exception):
    """Raised when ingestion encounters unparsable data with the exact field name."""

    def __init__(self, field_name: str) -> None:
        super().__init__(f"Ingestion failed on field: {field_name}")
        self.field_name = field_name


@dataclass(frozen=True)
class IdentityRecord:
    """Frozen record of an identity with all user and event-derived fields."""

    user_id: str
    username: str
    email: str
    department: str
    employment_status: str
    account_type: str
    owner_id: str
    source_system: str
    job_title: str = ""
    last_login: datetime | None = None
    created_at: datetime | None = None
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    mfa_enabled: bool = False
    sso_linked: bool = False
    login_count_30d: int = 0
    login_count_90d: int = 0
    systems_count: int = 0
    role_changes_90d: int = 0
    is_privileged: bool = False
    resource_sensitivity: str = "low"
    off_hours_access_pct: float = 0.0
    geo_anomaly: bool = False
    interactive_login: bool = False
    # Event-derived fields
    event_count_30d: int = 0
    event_count_90d: int = 0
    unique_resources_accessed: int = 0
    anomaly_event_count: int = 0
    failed_attempt_count: int = 0
    off_hours_event_pct: float = 0.0
    impossible_travel_detected: bool = False
    cross_system_impossible_travel: bool = False
    bulk_download_detected: bool = False
    max_resources_in_single_session: int = 0
    max_resources_in_single_day: int = 0
    avg_time_between_events_hours: float = 0.0


def _resolve_data_path(filename: str) -> str:
    """Return the path to a data file, preferring sample_data/ over data/raw/."""
    primary = os.path.join(_SAMPLE_DATA_DIR, filename)
    fallback = os.path.join(_RAW_DATA_DIR, filename)
    if os.path.exists(primary):
        return primary
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Neither {primary} nor {fallback} exists")


def _normalize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using alias map."""
    rename_map: dict[str, str] = {}
    for col in dataframe.columns:
        normalized = col.strip().lower().replace(" ", "_")
        if normalized in _COLUMN_ALIASES:
            rename_map[col] = _COLUMN_ALIASES[normalized]
        elif normalized in _COLUMN_ALIASES.values():
            rename_map[col] = normalized
        else:
            rename_map[col] = normalized
    return dataframe.rename(columns=rename_map)


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse a timestamp value accepting four formats.

    All returned datetimes are normalized to UTC. Timezone-aware inputs
    are converted; naive inputs are treated as UTC.

    Args:
        value: Raw timestamp string or numeric value.

    Returns:
        timezone-aware datetime in UTC, or None if empty.

    Raises:
        IngestionError: If value is present but unparsable.
    """
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(int(value), tz=timezone.utc)
        except (ValueError, OSError):
            raise IngestionError("last_login") from None
    if not isinstance(value, str):
        raise IngestionError("last_login")
    value = value.strip()
    if value == "":
        return None

    # ISO 8601 (handles "Z", "+HH:MM", "-HH:MM" natively in Python 3.11+)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed
    except (ValueError, AttributeError):
        pass
    # DD/MM/YYYY
    try:
        return datetime.strptime(value, "%d/%m/%Y").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    # MM-DD-YYYY
    try:
        return datetime.strptime(value, "%m-%d-%Y").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    # Unix epoch integer as string
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (ValueError, OSError):
        pass
    # YYYY-MM-DD (date only)
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    raise IngestionError("last_login")


def _parse_boolean(value: Any, default: bool = False) -> bool:
    """Parse a boolean value from various string representations."""
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return bool(value) if value is not None else default
    return value.strip().lower() in ("true", "1", "yes", "t", "y")


def _parse_int(value: Any, default: int = 0) -> int:
    """Safely parse an integer value."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return default
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        return default


def _parse_float(value: Any, default: float = 0.0) -> float:
    """Safely parse a float value."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_list_field(value: Any) -> list[str]:
    """Parse a pipe-separated list field."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    return [v.strip() for v in str(value).split("|") if v.strip()]


def _compute_event_derived_features(
    events_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compute per-user event-derived features from the event log.

    Args:
        events_df: DataFrame with event records.

    Returns:
        Dict mapping user_id to dict of event-derived feature values.
    """
    if events_df is None or events_df.empty:
        return {}

    if "timestamp" in events_df.columns:
        events_df = events_df.copy()
        events_df["_ts"] = pd.to_datetime(events_df["timestamp"], errors="coerce", utc=True)

        # Use maximum event timestamp as reference (not "now"), so time windows
        # capture the last N days of the dataset regardless of when it was generated.
        valid_ts = events_df["_ts"].dropna()
        if not valid_ts.empty:
            now = valid_ts.max()
        else:
            now = pd.Timestamp.now(tz="utc")

        # Event counts by time window
        cutoff_30d = now - pd.Timedelta(days=30)
        cutoff_90d = now - pd.Timedelta(days=90)

        recent_30d = events_df[events_df["_ts"] >= cutoff_30d]
        recent_90d = events_df[events_df["_ts"] >= cutoff_90d]

        count_30d = recent_30d.groupby("user_id").size() if not recent_30d.empty else pd.Series(dtype=int)
        count_90d = recent_90d.groupby("user_id").size() if not recent_90d.empty else pd.Series(dtype=int)

        # Unique resources accessed
        unique_resources = recent_30d.groupby("user_id")["resource"].nunique() if not recent_30d.empty else pd.Series(dtype=int)

        # Failed attempts
        if "success" in events_df.columns:
            failed_mask = ~recent_90d["success"].astype(str).str.lower().isin(("true", "1", "yes"))
            failed_count = recent_90d[failed_mask].groupby("user_id").size() if not recent_90d.empty else pd.Series(dtype=int)
        else:
            failed_count = pd.Series(dtype=int)

        # Off-hours events: use precomputed flag if available, else compute from hour
        if "off_hours" in events_df.columns:
            off_hours_total = recent_90d[recent_90d["off_hours"]].groupby("user_id").size() if not recent_90d.empty else pd.Series(dtype=int)
            total_90 = recent_90d.groupby("user_id").size() if not recent_90d.empty else pd.Series(dtype=float)
            off_hours_pct = (off_hours_total / total_90.replace(0, 1)).fillna(0.0)
        elif not recent_90d.empty:
            off_hours_mask = recent_90d["_ts"].dt.hour.isin([0, 1, 2, 3, 4, 5])
            off_hours_total = recent_90d[off_hours_mask].groupby("user_id").size()
            total_90 = recent_90d.groupby("user_id").size()
            off_hours_pct = (off_hours_total / total_90.replace(0, 1)).fillna(0.0)
        else:
            off_hours_pct = pd.Series(dtype=float)

        # Anomaly event count
        if "is_anomaly" in events_df.columns:
            anomaly_mask = recent_90d["is_anomaly"].astype(str).str.lower().isin(("true", "1", "yes"))
            anomaly_count = recent_90d[anomaly_mask].groupby("user_id").size() if not recent_90d.empty else pd.Series(dtype=int)
        else:
            anomaly_count = pd.Series(dtype=int)

        # Impossible travel (event label based)
        impossible_mask = recent_90d["anomaly_type"].astype(str).str.lower().eq("impossible_travel")
        impossible_flag = impossible_mask.groupby(events_df["user_id"]).any() if not recent_90d.empty else pd.Series(dtype=bool)

        # Cross-system impossible travel (geo-based, different source systems, <3 hours)
        cross_system_flag = pd.Series(dtype=bool)
        from core.geo_utils import REGION_BY_LOCATION
        if not events_df.empty and "location" in events_df.columns and "source_system" in events_df.columns:
            cs_events = events_df.sort_values(["_ts"]).reset_index(drop=True)
            THIRD = pd.Timedelta(hours=3)
            for uid in cs_events["user_id"].unique():
                user_events = cs_events[cs_events["user_id"] == uid]
                if len(user_events) < 2:
                    continue
                ts_arr = user_events["_ts"].values
                loc_arr = user_events["location"].values
                src_arr = user_events["source_system"].values
                for j in range(len(user_events) - 1):
                    if ts_arr[j + 1] - ts_arr[j] > THIRD:
                        continue
                    src_a, src_b = str(src_arr[j]), str(src_arr[j + 1])
                    if src_a == src_b:
                        continue
                    region_a = REGION_BY_LOCATION.get(str(loc_arr[j]), "unknown")
                    region_b = REGION_BY_LOCATION.get(str(loc_arr[j + 1]), "unknown")
                    if region_a == "unknown" or region_b == "unknown":
                        continue
                    if region_a != region_b:
                        cross_system_flag[uid] = True
                        break

        # Bulk download
        if not recent_90d.empty and "action" in events_df.columns:
            download_mask = recent_90d["action"].astype(str).str.lower().eq("download")
            bulk_flag = download_mask.groupby(events_df["user_id"]).sum().gt(100)
        else:
            bulk_flag = pd.Series(dtype=bool)

        # Max resources in single day (calendar-date based)
        max_resources_day = pd.Series(dtype=int)
        if not events_df.empty and "_ts" in events_df.columns:
            events_df["_date"] = events_df["_ts"].dt.date
            resources_per_day = events_df.groupby(["user_id", "_date"])["resource"].nunique()
            max_resources_day = resources_per_day.groupby("user_id").max()

        # Max resources in single session (90d window approximation)
        max_resources = pd.Series(dtype=int)
        if not recent_90d.empty:
            recent_90d["_date"] = recent_90d["_ts"].dt.date
            resources_per_day = recent_90d.groupby(["user_id", "_date"])["resource"].nunique()
            max_resources = resources_per_day.groupby("user_id").max()

        # Average time between events
        avg_time = pd.Series(dtype=float)
        if not recent_90d.empty:
            sorted_events = recent_90d.sort_values(["_ts"])
            for uid in sorted_events["user_id"].unique():
                user_ts = sorted_events[sorted_events["user_id"] == uid]["_ts"].dropna()
                if len(user_ts) > 1:
                    diffs = user_ts.diff().dropna()
                    avg_time[uid] = diffs.dt.total_seconds().mean() / 3600.0
                else:
                    avg_time[uid] = 0.0
            avg_time = avg_time.fillna(0.0)
    else:
        count_30d = pd.Series(dtype=int)
        count_90d = pd.Series(dtype=int)
        unique_resources = pd.Series(dtype=int)
        failed_count = pd.Series(dtype=int)
        off_hours_pct = pd.Series(dtype=float)
        anomaly_count = pd.Series(dtype=int)
        impossible_flag = pd.Series(dtype=bool)
        bulk_flag = pd.Series(dtype=bool)
        max_resources = pd.Series(dtype=int)
        max_resources_day = pd.Series(dtype=int)
        avg_time = pd.Series(dtype=float)

    result: dict[str, dict[str, Any]] = {}
    all_user_ids = events_df["user_id"].unique()
    for uid in all_user_ids:
        result[uid] = {
            "event_count_30d": int(count_30d.get(uid, 0)),
            "event_count_90d": int(count_90d.get(uid, 0)),
            "unique_resources_accessed": int(unique_resources.get(uid, 0)),
            "anomaly_event_count": int(anomaly_count.get(uid, 0)),
            "failed_attempt_count": int(failed_count.get(uid, 0)),
            "off_hours_event_pct": float(off_hours_pct.get(uid, 0.0)),
            "impossible_travel_detected": bool(impossible_flag.get(uid, False)),
            "cross_system_impossible_travel": bool(cross_system_flag.get(uid, False) if len(cross_system_flag) > 0 else False),
            "bulk_download_detected": bool(bulk_flag.get(uid, False)),
            "max_resources_in_single_session": int(max_resources.get(uid, 0)),
            "max_resources_in_single_day": int(max_resources_day.get(uid, 0)),
            "avg_time_between_events_hours": float(avg_time.get(uid, 0.0)),
        }
    return result


def _adapt_competition_users(users_df: pd.DataFrame) -> pd.DataFrame:
    """Map competition-schema user columns to the canonical IdentityRecord fields.

    Competition schema (11 cols): user_id, username, email, department, job_title,
    privilege_level, systems_access, last_login, days_inactive, is_active, hire_date

    Maps to original schema by deriving/computing missing fields with documented defaults.
    """
    df = users_df.copy()
    # privilege_level -> is_privileged + account_type
    df["is_privileged"] = df.get("privilege_level", "").astype(str).str.lower().eq("admin")
    acct_map = {"service-account": "service", "admin": "human", "power-user": "human", "user": "human"}
    df["account_type"] = df.get("privilege_level", "user").astype(str).str.lower().map(acct_map).fillna("human")
    # employment_status from is_active
    df["employment_status"] = df.get("is_active", True).apply(lambda x: "active" if x else "terminated")
    # systems_count from systems_access
    df["systems_count"] = df.get("systems_access", "").astype(str).apply(lambda s: len(s.split("|")) if s else 0)
    # source_system from first entry in systems_access
    df["source_system"] = df.get("systems_access", "").astype(str).apply(lambda s: s.split("|")[0] if s else "")
    # created_at from hire_date
    df["created_at"] = df.get("hire_date", "")
    # Unmappable fields — documented defaults (conservative, risk-increasing)
    df["owner_id"] = ""
    df["roles"] = ""
    df["permissions"] = ""
    df["mfa_enabled"] = "false"
    df["sso_linked"] = "false"
    df["login_count_30d"] = 0
    df["login_count_90d"] = 0
    df["role_changes_90d"] = 0
    df["resource_sensitivity"] = "low"
    df["off_hours_access_pct"] = 0.0
    df["geo_anomaly"] = "false"
    df["interactive_login"] = "false"
    return df


def _adapt_competition_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Map competition-schema event columns to the canonical event fields.

    Competition schema (9 cols): timestamp, user_id, username, action, resource,
    resource_sensitivity, status, source_ip, time_classification

    Maps to original schema by deriving missing fields.
    """
    df = events_df.copy()
    # Join department + job_title from users (handled later in ingest loop)
    # success from status
    df["success"] = df.get("status", "success").astype(str).str.lower().eq("success")
    # off_hours from time_classification
    off_hours_set = {"unusual_hours", "night", "weekend"}
    df["off_hours"] = df.get("time_classification", "business_hours").astype(str).str.lower().isin(off_hours_set)
    # No location, no is_anomaly, no anomaly_type — defaults
    df["location"] = ""
    df["is_anomaly"] = "false"
    df["anomaly_type"] = ""
    # No department/job_title — will be joined from users
    if "department" not in df.columns:
        df["department"] = ""
    if "job_title" not in df.columns:
        df["job_title"] = ""
    if "event_id" not in df.columns:
        df["event_id"] = [f"EVT{i:06d}" for i in range(len(df))]
    return df


def ingest(
    users_path: str | None = None,
    events_path: str | None = None,
) -> list[IdentityRecord]:
    """Ingest and join user and event data into IdentityRecord list.

    Args:
        users_path: Path to users CSV. Defaults to sample_data/ or data/raw/.
        events_path: Path to events CSV. Defaults to sample_data/ or data/raw/.

    Returns:
        List of IdentityRecord instances.

    Raises:
        IngestionError: If required columns are missing or parse failures occur.
        FileNotFoundError: If neither primary nor fallback paths exist.
    """
    if users_path is None:
        users_path = _resolve_data_path("identity_users.csv")
    if events_path is None:
        events_path = _resolve_data_path("identity_events.csv")

    users_df = pd.read_csv(users_path)
    events_df = pd.read_csv(events_path)

    if users_df.empty:
        raise IngestionError("empty dataset")

    users_df = _normalize_column_names(users_df)
    events_df = _normalize_column_names(events_df)

    # Detect schema variant: competition (simpler 11-col users, 9-col events with
    # privilege_level and time_classification) vs original (rich 23-col users,
    # 13-col events with roles/permissions/is_privileged/employment_status).
    # Require BOTH competition-unique columns AND absence of original-unique columns.
    _IS_COMPETITION = (
        "privilege_level" in users_df.columns
        and "time_classification" in events_df.columns
        and "is_privileged" not in users_df.columns
        and "employment_status" not in users_df.columns
    )

    if _IS_COMPETITION:
        users_df = _adapt_competition_users(users_df)
        events_df = _adapt_competition_events(events_df)
        # Join department and job_title from users into events (events lack these)
        if "department" not in events_df.columns or events_df["department"].isna().all():
            user_dep_map = dict(zip(users_df["user_id"].astype(str), users_df["department"].astype(str)))
            user_jt_map = dict(zip(users_df["user_id"].astype(str), users_df["job_title"].astype(str)))
            events_df["department"] = events_df["user_id"].astype(str).map(user_dep_map).fillna("")
            events_df["job_title"] = events_df["user_id"].astype(str).map(user_jt_map).fillna("")

    # Validate required columns
    required_user_cols = {"user_id", "username"}
    missing = required_user_cols - set(users_df.columns)
    if missing:
        raise IngestionError(list(missing)[0])

    required_event_cols = {"user_id"} if _IS_COMPETITION else {"event_id", "user_id"}
    missing_event = required_event_cols - set(events_df.columns)
    if missing_event:
        raise IngestionError(list(missing_event)[0])

    # Compute event-derived features
    event_features = _compute_event_derived_features(events_df)

    # Compute per-user job_title (most common from events)
    job_title_map: dict[str, str] = {}
    if "job_title" in events_df.columns:
        jt_grouped = (
            events_df.groupby("user_id")["job_title"]
            .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else "")
        )
        for uid, jt in jt_grouped.items():
            job_title_map[str(uid)] = str(jt) if jt else ""

    # Build IdentityRecord list
    records: list[IdentityRecord] = []
    for _, row in users_df.iterrows():
        uid = str(row.get("user_id", ""))
        if not uid:
            raise IngestionError("user_id")

        last_login = None
        try:
            last_login = _parse_timestamp(row.get("last_login"))
        except IngestionError as e:
            if e.field_name == "last_login":
                last_login = None

        created_at = None
        try:
            created_at = _parse_timestamp(row.get("created_at", row.get("last_login")))
        except IngestionError:
            created_at = None

        features = event_features.get(uid, {})

        record = IdentityRecord(
            user_id=uid,
            username=str(row.get("username", "")),
            email=str(row.get("email", "")),
            department=str(row.get("department", "")),
            employment_status=str(row.get("employment_status", "active")),
            account_type=str(row.get("account_type", "human")),
            owner_id=str(row.get("owner_id", "")) if str(row.get("owner_id", "")) not in ("", "nan", "None") else "",
            source_system=str(row.get("source_system", "")),
            job_title=job_title_map.get(uid, str(row.get("job_title", ""))),
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
            event_count_30d=features.get("event_count_30d", 0),
            event_count_90d=features.get("event_count_90d", 0),
            unique_resources_accessed=features.get("unique_resources_accessed", 0),
            anomaly_event_count=features.get("anomaly_event_count", 0),
            failed_attempt_count=features.get("failed_attempt_count", 0),
            off_hours_event_pct=features.get("off_hours_event_pct", 0.0),
            impossible_travel_detected=features.get("impossible_travel_detected", False),
            cross_system_impossible_travel=features.get("cross_system_impossible_travel", False),
            bulk_download_detected=features.get("bulk_download_detected", False),
            max_resources_in_single_session=features.get("max_resources_in_single_session", 0),
            max_resources_in_single_day=features.get("max_resources_in_single_day", 0),
            avg_time_between_events_hours=features.get("avg_time_between_events_hours", 0.0),
        )
        records.append(record)

    return records
