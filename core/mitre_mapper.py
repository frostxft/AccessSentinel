"""MITRE ATT&CK mapping module for AccessSentinel.

Maps triggered detection rules to MITRE ATT&CK techniques for security
posture reporting and threat intelligence integration.
"""

import json
import os
from dataclasses import dataclass

# ── Constants ────────────────────────────────────────────────────────────────

_DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data")

RULE_TO_MITRE: dict[str, str] = {
    "STALE_PRIVILEGED": "T1078",
    "ORPHANED_ACCOUNT": "T1136",
    "OVER_PRIVILEGED": "T1098",
    "SHADOW_ADMIN": "T1134",
    "PRIVILEGE_CREEP": "T1098.001",
    "SERVICE_ACCT_ABUSE": "T1078.003",
    "CREDENTIAL_SPRAWL": "T1078.004",
    "IMPOSSIBLE_TRAVEL": "T1078",
    "BULK_DOWNLOAD": "T1048",
    "SOD_VIOLATION": "T1134",
    "EXCESSIVE_ACCESS": "T1021",
    "LATERAL_MOVEMENT_SPIKE": "T1021",
}

_EXPECTED_MITRE_KEYS: frozenset[str] = frozenset({
    "T1078",
    "T1136",
    "T1098",
    "T1134",
    "T1098.001",
    "T1078.003",
    "T1078.004",
    "T1110",
    "T1021",
    "T1048",
})


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MitreTechnique:
    """MITRE ATT&CK technique linked to a triggered detection rule.

    Attributes:
        technique_id: The MITRE ATT&CK technique identifier (e.g. "T1078").
        name: Human-readable technique name.
        tactic: Parent MITRE ATT&CK tactic.
        url: Link to the technique page.
        triggered_by_rule: The AccessSentinel rule that triggered this
            technique.  Empty string when returned by
            :func:`get_all_mitre_techniques`.
    """

    technique_id: str
    name: str
    tactic: str
    url: str
    triggered_by_rule: str


# ── Module-level JSON loading with validation ────────────────────────────────

def _load_mitre_data() -> dict[str, dict[str, str]]:
    """Load and validate the MITRE ATT&CK JSON data file.

    Returns:
        Parsed dictionary from ``data/mitre_attack.json``.

    Raises:
        RuntimeError: If the file is missing / unparsable or any expected
            technique key is absent.
    """
    filepath = os.path.join(_DATA_DIR, "mitre_attack.json")
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise RuntimeError(f"MITRE data file not found: {filepath}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse MITRE data: {exc}") from exc

    for key in _EXPECTED_MITRE_KEYS:
        if key not in data:
            raise RuntimeError(f"Missing MITRE key: {key}")

    return data


_MITRE_DATA: dict[str, dict[str, str]] = _load_mitre_data()


# ── Public API ────────────────────────────────────────────────────────────────

def map_rules_to_mitre(triggered_rule_ids: list[str]) -> list[MitreTechnique]:
    """Map triggered rule IDs to their corresponding MITRE ATT&CK techniques.

    Args:
        triggered_rule_ids: List of AccessSentinel rule identifiers that
            have been triggered (non-suppressed).

    Returns:
        List of :class:`MitreTechnique` instances for every triggered rule
        whose identifier exists in :data:`RULE_TO_MITRE`. Rules not found
        in the mapping are silently skipped.
    """
    results: list[MitreTechnique] = []
    for rule_id in triggered_rule_ids:
        technique_id: str | None = RULE_TO_MITRE.get(rule_id)
        if technique_id is None:
            continue
        entry: dict[str, str] = _MITRE_DATA[technique_id]
        results.append(
            MitreTechnique(
                technique_id=technique_id,
                name=entry["name"],
                tactic=entry["tactic"],
                url=entry["url"],
                triggered_by_rule=rule_id,
            )
        )
    return results


def get_all_mitre_techniques() -> list[MitreTechnique]:
    """Return every MITRE ATT&CK technique defined in the loaded JSON.

    Returns:
        List of all 10 :class:`MitreTechnique` instances from the data file.
    """
    results: list[MitreTechnique] = []
    for technique_id, entry in _MITRE_DATA.items():
        results.append(
            MitreTechnique(
                technique_id=technique_id,
                name=entry["name"],
                tactic=entry["tactic"],
                url=entry["url"],
                triggered_by_rule="",
            )
        )
    return results
