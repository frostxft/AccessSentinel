"""Role exception registry for AccessSentinel.

Provides score softening for senior leadership roles (CTO, CISO, CEO, etc.)
whose broad or infrequent access patterns are job-appropriate rather than
indicative of compromise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.ingestion import IdentityRecord

SENIOR_ROLE_ADJUSTMENTS: dict[str, int] = {
    "cto": -25,
    "ciso": -25,
    "cfo": -20,
    "ceo": -25,
    "vp": -15,
    "svp": -15,
    "evp": -15,
    "director": -15,
    "chief": -20,
}


_TOKEN_SPLIT_PATTERN = re.compile(r"[\s/\\\-_|]+")


@dataclass(frozen=True)
class RoleException:
    """A senior-role exception that softens the risk score.

    Attributes:
        role_keyword: The matching senior-role token (e.g. ``"cto"``).
        score_adjustment: Points subtracted from the raw risk score.
        rules_softened: Rule IDs whose severity is softened by this exception.
        narrative_override: Human-readable justification text.
    """

    role_keyword: str
    score_adjustment: int
    rules_softened: list[str]
    narrative_override: str


def _tokenize(value: str) -> set[str]:
    """Split a string on whitespace/separators and lowercase each token.

    Args:
        value: A raw role or job title string.

    Returns:
        Set of lowercase, non-empty tokens.
    """
    return {
        token.lower()
        for token in _TOKEN_SPLIT_PATTERN.split(value)
        if token
    }


def resolve_role_exceptions(identity: IdentityRecord) -> list[RoleException]:
    """Check whether the identity holds a senior-role exception.

    Tokenizes ``identity.job_title`` and each entry in ``identity.roles``
    on whitespace and common separators.  Matches against
    :data:`SENIOR_ROLE_ADJUSTMENTS` using exact, case-insensitive token
    comparison.  If multiple keywords match, only the single largest-magnitude
    adjustment is returned (no stacking).

    Args:
        identity: The identity record to evaluate.

    Returns:
        List of :class:`RoleException` objects (either empty or a single
        element).
    """
    tokens: set[str] = set()
    if identity.job_title:
        tokens |= _tokenize(identity.job_title)
    for role_value in identity.roles:
        tokens |= _tokenize(role_value)

    matches: list[tuple[str, int]] = []
    for keyword, adjustment in SENIOR_ROLE_ADJUSTMENTS.items():
        if keyword in tokens:
            matches.append((keyword, adjustment))

    if not matches:
        return []

    # Deduplicate: keep only the single largest-magnitude adjustment.
    best = max(matches, key=lambda x: abs(x[1]))
    keyword, adjustment = best

    return [
        RoleException(
            role_keyword=keyword,
            score_adjustment=adjustment,
            rules_softened=["STALE_PRIVILEGED"],
            narrative_override=(
                f"Role exception applied: this identity holds a "
                f"{keyword.upper()} title. Broad or infrequent access may "
                f"be job-appropriate for this seniority level. Recommended "
                f"action: do not auto-disable; schedule an access review "
                f"with HR or the identity's manager within 48 hours."
            ),
        )
    ]
