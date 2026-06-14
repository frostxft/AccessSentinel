"""Geo-region utilities for cross-system impossible-travel detection.

Maintains an explicit, hand-curated mapping from location strings that
appear in sample_data/identity_events.csv to broad geographic regions.
Locations not in this mapping are treated as region="unknown" and are
excluded from cross-system travel detection (no false positives from
missing data).
"""

REGION_BY_LOCATION: dict[str, str] = {
    "New York, US": "North America",
    "Toronto, CA": "North America",
    "London, UK": "Europe",
    "Paris, FR": "Europe",
    "Berlin, DE": "Europe",
    "Mumbai, IN": "Asia-Pacific",
    "Singapore, SG": "Asia-Pacific",
    "Tokyo, JP": "Asia-Pacific",
    "Sydney, AU": "Asia-Pacific",
    "Sao Paulo, BR": "South America",
}
