"""Shared verdict vocabulary + parsing (used by the mock VLM and the debate).

Kept neutral (no dependency on vlm/ or debate/) so both layers can import it.
"""

import re

VERDICTS = ["endorse", "amend", "reject"]

# Conservatism ordering for tie-breaks / concessions: reject > amend > endorse.
CONSERVATISM = {"endorse": 0, "amend": 1, "reject": 2}

_VERDICT_RE = re.compile(r"VERDICT:\s*(endorse|amend|reject)", re.IGNORECASE)


def parse_verdict(text: str) -> str | None:
    """Extract the verdict from a reply's ``VERDICT:`` line, if present."""
    match = _VERDICT_RE.search(text or "")
    return match.group(1).lower() if match else None


def most_conservative(*verdicts: str) -> str:
    """Return the most conservative of the given verdicts."""
    return max(verdicts, key=lambda v: CONSERVATISM[v])
