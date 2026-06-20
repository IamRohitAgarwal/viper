"""Trace logging (SPEC section 11): write the full run trace to log.json."""

import json
import os
from dataclasses import asdict, is_dataclass


def _safe(obj):
    """JSON-serialise dataclasses while dropping non-serialisable fields (images)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _safe(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return f"<{type(obj).__name__}>"   # e.g. PIL images, omitted from the log


def log_run(data: dict, path: str) -> None:
    """Write ``data`` as pretty JSON to ``path``."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_safe(data), fh, indent=2)
