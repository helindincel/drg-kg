"""Strict-mode helper.

When ``DRG_STRICT=1`` (or any truthy value), optional/best-effort subsystems
(coreference resolution, entity resolution, implicit relation inference,
hub validation, ...) raise on failure instead of being silently downgraded.
This is intended for research / debugging where reproducibility matters more
than pipeline resilience.

Usage::

    from drg.utils.strict import is_strict

    try:
        ...
    except Exception as e:
        if is_strict():
            raise
        logger.warning("subsystem failed: %s", e, exc_info=True)
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "y", "on"}


def is_strict() -> bool:
    """Return True when ``DRG_STRICT`` env var is set to a truthy value."""
    return os.getenv("DRG_STRICT", "0").strip().lower() in _TRUTHY
