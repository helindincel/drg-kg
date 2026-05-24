"""Entity name normalization.

Strips honorifics (Dr., Prof., …), suffixes (Jr., II, …) and collapses
whitespace so that surface variants of the same entity end up with the same
canonical form. Domain-agnostic by design: we don't try to know the difference
between "Apple Inc." and "Apple"; we just normalize the cosmetic noise around
the underlying token.
"""

from __future__ import annotations

import re

# One pass-friendly group of regexes. Adding a new prefix/suffix here is the
# only place it needs to be touched — the resolver picks it up automatically.
_TITLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(dr|doctor|prof|professor|mr|mrs|miss|ms|sir|madam|lord|lady)\s*\.?\s*",
        re.IGNORECASE,
    ),
    re.compile(r"\s+(jr|sr|jr\.|sr\.|ii|iii|iv)$", re.IGNORECASE),
)


def normalize_entity_name(name: str) -> str:
    """Return a casefolded, title-stripped, whitespace-collapsed form of ``name``.

    Examples:
        - ``"Dr. Elena Vasquez"`` → ``"elena vasquez"``
        - ``"Prof. John Smith Jr."`` → ``"john smith"``
        - ``"  Cognitive   Enhancement  "`` → ``"cognitive enhancement"``

    The output is intended for comparison only; callers should keep the
    original ``name`` for display and use the normalized form as a cache key
    or similarity input.
    """
    normalized = name.lower().strip()
    for pattern in _TITLE_PATTERNS:
        normalized = pattern.sub("", normalized)
    return " ".join(normalized.split())
