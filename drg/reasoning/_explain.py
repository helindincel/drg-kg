"""Human-readable explanation helpers for inferred edges.

These functions live in a tiny module of their own so that:

- rules in ``_rules.py`` can share consistent wording without
  importing the engine;
- callers that want to re-render an existing inference (e.g. for a UI
  popover) can call the same helpers without re-running the rule.

Tone notes
----------
The explanations are written in **plain English** to keep the
provenance trail readable regardless of which language the source
documents were in. Document identifiers are surfaced verbatim so that
explanations remain auditable — they're not just nice prose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import EvidenceLink

__all__ = [
    "explain_composition",
    "explain_inverse",
    "explain_path_bridge",
    "explain_symmetric",
    "explain_transitive",
]


def _fmt_link(link: EvidenceLink) -> str:
    """Render a single ``EvidenceLink`` for inline citation."""
    s, r, o = link.triple
    ref = f" (source: {link.source_ref})" if link.source_ref else ""
    return f"{s} —[{r}]→ {o}{ref}"


def explain_path_bridge(
    *,
    src: str,
    dst: str,
    bridge: str,
    evidence: list[EvidenceLink],
) -> str:
    """Explanation for :class:`PathBridgeRule`.

    Example output::

        "Apple and Jimmy Iovine are both connected to Beats. "
        "Evidence: Apple —[ACQUIRED]→ Beats (source: doc_A); "
        "Jimmy Iovine —[FOUNDED]→ Beats (source: doc_B)."
    """
    rendered = "; ".join(_fmt_link(link) for link in evidence)
    return (
        f"{src} and {dst} are both connected to {bridge}. "
        f"Evidence: {rendered}."
    )


def explain_inverse(
    *,
    source: str,
    relation: str,
    target: str,
    original: EvidenceLink,
) -> str:
    """Explanation for :class:`InverseRule`."""
    return (
        f"{source} {relation} {target} follows by inverse from "
        f"{_fmt_link(original)}."
    )


def explain_symmetric(
    *,
    source: str,
    relation: str,
    target: str,
    original: EvidenceLink,
) -> str:
    """Explanation for :class:`SymmetricRule`."""
    return (
        f"{source} {relation} {target} holds because the relation is "
        f"symmetric and {_fmt_link(original)} was extracted."
    )


def explain_transitive(
    *,
    head: str,
    mid: str,
    tail: str,
    relation: str,
) -> str:
    """Explanation for :class:`TransitiveRule`."""
    return (
        f"{head} {relation} {tail} follows by transitivity through {mid} "
        f"({head} {relation} {mid}, {mid} {relation} {tail})."
    )


def explain_composition(
    *,
    owner: str,
    asset: str,
    location: str,
) -> str:
    """Explanation for :class:`CompositionRule`."""
    return (
        f"{owner} operates_in {location} because {owner} owns/controls "
        f"{asset}, which is located in {location}."
    )
