"""Knowledge graph builders.

Centralizes conversion from extracted (typed entities, triples) into EnhancedKG so that:
- Edge metadata consistently includes schema-based `relationship_description` when available.
- Edge `relationship_detail` can be populated with deterministic evidence snippets from the same input text.

This keeps behavior consistent across CLI, examples, and any other pipeline entrypoint.
"""

from __future__ import annotations

import os
import re
from typing import Any

from ..schema import DRGSchema, EnhancedDRGSchema
from .kg_core import EnhancedKG, KGEdge, KGNode


def extract_evidence_snippet(
    full_text: str,
    source: str,
    target: str,
    *,
    max_chars: int = 240,
    max_pair_distance: int = 2500,
) -> str | None:
    """Extract a short, deterministic evidence snippet containing source & target.

    This is NOT retrieval/RAG. It is a string-indexed snippet cut from the same input text.
    Conservative behavior: if we can't find a reasonable co-occurrence, return None.
    """
    if not full_text or not source or not target:
        return None
    s = source.strip()
    t = target.strip()
    if not s or not t:
        return None

    def _pattern(x: str) -> re.Pattern[str]:
        return re.compile(rf"(?<!\w){re.escape(x)}(?!\w)", re.IGNORECASE)

    ps = _pattern(s)
    pt = _pattern(t)

    s_matches = [m.start() for m in ps.finditer(full_text)]
    t_matches = [m.start() for m in pt.finditer(full_text)]
    if not s_matches or not t_matches:
        return None

    s_matches = s_matches[:20]
    t_matches = t_matches[:20]

    best: tuple[int, int, int] | None = None  # (distance, s_pos, t_pos)
    for sp in s_matches:
        for tp in t_matches:
            dist = abs(sp - tp)
            if best is None or dist < best[0]:
                best = (dist, sp, tp)
    if best is None:
        return None
    dist, sp, tp = best
    if dist > max_pair_distance:
        return None

    lo = min(sp, tp)
    hi = max(sp, tp)

    left = full_text.rfind("\n", 0, lo)
    punct_left = max(
        full_text.rfind(".", 0, lo), full_text.rfind("?", 0, lo), full_text.rfind("!", 0, lo)
    )
    left = max(left, punct_left)
    if left == -1:
        left = 0
    else:
        left = min(len(full_text), left + 1)

    right_candidates = [
        full_text.find("\n", hi),
        full_text.find(".", hi),
        full_text.find("?", hi),
        full_text.find("!", hi),
    ]
    right_candidates = [x for x in right_candidates if x != -1]
    right = min(right_candidates) + 1 if right_candidates else len(full_text)

    snippet = full_text[left:right].strip()
    truncated_left = False
    truncated_right = False

    if len(snippet) > max_chars:
        # Centered window around the closest co-occurrence.
        mid = (lo + hi) // 2
        half = max_chars // 2
        ws = max(0, mid - half)
        we = min(len(full_text), ws + max_chars)
        truncated_left = ws > 0
        truncated_right = we < len(full_text)
        snippet = full_text[ws:we].strip()

    # Normalize whitespace early.
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if not snippet:
        return None

    # Avoid cutting mid-word: if we truncated, trim to nearest whitespace boundary.
    if truncated_left and " " in snippet:
        first_space = snippet.find(" ")
        # If the cut likely started mid-word, drop leading fragment.
        if first_space != -1 and first_space < 20:
            snippet = snippet[first_space + 1 :].lstrip()
    if truncated_right and " " in snippet:
        last_space = snippet.rfind(" ")
        # If the cut likely ended mid-word, drop trailing fragment.
        if last_space != -1 and (len(snippet) - last_space) < 20:
            snippet = snippet[:last_space].rstrip()

    # Add ellipses to communicate truncation (UI-friendly).
    if truncated_left:
        snippet = "… " + snippet
    if truncated_right:
        snippet = snippet + " …"

    return snippet or None


def _relation_docs_from_schema(
    schema: DRGSchema | EnhancedDRGSchema | None,
    rel_name: str,
    src_type: str | None,
    dst_type: str | None,
) -> tuple[str | None, str | None]:
    """Best-effort lookup of relation description/detail from schema.

    Prefers exact (name, src, dst) match when possible; falls back to name-only match.
    """
    if schema is None or not hasattr(schema, "relation_groups"):
        return None, None

    candidates: list[tuple[str | None, str | None]] = []
    for rg in getattr(schema, "relation_groups", []):
        for r in getattr(rg, "relations", []):
            if getattr(r, "name", None) != rel_name:
                continue
            r_src = getattr(r, "src", None)
            r_dst = getattr(r, "dst", None)
            desc = getattr(r, "description", None)
            det = getattr(r, "detail", None)
            desc_s = desc if isinstance(desc, str) and desc.strip() else None
            det_s = det if isinstance(det, str) and det.strip() else None

            if (
                src_type is not None
                and dst_type is not None
                and r_src == src_type
                and r_dst == dst_type
            ):
                return desc_s, det_s
            candidates.append((desc_s, det_s))

    for desc_s, det_s in candidates:
        if desc_s or det_s:
            return desc_s, det_s
    return None, None


def build_enhanced_kg(
    *,
    entities_typed: list[tuple[str, str]],
    triples: list[tuple[str, str, str]],
    schema: DRGSchema | EnhancedDRGSchema | None = None,
    source_text: str | None = None,
) -> EnhancedKG:
    """Build EnhancedKG from typed entities and triples.

    Args:
        entities_typed: [(entity_name, entity_type), ...]
        triples: [(source, relation, target), ...]
        schema: Optional schema used to enrich edges with `relationship_description`
        source_text: Optional original text used to populate `relationship_detail` with evidence snippet

    Returns:
        EnhancedKG
    """
    kg = EnhancedKG()
    entity_type_map = dict(entities_typed)

    for name, etype in entities_typed:
        kg.add_node(KGNode(id=name, type=etype, properties={}, metadata={}))

    try:
        evidence_max_chars = int(os.getenv("DRG_EVIDENCE_MAX_CHARS", "240"))
    except Exception:
        evidence_max_chars = 240
    try:
        evidence_max_pair_distance = int(os.getenv("DRG_EVIDENCE_MAX_PAIR_DISTANCE", "2500"))
    except Exception:
        evidence_max_pair_distance = 2500

    for s, r, o in triples:
        if s not in kg.nodes:
            kg.add_node(KGNode(id=s, type=entity_type_map.get(s), properties={}, metadata={}))
        if o not in kg.nodes:
            kg.add_node(KGNode(id=o, type=entity_type_map.get(o), properties={}, metadata={}))

        src_type = entity_type_map.get(s)
        dst_type = entity_type_map.get(o)
        rel_desc, rel_det = _relation_docs_from_schema(schema, r, src_type, dst_type)

        md: dict[str, Any] = {"triple": [s, r, o]}
        if rel_desc:
            md["relationship_description"] = rel_desc
        if rel_det:
            md["schema_detail"] = rel_det

        evidence = None
        if source_text:
            evidence = extract_evidence_snippet(
                source_text,
                s,
                o,
                max_chars=evidence_max_chars,
                max_pair_distance=evidence_max_pair_distance,
            )
            if evidence:
                md["evidence"] = evidence

        # Always ensure a usable description field exists (sample-format alignment).
        if "relationship_description" not in md:
            md["relationship_description"] = f"Auto-extracted relation '{r}'."

        kg.add_edge(
            KGEdge(
                source=s,
                target=o,
                relationship_type=r,
                relationship_detail=evidence or f"{s} {r} {o}",
                metadata=md,
            )
        )

    return kg
