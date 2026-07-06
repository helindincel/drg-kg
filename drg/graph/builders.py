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

from ..confidence import ConfidenceStrategy, DefaultConfidenceStrategy
from ..schema import DRGSchema, EnhancedDRGSchema
from .kg_core import EnhancedKG, KGEdge, KGNode
from .provenance import attach_provenance, find_text_provenance


def extract_evidence_snippet(
    full_text: str,
    source: str,
    target: str,
    *,
    relation: str | None = None,
    max_chars: int = 240,
    max_pair_distance: int = 2500,
) -> str | None:
    """Extract a short, deterministic evidence snippet containing source & target.

    It is a string-indexed snippet cut from the same input text.
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

    def _relation_terms(rel: str | None) -> list[str]:
        if not rel:
            return []
        rel_norm = rel.strip().replace("-", "_").lower()
        if not rel_norm:
            return []
        parts = [p for p in re.split(r"[_\W]+", rel_norm) if p]
        terms: list[str] = []
        if len(parts) > 1:
            terms.append(" ".join(parts))
        for p in parts:
            if len(p) >= 3:
                terms.append(p)
        seen: set[str] = set()
        out: list[str] = []
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            out.append(term)
        return out

    relation_terms = _relation_terms(relation)

    # Prefer a sentence that explicitly contains source + target + relation cue.
    if relation_terms:
        sentence_pattern = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
        for m in sentence_pattern.finditer(full_text):
            sentence = m.group(0).strip()
            if not sentence:
                continue
            sent_l = sentence.lower()
            if s.lower() in sent_l and t.lower() in sent_l and any(rt in sent_l for rt in relation_terms):
                sentence = re.sub(r"\s+", " ", sentence).strip()
                if len(sentence) <= max_chars:
                    return sentence
                return sentence[:max_chars].rstrip() + " …"

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


def _is_valid_schema_triple(
    schema: DRGSchema | EnhancedDRGSchema,
    *,
    source: str,
    relation: str,
    target: str,
    entity_type_map: dict[str, str],
) -> bool:
    """Return True when a triple conforms to the schema ontology."""
    src_type = entity_type_map.get(source)
    dst_type = entity_type_map.get(target)
    if not (src_type and dst_type):
        return False
    if isinstance(schema, EnhancedDRGSchema):
        return schema.is_valid_relation(relation, src_type, dst_type)
    for rel in schema.relations:
        if rel.name == relation and rel.src == src_type and rel.dst == dst_type:
            return True
    return False


def _filter_extractions_for_schema(
    *,
    schema: DRGSchema | EnhancedDRGSchema | None,
    entities_typed: list[tuple[str, str]],
    triples: list[tuple[str, str, str]],
    enriched_relations: list[dict[str, Any]] | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]], list[dict[str, Any]] | None]:
    """Drop entities and triples that violate the provided schema."""
    if schema is None:
        return entities_typed, triples, enriched_relations

    if isinstance(schema, EnhancedDRGSchema):
        allowed_types = {et.name for et in schema.entity_types}
    else:
        allowed_types = {e.name for e in schema.entities}

    filtered_entities = [(name, etype) for name, etype in entities_typed if etype in allowed_types]
    entity_type_map = dict(filtered_entities)

    filtered_triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for triple in triples:
        if triple in seen:
            continue
        s, r, o = triple
        if _is_valid_schema_triple(schema, source=s, relation=r, target=o, entity_type_map=entity_type_map):
            filtered_triples.append(triple)
            seen.add(triple)

    filtered_enriched: list[dict[str, Any]] | None = None
    if enriched_relations is not None:
        valid_set = set(filtered_triples)
        filtered_enriched = [
            item
            for item in enriched_relations
            if isinstance(item.get("relation"), (list, tuple))
            and tuple(item["relation"]) in valid_set
        ]

    return filtered_entities, filtered_triples, filtered_enriched


def _normalize_evidence_text(evidence: str | None) -> str:
    if not evidence or not isinstance(evidence, str):
        return ""
    return re.sub(r"\s+", " ", evidence.strip().lower())


def _filter_redundant_relations(
    triples: list[tuple[str, str, str]],
    enriched_relations: list[dict[str, Any]] | None = None,
) -> tuple[list[tuple[str, str, str]], list[dict[str, Any]] | None]:
    """Drop parallel edges that restate the same fact with identical evidence.

    When multiple relations share the same unordered endpoint pair and the
    same normalized evidence span, keep the highest-confidence triple.
    """
    if not triples:
        return triples, enriched_relations

    enriched_by_triple: dict[tuple[str, str, str], dict[str, Any]] = {}
    if enriched_relations:
        for item in enriched_relations:
            rel = item.get("relation")
            if isinstance(rel, (list, tuple)) and len(rel) >= 3:
                enriched_by_triple[(str(rel[0]), str(rel[1]), str(rel[2]))] = item

    # Group by (sorted endpoints, evidence); triples without evidence are kept as-is.
    groups: dict[tuple[str, ...], list[tuple[str, str, str]]] = {}
    passthrough: list[tuple[str, str, str]] = []
    for triple in triples:
        s, r, o = triple
        evidence = _normalize_evidence_text(enriched_by_triple.get(triple, {}).get("evidence"))
        if not evidence:
            passthrough.append(triple)
            continue
        key = (s.lower(), o.lower()) if s.lower() <= o.lower() else (o.lower(), s.lower())
        group_key = (*key, evidence)
        groups.setdefault(group_key, []).append(triple)

    kept: list[tuple[str, str, str]] = list(passthrough)
    for group in groups.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        best = max(
            group,
            key=lambda t: float(enriched_by_triple.get(t, {}).get("confidence") or 0.0),
        )
        kept.append(best)

    kept_set = set(kept)
    kept_triples = [t for t in triples if t in kept_set]

    filtered_enriched: list[dict[str, Any]] | None = None
    if enriched_relations is not None:
        filtered_enriched = [
            item
            for item in enriched_relations
            if isinstance(item.get("relation"), (list, tuple))
            and tuple(item["relation"]) in kept_set
        ]

    return kept_triples, filtered_enriched


def _apply_name_mapping_to_extractions(
    *,
    entities_typed: list[tuple[str, str]],
    triples: list[tuple[str, str, str]],
    enriched_relations: list[dict[str, Any]] | None,
    name_mapping: dict[str, str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]], list[dict[str, Any]] | None]:
    """Rewrite extraction outputs to canonical entity ids before graph build."""
    if not name_mapping:
        return entities_typed, triples, enriched_relations

    remapped_entities: list[tuple[str, str]] = []
    seen_entities: set[tuple[str, str]] = set()
    for name, etype in entities_typed:
        canonical = name_mapping.get(name, name)
        key = (canonical, etype)
        if key not in seen_entities:
            remapped_entities.append(key)
            seen_entities.add(key)

    remapped_triples: list[tuple[str, str, str]] = []
    seen_triples: set[tuple[str, str, str]] = set()
    for s, r, o in triples:
        triple = (name_mapping.get(s, s), r, name_mapping.get(o, o))
        if triple[0] == triple[2] or triple in seen_triples:
            continue
        remapped_triples.append(triple)
        seen_triples.add(triple)

    remapped_enriched: list[dict[str, Any]] | None = None
    if enriched_relations is not None:
        remapped_enriched = []
        for item in enriched_relations:
            rel = item.get("relation")
            if not isinstance(rel, (list, tuple)) or len(rel) < 3:
                continue
            mapped = (
                name_mapping.get(str(rel[0]), str(rel[0])),
                str(rel[1]),
                name_mapping.get(str(rel[2]), str(rel[2])),
            )
            if mapped[0] == mapped[2]:
                continue
            remapped_item = dict(item)
            remapped_item["relation"] = mapped
            remapped_enriched.append(remapped_item)

    return remapped_entities, remapped_triples, remapped_enriched


def _prune_isolated_nodes(kg: EnhancedKG) -> int:
    """Remove nodes that participate in no edges (in-place).

    Improves information density by dropping extracted mentions that never
    licensed a relation. Returns the number of nodes removed.
    """
    if not kg.edges:
        return 0

    connected: set[str] = set()
    for edge in kg.edges:
        connected.add(edge.source)
        connected.add(edge.target)

    removed = 0
    for node_id in list(kg.nodes.keys()):
        if node_id not in connected:
            del kg.nodes[node_id]
            removed += 1

    if removed and kg.clusters:
        for cluster in kg.clusters.values():
            cluster.node_ids = {nid for nid in cluster.node_ids if nid in kg.nodes}

    return removed


def build_enhanced_kg(
    *,
    entities_typed: list[tuple[str, str]],
    triples: list[tuple[str, str, str]],
    schema: DRGSchema | EnhancedDRGSchema | None = None,
    source_text: str | None = None,
    enriched_relations: list[dict[str, Any]] | None = None,
    confidence_strategy: ConfidenceStrategy | None | str = "default",
    entity_confidences: dict[str, float] | None = None,
    relation_confidences: dict[tuple[str, str, str], float] | None = None,
    entity_properties: dict[str, dict[str, Any]] | None = None,
    document_id: str | None = None,
    events: list[Any] | None = None,
    name_mapping: dict[str, str] | None = None,
    entity_aliases: dict[str, list[str]] | None = None,
    filter_against_schema: bool = True,
    prune_isolated_nodes: bool = True,
    filter_redundant_relations: bool = True,
) -> EnhancedKG:
    """Build EnhancedKG from typed entities and triples.

    Args:
        entities_typed: [(entity_name, entity_type), ...]
        triples: [(source, relation, target), ...]
        schema: Optional schema used to enrich edges with `relationship_description`
        source_text: Optional original text used to populate `relationship_detail` with evidence snippet
        enriched_relations: Optional per-triple metadata as produced by
            :func:`drg.extract.extract_typed` (when ``return_enriched=True``).
            Used by the confidence strategy to honour upstream signals
            (negation, temporal cues, future LLM self-rating).
        confidence_strategy: How to compute confidence scores. One of:
            - ``"default"`` (the default): use
              :class:`drg.confidence.DefaultConfidenceStrategy` — heuristic
              placeholder, schema-aware, side-effect-free.
            - ``None``: do not compute or attach any confidence (legacy
              behaviour; existing call sites that don't opt-in stay
              unchanged).
            - A :class:`drg.confidence.ConfidenceStrategy` instance to
              plug in a custom scorer.
        entity_confidences: Optional explicit ``entity_name -> score`` map.
            When provided, these values override what the strategy would
            compute for matching entities. Useful for piping pre-computed
            scores from a custom upstream pipeline.
        relation_confidences: Optional explicit ``(s, r, o) -> score`` map.
            Same semantics as ``entity_confidences`` but for edges.
        entity_properties: Optional ``entity_name -> properties`` map populated
            from typed extraction outputs. Values are copied onto matching
            :class:`KGNode` objects.
        document_id: Optional identifier for the source document. When
            provided, every edge built here is stamped with
            ``metadata['source_ref'] = document_id`` (existing
            ``source_ref`` values from ``enriched_relations`` win), and
            nodes get ``metadata['source_documents']`` populated with
            ``[document_id]``. This is what later lets
            :mod:`drg.reasoning` distinguish edges that originated from
            **different documents** when running multi-document
            inference rules like
            :class:`drg.reasoning.PathBridgeRule`. Default ``None``
            preserves the legacy behaviour exactly.
        name_mapping: Optional ``original_name -> canonical_name`` map from
            entity resolution. Endpoints are rewritten before nodes/edges are
            created and the mapping is applied again on the assembled graph.
        entity_aliases: Optional ``canonical_name -> [alias, ...]`` map stored
            on node metadata for search and explainability.
        filter_against_schema: When ``True`` (default) and ``schema`` is set,
            drop entities whose types and triples whose relation endpoints are
            not valid under the schema before building the graph.
        prune_isolated_nodes: When ``True`` (default), remove nodes that do not
            participate in any edge after the graph is assembled. This favors
            information density over raw entity recall.
        filter_redundant_relations: When ``True`` (default), drop parallel edges
            that share the same endpoint pair and evidence span, keeping the
            highest-confidence relation.

    Returns:
        EnhancedKG. Nodes/edges carry ``confidence`` attributes when a
        strategy was applied (or explicit overrides were provided);
        otherwise both attributes are ``None`` (legacy behaviour).
    """
    if name_mapping:
        entities_typed, triples, enriched_relations = _apply_name_mapping_to_extractions(
            entities_typed=entities_typed,
            triples=triples,
            enriched_relations=enriched_relations,
            name_mapping=name_mapping,
        )

    if filter_against_schema and schema is not None:
        entities_typed, triples, enriched_relations = _filter_extractions_for_schema(
            schema=schema,
            entities_typed=entities_typed,
            triples=triples,
            enriched_relations=enriched_relations,
        )

    if filter_redundant_relations:
        triples, enriched_relations = _filter_redundant_relations(triples, enriched_relations)

    kg = EnhancedKG()
    entity_type_map = dict(entities_typed)
    entity_properties_map = {
        str(name): dict(props)
        for name, props in (entity_properties or {}).items()
        if isinstance(props, dict)
    }

    # Resolve the strategy. ``"default"`` is a sentinel rather than a real
    # default-arg-instance because instantiating the strategy at module
    # import time would be wasteful for callers that pass ``None``.
    strategy: ConfidenceStrategy | None
    if confidence_strategy == "default":
        strategy = DefaultConfidenceStrategy()
    elif isinstance(confidence_strategy, ConfidenceStrategy):
        strategy = confidence_strategy
    else:
        strategy = None  # explicit None or unknown sentinel

    # Compute confidence maps once per call. Strategies are pure, so this
    # is cheap; we batch by passing all entities/relations together.
    ent_score_map: dict[str, float] = dict(entity_confidences or {})
    rel_score_map: dict[tuple[str, str, str], float] = dict(relation_confidences or {})

    if strategy is not None:
        ctx = {
            "schema": schema,
            "source_text": source_text,
            "entity_types": entity_type_map,
        }
        ent_scores = strategy.score_entities(entities_typed, context=ctx)
        for ename, sc in ent_scores.items():
            # Explicit overrides win — they reflect caller intent, while
            # the strategy is the fallback scorer.
            ent_score_map.setdefault(ename, sc.value)

        rel_scores = strategy.score_relations(
            triples,
            enriched_relations=enriched_relations,
            context=ctx,
        )
        for triple, sc in rel_scores.items():
            rel_score_map.setdefault(triple, sc.value)

    try:
        from .._version import __version__ as extractor_version
    except Exception:  # pragma: no cover - only hit in unusual source checkouts
        extractor_version = None

    # When a document_id is supplied, every node introduced from this build
    # gets a `source_documents` provenance list. The reasoning layer reads
    # this to understand which graph nodes were touched by which documents
    # (used by hints / explanations); the merger later unions the lists.
    node_metadata_base: dict[str, Any] = {"source_documents": [document_id]} if document_id else {}

    for name, etype in entities_typed:
        node_metadata = attach_provenance(
            node_metadata_base,
            find_text_provenance(
                source_text,
                (name,),
                document_id=document_id,
                extractor_version=extractor_version,
            ),
        )
        aliases = entity_aliases.get(name) if entity_aliases else None
        if aliases:
            node_metadata = dict(node_metadata)
            node_metadata["aliases"] = sorted({str(a) for a in aliases if str(a).strip()}, key=str.lower)
        kg.add_node(
            KGNode(
                id=name,
                type=etype,
                properties=entity_properties_map.get(name, {}),
                metadata=node_metadata,
                confidence=ent_score_map.get(name),
            )
        )

    try:
        evidence_max_chars = int(os.getenv("DRG_EVIDENCE_MAX_CHARS", "240"))
    except Exception:
        evidence_max_chars = 240
    try:
        evidence_max_pair_distance = int(os.getenv("DRG_EVIDENCE_MAX_PAIR_DISTANCE", "2500"))
    except Exception:
        evidence_max_pair_distance = 2500

    enriched_by_triple: dict[tuple[str, str, str], dict[str, Any]] = {}
    if enriched_relations:
        for item in enriched_relations:
            rel = item.get("relation")
            if isinstance(rel, (list, tuple)) and len(rel) >= 3:
                enriched_by_triple[(str(rel[0]), str(rel[1]), str(rel[2]))] = item

    for s, r, o in triples:
        if s not in kg.nodes:
            # Synthetic node for a triple endpoint not present in entities_typed.
            # Apply any pre-computed confidence so even synthetic nodes
            # carry a score when available.
            node_metadata = attach_provenance(
                node_metadata_base,
                find_text_provenance(
                    source_text,
                    (s,),
                    document_id=document_id,
                    extractor_version=extractor_version,
                ),
            )
            kg.add_node(
                KGNode(
                    id=s,
                    type=entity_type_map.get(s),
                    properties=entity_properties_map.get(s, {}),
                    metadata=node_metadata,
                    confidence=ent_score_map.get(s),
                )
            )
        if o not in kg.nodes:
            node_metadata = attach_provenance(
                node_metadata_base,
                find_text_provenance(
                    source_text,
                    (o,),
                    document_id=document_id,
                    extractor_version=extractor_version,
                ),
            )
            kg.add_node(
                KGNode(
                    id=o,
                    type=entity_type_map.get(o),
                    properties=entity_properties_map.get(o, {}),
                    metadata=node_metadata,
                    confidence=ent_score_map.get(o),
                )
            )

        src_type = entity_type_map.get(s)
        dst_type = entity_type_map.get(o)
        rel_desc, rel_det = _relation_docs_from_schema(schema, r, src_type, dst_type)

        md: dict[str, Any] = {"triple": [s, r, o]}
        if rel_desc:
            md["relationship_description"] = rel_desc
        if rel_det:
            md["schema_detail"] = rel_det

        # Stamp the document of origin so multi-document inference
        # (`drg.reasoning.PathBridgeRule`) can distinguish edges that
        # originated from different documents after a merge. Edges that
        # already carry a `source_ref` (e.g. from `enriched_relations`)
        # keep their existing value.
        if document_id and "source_ref" not in md:
            md["source_ref"] = document_id

        evidence = None

        # Always ensure a usable description field exists (sample-format alignment).
        if "relationship_description" not in md:
            md["relationship_description"] = f"Auto-extracted relation '{r}'."

        enriched_item = enriched_by_triple.get((s, r, o))
        start_time: str | None = None
        end_time: str | None = None
        is_negated = False
        edge_provenance = find_text_provenance(
            source_text,
            (s, o),
            document_id=document_id,
            extractor_version=extractor_version,
        )
        if enriched_item:
            temporal = enriched_item.get("temporal")
            if isinstance(temporal, dict):
                start_time = temporal.get("valid_from") or temporal.get("start")
                end_time = temporal.get("valid_to") or temporal.get("end")
                if start_time or end_time:
                    from ..temporal import TemporalScope

                    scope = TemporalScope.from_legacy_temporal(temporal)
                    if scope is not None:
                        temporal_meta = scope.to_dict()
                        if edge_provenance.snippet and "raw_text" not in temporal_meta:
                            temporal_meta["raw_text"] = edge_provenance.snippet
                        md["temporal"] = temporal_meta
                        precision = temporal.get("precision")
                        if isinstance(precision, str) and precision:
                            md["temporal"]["precision"] = precision
            is_negated = bool(enriched_item.get("is_negated", False))
            existing_ref = enriched_item.get("source_ref")
            if isinstance(existing_ref, str) and existing_ref:
                md["source_ref"] = existing_ref
                edge_provenance = find_text_provenance(
                    source_text,
                    (s, o),
                    document_id=existing_ref,
                    extractor_version=extractor_version,
                )

        if source_text:
            enriched_evidence = None
            if enriched_item:
                raw_evidence = enriched_item.get("evidence")
                if isinstance(raw_evidence, str) and raw_evidence.strip():
                    enriched_evidence = raw_evidence.strip()

            evidence = enriched_evidence or evidence
            if evidence is None:
                evidence = extract_evidence_snippet(
                    source_text,
                    s,
                    o,
                    relation=r,
                    max_chars=evidence_max_chars,
                    max_pair_distance=evidence_max_pair_distance,
                )
            if evidence:
                md["evidence"] = evidence
                edge_provenance = find_text_provenance(
                    source_text,
                    (s, o),
                    document_id=md.get("source_ref") if isinstance(md.get("source_ref"), str) else document_id,
                    extractor_version=extractor_version,
                    preferred_snippet=evidence,
                )

        md = attach_provenance(md, edge_provenance)

        kg.add_edge(
            KGEdge(
                source=s,
                target=o,
                relationship_type=r,
                relationship_detail=md.get("evidence") or f"{s} {r} {o}",
                metadata=md,
                start_time=start_time,
                end_time=end_time,
                confidence=rel_score_map.get((s, r, o)),
                is_negated=is_negated,
            )
        )

    if events:
        # Lazy import — keeps the legacy builder dependency-light when no
        # events are supplied (preserves the byte-for-byte output of pre-
        # event-extraction graphs).
        from ..events._graph_mapping import (
            event_to_kg_node,
            event_to_role_edges,
        )

        for event in events:
            event_node = event_to_kg_node(event)
            if event_node.id not in kg.nodes:
                kg.add_node(event_node)
            for role_edge in event_to_role_edges(event):
                # Auto-create participant nodes that weren't in entities_typed
                # (defensive — extractor should have resolved them, but a
                # caller may pass raw events from elsewhere).
                if role_edge.target not in kg.nodes:
                    kg.add_node(
                        KGNode(
                            id=role_edge.target,
                            type=entity_type_map.get(role_edge.target),
                            properties={},
                            metadata=dict(node_metadata_base),
                            confidence=ent_score_map.get(role_edge.target),
                        )
                    )
                kg.add_edge(role_edge)

    if name_mapping:
        kg.canonicalize_entities(name_mapping)

    if prune_isolated_nodes:
        pruned = _prune_isolated_nodes(kg)
        if pruned:
            meta = dict(kg.metadata)
            build_meta = dict(meta.get("build", {}) or {})
            build_meta["pruned_isolated_nodes"] = pruned
            meta["build"] = build_meta
            kg.metadata = meta

    return kg
