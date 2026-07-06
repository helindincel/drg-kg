"""Declarative knowledge graph extraction using DSPy.

This package re-organizes the historical `drg/extract.py` monolith into focused
sub-modules while preserving the public API and the test-mock surface.

Mock-friendly attributes intentionally live at this top-level namespace
(``drg.extract.dspy``, ``drg.extract._get_extractor``,
``drg.extract._configure_llm_auto``, ``drg.extract.resolve_entities_and_relations``,
``drg.extract.resolve_coreferences``) so existing
``patch('drg.extract.X')`` calls keep working.
"""

from __future__ import annotations

import asyncio as _asyncio
import contextlib
import functools as _functools
import json
import os
from collections import Counter
from typing import Any

# Import Mock at module load (not inside function) so its identity is stable.
from unittest.mock import Mock as _Mock

# Top-level dspy import — tests patch this attribute.
import dspy

from ..errors import ExtractionError, GraphError, LLMConfigError
from ..schema import DRGSchema, EnhancedDRGSchema
from ..utils.llm_throttle import throttle_llm_calls
from ..utils.logging import get_logger, with_context
from ..utils.strict import is_strict
from ._adapters import _maybe_json_adapter_context, run_predict
from ._chunk_context import (
    _build_cross_chunk_context_snippets,
    _select_anchor_entities,
)
from ._heuristics import (
    _infer_relation_metadata_heuristic,
)
from ._relations import (
    REVERSE_RELATION_PATTERNS,
    _infer_reverse_relation_name,
    _normalize_schema,
)
from ._schema_gen import (
    SchemaGeneration,
    _sample_text_for_schema_generation,  # noqa: F401 — imported for test access
    generate_schema_from_text,
)
from ._schema_sanitizer import SanitizationReport, SchemaSanitizer
from ._signatures import (
    _create_coreference_signature,
    _create_document_relation_signature,
    _create_entity_signature,
    _create_implicit_relation_signature,
    _create_relation_signature,
)
from ._types import (
    EntityList,
    EntityMention,
    ExtractedRelation,
    ExtractionResult,
    RelationList,
    SchemaOutput,
    TemporalInfo,
)

logger = get_logger(__name__)

try:
    from ..coreference_resolution import resolve_coreferences
except ImportError:
    resolve_coreferences = None

try:
    from ..entity_resolution import resolve_entities_and_relations, resolve_entities_detailed
except ImportError:
    resolve_entities_and_relations = None
    resolve_entities_detailed = None


__all__ = [
    "EntityList",
    "ExtractionResult",
    "KGExtractor",
    "RelationList",
    "SanitizationReport",
    "SchemaGeneration",
    "SchemaOutput",
    "SchemaSanitizer",
    "create_kgedge_from_triple",
    "extract_from_chunks",
    "extract_from_chunks_async",
    "extract_triples",
    "extract_typed",
    "extract_typed_async",
    "generate_schema_from_text",
]


def _maybe_lm_context(lm: Any | None):
    """Return a context manager that scopes ``lm`` for DSPy calls, if possible.

    DSPy 2.5+ exposes a top-level ``dspy.context(lm=...)``; older releases use
    ``dspy.settings.context(lm=...)``. We probe for both and fall back to a
    :func:`contextlib.nullcontext` (i.e. the global LM is used) if neither is
    available — that path also covers heavily mocked test environments where
    patching ``dspy.context`` would otherwise raise.

    Passing ``lm=None`` always returns ``nullcontext()``, preserving the legacy
    behaviour for call sites that don't opt into dependency injection.
    """
    if lm is None:
        return contextlib.nullcontext()

    ctx_factory = getattr(dspy, "context", None)
    if ctx_factory is not None and not isinstance(ctx_factory, _Mock):
        try:
            return ctx_factory(lm=lm)
        except TypeError:
            pass

    settings = getattr(dspy, "settings", None)
    if settings is not None:
        sub_ctx = getattr(settings, "context", None)
        if sub_ctx is not None and not isinstance(sub_ctx, _Mock):
            try:
                return sub_ctx(lm=lm)
            except TypeError:
                pass

    logger.warning(
        "Injected LM provided but no dspy.context manager is available; "
        "falling back to globally configured LM."
    )
    return contextlib.nullcontext()


def _should_return_dspy_prediction() -> bool:
    """Decide whether it's safe/meaningful to return a real ``dspy.Prediction``.

    Defined at the package level (not in ``_types``) so it reads the module-level
    ``dspy`` symbol that tests patch via ``patch('drg.extract.dspy')``.

    In unit tests, ``dspy`` (or ``dspy.Prediction``) is often patched/mocked.
    Returning a mocked Prediction breaks attribute semantics. In real runs and
    optimizer runs, returning an actual ``dspy.Prediction`` keeps DSPy
    internals/optimizers happier.
    """
    pred_cls = getattr(dspy, "Prediction", None)
    if pred_cls is None:
        return False
    if isinstance(pred_cls, _Mock):
        return False
    if not isinstance(pred_cls, type):
        return False
    return getattr(pred_cls, "__module__", "").startswith("dspy")


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _require_lm() -> bool:
    """Return whether extraction should fail instead of mock-mode degrading."""
    return is_strict() or _truthy_env("DRG_REQUIRE_LM") or _truthy_env("DRG_PRODUCTION")


def _effective_lm(lm: Any | None) -> Any | None:
    """Return the injected or globally configured DSPy LM after auto config."""
    if lm is not None:
        return lm
    _configure_llm_auto()
    return getattr(getattr(dspy, "settings", None), "lm", None)


def _empty_extraction_result(return_enriched: bool):
    if return_enriched:
        return [], [], []
    return [], []


def _guard_lm_or_mock_empty(*, lm: Any | None, return_enriched: bool, operation: str):
    """Common LM guard for public extraction APIs.

    Tests and notebooks may intentionally run without a configured LM. In that
    case legacy mock mode still returns empty extraction unless strict/prod mode
    asks us to fail fast.
    """
    effective_lm = _effective_lm(lm)
    if effective_lm is not None or isinstance(_get_extractor, _Mock):
        return None

    message = (
        "No DSPy LM is loaded. Configure LM via environment variables "
        "(e.g., DRG_MODEL + API key) or unset DRG_REQUIRE_LM/DRG_STRICT to allow "
        "mock-mode empty extraction."
    )
    if _require_lm():
        raise LLMConfigError(message)

    logger.warning(
        "No DSPy LM configured for %s; returning empty extraction (mock mode).", operation
    )
    return _empty_extraction_result(return_enriched)


def _coerce_entity_mentions(raw_entities: Any) -> list[EntityMention]:
    """Normalize DSPy entity outputs while preserving the public tuple API."""
    if raw_entities is None:
        return []
    if not isinstance(raw_entities, list):
        raise ExtractionError(
            f"Entity extraction returned invalid type: {type(raw_entities).__name__}"
        )

    mentions: list[EntityMention] = []
    for item in raw_entities:
        if isinstance(item, EntityMention):
            mentions.append(item)
        elif hasattr(item, "model_dump"):
            data = item.model_dump()
            mentions.append(
                EntityMention(**{**data, "name": str(data["name"]), "type": str(data["type"])})
            )
        elif isinstance(item, dict):
            name = item.get("name") or item.get("entity") or item.get("entity_name")
            etype = item.get("type") or item.get("entity_type")
            if name and etype:
                mentions.append(EntityMention(**{**item, "name": str(name), "type": str(etype)}))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            mentions.append(EntityMention(name=str(item[0]), type=str(item[1])))
    return [m for m in mentions if m.name.strip()]


def _entity_mentions_to_tuples(mentions: list[EntityMention]) -> list[tuple[str, str]]:
    return [(m.name, m.type) for m in mentions if m.name.strip()]


def _entity_mentions_to_dspy_input(mentions: list[EntityMention]) -> list[dict[str, Any]]:
    return [
        {
            "name": m.name,
            "type": m.type,
            "aliases": m.aliases,
            "evidence": m.evidence,
            "properties": m.properties,
            "metadata": m.metadata,
        }
        for m in mentions
    ]


def _coerce_extracted_relations(raw_relations: Any) -> list[ExtractedRelation]:
    """Normalize typed relation outputs and tolerate legacy tuple-shaped mocks."""
    if raw_relations is None:
        return []
    if not isinstance(raw_relations, list):
        raise ExtractionError(
            f"Relation extraction returned invalid type: {type(raw_relations).__name__}"
        )

    relations: list[ExtractedRelation] = []
    for item in raw_relations:
        if isinstance(item, ExtractedRelation):
            relations.append(item)
        elif hasattr(item, "model_dump"):
            data = item.model_dump()
            relations.append(
                ExtractedRelation(
                    source=str(data["source"]),
                    relation=str(data["relation"]),
                    target=str(data["target"]),
                    confidence=data.get("confidence"),
                    evidence=data.get("evidence"),
                    temporal=data.get("temporal"),
                    is_negated=bool(data.get("is_negated", False)),
                    metadata=data.get("metadata") or {},
                )
            )
        elif isinstance(item, dict):
            source = item.get("source") or item.get("src")
            relation = item.get("relation") or item.get("predicate") or item.get("type")
            target = item.get("target") or item.get("dst") or item.get("object")
            if source and relation and target:
                relations.append(
                    ExtractedRelation(
                        source=str(source),
                        relation=str(relation),
                        target=str(target),
                        confidence=item.get("confidence"),
                        evidence=item.get("evidence"),
                        temporal=item.get("temporal") or item.get("temporal_info"),
                        is_negated=bool(item.get("is_negated", item.get("negation", False))),
                        metadata=item.get("metadata") or {},
                    )
                )
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            relations.append(
                ExtractedRelation(source=str(item[0]), relation=str(item[1]), target=str(item[2]))
            )
    return relations


def _relation_to_triple(relation: ExtractedRelation) -> tuple[str, str, str]:
    return (relation.source, relation.relation, relation.target)


def _temporal_to_dict(value: TemporalInfo | dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, TemporalInfo):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if v is not None}
    return None


def _dedupe_preserve_order(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _remap_enriched_to_triples(
    enriched_relations: list[dict[str, Any]] | None,
    triples: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """Align relation metadata with the current triple list after filtering/resolution."""
    if not triples:
        return []
    enriched_relations = enriched_relations or []
    exact = {
        rel_dict["relation"]: rel_dict
        for rel_dict in enriched_relations
        if isinstance(rel_dict, dict) and rel_dict.get("relation")
    }
    by_relation_name: dict[str, list[dict[str, Any]]] = {}
    for rel_dict in exact.values():
        rel = rel_dict.get("relation")
        if isinstance(rel, tuple) and len(rel) >= 3:
            by_relation_name.setdefault(rel[1], []).append(rel_dict)

    aligned: list[dict[str, Any]] = []
    for triple in triples:
        rel_dict = exact.get(triple)
        if rel_dict is None:
            candidates = by_relation_name.get(triple[1], [])
            rel_dict = candidates[0] if len(candidates) == 1 else None
        if rel_dict is None:
            aligned.append(
                {
                    "relation": triple,
                    "confidence": None,
                    "evidence": None,
                    "temporal": None,
                    "is_negated": False,
                    "metadata": {},
                }
            )
            continue
        copied = dict(rel_dict)
        copied["relation"] = triple
        copied.setdefault("evidence", None)
        copied.setdefault("metadata", {})
        aligned.append(copied)
    return aligned


def _filter_relation_metadata(
    triples: list[tuple[str, str, str]],
    enriched_relations: list[dict[str, Any]] | None,
    *,
    min_confidence: float | None = None,
    filter_negated: bool = True,
) -> tuple[list[tuple[str, str, str]], list[dict[str, Any]]]:
    """Apply a single relation metadata policy across single/chunk paths."""
    aligned = _remap_enriched_to_triples(enriched_relations, triples)
    kept_triples: list[tuple[str, str, str]] = []
    kept_enriched: list[dict[str, Any]] = []

    for triple, rel_dict in zip(triples, aligned, strict=False):
        if filter_negated and rel_dict.get("is_negated", False):
            logger.debug("Filtered out negated relation: %s", rel_dict.get("relation", triple))
            continue

        if min_confidence is not None:
            confidence = rel_dict.get("confidence")
            if isinstance(confidence, (int, float)) and confidence < min_confidence:
                logger.debug(
                    "Filtered out low-confidence relation %s (confidence %.3f < %.3f)",
                    rel_dict.get("relation", triple),
                    confidence,
                    min_confidence,
                )
                continue

        kept_triples.append(triple)
        kept_enriched.append(rel_dict)

    return kept_triples, kept_enriched


def _postprocess_schema_and_metadata(
    *,
    schema: DRGSchema | EnhancedDRGSchema,
    entities: list[tuple[str, str]],
    triples: list[tuple[str, str, str]],
    enriched_relations: list[dict[str, Any]] | None,
    min_confidence: float | None = None,
    filter_negated: bool = True,
    enable_reverse_relation_fallback: bool = False,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]], list[dict[str, Any]]]:
    """Shared schema validation and relation metadata filtering."""
    valid_entities, valid_triples = _filter_against_schema(
        schema=schema,
        entities_typed=entities,
        triples=triples,
        enable_reverse_relation_fallback=enable_reverse_relation_fallback,
    )
    valid_triples, valid_enriched = _filter_relation_metadata(
        valid_triples,
        enriched_relations,
        min_confidence=min_confidence,
        filter_negated=filter_negated,
    )
    return valid_entities, valid_triples, valid_enriched


def _relation_schema_triples(schema: DRGSchema | EnhancedDRGSchema) -> list[tuple[str, str, str]]:
    normalized = _normalize_schema(schema)
    return [(rel.name, rel.src, rel.dst) for rel in normalized.relations]


def _candidate_relation_entity_groups(
    schema: DRGSchema | EnhancedDRGSchema,
    entities: list[tuple[str, str]],
    *,
    max_pairs: int,
) -> list[dict[str, Any]]:
    """Build schema-valid relation candidates for windowed document extraction."""
    entities_by_type: dict[str, list[str]] = {}
    for name, etype in entities:
        if name.strip():
            entities_by_type.setdefault(etype, []).append(name)

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for relation_name, src_type, dst_type in _relation_schema_triples(schema):
        for source in entities_by_type.get(src_type, []):
            for target in entities_by_type.get(dst_type, []):
                if source == target:
                    continue
                key = (source, relation_name, target)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "relation": relation_name,
                        "source": source,
                        "source_type": src_type,
                        "target": target,
                        "target_type": dst_type,
                    }
                )
                if len(candidates) >= max_pairs:
                    return candidates
    return candidates


def _chunk_text(chunk: dict[str, Any] | str) -> str:
    return chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)


def _select_evidence_windows_for_pair(
    chunk_texts: list[str],
    *,
    source: str,
    target: str,
    max_windows: int,
) -> list[dict[str, Any]]:
    """Select compact chunks that mention both or either candidate endpoint."""
    source_l = source.lower()
    target_l = target.lower()
    both: list[dict[str, Any]] = []
    either: list[dict[str, Any]] = []

    for idx, text in enumerate(chunk_texts):
        if not text.strip():
            continue
        lowered = text.lower()
        item = {"chunk_id": idx, "text": text}
        has_source = source_l in lowered
        has_target = target_l in lowered
        if has_source and has_target:
            both.append(item)
        elif has_source or has_target:
            either.append(item)

    selected = both[:max_windows]
    if len(selected) < max_windows:
        selected.extend(either[: max_windows - len(selected)])
    if selected:
        return selected

    # Conservative fallback: give the extractor one non-empty chunk rather than
    # the whole document when string matching misses an alias.
    for idx, text in enumerate(chunk_texts):
        if text.strip():
            return [{"chunk_id": idx, "text": text}]
    return []


def _should_use_windowed_document_relations(
    *,
    chunks: list[dict[str, Any]] | list[str],
    entities: list[tuple[str, str]],
) -> bool:
    mode = os.getenv("DRG_WINDOWED_RELATION_EXTRACTION", "auto").strip().lower()
    if mode in {"1", "true", "yes", "on", "always"}:
        return True
    if mode in {"0", "false", "no", "off", "never"}:
        return False
    chunk_threshold = int(os.getenv("DRG_WINDOWED_RELATION_CHUNK_THRESHOLD", "6"))
    entity_threshold = int(os.getenv("DRG_WINDOWED_RELATION_ENTITY_THRESHOLD", "25"))
    return len(chunks) >= chunk_threshold or len(entities) >= entity_threshold


def _extract_document_relations_windowed(
    *,
    extractor: Any,
    schema: DRGSchema | EnhancedDRGSchema,
    chunks: list[dict[str, Any]] | list[str],
    entities: list[tuple[str, str]],
) -> ExtractionResult:
    """Extract document relations through schema-valid candidate windows."""
    max_pairs = int(os.getenv("DRG_MAX_RELATION_CANDIDATE_PAIRS", "160"))
    max_windows = int(os.getenv("DRG_MAX_RELATION_EVIDENCE_WINDOWS", "3"))
    chunk_texts = [_chunk_text(chunk) for chunk in chunks]
    entity_type = dict(entities)
    candidates = _candidate_relation_entity_groups(schema, entities, max_pairs=max_pairs)

    triples: list[tuple[str, str, str]] = []
    enriched: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for candidate in candidates:
        windows = _select_evidence_windows_for_pair(
            chunk_texts,
            source=candidate["source"],
            target=candidate["target"],
            max_windows=max_windows,
        )
        if not windows:
            continue

        pair_entities = _dedupe_preserve_order(
            [
                (candidate["source"], candidate["source_type"]),
                (candidate["target"], candidate["target_type"]),
            ]
        )
        throttle_llm_calls()
        result = extractor.extract_document_relations(chunks=windows, entities=pair_entities)
        result_relations = getattr(result, "relations", [])
        result_enriched = _remap_enriched_to_triples(
            getattr(result, "enriched_relations", None),
            result_relations,
        )

        for triple, rel_dict in zip(result_relations, result_enriched, strict=False):
            s, _r, o = triple
            if s not in entity_type or o not in entity_type or triple in seen:
                continue
            seen.add(triple)
            triples.append(triple)
            metadata = dict(rel_dict.get("metadata") or {})
            metadata.setdefault("source", "windowed_document_relation_extraction")
            metadata.setdefault(
                "candidate_relation",
                (candidate["source"], candidate["relation"], candidate["target"]),
            )
            metadata.setdefault("evidence_chunk_ids", [w["chunk_id"] for w in windows])
            copied = dict(rel_dict)
            copied["relation"] = triple
            copied["metadata"] = metadata
            enriched.append(copied)

    logger.info(
        "Windowed document relation extraction completed: %d candidates -> %d relations",
        len(candidates),
        len(triples),
    )
    return ExtractionResult(entities=entities, relations=triples, enriched_relations=enriched)


def _infer_implicit_relations_windowed(
    *,
    extractor: Any,
    schema: DRGSchema | EnhancedDRGSchema,
    chunks: list[dict[str, Any]] | list[str],
    entities: list[tuple[str, str]],
    existing_relations: list[tuple[str, str, str]],
) -> ExtractionResult:
    """Infer implicit relations through the same candidate-window budget."""
    max_pairs = int(os.getenv("DRG_MAX_IMPLICIT_CANDIDATE_PAIRS", "120"))
    max_windows = int(os.getenv("DRG_MAX_RELATION_EVIDENCE_WINDOWS", "3"))
    chunk_texts = [_chunk_text(chunk) for chunk in chunks]
    candidates = _candidate_relation_entity_groups(schema, entities, max_pairs=max_pairs)

    triples: list[tuple[str, str, str]] = []
    enriched: list[dict[str, Any]] = []
    seen = set(existing_relations)
    entity_type = dict(entities)

    for candidate in candidates:
        windows = _select_evidence_windows_for_pair(
            chunk_texts,
            source=candidate["source"],
            target=candidate["target"],
            max_windows=max_windows,
        )
        if not windows:
            continue
        pair_entities = [
            (candidate["source"], candidate["source_type"]),
            (candidate["target"], candidate["target_type"]),
        ]
        window_text = "\n\n".join(window["text"] for window in windows)
        throttle_llm_calls()
        result = extractor.infer_implicit_relations(
            text=window_text,
            entities=pair_entities,
            existing_relations=existing_relations,
        )
        result_relations = getattr(result, "relations", [])
        result_enriched = _remap_enriched_to_triples(
            getattr(result, "enriched_relations", None),
            result_relations,
        )
        for triple, rel_dict in zip(result_relations, result_enriched, strict=False):
            s, _r, o = triple
            if s not in entity_type or o not in entity_type or triple in seen:
                continue
            seen.add(triple)
            triples.append(triple)
            metadata = dict(rel_dict.get("metadata") or {})
            metadata.setdefault("source", "windowed_implicit_relation_inference")
            metadata.setdefault("evidence_chunk_ids", [w["chunk_id"] for w in windows])
            copied = dict(rel_dict)
            copied["relation"] = triple
            copied["metadata"] = metadata
            enriched.append(copied)

    logger.info(
        "Windowed implicit relation inference completed: %d candidates -> %d relations",
        len(candidates),
        len(triples),
    )
    return ExtractionResult(entities=entities, relations=triples, enriched_relations=enriched)


def _canonicalize_chunk_entities(
    chunk_entities_list: list[list[tuple[str, str]]],
    name_mapping: dict[str, str],
) -> list[list[tuple[str, str]]]:
    canonical_chunks: list[list[tuple[str, str]]] = []
    for entities in chunk_entities_list:
        canonical = [(name_mapping.get(name, name), etype) for name, etype in entities]
        canonical_chunks.append(_dedupe_preserve_order(canonical))
    return canonical_chunks


def _tuples_to_entity_mentions(entities: list[tuple[str, str]]) -> list[EntityMention]:
    return [EntityMention(name=name, type=etype) for name, etype in entities if name.strip()]


def _relations_to_dspy_input(relations: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    return [
        {"source": source, "relation": relation, "target": target}
        for source, relation, target in relations
    ]


def _chunks_to_dspy_input(chunks: list[dict[str, Any]] | list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            text = str(chunk.get("text", ""))
            metadata = {k: v for k, v in chunk.items() if k != "text"}
        else:
            text = str(chunk)
            metadata = {}
        out.append({"chunk_id": metadata.get("chunk_id", idx), "text": text, "metadata": metadata})
    return out


def _relations_to_extraction_result(
    *,
    entities: list[tuple[str, str]],
    relations: list[ExtractedRelation],
    metadata_source: str | None = None,
) -> ExtractionResult:
    triples = [_relation_to_triple(rel) for rel in relations]
    enriched: list[dict[str, Any]] = []
    for rel in relations:
        metadata = dict(rel.metadata)
        if metadata_source:
            metadata.setdefault("source", metadata_source)
        enriched.append(
            {
                "relation": _relation_to_triple(rel),
                "confidence": rel.confidence,
                "evidence": rel.evidence,
                "temporal": _temporal_to_dict(rel.temporal),
                "is_negated": rel.is_negated,
                "metadata": metadata,
            }
        )
    return ExtractionResult(entities=entities, relations=triples, enriched_relations=enriched)


# ---------------------------------------------------------------------------
# KGExtractor
# ---------------------------------------------------------------------------


class KGExtractor(dspy.Module):
    """DSPy module for extracting knowledge graphs from text.

    Builds entity- and relation-extraction signatures dynamically from the
    user-provided schema; the extraction itself is fully declarative.

    Dependency injection
    --------------------
    Pass ``lm=<dspy.LM-like>`` to scope extraction to a specific language model
    without touching ``dspy.settings`` globally. When ``lm`` is ``None``
    (the default), the globally configured DSPy LM is used, preserving the
    legacy behaviour and existing call sites.
    """

    def __init__(
        self,
        schema: DRGSchema | EnhancedDRGSchema,
        lm: Any | None = None,
    ):
        super().__init__()
        self.schema = schema
        self.lm = lm

        EntitySig = _create_entity_signature(schema)
        RelationSig = _create_relation_signature(schema)
        DocumentRelationSig = _create_document_relation_signature(schema)
        ImplicitRelationSig = _create_implicit_relation_signature(schema)
        CoreferenceSig = _create_coreference_signature(schema)
        self._entity_types = list(getattr(EntitySig, "_entity_types", []))
        self._relation_schema = list(getattr(RelationSig, "_relation_schema", []))

        # dspy.Predict with typed Signature fields is the standard DSPy 2.5+ pattern.
        try:
            self.entity_extractor = dspy.Predict(EntitySig)
            self.relation_extractor = dspy.Predict(RelationSig)
            self.document_relation_extractor = dspy.Predict(DocumentRelationSig)
            self.implicit_relation_extractor = dspy.Predict(ImplicitRelationSig)
            self.coreference_resolver = dspy.Predict(CoreferenceSig)
        except Exception as e:
            logger.error("Predictor initialization failed: %s", e, exc_info=True)
            raise

    def forward(
        self,
        text: str,
        context_entities: list[tuple[str, str]] | None = None,
    ) -> ExtractionResult:
        """Extract entities and relations using DSPy structured outputs.

        When ``self.lm`` is set (via constructor injection), the extraction is
        scoped to that LM; otherwise the globally configured DSPy LM is used.
        """
        with contextlib.ExitStack() as stack:
            stack.enter_context(_maybe_lm_context(self.lm))
            stack.enter_context(_maybe_json_adapter_context())
            return self._forward_impl(text, context_entities)

    def _forward_impl(
        self,
        text: str,
        context_entities: list[tuple[str, str]] | None = None,
    ) -> ExtractionResult:
        """Actual extraction logic (kept separate so ``forward`` can scope it
        with an injected LM context)."""
        logger.info("Starting entity extraction...")

        entity_result = run_predict(
            self.entity_extractor,
            salvage_fields=("entities",),
            text=text,
            entity_types=self._entity_types,
        )
        entities_raw = (
            entity_result.entities
            if isinstance(entity_result, EntityList)
            else getattr(entity_result, "entities", [])
        )
        entity_mentions = _coerce_entity_mentions(entities_raw)
        entities_list = _entity_mentions_to_tuples(entity_mentions)

        # Merge with context entities (for cross-chunk relation discovery).
        if context_entities:
            existing_entity_names = {(name.lower(), etype) for name, etype in entities_list}
            for name, etype in context_entities:
                if (name.lower(), etype) not in existing_entity_names:
                    entities_list.append((name, etype))
                    entity_mentions.append(EntityMention(name=name, type=etype))
            logger.info(
                f"Merged {len(context_entities)} context entities, "
                f"total: {len(entities_list)} entities"
            )

        if entities_list:
            entities_list = [(name, etype) for name, etype in entities_list if name.strip()]

        logger.info(f"Entity extraction complete: {len(entities_list)} entities found")

        # Relation extraction with full entity context.
        context_count = len(context_entities) if context_entities else 0
        current_count = len(entities_list) - context_count
        logger.info(
            f"Starting relation extraction: {current_count} current + "
            f"{context_count} context = {len(entities_list)} total entities."
        )

        relation_result = run_predict(
            self.relation_extractor,
            salvage_fields=("relations",),
            text=text,
            entities=_entity_mentions_to_dspy_input(entity_mentions),
            relation_schema=self._relation_schema,
        )
        relations_raw = (
            relation_result.relations
            if isinstance(relation_result, RelationList)
            else getattr(relation_result, "relations", [])
        )
        extracted_relations = _coerce_extracted_relations(relations_raw)
        relations_list = [_relation_to_triple(rel) for rel in extracted_relations]

        # Relation metadata is primarily supplied by typed DSPy outputs. The
        # heuristic fills only missing temporal/negation fields for legacy mocks.
        heur = _infer_relation_metadata_heuristic(text=text, relations=relations_list)
        temporal_info = heur.get("temporal_info")
        negations = heur.get("negations")

        logger.info(f"Relation extraction complete: {len(relations_list)} relations found")

        enriched_relations = []
        for i, rel in enumerate(extracted_relations):
            typed_temporal = _temporal_to_dict(rel.temporal)
            heuristic_temporal = temporal_info[i] if temporal_info else None
            heuristic_negated = negations[i] if negations is not None else False
            metadata = dict(rel.metadata)
            if typed_temporal is None and heuristic_temporal is not None:
                metadata.setdefault("temporal_source", "heuristic")
            if not rel.is_negated and heuristic_negated:
                metadata.setdefault("negation_source", "heuristic")
            enriched_relations.append(
                {
                    "relation": _relation_to_triple(rel),
                    "confidence": rel.confidence,
                    "evidence": rel.evidence,
                    "temporal": typed_temporal or heuristic_temporal,
                    "is_negated": rel.is_negated or heuristic_negated,
                    "metadata": metadata,
                }
            )

        if _should_return_dspy_prediction():
            return dspy.Prediction(
                entities=entities_list,
                relations=relations_list,
                enriched_relations=enriched_relations,
            )
        return ExtractionResult(
            entities=entities_list,
            relations=relations_list,
            enriched_relations=enriched_relations,
        )

    def extract_document_relations(
        self,
        *,
        chunks: list[dict[str, Any]] | list[str],
        entities: list[tuple[str, str]],
    ) -> ExtractionResult:
        """Extract relations once over the document using structured chunk inputs."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(_maybe_lm_context(self.lm))
            stack.enter_context(_maybe_json_adapter_context())
            entity_mentions = _tuples_to_entity_mentions(entities)
            result = run_predict(
                self.document_relation_extractor,
                salvage_fields=("relations",),
                document_chunks=_chunks_to_dspy_input(chunks),
                entities=_entity_mentions_to_dspy_input(entity_mentions),
                relation_schema=self._relation_schema,
            )
            raw_relations = (
                result.relations
                if isinstance(result, RelationList)
                else getattr(result, "relations", [])
            )
            return _relations_to_extraction_result(
                entities=entities,
                relations=_coerce_extracted_relations(raw_relations),
                metadata_source="document_relation_extraction",
            )

    def infer_implicit_relations(
        self,
        *,
        text: str,
        entities: list[tuple[str, str]],
        existing_relations: list[tuple[str, str, str]],
    ) -> ExtractionResult:
        """Infer implicit relations with a typed DSPy program."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(_maybe_lm_context(self.lm))
            stack.enter_context(_maybe_json_adapter_context())
            entity_mentions = _tuples_to_entity_mentions(entities)
            result = run_predict(
                self.implicit_relation_extractor,
                salvage_fields=("relations",),
                text=text,
                entities=_entity_mentions_to_dspy_input(entity_mentions),
                existing_relations=_relations_to_dspy_input(existing_relations),
                relation_schema=self._relation_schema,
            )
            raw_relations = (
                result.relations
                if isinstance(result, RelationList)
                else getattr(result, "relations", [])
            )
            return _relations_to_extraction_result(
                entities=entities,
                relations=_coerce_extracted_relations(raw_relations),
                metadata_source="implicit_relation_extraction",
            )

    def resolve_coreferences_dspy(
        self,
        *,
        text: str,
        entities: list[tuple[str, str]],
        relations: list[tuple[str, str, str]],
    ) -> ExtractionResult:
        """Resolve relation endpoints with a typed DSPy coreference pass."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(_maybe_lm_context(self.lm))
            stack.enter_context(_maybe_json_adapter_context())
            entity_mentions = _tuples_to_entity_mentions(entities)
            result = run_predict(
                self.coreference_resolver,
                salvage_fields=("resolved_relations", "relations"),
                text=text,
                entities=_entity_mentions_to_dspy_input(entity_mentions),
                relations=_relations_to_dspy_input(relations),
                relation_schema=self._relation_schema,
            )
            raw_relations = (
                getattr(result, "resolved_relations", None)
                if not isinstance(result, RelationList)
                else result.relations
            )
            if raw_relations is None:
                raw_relations = getattr(result, "relations", [])
            return _relations_to_extraction_result(
                entities=entities,
                relations=_coerce_extracted_relations(raw_relations),
                metadata_source="coreference_resolution",
            )


# ---------------------------------------------------------------------------
# LM configuration + global extractor cache
# ---------------------------------------------------------------------------

_extractor: KGExtractor | None = None


def _schema_fingerprint(schema: DRGSchema | EnhancedDRGSchema) -> str:
    """Return a stable cache key for all extraction-relevant schema metadata."""
    if isinstance(schema, EnhancedDRGSchema):
        payload = schema.to_dict()
    else:
        payload = {
            "entities": [{"name": e.name} for e in schema.entities],
            "relations": [
                {
                    "name": r.name,
                    "src": r.src,
                    "dst": r.dst,
                    "description": r.description,
                    "detail": r.detail,
                    "properties": r.properties,
                }
                for r in schema.relations
            ],
        }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _configure_llm_auto() -> None:
    """Auto-configure DSPy LM from environment variables (delegates to LMConfig)."""
    try:
        from ..config import configure_lm

        configure_lm()
    except ImportError:
        logger.warning("drg.config module not available, skipping LM configuration")


def _get_extractor(
    schema: DRGSchema | EnhancedDRGSchema,
    lm: Any | None = None,
) -> KGExtractor:
    """Return a cached `KGExtractor` for ``(schema, lm)``.

    Cache invalidation rule: rebuild when either the schema **or** the injected
    LM changes. When ``lm`` is ``None``, ``_configure_llm_auto()`` is called so
    the global DSPy state still picks up environment variables; when ``lm`` is
    provided we deliberately skip that step (the caller has already supplied
    the LM and we don't want to clobber test/global state).
    """
    global _extractor

    if lm is None:
        _configure_llm_auto()

    if _extractor is None:
        _extractor = KGExtractor(schema, lm=lm)
        return _extractor

    schema_changed = _schema_fingerprint(_extractor.schema) != _schema_fingerprint(schema)
    lm_changed = _extractor.lm is not lm

    if schema_changed or lm_changed:
        _extractor = KGExtractor(schema, lm=lm)
    return _extractor


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------


def extract_from_chunks(
    chunks: list[dict[str, Any]],
    schema: DRGSchema | EnhancedDRGSchema,
    enable_cross_chunk_relationships: bool = True,
    enable_entity_resolution: bool = True,
    enable_coreference_resolution: bool = False,
    enable_implicit_relationships: bool = True,
    enable_cross_chunk_context_snippets: bool = True,
    max_cross_chunk_context_chunks: int = 3,
    cross_chunk_snippet_chars: int = 350,
    max_cross_chunk_context_chars: int = 1200,
    min_anchor_entity_len: int = 3,
    max_anchor_entities: int = 8,
    two_pass_extraction: bool = True,
    embedding_provider: Any = None,
    return_enriched: bool = False,
    min_confidence: float | None = None,
    filter_negated: bool = True,
    lm: Any | None = None,
) -> (
    tuple[list[tuple[str, str]], list[tuple[str, str, str]]]
    | tuple[list[tuple[str, str]], list[tuple[str, str, str]], list[dict[str, Any]]]
):
    """Extract entities and relations from multiple chunks with cross-chunk support.

    See the full docstring in the package README / source history for the
    detailed behaviour of two-pass vs. single-pass mode and the cross-chunk
    context snippet mechanism.

    Args (DI):
        lm: Optional DSPy-compatible language model. When supplied, the
            underlying ``KGExtractor`` scopes all DSPy calls to this LM instead
            of reading from the global ``dspy.settings``. Default ``None``
            preserves the legacy auto-configuration behaviour.
    """
    guarded_empty = _guard_lm_or_mock_empty(
        lm=lm,
        return_enriched=return_enriched,
        operation="extract_from_chunks",
    )
    if guarded_empty is not None:
        return guarded_empty

    extractor = _get_extractor(schema, lm=lm)

    all_entities: list[tuple[str, str]] = []
    all_triples: list[tuple[str, str, str]] = []
    all_enriched: list[dict[str, Any]] = []

    if two_pass_extraction:
        logger.info("Using two-pass extraction mode")

        # PASS 1: extract all entities from all chunks.
        logger.info("Pass 1: Extracting entities from all chunks...")
        all_entities = []
        chunk_texts: list[str] = []
        chunk_entities_list: list[list[tuple[str, str]]] = []

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            if not chunk_text.strip():
                chunk_texts.append("")
                chunk_entities_list.append([])
                continue

            chunk_texts.append(chunk_text)
            chunk_log = with_context(logger, pass_="entities", chunk_id=i, total_chunks=len(chunks))
            chunk_log.info(f"Pass 1 - Processing chunk {i + 1}/{len(chunks)} for entities...")

            throttle_llm_calls()
            result = extractor(text=chunk_text)
            chunk_entities = result.entities if hasattr(result, "entities") else []
            chunk_entities_list.append(chunk_entities)

            existing_entities = {(name.lower(), etype) for name, etype in all_entities}
            for name, etype in chunk_entities:
                if (name.lower(), etype) not in existing_entities:
                    all_entities.append((name, etype))
                    existing_entities.add((name.lower(), etype))

        logger.info(f"Pass 1 complete: {len(all_entities)} unique entities extracted")

        # Reconcile globally before relation extraction so the second pass uses
        # canonical entities across all chunks instead of local surface forms.
        if enable_entity_resolution and all_entities and resolve_entities_detailed:
            try:
                detailed = resolve_entities_detailed(
                    all_entities,
                    similarity_threshold=0.65,
                    adaptive_threshold=True,
                    embedding_provider=embedding_provider,
                    use_embedding=bool(embedding_provider),
                )
                all_entities = list(detailed.entities)
                chunk_entities_list = _canonicalize_chunk_entities(
                    chunk_entities_list,
                    detailed.name_mapping,
                )
                logger.info(
                    "Global entity reconciliation before relation pass: %d canonical entities",
                    len(all_entities),
                )
            except Exception as e:
                if is_strict():
                    raise
                logger.warning("Global entity reconciliation failed: %s", e, exc_info=True)

        # Build entity-to-chunks index for cross-chunk snippet injection.
        # Used in the per-chunk Pass 2 path when enable_cross_chunk_relationships=False.
        entity_to_chunks_idx: dict[str, list[int]] = {}
        if enable_cross_chunk_context_snippets:
            for ci, chunk_ents in enumerate(chunk_entities_list):
                for cname, _ in chunk_ents:
                    entity_to_chunks_idx.setdefault(cname.lower(), []).append(ci)
            if entity_to_chunks_idx:
                logger.info(
                    "Built entity-to-chunks index for snippet injection: "
                    "%d entity entries across %d chunks.",
                    len(entity_to_chunks_idx),
                    len(chunk_texts),
                )

        # PASS 2: extract relations at document scope with canonical entities.
        logger.info(
            f"Pass 2: Extracting document relations with {len(all_entities)} global entities..."
        )
        all_triples = []

        if enable_cross_chunk_relationships:
            if not hasattr(extractor, "extract_document_relations"):
                raise ExtractionError(
                    "Extractor does not support document-level relation extraction"
                )
            if _should_use_windowed_document_relations(chunks=chunk_texts, entities=all_entities):
                document_result = _extract_document_relations_windowed(
                    extractor=extractor,
                    schema=schema,
                    chunks=chunk_texts,
                    entities=all_entities,
                )
            else:
                throttle_llm_calls()
                document_result = extractor.extract_document_relations(
                    chunks=_chunks_to_dspy_input(chunk_texts),
                    entities=all_entities,
                )
            all_triples.extend(getattr(document_result, "relations", []))
            document_enriched = getattr(document_result, "enriched_relations", None)
            if isinstance(document_enriched, list):
                all_enriched.extend(document_enriched)
        else:
            for i, chunk_text in enumerate(chunk_texts):
                if not chunk_text.strip():
                    continue

                chunk_log = with_context(
                    logger, pass_="relations", chunk_id=i, total_chunks=len(chunks)
                )
                chunk_log.info(f"Pass 2 - Processing chunk {i + 1}/{len(chunks)} for relations...")

                # Inject cross-chunk context snippets when available.
                augmented_text = chunk_text
                if enable_cross_chunk_context_snippets and entity_to_chunks_idx:
                    _anchor_ents = _select_anchor_entities(
                        chunk_text,
                        chunk_entities_list[i] if i < len(chunk_entities_list) else [],
                        entity_to_chunks_idx,
                        len(chunk_texts),
                        min_anchor_len=min_anchor_entity_len,
                        max_anchors=max_anchor_entities,
                    )
                    _snippets = _build_cross_chunk_context_snippets(
                        chunk_texts,
                        entity_to_chunks_idx,
                        _anchor_ents,
                        current_chunk_index=i,
                        max_chunks=max_cross_chunk_context_chunks,
                        snippet_chars=cross_chunk_snippet_chars,
                        max_total_chars=max_cross_chunk_context_chars,
                    )
                    if _snippets:
                        augmented_text = (
                            chunk_text + "\n\n[Cross-document context]\n" + "\n".join(_snippets)
                        )
                        logger.debug(
                            "Injected %d cross-chunk snippet(s) into chunk %d.",
                            len(_snippets),
                            i,
                        )

                throttle_llm_calls()
                result = extractor(text=augmented_text)

                chunk_relations = result.relations if hasattr(result, "relations") else []
                all_triples.extend(chunk_relations)
                chunk_enriched = getattr(result, "enriched_relations", None)
                if isinstance(chunk_enriched, list):
                    all_enriched.extend(chunk_enriched)

        logger.info(f"Pass 2 complete: {len(all_triples)} relations extracted")

    else:
        logger.info("Using single-pass extraction mode")
        context_entities: list[tuple[str, str]] = []
        # Running entity-to-chunks index and chunk text list for snippet injection.
        sp_chunk_texts: list[str] = []
        sp_entity_to_chunks: dict[str, list[int]] = {}

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            if not chunk_text.strip():
                sp_chunk_texts.append("")
                continue

            # Inject cross-chunk context snippets from previously processed chunks.
            augmented_text = chunk_text
            if enable_cross_chunk_context_snippets and sp_entity_to_chunks:
                _anchor_ents = _select_anchor_entities(
                    chunk_text,
                    context_entities,
                    sp_entity_to_chunks,
                    len(chunks),
                    min_anchor_len=min_anchor_entity_len,
                    max_anchors=max_anchor_entities,
                )
                _snippets = _build_cross_chunk_context_snippets(
                    sp_chunk_texts,
                    sp_entity_to_chunks,
                    _anchor_ents,
                    current_chunk_index=i,
                    max_chunks=max_cross_chunk_context_chunks,
                    snippet_chars=cross_chunk_snippet_chars,
                    max_total_chars=max_cross_chunk_context_chars,
                )
                if _snippets:
                    augmented_text = (
                        chunk_text + "\n\n[Cross-document context]\n" + "\n".join(_snippets)
                    )
                    logger.debug(
                        "Injected %d cross-chunk snippet(s) into single-pass chunk %d.",
                        len(_snippets),
                        i,
                    )

            logger.info(f"Processing chunk {i + 1}/{len(chunks)}...")

            throttle_llm_calls()
            result = extractor(
                text=augmented_text,
                context_entities=(
                    context_entities
                    if enable_cross_chunk_relationships and context_entities
                    else None
                ),
            )

            chunk_entities = result.entities if hasattr(result, "entities") else []
            chunk_relations = result.relations if hasattr(result, "relations") else []
            all_entities.extend(chunk_entities)
            all_triples.extend(chunk_relations)
            chunk_enriched = getattr(result, "enriched_relations", None)
            if isinstance(chunk_enriched, list):
                all_enriched.extend(chunk_enriched)

            # Update running index for subsequent chunks.
            sp_chunk_texts.append(chunk_text)
            if enable_cross_chunk_context_snippets:
                for cname, _ in chunk_entities:
                    sp_entity_to_chunks.setdefault(cname.lower(), []).append(i)

            if enable_cross_chunk_relationships:
                existing_names = {(name.lower(), etype) for name, etype in context_entities}
                for name, etype in chunk_entities:
                    if (name.lower(), etype) not in existing_names:
                        context_entities.append((name, etype))

                logger.info(
                    f"Context entities updated: {len(context_entities)} total "
                    f"(chunk {i + 1} sees entities from chunks 1-{i + 1})"
                )

        if enable_cross_chunk_relationships and all_entities:
            if not hasattr(extractor, "extract_document_relations"):
                raise ExtractionError(
                    "Extractor does not support document-level relation extraction"
                )
            document_entities = _dedupe_preserve_order(all_entities)
            document_chunks = [_chunk_text(chunk) for chunk in chunks]
            if _should_use_windowed_document_relations(
                chunks=document_chunks,
                entities=document_entities,
            ):
                document_result = _extract_document_relations_windowed(
                    extractor=extractor,
                    schema=schema,
                    chunks=document_chunks,
                    entities=document_entities,
                )
            else:
                throttle_llm_calls()
                document_result = extractor.extract_document_relations(
                    chunks=_chunks_to_dspy_input(document_chunks),
                    entities=document_entities,
                )
            all_triples.extend(getattr(document_result, "relations", []))
            document_enriched = getattr(document_result, "enriched_relations", None)
            if isinstance(document_enriched, list):
                all_enriched.extend(document_enriched)

    # Deduplicate.
    all_entities = _dedupe_preserve_order(all_entities)
    all_triples = _dedupe_preserve_order(all_triples)
    all_enriched = _remap_enriched_to_triples(all_enriched, all_triples)

    # Schema validation: keep entities/triples within the declared ontology.
    try:
        all_entities, all_triples, all_enriched = _postprocess_schema_and_metadata(
            schema=schema,
            entities=all_entities,
            triples=all_triples,
            enriched_relations=all_enriched,
            # Final relation metadata filtering runs after implicit inference.
            filter_negated=False,
        )
    except Exception as _schema_filter_exc:
        if is_strict():
            raise
        logger.warning(
            "Schema validation filter failed in extract_from_chunks: %s",
            _schema_filter_exc,
            exc_info=True,
        )

    # Post-processing: coreference resolution.
    if enable_coreference_resolution:
        try:
            full_text = "\n\n".join(
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in chunks
            )
            if not isinstance(extractor, _Mock) and hasattr(extractor, "resolve_coreferences_dspy"):
                throttle_llm_calls()
                coref_result = extractor.resolve_coreferences_dspy(
                    text=full_text,
                    entities=all_entities,
                    relations=all_triples,
                )
                all_triples = getattr(coref_result, "relations", all_triples)
                coref_enriched = getattr(coref_result, "enriched_relations", None)
                if isinstance(coref_enriched, list):
                    all_enriched = coref_enriched
            if resolve_coreferences:
                all_entities, all_triples = resolve_coreferences(
                    text=full_text,
                    entities=all_entities,
                    relations=all_triples,
                    use_nlp=True,
                    use_neural_coref=True,
                    embedding_provider=embedding_provider,
                    language=os.getenv("DRG_LANGUAGE", "en"),
                )
            all_enriched = _remap_enriched_to_triples(all_enriched, all_triples)
        except Exception as e:
            if is_strict():
                raise
            logger.warning("Coreference resolution failed: %s", e, exc_info=True)

    # Post-processing: entity resolution.
    if enable_entity_resolution and resolve_entities_and_relations:
        try:
            all_entities, all_triples = resolve_entities_and_relations(
                all_entities,
                all_triples,
                similarity_threshold=0.65,
                adaptive_threshold=True,
                embedding_provider=embedding_provider,
                use_embedding=True,
            )
            all_enriched = _remap_enriched_to_triples(all_enriched, all_triples)
        except Exception as e:
            if is_strict():
                raise
            logger.warning("Entity resolution failed: %s", e, exc_info=True)

    # DSPy implicit relationship inference (schema-gated).
    if enable_implicit_relationships and all_entities:
        try:
            full_text = "\n\n".join(
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in chunks
            )
            if isinstance(extractor, _Mock) or not hasattr(extractor, "infer_implicit_relations"):
                inferred_result = ExtractionResult(entities=all_entities, relations=[])
            elif _should_use_windowed_document_relations(chunks=chunks, entities=all_entities):
                inferred_result = _infer_implicit_relations_windowed(
                    extractor=extractor,
                    schema=schema,
                    chunks=chunks,
                    entities=all_entities,
                    existing_relations=all_triples,
                )
            else:
                throttle_llm_calls()
                inferred_result = extractor.infer_implicit_relations(
                    text=full_text,
                    entities=all_entities,
                    existing_relations=all_triples,
                )
            inferred = getattr(inferred_result, "relations", [])
            inferred_enriched = getattr(inferred_result, "enriched_relations", None)
            enriched_by_triple = {
                rel_dict.get("relation"): rel_dict
                for rel_dict in inferred_enriched or []
                if isinstance(rel_dict, dict)
            }
            if inferred:
                existing = set(all_triples)
                for t in inferred:
                    if t not in existing:
                        all_triples.append(t)
                        all_enriched.append(enriched_by_triple.get(t, {"relation": t}))
                        existing.add(t)
        except Exception as e:
            if is_strict():
                raise
            logger.debug("Implicit relationship inference failed: %s", e, exc_info=True)

    all_entities, all_triples, all_enriched = _postprocess_schema_and_metadata(
        schema=schema,
        entities=all_entities,
        triples=all_triples,
        enriched_relations=all_enriched,
        min_confidence=min_confidence,
        filter_negated=filter_negated,
    )

    # Optional hub-dominance QA gate (off by default).
    _validate_hub_dominance(all_triples)

    if return_enriched:
        return all_entities, all_triples, _remap_enriched_to_triples(all_enriched, all_triples)
    return all_entities, all_triples


def _validate_hub_dominance(all_triples: list[tuple[str, str, str]]) -> None:
    """Optional QA gate that raises if a single entity dominates the graph.

    Off by default; enable with ``DRG_VALIDATE_HUB_DOMINANCE=1``. Configure via:
        - ``DRG_HUB_VALIDATION_MODE``: "error" | "warn" (default: "error")
        - ``DRG_MAX_HUB_RATIO``: float (default: 0.30)
        - ``DRG_MIN_DIVERSITY_RATIO``: float (default: 0.50)
    """
    if not all_triples:
        return

    validate_hub = os.getenv("DRG_VALIDATE_HUB_DOMINANCE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    if not validate_hub:
        return

    entity_counts: Counter = Counter()
    for s, _r, t in all_triples:
        entity_counts[s] += 1
        entity_counts[t] += 1

    total_edges = len(all_triples)
    validation_mode = os.getenv("DRG_HUB_VALIDATION_MODE", "error").strip().lower()
    try:
        max_hub_ratio = float(os.getenv("DRG_MAX_HUB_RATIO", "0.30"))
    except ValueError:
        max_hub_ratio = 0.30
    try:
        min_diversity_ratio = float(os.getenv("DRG_MIN_DIVERSITY_RATIO", "0.50"))
    except ValueError:
        min_diversity_ratio = 0.50

    top = entity_counts.most_common(1)
    if not top:
        return
    top_entity, top_count = top[0]
    hub_ratio = top_count / total_edges
    diversity_ratio = (total_edges - top_count) / total_edges

    if hub_ratio > max_hub_ratio or diversity_ratio < min_diversity_ratio:
        msg = (
            "Hub dominance validation failed: "
            f"top_entity={top_entity}, hub_ratio={hub_ratio:.2f} "
            f"(max={max_hub_ratio:.2f}), diversity_ratio={diversity_ratio:.2f} "
            f"(min={min_diversity_ratio:.2f})."
        )
        if validation_mode == "warn":
            logger.warning(f"{msg} Proceeding (mode=warn).")
            return
        logger.error(msg)
        logger.error(
            "If this document is naturally hub-like, disable via "
            "DRG_VALIDATE_HUB_DOMINANCE=0 or set DRG_HUB_VALIDATION_MODE=warn."
        )
        raise GraphError("Hub dominance validation failed")


def extract_typed(
    text: str,
    schema: DRGSchema | EnhancedDRGSchema,
    enable_entity_resolution: bool = True,
    enable_coreference_resolution: bool = False,
    enable_implicit_relationships: bool = True,
    embedding_provider: Any = None,
    return_enriched: bool = False,
    min_confidence: float | None = None,
    filter_negated: bool = True,
    enable_reverse_relation_fallback: bool = False,
    lm: Any | None = None,
) -> (
    tuple[list[tuple[str, str]], list[tuple[str, str, str]]]
    | tuple[list[tuple[str, str]], list[tuple[str, str, str]], list[dict[str, Any]]]
):
    """Extract entities and relations from a single text chunk.

    For long texts that need chunking, prefer :func:`extract_from_chunks`,
    which enables cross-chunk relationship discovery.

    Args (DI):
        lm: Optional DSPy-compatible language model. When supplied, extraction
            runs inside a DSPy context bound to this LM, bypassing the global
            ``dspy.settings`` (useful for tests and multi-tenant workflows).
    """
    if not text or not text.strip():
        if return_enriched:
            return [], [], []
        return [], []

    # Input length guard — protects against prompt injection via oversized inputs.
    _max_chars = int(os.getenv("DRG_MAX_TEXT_CHARS", "100000"))
    if len(text) > _max_chars:
        raise ValueError(
            f"Input text is too long ({len(text):,} chars). "
            f"Maximum allowed: {_max_chars:,} chars (set DRG_MAX_TEXT_CHARS to override)."
        )

    guarded_empty = _guard_lm_or_mock_empty(
        lm=lm,
        return_enriched=return_enriched,
        operation="extract_typed",
    )
    if guarded_empty is not None:
        return guarded_empty

    extractor = _get_extractor(schema, lm=lm)

    result = extractor(text=text)

    entities_typed = result.entities if hasattr(result, "entities") and result.entities else []
    triples = result.relations if hasattr(result, "relations") and result.relations else []

    enriched_relations: list[dict[str, Any]] | None = None
    enriched_raw = getattr(result, "enriched_relations", None)
    if isinstance(enriched_raw, list) and enriched_raw:
        enriched_relations = enriched_raw

    if not isinstance(entities_typed, list):
        raise ExtractionError(
            f"Extraction returned invalid entities type: {type(entities_typed).__name__}"
        )
    if not isinstance(triples, list):
        raise ExtractionError(f"Extraction returned invalid triples type: {type(triples).__name__}")

    # Schema validation: keep only entities/relations the schema allows. Final
    # metadata filters run after optional implicit relation inference below.
    valid_entities, valid_triples, enriched_relations = _postprocess_schema_and_metadata(
        schema=schema,
        entities=entities_typed,
        triples=triples,
        enriched_relations=enriched_relations,
        filter_negated=False,
        enable_reverse_relation_fallback=enable_reverse_relation_fallback,
    )

    # Coreference resolution (BEFORE entity resolution).
    if enable_coreference_resolution and valid_entities:
        try:
            if not isinstance(extractor, _Mock) and hasattr(extractor, "resolve_coreferences_dspy"):
                coref_result = extractor.resolve_coreferences_dspy(
                    text=text,
                    entities=valid_entities,
                    relations=valid_triples,
                )
                valid_triples = getattr(coref_result, "relations", valid_triples)
                coref_enriched = getattr(coref_result, "enriched_relations", None)
                if isinstance(coref_enriched, list):
                    enriched_relations = coref_enriched
            if resolve_coreferences is None:
                logger.warning("Coreference resolution module not available, skipping resolution")
            else:
                valid_entities, valid_triples = resolve_coreferences(
                    text=text,
                    entities=valid_entities,
                    relations=valid_triples,
                    use_nlp=True,
                    embedding_provider=embedding_provider,
                    language=os.getenv("DRG_LANGUAGE", "en"),
                )
                logger.info("Coreference resolution applied successfully")
        except Exception as e:
            if is_strict():
                raise
            logger.warning(
                "Coreference resolution failed: %s, continuing without resolution",
                e,
                exc_info=True,
            )

    # Entity resolution.
    if enable_entity_resolution and valid_entities:
        if resolve_entities_and_relations is None:
            logger.warning("Entity resolution module not available, skipping resolution")
        else:
            try:
                valid_entities, valid_triples = resolve_entities_and_relations(
                    valid_entities,
                    valid_triples,
                    similarity_threshold=0.65,
                    adaptive_threshold=True,
                    embedding_provider=embedding_provider,
                    use_embedding=bool(embedding_provider),
                )
                logger.info("Entity resolution applied successfully")
            except Exception as e:
                if is_strict():
                    raise
                logger.warning(
                    "Entity resolution failed: %s, continuing without resolution",
                    e,
                    exc_info=True,
                )

    # Optional DSPy implicit relationship inference.
    if enable_implicit_relationships and valid_entities and text:
        if isinstance(extractor, _Mock) or not hasattr(extractor, "infer_implicit_relations"):
            inferred_result = ExtractionResult(entities=valid_entities, relations=[])
        else:
            inferred_result = extractor.infer_implicit_relations(
                text=text,
                entities=valid_entities,
                existing_relations=valid_triples,
            )
        inferred = getattr(inferred_result, "relations", [])
        inferred_enriched = getattr(inferred_result, "enriched_relations", None)
        enriched_by_triple = {
            rel_dict.get("relation"): rel_dict
            for rel_dict in inferred_enriched or []
            if isinstance(rel_dict, dict)
        }
        if inferred:
            if return_enriched and enriched_relations is None:
                enriched_relations = []
            existing = set(valid_triples)
            for t in inferred:
                if t not in existing:
                    valid_triples.append(t)
                    if enriched_relations is not None:
                        enriched_relations.append(enriched_by_triple.get(t, {"relation": t}))
                    existing.add(t)

    valid_entities, valid_triples, valid_enriched = _postprocess_schema_and_metadata(
        schema=schema,
        entities=valid_entities,
        triples=valid_triples,
        enriched_relations=enriched_relations,
        min_confidence=min_confidence,
        filter_negated=filter_negated,
        enable_reverse_relation_fallback=enable_reverse_relation_fallback,
    )

    if return_enriched:
        return valid_entities, valid_triples, valid_enriched
    return valid_entities, valid_triples


def _filter_against_schema(
    schema: DRGSchema | EnhancedDRGSchema,
    entities_typed: list[tuple[str, str]],
    triples: list[tuple[str, str, str]],
    *,
    enable_reverse_relation_fallback: bool = False,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Keep only entities and triples that are valid under `schema`.

    Reverse-relation conversion is opt-in. The default path keeps schema
    validation literal so hardcoded relation maps do not silently steer
    extraction correctness.
    """
    normalized = _normalize_schema(schema)
    entity_names = {e.name for e in normalized.entities}
    valid_entities = [(name, etype) for name, etype in entities_typed if etype in entity_names]

    reverse_patterns_inv = {v: k for k, v in REVERSE_RELATION_PATTERNS.items() if k != v}

    if isinstance(schema, EnhancedDRGSchema):
        return valid_entities, _filter_triples_enhanced(
            schema,
            valid_entities,
            triples,
            reverse_patterns_inv,
            enable_reverse_relation_fallback=enable_reverse_relation_fallback,
        )
    return valid_entities, _filter_triples_legacy(
        normalized,
        valid_entities,
        triples,
        reverse_patterns_inv,
        enable_reverse_relation_fallback=enable_reverse_relation_fallback,
    )


def _filter_triples_enhanced(
    schema: EnhancedDRGSchema,
    valid_entities: list[tuple[str, str]],
    triples: list[tuple[str, str, str]],
    reverse_patterns_inv: dict[str, str],
    *,
    enable_reverse_relation_fallback: bool = False,
) -> list[tuple[str, str, str]]:
    """Schema-validate triples against an `EnhancedDRGSchema` (with reverse fallback)."""
    valid_triples: list[tuple[str, str, str]] = []
    name_to_type = dict(valid_entities)

    for s, r, o in triples:
        s_type = name_to_type.get(s)
        o_type = name_to_type.get(o)
        if not (s_type and o_type):
            continue

        if schema.is_valid_relation(r, s_type, o_type):
            valid_triples.append((s, r, o))
            continue

        if not enable_reverse_relation_fallback:
            continue

        # Reverse-relation conversion via pattern table.
        if r in reverse_patterns_inv:
            reverse_rel = reverse_patterns_inv[r]
            if schema.is_valid_relation(reverse_rel, o_type, s_type):
                valid_triples.append((o, reverse_rel, s))
                logger.debug(
                    f"Converted reverse relation (pattern): ({s},{r},{o}) -> "
                    f"({o},{reverse_rel},{s})"
                )
                continue
            if reverse_rel in REVERSE_RELATION_PATTERNS:
                direct_rel = REVERSE_RELATION_PATTERNS[reverse_rel]
                if schema.is_valid_relation(direct_rel, o_type, s_type):
                    valid_triples.append((o, direct_rel, s))
                    logger.debug(
                        f"Converted reverse relation (pattern): ({s},{r},{o}) -> "
                        f"({o},{direct_rel},{s})"
                    )
                    continue

        # Generic reverse inference.
        generic_reverse_rel = _infer_reverse_relation_name(r)
        if generic_reverse_rel and generic_reverse_rel != r:
            if schema.is_valid_relation(generic_reverse_rel, o_type, s_type):
                valid_triples.append((o, generic_reverse_rel, s))
                logger.debug(
                    f"Converted reverse relation (generic): ({s},{r},{o}) -> "
                    f"({o},{generic_reverse_rel},{s})"
                )
                continue
            if r.endswith(("_by", "_of", "_from")):
                base_name = r.rsplit("_", 1)[0] if "_" in r else r
                if schema.is_valid_relation(base_name, o_type, s_type):
                    valid_triples.append((o, base_name, s))
                    logger.debug(
                        f"Converted reverse relation (suffix): ({s},{r},{o}) -> "
                        f"({o},{base_name},{s})"
                    )
                    continue

    return valid_triples


def _filter_triples_legacy(
    normalized: DRGSchema,
    valid_entities: list[tuple[str, str]],
    triples: list[tuple[str, str, str]],
    reverse_patterns_inv: dict[str, str],
    *,
    enable_reverse_relation_fallback: bool = False,
) -> list[tuple[str, str, str]]:
    """Schema-validate triples against a legacy `DRGSchema` (with reverse fallback)."""
    rel_types = {(r.src, r.name, r.dst) for r in normalized.relations}
    name_to_type = dict(valid_entities)
    valid_triples: list[tuple[str, str, str]] = []

    for s, r, o in triples:
        s_type = name_to_type.get(s)
        o_type = name_to_type.get(o)
        if not (s_type and o_type):
            continue

        if (s_type, r, o_type) in rel_types:
            valid_triples.append((s, r, o))
            continue

        if not enable_reverse_relation_fallback:
            continue

        if r in reverse_patterns_inv:
            reverse_rel = reverse_patterns_inv[r]
            if (o_type, reverse_rel, s_type) in rel_types:
                valid_triples.append((o, reverse_rel, s))
                logger.debug(
                    f"Converted reverse relation (pattern): ({s},{r},{o}) -> "
                    f"({o},{reverse_rel},{s})"
                )
                continue
            if reverse_rel in REVERSE_RELATION_PATTERNS:
                direct_rel = REVERSE_RELATION_PATTERNS[reverse_rel]
                if (o_type, direct_rel, s_type) in rel_types:
                    valid_triples.append((o, direct_rel, s))
                    logger.debug(
                        f"Converted reverse relation (pattern): ({s},{r},{o}) -> "
                        f"({o},{direct_rel},{s})"
                    )
                    continue

        generic_reverse_rel = _infer_reverse_relation_name(r)
        if generic_reverse_rel and generic_reverse_rel != r:
            if (o_type, generic_reverse_rel, s_type) in rel_types:
                valid_triples.append((o, generic_reverse_rel, s))
                logger.debug(
                    f"Converted reverse relation (generic): ({s},{r},{o}) -> "
                    f"({o},{generic_reverse_rel},{s})"
                )
                continue
            if r.endswith(("_by", "_of", "_from")):
                base_name = r.rsplit("_", 1)[0] if "_" in r else r
                if (o_type, base_name, s_type) in rel_types:
                    valid_triples.append((o, base_name, s))
                    logger.debug(
                        f"Converted reverse relation (suffix): ({s},{r},{o}) -> "
                        f"({o},{base_name},{s})"
                    )
                    continue

    return valid_triples


def extract_triples(
    text: str,
    schema: DRGSchema | EnhancedDRGSchema,
) -> list[tuple[str, str, str]]:
    """Backward-compatible wrapper: return only the triples list."""
    _, triples = extract_typed(text, schema)
    return triples


def create_kgedge_from_triple(
    triple: tuple[str, str, str],
    enriched_metadata: dict[str, Any] | None = None,
    relationship_detail: str | None = None,
) -> Any:
    """Create a `KGEdge` from a triple, optionally enriched with temporal metadata.

    Returns:
        A `drg.graph.kg_core.KGEdge` instance.
    """
    from ..graph.kg_core import KGEdge  # lazy import — avoids heavy graph deps at module load

    source, relation, target = triple
    if relationship_detail is None:
        relationship_detail = f"{source} {relation} {target}"

    start_time: str | None = None
    end_time: str | None = None
    if enriched_metadata and enriched_metadata.get("temporal"):
        temporal = enriched_metadata["temporal"]
        if isinstance(temporal, dict):
            start_time = temporal.get("valid_from") or temporal.get("start")
            end_time = temporal.get("valid_to") or temporal.get("end")

    confidence = enriched_metadata.get("confidence") if enriched_metadata else None
    is_negated = enriched_metadata.get("is_negated", False) if enriched_metadata else False

    return KGEdge(
        source=source,
        target=target,
        relationship_type=relation,
        relationship_detail=relationship_detail,
        start_time=start_time,
        end_time=end_time,
        confidence=confidence,
        is_negated=is_negated,
    )


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


async def extract_typed_async(
    text: str,
    schema: DRGSchema | EnhancedDRGSchema,
    **kwargs: Any,
) -> (
    tuple[list[tuple[str, str]], list[tuple[str, str, str]]]
    | tuple[list[tuple[str, str]], list[tuple[str, str, str]], list[dict[str, Any]]]
):
    """Async version of :func:`extract_typed`.

    Runs the synchronous extraction in a thread pool so the event loop is not
    blocked during LLM calls.  All keyword arguments are forwarded to
    :func:`extract_typed`.

    Example::

        entities, triples = await extract_typed_async(text, schema)
    """
    return await _asyncio.to_thread(_functools.partial(extract_typed, text, schema, **kwargs))


async def extract_from_chunks_async(
    chunks: list[dict[str, Any]],
    schema: DRGSchema | EnhancedDRGSchema,
    **kwargs: Any,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Async version of :func:`extract_from_chunks`.

    Runs the synchronous multi-chunk extraction in a thread pool.
    All keyword arguments are forwarded to :func:`extract_from_chunks`.

    Example::

        entities, triples = await extract_from_chunks_async(chunks, schema)
    """
    return await _asyncio.to_thread(
        _functools.partial(extract_from_chunks, chunks, schema, **kwargs)
    )
