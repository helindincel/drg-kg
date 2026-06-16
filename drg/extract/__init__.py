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
from ._chunk_context import (
    _build_cross_chunk_context_snippets,
    _select_anchor_entities,
)
from ._heuristics import (
    _infer_implicit_relations,
    _infer_relation_metadata_heuristic,
)
from ._parsing import _parse_json_output
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
from ._signatures import _create_entity_signature, _create_relation_signature
from ._types import (
    EntityList,
    ExtractionResult,
    RelationList,
    SchemaOutput,
)

logger = get_logger(__name__)

# Lazy optional imports — also exposed for test patching at this namespace.
try:
    from ..optimizer import DRGOptimizer, OptimizerConfig
except ImportError:
    DRGOptimizer = None
    OptimizerConfig = None

try:
    from ..coreference_resolution import resolve_coreferences
except ImportError:
    resolve_coreferences = None

try:
    from ..entity_resolution import resolve_entities_and_relations
except ImportError:
    resolve_entities_and_relations = None


__all__ = [
    "EntityList",
    # Types
    "ExtractionResult",
    # Public API
    "KGExtractor",
    "RelationList",
    "SchemaGeneration",
    "SchemaOutput",
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

        # Prefer TypedPredictor (DSPy 2.5+); fall back to Predict otherwise.
        try:
            if hasattr(dspy, "TypedPredictor"):
                self.entity_extractor = dspy.TypedPredictor(EntitySig, output_type=EntityList)
                self.relation_extractor = dspy.TypedPredictor(RelationSig, output_type=RelationList)
                self._use_typed_predictor = True
            else:
                logger.warning("TypedPredictor not available, falling back to Predict")
                self.entity_extractor = dspy.Predict(EntitySig)
                self.relation_extractor = dspy.Predict(RelationSig)
                self._use_typed_predictor = False
        except Exception as e:
            if is_strict():
                raise
            logger.warning(
                "TypedPredictor initialization failed: %s, using Predict",
                e,
                exc_info=True,
            )
            self.entity_extractor = dspy.Predict(EntitySig)
            self.relation_extractor = dspy.Predict(RelationSig)
            self._use_typed_predictor = False

    def forward(
        self,
        text: str,
        context_entities: list[tuple[str, str]] | None = None,
    ) -> ExtractionResult:
        """Extract entities and relations using DSPy TypedPredictor.

        When ``self.lm`` is set (via constructor injection), the extraction is
        scoped to that LM; otherwise the globally configured DSPy LM is used.
        """
        with _maybe_lm_context(self.lm):
            return self._forward_impl(text, context_entities)

    def _forward_impl(
        self,
        text: str,
        context_entities: list[tuple[str, str]] | None = None,
    ) -> ExtractionResult:
        """Actual extraction logic (kept separate so ``forward`` can scope it
        with an injected LM context)."""
        logger.info("Starting entity extraction...")

        if self._use_typed_predictor:
            entity_result = self.entity_extractor(text=text)
            if isinstance(entity_result, EntityList):
                entities_list = entity_result.entities
            else:
                entities_list = getattr(entity_result, "entities", [])
                logger.warning(f"Expected EntityList, got {type(entity_result).__name__}")

            if not isinstance(entities_list, list):
                error_msg = f"Expected list, got {type(entities_list).__name__}"
                logger.error(f"Entity extraction: {error_msg}")
                raise ExtractionError(f"Entity extraction returned invalid type: {error_msg}")
        else:
            entity_result = self.entity_extractor(text=text)
            entities_raw = getattr(entity_result, "entities", "[]")

            if isinstance(entities_raw, list):
                entities_list = [
                    (str(item[0]), str(item[1]))
                    for item in entities_raw
                    if isinstance(item, (list, tuple)) and len(item) >= 2
                ]
            elif isinstance(entities_raw, str):
                try:
                    parsed = _parse_json_output(entities_raw, expected_format="array")
                    entities_list = [
                        (str(item[0]), str(item[1]))
                        for item in parsed
                        if isinstance(item, (list, tuple)) and len(item) >= 2
                    ]
                except ValueError as e:
                    logger.error(f"Entity extraction JSON parsing failed: {e}")
                    raise ExtractionError(f"Failed to parse entity extraction output: {e}") from e
            else:
                logger.error(f"Entity extraction returned unexpected type: {type(entities_raw)}")
                raise ExtractionError(
                    f"Entity extraction returned invalid type: {type(entities_raw)}"
                )

        # Merge with context entities (for cross-chunk relation discovery).
        if context_entities:
            existing_entity_names = {(name.lower(), etype) for name, etype in entities_list}
            for name, etype in context_entities:
                if (name.lower(), etype) not in existing_entity_names:
                    entities_list.append((name, etype))
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

        if self._use_typed_predictor:
            relation_result = self.relation_extractor(text=text, entities=entities_list)
            if isinstance(relation_result, RelationList):
                relations_list = relation_result.relations
            else:
                relations_list = getattr(relation_result, "relations", [])
                logger.warning(f"Expected RelationList, got {type(relation_result).__name__}")

            if not isinstance(relations_list, list):
                error_msg = f"Expected list, got {type(relations_list).__name__}"
                logger.error(f"Relation extraction: {error_msg}")
                raise ExtractionError(f"Relation extraction returned invalid type: {error_msg}")
        else:
            entities_json = json.dumps(entities_list)
            relation_result = self.relation_extractor(text=text, entities=entities_json)
            relations_raw = getattr(relation_result, "relations", "[]")

            if isinstance(relations_raw, list):
                relations_list = [
                    (str(item[0]), str(item[1]), str(item[2]))
                    for item in relations_raw
                    if isinstance(item, (list, tuple)) and len(item) >= 3
                ]
            elif isinstance(relations_raw, str):
                try:
                    parsed = _parse_json_output(relations_raw, expected_format="array")
                    relations_list = [
                        (str(item[0]), str(item[1]), str(item[2]))
                        for item in parsed
                        if isinstance(item, (list, tuple)) and len(item) >= 3
                    ]
                except ValueError as e:
                    logger.error(f"Relation extraction JSON parsing failed: {e}")
                    raise ExtractionError(f"Failed to parse relation extraction output: {e}") from e
            else:
                logger.error(f"Relation extraction returned unexpected type: {type(relations_raw)}")
                raise ExtractionError(
                    f"Relation extraction returned invalid type: {type(relations_raw)}"
                )

        # Deterministic heuristics for negation/temporal (LLM provides only triples).
        heur = _infer_relation_metadata_heuristic(text=text, relations=relations_list)
        confidence_scores = None
        temporal_info = heur.get("temporal_info")
        negations = heur.get("negations")

        logger.info(f"Relation extraction complete: {len(relations_list)} relations found")

        enriched_relations = [
            {
                "relation": rel,
                "confidence": confidence_scores[i] if confidence_scores else None,
                "temporal": temporal_info[i] if temporal_info else None,
                "is_negated": negations[i] if negations is not None else False,
            }
            for i, rel in enumerate(relations_list)
        ]

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


# ---------------------------------------------------------------------------
# LM configuration + global extractor cache
# ---------------------------------------------------------------------------

_extractor: KGExtractor | None = None


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

    normalized_old = _normalize_schema(_extractor.schema)
    normalized_new = _normalize_schema(schema)

    old_entities = {e.name for e in normalized_old.entities}
    old_relations = {(r.name, r.src, r.dst) for r in normalized_old.relations}
    new_entities = {e.name for e in normalized_new.entities}
    new_relations = {(r.name, r.src, r.dst) for r in normalized_new.relations}

    schema_changed = old_entities != new_entities or old_relations != new_relations
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
    lm: Any | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
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
    extractor = _get_extractor(schema, lm=lm)

    # Mock-mode short-circuit: if no LM is configured, return empty results.
    effective_lm = lm if lm is not None else getattr(getattr(dspy, "settings", None), "lm", None)
    if effective_lm is None and isinstance(extractor, KGExtractor):
        logger.warning("No DSPy LM configured; returning empty extraction (mock mode).")
        return [], []

    if two_pass_extraction:
        logger.info("Using two-pass extraction mode")

        # PASS 1: extract all entities from all chunks.
        logger.info("Pass 1: Extracting entities from all chunks...")
        all_entities: list[tuple[str, str]] = []
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

        # Build entity -> chunk indices map for deterministic context snippets.
        entity_to_chunks: dict[str, list[int]] = {}
        for idx, ents in enumerate(chunk_entities_list):
            for name, _ in ents:
                if not name:
                    continue
                key = name.lower()
                entity_to_chunks.setdefault(key, [])
                if not entity_to_chunks[key] or entity_to_chunks[key][-1] != idx:
                    entity_to_chunks[key].append(idx)

        # PASS 2: extract relations with global entity context.
        logger.info(f"Pass 2: Extracting relations with {len(all_entities)} global entities...")
        all_triples: list[tuple[str, str, str]] = []

        for i, chunk_text in enumerate(chunk_texts):
            if not chunk_text.strip():
                continue

            chunk_log = with_context(
                logger, pass_="relations", chunk_id=i, total_chunks=len(chunks)
            )
            chunk_log.info(f"Pass 2 - Processing chunk {i + 1}/{len(chunks)} for relations...")

            # Deterministic intra-document evidence injection — NOT retrieval/RAG.
            augmented_text = chunk_text
            if (
                enable_cross_chunk_relationships
                and enable_cross_chunk_context_snippets
                and max_cross_chunk_context_chunks > 0
            ):
                current_entities = _select_anchor_entities(
                    chunk_text=chunk_text,
                    chunk_entities=chunk_entities_list[i],
                    entity_to_chunks=entity_to_chunks,
                    total_chunks=len(chunk_texts),
                    min_anchor_len=min_anchor_entity_len,
                    max_anchors=max_anchor_entities,
                )
                snippets = _build_cross_chunk_context_snippets(
                    chunk_texts=chunk_texts,
                    entity_to_chunks=entity_to_chunks,
                    anchor_entities=current_entities,
                    current_chunk_index=i,
                    max_chunks=max_cross_chunk_context_chunks,
                    snippet_chars=cross_chunk_snippet_chars,
                    max_total_chars=max_cross_chunk_context_chars,
                    min_anchor_len=min_anchor_entity_len,
                )
                if snippets:
                    augmented_text = (
                        "[CROSS-CHUNK CONTEXT]\n"
                        + "\n\n".join(snippets)
                        + "\n\n[CURRENT CHUNK]\n"
                        + chunk_text
                    )

            if enable_cross_chunk_relationships:
                throttle_llm_calls()
                result = extractor(text=augmented_text, context_entities=all_entities)
            else:
                throttle_llm_calls()
                result = extractor(text=chunk_text)

            chunk_relations = result.relations if hasattr(result, "relations") else []
            all_triples.extend(chunk_relations)

        logger.info(f"Pass 2 complete: {len(all_triples)} relations extracted")

    else:
        logger.info("Using single-pass extraction mode")
        all_entities: list[tuple[str, str]] = []
        all_triples: list[tuple[str, str, str]] = []
        context_entities: list[tuple[str, str]] = []

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            if not chunk_text.strip():
                continue

            logger.info(f"Processing chunk {i + 1}/{len(chunks)}...")

            if enable_cross_chunk_relationships and context_entities:
                throttle_llm_calls()
                result = extractor(text=chunk_text, context_entities=context_entities)
            else:
                throttle_llm_calls()
                result = extractor(text=chunk_text)

            chunk_entities = result.entities if hasattr(result, "entities") else []
            chunk_relations = result.relations if hasattr(result, "relations") else []
            all_entities.extend(chunk_entities)
            all_triples.extend(chunk_relations)

            if enable_cross_chunk_relationships:
                existing_names = {(name.lower(), etype) for name, etype in context_entities}
                for name, etype in chunk_entities:
                    if (name.lower(), etype) not in existing_names:
                        context_entities.append((name, etype))

                logger.info(
                    f"Context entities updated: {len(context_entities)} total "
                    f"(chunk {i + 1} sees entities from chunks 1-{i + 1})"
                )

    # Deduplicate.
    all_entities = list(set(all_entities))
    all_triples = list(set(all_triples))

    # Post-processing: coreference resolution.
    if enable_coreference_resolution and resolve_coreferences:
        try:
            full_text = "\n\n".join(
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in chunks
            )
            all_entities, all_triples = resolve_coreferences(
                text=full_text,
                entities=all_entities,
                relations=all_triples,
                use_nlp=True,
                use_neural_coref=True,
                embedding_provider=embedding_provider,
                language=os.getenv("DRG_LANGUAGE", "en"),
            )
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
        except Exception as e:
            if is_strict():
                raise
            logger.warning("Entity resolution failed: %s", e, exc_info=True)

    # Deterministic implicit relationship inference (schema-gated).
    if enable_implicit_relationships and all_entities:
        try:
            full_text = "\n\n".join(
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in chunks
            )
            inferred = _infer_implicit_relations(
                text=full_text,
                entities=all_entities,
                schema=schema,
                existing_triples=all_triples,
            )
            if inferred:
                existing = set(all_triples)
                for t in inferred:
                    if t not in existing:
                        all_triples.append(t)
                        existing.add(t)
        except Exception as e:
            if is_strict():
                raise
            logger.debug("Implicit relationship inference failed: %s", e, exc_info=True)

    # Optional hub-dominance QA gate (off by default).
    _validate_hub_dominance(all_triples)

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
    use_optimizer: bool = False,
    optimizer_config: Any | None = None,
    training_examples: list[dict[str, Any]] | None = None,
    return_enriched: bool = False,
    min_confidence: float | None = None,
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

    extractor = _get_extractor(schema, lm=lm)

    # Mock-mode short-circuit: if no LM is available (neither injected nor
    # globally configured) and the extractor is real, return empty extraction
    # instead of crashing on the DSPy call.
    effective_lm = lm if lm is not None else getattr(getattr(dspy, "settings", None), "lm", None)
    if effective_lm is None and isinstance(extractor, KGExtractor):
        if os.getenv("DRG_REQUIRE_LM", "").lower() in {"1", "true", "yes"}:
            raise LLMConfigError(
                "No DSPy LM is loaded. Configure LM via environment variables "
                "(e.g., DRG_MODEL + API key) or unset DRG_REQUIRE_LM to allow "
                "mock-mode empty extraction."
            )
        logger.warning("No DSPy LM configured; returning empty extraction (mock mode).")
        if return_enriched:
            return [], [], []
        return [], []

    # Optimizer path. Note: optimizer failures are always raised — disable
    # optimization via `use_optimizer=False` if you want to ignore them.
    if use_optimizer and training_examples:
        if DRGOptimizer is None or OptimizerConfig is None:
            logger.warning("Optimizer module not available, falling back to base extractor")
        else:
            try:
                if optimizer_config is None:
                    optimizer_config = OptimizerConfig()
                optimizer = DRGOptimizer(
                    schema=schema,
                    config=optimizer_config,
                    training_examples=training_examples,
                )
                logger.info(
                    f"Optimizing extractor with {len(training_examples)} training examples..."
                )
                extractor = optimizer.optimize()
                logger.info("Optimization completed, using optimized extractor")
            except Exception as e:
                logger.error("Optimizer failed: %s", e, exc_info=True)
                raise ExtractionError(
                    f"Optimizer optimization failed: {e}. "
                    "If you want to proceed without optimization, set use_optimizer=False. "
                    "Otherwise, check your training examples format and optimizer configuration."
                ) from e

    result = extractor(text=text)

    entities_typed = result.entities if hasattr(result, "entities") and result.entities else []
    triples = result.relations if hasattr(result, "relations") and result.relations else []

    # Enriched relation filtering (negation + min_confidence).
    enriched_relations: list[dict[str, Any]] | None = None
    enriched_raw = getattr(result, "enriched_relations", None)
    if isinstance(enriched_raw, list) and enriched_raw:
        enriched_relations = enriched_raw
        filtered_triples: list[tuple[str, str, str]] = []
        filtered_enriched: list[dict[str, Any]] = []
        for i, rel_dict in enumerate(enriched_relations):
            if rel_dict.get("is_negated", False):
                logger.debug(f"Filtered out negated relation: {rel_dict.get('relation')}")
                continue

            if min_confidence is not None:
                confidence = rel_dict.get("confidence")
                if confidence is None:
                    logger.debug(f"No confidence score for {rel_dict.get('relation')}, keeping")
                elif confidence < min_confidence:
                    logger.debug(
                        f"Filtered out low-confidence relation {rel_dict.get('relation')} "
                        f"(confidence: {confidence:.2f} < {min_confidence:.2f})"
                    )
                    continue

            filtered_triples.append(triples[i] if i < len(triples) else rel_dict["relation"])
            filtered_enriched.append(rel_dict)

        triples = filtered_triples
        enriched_relations = filtered_enriched

        if min_confidence is not None:
            filtered_count = len(enriched_raw) - len(filtered_enriched)
            if filtered_count > 0:
                logger.info(
                    f"Confidence filtering: {filtered_count} relationships filtered out "
                    f"(confidence < {min_confidence:.2f}), {len(filtered_enriched)} remaining"
                )

    if not isinstance(entities_typed, list):
        raise ExtractionError(
            f"Extraction returned invalid entities type: {type(entities_typed).__name__}"
        )
    if not isinstance(triples, list):
        raise ExtractionError(f"Extraction returned invalid triples type: {type(triples).__name__}")

    # Schema validation: keep only entities/relations the schema allows.
    valid_entities, valid_triples = _filter_against_schema(
        schema=schema,
        entities_typed=entities_typed,
        triples=triples,
    )

    # Coreference resolution (BEFORE entity resolution).
    if enable_coreference_resolution and valid_entities:
        if resolve_coreferences is None:
            logger.warning("Coreference resolution module not available, skipping resolution")
        else:
            try:
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

    # Optional deterministic implicit relationship inference.
    if enable_implicit_relationships and valid_entities and text:
        inferred = _infer_implicit_relations(
            text=text,
            entities=valid_entities,
            schema=schema,
            existing_triples=valid_triples,
        )
        if inferred:
            existing = set(valid_triples)
            for t in inferred:
                if t not in existing:
                    valid_triples.append(t)
                    existing.add(t)

    # Map enriched_relations to valid triples (after schema validation, before resolution).
    valid_enriched: list[dict[str, Any]] = []
    if return_enriched and enriched_relations:
        triple_to_enriched = {
            rel_dict["relation"]: rel_dict
            for rel_dict in enriched_relations
            if rel_dict.get("relation")
        }
        for triple in valid_triples:
            if triple in triple_to_enriched:
                valid_enriched.append(triple_to_enriched[triple])
            else:
                valid_enriched.append(
                    {
                        "relation": triple,
                        "confidence": None,
                        "temporal": None,
                        "is_negated": False,
                    }
                )

    if return_enriched:
        return valid_entities, valid_triples, valid_enriched
    return valid_entities, valid_triples


def _filter_against_schema(
    schema: DRGSchema | EnhancedDRGSchema,
    entities_typed: list[tuple[str, str]],
    triples: list[tuple[str, str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Keep only entities and triples that are valid under `schema`.

    Handles reverse-relation conversion using `REVERSE_RELATION_PATTERNS` and
    `_infer_reverse_relation_name` (domain-agnostic).
    """
    normalized = _normalize_schema(schema)
    entity_names = {e.name for e in normalized.entities}
    valid_entities = [(name, etype) for name, etype in entities_typed if etype in entity_names]

    reverse_patterns_inv = {v: k for k, v in REVERSE_RELATION_PATTERNS.items() if k != v}

    if isinstance(schema, EnhancedDRGSchema):
        return valid_entities, _filter_triples_enhanced(
            schema, valid_entities, triples, reverse_patterns_inv
        )
    return valid_entities, _filter_triples_legacy(
        normalized, valid_entities, triples, reverse_patterns_inv
    )


def _filter_triples_enhanced(
    schema: EnhancedDRGSchema,
    valid_entities: list[tuple[str, str]],
    triples: list[tuple[str, str, str]],
    reverse_patterns_inv: dict[str, str],
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
            start_time = temporal.get("start")
            end_time = temporal.get("end")

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
