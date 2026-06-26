"""Automatic schema generation from text via DSPy.

Two responsibilities:
    1. Sample a long input deterministically so it fits the LLM context budget.
       (`_sample_text_for_schema_generation`)
    2. Run a DSPy program that returns a structured `EnhancedDRGSchema` JSON.
       (`generate_schema_from_text`)
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any
from unittest.mock import Mock as _Mock

import dspy

from ..errors import SchemaGenerationError
from ..schema import EnhancedDRGSchema, EntityType, Relation, RelationGroup
from ..utils.llm_throttle import throttle_llm_calls
from ._parsing import _parse_json_output
from ._types import SchemaEntityType, SchemaOutput, SchemaRelationGroup

logger = logging.getLogger(__name__)


def _maybe_json_adapter_context():
    """Use DSPy 3's JSONAdapter for typed schema-generation outputs when available."""
    json_adapter_cls = getattr(dspy, "JSONAdapter", None)
    if json_adapter_cls is None or isinstance(json_adapter_cls, _Mock):
        return contextlib.nullcontext()

    settings = getattr(dspy, "settings", None)
    if settings is not None and getattr(settings, "adapter", None) is not None:
        return contextlib.nullcontext()

    try:
        adapter = json_adapter_cls()
    except Exception:
        return contextlib.nullcontext()

    ctx_factory = getattr(dspy, "context", None)
    if ctx_factory is not None and not isinstance(ctx_factory, _Mock):
        try:
            return ctx_factory(adapter=adapter)
        except TypeError:
            pass

    if settings is not None:
        sub_ctx = getattr(settings, "context", None)
        if sub_ctx is not None and not isinstance(sub_ctx, _Mock):
            try:
                return sub_ctx(adapter=adapter)
            except TypeError:
                pass

    return contextlib.nullcontext()


class SchemaGeneration(dspy.Signature):
    """Generate EnhancedDRGSchema from the given text.

    Build a compact, reusable ontology for knowledge graph extraction by
    reasoning from the semantics of the input text. The ontology must remain
    declarative, domain-agnostic, and zero-shot: do not use hard-coded entity
    types, relation types, properties, domain rules, or examples. Derive every
    type and relation from reusable semantic patterns observed in the text, and
    optimize for the smallest ontology that can accurately generalize to unseen
    documents with similar semantic structure.

    Semantic completeness and recall:
    - Favor semantic coverage of important roles, interactions, events,
      attributes, and temporal meanings over a minimal classification-only
      schema. The ontology should make important extractable information visible
      to downstream knowledge graph construction rather than silently omitting
      it.
    - Keep the ontology compact, but do not merge or discard distinctions that
      would cause systematic loss of meaning during extraction.

    Extractability:
    - Include ontology elements only when they can be reliably recognized and
      extracted from text using semantic evidence, not merely because they are
      theoretically meaningful.
    - Prefer definitions whose boundaries, expected evidence, and reusable
      extraction cues are clear enough to work across different documents with
      similar patterns.

    Canonical representation:
    - Prefer a single canonical representation for each semantic fact. Avoid
      modeling the same information redundantly as both entity types,
      properties, and relations unless the distinction carries independent
      semantic meaning.

    Entity type discovery:
    - Prefer reusable semantic roles over surface nouns or one-off labels.
    - Avoid generic catch-all categories when the text supports a clearer
      abstraction; split only when the distinction changes expected properties,
      relations, behavior, or graph structure.
    - Distinguish autonomous actors from passive objects by observed semantic
      behavior such as independent action, decision making, planning,
      communication, coordination, or responsibility.
    - Represent significant events as first-class entity types when they have
      participants, temporal ordering, causes, consequences, locations, state
      changes, or multiple relationships. Events need not be explicitly named.
    - Model abstract concepts with meaningful reusable distinctions when the
      text differentiates roles such as ideas, states, qualities, processes,
      plans, obligations, claims, methods, or values; do not collapse distinct
      abstractions into a single overloaded type.
    - Introduce hierarchical abstractions only when they help reuse: parent
      concepts should explain shared semantics across child types without
      over-fragmenting the ontology.

    Relation discovery:
    - Define relations as recurring semantic interactions between entity types,
      not merely lexical verb mappings.
    - Include structural, organizational, causal, spatial, dependency,
      participation, communication, containment, functional, temporal, and
      lifecycle relationships whenever justified by the document semantics.
    - Prefer relation names that describe reusable graph semantics across
      documents; avoid document-specific phrasing and overly broad relations
      that hide multiple distinct interactions.
    - Relation source and target must each be one of the generated entity type
      names. Do not use primitive value types such as string, number, date, or
      boolean as relation endpoints; represent those values as entity or
      relation properties unless they are semantically independent entities.
    - Use relation groups to organize related interactions by semantic function.

    Property discovery:
    - Add reusable entity properties when information is intrinsic to instances
      of a type rather than a relationship to another entity.
    - Add reusable relation properties when qualifiers such as strength, role,
      manner, status, scope, confidence-bearing wording, temporal validity, or
      other modifiers belong to an interaction rather than either endpoint.
    - Do not turn every attribute-like phrase into a property; include properties
      that improve repeated extraction and downstream graph utility.

    Temporal modeling:
    - Preserve semantically important dates, durations, periods, versions,
      ordering, lifecycle stages, recurrence, deadlines, and validity windows
      through event types, temporal relations, or relation/entity properties as
      appropriate.
    - Do not omit temporal semantics when they change the meaning of an entity,
      event, state, or relation.

    Before returning the ontology, internally validate that important semantic
    roles are represented; recurring interactions have reusable relation types;
    major events are modeled when appropriate; reusable entity and relation
    properties have been discovered where applicable; temporal semantics are not
    omitted; semantic completeness has not been sacrificed in ways that would
    reduce extraction recall; each ontology element is reliably extractable from
    textual evidence; semantic facts are not redundantly modeled across entity
    types, properties, and relations without independent meaning; generic
    categories are not semantically overloaded; every relation endpoint is a
    generated entity type rather than a primitive property value; and the
    ontology generalizes beyond the current document instead of overfitting to
    document-specific wording.

    The output is intentionally split into typed fields instead of one JSON blob.
    """

    text: str = dspy.InputField(desc="Input text")
    entity_types: list[SchemaEntityType] = dspy.OutputField(desc="Generated entity types")
    relation_groups: list[SchemaRelationGroup] = dspy.OutputField(desc="Generated relation groups")


def generate_schema_from_text(text: str) -> EnhancedDRGSchema:
    """Generate an `EnhancedDRGSchema` from input text via DSPy.

    Domain-agnostic: works across domains (technology, business, science,
    medicine, history, literature, etc.) and across input types (articles,
    reports, books, transcripts).

    For long inputs, applies intelligent sampling (see
    `_sample_text_for_schema_generation`) before invoking the LLM.

    NOTE: This function relies on DSPy's `Predict` with typed Signature fields
    for structured output. It does NOT implement application-level retries;
    configure retries via DSPy LM settings if needed.

    Raises:
        ValueError: If the input text exceeds the configured character limit
            (``DRG_MAX_TEXT_CHARS``, default 100 000).
        RuntimeError: If LLM invocation, JSON parsing, or schema validation
            fails. The error message contains diagnostic info for the user.
    """
    # Input length guard — protects against prompt injection via oversized inputs.
    _max_chars = int(os.getenv("DRG_MAX_TEXT_CHARS", "100000"))
    if len(text) > _max_chars:
        raise ValueError(
            f"Input text is too long ({len(text):,} chars). "
            f"Maximum allowed: {_max_chars:,} chars (set DRG_MAX_TEXT_CHARS to override)."
        )

    # Lazy import to avoid circular dependency with the package __init__.
    from . import _configure_llm_auto

    _configure_llm_auto()

    sample_text = _sample_text_for_schema_generation(text)

    try:
        schema_generator = dspy.Predict(SchemaGeneration)
        throttle_llm_calls()
        with _maybe_json_adapter_context():
            schema_result = schema_generator(text=sample_text)
        schema_data = _schema_prediction_to_dict(schema_result)
        logger.info("Schema generation completed")
    except Exception as e:
        logger.error(f"Schema generation failed: {e}")
        raise SchemaGenerationError(
            f"Schema generation failed: {e}. Check your LLM configuration and API keys."
        ) from e

    if not schema_data or (isinstance(schema_data, dict) and not schema_data):
        logger.error("Parsed schema JSON is empty")
        logger.error(f"Schema data type: {type(schema_data)}, value: {schema_data}")
        raise SchemaGenerationError(
            "Schema generation returned empty schema. "
            "The LLM may need a different typed-output configuration."
        )

    # Parse / validate schema strictly.
    try:
        schema = EnhancedDRGSchema.from_dict(schema_data)
    except Exception as e:
        # Backward-compatible conversion from legacy schema shape if present.
        if (
            isinstance(schema_data, dict)
            and "entities" in schema_data
            and "relations" in schema_data
        ):
            entity_types = [
                EntityType(
                    name=e_dict["name"],
                    description=e_dict.get("description") or "Auto-generated entity type",
                    examples=(
                        e_dict.get("examples", [])
                        if isinstance(e_dict.get("examples", []), list)
                        else []
                    ),
                    properties=(
                        e_dict.get("properties", {})
                        if isinstance(e_dict.get("properties", {}), dict)
                        else {}
                    ),
                )
                for e_dict in schema_data.get("entities", [])
                if isinstance(e_dict, dict) and e_dict.get("name")
            ]
            relations = [
                Relation(
                    name=r_dict["name"],
                    src=r_dict.get("source", r_dict.get("src", "")),
                    dst=r_dict.get("target", r_dict.get("dst", "")),
                    description=r_dict.get("description", ""),
                    detail=r_dict.get("detail", ""),
                    properties=(
                        r_dict.get("properties", {})
                        if isinstance(r_dict.get("properties", {}), dict)
                        else {}
                    ),
                )
                for r_dict in schema_data.get("relations", [])
                if isinstance(r_dict, dict) and r_dict.get("name")
            ]
            if not entity_types or not relations:
                raise SchemaGenerationError(f"Legacy schema conversion failed: {e}") from e
            schema = EnhancedDRGSchema(
                entity_types=entity_types,
                relation_groups=[
                    RelationGroup(
                        name="general",
                        description="General relations",
                        relations=relations,
                    )
                ],
                auto_discovery=bool(schema_data.get("auto_discovery", False)),
            )
        else:
            raise SchemaGenerationError(
                f"Schema generation output is not a valid EnhancedDRGSchema JSON: {e}"
            ) from e

    total_relations_count = sum(len(rg.relations) for rg in schema.relation_groups)
    logger.info(
        f"Enhanced schema created: {len(schema.entity_types)} entity types, "
        f"{len(schema.relation_groups)} relation groups, {total_relations_count} relations"
    )
    return schema


def _schema_prediction_to_dict(schema_result: Any) -> dict[str, Any]:
    """Normalize DSPy schema-generation outputs into EnhancedDRGSchema dict shape.

    The primary path is typed fields (`entity_types`, `relation_groups`). A
    narrow legacy fallback accepts the old `generated_schema` JSON string so old
    saved examples and heavily mocked tests fail predictably rather than
    breaking at attribute access.
    """
    if isinstance(schema_result, dict):
        return dict(schema_result)

    if isinstance(schema_result, SchemaOutput):
        return schema_result.model_dump()

    entity_types = getattr(schema_result, "entity_types", None)
    relation_groups = getattr(schema_result, "relation_groups", None)
    if entity_types is not None or relation_groups is not None:
        return {
            "entity_types": _coerce_model_list(entity_types),
            "relation_groups": _coerce_model_list(relation_groups),
        }

    generated_schema = getattr(schema_result, "generated_schema", None)
    if isinstance(generated_schema, str):
        try:
            return _parse_json_output(generated_schema, expected_format="object")
        except ValueError as e:
            raise SchemaGenerationError(
                f"Legacy schema JSON parsing failed: {e}. Prefer typed schema-generation outputs."
            ) from e

    return {}


def _coerce_model_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump())
        elif isinstance(item, dict):
            out.append(dict(item))
    return out


def _sample_text_for_schema_generation(text: str) -> str:
    """Deterministic, input-agnostic sampling for schema generation.

    Goals:
        - Keep behavior deterministic (no randomness).
        - Maximize document coverage for long inputs (avoid missing late-section
          types/relations).
        - Enforce a strict budget so any input size is safe.
    """
    if not text or not text.strip():
        return ""

    sample_fraction = float(os.getenv("DRG_SCHEMA_SAMPLE_FRACTION", "0.60"))
    max_total_chars = int(os.getenv("DRG_SCHEMA_MAX_SAMPLE_CHARS", "100000"))
    max_parts = int(os.getenv("DRG_SCHEMA_MAX_PARTS", "20"))
    min_part_chars = int(os.getenv("DRG_SCHEMA_MIN_PART_CHARS", "2500"))
    max_part_chars = int(os.getenv("DRG_SCHEMA_MAX_PART_CHARS", "5000"))

    doc_len = len(text)
    if doc_len <= max_total_chars:
        logger.info(f"Text is short/medium ({doc_len:,} chars), using full text...")
        return text

    desired = min(
        max_total_chars,
        max(min_part_chars * 4, int(doc_len * max(0.0, min(1.0, sample_fraction)))),
    )
    per_part = max(min_part_chars, min(max_part_chars, desired // max(4, min(max_parts, 12))))
    num_parts = max(4, min(max_parts, max(4, round(desired / max(1, per_part)))))
    part_len = max(min_part_chars, min(max_part_chars, desired // num_parts))

    all_parts: list[tuple[int, str]] = []
    for i in range(num_parts):
        if i == 0:
            start = 0
        elif i == num_parts - 1:
            start = max(0, doc_len - part_len)
        else:
            ratio = i / (num_parts - 1)
            center = int(doc_len * ratio)
            start = max(0, center - part_len // 2)
        end = min(start + part_len, doc_len)
        if end > start:
            all_parts.append((i, text[start:end]))

    # Budget enforcement: always include FIRST and LAST, then fill from middle out.
    sep = "\n\n[... truncated ...]\n\n"
    parts_by_idx = dict(all_parts)
    chosen_idxs: list[int] = []
    if 0 in parts_by_idx:
        chosen_idxs.append(0)
    last_idx = num_parts - 1
    if last_idx in parts_by_idx and last_idx not in chosen_idxs:
        chosen_idxs.append(last_idx)

    mids = [i for i in range(1, last_idx) if i in parts_by_idx]
    mid_center = last_idx / 2.0
    mids.sort(key=lambda i: (abs(i - mid_center), i))

    def _serialized_len(idxs: list[int]) -> int:
        if not idxs:
            return 0
        return sum(len(parts_by_idx[i]) for i in idxs) + (len(idxs) - 1) * len(sep)

    for i in mids:
        trial = sorted({*chosen_idxs, i})
        if _serialized_len(trial) <= max_total_chars:
            chosen_idxs = trial

    if last_idx in parts_by_idx and last_idx not in chosen_idxs:
        chosen_idxs.append(last_idx)
        chosen_idxs = sorted(set(chosen_idxs))
        # If still over budget, drop middle indices.
        while _serialized_len(chosen_idxs) > max_total_chars and len(chosen_idxs) > 2:
            removable = [i for i in chosen_idxs if i not in (0, last_idx)]
            if not removable:
                break
            removable.sort(key=lambda i: (abs(i - mid_center), i), reverse=True)
            chosen_idxs.remove(removable[0])

    out_parts = [parts_by_idx[i] for i in chosen_idxs]
    sampled = sep.join(out_parts)
    coverage = (sum(len(p) for p in out_parts) / doc_len) * 100.0
    logger.info(
        f"Text too long ({doc_len:,} chars), sampling {len(out_parts)} parts "
        f"(~{sum(len(p) for p in out_parts):,} chars, {coverage:.1f}% coverage, "
        f"budget={max_total_chars:,})..."
    )
    return sampled
