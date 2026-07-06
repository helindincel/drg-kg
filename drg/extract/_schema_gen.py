"""Automatic schema generation from text via DSPy.

Responsibilities are split across modules:
    - ``_schema_prompts`` — LLM instruction text (core rules + pass-specific tasks)
    - ``_sample_text_for_schema_generation`` — deterministic text sampling
    - ``generate_schema_from_text`` — orchestrates generation, post-processing,
      review, and coverage enrichment
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
from ._adapters import _maybe_json_adapter_context, run_predict
from ._parsing import _parse_json_output
from ._schema_prompts import (
    SCHEMA_COVERAGE_AUDIT_INSTRUCTIONS,
    SCHEMA_GENERATION_INSTRUCTIONS,
    SCHEMA_INTERACTION_FAMILIES,
    SCHEMA_RETRY_GUIDANCE_EMPTY,
    SCHEMA_RETRY_GUIDANCE_TEMPLATE,
    SCHEMA_REVIEW_INSTRUCTIONS,
)
from ._schema_sanitizer import SchemaSanitizer
from ._types import SchemaEntityType, SchemaOutput, SchemaRelationGroup

_sanitizer = SchemaSanitizer()

logger = logging.getLogger(__name__)


def _schema_output_limits() -> dict[str, int]:
    """Configurable ontology size budget for declarative schema generation."""
    return {
        "max_entity_types": int(os.getenv("DRG_SCHEMA_MAX_ENTITY_TYPES", "10")),
        "max_relation_groups": int(os.getenv("DRG_SCHEMA_MAX_RELATION_GROUPS", "6")),
        "max_relations": int(os.getenv("DRG_SCHEMA_MAX_RELATIONS", "32")),
        "max_examples_per_entity_type": int(os.getenv("DRG_SCHEMA_MAX_ENTITY_EXAMPLES", "3")),
    }


def _schema_ontology_budget_text(limits: dict[str, int] | None = None) -> str:
    limits = limits or _schema_output_limits()
    return (
        f"Return at most {limits['max_entity_types']} entity_types, "
        f"at most {limits['max_relation_groups']} relation_groups, and "
        f"at most {limits['max_relations']} relations total across all groups. "
        f"Use at most {limits['max_examples_per_entity_type']} examples per entity type."
    )


def _schema_coverage_pass_enabled() -> bool:
    return os.getenv("DRG_SCHEMA_COVERAGE_PASS", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _schema_max_tokens(*, attempt_idx: int = 1) -> int:
    """Token budget for schema generation (larger than default extraction)."""
    base = int(os.getenv("DRG_SCHEMA_MAX_TOKENS", "8192"))
    if attempt_idx > 1:
        bump = int(os.getenv("DRG_SCHEMA_MAX_TOKENS_RETRY_BUMP", "4096"))
        base += bump * (attempt_idx - 1)
    return max(base, int(os.getenv("DRG_MAX_TOKENS", "1500")))


def _lm_max_tokens(lm: Any) -> int | None:
    kwargs = getattr(lm, "kwargs", None)
    if isinstance(kwargs, dict) and kwargs.get("max_tokens") is not None:
        return int(kwargs["max_tokens"])
    direct = getattr(lm, "max_tokens", None)
    return int(direct) if direct is not None else None


def _lm_model_name(lm: Any) -> str | None:
    for attr in ("model", "model_name"):
        value = getattr(lm, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _schema_model_name() -> str | None:
    """Return the model name to use for schema generation.

    Checks ``DRG_SCHEMA_MODEL`` first; falls back to the active DSPy LM model.
    Setting ``DRG_SCHEMA_MODEL`` allows schema generation to use a different
    (typically higher-quality) model than the extraction pipeline, e.g.::

        DRG_MODEL=openrouter/google/gemini-2.5-flash   # fast, cheap extraction
        DRG_SCHEMA_MODEL=openrouter/anthropic/claude-sonnet-4-5  # schema only
    """
    return os.getenv("DRG_SCHEMA_MODEL", "").strip() or None


def _maybe_schema_lm_context(*, attempt_idx: int = 1):
    """Scope a dedicated LM for schema generation.

    Applies two independent overrides (both optional):
    1. **Model override** – if ``DRG_SCHEMA_MODEL`` is set, switches to that
       model.  This lets you use a higher-quality model (e.g. Claude) for
       schema generation while keeping a faster/cheaper model for extraction.
    2. **Token budget** – always bumps ``max_tokens`` to the schema budget so
       the LLM has enough headroom to emit a full structured JSON response.
    """
    target_tokens = _schema_max_tokens(attempt_idx=attempt_idx)
    schema_model = _schema_model_name()

    settings = getattr(dspy, "settings", None)
    lm = getattr(settings, "lm", None) if settings is not None else None
    if lm is None or isinstance(lm, _Mock):
        return contextlib.nullcontext()

    # Determine the effective model name for the schema LM.
    active_model = _lm_model_name(lm)
    effective_model = schema_model or active_model
    if not effective_model:
        return contextlib.nullcontext()

    current_tokens = _lm_max_tokens(lm)
    model_unchanged = (not schema_model) or (schema_model == active_model)
    tokens_sufficient = current_tokens is not None and current_tokens >= target_tokens

    # Skip if nothing needs changing.
    if model_unchanged and tokens_sufficient:
        return contextlib.nullcontext()

    lm_kwargs: dict[str, Any] = {}
    source_kwargs = getattr(lm, "kwargs", None)
    if isinstance(source_kwargs, dict):
        lm_kwargs.update(source_kwargs)
    lm_kwargs["max_tokens"] = target_tokens

    if schema_model:
        logger.info("DSPy schema LM configured: %s (max_tokens=%d)", effective_model, target_tokens)
    try:
        schema_lm = dspy.LM(effective_model, **lm_kwargs)
    except Exception:
        logger.debug("Could not build schema-scoped dspy.LM", exc_info=True)
        return contextlib.nullcontext()

    ctx_factory = getattr(dspy, "context", None)
    if ctx_factory is not None and not isinstance(ctx_factory, _Mock):
        try:
            return ctx_factory(lm=schema_lm)
        except TypeError:
            pass

    if settings is not None:
        sub_ctx = getattr(settings, "context", None)
        if sub_ctx is not None and not isinstance(sub_ctx, _Mock):
            try:
                return sub_ctx(lm=schema_lm)
            except TypeError:
                pass

    return contextlib.nullcontext()


class SchemaGeneration(dspy.Signature):
    """Generate EnhancedDRGSchema from input text (see generation_instructions)."""

    text: str = dspy.InputField(desc="Input text")
    ontology_budget: str = dspy.InputField(
        desc="Hard per-run limits on entity types, relation groups, and relations."
    )
    interaction_families: str = dspy.InputField(
        desc="Semantic interaction families to survey when discovering relations."
    )
    generation_instructions: str = dspy.InputField(desc="Full ontology design rules for this pass.")
    retry_guidance: str = dspy.InputField(
        desc="Empty on first attempt; corrective guidance after a failed attempt."
    )
    entity_types: list[SchemaEntityType] = dspy.OutputField(
        desc=(
            "Canonical, reusable entity types obeying ontology_budget. Prefer one "
            "broad type over overlapping variants. Examples MUST be canonical "
            "entity names (not pronouns, aliases, or generic noun phrases). Use "
            "empty properties {} unless essential."
        )
    )
    relation_groups: list[SchemaRelationGroup] = dspy.OutputField(
        desc=(
            "Semantically cohesive relation groups obeying ontology_budget. One "
            "canonical relation per interaction; relation names must be "
            "endpoint-free (e.g. 'develops', 'monitors' — never "
            "'develops_product' or 'organization_monitors_person'). Each relation "
            "needs detail and groups should include grounded example triples."
        )
    )


class SchemaReview(dspy.Signature):
    """Review and fix a draft schema (see review_instructions)."""

    text: str = dspy.InputField(desc="Source text the schema was derived from.")
    draft_entity_types: list[dict] = dspy.InputField(
        desc="Draft entity types to review and correct."
    )
    draft_relation_groups: list[dict] = dspy.InputField(
        desc="Draft relation groups to review and correct."
    )
    review_instructions: str = dspy.InputField(desc="Self-critique checklist.")
    entity_types: list[SchemaEntityType] = dspy.OutputField(
        desc="Corrected entity types.  Preserve anything already correct."
    )
    relation_groups: list[SchemaRelationGroup] = dspy.OutputField(
        desc="Corrected relation groups.  Preserve anything already correct."
    )


class SchemaCoverageAudit(dspy.Signature):
    """Extend a draft ontology with missing entity types and relations."""

    text: str = dspy.InputField(desc="Source text")
    entity_types: list[dict] = dspy.InputField(
        desc="Existing entity types from the draft schema (do not modify)."
    )
    relation_groups: list[dict] = dspy.InputField(
        desc="Existing relation groups from the draft schema (do not modify)."
    )
    interaction_families: str = dspy.InputField(
        desc="Semantic interaction families to check for coverage gaps."
    )
    additional_entity_budget: str = dspy.InputField(
        desc="Maximum number of NEW entity types to add in this pass."
    )
    additional_relation_budget: str = dspy.InputField(
        desc="Maximum number of NEW relations to add in this pass."
    )
    coverage_instructions: str = dspy.InputField(desc="Coverage audit rules.")
    additional_entity_types: list[SchemaEntityType] = dspy.OutputField(
        desc="Only newly needed entity types; empty list if none are required."
    )
    additional_relation_groups: list[SchemaRelationGroup] = dspy.OutputField(
        desc="Only newly needed relation groups; empty list if coverage is sufficient."
    )


# Backward-compatible alias for tests and external references.
SchemaRelationCoverageAudit = SchemaCoverageAudit


def generate_schema_from_text(text: str) -> EnhancedDRGSchema:
    """Generate an `EnhancedDRGSchema` from input text via DSPy.

    Orchestrates sampling, LLM generation, sanitization, optional review, and
    optional coverage enrichment. Token budget is scoped per call via
    ``_maybe_schema_lm_context`` — the global ``DRG_MAX_TOKENS`` env var is not
    mutated.

    Raises:
        ValueError: If the input text exceeds ``DRG_MAX_TEXT_CHARS``.
        SchemaGenerationError: If all generation attempts fail.
    """
    _validate_schema_input_length(text)

    from . import _configure_llm_auto

    _configure_llm_auto()

    sample_text = _sample_text_for_schema_generation(text)
    limits = _schema_output_limits()
    schema = _generate_schema_with_retries(sample_text, limits)

    total_relations_count = sum(len(rg.relations) for rg in schema.relation_groups)
    logger.info(
        "Enhanced schema created: %d entity types, %d relation groups, %d relations",
        len(schema.entity_types),
        len(schema.relation_groups),
        total_relations_count,
    )
    return schema


def _validate_schema_input_length(text: str) -> None:
    max_chars = int(os.getenv("DRG_MAX_TEXT_CHARS", "100000"))
    if len(text) > max_chars:
        raise ValueError(
            f"Input text is too long ({len(text):,} chars). "
            f"Maximum allowed: {max_chars:,} chars (set DRG_MAX_TEXT_CHARS to override)."
        )


def _schema_generation_inputs(limits: dict[str, int]) -> dict[str, str]:
    return {
        "ontology_budget": _schema_ontology_budget_text(limits),
        "interaction_families": SCHEMA_INTERACTION_FAMILIES,
        "generation_instructions": SCHEMA_GENERATION_INSTRUCTIONS,
    }


def _generate_schema_with_retries(
    sample_text: str,
    limits: dict[str, int],
) -> EnhancedDRGSchema:
    schema_inputs = _schema_generation_inputs(limits)
    schema_generator = dspy.Predict(SchemaGeneration)
    attempt_texts = _schema_attempt_texts(sample_text)
    last_error: Exception | None = None

    for attempt_idx, attempt_text in enumerate(attempt_texts, start=1):
        retry_guidance = _schema_retry_guidance(last_error)
        try:
            schema = _invoke_schema_generation(
                schema_generator,
                attempt_text=attempt_text,
                attempt_idx=attempt_idx,
                retry_guidance=retry_guidance,
                schema_inputs=schema_inputs,
            )
            schema = _postprocess_generated_schema(
                text=attempt_text,
                schema=schema,
                limits=limits,
                attempt_idx=attempt_idx,
            )
            logger.info("Schema generation completed (attempt %d)", attempt_idx)
            return schema
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Schema generation attempt %d/%d failed: %s",
                attempt_idx,
                len(attempt_texts),
                exc,
            )
            if _looks_like_rate_limit_error(exc):
                break

    logger.error(
        "Schema generation failed after %d attempt(s): %s",
        len(attempt_texts),
        last_error,
    )
    raise SchemaGenerationError(
        f"Schema generation failed: {last_error}. Check your LLM configuration and API keys."
    ) from last_error


def _invoke_schema_generation(
    schema_generator: Any,
    *,
    attempt_text: str,
    attempt_idx: int,
    retry_guidance: str,
    schema_inputs: dict[str, str],
) -> EnhancedDRGSchema:
    from . import _configure_llm_auto

    if attempt_idx > 1:
        _configure_llm_auto()

    throttle_llm_calls()
    with contextlib.ExitStack() as stack:
        stack.enter_context(_maybe_schema_lm_context(attempt_idx=attempt_idx))
        stack.enter_context(_maybe_json_adapter_context())
        schema_result = run_predict(
            schema_generator,
            salvage_fields=("entity_types", "relation_groups"),
            text=attempt_text,
            retry_guidance=retry_guidance,
            **schema_inputs,
        )

    schema_data = _schema_prediction_to_dict(schema_result)
    if not schema_data or (isinstance(schema_data, dict) and not schema_data):
        raise SchemaGenerationError(
            "Schema generation returned empty schema. "
            "The LLM may need a different typed-output configuration."
        )
    return _schema_from_payload(schema_data)


def _postprocess_generated_schema(
    *,
    text: str,
    schema: EnhancedDRGSchema,
    limits: dict[str, int],
    attempt_idx: int,
) -> EnhancedDRGSchema:
    schema, _report = _sanitizer.sanitize(schema)
    schema = _maybe_review_schema(text=text, schema=schema, attempt_idx=attempt_idx)
    schema, _report = _sanitizer.sanitize(schema)
    schema = _maybe_enrich_schema_coverage(
        text=text,
        schema=schema,
        limits=limits,
        attempt_idx=attempt_idx,
    )
    schema, _report = _sanitizer.sanitize(schema)
    return schema


def _relation_count(schema: EnhancedDRGSchema) -> int:
    return sum(len(rg.relations) for rg in schema.relation_groups)


def _schema_entity_types_for_audit(schema: EnhancedDRGSchema) -> list[dict[str, Any]]:
    return [
        {
            "name": et.name,
            "description": et.description,
            "examples": list(et.examples),
            "properties": dict(et.properties),
        }
        for et in schema.entity_types
    ]


def _schema_relation_groups_for_audit(schema: EnhancedDRGSchema) -> list[dict[str, Any]]:
    return [
        {
            "name": rg.name,
            "description": rg.description,
            "relations": [
                {
                    "name": rel.name,
                    "source": rel.src,
                    "target": rel.dst,
                    "description": rel.description,
                    "detail": rel.detail,
                    "properties": dict(rel.properties),
                }
                for rel in rg.relations
            ],
            "examples": list(rg.examples),
        }
        for rg in schema.relation_groups
    ]


def _normalize_relation_name(name: str) -> str:
    return (name or "").strip().lower().replace("-", "_")


def _relation_key(name: str, src: str, dst: str) -> tuple[str, str, str]:
    return (
        _normalize_relation_name(name),
        (src or "").strip(),
        (dst or "").strip(),
    )


def _merge_coverage_extensions(
    schema: EnhancedDRGSchema,
    *,
    additional_entity_types: list[dict[str, Any]],
    additional_groups: list[dict[str, Any]],
    limits: dict[str, int],
) -> EnhancedDRGSchema:
    """Merge coverage-pass entity types and relations into the draft schema."""
    entity_types = list(schema.entity_types)
    relation_groups = list(schema.relation_groups)

    existing_type_names = {et.name for et in entity_types}
    lower_to_canonical = {et.name.lower(): et.name for et in entity_types}
    existing_relations = {
        _relation_key(rel.name, rel.src, rel.dst) for rg in relation_groups for rel in rg.relations
    }

    entity_slots = max(0, limits["max_entity_types"] - len(entity_types))
    added_types = 0
    for et_data in additional_entity_types:
        if added_types >= entity_slots or not isinstance(et_data, dict):
            continue
        name = str(et_data.get("name", "")).strip()
        if not name or name in existing_type_names or name.lower() in lower_to_canonical:
            continue
        entity_types.append(
            EntityType(
                name=name,
                description=str(et_data.get("description", "")),
                examples=(
                    et_data.get("examples", [])
                    if isinstance(et_data.get("examples", []), list)
                    else []
                ),
                properties=(
                    et_data.get("properties", {})
                    if isinstance(et_data.get("properties", {}), dict)
                    else {}
                ),
            )
        )
        existing_type_names.add(name)
        lower_to_canonical[name.lower()] = name
        added_types += 1

    valid_types = {et.name for et in entity_types}
    lower_to_canonical = {et.name.lower(): et.name for et in entity_types}
    relation_slots = max(0, limits["max_relations"] - _relation_count(schema))
    added_relations = 0

    for group_data in additional_groups:
        if added_relations >= relation_slots or not isinstance(group_data, dict):
            break
        new_relations: list[Relation] = []
        for rel_data in group_data.get("relations", []):
            if added_relations >= relation_slots or not isinstance(rel_data, dict):
                continue
            rel_name = str(rel_data.get("name", "")).strip()
            src = str(rel_data.get("source", rel_data.get("src", ""))).strip()
            dst = str(rel_data.get("target", rel_data.get("dst", ""))).strip()
            if not rel_name or not src or not dst:
                continue
            resolved_src = src if src in valid_types else lower_to_canonical.get(src.lower())
            resolved_dst = dst if dst in valid_types else lower_to_canonical.get(dst.lower())
            if resolved_src is None or resolved_dst is None:
                logger.debug(
                    "Coverage merge: dropping relation %r with undefined endpoint(s) "
                    "src=%r dst=%r.",
                    rel_name,
                    src,
                    dst,
                )
                continue
            rel_key = _relation_key(rel_name, resolved_src, resolved_dst)
            if rel_key in existing_relations:
                continue
            new_relations.append(
                Relation(
                    name=rel_name,
                    src=resolved_src,
                    dst=resolved_dst,
                    description=str(rel_data.get("description", "")),
                    detail=str(rel_data.get("detail", "")),
                    properties=(
                        rel_data.get("properties", {})
                        if isinstance(rel_data.get("properties", {}), dict)
                        else {}
                    ),
                )
            )
            existing_relations.add(rel_key)
            added_relations += 1
        if not new_relations:
            continue
        relation_groups.append(
            RelationGroup(
                name=str(group_data.get("name", "coverage_extensions")).strip()
                or "coverage_extensions",
                description=str(
                    group_data.get("description", "Relations added by coverage audit.")
                ),
                relations=new_relations,
                examples=(
                    group_data.get("examples", [])
                    if isinstance(group_data.get("examples", []), list)
                    else []
                ),
            )
        )

    if added_types == 0 and added_relations == 0:
        return schema

    return EnhancedDRGSchema(
        entity_types=entity_types,
        relation_groups=relation_groups,
        entity_groups=list(schema.entity_groups),
        property_groups=list(schema.property_groups),
        auto_discovery=schema.auto_discovery,
    )


def _merge_additional_relation_groups(
    schema: EnhancedDRGSchema,
    additional_groups: list[dict[str, Any]],
    *,
    max_relations: int,
) -> EnhancedDRGSchema:
    """Merge declarative coverage-pass relations into the draft schema."""
    limits = {
        "max_entity_types": len(schema.entity_types),
        "max_relations": max_relations,
    }
    return _merge_coverage_extensions(
        schema,
        additional_entity_types=[],
        additional_groups=additional_groups,
        limits=limits,
    )


def _schema_review_pass_enabled() -> bool:
    explicit = os.getenv("DRG_SCHEMA_REVIEW_PASS", "").strip().lower()
    if explicit in {"1", "true", "yes", "on"}:
        return True
    if explicit in {"0", "false", "no", "off"}:
        return False
    # Auto-enable when a dedicated schema model is configured (typically higher quality).
    return bool(_schema_model_name())


def _maybe_review_schema(
    *,
    text: str,
    schema: EnhancedDRGSchema,
    attempt_idx: int,
) -> EnhancedDRGSchema:
    """Self-critique pass: ask the LLM to review and fix the draft schema.

    Runs between the initial sanitization and the coverage pass so that:
    - The LLM works on a clean schema (primitives and orphans already removed).
    - Any new entity types introduced here (e.g. LegalCase) are available for
      the coverage pass to add relations to.

    Controlled by ``DRG_SCHEMA_REVIEW_PASS`` or auto-enabled when
    ``DRG_SCHEMA_MODEL`` is set.
    """
    if not _schema_review_pass_enabled():
        return schema

    from . import _configure_llm_auto

    _configure_llm_auto()
    review_generator = dspy.Predict(SchemaReview)
    throttle_llm_calls()
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(_maybe_schema_lm_context(attempt_idx=attempt_idx))
            stack.enter_context(_maybe_json_adapter_context())
            review_result = run_predict(
                review_generator,
                salvage_fields=("entity_types", "relation_groups"),
                text=text,
                draft_entity_types=_schema_entity_types_for_audit(schema),
                draft_relation_groups=_schema_relation_groups_for_audit(schema),
                review_instructions=SCHEMA_REVIEW_INSTRUCTIONS,
            )
    except Exception as exc:
        logger.warning("Schema review pass skipped: %s", exc)
        return schema

    # Parse the corrected schema — fall back to the draft if parsing fails.
    entity_types_raw = _coerce_model_list(getattr(review_result, "entity_types", None))
    relation_groups_raw = _coerce_model_list(getattr(review_result, "relation_groups", None))
    if not entity_types_raw and not relation_groups_raw:
        logger.debug("Schema review pass returned empty result; keeping draft.")
        return schema

    try:
        reviewed = _schema_from_payload(
            {
                "entity_types": entity_types_raw,
                "relation_groups": relation_groups_raw,
                "entity_groups": list(schema.entity_groups),
                "property_groups": list(schema.property_groups),
                "auto_discovery": schema.auto_discovery,
            }
        )
    except Exception as exc:
        logger.warning("Schema review pass result could not be parsed (%s); keeping draft.", exc)
        return schema

    # Count changes for logging.
    orig_et = {et.name for et in schema.entity_types}
    new_et = {et.name for et in reviewed.entity_types}
    added_et = new_et - orig_et
    removed_et = orig_et - new_et
    orig_rel = {rel.name for rg in schema.relation_groups for rel in rg.relations}
    new_rel = {rel.name for rg in reviewed.relation_groups for rel in rg.relations}
    added_rel = new_rel - orig_rel
    removed_rel = orig_rel - new_rel

    changes: list[str] = []
    if added_et:
        changes.append(f"+entity_types: {sorted(added_et)}")
    if removed_et:
        changes.append(f"-entity_types: {sorted(removed_et)}")
    if added_rel:
        changes.append(f"+relations: {sorted(added_rel)}")
    if removed_rel:
        changes.append(f"-relations: {sorted(removed_rel)}")

    if changes:
        logger.info("Schema review pass applied changes — %s", "; ".join(changes))
    else:
        logger.debug("Schema review pass: no changes.")

    return reviewed


def _maybe_enrich_schema_coverage(
    *,
    text: str,
    schema: EnhancedDRGSchema,
    limits: dict[str, int],
    attempt_idx: int,
) -> EnhancedDRGSchema:
    """Declarative second DSPy pass to close entity-type and relation coverage gaps."""
    if not _schema_coverage_pass_enabled():
        return schema

    remaining_relations = limits["max_relations"] - _relation_count(schema)
    remaining_entity_types = limits["max_entity_types"] - len(schema.entity_types)
    if remaining_relations <= 0 and remaining_entity_types <= 0:
        return schema

    coverage_relation_budget = min(max(remaining_relations, 0), 4)
    coverage_entity_budget = min(max(remaining_entity_types, 0), 2)

    from . import _configure_llm_auto

    _configure_llm_auto()
    coverage_generator = dspy.Predict(SchemaCoverageAudit)
    throttle_llm_calls()
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(_maybe_schema_lm_context(attempt_idx=attempt_idx))
            stack.enter_context(_maybe_json_adapter_context())
            coverage_result = run_predict(
                coverage_generator,
                salvage_fields=("additional_entity_types", "additional_relation_groups"),
                text=text,
                entity_types=_schema_entity_types_for_audit(schema),
                relation_groups=_schema_relation_groups_for_audit(schema),
                interaction_families=SCHEMA_INTERACTION_FAMILIES,
                additional_entity_budget=(
                    f"Add at most {coverage_entity_budget} new entity types."
                ),
                additional_relation_budget=(
                    f"Add at most {coverage_relation_budget} new relations across new groups."
                ),
                coverage_instructions=SCHEMA_COVERAGE_AUDIT_INSTRUCTIONS,
            )
    except Exception as exc:
        logger.warning("Schema coverage pass skipped: %s", exc)
        return schema

    additional_entity_types = _coerce_model_list(
        getattr(coverage_result, "additional_entity_types", None)
    )
    additional_groups = _coerce_model_list(
        getattr(coverage_result, "additional_relation_groups", None)
    )
    if not additional_entity_types and not additional_groups:
        return schema

    enriched = _merge_coverage_extensions(
        schema,
        additional_entity_types=additional_entity_types,
        additional_groups=additional_groups,
        limits=limits,
    )
    added_types = len(enriched.entity_types) - len(schema.entity_types)
    added_relations = _relation_count(enriched) - _relation_count(schema)
    if added_types or added_relations:
        logger.info(
            "Schema coverage pass added %d entity type(s) and %d relation(s)",
            added_types,
            added_relations,
        )
    return enriched


def _maybe_enrich_schema_relation_coverage(
    *,
    text: str,
    schema: EnhancedDRGSchema,
    limits: dict[str, int],
    attempt_idx: int,
) -> EnhancedDRGSchema:
    """Backward-compatible alias."""
    return _maybe_enrich_schema_coverage(
        text=text,
        schema=schema,
        limits=limits,
        attempt_idx=attempt_idx,
    )


def _schema_retry_guidance(last_error: Exception | None) -> str:
    if last_error is None:
        return SCHEMA_RETRY_GUIDANCE_EMPTY
    reason = str(last_error).strip() or last_error.__class__.__name__
    return SCHEMA_RETRY_GUIDANCE_TEMPLATE.format(reason=reason)


def _schema_attempt_texts(sample_text: str) -> list[str]:
    """Build deterministic fallback prompts for schema generation."""
    text = sample_text or ""
    if not text:
        return [text]

    lengths = [len(text)]
    lengths.extend(
        [
            max(800, int(len(text) * 0.75)),
            max(800, int(len(text) * 0.55)),
            min(1200, len(text)),
        ]
    )
    seen: set[int] = set()
    attempts: list[str] = []
    for ln in lengths:
        ln = max(1, min(len(text), ln))
        if ln in seen:
            continue
        seen.add(ln)
        attempts.append(_truncate_text_at_boundary(text, ln))
    return attempts


def _truncate_text_at_boundary(text: str, max_len: int) -> str:
    """Truncate text at or before ``max_len``, preferring paragraph/sentence breaks."""
    if max_len >= len(text):
        return text
    window = text[:max_len]
    for sep in ("\n\n", "\n", ". ", "! ", "? ", "; "):
        pos = window.rfind(sep)
        if pos >= max(0, max_len // 2):
            return window[: pos + len(sep)].rstrip()
    last_space = window.rfind(" ")
    if last_space >= max(0, max_len // 2):
        return window[:last_space].rstrip()
    return window.rstrip()


def _align_slice_to_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    """Snap a [start, end) slice to nearby paragraph or sentence boundaries."""
    start = max(0, min(start, len(text)))
    end = max(start + 1, min(end, len(text)))

    para_start = text.rfind("\n\n", 0, start)
    if para_start != -1 and start - para_start <= 400:
        start = para_start + 2
    else:
        for punct in (". ", "! ", "? "):
            sent_start = text.rfind(punct, max(0, start - 300), start)
            if sent_start != -1:
                start = sent_start + len(punct)
                break

    para_end = text.find("\n\n", end, min(len(text), end + 400))
    if para_end != -1:
        end = para_end
    else:
        for punct in (". ", "! ", "? "):
            sent_end = text.find(punct, end, min(len(text), end + 300))
            if sent_end != -1:
                end = sent_end + len(punct)
                break

    end = max(start + 1, min(end, len(text)))
    return start, end


def _schema_from_payload(schema_data: dict[str, Any]) -> EnhancedDRGSchema:
    """Parse schema payload, including legacy compatibility conversion."""
    try:
        return EnhancedDRGSchema.from_dict(schema_data)
    except Exception as e:
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
            return EnhancedDRGSchema(
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
        raise SchemaGenerationError(
            f"Schema generation output is not a valid EnhancedDRGSchema JSON: {e}"
        ) from e


def _looks_like_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    tokens = ("rate limit", "ratelimit", "resource_exhausted", "429", "quota exceeded")
    return any(tok in msg for tok in tokens)


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
        start, end = _align_slice_to_boundaries(text, start, end)
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
