"""DSPy signature factories for entity and relation extraction.

Signatures are generated dynamically from the user-provided schema.

NOTE on DSPy patching:
    Tests patch `drg.extract.dspy`; the signature classes constructed here are
    instantiated by `KGExtractor` (defined in `drg.extract.__init__`), so the
    relevant `dspy` lookups happen inside that module's namespace.
    The signatures themselves are short-lived class objects.
"""

from __future__ import annotations

import dspy

from ..schema import DRGSchema, EnhancedDRGSchema
from ._relations import _normalize_schema


def _create_entity_signature(schema: DRGSchema | EnhancedDRGSchema) -> type:
    """Build a dspy.Signature class for entity extraction from the given schema.

    DSPy best practice: keep the signature minimal (InputField/OutputField only).
    Avoid prompt engineering; let the schema speak for itself.
    """
    normalized = _normalize_schema(schema)

    if isinstance(schema, EnhancedDRGSchema):
        entity_types_list = [et.name for et in schema.entity_types]
    else:
        entity_types_list = [e.name for e in normalized.entities]
    entity_types = ", ".join(entity_types_list)

    class EntityExtraction(dspy.Signature):
        """Extract entities from text according to the schema."""

        text: str = dspy.InputField(desc="Input text")
        entities: list[tuple[str, str]] = dspy.OutputField(
            desc=(
                f"Return entities as [(entity_name, entity_type), ...]. "
                f"entity_type must be one of: {entity_types}."
            )
        )

    EntityExtraction._entity_types = entity_types_list
    return EntityExtraction


def _create_relation_signature(schema: DRGSchema | EnhancedDRGSchema) -> type:
    """Build a dspy.Signature class for relation extraction from the given schema.

    Minimal signature: we do NOT request extra metadata fields from the LLM
    (e.g., confidence/temporal/negation). Those are computed by deterministic
    post-processing in `_heuristics`.
    """
    normalized = _normalize_schema(schema)

    if isinstance(schema, EnhancedDRGSchema):
        schema_info = "\n".join(
            f"{r.name}: {r.src} -> {r.dst}" for rg in schema.relation_groups for r in rg.relations
        )
    else:
        schema_info = "\n".join(f"{r.name}: {r.src} -> {r.dst}" for r in normalized.relations)

    class RelationExtraction(dspy.Signature):
        """Extract relationships between provided entities under the schema."""

        text: str = dspy.InputField(desc="Input text (current chunk)")
        entities: list[tuple[str, str]] = dspy.InputField(desc="Entities as [(name, type), ...].")
        relations: list[tuple[str, str, str]] = dspy.OutputField(
            desc=(
                "Return relations as [(source, relation, target), ...]. "
                f"Must be valid under schema:\n{schema_info}"
            )
        )

    RelationExtraction._relation_info = schema_info
    return RelationExtraction
