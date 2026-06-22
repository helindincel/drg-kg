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


def _relation_schema_for(schema: DRGSchema | EnhancedDRGSchema) -> list[dict[str, str]]:
    normalized = _normalize_schema(schema)
    if isinstance(schema, EnhancedDRGSchema):
        return [
            {"name": r.name, "source_type": r.src, "target_type": r.dst}
            for rg in schema.relation_groups
            for r in rg.relations
        ]
    return [
        {"name": r.name, "source_type": r.src, "target_type": r.dst}
        for r in normalized.relations
    ]


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
    class EntityExtraction(dspy.Signature):
        """Extract entities from text according to the schema."""

        text: str = dspy.InputField(desc="Input text")
        entity_types: list[str] = dspy.InputField(desc="Available entity types")
        entities: list[dict] = dspy.OutputField(desc="Extracted entity mentions")

    EntityExtraction._entity_types = entity_types_list
    return EntityExtraction


def _create_relation_signature(schema: DRGSchema | EnhancedDRGSchema) -> type:
    """Build a dspy.Signature class for relation extraction from the given schema.

    Minimal signature: we do NOT request extra metadata fields from the LLM
    (e.g., confidence/temporal/negation). Those are computed by deterministic
    post-processing in `_heuristics`.
    """
    relation_schema = _relation_schema_for(schema)

    class RelationExtraction(dspy.Signature):
        """Extract relationships between provided entities under the schema."""

        text: str = dspy.InputField(desc="Input text (current chunk)")
        entities: list[dict] = dspy.InputField(desc="Available entity mentions")
        relation_schema: list[dict] = dspy.InputField(desc="Allowed relation types")
        relations: list[dict] = dspy.OutputField(desc="Extracted relations")

    RelationExtraction._relation_schema = relation_schema
    return RelationExtraction


def _create_document_relation_signature(schema: DRGSchema | EnhancedDRGSchema) -> type:
    """Build a document-level relation extraction signature."""
    relation_schema = _relation_schema_for(schema)

    class DocumentRelationExtraction(dspy.Signature):
        """Extract document-level relationships between canonical entities."""

        document_chunks: list[dict] = dspy.InputField(desc="Document chunks")
        entities: list[dict] = dspy.InputField(desc="Canonical entity mentions")
        relation_schema: list[dict] = dspy.InputField(desc="Allowed relation types")
        relations: list[dict] = dspy.OutputField(desc="Extracted document-level relations")

    DocumentRelationExtraction._relation_schema = relation_schema
    return DocumentRelationExtraction


def _create_implicit_relation_signature(schema: DRGSchema | EnhancedDRGSchema) -> type:
    """Build a DSPy signature for schema-gated implicit relation extraction."""
    relation_schema = _relation_schema_for(schema)

    class ImplicitRelationExtraction(dspy.Signature):
        """Infer implicit relationships licensed by the text and existing graph."""

        text: str = dspy.InputField(desc="Input text")
        entities: list[dict] = dspy.InputField(desc="Canonical entity mentions")
        existing_relations: list[dict] = dspy.InputField(desc="Already extracted relations")
        relation_schema: list[dict] = dspy.InputField(desc="Allowed relation types")
        relations: list[dict] = dspy.OutputField(desc="Inferred implicit relations")

    ImplicitRelationExtraction._relation_schema = relation_schema
    return ImplicitRelationExtraction


def _create_coreference_signature(schema: DRGSchema | EnhancedDRGSchema) -> type:
    """Build a DSPy signature for document-aware relation endpoint resolution."""
    relation_schema = _relation_schema_for(schema)

    class CoreferenceResolution(dspy.Signature):
        """Resolve pronoun and alias endpoints in extracted relations."""

        text: str = dspy.InputField(desc="Input text")
        entities: list[dict] = dspy.InputField(desc="Canonical entity mentions")
        relations: list[dict] = dspy.InputField(desc="Relations to resolve")
        relation_schema: list[dict] = dspy.InputField(desc="Allowed relation types")
        resolved_relations: list[dict] = dspy.OutputField(desc="Relations with resolved endpoints")

    CoreferenceResolution._relation_schema = relation_schema
    return CoreferenceResolution
