"""DSPy signature factories for entity and relation extraction.

Signatures are generated dynamically from the user-provided schema.

Design: Signatures carry the full schema context so the LLM has precise
guidance — entity type descriptions/examples/properties and relation
descriptions/details/properties/examples are passed as structured InputField data, not
discarded. OutputField annotations use the typed Pydantic models
(EntityMention, ExtractedRelation) so DSPy generates a rich JSON schema for the
output prompt, making properties, confidence, evidence, negation, and temporal
fields visible to the LLM.

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
from ._types import EntityMention, ExtractedRelation


def _relation_schema_for(schema: DRGSchema | EnhancedDRGSchema) -> list[dict]:
    """Build the relation schema list passed to every relation extraction signature.

    For EnhancedDRGSchema, includes group-level context (group name and
    description/examples) and per-relation semantics (description and
    detail/example).
    For legacy DRGSchema, includes per-relation description and detail.
    """
    normalized = _normalize_schema(schema)
    if isinstance(schema, EnhancedDRGSchema):
        return [
            {
                "name": r.name,
                "source_type": r.src,
                "target_type": r.dst,
                "description": r.description,
                "example": r.detail,
                "properties": r.properties,
                "group": rg.name,
                "group_description": rg.description,
                "group_examples": rg.examples,
            }
            for rg in schema.relation_groups
            for r in rg.relations
        ]
    return [
        {
            "name": r.name,
            "source_type": r.src,
            "target_type": r.dst,
            "description": r.description,
            "example": r.detail,
            "properties": r.properties,
        }
        for r in normalized.relations
    ]


def _entity_schema_for(schema: DRGSchema | EnhancedDRGSchema) -> list[dict]:
    """Build the entity type list passed to the entity extraction signature.

    For EnhancedDRGSchema, includes description, up to 5 examples per type,
    and schema-defined properties. For EntityGroup-aware schemas, includes the
    group name and description so the LLM can distinguish semantically grouped
    types.
    For legacy DRGSchema, includes only the name (no metadata available).
    """
    normalized = _normalize_schema(schema)
    if isinstance(schema, EnhancedDRGSchema):
        # Build a group membership index for enrichment.
        entity_to_group: dict[str, tuple[str, str]] = {}
        for eg in schema.entity_groups:
            for et in eg.entity_types:
                entity_to_group[et.name] = (eg.name, eg.description)

        result = []
        for et in schema.entity_types:
            entry: dict = {
                "name": et.name,
                "description": et.description,
                "examples": et.examples[:5],
                "properties": et.properties,
            }
            if et.name in entity_to_group:
                grp_name, grp_desc = entity_to_group[et.name]
                entry["group"] = grp_name
                entry["group_description"] = grp_desc
            result.append(entry)
        return result

    # Legacy DRGSchema — Entity dataclass has only a name field.
    return [
        {"name": e.name, "description": "", "examples": [], "properties": {}}
        for e in normalized.entities
    ]


def _create_entity_signature(schema: DRGSchema | EnhancedDRGSchema) -> type:
    """Build a dspy.Signature class for entity extraction from the given schema.

    The entity_types InputField receives full schema metadata (descriptions,
    examples, properties, optional group context) so the LLM can distinguish
    between similar entity types. The entities OutputField is annotated with
    EntityMention so DSPy exposes all output fields (aliases, evidence,
    properties) to the LLM in the output JSON schema.
    """
    entity_types_list = _entity_schema_for(schema)

    class EntityExtraction(dspy.Signature):
        """Extract named entities from the text that match the provided schema.

        For each entity found:
        - Set name to the form used in the text.
        - Set type to exactly one value from entity_types[*].name.
        - Set aliases to any abbreviations or alternate forms for the same entity in the text.
        - Set evidence to the shortest verbatim text span that identifies the entity.
        - Set properties with values found in the text for keys declared in the matching entity type.
        Only extract entities whose type matches one of the provided entity types.
        """

        text: str = dspy.InputField(desc="Input text to analyze")
        entity_types: list[dict] = dspy.InputField(
            desc=(
                "Entity type schema. Each entry: name (type identifier), "
                "description (what this type means), examples (typical instances), "
                "properties (schema-defined attributes to populate when present). "
                "Only extract entities whose type exactly matches one of these names."
            )
        )
        entities: list[EntityMention] = dspy.OutputField(
            desc=(
                "Extracted entity mentions. Each must have: "
                "name (as it appears in text), "
                "type (exactly matching one of the provided entity_types names), "
                "aliases (list of other forms or abbreviations in the text, empty list if none), "
                "evidence (shortest verbatim span from the text that identifies the entity), "
                "properties (only schema-defined keys with values supported by the text), "
                "metadata (non-schema auxiliary details if needed)."
            )
        )

    EntityExtraction._entity_types = entity_types_list  # type: ignore[attr-defined]
    return EntityExtraction


def _create_relation_signature(schema: DRGSchema | EnhancedDRGSchema) -> type:
    """Build a dspy.Signature class for relation extraction from the given schema.

    The relation_schema InputField receives full relation semantics (description,
    example, group context, group examples). The relations OutputField is annotated with
    ExtractedRelation so the LLM sees confidence, evidence, is_negated, and
    temporal as required output fields.
    """
    relation_schema = _relation_schema_for(schema)

    class RelationExtraction(dspy.Signature):
        """Extract relationships between the provided entities that are explicitly supported by the text.

        Rules:
        - Only assert relations whose name appears in relation_schema.
        - Set is_negated=True when the text explicitly denies or revokes the relation
          (e.g. 'no longer', 'never', 'stopped', 'ceased').
        - Set confidence (0.0–1.0) to your certainty that the relation holds as stated.
        - Set evidence to the shortest verbatim text span that licenses the relation.
        - Set temporal.start/end (ISO 8601 or year) when the relation is time-bounded.
        - Do NOT infer relations unsupported by the text.
        """

        text: str = dspy.InputField(desc="Input text (current chunk)")
        entities: list[dict] = dspy.InputField(desc="Available entity mentions with name and type")
        relation_schema: list[dict] = dspy.InputField(
            desc=(
                "Allowed relation types. Each entry: name, source_type, target_type, "
                "description (what the relation means), example (evidence text), "
                "group (semantic category), group_description, group_examples."
            )
        )
        relations: list[ExtractedRelation] = dspy.OutputField(
            desc=(
                "Extracted relations. Each must have: "
                "source (entity name), relation (from schema), target (entity name), "
                "confidence (0.0–1.0), "
                "evidence (verbatim span), "
                "is_negated (bool), "
                "temporal (object with start/end if time-bounded, else null)."
            )
        )

    RelationExtraction._relation_schema = relation_schema  # type: ignore[attr-defined]
    return RelationExtraction


def _create_document_relation_signature(schema: DRGSchema | EnhancedDRGSchema) -> type:
    """Build a document-level relation extraction signature."""
    relation_schema = _relation_schema_for(schema)

    class DocumentRelationExtraction(dspy.Signature):
        """Extract relationships between canonical entities across all document chunks.

        Rules:
        - Only assert relations whose name appears in relation_schema.
        - A relation may span multiple chunks; use evidence from the most direct chunk.
        - Set is_negated=True when the text explicitly denies the relation.
        - Set confidence (0.0–1.0) and evidence (verbatim span).
        - Set temporal.start/end when the relation is time-bounded.
        """

        document_chunks: list[dict] = dspy.InputField(
            desc="Document chunks, each with chunk_id and text"
        )
        entities: list[dict] = dspy.InputField(desc="Canonical entity mentions with name and type")
        relation_schema: list[dict] = dspy.InputField(
            desc=(
                "Allowed relation types. Each entry: name, source_type, target_type, "
                "description, example, group, group_description, group_examples."
            )
        )
        relations: list[ExtractedRelation] = dspy.OutputField(
            desc=(
                "Extracted document-level relations. Each must have: "
                "source, relation (from schema), target, confidence (0.0–1.0), "
                "evidence (verbatim span), is_negated (bool), temporal."
            )
        )

    DocumentRelationExtraction._relation_schema = relation_schema  # type: ignore[attr-defined]
    return DocumentRelationExtraction


def _create_implicit_relation_signature(schema: DRGSchema | EnhancedDRGSchema) -> type:
    """Build a DSPy signature for schema-gated implicit relation extraction."""
    relation_schema = _relation_schema_for(schema)

    class ImplicitRelationExtraction(dspy.Signature):
        """Infer implicit relationships licensed by the text and existing graph.

        Rules:
        - Only assert relations whose name appears in relation_schema.
        - Do NOT repeat relations already in existing_relations.
        - An implicit relation must be inferable from the text, not invented.
        - Set confidence, evidence, is_negated, and temporal as appropriate.
        """

        text: str = dspy.InputField(desc="Input text")
        entities: list[dict] = dspy.InputField(desc="Canonical entity mentions with name and type")
        existing_relations: list[dict] = dspy.InputField(
            desc="Already extracted relations (do not repeat these)"
        )
        relation_schema: list[dict] = dspy.InputField(
            desc=(
                "Allowed relation types. Each entry: name, source_type, target_type, "
                "description, example, group, group_description, group_examples."
            )
        )
        relations: list[ExtractedRelation] = dspy.OutputField(
            desc=(
                "Inferred implicit relations not already in existing_relations. "
                "Each must have: source, relation (from schema), target, "
                "confidence (0.0–1.0), evidence (verbatim span), is_negated (bool), temporal."
            )
        )

    ImplicitRelationExtraction._relation_schema = relation_schema  # type: ignore[attr-defined]
    return ImplicitRelationExtraction


def _create_coreference_signature(schema: DRGSchema | EnhancedDRGSchema) -> type:
    """Build a DSPy signature for document-aware relation endpoint resolution."""
    relation_schema = _relation_schema_for(schema)

    class CoreferenceResolution(dspy.Signature):
        """Resolve pronoun and alias endpoints in extracted relations.

        Replace pronouns, abbreviations, or aliases in relation source/target
        with the canonical entity name from the entities list.
        Preserve all other relation fields (confidence, evidence, is_negated, temporal).
        """

        text: str = dspy.InputField(desc="Input text")
        entities: list[dict] = dspy.InputField(desc="Canonical entity mentions with name and type")
        relations: list[dict] = dspy.InputField(
            desc="Relations whose endpoints may contain pronouns or aliases"
        )
        relation_schema: list[dict] = dspy.InputField(desc="Allowed relation types for validation")
        resolved_relations: list[ExtractedRelation] = dspy.OutputField(
            desc=(
                "Relations with resolved endpoints. Replace pronoun/alias source or target "
                "with the matching canonical entity name. "
                "Each must have: source, relation, target, confidence, evidence, is_negated, temporal."
            )
        )

    CoreferenceResolution._relation_schema = relation_schema  # type: ignore[attr-defined]
    return CoreferenceResolution
