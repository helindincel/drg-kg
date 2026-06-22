from unittest.mock import Mock, patch

from drg.extract import ExtractionResult, extract_from_chunks, extract_typed
from drg.schema import DRGSchema, Entity, Relation


class _ImplicitExtractor:
    def __call__(self, text):
        return ExtractionResult(
            entities=[("Tesla", "Company"), ("Gigafactory", "Facility")],
            relations=[],
            enriched_relations=[],
        )

    def infer_implicit_relations(self, *, text, entities, existing_relations):
        return ExtractionResult(
            entities=entities,
            relations=[("Tesla", "owns", "Gigafactory")],
            enriched_relations=[
                {
                    "relation": ("Tesla", "owns", "Gigafactory"),
                    "confidence": 0.82,
                    "evidence": "Tesla's Gigafactory",
                    "metadata": {"source": "implicit_relation_extraction"},
                }
            ],
        )


class _DocumentRelationExtractor:
    def __init__(self):
        self.document_calls = []

    def __call__(self, text):
        if "Apple" in text:
            entities = [("Apple", "Company")]
        elif "Jony Ive" in text:
            entities = [("Jony Ive", "Person"), ("iPhone", "Product")]
        else:
            entities = []
        return ExtractionResult(entities=entities, relations=[], enriched_relations=[])

    def extract_document_relations(self, *, chunks, entities):
        self.document_calls.append({"chunks": chunks, "entities": entities})
        return ExtractionResult(
            entities=entities,
            relations=[("Jony Ive", "designed", "iPhone")],
            enriched_relations=[
                {
                    "relation": ("Jony Ive", "designed", "iPhone"),
                    "evidence": "document-level relation",
                    "metadata": {"source": "document_relation_extraction"},
                }
            ],
        )


class _CoreferenceExtractor:
    def __call__(self, text):
        return ExtractionResult(
            entities=[("John", "Person"), ("He", "Person"), ("Tesla", "Company")],
            relations=[("He", "works_at", "Tesla")],
            enriched_relations=[],
        )

    def resolve_coreferences_dspy(self, *, text, entities, relations):
        return ExtractionResult(
            entities=entities,
            relations=[("John", "works_at", "Tesla")],
            enriched_relations=[
                {
                    "relation": ("John", "works_at", "Tesla"),
                    "metadata": {"source": "coreference_resolution"},
                }
            ],
        )

    def infer_implicit_relations(self, *, text, entities, existing_relations):
        return ExtractionResult(entities=entities, relations=[], enriched_relations=[])


def test_implicit_relations_are_inferred_by_dspy_extractor():
    schema = DRGSchema(
        entities=[Entity("Company"), Entity("Facility")],
        relations=[Relation("owns", "Company", "Facility")],
    )

    with (
        patch("drg.extract._get_extractor", return_value=_ImplicitExtractor()),
        patch("drg.extract.dspy") as mock_dspy,
    ):
        mock_dspy.settings.lm = Mock()
        entities, relations, enriched = extract_typed(
            "Tesla's Gigafactory is in Nevada.",
            schema,
            enable_entity_resolution=False,
            return_enriched=True,
        )

    assert ("Tesla", "Company") in entities
    assert ("Tesla", "owns", "Gigafactory") in relations
    assert enriched[0]["metadata"]["source"] == "implicit_relation_extraction"


def test_cross_chunk_relations_use_document_level_extractor_without_context_stuffing():
    schema = DRGSchema(
        entities=[Entity("Company"), Entity("Person"), Entity("Product")],
        relations=[
            Relation("designed", "Person", "Product"),
        ],
    )
    extractor = _DocumentRelationExtractor()

    with (
        patch("drg.extract._get_extractor", return_value=extractor),
        patch("drg.extract.dspy") as mock_dspy,
    ):
        mock_dspy.settings.lm = Mock()
        _entities, relations = extract_from_chunks(
            [
                {"text": "Apple released the iPhone."},
                {"text": "Jony Ive designed it."},
            ],
            schema,
            enable_entity_resolution=False,
            enable_implicit_relationships=False,
        )

    assert relations == [("Jony Ive", "designed", "iPhone")]
    assert len(extractor.document_calls) == 1
    assert all("[CROSS-CHUNK CONTEXT]" not in c["text"] for c in extractor.document_calls[0]["chunks"])


def test_coreference_uses_dspy_resolution_before_entity_resolution():
    schema = DRGSchema(
        entities=[Entity("Person"), Entity("Company")],
        relations=[Relation("works_at", "Person", "Company")],
    )

    with (
        patch("drg.extract._get_extractor", return_value=_CoreferenceExtractor()),
        patch("drg.extract.resolve_coreferences", lambda **kwargs: (kwargs["entities"], kwargs["relations"])),
        patch("drg.extract.dspy") as mock_dspy,
    ):
        mock_dspy.settings.lm = Mock()
        _entities, relations = extract_typed(
            "John joined Tesla. He works there.",
            schema,
            enable_coreference_resolution=True,
            enable_entity_resolution=False,
            enable_implicit_relationships=False,
        )

    assert relations == [("John", "works_at", "Tesla")]
