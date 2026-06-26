"""Lightweight ontology evaluation for generated DRG schemas.

The evaluators in this module are intentionally domain-agnostic. They inspect
the ontology structure, names, descriptions, properties, and relation topology
without relying on document-specific gold labels or expected entity types.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..schema import DRGSchema, EnhancedDRGSchema

__all__ = [
    "DEFAULT_ONTOLOGY_EVALUATORS",
    "EventModelingQualityEvaluator",
    "ExtractionSuitabilityEvaluator",
    "OntologyEvaluationReport",
    "OntologyEvaluationResult",
    "OntologyEvaluatorContext",
    "OntologyMetricEvaluator",
    "PropertyModelingQualityEvaluator",
    "RelationQualityEvaluator",
    "ReusabilityEvaluator",
    "SemanticCompletenessEvaluator",
    "SemanticConsistencyEvaluator",
    "evaluate_ontology",
]


@dataclass(frozen=True)
class _EntityView:
    name: str
    description: str = ""
    examples: tuple[str, ...] = ()
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _RelationView:
    name: str
    source: str
    target: str
    description: str = ""
    detail: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    group: str = ""


@dataclass(frozen=True)
class OntologyEvaluatorContext:
    """Normalized, read-only view consumed by ontology metric evaluators."""

    entities: tuple[_EntityView, ...]
    relations: tuple[_RelationView, ...]
    relation_groups: tuple[str, ...] = ()
    property_groups: tuple[dict[str, Any], ...] = ()

    @property
    def entity_names(self) -> set[str]:
        return {entity.name for entity in self.entities}

    @property
    def connected_entity_names(self) -> set[str]:
        names: set[str] = set()
        for relation in self.relations:
            names.add(relation.source)
            names.add(relation.target)
        return names & self.entity_names


@dataclass(frozen=True)
class OntologyEvaluationResult:
    """Result returned by one ontology metric evaluator."""

    metric: str
    score: float
    explanation: str
    findings: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "score": round(_clamp01(self.score), 4),
            "explanation": self.explanation,
            "findings": list(self.findings),
            "recommendations": list(self.recommendations),
            "weight": self.weight,
        }


@dataclass(frozen=True)
class OntologyEvaluationReport:
    """Aggregated ontology evaluation report."""

    overall_score: float
    metrics: tuple[OntologyEvaluationResult, ...]
    strengths: tuple[str, ...] = ()
    weaknesses: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()

    @property
    def per_metric_scores(self) -> dict[str, float]:
        return {result.metric: round(_clamp01(result.score), 4) for result in self.metrics}

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(_clamp01(self.overall_score), 4),
            "per_metric_scores": self.per_metric_scores,
            "metrics": [result.to_dict() for result in self.metrics],
            "strengths": list(self.strengths),
            "weaknesses": list(self.weaknesses),
            "recommendations": list(self.recommendations),
        }


class OntologyMetricEvaluator(Protocol):
    """Protocol implemented by independent ontology metric evaluators."""

    name: str
    weight: float

    def evaluate(self, context: OntologyEvaluatorContext) -> OntologyEvaluationResult:
        """Evaluate one metric against a normalized ontology context."""


class _BaseOntologyMetric:
    name = "ontology_metric"
    weight = 1.0

    def _result(
        self,
        score: float,
        explanation: str,
        findings: list[str] | None = None,
        recommendations: list[str] | None = None,
    ) -> OntologyEvaluationResult:
        return OntologyEvaluationResult(
            metric=self.name,
            score=_clamp01(score),
            explanation=explanation,
            findings=tuple(findings or ()),
            recommendations=tuple(recommendations or ()),
            weight=self.weight,
        )


class SemanticCompletenessEvaluator(_BaseOntologyMetric):
    """Estimate whether major semantic surfaces are represented in the ontology."""

    name = "semantic_completeness"

    def evaluate(self, context: OntologyEvaluatorContext) -> OntologyEvaluationResult:
        findings: list[str] = []
        recommendations: list[str] = []

        entity_count = len(context.entities)
        relation_count = len(context.relations)
        connected_ratio = _ratio(len(context.connected_entity_names), entity_count)
        described_entities = _ratio(
            sum(1 for entity in context.entities if _meaningful_text(entity.description)),
            entity_count,
        )
        described_relations = _ratio(
            sum(1 for relation in context.relations if _meaningful_text(relation.description)),
            relation_count,
        )
        relation_coverage = min(1.0, relation_count / max(entity_count - 1, 1))
        relation_variety = _ratio(
            len({relation.name for relation in context.relations}), relation_count
        )

        score = (
            0.30 * connected_ratio
            + 0.25 * relation_coverage
            + 0.20 * described_entities
            + 0.15 * described_relations
            + 0.10 * relation_variety
        )

        isolated = sorted(context.entity_names - context.connected_entity_names)
        if isolated:
            findings.append(f"{len(isolated)} entity type has no relation participation.")
            recommendations.append(
                "Connect isolated entity types or remove them if they are not extractable."
            )
        if relation_count < max(1, entity_count - 1):
            findings.append("Relation coverage is sparse relative to the number of entity types.")
            recommendations.append(
                "Add reusable relations for recurring interactions and dependencies."
            )

        return self._result(
            score,
            "Scores ontology coverage through connected entity types, relation coverage, descriptions, and relation variety.",
            findings,
            recommendations,
        )


class SemanticConsistencyEvaluator(_BaseOntologyMetric):
    """Check internal ontology coherence without requiring a gold schema."""

    name = "semantic_consistency"

    def evaluate(self, context: OntologyEvaluatorContext) -> OntologyEvaluationResult:
        findings: list[str] = []
        recommendations: list[str] = []
        penalties = 0.0

        entity_counts = Counter(entity.name for entity in context.entities)
        duplicate_entities = [name for name, count in entity_counts.items() if count > 1]
        if duplicate_entities:
            penalties += 0.25
            findings.append(
                f"Duplicate entity type names found: {', '.join(duplicate_entities[:5])}."
            )
            recommendations.append("Use one canonical entity type name for each semantic role.")

        invalid_relations = [
            relation
            for relation in context.relations
            if relation.source not in context.entity_names
            or relation.target not in context.entity_names
        ]
        if invalid_relations:
            penalties += min(0.40, 0.12 * len(invalid_relations))
            findings.append(f"{len(invalid_relations)} relation endpoint reference is invalid.")
            recommendations.append("Ensure every relation source and target names an entity type.")

        triple_counts = Counter((r.name, r.source, r.target) for r in context.relations)
        duplicate_triples = [triple for triple, count in triple_counts.items() if count > 1]
        if duplicate_triples:
            penalties += min(0.20, 0.05 * len(duplicate_triples))
            findings.append("Duplicate relation triples reduce canonical representation quality.")
            recommendations.append("Deduplicate repeated relation definitions.")

        missing_relation_names = sum(
            1 for relation in context.relations if not relation.name.strip()
        )
        missing_entity_names = sum(1 for entity in context.entities if not entity.name.strip())
        if missing_relation_names or missing_entity_names:
            penalties += 0.25
            findings.append("Some ontology elements have empty names.")
            recommendations.append("Require stable names for all entity and relation types.")

        return self._result(
            1.0 - penalties,
            "Checks duplicate definitions, invalid relation endpoints, and empty ontology element names.",
            findings,
            recommendations,
        )


class ReusabilityEvaluator(_BaseOntologyMetric):
    """Assess whether ontology elements look reusable beyond one document."""

    name = "reusability"

    def evaluate(self, context: OntologyEvaluatorContext) -> OntologyEvaluationResult:
        findings: list[str] = []
        recommendations: list[str] = []
        names = [entity.name for entity in context.entities] + [
            relation.name for relation in context.relations
        ]

        concise_names = _ratio(sum(1 for name in names if 1 <= len(name.split()) <= 4), len(names))
        stable_names = _ratio(
            sum(1 for name in names if not _looks_instance_specific(name)), len(names)
        )
        generic_penalty = _ratio(
            sum(1 for name in names if _is_low_information_name(name)), len(names)
        )
        descriptions = [entity.description for entity in context.entities] + [
            relation.description for relation in context.relations
        ]
        reusable_descriptions = _ratio(
            sum(
                1
                for text in descriptions
                if _meaningful_text(text) and not _looks_instance_specific(text)
            ),
            len(descriptions),
        )
        score = 0.35 * concise_names + 0.35 * stable_names + 0.25 * reusable_descriptions
        score -= 0.20 * generic_penalty

        instance_like = [name for name in names if _looks_instance_specific(name)]
        if instance_like:
            findings.append("Some type names look instance-specific rather than reusable.")
            recommendations.append(
                "Prefer semantic role names over document-specific labels or examples."
            )
        if generic_penalty:
            findings.append("Some names are low-information catch-all labels.")
            recommendations.append(
                "Replace generic labels with clearer reusable semantic distinctions."
            )

        return self._result(
            score,
            "Scores concise, stable naming and reusable descriptions while penalizing instance-like or catch-all labels.",
            findings,
            recommendations,
        )


class RelationQualityEvaluator(_BaseOntologyMetric):
    """Evaluate relation specificity, endpoint use, and organization."""

    name = "relation_quality"

    def evaluate(self, context: OntologyEvaluatorContext) -> OntologyEvaluationResult:
        relation_count = len(context.relations)
        findings: list[str] = []
        recommendations: list[str] = []

        if relation_count == 0:
            return self._result(
                0.0,
                "No relations are available to evaluate.",
                ["The ontology does not define relation types."],
                ["Add reusable relation types between generated entity types."],
            )

        named = _ratio(
            sum(1 for relation in context.relations if relation.name.strip()), relation_count
        )
        described = _ratio(
            sum(1 for relation in context.relations if _meaningful_text(relation.description)),
            relation_count,
        )
        valid_endpoints = _ratio(
            sum(
                1
                for relation in context.relations
                if relation.source in context.entity_names
                and relation.target in context.entity_names
            ),
            relation_count,
        )
        specific_names = _ratio(
            sum(1 for relation in context.relations if not _is_low_information_name(relation.name)),
            relation_count,
        )
        grouped = _ratio(sum(1 for relation in context.relations if relation.group), relation_count)
        score = (
            0.20 * named
            + 0.25 * described
            + 0.25 * valid_endpoints
            + 0.20 * specific_names
            + 0.10 * grouped
        )

        weak_names = [
            relation.name
            for relation in context.relations
            if _is_low_information_name(relation.name)
        ]
        if weak_names:
            findings.append("Some relation names are too broad to carry clear graph semantics.")
            recommendations.append(
                "Use relation names that express a reusable interaction, not just association."
            )
        if described < 0.8:
            findings.append("Several relations lack meaningful descriptions.")
            recommendations.append(
                "Add short descriptions that define relation semantics and extraction cues."
            )

        return self._result(
            score,
            "Scores relation naming, descriptions, valid endpoints, semantic specificity, and grouping.",
            findings,
            recommendations,
        )


class PropertyModelingQualityEvaluator(_BaseOntologyMetric):
    """Evaluate whether properties are present and modeled in a reusable way."""

    name = "property_modeling_quality"

    def evaluate(self, context: OntologyEvaluatorContext) -> OntologyEvaluationResult:
        findings: list[str] = []
        recommendations: list[str] = []
        entity_properties = [
            prop for entity in context.entities for prop in entity.properties.items()
        ]
        relation_properties = [
            prop for relation in context.relations for prop in relation.properties.items()
        ]
        all_properties = entity_properties + relation_properties

        if not all_properties:
            score = 0.55 if len(context.entities) <= 3 else 0.40
            findings.append("No entity or relation properties are defined.")
            recommendations.append(
                "Add reusable properties for intrinsic attributes and relation qualifiers."
            )
            return self._result(
                score,
                "No properties are defined; this may be acceptable for very small ontologies but limits extraction utility.",
                findings,
                recommendations,
            )

        named = _ratio(
            sum(1 for name, _ in all_properties if str(name).strip()), len(all_properties)
        )
        documented = _ratio(
            sum(1 for _, spec in all_properties if _property_has_description(spec)),
            len(all_properties),
        )
        scoped = 0.65
        if entity_properties and relation_properties:
            scoped = 1.0
        elif relation_properties:
            scoped = 0.8
        score = 0.35 * named + 0.45 * documented + 0.20 * scoped

        if documented < 0.8:
            findings.append("Some properties lack descriptions or structured specifications.")
            recommendations.append(
                "Document property meaning so extraction can distinguish attributes from relations."
            )

        return self._result(
            score,
            "Scores property naming, documentation, and whether entity and relation qualifiers are modeled explicitly.",
            findings,
            recommendations,
        )


class EventModelingQualityEvaluator(_BaseOntologyMetric):
    """Evaluate explicit event modeling and temporal/causal support."""

    name = "event_modeling_quality"

    def evaluate(self, context: OntologyEvaluatorContext) -> OntologyEvaluationResult:
        findings: list[str] = []
        recommendations: list[str] = []
        event_entities = [
            entity for entity in context.entities if _contains_any(entity.name, _EVENT_TERMS)
        ]
        temporal_or_causal_relations = [
            relation
            for relation in context.relations
            if _contains_any(relation.name + " " + relation.description, _EVENT_RELATION_TERMS)
        ]

        if not event_entities and not temporal_or_causal_relations:
            return self._result(
                0.70,
                "No explicit event or temporal/causal modeling need is visible from the ontology structure.",
                (),
                (),
            )

        event_connected = _ratio(
            sum(1 for entity in event_entities if entity.name in context.connected_entity_names),
            len(event_entities),
        )
        event_described = _ratio(
            sum(1 for entity in event_entities if _meaningful_text(entity.description)),
            len(event_entities),
        )
        temporal_causal_support = min(
            1.0, len(temporal_or_causal_relations) / max(len(event_entities), 1)
        )

        if event_entities:
            score = 0.40 * event_connected + 0.30 * event_described + 0.30 * temporal_causal_support
        else:
            score = 0.45
            findings.append(
                "Temporal or causal relations exist, but no first-class event type is modeled."
            )
            recommendations.append(
                "Model significant events as entity types when they have participants or timing."
            )

        if event_entities and event_connected < 1.0:
            findings.append("Some event-like entity types are not connected by relations.")
            recommendations.append(
                "Connect event types to participants, causes, consequences, or temporal context."
            )

        return self._result(
            score,
            "Scores event-like entity types by connectivity, descriptions, and temporal/causal relation support.",
            findings,
            recommendations,
        )


class ExtractionSuitabilityEvaluator(_BaseOntologyMetric):
    """Estimate whether the ontology is practical for downstream extraction."""

    name = "extraction_suitability"

    def evaluate(self, context: OntologyEvaluatorContext) -> OntologyEvaluationResult:
        findings: list[str] = []
        recommendations: list[str] = []

        elements = len(context.entities) + len(context.relations)
        described_elements = sum(
            1 for entity in context.entities if _meaningful_text(entity.description)
        ) + sum(1 for relation in context.relations if _meaningful_text(relation.description))
        example_or_detail = sum(1 for entity in context.entities if entity.examples) + sum(
            1 for relation in context.relations if _meaningful_text(relation.detail)
        )
        stable_names = sum(
            1
            for name in [entity.name for entity in context.entities]
            + [relation.name for relation in context.relations]
            if name.strip() and not _looks_instance_specific(name)
        )
        compactness = _compactness_score(len(context.entities), len(context.relations))

        score = (
            0.30 * _ratio(described_elements, elements)
            + 0.20 * _ratio(example_or_detail, elements)
            + 0.25 * _ratio(stable_names, elements)
            + 0.25 * compactness
        )

        if _ratio(example_or_detail, elements) < 0.4:
            findings.append(
                "Few elements include examples or relation details that guide extraction."
            )
            recommendations.append(
                "Add compact examples or evidence-style details for ambiguous elements."
            )
        if compactness < 0.6:
            findings.append("Ontology size looks imbalanced for its relation coverage.")
            recommendations.append(
                "Merge overly fragmented types or add relations that justify the type split."
            )

        return self._result(
            score,
            "Scores descriptions, examples/details, stable names, and compactness for extraction use.",
            findings,
            recommendations,
        )


DEFAULT_ONTOLOGY_EVALUATORS: tuple[OntologyMetricEvaluator, ...] = (
    SemanticCompletenessEvaluator(),
    SemanticConsistencyEvaluator(),
    ReusabilityEvaluator(),
    RelationQualityEvaluator(),
    PropertyModelingQualityEvaluator(),
    EventModelingQualityEvaluator(),
    ExtractionSuitabilityEvaluator(),
)


def evaluate_ontology(
    ontology: EnhancedDRGSchema | DRGSchema | dict[str, Any],
    *,
    evaluators: tuple[OntologyMetricEvaluator, ...] | list[OntologyMetricEvaluator] | None = None,
) -> OntologyEvaluationReport:
    """Evaluate an ontology/schema with reusable, domain-agnostic metrics."""

    context = _build_context(ontology)
    metric_results = tuple(
        evaluator.evaluate(context) for evaluator in (evaluators or DEFAULT_ONTOLOGY_EVALUATORS)
    )
    total_weight = sum(max(result.weight, 0.0) for result in metric_results)
    overall = (
        sum(_clamp01(result.score) * max(result.weight, 0.0) for result in metric_results)
        / total_weight
        if total_weight
        else 0.0
    )
    strengths = tuple(
        f"{result.metric}: {result.explanation}"
        for result in metric_results
        if result.score >= 0.80
    )
    weaknesses = tuple(
        f"{result.metric}: {result.explanation}" for result in metric_results if result.score < 0.60
    )
    recommendations = _unique(
        recommendation for result in metric_results for recommendation in result.recommendations
    )

    return OntologyEvaluationReport(
        overall_score=overall,
        metrics=metric_results,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendations=recommendations,
    )


def _build_context(
    ontology: EnhancedDRGSchema | DRGSchema | dict[str, Any],
) -> OntologyEvaluatorContext:
    if isinstance(ontology, EnhancedDRGSchema):
        entities = tuple(
            _EntityView(
                name=entity.name,
                description=entity.description,
                examples=tuple(entity.examples),
                properties=dict(entity.properties),
            )
            for entity in ontology.entity_types
        )
        relations = tuple(
            _RelationView(
                name=relation.name,
                source=relation.src,
                target=relation.dst,
                description=relation.description,
                detail=relation.detail,
                properties=dict(relation.properties),
                group=group.name,
            )
            for group in ontology.relation_groups
            for relation in group.relations
        )
        return OntologyEvaluatorContext(
            entities=entities,
            relations=relations,
            relation_groups=tuple(group.name for group in ontology.relation_groups),
            property_groups=tuple(group.properties for group in ontology.property_groups),
        )

    if isinstance(ontology, DRGSchema):
        return OntologyEvaluatorContext(
            entities=tuple(_EntityView(name=entity.name) for entity in ontology.entities),
            relations=tuple(
                _RelationView(
                    name=relation.name,
                    source=relation.src,
                    target=relation.dst,
                    description=relation.description,
                    detail=relation.detail,
                    properties=dict(relation.properties),
                )
                for relation in ontology.relations
            ),
        )

    if isinstance(ontology, dict):
        return _context_from_dict(ontology)

    raise TypeError(f"Unsupported ontology type: {type(ontology).__name__}")


def _context_from_dict(data: dict[str, Any]) -> OntologyEvaluatorContext:
    raw_entities = data.get("entity_types", data.get("entities", []))
    entities: list[_EntityView] = []
    if isinstance(raw_entities, list):
        for item in raw_entities:
            if isinstance(item, str):
                entities.append(_EntityView(name=item))
            elif isinstance(item, dict):
                examples = item.get("examples", ())
                if not isinstance(examples, list):
                    examples = []
                properties = item.get("properties", {})
                if not isinstance(properties, dict):
                    properties = {}
                entities.append(
                    _EntityView(
                        name=str(item.get("name", "")).strip(),
                        description=str(item.get("description", "")).strip(),
                        examples=tuple(str(example) for example in examples),
                        properties=properties,
                    )
                )

    relations: list[_RelationView] = []
    relation_groups: list[str] = []
    raw_relation_groups = data.get("relation_groups", [])
    if isinstance(raw_relation_groups, list):
        for group in raw_relation_groups:
            if not isinstance(group, dict):
                continue
            group_name = str(group.get("name", "")).strip()
            relation_groups.append(group_name)
            raw_relations = group.get("relations", [])
            if not isinstance(raw_relations, list):
                continue
            for relation in raw_relations:
                if isinstance(relation, dict):
                    relations.append(_relation_from_dict(relation, group_name))

    raw_relations = data.get("relations", [])
    if isinstance(raw_relations, list):
        for relation in raw_relations:
            if isinstance(relation, dict):
                relations.append(_relation_from_dict(relation, ""))

    raw_property_groups = data.get("property_groups", [])
    property_groups = tuple(group for group in raw_property_groups if isinstance(group, dict))
    return OntologyEvaluatorContext(
        entities=tuple(entities),
        relations=tuple(relations),
        relation_groups=tuple(name for name in relation_groups if name),
        property_groups=property_groups,
    )


def _relation_from_dict(data: dict[str, Any], group: str) -> _RelationView:
    properties = data.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    return _RelationView(
        name=str(data.get("name", data.get("relation", ""))).strip(),
        source=str(data.get("source", data.get("src", ""))).strip(),
        target=str(data.get("target", data.get("dst", ""))).strip(),
        description=str(data.get("description", "")).strip(),
        detail=str(data.get("detail", "")).strip(),
        properties=properties,
        group=group,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return 1.0 if denominator == 0 else _clamp01(float(numerator) / float(denominator))


def _meaningful_text(text: str) -> bool:
    words = [word for word in str(text).strip().split() if word]
    return len(words) >= 3


def _looks_instance_specific(text: str) -> bool:
    clean = str(text).strip()
    if not clean:
        return False
    words = clean.replace("_", " ").split()
    has_many_words = len(words) > 5
    has_digits = any(ch.isdigit() for ch in clean)
    has_quote_or_parenthetical = any(ch in clean for ch in "'\"()")
    mostly_title_tokens = (
        len(words) >= 4 and sum(word[:1].isupper() for word in words) >= len(words) - 1
    )
    return has_many_words or has_digits or has_quote_or_parenthetical or mostly_title_tokens


_LOW_INFORMATION_NAMES = {
    "entity",
    "object",
    "thing",
    "concept",
    "item",
    "relation",
    "relationship",
    "related_to",
    "associated_with",
    "has",
    "is",
    "type",
    "misc",
    "other",
}


def _is_low_information_name(name: str) -> bool:
    normalized = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    return normalized in _LOW_INFORMATION_NAMES


def _property_has_description(spec: Any) -> bool:
    if isinstance(spec, str):
        return _meaningful_text(spec)
    if isinstance(spec, dict):
        return any(
            _meaningful_text(str(spec.get(key, ""))) for key in ("description", "desc", "detail")
        )
    return False


_EVENT_TERMS = {
    "event",
    "process",
    "activity",
    "action",
    "incident",
    "transition",
    "change",
    "occurrence",
}

_EVENT_RELATION_TERMS = {
    "before",
    "after",
    "during",
    "causes",
    "caused",
    "leads",
    "trigger",
    "result",
    "participant",
    "occurs",
    "happens",
    "temporal",
    "period",
}


def _contains_any(text: str, terms: set[str]) -> bool:
    normalized = str(text).lower().replace("_", " ")
    return any(term in normalized for term in terms)


def _compactness_score(entity_count: int, relation_count: int) -> float:
    if entity_count == 0:
        return 0.0
    ratio = relation_count / entity_count
    if 0.5 <= ratio <= 2.5:
        return 1.0
    if ratio < 0.5:
        return _clamp01(ratio / 0.5)
    return _clamp01(2.5 / ratio)


def _unique(items: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return tuple(unique)
