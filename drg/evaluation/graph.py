"""Graph quality evaluation for persisted EnhancedKG-style JSON.

This evaluator is intentionally deterministic and gold-label-free. It answers a
different question than extraction F1: "does this graph look internally usable
and auditable enough to trust?" Controlled bad-graph tests should make the score
drop and produce actionable recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..graph.validation import validate_graph_data

__all__ = [
    "GraphEvaluationReport",
    "evaluate_graph_quality",
]


@dataclass(frozen=True)
class GraphEvaluationReport:
    """Aggregated graph quality score, findings, and recommendations."""

    overall_score: float
    findings: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    validation_summary: dict[str, int] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": round(_clamp01(self.overall_score), 4),
            "findings": list(self.findings),
            "recommendations": list(self.recommendations),
            "validation_summary": dict(self.validation_summary),
            "checks": self.checks,
        }


def evaluate_graph_quality(
    graph: dict[str, Any],
    *,
    schema: Any | None = None,
) -> GraphEvaluationReport:
    """Evaluate structural, ontology, confidence, evidence, and provenance quality."""

    validation = validate_graph_data(graph)
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
    node_types = _node_types(nodes)
    allowed_entity_types, allowed_relations = _schema_constraints(schema)

    findings: list[str] = []
    recommendations: list[str] = []
    penalties = 0.0

    for issue in validation.issues:
        if issue.severity == "info":
            continue
        findings.append(f"{issue.code}: {issue.message}")
        penalties += 0.12 if issue.severity == "error" else 0.05
        if issue.code == "duplicate_node_id":
            recommendations.append("Deduplicate node ids before publishing benchmark results.")

    unknown_types = sorted(
        {
            str(node.get("type"))
            for node in nodes
            if node.get("type")
            and allowed_entity_types
            and node.get("type") not in allowed_entity_types
        }
    )
    if unknown_types:
        penalties += 0.08 * len(unknown_types)
        findings.append(f"unknown_entity_type: {', '.join(unknown_types)}")
        recommendations.append("Map graph node types to the generated ontology before evaluation.")

    invalid_relations = _invalid_relations(edges, node_types, allowed_relations)
    if invalid_relations:
        penalties += 0.12 * len(invalid_relations)
        findings.append(
            f"invalid_relation: {len(invalid_relations)} relation violates the ontology."
        )
        recommendations.append(
            "Remove invalid relations or update the ontology with valid endpoints."
        )

    zero_confidence = _zero_confidence_count(nodes, edges)
    if zero_confidence:
        penalties += 0.06 * zero_confidence
        findings.append(f"zero_confidence: {zero_confidence} graph element has confidence 0.")
        recommendations.append(
            "Calibrate confidence scores; reserve 0.0 for known-bad extractions."
        )

    missing_evidence = _missing_metadata_count(nodes, edges, "evidence")
    if missing_evidence:
        penalties += 0.05 * missing_evidence
        findings.append(f"missing_evidence: {missing_evidence} graph element lacks evidence.")
        recommendations.append("Attach evidence snippets or source spans to every extracted fact.")

    missing_provenance = _missing_provenance_count(nodes, edges)
    if missing_provenance:
        penalties += 0.05 * missing_provenance
        findings.append(f"missing_provenance: {missing_provenance} graph element lacks provenance.")
        recommendations.append("Attach document/chunk provenance so graph facts are auditable.")

    score = _clamp01(1.0 - penalties)
    if not findings:
        findings.append("graph_quality_ok: graph passed quality checks.")

    return GraphEvaluationReport(
        overall_score=score,
        findings=tuple(findings),
        recommendations=tuple(dict.fromkeys(recommendations)),
        validation_summary={
            "errors": validation.error_count,
            "warnings": validation.warning_count,
            "info": validation.info_count,
        },
        checks={
            "unknown_entity_types": unknown_types,
            "invalid_relations": invalid_relations,
            "zero_confidence": zero_confidence,
            "missing_evidence": missing_evidence,
            "missing_provenance": missing_provenance,
        },
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _node_types(nodes: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(node.get("id")): str(node.get("type"))
        for node in nodes
        if isinstance(node.get("id"), str) and node.get("type")
    }


def _schema_constraints(schema: Any | None) -> tuple[set[str], set[tuple[str, str, str]]]:
    if schema is None:
        return set(), set()

    if isinstance(schema, dict):
        raw_entities = schema.get("entity_types", schema.get("entities", []))
        if not isinstance(raw_entities, list):
            raw_entities = []
        entity_types = {
            str(item.get("name"))
            for item in raw_entities
            if isinstance(item, dict) and item.get("name")
        }
        relations: set[tuple[str, str, str]] = set()
        raw_relation_groups = schema.get("relation_groups", [])
        if not isinstance(raw_relation_groups, list):
            raw_relation_groups = []
        for group in raw_relation_groups:
            if not isinstance(group, dict):
                continue
            raw_relations = group.get("relations", [])
            if not isinstance(raw_relations, list):
                continue
            for rel in raw_relations:
                if isinstance(rel, dict):
                    relations.add(
                        (
                            str(rel.get("source", rel.get("src", ""))),
                            str(rel.get("name", rel.get("type", ""))),
                            str(rel.get("target", rel.get("dst", ""))),
                        )
                    )
        top_level_relations = schema.get("relations", [])
        if not isinstance(top_level_relations, list):
            top_level_relations = []
        for rel in top_level_relations:
            if isinstance(rel, dict):
                relations.add(
                    (
                        str(rel.get("source", rel.get("src", ""))),
                        str(rel.get("name", rel.get("type", ""))),
                        str(rel.get("target", rel.get("dst", ""))),
                    )
                )
        return entity_types, {rel for rel in relations if all(rel)}

    entity_types = {item.name for item in getattr(schema, "entity_types", [])}
    entity_types.update(item.name for item in getattr(schema, "entities", []))
    relations = {
        (getattr(rel, "src", ""), getattr(rel, "name", ""), getattr(rel, "dst", ""))
        for group in getattr(schema, "relation_groups", [])
        for rel in getattr(group, "relations", [])
    }
    relations.update(
        (getattr(rel, "src", ""), getattr(rel, "name", ""), getattr(rel, "dst", ""))
        for rel in getattr(schema, "relations", [])
    )
    return {str(item) for item in entity_types if item}, {rel for rel in relations if all(rel)}


def _invalid_relations(
    edges: list[dict[str, Any]],
    node_types: dict[str, str],
    allowed_relations: set[tuple[str, str, str]],
) -> list[dict[str, str]]:
    if not allowed_relations:
        return []
    invalid: list[dict[str, str]] = []
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        relation = str(edge.get("relationship_type", ""))
        key = (node_types.get(source, ""), relation, node_types.get(target, ""))
        if all(key) and key not in allowed_relations:
            invalid.append(
                {
                    "source": source,
                    "relationship_type": relation,
                    "target": target,
                    "source_type": key[0],
                    "target_type": key[2],
                }
            )
    return invalid


def _zero_confidence_count(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> int:
    return sum(1 for item in [*nodes, *edges] if item.get("confidence") == 0)


def _missing_metadata_count(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    key: str,
) -> int:
    count = 0
    for item in [*nodes, *edges]:
        metadata = item.get("metadata")
        if not (isinstance(metadata, dict) and metadata.get(key)):
            count += 1
    return count


def _missing_provenance_count(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> int:
    count = 0
    for item in [*nodes, *edges]:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            count += 1
            continue
        provenance = metadata.get("provenance")
        legacy_docs = metadata.get("source_documents")
        has_structured = isinstance(provenance, dict) and bool(provenance)
        has_legacy = isinstance(legacy_docs, list) and bool(legacy_docs)
        if not (has_structured or has_legacy or metadata.get("source_ref")):
            count += 1
    return count
