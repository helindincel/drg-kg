"""Tests for the multi-document reasoning layer (``drg.reasoning``).

All tests are deterministic and require **no LLM** and **no optional
heavy dependencies**. They exercise:

- The headline use case from the design doc — Apple/Beats/Jimmy Iovine
  cross-document reasoning.
- Every built-in :class:`drg.reasoning.InferenceRule` in isolation.
- Engine invariants: idempotency, JSON round-trip, conservative
  defaults, provenance shape, dry-run mode.
- Pipeline glue: ``build_enhanced_kg(document_id=...)`` stamps
  ``source_ref`` on edges and ``source_documents`` on nodes; the
  ``GraphMerger`` propagates them across documents.
- Backward-compatibility safety net: calling the legacy entry points
  without any of the new options preserves the historical behaviour.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from drg.graph import (
    EnhancedKG,
    GraphMerger,
    KGEdge,
    KGNode,
)
from drg.graph.builders import build_enhanced_kg
from drg.reasoning import (
    INVERSE_RELATION_PAIRS,
    EvidenceLink,
    InferenceReport,
    InferenceRule,
    InferredEdge,
    InverseRule,
    MultiDocumentReasoner,
    PathBridgeRule,
    ReasoningConfig,
    default_rules,
    reason_over_graph,
)
from drg.reasoning._explain import (
    explain_composition,
    explain_inverse,
    explain_path_bridge,
    explain_symmetric,
    explain_transitive,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_edge(
    source: str,
    rel: str,
    target: str,
    *,
    source_ref: str | None = None,
    confidence: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> KGEdge:
    metadata: dict[str, Any] = {}
    if source_ref:
        metadata["source_ref"] = source_ref
    if extra_metadata:
        metadata.update(extra_metadata)
    return KGEdge(
        source=source,
        target=target,
        relationship_type=rel,
        relationship_detail=f"{source} {rel} {target}",
        metadata=metadata,
        confidence=confidence,
    )


def _make_two_doc_kg() -> EnhancedKG:
    """Apple/Beats/Jimmy Iovine across two documents."""
    base = EnhancedKG()
    ka = EnhancedKG()
    ka.add_node(KGNode(id="Apple", type="Company"))
    ka.add_node(KGNode(id="Beats", type="Company"))
    ka.add_edge(
        _make_edge("Apple", "ACQUIRED", "Beats", source_ref="doc_A", confidence=0.95)
    )

    kb = EnhancedKG()
    kb.add_node(KGNode(id="Jimmy Iovine", type="Person"))
    kb.add_node(KGNode(id="Beats", type="Company"))
    kb.add_edge(
        _make_edge(
            "Jimmy Iovine", "FOUNDED", "Beats", source_ref="doc_B", confidence=0.9
        )
    )

    GraphMerger().merge(base, ka, document_id="doc_A")
    GraphMerger().merge(base, kb, document_id="doc_B")
    return base


def _inferred_edges(kg: EnhancedKG) -> list[KGEdge]:
    return [e for e in kg.edges if e.metadata.get("inferred")]


def _extracted_edges(kg: EnhancedKG) -> list[KGEdge]:
    return [e for e in kg.edges if not e.metadata.get("inferred")]


# ---------------------------------------------------------------------------
# Headline use case
# ---------------------------------------------------------------------------


class TestAppleBeatsJimmyIovine:
    """Verify the multi-document inference example from the design doc.

    Doc A: Apple ACQUIRED Beats.
    Doc B: Jimmy Iovine FOUNDED Beats.
    Expectation: a single `connected_via_beats` edge linking Apple and
    Jimmy Iovine, with full provenance covering both documents.
    """

    def test_bridge_rule_creates_apple_to_jimmy_edge(self):
        kg = _make_two_doc_kg()

        # Disable everything except path_bridge so the assertion isolates
        # the cross-document inference path.
        config = ReasoningConfig(
            disabled_rules=frozenset({"inverse", "symmetric", "transitive", "composition"})
        )
        report = MultiDocumentReasoner(config=config).reason(kg)
        bridge_edges = [
            e for e in _inferred_edges(kg) if e.relationship_type.startswith("connected_via_")
        ]
        assert len(bridge_edges) == 1
        edge = bridge_edges[0]
        assert {edge.source, edge.target} == {"Apple", "Jimmy Iovine"}
        assert edge.relationship_type == "connected_via_beats"
        assert report.added_edges  # populated

    def test_bridge_edge_carries_provenance(self):
        kg = _make_two_doc_kg()
        config = ReasoningConfig(
            disabled_rules=frozenset({"inverse", "symmetric", "transitive", "composition"})
        )
        MultiDocumentReasoner(config=config).reason(kg)
        edge = next(
            e for e in _inferred_edges(kg) if e.relationship_type.startswith("connected_via_")
        )

        inf = edge.metadata["inference"]
        assert inf["rule"] == "path_bridge"
        assert inf["bridge_entity"] == "Beats"
        assert set(inf["source_documents"]) == {"doc_A", "doc_B"}
        assert len(inf["evidence_chain"]) == 2
        triples_cited = {tuple(link["triple"]) for link in inf["evidence_chain"]}
        assert ("Apple", "ACQUIRED", "Beats") in triples_cited
        assert ("Jimmy Iovine", "FOUNDED", "Beats") in triples_cited
        assert "Beats" in inf["explanation"]
        assert "doc_A" in inf["explanation"]
        assert "doc_B" in inf["explanation"]

    def test_bridge_edge_is_distinguishable_from_extracted_edges(self):
        kg = _make_two_doc_kg()
        MultiDocumentReasoner().reason(kg)

        extracted = _extracted_edges(kg)
        inferred = _inferred_edges(kg)
        assert len(extracted) == 2
        assert all(not e.metadata.get("inferred") for e in extracted)
        assert all(e.metadata.get("inferred") is True for e in inferred)
        for e in inferred:
            assert "inference" in e.metadata
            assert {
                "rule",
                "evidence_chain",
                "source_documents",
                "explanation",
                "confidence",
            }.issubset(e.metadata["inference"].keys())


# ---------------------------------------------------------------------------
# PathBridgeRule edge cases
# ---------------------------------------------------------------------------


class TestPathBridgeRule:
    def test_does_not_fire_when_both_edges_from_same_document(self):
        """A bridge entity from the same document is not multi-document
        evidence — the rule must abstain to avoid fabricating links from
        in-document signal that the LLM didn't extract directly."""
        kg = EnhancedKG()
        kg.add_node(KGNode(id="A"))
        kg.add_node(KGNode(id="B"))
        kg.add_node(KGNode(id="X"))
        kg.add_edge(_make_edge("A", "r1", "X", source_ref="doc_same", confidence=0.9))
        kg.add_edge(_make_edge("B", "r2", "X", source_ref="doc_same", confidence=0.9))

        MultiDocumentReasoner(
            config=ReasoningConfig(
                disabled_rules=frozenset({"inverse", "symmetric", "transitive", "composition"})
            )
        ).reason(kg)
        assert not _inferred_edges(kg)

    def test_does_not_fire_when_edge_missing_source_ref(self):
        """No source_ref means we can't know which document an edge came
        from. The bridge rule must abstain rather than guess."""
        kg = EnhancedKG()
        kg.add_node(KGNode(id="A"))
        kg.add_node(KGNode(id="B"))
        kg.add_node(KGNode(id="X"))
        kg.add_edge(_make_edge("A", "r1", "X", source_ref="doc_A", confidence=0.9))
        kg.add_edge(_make_edge("B", "r2", "X", confidence=0.9))  # no source_ref

        MultiDocumentReasoner(
            config=ReasoningConfig(
                disabled_rules=frozenset({"inverse", "symmetric", "transitive", "composition"})
            )
        ).reason(kg)
        assert not _inferred_edges(kg)

    def test_does_not_fire_on_identical_relations_when_distinct_relations_required(self):
        kg = EnhancedKG()
        kg.add_node(KGNode(id="A"))
        kg.add_node(KGNode(id="B"))
        kg.add_node(KGNode(id="X"))
        kg.add_edge(_make_edge("A", "OWNS", "X", source_ref="doc_A", confidence=0.9))
        kg.add_edge(_make_edge("B", "OWNS", "X", source_ref="doc_B", confidence=0.9))

        MultiDocumentReasoner(
            config=ReasoningConfig(
                disabled_rules=frozenset({"inverse", "symmetric", "transitive", "composition"})
            )
        ).reason(kg)
        # Same relation; conservative default refuses
        assert not [e for e in _inferred_edges(kg) if e.metadata["inference"]["rule"] == "path_bridge"]

    def test_fires_on_identical_relations_when_distinct_check_disabled(self):
        kg = EnhancedKG()
        kg.add_node(KGNode(id="A"))
        kg.add_node(KGNode(id="B"))
        kg.add_node(KGNode(id="X"))
        kg.add_edge(_make_edge("A", "OWNS", "X", source_ref="doc_A", confidence=0.9))
        kg.add_edge(_make_edge("B", "OWNS", "X", source_ref="doc_B", confidence=0.9))

        config = ReasoningConfig(
            disabled_rules=frozenset({"inverse", "symmetric", "transitive", "composition"}),
            require_distinct_bridge_relations=False,
        )
        MultiDocumentReasoner(config=config).reason(kg)
        bridge = [
            e for e in _inferred_edges(kg)
            if e.metadata["inference"]["rule"] == "path_bridge"
        ]
        assert len(bridge) == 1

    def test_deterministic_endpoint_ordering(self):
        """The rule should pick the lexicographic-smaller endpoint as the
        source so reruns produce identical edges."""
        kg = EnhancedKG()
        kg.add_node(KGNode(id="Zeta"))
        kg.add_node(KGNode(id="Alpha"))
        kg.add_node(KGNode(id="Bridge"))
        kg.add_edge(_make_edge("Zeta", "r1", "Bridge", source_ref="d1", confidence=0.9))
        kg.add_edge(_make_edge("Alpha", "r2", "Bridge", source_ref="d2", confidence=0.9))

        MultiDocumentReasoner(
            config=ReasoningConfig(
                disabled_rules=frozenset({"inverse", "symmetric", "transitive", "composition"})
            )
        ).reason(kg)
        bridge = next(
            e for e in _inferred_edges(kg)
            if e.metadata["inference"]["rule"] == "path_bridge"
        )
        assert bridge.source == "Alpha"
        assert bridge.target == "Zeta"

    def test_per_bridge_cap_limits_combinatorial_blowup(self):
        kg = EnhancedKG()
        kg.add_node(KGNode(id="Hub"))
        for i in range(10):
            kg.add_node(KGNode(id=f"N{i}"))
            kg.add_edge(
                _make_edge(f"N{i}", f"r{i}", "Hub", source_ref=f"doc_{i}", confidence=0.9)
            )

        # cap=3 selects only the first 3 incident edges; 3 incident
        # edges -> C(3, 2) = 3 candidate pairs.
        config = ReasoningConfig(
            disabled_rules=frozenset({"inverse", "symmetric", "transitive", "composition"}),
            max_bridge_candidates_per_node=3,
        )
        MultiDocumentReasoner(config=config).reason(kg)
        assert len(_inferred_edges(kg)) == 3


# ---------------------------------------------------------------------------
# InverseRule
# ---------------------------------------------------------------------------


class TestInverseRule:
    def test_inverse_pair_emitted(self):
        kg = EnhancedKG()
        kg.add_node(KGNode(id="Steve Jobs"))
        kg.add_node(KGNode(id="Apple"))
        kg.add_edge(_make_edge("Steve Jobs", "founded", "Apple", source_ref="d1", confidence=0.92))

        MultiDocumentReasoner(
            config=ReasoningConfig(
                disabled_rules=frozenset({"path_bridge", "symmetric", "transitive", "composition"})
            )
        ).reason(kg)
        inferred = _inferred_edges(kg)
        assert len(inferred) == 1
        e = inferred[0]
        assert e.source == "Apple"
        assert e.target == "Steve Jobs"
        assert e.relationship_type == "founded_by"
        assert e.metadata["inference"]["rule"] == "inverse"
        assert e.confidence == 0.92

    def test_does_not_emit_when_inverse_already_present(self):
        kg = EnhancedKG()
        kg.add_node(KGNode(id="A"))
        kg.add_node(KGNode(id="B"))
        kg.add_edge(_make_edge("A", "founded", "B", source_ref="d1"))
        kg.add_edge(_make_edge("B", "founded_by", "A", source_ref="d1"))

        MultiDocumentReasoner(
            config=ReasoningConfig(
                disabled_rules=frozenset({"path_bridge", "symmetric", "transitive", "composition"})
            )
        ).reason(kg)
        assert not _inferred_edges(kg)

    @pytest.mark.parametrize(
        "src_rel,expected_inverse",
        list(INVERSE_RELATION_PAIRS.items())[:6],
    )
    def test_inverse_pair_table_covers_common_predicates(self, src_rel, expected_inverse):
        kg = EnhancedKG()
        kg.add_node(KGNode(id="A"))
        kg.add_node(KGNode(id="B"))
        kg.add_edge(_make_edge("A", src_rel, "B", source_ref="d1"))

        MultiDocumentReasoner(
            config=ReasoningConfig(
                disabled_rules=frozenset({"path_bridge", "symmetric", "transitive", "composition"})
            )
        ).reason(kg)
        assert any(
            e.relationship_type == expected_inverse and e.source == "B" and e.target == "A"
            for e in _inferred_edges(kg)
        )


# ---------------------------------------------------------------------------
# SymmetricRule
# ---------------------------------------------------------------------------


class TestSymmetricRule:
    def test_emits_back_edge_for_symmetric_predicate(self):
        kg = EnhancedKG()
        kg.add_node(KGNode(id="Alice"))
        kg.add_node(KGNode(id="Bob"))
        kg.add_edge(_make_edge("Alice", "collaborates_with", "Bob", source_ref="d1"))

        MultiDocumentReasoner(
            config=ReasoningConfig(
                disabled_rules=frozenset({"path_bridge", "inverse", "transitive", "composition"})
            )
        ).reason(kg)
        inferred = _inferred_edges(kg)
        assert len(inferred) == 1
        assert inferred[0].source == "Bob"
        assert inferred[0].target == "Alice"
        assert inferred[0].relationship_type == "collaborates_with"

    def test_does_not_emit_for_asymmetric_predicate(self):
        kg = EnhancedKG()
        kg.add_node(KGNode(id="A"))
        kg.add_node(KGNode(id="B"))
        kg.add_edge(_make_edge("A", "owns", "B", source_ref="d1"))

        MultiDocumentReasoner(
            config=ReasoningConfig(
                disabled_rules=frozenset({"path_bridge", "inverse", "transitive", "composition"})
            )
        ).reason(kg)
        assert not _inferred_edges(kg)

    def test_does_not_double_emit_when_both_directions_present(self):
        kg = EnhancedKG()
        kg.add_node(KGNode(id="A"))
        kg.add_node(KGNode(id="B"))
        kg.add_edge(_make_edge("A", "works_with", "B", source_ref="d1"))
        kg.add_edge(_make_edge("B", "works_with", "A", source_ref="d2"))

        MultiDocumentReasoner(
            config=ReasoningConfig(
                disabled_rules=frozenset({"path_bridge", "inverse", "transitive", "composition"})
            )
        ).reason(kg)
        assert not _inferred_edges(kg)


# ---------------------------------------------------------------------------
# TransitiveRule
# ---------------------------------------------------------------------------


class TestTransitiveRule:
    def test_single_hop_closure(self):
        kg = EnhancedKG()
        kg.add_node(KGNode(id="Eiffel Tower"))
        kg.add_node(KGNode(id="Paris"))
        kg.add_node(KGNode(id="France"))
        kg.add_edge(_make_edge("Eiffel Tower", "part_of", "Paris", source_ref="d1"))
        kg.add_edge(_make_edge("Paris", "part_of", "France", source_ref="d2"))

        MultiDocumentReasoner(
            config=ReasoningConfig(
                disabled_rules=frozenset({"path_bridge", "inverse", "symmetric", "composition"})
            )
        ).reason(kg)
        inferred = _inferred_edges(kg)
        triples = {(e.source, e.relationship_type, e.target) for e in inferred}
        assert ("Eiffel Tower", "part_of", "France") in triples

    def test_one_hop_per_run_for_chain_of_three(self):
        """Three-step chain → one hop emitted per run (deterministic;
        callers can re-run to reach the full closure)."""
        kg = EnhancedKG()
        for n in ["A", "B", "C", "D"]:
            kg.add_node(KGNode(id=n))
        kg.add_edge(_make_edge("A", "part_of", "B", source_ref="d1"))
        kg.add_edge(_make_edge("B", "part_of", "C", source_ref="d2"))
        kg.add_edge(_make_edge("C", "part_of", "D", source_ref="d3"))

        config = ReasoningConfig(
            disabled_rules=frozenset({"path_bridge", "inverse", "symmetric", "composition"})
        )
        reasoner = MultiDocumentReasoner(config=config)
        reasoner.reason(kg)
        # After 1 pass: have A-B, B-C, C-D (extracted) + A-C, B-D (inferred). NOT A-D yet.
        triples = {(e.source, e.relationship_type, e.target) for e in kg.edges}
        assert ("A", "part_of", "C") in triples
        assert ("B", "part_of", "D") in triples
        # A-D requires chaining off an inferred edge; default config refuses.
        assert ("A", "part_of", "D") not in triples

    def test_allows_inferred_chaining_when_opted_in(self):
        kg = EnhancedKG()
        for n in ["A", "B", "C", "D"]:
            kg.add_node(KGNode(id=n))
        kg.add_edge(_make_edge("A", "part_of", "B", source_ref="d1"))
        kg.add_edge(_make_edge("B", "part_of", "C", source_ref="d2"))
        kg.add_edge(_make_edge("C", "part_of", "D", source_ref="d3"))

        config = ReasoningConfig(
            disabled_rules=frozenset({"path_bridge", "inverse", "symmetric", "composition"}),
            allow_inferred_in_evidence=True,
        )
        reasoner = MultiDocumentReasoner(config=config)
        # Two passes — one to produce A-C / B-D, another to consume them
        reasoner.reason(kg)
        reasoner.reason(kg)
        triples = {(e.source, e.relationship_type, e.target) for e in kg.edges}
        assert ("A", "part_of", "D") in triples


# ---------------------------------------------------------------------------
# CompositionRule
# ---------------------------------------------------------------------------


class TestCompositionRule:
    def test_owns_plus_located_in_emits_operates_in(self):
        kg = EnhancedKG()
        for n in ["Apple", "Apple Park", "Cupertino"]:
            kg.add_node(KGNode(id=n))
        kg.add_edge(_make_edge("Apple", "owns", "Apple Park", source_ref="d1", confidence=0.9))
        kg.add_edge(
            _make_edge("Apple Park", "located_in", "Cupertino", source_ref="d2", confidence=0.9)
        )

        MultiDocumentReasoner(
            config=ReasoningConfig(
                disabled_rules=frozenset({"path_bridge", "inverse", "symmetric", "transitive"})
            )
        ).reason(kg)
        inferred = _inferred_edges(kg)
        assert any(
            e.source == "Apple"
            and e.target == "Cupertino"
            and e.relationship_type == "operates_in"
            for e in inferred
        )

    def test_does_not_emit_when_intermediate_missing(self):
        kg = EnhancedKG()
        for n in ["A", "L"]:
            kg.add_node(KGNode(id=n))
        # Only one edge — no composition possible
        kg.add_edge(_make_edge("A", "owns", "L", source_ref="d1"))

        MultiDocumentReasoner(
            config=ReasoningConfig(
                disabled_rules=frozenset({"path_bridge", "inverse", "symmetric", "transitive"})
            )
        ).reason(kg)
        assert not _inferred_edges(kg)


# ---------------------------------------------------------------------------
# Engine invariants
# ---------------------------------------------------------------------------


class TestEngineInvariants:
    def test_idempotent_across_runs(self):
        kg = _make_two_doc_kg()
        report1 = MultiDocumentReasoner().reason(kg)
        edge_count_after_first = len(kg.edges)
        report2 = MultiDocumentReasoner().reason(kg)

        # First call adds edges; second call must be a no-op.
        assert report1.added_edges
        assert not report2.added_edges
        assert len(kg.edges) == edge_count_after_first

    def test_does_not_mutate_extracted_edges(self):
        kg = _make_two_doc_kg()
        snapshot = [
            (e.source, e.relationship_type, e.target, dict(e.metadata), e.confidence)
            for e in kg.edges
        ]
        MultiDocumentReasoner().reason(kg)

        # Re-check the first N edges (the original extracted ones) are byte-identical
        for original, edge in zip(snapshot, kg.edges[: len(snapshot)], strict=True):
            o_src, o_rel, o_tgt, o_meta, o_conf = original
            assert edge.source == o_src
            assert edge.relationship_type == o_rel
            assert edge.target == o_tgt
            assert edge.metadata == o_meta
            assert edge.confidence == o_conf

    def test_dry_run_does_not_mutate(self):
        kg = _make_two_doc_kg()
        before = len(kg.edges)
        report = MultiDocumentReasoner().reason(kg, dry_run=True)
        assert len(kg.edges) == before
        assert report.added_edges  # candidates reported

    def test_endpoint_validation_drops_missing_nodes(self):
        kg = EnhancedKG()
        kg.add_node(KGNode(id="A"))
        kg.add_node(KGNode(id="X"))

        # Construct a rule that emits an edge whose endpoint is missing.
        class GhostRule(InferenceRule):
            name = "ghost"

            def apply(self, kg, config):
                return [
                    InferredEdge(
                        source="A",
                        target="GHOST",  # not in graph
                        relationship_type="r",
                        rule_name=self.name,
                        evidence_chain=[EvidenceLink(triple=("A", "r", "X"), source_ref="d1")],
                        explanation="ghost",
                        confidence=0.9,
                    )
                ]

        report = MultiDocumentReasoner(rules=[GhostRule()]).reason(kg)
        assert not report.added_edges
        assert report.skipped_missing_endpoint

    def test_disabled_rule_is_not_invoked(self):
        kg = EnhancedKG()
        kg.add_node(KGNode(id="A"))
        kg.add_node(KGNode(id="B"))
        kg.add_edge(_make_edge("A", "founded", "B", source_ref="d1"))

        config = ReasoningConfig(disabled_rules=frozenset({"inverse"}))
        MultiDocumentReasoner(config=config).reason(kg)
        # InverseRule disabled -> no founded_by inferred
        assert not _inferred_edges(kg)

    def test_low_confidence_candidate_is_dropped(self):
        kg = EnhancedKG()
        kg.add_node(KGNode(id="A"))
        kg.add_node(KGNode(id="B"))
        kg.add_edge(_make_edge("A", "founded", "B", source_ref="d1", confidence=0.2))

        config = ReasoningConfig(
            disabled_rules=frozenset({"path_bridge", "symmetric", "transitive", "composition"}),
            min_confidence=0.5,
        )
        report = MultiDocumentReasoner(config=config).reason(kg)
        assert not _inferred_edges(kg)
        assert report.skipped_low_confidence

    def test_max_inferences_caps_output(self):
        kg = EnhancedKG()
        for i in range(5):
            kg.add_node(KGNode(id=f"P{i}"))
            kg.add_node(KGNode(id=f"C{i}"))
            kg.add_edge(_make_edge(f"P{i}", "founded", f"C{i}", source_ref=f"d{i}"))

        config = ReasoningConfig(
            disabled_rules=frozenset({"path_bridge", "symmetric", "transitive", "composition"}),
            max_inferences_per_run=2,
        )
        report = MultiDocumentReasoner(config=config).reason(kg)
        assert len(report.added_edges) == 2

    def test_rule_exception_is_isolated(self):
        """A buggy rule must not crash the engine — it skips and logs."""

        class CrashingRule(InferenceRule):
            name = "crash"

            def apply(self, kg, config):
                raise RuntimeError("simulated crash")

        kg = EnhancedKG()
        kg.add_node(KGNode(id="A"))
        kg.add_node(KGNode(id="B"))
        kg.add_edge(_make_edge("A", "founded", "B", source_ref="d1"))

        # Combine the crashing rule with InverseRule so we can prove the
        # latter still runs after the former fails.
        report = MultiDocumentReasoner(
            rules=[CrashingRule(), InverseRule()],
        ).reason(kg)
        assert report.per_rule_counts.get("inverse") == 1


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_inferred_edges_survive_save_and_load(self, tmp_path):
        kg = _make_two_doc_kg()
        MultiDocumentReasoner().reason(kg)

        original_inferred = sorted(
            (e.source, e.relationship_type, e.target) for e in _inferred_edges(kg)
        )
        path = tmp_path / "kg.json"
        kg.save_json(str(path))
        reloaded = EnhancedKG.load_json(str(path))

        roundtrip_inferred = sorted(
            (e.source, e.relationship_type, e.target) for e in _inferred_edges(reloaded)
        )
        assert original_inferred == roundtrip_inferred

        # Every inferred edge must still carry full provenance.
        for e in _inferred_edges(reloaded):
            inf = e.metadata["inference"]
            assert "rule" in inf
            assert "evidence_chain" in inf
            assert "source_documents" in inf
            assert "explanation" in inf

    def test_inference_metadata_is_json_serialisable(self):
        kg = _make_two_doc_kg()
        MultiDocumentReasoner().reason(kg)
        for e in _inferred_edges(kg):
            json.dumps(e.metadata)  # raises if non-serialisable


# ---------------------------------------------------------------------------
# Builders + merger integration
# ---------------------------------------------------------------------------


class TestBuildersIntegration:
    def test_build_enhanced_kg_stamps_source_ref(self):
        kg = build_enhanced_kg(
            entities_typed=[("A", "T"), ("B", "T")],
            triples=[("A", "rel", "B")],
            document_id="doc_X",
        )
        edge = kg.edges[0]
        assert edge.metadata["source_ref"] == "doc_X"
        assert kg.nodes["A"].metadata["source_documents"] == ["doc_X"]
        assert kg.nodes["B"].metadata["source_documents"] == ["doc_X"]

    def test_build_enhanced_kg_without_document_id_is_byte_compatible(self):
        kg = build_enhanced_kg(
            entities_typed=[("A", "T"), ("B", "T")],
            triples=[("A", "rel", "B")],
        )
        # Legacy behaviour — no source_ref / source_documents keys.
        assert "source_ref" not in kg.edges[0].metadata
        assert "source_documents" not in kg.nodes["A"].metadata

    def test_merger_unions_source_documents_on_existing_nodes(self):
        base = EnhancedKG()
        base.add_node(KGNode(id="Apple", metadata={"source_documents": ["doc_A"]}))
        incoming = EnhancedKG()
        incoming.add_node(KGNode(id="Apple", metadata={"source_documents": ["doc_B"]}))

        GraphMerger().merge(base, incoming, document_id="doc_B")
        assert base.nodes["Apple"].metadata["source_documents"] == ["doc_A", "doc_B"]

    def test_merger_stamps_source_ref_on_edge_lacking_one(self):
        base = EnhancedKG()
        base.add_node(KGNode(id="A"))
        base.add_node(KGNode(id="B"))
        incoming = EnhancedKG()
        incoming.add_node(KGNode(id="A"))
        incoming.add_node(KGNode(id="B"))
        # Edge built without a source_ref:
        incoming.add_edge(KGEdge(source="A", target="B", relationship_type="r",
                                  relationship_detail="A r B"))
        GraphMerger().merge(base, incoming, document_id="doc_X")
        assert base.edges[0].metadata.get("source_ref") == "doc_X"

    def test_pipeline_apple_beats_jimmy_via_builders_and_merger(self):
        # Document A: Apple ACQUIRED Beats
        doc_a = build_enhanced_kg(
            entities_typed=[("Apple", "Company"), ("Beats", "Company")],
            triples=[("Apple", "ACQUIRED", "Beats")],
            document_id="doc_A",
        )
        # Document B: Jimmy Iovine FOUNDED Beats
        doc_b = build_enhanced_kg(
            entities_typed=[("Jimmy Iovine", "Person"), ("Beats", "Company")],
            triples=[("Jimmy Iovine", "FOUNDED", "Beats")],
            document_id="doc_B",
        )
        base = EnhancedKG()
        GraphMerger().merge(base, doc_a, document_id="doc_A")
        GraphMerger().merge(base, doc_b, document_id="doc_B")

        MultiDocumentReasoner(
            config=ReasoningConfig(
                disabled_rules=frozenset({"inverse", "symmetric", "transitive", "composition"})
            )
        ).reason(base)
        bridges = [
            e for e in _inferred_edges(base) if e.metadata["inference"]["rule"] == "path_bridge"
        ]
        assert len(bridges) == 1
        edge = bridges[0]
        assert {edge.source, edge.target} == {"Apple", "Jimmy Iovine"}
        assert set(edge.metadata["inference"]["source_documents"]) == {"doc_A", "doc_B"}


# ---------------------------------------------------------------------------
# History & explain
# ---------------------------------------------------------------------------


class TestHistoryAndExplain:
    def test_history_entry_is_appended_after_run(self):
        kg = _make_two_doc_kg()
        before = list(kg.metadata.get("history", []))
        MultiDocumentReasoner().reason(kg, document_id="doc_AB")
        after = kg.metadata["history"]
        assert len(after) == len(before) + 1
        entry = after[-1]
        assert entry["operation"] == "reasoning"
        assert entry["document_id"] == "doc_AB"
        assert entry["added_edges"] > 0

    def test_no_history_entry_when_report_empty(self):
        # No bridge edges possible — bridge entity missing.
        kg = EnhancedKG()
        kg.add_node(KGNode(id="X"))
        before_history = list(kg.metadata.get("history", []))
        MultiDocumentReasoner().reason(kg, document_id="doc_empty")
        # Empty report => no history entry written
        assert kg.metadata.get("history", []) == before_history

    def test_explain_helpers_render_doc_ids(self):
        link_a = EvidenceLink(
            triple=("Apple", "ACQUIRED", "Beats"), source_ref="doc_A", confidence=0.95
        )
        link_b = EvidenceLink(
            triple=("Jimmy Iovine", "FOUNDED", "Beats"), source_ref="doc_B", confidence=0.9
        )

        bridge = explain_path_bridge(
            src="Apple", dst="Jimmy Iovine", bridge="Beats", evidence=[link_a, link_b]
        )
        assert "Beats" in bridge
        assert "doc_A" in bridge
        assert "doc_B" in bridge

        inv = explain_inverse(source="Apple", relation="acquired_by", target="Beats", original=link_a)
        assert "acquired_by" in inv
        sym = explain_symmetric(
            source="A",
            relation="works_with",
            target="B",
            original=EvidenceLink(triple=("B", "works_with", "A"), source_ref="d1"),
        )
        assert "symmetric" in sym
        tr = explain_transitive(head="A", mid="B", tail="C", relation="part_of")
        assert "transitivity" in tr
        comp = explain_composition(owner="A", asset="X", location="L")
        assert "operates_in" in comp


# ---------------------------------------------------------------------------
# Data-model validation
# ---------------------------------------------------------------------------


class TestDataModelValidation:
    def test_inferred_edge_requires_evidence(self):
        with pytest.raises(ValueError, match="evidence"):
            InferredEdge(
                source="A",
                target="B",
                relationship_type="r",
                rule_name="x",
                evidence_chain=[],  # empty
                explanation="x",
            )

    def test_inferred_edge_requires_explanation(self):
        with pytest.raises(ValueError, match="explanation"):
            InferredEdge(
                source="A",
                target="B",
                relationship_type="r",
                rule_name="x",
                evidence_chain=[EvidenceLink(triple=("A", "r", "B"))],
                explanation="",
            )

    def test_inferred_edge_forbids_self_loop(self):
        with pytest.raises(ValueError, match="same"):
            InferredEdge(
                source="A",
                target="A",
                relationship_type="r",
                rule_name="x",
                evidence_chain=[EvidenceLink(triple=("A", "r", "B"))],
                explanation="x",
            )

    def test_inferred_edge_clamps_confidence(self):
        with pytest.raises(ValueError, match="confidence"):
            InferredEdge(
                source="A",
                target="B",
                relationship_type="r",
                rule_name="x",
                evidence_chain=[EvidenceLink(triple=("A", "r", "B"))],
                explanation="x",
                confidence=1.5,
            )

    def test_report_summary_is_empty_for_no_op(self):
        report = InferenceReport()
        assert report.is_empty()
        assert all(v == 0 or v == {} for v in report.summary().values())

    def test_rule_subclass_requires_name(self):
        with pytest.raises(TypeError, match="name"):
            class BadRule(InferenceRule):  # type: ignore[misc]
                # no name -> __init_subclass__ raises
                def apply(self, kg, config):  # pragma: no cover
                    return []


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------


def test_reason_over_graph_convenience():
    kg = _make_two_doc_kg()
    report = reason_over_graph(kg)
    assert isinstance(report, InferenceReport)
    assert report.added_edges


def test_default_rules_contains_all_builtins():
    names = {r.name for r in default_rules()}
    assert names == {"path_bridge", "inverse", "symmetric", "transitive", "composition"}


def test_top_level_lazy_import():
    import drg

    assert drg.MultiDocumentReasoner is MultiDocumentReasoner
    assert drg.PathBridgeRule is PathBridgeRule
    assert drg.reason_over_graph is reason_over_graph


# ---------------------------------------------------------------------------
# Rules cover both directions
# ---------------------------------------------------------------------------


def test_path_bridge_handles_both_outgoing_and_incoming_edges_at_bridge():
    """The bridge entity may be either the target or the source of an
    evidence edge — the rule must consider both roles."""
    kg = EnhancedKG()
    kg.add_node(KGNode(id="Apple"))
    kg.add_node(KGNode(id="Beats"))
    kg.add_node(KGNode(id="Music Industry"))
    # Bridge=Beats appears as target in one edge and source in the other
    kg.add_edge(_make_edge("Apple", "ACQUIRED", "Beats", source_ref="d1", confidence=0.9))
    kg.add_edge(
        _make_edge("Beats", "PART_OF", "Music Industry", source_ref="d2", confidence=0.9)
    )

    config = ReasoningConfig(
        disabled_rules=frozenset({"inverse", "symmetric", "transitive", "composition"})
    )
    MultiDocumentReasoner(config=config).reason(kg)
    bridge = [
        e for e in _inferred_edges(kg) if e.metadata["inference"]["rule"] == "path_bridge"
    ]
    assert len(bridge) == 1
    assert {bridge[0].source, bridge[0].target} == {"Apple", "Music Industry"}
