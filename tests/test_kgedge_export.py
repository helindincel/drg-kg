import json

from drg.graph.kg_core import EnhancedKG, KGEdge, KGNode


def test_enriched_format_includes_temporal_confidence_negation():
    kg = EnhancedKG()
    kg.add_node(KGNode(id="A", type="Company"))
    kg.add_node(KGNode(id="B", type="Product"))

    kg.add_edge(
        KGEdge(
            source="A",
            target="B",
            relationship_type="produces",
            relationship_detail="A produces B",
            start_time="1976-01-01",
            end_time="2011-12-31",
            confidence=0.9,
            is_negated=True,
            metadata={"source_ref": "test"},
        )
    )

    enriched = json.loads(kg.to_enriched_format())
    rel = enriched["relationships"][0]
    assert rel["start_time"] == "1976-01-01"
    assert rel["end_time"] == "2011-12-31"
    assert rel["confidence"] == 0.9
    assert rel["is_negated"] is True
    assert rel["source_ref"] == "test"
