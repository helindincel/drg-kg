#!/usr/bin/env python3
"""
Incremental Knowledge Graph Update — End-to-end demo
=====================================================

Bu örnek, **LLM çağrısı gerektirmez** — saf veri katmanını gösterir.
Üç farklı dökümandan çıkarıldığını varsaydığımız üç ayrı küçük KG'yi tek
bir kalıcı KG'ye merge eder. Gerçek kullanımda ``new_kg`` adımları
``build_enhanced_kg(...)`` ya da ``extract_typed(...)`` gibi mevcut
extraction fonksiyonlarınızın çıktısı olur; merge mantığı aynı kalır.

Çalıştırma::

    python examples/incremental_update_example.py

Çıktı:
- Merge sonrası graph istatistikleri her ingestion sonrasında
- Persisted KG dosyası: ``examples/incremental_update_demo.kg.json``
- Sürüm + history metadata, dedup raporu (KGDiff)

Pipeline ile entegrasyon (gerçek dökümanlarda)::

    from drg.extract import extract_typed
    from drg.graph import EnhancedKG, GraphMerger
    from drg.graph.builders import build_enhanced_kg

    kg_path = "outputs/global_kg.json"
    base = EnhancedKG.load_json(kg_path) if Path(kg_path).exists() else EnhancedKG()

    for doc_id, text in your_documents:
        entities, triples = extract_typed(text, schema)
        new_kg = build_enhanced_kg(
            entities_typed=entities,
            triples=triples,
            source_text=text,
            schema=schema,
        )
        diff = GraphMerger().merge(base, new_kg, document_id=doc_id)
        print(f"{doc_id}: {diff.summary()}")

    base.save_json(kg_path)

Bu pattern, mevcut "build from scratch" iş akışını bozmaz — sadece
``GraphMerger`` opsiyonel bir katman olarak ekleniyor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo root'u path'e ekle (pip install -e olmadan da çalışsın).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drg.graph import (  # noqa: E402
    EnhancedKG,
    GraphMerger,
    KGEdge,
    KGNode,
    MergeStrategy,
    NodeMergePolicy,
)


def _print_section(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def _print_kg_summary(kg: EnhancedKG, label: str) -> None:
    print(f"  [{label}] nodes={len(kg.nodes)}  edges={len(kg.edges)}  ", end="")
    print(f"clusters={len(kg.clusters)}  version={kg.metadata.get('version', '-')}")


# ---------------------------------------------------------------------------
# Doc 1 — initial document
# ---------------------------------------------------------------------------


def doc1_kg() -> EnhancedKG:
    """Simulates: 'Apple Inc. was founded by Steve Jobs and Steve Wozniak.
    Tim Cook is the CEO.' extracted into a KG."""
    kg = EnhancedKG()
    kg.add_node(KGNode(id="Apple Inc", type="Company", properties={"sector": "tech"}))
    kg.add_node(KGNode(id="Steve Jobs", type="Person"))
    kg.add_node(KGNode(id="Steve Wozniak", type="Person"))
    kg.add_node(KGNode(id="Tim Cook", type="Person"))
    kg.add_edge(
        KGEdge(
            source="Steve Jobs",
            target="Apple Inc",
            relationship_type="FOUNDED",
            relationship_detail="co-founder",
            confidence=0.92,
            metadata={"source_ref": "doc_1"},
        )
    )
    kg.add_edge(
        KGEdge(
            source="Steve Wozniak",
            target="Apple Inc",
            relationship_type="FOUNDED",
            relationship_detail="co-founder",
            confidence=0.92,
            metadata={"source_ref": "doc_1"},
        )
    )
    kg.add_edge(
        KGEdge(
            source="Tim Cook",
            target="Apple Inc",
            relationship_type="WORKS_AT",
            relationship_detail="CEO",
            confidence=0.95,
            metadata={"source_ref": "doc_1"},
        )
    )
    return kg


# ---------------------------------------------------------------------------
# Doc 2 — overlapping document with new entities + duplicate fact
# ---------------------------------------------------------------------------


def doc2_kg() -> EnhancedKG:
    """Simulates a second article that mentions the same Apple in lowercase
    surface form ('apple inc'), introduces 'iPhone' as a product, and
    *re-states* the Tim Cook -> Apple WORKS_AT fact (different casing on
    the relation type to model real-world extraction noise)."""
    kg = EnhancedKG()
    kg.add_node(
        KGNode(
            id="apple inc",
            type="Company",
            properties={"industry": "technology"},
        )
    )
    kg.add_node(KGNode(id="iPhone", type="Product"))
    kg.add_node(KGNode(id="Tim Cook", type="Person"))
    kg.add_edge(
        KGEdge(
            source="apple inc",
            target="iPhone",
            relationship_type="PRODUCES",
            relationship_detail="flagship smartphone",
            confidence=0.9,
            metadata={"source_ref": "doc_2"},
        )
    )
    kg.add_edge(
        KGEdge(
            source="Tim Cook",
            target="apple inc",
            relationship_type="works_at",  # case differs from doc_1 on purpose
            relationship_detail="CEO",
            confidence=0.88,
            metadata={"source_ref": "doc_2"},
        )
    )
    return kg


# ---------------------------------------------------------------------------
# Doc 3 — adds a brand-new sub-graph that references existing nodes
# ---------------------------------------------------------------------------


def doc3_kg() -> EnhancedKG:
    """Simulates: 'Apple Inc. acquired Beats Electronics in 2014. Beats'
    headquarters are in Culver City.' The 'Apple Inc.' here is yet another
    surface form ('Apple Inc.' with trailing period — which intentionally
    *does NOT* normalize to the same canonical id, demonstrating the
    conservative default behaviour). 'Tim Cook' is referenced again."""
    kg = EnhancedKG()
    kg.add_node(KGNode(id="Apple Inc", type="Company"))  # exact id match -> reused
    kg.add_node(KGNode(id="Beats Electronics", type="Company"))
    kg.add_node(KGNode(id="Culver City", type="Place"))
    kg.add_edge(
        KGEdge(
            source="Apple Inc",
            target="Beats Electronics",
            relationship_type="ACQUIRED",
            relationship_detail="2014 acquisition",
            confidence=0.97,
            metadata={"source_ref": "doc_3"},
        )
    )
    kg.add_edge(
        KGEdge(
            source="Beats Electronics",
            target="Culver City",
            relationship_type="HEADQUARTERED_IN",
            relationship_detail="HQ location",
            confidence=0.85,
            metadata={"source_ref": "doc_3"},
        )
    )
    return kg


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> int:
    out_path = Path(__file__).with_suffix(".kg.json")

    _print_section("1. Bootstrapping a fresh persisted KG")
    print(
        f"  Output file: {out_path}\n"
        f"  (deletes any previous demo run so the example is reproducible)"
    )
    if out_path.exists():
        out_path.unlink()
    base = EnhancedKG()
    _print_kg_summary(base, "empty")

    merger = GraphMerger(MergeStrategy(node_policy=NodeMergePolicy.PREFER_EXISTING))

    _print_section("2. Ingesting doc_1")
    new_kg = doc1_kg()
    _print_kg_summary(new_kg, "doc_1 in-memory")
    diff = merger.merge(base, new_kg, document_id="doc_1")
    print("  Diff:", json.dumps(diff.summary()))
    _print_kg_summary(base, "base after doc_1")

    _print_section("3. Ingesting doc_2 (overlapping with doc_1)")
    new_kg = doc2_kg()
    _print_kg_summary(new_kg, "doc_2 in-memory")
    diff = merger.merge(base, new_kg, document_id="doc_2")
    print("  Diff:", json.dumps(diff.summary()))
    print("  -> 'apple inc' fold edildi mi:", "apple inc" not in base.nodes)
    print("  -> 'Apple Inc' canonical olarak korundu mu:", "Apple Inc" in base.nodes)
    print(
        "  -> Duplicate WORKS_AT kenarı (case-insensitive dedup):",
        diff.summary()["skipped_edges"],
        "skipped",
    )
    _print_kg_summary(base, "base after doc_2")

    _print_section("4. Ingesting doc_3 (new sub-graph touching existing nodes)")
    new_kg = doc3_kg()
    _print_kg_summary(new_kg, "doc_3 in-memory")
    diff = merger.merge(base, new_kg, document_id="doc_3")
    print("  Diff:", json.dumps(diff.summary()))
    _print_kg_summary(base, "base after doc_3")

    _print_section("5. Persisting the updated KG")
    base.save_json(str(out_path))
    print(f"  Wrote: {out_path}")

    _print_section("6. Round-trip — load and continue updating")
    reloaded = EnhancedKG.load_json(str(out_path))
    print(f"  Reloaded  nodes={len(reloaded.nodes)}  edges={len(reloaded.edges)}")
    print(f"  Version:  {reloaded.metadata.get('version')}")
    print(f"  History:  {len(reloaded.metadata.get('history', []))} entries")
    print("  Documents seen so far:")
    for h in reloaded.metadata.get("history", []):
        print(
            f"    - v{h['version']} {h['operation']:5s} doc={h.get('document_id')}  "
            f"+{h['added_nodes']}n/{h['added_edges']}e  "
            f"~{h['merged_nodes']}n  -{h['skipped_edges']}e"
        )

    _print_section("7. Provenance — what was merged into 'Apple Inc'?")
    apple = reloaded.nodes["Apple Inc"]
    if "merged_from" in apple.metadata:
        print("  Apple Inc.metadata['merged_from'] (audit trail):")
        for entry in apple.metadata["merged_from"]:
            print(f"    - {entry}")
    else:
        print("  No merge events recorded for Apple Inc (only exact-id matches).")

    print()
    print("OK — incremental update demo finished without an LLM call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
