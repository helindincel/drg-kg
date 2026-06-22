"""Generate deterministic scale benchmark datasets.

Usage:
    python examples/benchmarks/generate_scale_dataset.py --documents 1000 -o /tmp/drg_scale_1k.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_dataset(*, documents: int, entities_per_document: int) -> dict:
    if documents < 1:
        raise ValueError("documents must be >= 1")
    if entities_per_document < 2:
        raise ValueError("entities-per-document must be >= 2")

    text_parts: list[str] = []
    gold_entities: list[dict[str, str]] = []
    gold_relations: list[dict[str, str]] = []

    for doc_idx in range(documents):
        company = f"Company{doc_idx:06d}"
        product = f"Product{doc_idx:06d}"
        person = f"Person{doc_idx:06d}"
        market = f"Market{doc_idx % max(1, entities_per_document):03d}"

        text_parts.append(
            f"{company} launched {product} with {person} for {market}. "
            f"{person} leads product strategy at {company}."
        )
        gold_entities.extend(
            [
                {"name": company, "type": "Company"},
                {"name": product, "type": "Product"},
                {"name": person, "type": "Person"},
                {"name": market, "type": "Market"},
            ]
        )
        gold_relations.extend(
            [
                {
                    "source": company,
                    "relationship_type": "LAUNCHED",
                    "target": product,
                },
                {
                    "source": person,
                    "relationship_type": "LEADS_STRATEGY_AT",
                    "target": company,
                },
                {
                    "source": product,
                    "relationship_type": "TARGETS",
                    "target": market,
                },
            ]
        )

    return {
        "name": f"synthetic_scale_{documents}",
        "text": "\n".join(text_parts),
        "gold_entities": gold_entities,
        "gold_relations": gold_relations,
        "gold_events": [],
        "metadata": {
            "domain": "synthetic_scale",
            "task": "large_dataset_performance",
            "document_count": documents,
            "chunk_count": documents,
            "entity_count": len(gold_entities),
            "relation_count": len(gold_relations),
            "generator": "examples/benchmarks/generate_scale_dataset.py",
        },
    }


def build_oracle_predictions(dataset: dict) -> dict:
    return {
        "adapter": "oracle",
        "model": "deterministic-gold",
        "predictions": {
            dataset["name"]: {
                "entities": [
                    [entity["name"], entity.get("type")] for entity in dataset["gold_entities"]
                ],
                "relations": [
                    [
                        relation["source"],
                        relation["relationship_type"],
                        relation["target"],
                    ]
                    for relation in dataset["gold_relations"]
                ],
                "metadata": {
                    "source": "gold_oracle",
                },
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a DRG scale benchmark dataset")
    parser.add_argument(
        "--documents", type=int, required=True, help="Number of synthetic documents"
    )
    parser.add_argument(
        "--entities-per-document",
        type=int,
        default=4,
        help="Controls market cardinality in the generated corpus",
    )
    parser.add_argument("-o", "--output", type=str, required=True, help="Output dataset JSON path")
    parser.add_argument(
        "--oracle-output",
        type=str,
        help="Optional prediction artifact path containing gold/oracle predictions",
    )
    args = parser.parse_args()

    dataset = build_dataset(
        documents=args.documents,
        entities_per_document=args.entities_per_document,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {dataset['name']} to {output}")
    if args.oracle_output:
        oracle_output = Path(args.oracle_output)
        oracle_output.parent.mkdir(parents=True, exist_ok=True)
        oracle_output.write_text(
            json.dumps(build_oracle_predictions(dataset), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote oracle predictions to {oracle_output}")


if __name__ == "__main__":
    main()
