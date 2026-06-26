#!/usr/bin/env python3
"""DSPy optimizer demo — requires DSPy and an LLM provider.

Demonstrates how to use :mod:`drg.optimizer` to optimise a
:class:`~drg.extract.KGExtractor` against labelled training examples.

Covers:
- Preparing training data in the optimizer format
- BootstrapFewShot optimisation (fast, default)
- MIPRO optimisation (slower, higher quality)
- Evaluating the optimised extractor on a held-out set
- Comparing raw vs. optimised extractor metrics

Prerequisites::

    pip install drg-kg[extract]
    export OPENAI_API_KEY=sk-...   # or any DSPy-supported provider

Run::

    python examples/optimizer_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Training data (labelled examples)
# ---------------------------------------------------------------------------

TRAINING_DATA = [
    {
        "text": (
            "Marie Curie was born in Warsaw in 1867. She studied at the University "
            "of Paris and discovered both polonium and radium."
        ),
        "expected_entities": [
            {"name": "Marie Curie", "type": "Person"},
            {"name": "Warsaw", "type": "Place"},
            {"name": "University of Paris", "type": "Organization"},
            {"name": "polonium", "type": "Element"},
            {"name": "radium", "type": "Element"},
        ],
        "expected_relations": [
            {"source": "Marie Curie", "relation": "born_in", "target": "Warsaw"},
            {"source": "Marie Curie", "relation": "studied_at", "target": "University of Paris"},
            {"source": "Marie Curie", "relation": "discovered", "target": "polonium"},
            {"source": "Marie Curie", "relation": "discovered", "target": "radium"},
        ],
    },
    {
        "text": (
            "Apple Inc. acquired Beats Electronics in 2014 for $3 billion. "
            "Beats was co-founded by Dr. Dre and Jimmy Iovine."
        ),
        "expected_entities": [
            {"name": "Apple Inc.", "type": "Company"},
            {"name": "Beats Electronics", "type": "Company"},
            {"name": "Dr. Dre", "type": "Person"},
            {"name": "Jimmy Iovine", "type": "Person"},
        ],
        "expected_relations": [
            {"source": "Apple Inc.", "relation": "acquired", "target": "Beats Electronics"},
            {"source": "Dr. Dre", "relation": "co-founded", "target": "Beats Electronics"},
            {"source": "Jimmy Iovine", "relation": "co-founded", "target": "Beats Electronics"},
        ],
    },
    {
        "text": (
            "OpenAI was founded in San Francisco by Sam Altman and others. "
            "Microsoft has invested heavily in OpenAI."
        ),
        "expected_entities": [
            {"name": "OpenAI", "type": "Company"},
            {"name": "San Francisco", "type": "Place"},
            {"name": "Sam Altman", "type": "Person"},
            {"name": "Microsoft", "type": "Company"},
        ],
        "expected_relations": [
            {"source": "OpenAI", "relation": "founded_in", "target": "San Francisco"},
            {"source": "Sam Altman", "relation": "founded", "target": "OpenAI"},
            {"source": "Microsoft", "relation": "invested_in", "target": "OpenAI"},
        ],
    },
]

# Held-out test set (not used during optimisation)
TEST_EXAMPLES = [
    {
        "text": "SpaceX was founded by Elon Musk in 2002 in Hawthorne, California.",
        "expected_entities": [
            {"name": "SpaceX", "type": "Company"},
            {"name": "Elon Musk", "type": "Person"},
            {"name": "Hawthorne", "type": "Place"},
        ],
        "expected_relations": [
            {"source": "Elon Musk", "relation": "founded", "target": "SpaceX"},
            {"source": "SpaceX", "relation": "located_in", "target": "Hawthorne"},
        ],
    },
]


def _score_predictions(predictions, test_examples) -> dict:
    """Quick entity/relation F1 scorer (no DSPy dependency)."""
    from drg.optimizer.metrics import EntityExtractionMetric, RelationExtractionMetric

    entity_metric = EntityExtractionMetric()
    relation_metric = RelationExtractionMetric()

    entity_f1s = []
    relation_f1s = []

    for pred, ex_dict in zip(predictions, test_examples, strict=False):
        # Wrap raw dict as a simple namespace for the metric callables
        class _Ex:
            expected_entities = ex_dict["expected_entities"]
            expected_relations = ex_dict["expected_relations"]

        ef = entity_metric(_Ex(), pred)
        rf = relation_metric(_Ex(), pred)
        entity_f1s.append(ef)
        relation_f1s.append(rf)

    return {
        "entity_f1": sum(entity_f1s) / len(entity_f1s) if entity_f1s else 0.0,
        "relation_f1": sum(relation_f1s) / len(relation_f1s) if relation_f1s else 0.0,
    }


def main() -> None:
    # Guard against missing dependencies
    try:
        import dspy  # noqa: F401
    except ImportError:
        print("ERROR: DSPy is required. Install with: pip install drg-kg[extract]")
        sys.exit(1)

    from drg.config import configure_lm
    from drg.extract import KGExtractor
    from drg.optimizer import KGOptimizerConfig, optimize_extractor
    from drg.schema import DRGSchema, Entity, Relation

    # ------------------------------------------------------------------
    # 1. Configure LLM
    # ------------------------------------------------------------------
    _section("Configuring LLM")
    try:
        configure_lm()
        print("  LLM configured from environment variables.")
    except Exception as e:
        print(f"  WARNING: LLM config failed: {e}")
        print("  Set OPENAI_API_KEY (or another provider) and retry.")
        return

    # ------------------------------------------------------------------
    # 2. Build a schema for extraction
    # ------------------------------------------------------------------
    _section("Schema")
    schema = DRGSchema(
        entities=[
            Entity("Person"),
            Entity("Company"),
            Entity("Place"),
            Entity("Organization"),
            Entity("Element"),
        ],
        relations=[
            Relation("founded", src="Person", dst="Company"),
            Relation("acquired", src="Company", dst="Company"),
            Relation("born_in", src="Person", dst="Place"),
            Relation("located_in", src="Company", dst="Place"),
            Relation("discovered", src="Person", dst="Element"),
            Relation("studied_at", src="Person", dst="Organization"),
            Relation("invested_in", src="Company", dst="Company"),
        ],
    )
    print(f"  Entities : {[e.name for e in schema.entities]}")
    print(f"  Relations: {[r.name for r in schema.relations]}")

    # ------------------------------------------------------------------
    # 3. Baseline extractor (unoptimised)
    # ------------------------------------------------------------------
    _section("Baseline extractor (unoptimised)")
    base_extractor = KGExtractor(schema)
    print("  KGExtractor created (no few-shot examples).")

    # ------------------------------------------------------------------
    # 4. Optimise with BootstrapFewShot
    # ------------------------------------------------------------------
    _section("Optimising with BootstrapFewShot")
    config = KGOptimizerConfig(
        optimizer_type="bootstrap",
        max_bootstrapped_demos=3,
        max_labeled_demos=1,
        metric_threshold=0.6,
    )
    print(f"  Optimizer type   : {config.optimizer_type}")
    print(f"  Max demos        : {config.max_bootstrapped_demos}")
    print(f"  Metric threshold : {config.metric_threshold}")

    try:
        optimised = optimize_extractor(TRAINING_DATA, config=config, extractor=base_extractor)
        print("  Optimisation complete.")
    except Exception as e:
        print(f"  Optimisation failed: {e}")
        print("  (This is expected in CI without a live LLM)")
        return

    # ------------------------------------------------------------------
    # 5. Evaluate on test set
    # ------------------------------------------------------------------
    _section("Evaluating on test set")

    test_preds = []
    for ex in TEST_EXAMPLES:
        try:
            result = optimised(ex["text"])
            entities = getattr(result, "entities", [])
            triples = getattr(result, "relations", [])
            pred = type(
                "P",
                (),
                {
                    "entities": [{"name": n, "type": t} for n, t in entities],
                    "relations": [
                        {"source": s, "relation": r, "target": tgt} for s, r, tgt in triples
                    ],
                },
            )()
            test_preds.append(pred)
        except Exception as e:
            print(f"  Extraction failed for test example: {e}")
            return

    scores = _score_predictions(test_preds, TEST_EXAMPLES)
    print(f"  Entity F1   : {scores['entity_f1']:.4f}")
    print(f"  Relation F1 : {scores['relation_f1']:.4f}")

    print()
    print("Done! The optimised extractor has been compiled with few-shot demonstrations.")
    print("Save it with: dspy.save(optimised, 'optimised_extractor.json')")


if __name__ == "__main__":
    main()
