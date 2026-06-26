"""DSPy optimizer for DRG knowledge graph extraction pipelines.

Wraps DSPy teleprompters (BootstrapFewShot, MIPRO, COPRO, LabeledFewShot)
to optimise :class:`~drg.extract.KGExtractor` prompts against labelled data.

Quick start::

    from drg.optimizer import KGOptimizerConfig, optimize_extractor

    config = KGOptimizerConfig(optimizer_type="bootstrap", max_bootstrapped_demos=4)
    examples = [
        {
            "text": "Marie Curie discovered radium.",
            "expected_entities": [{"name": "Marie Curie", "type": "Person"}, ...],
            "expected_relations": [{"source": "Marie Curie", "relation": "discovered", "target": "radium"}],
        },
    ]
    optimised = optimize_extractor(examples, config=config, schema=schema)
"""

from .metrics import EntityExtractionMetric, RelationExtractionMetric, weighted_f1_metric
from .optimizer import KGOptimizerConfig, optimize_extractor

__all__ = [
    "EntityExtractionMetric",
    "KGOptimizerConfig",
    "RelationExtractionMetric",
    "optimize_extractor",
    "weighted_f1_metric",
]
