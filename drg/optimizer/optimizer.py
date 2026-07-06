"""DSPy optimizer integration for DRG knowledge graph extraction.

This module wraps DSPy's teleprompter family so you can optimise the
``KGExtractor`` prompts against labelled training examples without
writing DSPy boilerplate.

Supported optimiser types
--------------------------
``bootstrap``        — BootstrapFewShot (default, fast, no labels required)
``mipro``            — MIPRO (instruction + few-shot, slower, higher quality)
``copro``            — COPRO (coordinate ascent, good for small datasets)
``labeled_few_shot`` — LabeledFewShot (requires gold labels for every demo)

Quick start::

    from drg.optimizer import KGOptimizerConfig, optimize_extractor

    config = KGOptimizerConfig(
        optimizer_type="bootstrap",
        max_bootstrapped_demos=4,
        metric_threshold=0.7,
    )
    training_examples = [
        {"text": "Apple was founded by Steve Jobs.", "expected_entities": [...], "expected_relations": [...]},
    ]
    optimised_extractor = optimize_extractor(training_examples, config=config, schema=schema)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .metrics import weighted_f1_metric

__all__ = [
    "KGOptimizerConfig",
    "PipelineStage",
    "optimize_extractor",
]

OptimizerType = Literal["bootstrap", "mipro", "copro", "labeled_few_shot"]
PipelineStage = Literal["single_text", "document_relations", "implicit_relations", "coreference"]


@dataclass
class KGOptimizerConfig:
    """Configuration for the DSPy KG extraction optimizer.

    Attributes:
        optimizer_type: Which DSPy teleprompter to use.
        max_bootstrapped_demos: Maximum number of few-shot demonstrations to
            bootstrap (applies to ``bootstrap`` and ``mipro``).
        max_labeled_demos: Maximum labelled demonstrations for ``labeled_few_shot``
            and ``mipro``.
        metric_threshold: Minimum weighted F1 required for a bootstrapped demo
            to be accepted.  Irrelevant for ``labeled_few_shot``.
        max_iterations: Maximum optimisation iterations (``mipro`` / ``copro``).
        entity_weight: Entity F1 weight in the composite metric (default 0.6).
        relation_weight: Relation F1 weight in the composite metric (default 0.4).
        teacher_settings: Extra kwargs forwarded to the DSPy teacher LM.
        trainset_fraction: Fraction of training examples to use as the
            bootstrap seed (rest used as validation).
        pipeline_stage: Which DSPy program to compile. ``single_text`` keeps
            the legacy ``KGExtractor.forward`` behavior; the other stages tune
            document relation, implicit relation, or coreference sub-programs.
    """

    optimizer_type: OptimizerType = "bootstrap"
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 2
    metric_threshold: float = 0.7
    max_iterations: int = 5
    entity_weight: float = 0.6
    relation_weight: float = 0.4
    teacher_settings: dict[str, Any] = field(default_factory=dict)
    trainset_fraction: float = 0.8
    pipeline_stage: PipelineStage = "single_text"


def _make_dspy_examples(
    training_data: list[dict[str, Any]],
    *,
    pipeline_stage: PipelineStage = "single_text",
) -> list[Any]:
    """Convert raw training dicts to ``dspy.Example`` objects."""
    import dspy

    examples = []
    for item in training_data:
        payload = {
            "text": item.get("text", ""),
            "chunks": item.get("chunks", []),
            "entities": item.get("entities", item.get("expected_entities", [])),
            "relations": item.get("relations", []),
            "existing_relations": item.get("existing_relations", item.get("relations", [])),
            "expected_entities": item.get("expected_entities", []),
            "expected_relations": item.get("expected_relations", []),
            "expected_enriched_relations": item.get("expected_enriched_relations", []),
        }
        if pipeline_stage == "single_text":
            ex = dspy.Example(**payload).with_inputs("text")
        elif pipeline_stage == "document_relations":
            ex = dspy.Example(**payload).with_inputs("chunks", "entities")
        elif pipeline_stage == "implicit_relations":
            ex = dspy.Example(**payload).with_inputs("text", "entities", "existing_relations")
        elif pipeline_stage == "coreference":
            ex = dspy.Example(**payload).with_inputs("text", "entities", "relations")
        else:
            raise ValueError(f"Unknown pipeline_stage: {pipeline_stage!r}")
        examples.append(ex)
    return examples


def _build_optimizable_program(dspy_module, extractor, pipeline_stage: PipelineStage):
    """Wrap a concrete extraction stage in a compile-able DSPy module."""

    if pipeline_stage == "single_text":
        return extractor

    class OptimizableExtractionStage(dspy_module.Module):
        def __init__(self, wrapped_extractor, stage: PipelineStage):
            super().__init__()
            self.extractor = wrapped_extractor
            self.pipeline_stage = stage

        def forward(
            self,
            text: str = "",
            chunks: list[dict[str, Any]] | None = None,
            entities: list[tuple[str, str]] | None = None,
            relations: list[tuple[str, str, str]] | None = None,
            existing_relations: list[tuple[str, str, str]] | None = None,
        ):
            entities = entities or []
            relations = relations or []
            existing_relations = existing_relations or []
            chunks = chunks or []

            if self.pipeline_stage == "document_relations":
                return self.extractor.extract_document_relations(chunks=chunks, entities=entities)
            if self.pipeline_stage == "implicit_relations":
                return self.extractor.infer_implicit_relations(
                    text=text,
                    entities=entities,
                    existing_relations=existing_relations,
                )
            if self.pipeline_stage == "coreference":
                return self.extractor.resolve_coreferences_dspy(
                    text=text,
                    entities=entities,
                    relations=relations,
                )
            raise ValueError(f"Unknown pipeline_stage: {self.pipeline_stage!r}")

    return OptimizableExtractionStage(extractor, pipeline_stage)


def _get_dspy_optimizer(dspy_module, name: str):
    optimizer = getattr(dspy_module, name, None)
    if optimizer is not None:
        return optimizer

    teleprompt = getattr(dspy_module, "teleprompt", None)
    if teleprompt is not None:
        optimizer = getattr(teleprompt, name, None)
        if optimizer is not None:
            return optimizer

    raise AttributeError(f"DSPy optimizer {name!r} is not available")


def _build_metric(config: KGOptimizerConfig):
    """Return a metric callable with weights from config."""
    entity_w = config.entity_weight
    relation_w = config.relation_weight

    def metric(example, prediction, trace=None):
        return weighted_f1_metric(
            example,
            prediction,
            trace,
            entity_weight=entity_w,
            relation_weight=relation_w,
        )

    return metric


def optimize_extractor(
    training_data: list[dict[str, Any]],
    *,
    config: KGOptimizerConfig | None = None,
    extractor=None,
    schema=None,
) -> Any:
    """Optimise a :class:`~drg.extract.KGExtractor` using DSPy teleprompters.

    Parameters
    ----------
    training_data:
        List of dicts with ``text``, ``expected_entities``, and
        ``expected_relations`` keys.
    config:
        Optimizer configuration.  Defaults applied when ``None``.
    extractor:
        Pre-built :class:`~drg.extract.KGExtractor` instance.  A new one
        is created from ``schema`` when ``None``.
    schema:
        Schema used to build a new :class:`~drg.extract.KGExtractor` when
        ``extractor`` is not supplied.

    Returns
    -------
    The compiled (optimised) extractor module.
    """
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "DSPy is required for the optimizer. Install it with: pip install drg-kg[extract]"
        ) from exc

    cfg = config or KGOptimizerConfig()

    if extractor is None:
        from drg.extract import KGExtractor

        if schema is None:
            raise ValueError("schema is required when extractor is not supplied.")
        extractor = KGExtractor(schema)

    program = _build_optimizable_program(dspy, extractor, cfg.pipeline_stage)

    examples = _make_dspy_examples(training_data, pipeline_stage=cfg.pipeline_stage)
    if not examples:
        raise ValueError("training_data must contain at least one example.")

    split = max(1, int(len(examples) * cfg.trainset_fraction))
    trainset = examples[:split]
    devset = examples[split:] or examples[:1]

    metric = _build_metric(cfg)

    if cfg.optimizer_type == "bootstrap":
        BootstrapFewShot = _get_dspy_optimizer(dspy, "BootstrapFewShot")
        teleprompter = BootstrapFewShot(
            metric=metric,
            metric_threshold=cfg.metric_threshold,
            max_bootstrapped_demos=cfg.max_bootstrapped_demos,
            max_labeled_demos=cfg.max_labeled_demos,
            teacher_settings=cfg.teacher_settings or {},
        )
        return teleprompter.compile(program, trainset=trainset)

    elif cfg.optimizer_type == "labeled_few_shot":
        LabeledFewShot = _get_dspy_optimizer(dspy, "LabeledFewShot")
        teleprompter = LabeledFewShot(k=cfg.max_labeled_demos)
        return teleprompter.compile(program, trainset=trainset)

    elif cfg.optimizer_type == "copro":
        COPRO = _get_dspy_optimizer(dspy, "COPRO")
        teleprompter = COPRO(
            metric=metric,
            verbose=False,
        )
        return teleprompter.compile(
            program,
            trainset=trainset,
            eval_kwargs={"devset": devset},
        )

    elif cfg.optimizer_type == "mipro":
        try:
            MIPROv2 = _get_dspy_optimizer(dspy, "MIPROv2")
            teleprompter = MIPROv2(
                metric=metric,
                auto="light",
                num_candidates=cfg.max_bootstrapped_demos,
            )
        except AttributeError:
            # Fall back to MIPRO for older DSPy versions
            MIPRO = _get_dspy_optimizer(dspy, "MIPRO")
            teleprompter = MIPRO(metric=metric)
        return teleprompter.compile(
            program,
            trainset=trainset,
            max_bootstrapped_demos=cfg.max_bootstrapped_demos,
            max_labeled_demos=cfg.max_labeled_demos,
        )

    else:
        raise ValueError(
            f"Unknown optimizer_type: {cfg.optimizer_type!r}. "
            "Choose from: 'bootstrap', 'mipro', 'copro', 'labeled_few_shot'."
        )
