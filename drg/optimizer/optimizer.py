"""DSPy optimizer integration for iterative learning."""

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import dspy

from ..extract import KGExtractor
from ..schema import DRGSchema, EnhancedDRGSchema

logger = logging.getLogger(__name__)


def _supports_var_kwargs(callable_obj: Callable) -> bool:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _filter_supported_kwargs(callable_obj: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if _supports_var_kwargs(callable_obj):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _call_with_supported_kwargs(callable_obj: Callable, **kwargs: Any):
    return callable_obj(**_filter_supported_kwargs(callable_obj, kwargs))


def _instantiate_with_supported_kwargs(factory: Callable, **kwargs: Any):
    return _call_with_supported_kwargs(factory, **kwargs)


class OptimizerType(str, Enum):
    """Supported optimizer types."""

    BOOTSTRAP_FEW_SHOT = "bootstrap_few_shot"
    MIPRO = "mipro"
    COPRO = "copro"
    LABELED_FEW_SHOT = "labeled_few_shot"


@dataclass
class OptimizerConfig:
    """Configuration for DSPy optimizer."""

    optimizer_type: OptimizerType = OptimizerType.BOOTSTRAP_FEW_SHOT
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 16
    num_candidates: int = 10
    init_temperature: float = 1.0
    metric_threshold: float = 0.7
    max_iterations: int = 5


@dataclass
class EvaluationResult:
    """Result of evaluation."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    details: dict[str, Any]


class DRGOptimizer:
    """DSPy optimizer wrapper for DRG extraction improvement."""

    def __init__(
        self,
        schema: DRGSchema | EnhancedDRGSchema,
        config: OptimizerConfig | None = None,
        training_examples: list[dict[str, Any]] | None = None,
        lm: Any | None = None,
    ):
        """Initialize DRG optimizer.

        Args:
            schema: DRG schema for extraction
            config: Optimizer configuration
            training_examples: Optional training examples (can also be added via add_training_example)
        """
        self.schema = schema
        self.config = config or OptimizerConfig()
        self.lm = lm

        # Create base extractor
        self.base_extractor = KGExtractor(schema, lm=lm)

        # Optimized extractor (will be set after optimization)
        self.optimized_extractor: KGExtractor | None = None

        # Training examples
        self.training_examples: list[dict[str, Any]] = (
            training_examples if training_examples is not None else []
        )

        # Evaluation history
        self.evaluation_history: list[EvaluationResult] = []
        self.last_compile_config: dict[str, Any] | None = None

    def add_training_example(
        self,
        text: str,
        expected_entities: list[tuple[str, str]],
        expected_relations: list[tuple[str, str, str]],
    ):
        """Add a training example.

        Args:
            text: Input text
            expected_entities: Expected (entity_name, entity_type) tuples
            expected_relations: Expected (source, relation, target) tuples
        """
        self.training_examples.append(
            {
                "text": text,
                "expected_entities": expected_entities,
                "expected_relations": expected_relations,
            }
        )

    def optimize(
        self,
        metric: Callable | None = None,
        validation_examples: list[dict[str, Any]] | None = None,
    ) -> KGExtractor:
        """Optimize extraction using DSPy optimizer.

        Args:
            metric: Custom evaluation metric function
            validation_examples: Optional validation examples

        Returns:
            Optimized KGExtractor
        """
        if not self.training_examples:
            raise ValueError("DSPy optimization requires at least one training example")

        logger.info(f"Starting optimization with {len(self.training_examples)} training examples")

        # Use default metric if not provided. Metric is known before optimizer
        # construction because several DSPy 2.x optimizers accept it in
        # `__init__` rather than `compile(...)`.
        if metric is None:
            metric = self._default_metric

        # Create optimizer based on type
        if self.config.optimizer_type == OptimizerType.BOOTSTRAP_FEW_SHOT:
            optimizer = self._create_bootstrap_optimizer(metric=metric)
        elif self.config.optimizer_type == OptimizerType.MIPRO:
            optimizer = self._create_mipro_optimizer(metric=metric)
        elif self.config.optimizer_type == OptimizerType.COPRO:
            optimizer = self._create_copro_optimizer(metric=metric)
        elif self.config.optimizer_type == OptimizerType.LABELED_FEW_SHOT:
            optimizer = self._create_labeled_few_shot_optimizer(metric=metric)
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

        # Prepare training set
        trainset = self._prepare_trainset()

        # Optimize using DSPy optimizer
        # DSPy 2.4+ uses compile() method for all optimizers (BootstrapFewShot, MIPRO, COPRO, etc.)
        # BootstrapFewShot is a teleprompter that wraps the module during forward pass
        try:
            if not hasattr(optimizer, "compile"):
                raise RuntimeError(
                    f"{type(optimizer).__name__} does not expose compile(); "
                    "DRG requires DSPy optimizers that actually compile."
                )
            self.optimized_extractor = self._compile_optimizer(
                optimizer=optimizer,
                trainset=trainset,
                metric=metric,
                validation_examples=validation_examples,
            )

            logger.info("Optimization completed successfully")
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise RuntimeError(
                f"DSPy optimization failed: {e}. "
                "Check your training examples format and metric function."
            ) from e

        # All three try-branches assign self.optimized_extractor before
        # reaching this point; the assert both documents that contract and
        # narrows Optional[KGExtractor] -> KGExtractor for type checkers.
        assert self.optimized_extractor is not None
        return self.optimized_extractor

    def _create_bootstrap_optimizer(self, *, metric: Callable) -> dspy.BootstrapFewShot:
        """Create BootstrapFewShot optimizer."""
        # BootstrapFewShot is a teleprompter, not a direct optimizer
        # It optimizes during forward pass
        return _instantiate_with_supported_kwargs(
            dspy.BootstrapFewShot,
            metric=metric,
            max_bootstrapped_demos=self.config.max_bootstrapped_demos,
            max_labeled_demos=self.config.max_labeled_demos,
        )

    def _create_mipro_optimizer(self, *, metric: Callable):
        """Create MIPRO optimizer.

        Note: MIPRO may not be available in all DSPy versions.
        """
        if hasattr(dspy, "MIPRO"):
            return _instantiate_with_supported_kwargs(
                dspy.MIPRO,
                metric=metric,
                num_candidates=self.config.num_candidates,
                init_temperature=self.config.init_temperature,
            )
        raise RuntimeError("DSPy MIPRO optimizer is not available")

    def _create_copro_optimizer(self, *, metric: Callable) -> dspy.COPRO:
        """Create COPRO optimizer."""
        return _instantiate_with_supported_kwargs(
            dspy.COPRO,
            metric=metric,
            num_candidates=self.config.num_candidates,
            init_temperature=self.config.init_temperature,
        )

    def _create_labeled_few_shot_optimizer(self, *, metric: Callable) -> dspy.LabeledFewShot:
        """Create LabeledFewShot optimizer."""
        return _instantiate_with_supported_kwargs(
            dspy.LabeledFewShot,
            metric=metric,
            k=self.config.max_labeled_demos,
        )

    def _prepare_trainset(self) -> list[dspy.Example]:
        """Prepare training set for DSPy.

        Important: The trainset format must match the output format of KGExtractor(...).
        The extractor returns a prediction-like object with 'entities' and 'relations' attributes,
        where both are lists of tuples. This matches the expected format from training examples.
        """
        trainset = []
        for example in self.training_examples:
            # Create dspy.Example with correct format
            # Input: text (str)
            # Output: entities (List[Tuple[str, str]]), relations (List[Tuple[str, str, str]])
            # This matches KGExtractor.forward() return type
            trainset.append(
                dspy.Example(
                    text=example["text"],
                    entities=example["expected_entities"],  # List[Tuple[str, str]]
                    relations=example["expected_relations"],  # List[Tuple[str, str, str]]
                ).with_inputs("text")
            )
        return trainset

    def _compile_optimizer(
        self,
        *,
        optimizer: Any,
        trainset: list[dspy.Example],
        metric: Callable,
        validation_examples: list[dict[str, Any]] | None,
    ) -> KGExtractor:
        """Compile a DSPy optimizer and fail if required configuration is dropped."""
        compile_kwargs: dict[str, Any] = {
            "student": self.base_extractor,
            "trainset": trainset,
            "metric": metric,
        }
        if validation_examples:
            compile_kwargs["valset"] = self._prepare_examples(validation_examples)
        supported_kwargs = _filter_supported_kwargs(optimizer.compile, compile_kwargs)
        missing_core = [k for k in ("student", "trainset") if k not in supported_kwargs]
        if missing_core:
            raise RuntimeError(
                f"{type(optimizer).__name__}.compile does not accept required kwargs: {missing_core}"
            )
        if "metric" not in supported_kwargs and not self._optimizer_has_metric(optimizer, metric):
            raise RuntimeError(
                f"{type(optimizer).__name__} does not preserve the metric configuration"
            )
        if validation_examples and "valset" not in supported_kwargs:
            raise RuntimeError(
                f"{type(optimizer).__name__}.compile does not accept validation examples"
            )
        self.last_compile_config = {
            "lm": self.lm,
            "trainset": trainset,
            "valset": supported_kwargs.get("valset"),
            "metric": metric,
        }
        return optimizer.compile(**supported_kwargs)

    def _optimizer_has_metric(self, optimizer: Any, metric: Callable) -> bool:
        return getattr(optimizer, "metric", None) is metric

    def _prepare_examples(self, examples: list[dict[str, Any]]) -> list[dspy.Example]:
        old_examples = self.training_examples
        try:
            self.training_examples = examples
            return self._prepare_trainset()
        finally:
            self.training_examples = old_examples

    def _default_metric(
        self,
        example: dspy.Example,
        pred: Any,
        trace: Any | None = None,
    ) -> float:
        """Default evaluation metric.

        Args:
            example: Ground truth example
            pred: Prediction from model
            trace: Optional trace

        Returns:
            Metric score (0-1)
        """
        # Extract expected and predicted entities/relations
        expected_entities = set(example.entities)
        expected_relations = set(example.relations)

        pred_entities = set(getattr(pred, "entities", []))
        pred_relations = set(getattr(pred, "relations", []))

        # Calculate F1 score
        entity_precision = (
            len(expected_entities & pred_entities) / len(pred_entities) if pred_entities else 0.0
        )
        entity_recall = (
            len(expected_entities & pred_entities) / len(expected_entities)
            if expected_entities
            else 0.0
        )
        entity_f1 = (
            2 * entity_precision * entity_recall / (entity_precision + entity_recall)
            if (entity_precision + entity_recall) > 0
            else 0.0
        )

        relation_precision = (
            len(expected_relations & pred_relations) / len(pred_relations)
            if pred_relations
            else 0.0
        )
        relation_recall = (
            len(expected_relations & pred_relations) / len(expected_relations)
            if expected_relations
            else 0.0
        )
        relation_f1 = (
            2 * relation_precision * relation_recall / (relation_precision + relation_recall)
            if (relation_precision + relation_recall) > 0
            else 0.0
        )

        # Combined F1 (weighted average)
        combined_f1 = 0.6 * entity_f1 + 0.4 * relation_f1

        return combined_f1

    def evaluate(
        self,
        test_examples: list[dict[str, Any]],
        use_optimized: bool = True,
    ) -> EvaluationResult:
        """Evaluate extractor on test examples.

        Args:
            test_examples: List of test examples with expected results
            use_optimized: Whether to use optimized extractor

        Returns:
            EvaluationResult
        """
        extractor = (
            self.optimized_extractor
            if use_optimized and self.optimized_extractor
            else self.base_extractor
        )

        all_precisions = []
        all_recalls = []
        all_f1s = []
        all_accuracies = []

        details: dict[str, list[float]] = {
            "entity_precisions": [],
            "entity_recalls": [],
            "relation_precisions": [],
            "relation_recalls": [],
        }

        for example in test_examples:
            text = example["text"]
            expected_entities = {tuple(e) for e in example.get("expected_entities", [])}
            expected_relations = {tuple(r) for r in example.get("expected_relations", [])}

            # Extract
            result = extractor(text=text)
            pred_entities = {
                tuple(e) for e in (result.entities if hasattr(result, "entities") else [])
            }
            pred_relations = {
                tuple(r) for r in (result.relations if hasattr(result, "relations") else [])
            }

            # Calculate metrics
            entity_precision = (
                len(expected_entities & pred_entities) / len(pred_entities)
                if pred_entities
                else 0.0
            )
            entity_recall = (
                len(expected_entities & pred_entities) / len(expected_entities)
                if expected_entities
                else 0.0
            )
            entity_f1 = (
                2 * entity_precision * entity_recall / (entity_precision + entity_recall)
                if (entity_precision + entity_recall) > 0
                else 0.0
            )

            relation_precision = (
                len(expected_relations & pred_relations) / len(pred_relations)
                if pred_relations
                else 0.0
            )
            relation_recall = (
                len(expected_relations & pred_relations) / len(expected_relations)
                if expected_relations
                else 0.0
            )
            relation_f1 = (
                2 * relation_precision * relation_recall / (relation_precision + relation_recall)
                if (relation_precision + relation_recall) > 0
                else 0.0
            )

            # Combined metrics
            precision = 0.6 * entity_precision + 0.4 * relation_precision
            recall = 0.6 * entity_recall + 0.4 * relation_recall
            f1 = 0.6 * entity_f1 + 0.4 * relation_f1
            accuracy = (
                (len(expected_entities & pred_entities) + len(expected_relations & pred_relations))
                / (len(expected_entities) + len(expected_relations))
                if (expected_entities or expected_relations)
                else 0.0
            )

            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            all_accuracies.append(accuracy)

            details["entity_precisions"].append(entity_precision)
            details["entity_recalls"].append(entity_recall)
            details["relation_precisions"].append(relation_precision)
            details["relation_recalls"].append(relation_recall)

        # Average metrics
        result = EvaluationResult(
            precision=sum(all_precisions) / len(all_precisions) if all_precisions else 0.0,
            recall=sum(all_recalls) / len(all_recalls) if all_recalls else 0.0,
            f1=sum(all_f1s) / len(all_f1s) if all_f1s else 0.0,
            accuracy=sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0,
            details=details,
        )

        self.evaluation_history.append(result)
        return result

    def iterative_improve(
        self,
        test_examples: list[dict[str, Any]],
        max_iterations: int | None = None,
    ) -> list[EvaluationResult]:
        """Iteratively improve extraction through multiple optimization cycles.

        Args:
            test_examples: Test examples for evaluation
            max_iterations: Maximum number of iterations (default: from config)

        Returns:
            List of evaluation results for each iteration
        """
        max_iter = max_iterations or self.config.max_iterations
        results = []

        logger.info(f"Starting iterative improvement with max {max_iter} iterations")

        for iteration in range(max_iter):
            logger.info(f"Iteration {iteration + 1}/{max_iter}")

            # Optimize
            self.optimize()

            # Evaluate
            result = self.evaluate(test_examples, use_optimized=True)
            results.append(result)

            logger.info(
                f"Iteration {iteration + 1} - F1: {result.f1:.3f}, "
                f"Precision: {result.precision:.3f}, Recall: {result.recall:.3f}"
            )

            # Check if threshold met
            if result.f1 >= self.config.metric_threshold:
                logger.info(f"Target metric threshold ({self.config.metric_threshold}) reached")
                break

        return results

    def compare_before_after(
        self,
        test_examples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare base and optimized extractor performance.

        Args:
            test_examples: Test examples

        Returns:
            Comparison dictionary
        """
        # Evaluate base
        base_result = self.evaluate(test_examples, use_optimized=False)

        # Evaluate optimized
        if self.optimized_extractor:
            optimized_result = self.evaluate(test_examples, use_optimized=True)
        else:
            logger.warning("No optimized extractor available, running optimization first")
            self.optimize()
            optimized_result = self.evaluate(test_examples, use_optimized=True)

        # Calculate improvements
        improvement = {
            "precision": optimized_result.precision - base_result.precision,
            "recall": optimized_result.recall - base_result.recall,
            "f1": optimized_result.f1 - base_result.f1,
            "accuracy": optimized_result.accuracy - base_result.accuracy,
        }

        return {
            "base": {
                "precision": base_result.precision,
                "recall": base_result.recall,
                "f1": base_result.f1,
                "accuracy": base_result.accuracy,
            },
            "optimized": {
                "precision": optimized_result.precision,
                "recall": optimized_result.recall,
                "f1": optimized_result.f1,
                "accuracy": optimized_result.accuracy,
            },
            "improvement": improvement,
            "improvement_percent": {
                "precision": (improvement["precision"] / base_result.precision * 100)
                if base_result.precision > 0
                else 0.0,
                "recall": (improvement["recall"] / base_result.recall * 100)
                if base_result.recall > 0
                else 0.0,
                "f1": (improvement["f1"] / base_result.f1 * 100) if base_result.f1 > 0 else 0.0,
                "accuracy": (improvement["accuracy"] / base_result.accuracy * 100)
                if base_result.accuracy > 0
                else 0.0,
            },
        }


def create_optimizer(
    schema: DRGSchema | EnhancedDRGSchema, optimizer_type: str = "bootstrap_few_shot", **kwargs
) -> DRGOptimizer:
    """Factory function to create optimizer.

    Args:
        schema: DRG schema
        optimizer_type: Optimizer type name
        **kwargs: Additional config parameters

    Returns:
        DRGOptimizer instance
    """
    config = OptimizerConfig(optimizer_type=OptimizerType(optimizer_type), **kwargs)
    return DRGOptimizer(schema=schema, config=config)


def evaluate_extraction(
    extractor: KGExtractor,
    test_examples: list[dict[str, Any]],
) -> EvaluationResult:
    """Evaluate extraction performance.

    Args:
        extractor: KGExtractor to evaluate
        test_examples: Test examples with expected results

    Returns:
        EvaluationResult
    """
    optimizer = DRGOptimizer(schema=extractor.schema)
    optimizer.base_extractor = extractor
    return optimizer.evaluate(test_examples, use_optimized=False)
