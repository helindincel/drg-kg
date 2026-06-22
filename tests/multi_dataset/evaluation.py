"""Multi-dataset evaluation infrastructure."""

import logging
from dataclasses import dataclass
from typing import Any

from drg.chunking import Chunk, ChunkingStrategy
from drg.embedding import EmbeddingProvider

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetrics:
    """Metrics for a single dataset."""

    dataset_name: str
    chunking_quality: dict[str, float]
    entity_extraction: dict[str, float]
    total_chunks: int
    total_entities: int
    total_relations: int


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    dataset_metrics: list[DatasetMetrics]
    comparison_table: dict[str, dict[str, Any]]
    observations: list[str]
    failure_cases: list[dict[str, Any]]


class MultiDatasetEvaluator:
    """Evaluates pipeline performance across multiple datasets."""

    def __init__(
        self,
        chunker: ChunkingStrategy,
        embedding_provider: EmbeddingProvider,
    ):
        """Initialize evaluator.

        Args:
            chunker: Chunking strategy
            embedding_provider: Embedding provider
        """
        self.chunker = chunker
        self.embedding_provider = embedding_provider

    def evaluate_dataset(
        self,
        dataset_name: str,
        text: str,
        ground_truth_entities: list[str] | None = None,
        ground_truth_chunks: list[str] | None = None,
        test_queries: list[dict[str, Any]] | None = None,
    ) -> DatasetMetrics:
        """Evaluate a single dataset.

        Args:
            dataset_name: Name of the dataset
            text: Input text
            ground_truth_entities: Optional list of ground truth entity names
            ground_truth_chunks: Optional list of ground truth chunk IDs
            test_queries: Optional list of test queries with expected results

        Returns:
            DatasetMetrics object
        """
        logger.info(f"Evaluating dataset: {dataset_name}")

        # Chunking
        chunks = self.chunker.chunk(
            text=text,
            origin_dataset=dataset_name,
            origin_file=f"{dataset_name}.txt",
        )

        # Chunking quality metrics
        chunking_quality = self._evaluate_chunking_quality(chunks, ground_truth_entities)

        # Entity extraction (simplified - would use actual extraction)
        total_entities = len(ground_truth_entities) if ground_truth_entities else 0
        total_relations = 0  # Would be extracted from KG

        # Entity extraction effectiveness
        entity_extraction = {}
        if ground_truth_entities:
            entity_extraction = self._evaluate_entity_extraction(chunks, ground_truth_entities)

        return DatasetMetrics(
            dataset_name=dataset_name,
            chunking_quality=chunking_quality,
            entity_extraction=entity_extraction,
            total_chunks=len(chunks),
            total_entities=total_entities,
            total_relations=total_relations,
        )

    def _evaluate_chunking_quality(
        self,
        chunks: list[Chunk],
        ground_truth_entities: list[str] | None = None,
    ) -> dict[str, float]:
        """Evaluate chunking quality.

        Args:
            chunks: List of chunks
            ground_truth_entities: Optional ground truth entities

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        # Token distribution
        token_counts = [chunk.token_count for chunk in chunks]
        if token_counts:
            metrics["mean_token_count"] = sum(token_counts) / len(token_counts)
            metrics["std_token_count"] = (
                sum((x - metrics["mean_token_count"]) ** 2 for x in token_counts)
                / len(token_counts)
            ) ** 0.5

        # Entity boundary preservation (simplified)
        if ground_truth_entities:
            violations = 0
            for chunk in chunks:
                # Check if entities are split across chunks (simplified check)
                chunk_text_lower = chunk.text.lower()
                for entity in ground_truth_entities:
                    entity_lower = entity.lower()
                    # Simple heuristic: if entity appears partially
                    if entity_lower in chunk_text_lower:
                        # Check if it's at boundary (simplified)
                        if chunk_text_lower.startswith(entity_lower) or chunk_text_lower.endswith(
                            entity_lower
                        ):
                            violations += 1

            total_entities = len(ground_truth_entities)
            metrics["entity_boundary_violation_rate"] = (
                violations / total_entities if total_entities > 0 else 0.0
            )
        else:
            metrics["entity_boundary_violation_rate"] = 0.0

        return metrics

    def _evaluate_entity_extraction(
        self,
        chunks: list[Chunk],
        ground_truth_entities: list[str],
    ) -> dict[str, float]:
        """Evaluate entity extraction effectiveness.

        Args:
            chunks: List of chunks
            ground_truth_entities: Ground truth entity names

        Returns:
            Dictionary of extraction metrics
        """
        # Simplified: check if entities appear in chunks
        extracted_entities = set()
        for chunk in chunks:
            chunk_text_lower = chunk.text.lower()
            for entity in ground_truth_entities:
                if entity.lower() in chunk_text_lower:
                    extracted_entities.add(entity)

        precision = len(extracted_entities) / len(chunks) if chunks else 0.0
        recall = (
            len(extracted_entities) / len(ground_truth_entities) if ground_truth_entities else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def evaluate_multiple_datasets(
        self,
        datasets: list[dict[str, Any]],
    ) -> EvaluationResult:
        """Evaluate multiple datasets.

        Args:
            datasets: List of dataset dictionaries with text, ground truth, etc.

        Returns:
            EvaluationResult with comparison table and observations
        """
        all_metrics = []

        for dataset_data in datasets:
            metrics = self.evaluate_dataset(
                dataset_name=dataset_data["name"],
                text=dataset_data["text"],
                ground_truth_entities=dataset_data.get("ground_truth_entities"),
                test_queries=dataset_data.get("test_queries"),
            )
            all_metrics.append(metrics)

        # Create comparison table
        comparison_table = self._create_comparison_table(all_metrics)

        # Generate observations
        observations = self._generate_observations(all_metrics)

        # Identify failure cases
        failure_cases = self._identify_failure_cases(all_metrics)

        return EvaluationResult(
            dataset_metrics=all_metrics,
            comparison_table=comparison_table,
            observations=observations,
            failure_cases=failure_cases,
        )

    def _create_comparison_table(
        self,
        metrics_list: list[DatasetMetrics],
    ) -> dict[str, dict[str, Any]]:
        """Create comparison table from metrics.

        Args:
            metrics_list: List of dataset metrics

        Returns:
            Comparison table dictionary
        """
        table = {}

        for metrics in metrics_list:
            table[metrics.dataset_name] = {
                "chunking_quality": metrics.chunking_quality,
                "entity_extraction": metrics.entity_extraction,
                "total_chunks": metrics.total_chunks,
                "total_entities": metrics.total_entities,
            }

        return table

    def _generate_observations(
        self,
        metrics_list: list[DatasetMetrics],
    ) -> list[str]:
        """Generate observations from metrics.

        Args:
            metrics_list: List of dataset metrics

        Returns:
            List of observation strings
        """
        observations = []

        # Compare chunking quality
        violation_rates = [
            m.chunking_quality.get("entity_boundary_violation_rate", 0.0) for m in metrics_list
        ]
        if violation_rates:
            avg_violation = sum(violation_rates) / len(violation_rates)
            observations.append(f"Average entity boundary violation rate: {avg_violation:.2%}")

        return observations

    def _identify_failure_cases(
        self,
        metrics_list: list[DatasetMetrics],
    ) -> list[dict[str, Any]]:
        """Identify failure cases.

        Args:
            metrics_list: List of dataset metrics

        Returns:
            List of failure case dictionaries
        """
        failures = []

        for metrics in metrics_list:
            # Check for high violation rate
            violation_rate = metrics.chunking_quality.get("entity_boundary_violation_rate", 0.0)
            if violation_rate > 0.1:  # > 10%
                failures.append(
                    {
                        "dataset": metrics.dataset_name,
                        "type": "high_entity_boundary_violation",
                        "value": violation_rate,
                        "description": f"Entity boundary violation rate is {violation_rate:.2%}",
                    }
                )

        return failures
