"""DSPy optimizer module for iterative learning and improvement."""

from .metrics import (
    ExtractionMetrics,
    calculate_metrics,
    compare_metrics,
)
from .optimizer import (
    DRGOptimizer,
    OptimizerConfig,
    create_optimizer,
    evaluate_extraction,
)

__all__ = [
    "DRGOptimizer",
    "ExtractionMetrics",
    "OptimizerConfig",
    "calculate_metrics",
    "compare_metrics",
    "create_optimizer",
    "evaluate_extraction",
]
