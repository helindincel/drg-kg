"""I/O helpers: load/save benchmark datasets, suites, evaluation reports, predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._types import (
    BenchmarkDataset,
    BenchmarkSuite,
    EvaluationReport,
    PipelinePrediction,
)

__all__ = [
    "load_benchmark_datasets",
    "load_benchmark_suite",
    "load_evaluation_report",
    "load_official_benchmark_suite",
    "load_prediction_artifact",
    "save_json_report",
    "save_markdown_report",
]

# ---------------------------------------------------------------------------
# Official built-in benchmark suite (shipped inside the package)
# ---------------------------------------------------------------------------

_OFFICIAL_SUITE = BenchmarkSuite(
    name="drg-official-v1",
    adapters=[],
    metadata={"version": "1.0", "description": "DRG built-in benchmark suite"},
    datasets=[
        BenchmarkDataset(
            name="person_biography",
            text=(
                "Marie Curie was born in Warsaw in 1867. She studied at the University "
                "of Paris and later became a professor there. She discovered polonium "
                "and radium, and won the Nobel Prize in Physics in 1903 and the Nobel "
                "Prize in Chemistry in 1911."
            ),
            gold_entities=[
                {"name": "Marie Curie", "type": "Person"},
                {"name": "Warsaw", "type": "Place"},
                {"name": "University of Paris", "type": "Organization"},
                {"name": "polonium", "type": "Discovery"},
                {"name": "radium", "type": "Discovery"},
            ],
            gold_relations=[
                {"source": "Marie Curie", "type": "born_in", "target": "Warsaw"},
                {
                    "source": "Marie Curie",
                    "type": "studied_at",
                    "target": "University of Paris",
                },
                {
                    "source": "Marie Curie",
                    "type": "discovered",
                    "target": "polonium",
                },
                {"source": "Marie Curie", "type": "discovered", "target": "radium"},
            ],
            metadata={"domain": "biography", "task": "entity_relation"},
        ),
        BenchmarkDataset(
            name="company_acquisition",
            text=(
                "Apple Inc. acquired Beats Electronics in 2014 for approximately "
                "$3 billion. Beats was founded by Dr. Dre and Jimmy Iovine in "
                "Santa Monica, California."
            ),
            gold_entities=[
                {"name": "Apple Inc.", "type": "Organization"},
                {"name": "Beats Electronics", "type": "Organization"},
                {"name": "Dr. Dre", "type": "Person"},
                {"name": "Jimmy Iovine", "type": "Person"},
                {"name": "Santa Monica", "type": "Place"},
            ],
            gold_relations=[
                {
                    "source": "Apple Inc.",
                    "type": "acquired",
                    "target": "Beats Electronics",
                },
                {
                    "source": "Dr. Dre",
                    "type": "founded",
                    "target": "Beats Electronics",
                },
                {
                    "source": "Jimmy Iovine",
                    "type": "founded",
                    "target": "Beats Electronics",
                },
                {
                    "source": "Beats Electronics",
                    "type": "located_in",
                    "target": "Santa Monica",
                },
            ],
            metadata={"domain": "business", "task": "entity_relation"},
        ),
    ],
)


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------


def load_official_benchmark_suite() -> BenchmarkSuite:
    """Return the built-in DRG benchmark suite."""
    return _OFFICIAL_SUITE


def load_benchmark_suite(path: str | Path) -> BenchmarkSuite:
    """Load a :class:`~drg.evaluation.BenchmarkSuite` from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return BenchmarkSuite.from_dict(data)


def load_benchmark_datasets(path: str | Path | None) -> list[BenchmarkDataset]:
    """Load benchmark datasets from a JSON file (array or suite).

    Accepts either a suite JSON (``{"name": ..., "datasets": [...]}``), a bare
    array of dataset dicts, or a single dataset dict.  Returns ``None`` path as
    the official suite's datasets.
    """
    if path is None:
        return list(_OFFICIAL_SUITE.datasets)
    data: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [BenchmarkDataset.from_dict(d) for d in data]
    if isinstance(data, dict) and "datasets" in data:
        return [BenchmarkDataset.from_dict(d) for d in data["datasets"]]
    if isinstance(data, dict) and "name" in data and "text" in data:
        return [BenchmarkDataset.from_dict(data)]
    raise ValueError(f"Unrecognised benchmark JSON format in {path}")


def load_evaluation_report(path: str | Path) -> EvaluationReport:
    """Deserialise an :class:`~drg.evaluation.EvaluationReport` from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvaluationReport.from_dict(data)


def load_prediction_artifact(
    path: str | Path,
) -> tuple[list[PipelinePrediction], dict[str, Any]]:
    """Load pre-computed predictions from a JSON artifact.

    Returns a ``(predictions, metadata)`` tuple.  The JSON may be:

    * An array of prediction dicts.
    * A dict with ``"predictions"`` key (and optional ``"metadata"``).
    """
    data: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [PipelinePrediction.from_dict(p) for p in data], {}
    if isinstance(data, dict) and "predictions" in data:
        raw_preds = data["predictions"]
        # Accept either a list of prediction dicts or a dict keyed by dataset name
        if isinstance(raw_preds, dict):
            predictions = [PipelinePrediction.from_dict(v) for v in raw_preds.values()]
        else:
            predictions = [PipelinePrediction.from_dict(p) for p in raw_preds]
        # Extract metadata: prefer explicit "metadata" key, but also surface
        # any top-level scalar fields (e.g. "adapter") the caller can merge
        meta: dict[str, Any] = dict(data.get("metadata") or {})
        for k, v in data.items():
            if k not in ("predictions", "metadata") and not isinstance(v, (dict, list)):
                meta.setdefault(k, v)
        return predictions, meta
    raise ValueError(f"Unrecognised prediction artifact format in {path}")


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------


def save_json_report(report: EvaluationReport, path: str | Path) -> None:
    """Serialise *report* to a JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_markdown_report(report: EvaluationReport, path: str | Path) -> None:
    """Write a human-readable Markdown report to *path*."""
    from ._compare import render_markdown_report

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown_report(report), encoding="utf-8")
