#!/usr/bin/env python3
"""Run the evaluation framework on a tiny synthetic benchmark.

Run:

    python3 examples/evaluation_framework_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drg.evaluation import (
    BenchmarkRunner,
    PipelinePrediction,
    load_benchmark_dataset,
    render_markdown_report,
)


def main() -> int:
    dataset = load_benchmark_dataset("examples/benchmarks/synthetic_kg_benchmark.json")

    prediction = PipelinePrediction(
        entities=[
            ("Sam Altman", "Person"),
            ("OpenAI", "Company"),
            ("Microsoft", "Company"),
            ("GitHub", "Company"),
            ("GitHub Acquisition", "Event:Acquisition"),
        ],
        relations=[
            ("Sam Altman", "WORKED_WITH", "OpenAI"),
            ("Microsoft", "INVESTED_IN", "OpenAI"),
            ("Microsoft", "ACQUIRED", "GitHub"),
        ],
        events=[
            {
                "event_type": "Acquisition",
                "participants": {
                    "acquirer": ["Microsoft"],
                    "acquired": ["GitHub"],
                },
                "timestamp": "2018",
            }
        ],
        inferred_relations=[
            ("Sam Altman", "CONNECTED_TO", "Microsoft"),
        ],
        resolved_clusters={
            "Sam Altman": "ai",
            "OpenAI": "ai",
            "Microsoft": "ai",
            "GitHub": "developer_tools",
        },
        communities={
            "Sam Altman": "ai",
            "OpenAI": "ai",
            "Microsoft": "ai",
            "GitHub": "developer_tools",
        },
        metadata={"model": "deterministic_example"},
    )

    report = BenchmarkRunner(
        run_id="example_run",
        metadata={"pipeline": "manual_prediction"},
    ).evaluate([dataset], predictions={dataset.name: prediction})

    print(render_markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
