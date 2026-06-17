# Evaluation Framework

The evaluation framework measures DRG quality across extraction, graph
construction, reasoning, retrieval, and hybrid retrieval.

Status: opt-in, pure Python, independent from production extraction/query code.

## Current Limitations

Verified from source:

- `tests/multi_dataset/evaluation.py` is a simplified test helper, not a
  reusable benchmark framework.
- `drg.optimizer.metrics` computes entity/relation F1 for optimizer workflows,
  but does not evaluate events, graph quality, reasoning, retrieval, hybrid
  retrieval, benchmark datasets, reports, or regressions.
- Existing extraction/query modules expose enough outputs to evaluate, but the
  evaluation logic was not centralized.

## Architecture

```
BenchmarkDataset
      │
      ▼
PipelinePrediction  ──► BenchmarkRunner ──► EvaluationReport
                                      │
                                      ├── JSON report
                                      ├── Markdown report
                                      └── RegressionComparison
```

The runner accepts either:

- `PipelinePrediction` objects keyed by dataset name.
- A callable `runner(dataset) -> PipelinePrediction`.

That lets teams compare old vs new versions, model A vs model B, prompt A vs
prompt B, or any custom pipeline without coupling evaluation to extraction
internals.

## Components Evaluated

| Component | Metrics |
|---|---|
| Entity extraction | precision, recall, F1 |
| Relationship extraction | precision, recall, F1 |
| Event extraction | precision, recall, F1 |
| Entity resolution | pairwise precision, recall, F1 |
| Graph construction | entity coverage, relation coverage, density, orphan-node rate |
| Query & reasoning | inference precision, recall, F1 |
| Retrieval | Precision@K, Recall@K, MRR, NDCG |
| Hybrid retrieval | Precision@K, Recall@K, MRR, NDCG |
| Community quality | pairwise community precision, recall, F1 |

## Benchmark Dataset Schema

```json
{
  "name": "dataset_name",
  "text": "source text",
  "gold_entities": [{"name": "OpenAI", "type": "Company"}],
  "gold_relations": [
    {"source": "Microsoft", "relationship_type": "INVESTED_IN", "target": "OpenAI"}
  ],
  "gold_events": [
    {
      "event_type": "Acquisition",
      "participants": {"acquirer": ["Microsoft"], "acquired": ["GitHub"]},
      "timestamp": "2018"
    }
  ],
  "gold_inferred_relations": [
    {"source": "A", "relationship_type": "CONNECTED_TO", "target": "B"}
  ],
  "query_cases": [
    {
      "query": "What acquisitions involve Microsoft?",
      "relevant_entities": ["Microsoft", "GitHub"],
      "relevant_chunks": ["doc_github_chunk_000"]
    }
  ],
  "gold_communities": {"OpenAI": "ai", "Microsoft": "ai"}
}
```

See `examples/benchmarks/synthetic_kg_benchmark.json`.

## Usage

```python
from drg.evaluation import BenchmarkRunner, PipelinePrediction, load_benchmark_dataset

dataset = load_benchmark_dataset("examples/benchmarks/synthetic_kg_benchmark.json")
prediction = PipelinePrediction(
    entities=[("OpenAI", "Company")],
    relations=[("Microsoft", "INVESTED_IN", "OpenAI")],
    query_results={"Which companies are connected to OpenAI?": ["Microsoft", "OpenAI"]},
)

report = BenchmarkRunner(run_id="candidate").evaluate(
    [dataset],
    predictions={dataset.name: prediction},
)
```

Runnable demo:

```bash
python3 examples/evaluation_framework_example.py
```

## Regression Comparison

```python
from drg.evaluation import compare_reports

comparison = compare_reports(old_report, new_report, regression_threshold=0.01)
for regression in comparison.regressions:
    print(regression)
```

Use this in CI to fail a run when key metrics drop by more than an accepted
tolerance.

## Reproducibility

- Keep benchmark files in version control.
- Record model, prompt, schema, and pipeline configuration in
  `EvaluationReport.metadata`.
- Store JSON reports for historical comparison.
- Use deterministic synthetic datasets for fast regression tests and larger
  gold-standard datasets for scheduled benchmarks.

