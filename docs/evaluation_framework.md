# Evaluation Framework

The evaluation framework measures DRG quality across extraction, graph
construction, reasoning, deterministic graph-query behavior, and performance.

Status: opt-in, pure Python, independent from production extraction/query code.

## Current Limitations

Verified from source:

- `tests/multi_dataset/evaluation.py` is a simplified test helper, not a
  reusable benchmark framework.
- `drg.optimizer.metrics` computes entity/relation F1 for optimizer workflows,
  but does not evaluate events, graph quality, reasoning, graph-query metrics,
  benchmark datasets, reports, or regressions.
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
- A suite manifest loaded with `load_benchmark_suite()` or
  `load_official_benchmark_suite()`.

That lets teams compare old vs new versions, model A vs model B, prompt A vs
prompt B, or any custom pipeline without coupling evaluation to extraction
internals.

## Components Evaluated

| Component | Metrics |
|---|---|
| Entity extraction | alias-aware precision, recall, F1; optional span-overlap matching |
| Relationship extraction | alias-aware source/type/target precision, recall, F1 |
| Event extraction | event type, participant role, and timestamp precision, recall, F1 |
| Entity resolution | pairwise precision, recall, F1 |
| Graph construction | entity coverage, relation coverage, density, orphan-node rate |
| Query & reasoning | inference precision, recall, F1 |
| Community quality | pairwise community precision, recall, F1 |
| Evidence quality | evidence/span precision, recall, F1 |
| Confidence calibration | ECE, Brier score, reliability bins |
| Runtime performance | mean wall time, total wall time, dataset count |

## Benchmark Dataset Schema

```json
{
  "name": "dataset_name",
  "text": "source text",
  "documents": [{"id": "doc-1", "text": "source text"}],
  "gold_entities": [
    {
      "id": "ent:openai",
      "name": "OpenAI",
      "type": "Company",
      "aliases": ["Open AI"],
      "span": [0, 6],
      "provenance": {"document_id": "doc-1", "source_span": [0, 6]}
    }
  ],
  "gold_relations": [
    {
      "id": "rel:investment",
      "source": "Microsoft",
      "relationship_type": "INVESTED_IN",
      "target": "OpenAI",
      "provenance": {"document_id": "doc-1"}
    }
  ],
  "gold_events": [
    {
      "id": "event:acquisition",
      "event_type": "Acquisition",
      "participants": {"acquirer": ["Microsoft"], "acquired": ["GitHub"]},
      "timestamp": "2018",
      "provenance": {"document_id": "doc-1"}
    }
  ],
  "gold_evidence": [
    {
      "fact_id": "rel:investment",
      "snippet": "Microsoft invested in OpenAI.",
      "source_span": [0, 29]
    }
  ],
  "gold_inferred_relations": [
    {"source": "A", "relationship_type": "CONNECTED_TO", "target": "B"}
  ],
  "gold_communities": {"OpenAI": "ai", "Microsoft": "ai"},
  "metadata": {"domain": "business", "difficulty": "smoke"}
}
```

See `examples/benchmarks/synthetic_kg_benchmark.json`.

The minimal official suite manifest lives at
`examples/benchmarks/official_suite.json`. It references deterministic DRG
fixtures and lists adapter targets (`drg`, `external-baseline`, `kg-builder`)
so external tools can be compared by writing a small adapter that returns
`PipelinePrediction`.

## Usage

```python
from drg.evaluation import BenchmarkRunner, PipelinePrediction, load_benchmark_datasets

dataset = load_benchmark_datasets("examples/benchmarks/synthetic_kg_benchmark.json")[0]
prediction = PipelinePrediction(
    entities=[("OpenAI", "Company")],
    relations=[("Microsoft", "INVESTED_IN", "OpenAI")],
)

report = BenchmarkRunner(run_id="candidate").evaluate(
    [dataset],
    predictions=[prediction],
)
```

Suite usage:

```python
from drg.evaluation import BenchmarkRunner, load_official_benchmark_suite

suite = load_official_benchmark_suite()
report = BenchmarkRunner(run_id="candidate").evaluate(suite.datasets, runner=my_adapter)
```

CLI usage with JSON and Markdown artifacts:

```bash
drg eval run examples/benchmarks/synthetic_kg_benchmark.json \
  --measure-performance \
  -o reports/current.json \
  --markdown-output reports/current.md
```

External adapter usage:

```bash
drg eval run examples/benchmarks/synthetic_kg_benchmark.json \
  --predictions external_predictions.json \
  --adapter external-baseline \
  -o reports/external-baseline.json
```

See `docs/benchmarking.md` for the prediction artifact contract, large synthetic
dataset generation, and scheduled benchmark workflow.

CLI catalog:

```bash
drg eval list
drg eval list --json
```

Runnable demo:

```bash
python3 examples/evaluation_framework_example.py
```

## Regression Comparison

```python
from drg.evaluation import compare_reports

comparison = compare_reports(old_report, new_report, regression_threshold=0.01)
for delta in comparison.deltas:
    print(delta.metric, delta.status)
```

Use this in CI to fail a run when key metrics drop by more than an accepted
tolerance.

Metric- and dataset-specific thresholds are supported for release gates:

```bash
drg eval compare reports/baseline.json reports/candidate.json \
  --thresholds-json reports/thresholds.json \
  --fail-on-regression
```

## Reproducibility

- Keep benchmark files in version control.
- Record model, prompt, schema, and pipeline configuration in
  `EvaluationReport.metadata`.
- Store JSON reports for historical comparison.
- Use deterministic synthetic datasets for fast regression tests and larger
  gold-standard datasets for scheduled benchmarks.
