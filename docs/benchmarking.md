# Benchmarking and Technical Evidence

DRG benchmark artifacts are designed to answer five adoption questions:

- How accurate is extraction on a known gold dataset?
- How does a candidate run compare with a baseline?
- What are the latency, throughput, and memory costs?
- Can a large dataset run be reproduced?
- Can external systems be scored with the same metrics?

## Official Report Format

Use `drg eval run` to produce JSON and optional Markdown artifacts:

```bash
drg eval run examples/benchmarks/synthetic_kg_benchmark.json \
  --measure-performance \
  -o reports/current.json \
  --markdown-output reports/current.md
```

The JSON report includes:

- `aggregate`: extraction, graph, reasoning, and overall scores.
- `datasets[].components`: per-component metrics and diagnostics.
- `datasets[].metadata.performance`: per-dataset wall time, p50/p95 latency, Python heap, optional process RSS, and throughput.
- `metadata.performance`: run-level latency and peak-memory summary.

## Extraction Accuracy

The extraction report separates:

- entity extraction precision, recall, and F1
- relationship extraction precision, recall, and F1
- event extraction precision, recall, and F1
- graph coverage and orphan-node rate

Failure diagnostics classify common misses as `missing_entity`, `wrong_type`,
`missing_relation`, `wrong_relation_type`, `hallucinated_entity`,
`hallucinated_edge`, `missing_event`, or `extra_event`.

## Large Dataset Runs

Generate deterministic scale datasets without committing large JSON files:

```bash
python examples/benchmarks/generate_scale_dataset.py \
  --documents 10000 \
  --output /tmp/drg_scale_10k.json \
  --oracle-output /tmp/drg_scale_10k_oracle.json
```

Then evaluate either a live DRG run:

```bash
drg eval run /tmp/drg_scale_10k.json --measure-performance -o reports/scale_10k.json
```

or an existing prediction artifact:

```bash
drg eval run /tmp/drg_scale_10k.json \
  --predictions /tmp/drg_scale_10k_oracle.json \
  --adapter oracle \
  -o reports/scale_10k_oracle.json
```

Heavy live runs should stay out of normal PR CI. The scheduled benchmark workflow
in `.github/workflows/benchmarks.yml` runs a deterministic smoke benchmark and
uploads artifacts; use `workflow_dispatch` with a larger document count for
manual scale checks.

## External Adapter Contract

External systems do not need to import DRG. They only need to write prediction
JSON:

```json
{
  "adapter": "external-baseline",
  "model": "baseline-model",
  "predictions": {
    "dataset_name": {
      "entities": [["Apple", "Company"]],
      "relations": [["Apple", "ACQUIRED", "Beats"]],
      "events": []
    }
  }
}
```

Score that artifact against the same gold dataset:

```bash
drg eval run examples/benchmarks/corporate_acquisition_benchmark.json \
  --predictions external_predictions.json \
  --adapter external-baseline \
  -o reports/external-baseline.json \
  --markdown-output reports/external-baseline.md
```

Compare two reports with:

```bash
drg eval compare reports/drg.json reports/external-baseline.json -o reports/drg_vs_external.md
```

For fair comparisons, keep corpus, model budget, chunking assumptions, and
reported metrics fixed across systems.
