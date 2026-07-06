# Case Studies

Use this page to publish reproducible DRG runs from realistic user scenarios.
Each case study should include input, schema, command, output snapshot, accuracy
report, and performance report.

## Financial News

Source candidates:

- `inputs/input2.txt`
- `examples/quickstarts/`
- `examples/benchmarks/corporate_acquisition_benchmark.json`

Recommended artifact set:

- input text
- domain schema
- `drg extract ...` command
- generated KG JSON snapshot
- `drg eval run ... --measure-performance` JSON and Markdown reports
- notes on missed entities, wrong relation types, and hallucinated edges

## Biomedical Abstracts

Source candidates:

- `inputs/`
- project-specific benchmark JSON under `examples/benchmarks/`

Recommended artifact set:

- entity types and relation groups used for the domain
- extraction model/provider and temperature
- extraction accuracy report
- latency and memory report
- error analysis for abbreviations and aliases

## Wikipedia Or Corporate Docs

Source candidates:

- `inputs/input2.txt`
- `outputs/output1_kg.json`
- `examples/full_pipeline_example.py`

Recommended artifact set:

- multi-document ingestion command
- graph validation output
- community summary or cluster report
- benchmark report when gold annotations are available

## Adoption Metrics

Until the project has organic community adoption, track:

- reproducible case studies merged
- external prediction adapters submitted
- benchmark datasets contributed
- public reports generated from the official artifact format
- issues labeled `good first issue` and closed by new contributors
