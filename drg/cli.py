#!/usr/bin/env python3
"""CLI interface for DRG - Declarative Relationship Generation"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from .chunking import create_chunker
from .graph import KG
from .graph.builders import build_enhanced_kg
from .schema import DRGSchema, Entity, Relation, load_schema_from_json
from .utils.env_loader import load_dotenv

extract_from_chunks = None
extract_typed = None
generate_schema_from_text = None


def _ensure_extraction_imports() -> None:
    """Load DSPy-backed extraction only when an extraction command runs."""

    global extract_from_chunks, extract_typed, generate_schema_from_text
    if extract_from_chunks and extract_typed and generate_schema_from_text:
        return

    from .extract import (
        extract_from_chunks as _extract_from_chunks,
    )
    from .extract import (
        extract_typed as _extract_typed,
    )
    from .extract import (
        generate_schema_from_text as _generate_schema_from_text,
    )

    extract_from_chunks = extract_from_chunks or _extract_from_chunks
    extract_typed = extract_typed or _extract_typed
    generate_schema_from_text = generate_schema_from_text or _generate_schema_from_text


def create_default_schema():
    """Default schema: Company -> Product."""
    return DRGSchema(
        entities=[Entity("Company"), Entity("Product")],
        relations=[Relation("produces", "Company", "Product")],
    )


def main():
    # Load local .env if present (keeps API keys out of code)
    load_dotenv(".env", override=False)

    parser = argparse.ArgumentParser(
        description="DRG - Declarative Relationship Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Common commands:
  drg extract sample.txt --auto-schema -o output_kg.json
  drg validate output_kg.json
  drg diff old_kg.json new_kg.json --json
  drg versions list output_kg.json
  drg eval list

Use `drg <command> --help` for command-specific options.""",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- Extract command ---
    extract_parser = subparsers.add_parser("extract", help="Extract KG from text")

    extract_parser.add_argument("input", type=str, help="Input text file or '-' for stdin")
    extract_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="-",
        help="Output JSON file (default: stdout)",
    )
    extract_parser.add_argument(
        "--schema",
        type=str,
        help="Custom schema JSON file",
    )
    extract_parser.add_argument(
        "--auto-schema",
        action="store_true",
        help="Generate an EnhancedDRGSchema from the input text",
    )
    extract_parser.add_argument(
        "--output-format",
        type=str,
        default=None,
        choices=["legacy", "enhancedkg"],
        help=(
            "JSON format to write. Defaults to enhancedkg for --auto-schema and "
            "legacy for simple schema extraction."
        ),
    )
    extract_parser.add_argument(
        "--no-hub-validation",
        action="store_true",
        help="Disable hub-dominance validation gate",
    )
    extract_parser.add_argument(
        "--update",
        type=str,
        default=None,
        metavar="EXISTING_KG_JSON",
        help="Merge extracted facts into an existing EnhancedKG JSON file",
    )
    extract_parser.add_argument(
        "--update-strategy",
        type=str,
        default="prefer_existing",
        choices=["prefer_existing", "prefer_new", "union"],
        help="Conflict strategy for --update merges (default: prefer_existing)",
    )
    extract_parser.add_argument(
        "--update-document-id",
        type=str,
        default=None,
        help="Document identifier to store in provenance metadata during --update",
    )
    extract_parser.add_argument(
        "--diff-output",
        type=str,
        default=None,
        help="Write the incremental merge diff report to this JSON file when --update is used",
    )
    extract_parser.add_argument(
        "--infer",
        action="store_true",
        help="Run the reasoning layer after extraction/update to add inferred edges",
    )
    extract_parser.add_argument(
        "--extract-events",
        action="store_true",
        help="Extract event frames and map them into the KG",
    )
    extract_parser.add_argument(
        "--events-registry",
        type=str,
        default=None,
        help="Path to a custom event registry JSON file",
    )
    extract_parser.add_argument(
        "--events-use-example",
        action="store_true",
        help="Use the bundled example event registry",
    )
    extract_parser.add_argument(
        "--infer-min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for inferred edges (default: 0.5)",
    )
    extract_parser.add_argument(
        "--infer-disable-rule",
        action="append",
        default=[],
        help="Disable a reasoning rule by name; may be provided multiple times",
    )
    extract_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model override, e.g. openai/gpt-4o-mini or gemini/gemini-2.0-flash-exp",
    )
    extract_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Provider API key override for this invocation",
    )
    extract_parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Provider base URL override, useful for local gateways or Ollama",
    )
    extract_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature override (default: 0.0)",
    )

    # --- Eval command ---
    eval_parser = subparsers.add_parser("eval", help="Evaluation framework")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command")

    # eval run
    eval_run_parser = eval_subparsers.add_parser("run", help="Run benchmark")
    eval_run_parser.add_argument("dataset", type=str, help="Dataset JSON file or directory")
    eval_run_parser.add_argument("-o", "--output", type=str, help="Output report file")
    eval_run_parser.add_argument("--run-id", type=str, help="Run ID")
    eval_run_parser.add_argument("--model", type=str, help="Model (overrides env)")
    eval_run_parser.add_argument("--api-key", type=str, help="API Key (overrides env)")
    eval_run_parser.add_argument(
        "--predictions",
        type=str,
        help="Prediction artifact JSON from DRG or an external adapter",
    )
    eval_run_parser.add_argument(
        "--adapter", type=str, help="Adapter/system name for report metadata"
    )
    eval_run_parser.add_argument(
        "--measure-performance",
        action="store_true",
        help="Include wall-clock latency, throughput, and memory metrics in the report",
    )
    eval_run_parser.add_argument(
        "--markdown-output",
        type=str,
        help="Optional Markdown report path written in addition to the main output",
    )

    # eval compare
    eval_comp_parser = eval_subparsers.add_parser("compare", help="Compare reports")
    eval_comp_parser.add_argument("baseline", type=str, help="Baseline JSON")
    eval_comp_parser.add_argument("candidate", type=str, help="Candidate JSON")
    eval_comp_parser.add_argument("-o", "--output", type=str, help="Output markdown")
    eval_comp_parser.add_argument("--threshold", type=float, default=0.01)

    # eval list
    eval_list_parser = eval_subparsers.add_parser("list", help="List benchmark suite datasets")
    eval_list_parser.add_argument("--suite", type=str, default=None, help="Suite manifest JSON")
    eval_list_parser.add_argument("--json", action="store_true")

    # --- Validate command ---
    validate_parser = subparsers.add_parser("validate", help="Validate a Knowledge Graph JSON file")
    validate_parser.add_argument("graph", type=str, help="EnhancedKG JSON file")
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON validation report",
    )

    # --- Diff command ---
    diff_parser = subparsers.add_parser("diff", help="Diff two Knowledge Graph JSON snapshots")
    diff_parser.add_argument("old_graph", type=str, help="Baseline EnhancedKG JSON file")
    diff_parser.add_argument("new_graph", type=str, help="Candidate EnhancedKG JSON file")
    diff_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON diff report",
    )
    diff_parser.add_argument(
        "--fail-on-change",
        action="store_true",
        help="Exit with code 1 when the snapshots differ",
    )

    # --- Versions command ---
    versions_parser = subparsers.add_parser("versions", help="Manage graph version snapshots")
    versions_subparsers = versions_parser.add_subparsers(dest="versions_command")

    versions_list = versions_subparsers.add_parser("list", help="List graph versions")
    versions_list.add_argument("graph", type=str, help="EnhancedKG JSON file")
    versions_list.add_argument(
        "--versions-dir",
        type=str,
        default=None,
        help="Directory containing graph version snapshots",
    )
    versions_list.add_argument("--json", action="store_true", help="Emit JSON output")

    versions_diff = versions_subparsers.add_parser("diff", help="Diff two graph versions")
    versions_diff.add_argument("graph", type=str, help="EnhancedKG JSON file")
    versions_diff.add_argument("old_version", type=str, help="Baseline version id")
    versions_diff.add_argument("new_version", type=str, help="Candidate version id")
    versions_diff.add_argument(
        "--versions-dir",
        type=str,
        default=None,
        help="Directory containing graph version snapshots",
    )
    versions_diff.add_argument("--json", action="store_true", help="Emit JSON output")
    versions_diff.add_argument(
        "--fail-on-change",
        action="store_true",
        help="Exit with code 1 when the versions differ",
    )

    versions_rollback = versions_subparsers.add_parser(
        "rollback", help="Rollback graph file to a version"
    )
    versions_rollback.add_argument("graph", type=str, help="EnhancedKG JSON file")
    versions_rollback.add_argument("version", type=str, help="Version id to restore")
    versions_rollback.add_argument(
        "--versions-dir",
        type=str,
        default=None,
        help="Directory containing graph version snapshots",
    )

    # Legacy support: keep `drg <file>` as shorthand for `drg extract <file>`.
    if len(sys.argv) > 1 and sys.argv[1] not in [
        "extract",
        "eval",
        "validate",
        "diff",
        "versions",
        "-h",
        "--help",
    ]:
        sys.argv.insert(1, "extract")

    args = parser.parse_args()

    if args.command == "extract":
        _handle_extract(args)
    elif args.command == "eval":
        _handle_eval(args)
    elif args.command == "validate":
        _handle_validate(args)
    elif args.command == "diff":
        _handle_diff(args)
    elif args.command == "versions":
        _handle_versions(args)
    else:
        parser.print_help()


def _handle_extract(args):
    _ensure_extraction_imports()

    # Read input
    if args.input == "-":
        text = sys.stdin.read()
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        text = input_path.read_text(encoding="utf-8")

    # Load/generate schema
    if args.auto_schema:
        schema = None
    elif args.schema:
        try:
            schema = load_schema_from_json(args.schema)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error: Invalid schema file: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to load schema: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        schema = create_default_schema()

    if args.no_hub_validation:
        os.environ["DRG_VALIDATE_HUB_DOMINANCE"] = "0"
    if args.model:
        os.environ["DRG_MODEL"] = args.model
    if args.api_key:
        model = args.model or os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
        if "gemini" in model.lower():
            os.environ["GEMINI_API_KEY"] = args.api_key
            os.environ["GOOGLE_API_KEY"] = args.api_key
        elif "anthropic" in model.lower() or "claude" in model.lower():
            os.environ["ANTHROPIC_API_KEY"] = args.api_key
        elif "openrouter" in model.lower():
            os.environ["OPENROUTER_API_KEY"] = args.api_key
        else:
            os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["DRG_BASE_URL"] = args.base_url
    if args.temperature != 0.0:
        os.environ["DRG_TEMPERATURE"] = str(args.temperature)

    api_key = (
        args.api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("PERPLEXITY_API_KEY")
    )
    model = args.model or os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
    if not api_key and not model.lower().startswith("ollama"):
        print("Warning: No API key found. Cloud models require an API key.", file=sys.stderr)
        print(
            "For local models, use: --model ollama_chat/llama3 --base-url http://localhost:11434",
            file=sys.stderr,
        )

    # Determine format
    inferred_format = None
    if args.output != "-" and args.output.lower().endswith("_kg.json"):
        inferred_format = "enhancedkg"
    output_format = args.output_format or inferred_format or "legacy"

    if args.update:
        output_format = "enhancedkg"

    # Extraction
    try:
        if args.auto_schema:
            if not os.getenv("DRG_MAX_TOKENS"):
                os.environ["DRG_MAX_TOKENS"] = "4096"
            schema = generate_schema_from_text(text)

        if args.auto_schema:
            chunk_size = int(os.getenv("DRG_CHUNK_SIZE", "768"))
            chunker = create_chunker(strategy="token_based", chunk_size=chunk_size)
            chunks = chunker.chunk(text, origin_dataset="cli", origin_file=args.input)
            entities_typed, triples = extract_from_chunks(
                chunks=[{"text": c.text, "chunk_id": c.chunk_id} for c in chunks],
                schema=schema,
                enable_cross_chunk_relationships=True,
                enable_entity_resolution=True,
                two_pass_extraction=True,
            )
        else:
            if output_format == "enhancedkg":
                entities_typed, triples, enriched_relations = extract_typed(
                    text,
                    schema,
                    return_enriched=True,
                )
            else:
                entities_typed, triples = extract_typed(text, schema)
                enriched_relations = None

        triples = list(dict.fromkeys(triples))

        if output_format == "enhancedkg":
            effective_doc_id = args.update_document_id or (
                args.input if args.input != "-" else "<stdin>"
            )
            extracted_events = []
            if args.extract_events:
                from .events import EventTypeRegistry, example_event_registry, extract_events

                registry = None
                if args.events_registry:
                    registry = EventTypeRegistry.from_json(args.events_registry)
                elif args.events_use_example:
                    registry = example_event_registry()

                if registry:
                    extracted_events = extract_events(
                        text=text,
                        entities_typed=entities_typed,
                        registry=registry,
                        document_id=effective_doc_id,
                    )

            target_kg = build_enhanced_kg(
                entities_typed=entities_typed,
                triples=triples,
                schema=schema,
                source_text=text,
                enriched_relations=enriched_relations,
                document_id=effective_doc_id,
                events=extracted_events or None,
            )

            if args.update:
                from .graph import EnhancedKG, GraphMerger, MergeStrategy, NodeMergePolicy
                from .graph.versioning import create_snapshot

                update_path = Path(args.update)
                base_kg = (
                    EnhancedKG.load_json(str(update_path)) if update_path.exists() else EnhancedKG()
                )
                strategy = MergeStrategy(node_policy=NodeMergePolicy(args.update_strategy))
                merge_diff = GraphMerger(strategy).merge(
                    base_kg, target_kg, document_id=effective_doc_id
                )
                if args.diff_output:
                    diff_path = Path(args.diff_output)
                    diff_path.parent.mkdir(parents=True, exist_ok=True)
                    diff_path.write_text(
                        json.dumps(
                            {
                                "type": "merge",
                                "summary": merge_diff.summary(),
                                "diff": merge_diff.to_dict(),
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                target_kg = base_kg
                create_snapshot(
                    target_kg,
                    update_path,
                    operation="merge",
                    document_id=effective_doc_id,
                    diff_summary=merge_diff.summary(),
                )

            if args.infer:
                from .reasoning import MultiDocumentReasoner, ReasoningConfig

                infer_cfg = ReasoningConfig(
                    min_confidence=args.infer_min_confidence,
                    disabled_rules=frozenset(args.infer_disable_rule or []),
                )
                MultiDocumentReasoner(config=infer_cfg).reason(
                    target_kg, document_id=effective_doc_id
                )

            output_json = target_kg.to_json()
        else:
            kg = KG.from_typed(entities_typed, triples)
            output_json = kg.to_json()

        output_target = args.update if args.update and args.output == "-" else args.output
        if output_target == "-":
            print(output_json)
        else:
            output_path = Path(output_target)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_json, encoding="utf-8")
            print(f"Knowledge graph written to: {output_path}", file=sys.stderr)

    except Exception as e:
        import traceback

        def _redact_secrets(s: str) -> str:
            if not s:
                return s
            s = re.sub(r"(?i)(key=)[^&\s]+", r"\1REDACTED", s)
            s = re.sub(r"AIzaSy[0-9A-Za-z_-]{20,}", "REDACTED_GOOGLE_API_KEY", s)
            s = re.sub(r"sk-or-v1-[0-9a-fA-F]{20,}", "REDACTED_OPENROUTER_KEY", s)
            return s

        raw_msg = f"{type(e).__name__}: {e}"
        raw_tb = traceback.format_exc()
        print(f"Error during extraction: {_redact_secrets(raw_msg)}", file=sys.stderr)
        if os.getenv("DRG_DEBUG", "").lower() in {"1", "true", "yes"}:
            print(_redact_secrets(raw_tb), file=sys.stderr)
        sys.exit(1)


def _handle_eval(args):
    from .evaluation import (
        BenchmarkRunner,
        PipelinePrediction,
        compare_reports,
        load_benchmark_datasets,
        load_benchmark_suite,
        load_evaluation_report,
        load_official_benchmark_suite,
        load_prediction_artifact,
        render_markdown_report,
        render_regression_markdown,
        save_json_report,
        save_markdown_report,
    )

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.model:
        os.environ["DRG_MODEL"] = args.model

    if args.eval_command == "run":
        datasets = load_benchmark_datasets(args.dataset)
        metadata = {"model": os.getenv("DRG_MODEL")}
        if args.adapter:
            metadata["adapter"] = args.adapter
        predictions = None
        if args.predictions:
            predictions, prediction_metadata = load_prediction_artifact(args.predictions)
            metadata.update(prediction_metadata)
        else:
            _ensure_extraction_imports()
        runner = BenchmarkRunner(
            run_id=args.run_id,
            measure_performance=args.measure_performance,
            metadata=metadata,
        )

        def extraction_runner(ds):
            # Real extraction runner for evaluation
            e, t = extract_typed(ds.text, create_default_schema())
            return PipelinePrediction(entities=e, relations=t)

        if predictions is not None:
            report = runner.evaluate(datasets, predictions=predictions)
        else:
            report = runner.evaluate(datasets, runner=extraction_runner)

        if args.output:
            if args.output.endswith(".md"):
                save_markdown_report(report, args.output)
            else:
                save_json_report(report, args.output)
                if args.markdown_output:
                    save_markdown_report(report, args.markdown_output)
        else:
            print(render_markdown_report(report))
            if args.markdown_output:
                save_markdown_report(report, args.markdown_output)

    elif args.eval_command == "compare":
        comparison = compare_reports(
            load_evaluation_report(args.baseline),
            load_evaluation_report(args.candidate),
            regression_threshold=args.threshold,
        )
        output = render_regression_markdown(comparison)
        if args.output:
            Path(args.output).write_text(output)
        else:
            print(output)

    elif args.eval_command == "list":
        suite = load_benchmark_suite(args.suite) if args.suite else load_official_benchmark_suite()
        if args.json:
            print(json.dumps(suite.to_dict(), indent=2, ensure_ascii=False))
            return
        print(f"Benchmark suite: {suite.name}")
        if suite.adapters:
            print("Adapters: " + ", ".join(suite.adapters))
        for dataset in suite.datasets:
            task = dataset.metadata.get("task") or dataset.metadata.get("domain") or "general"
            print(f"- {dataset.name} ({task})")


def _print_validation_report(report, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return

    status = "valid" if report.valid else "invalid"
    print(f"Knowledge graph is {status}: {report.path}")
    for issue in report.issues:
        print(f"[{issue.severity}] {issue.code} at {issue.path}: {issue.message}")


def _handle_validate(args) -> None:
    from .graph.validation import validate_graph_file

    try:
        report = validate_graph_file(args.graph)
    except FileNotFoundError:
        print(f"Error: Graph file not found: {args.graph}", file=sys.stderr)
        sys.exit(2)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: Invalid graph JSON: {e}", file=sys.stderr)
        sys.exit(2)

    _print_validation_report(report, as_json=args.json)
    if not report.valid:
        sys.exit(1)


def _load_validated_graph(path: str) -> dict[str, Any]:
    from .graph.validation import load_graph_json, validate_graph_data

    data = load_graph_json(path)
    report = validate_graph_data(data, path=path)
    if not report.valid:
        messages = "; ".join(
            f"{issue.code} at {issue.path}: {issue.message}"
            for issue in report.issues
            if issue.severity == "error"
        )
        raise ValueError(messages)
    return data


def _print_diff_report(diff, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(diff.to_dict(), indent=2, ensure_ascii=False))
        return

    if not diff.changed:
        print("No graph changes detected.")
        return

    print("Knowledge graph changes detected:")
    for key, value in diff.summary().items():
        if value:
            print(f"- {key}: {value}")


def _handle_diff(args) -> None:
    from .graph.diff import diff_graph_data

    try:
        old_data = _load_validated_graph(args.old_graph)
        new_data = _load_validated_graph(args.new_graph)
    except FileNotFoundError as e:
        print(f"Error: Graph file not found: {e.filename}", file=sys.stderr)
        sys.exit(2)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: Cannot diff invalid graph input: {e}", file=sys.stderr)
        sys.exit(2)

    diff = diff_graph_data(old_data, new_data)
    _print_diff_report(diff, as_json=args.json)
    if args.fail_on_change and diff.changed:
        sys.exit(1)


def _handle_versions(args) -> None:
    from .graph.versioning import diff_versions, list_versions, rollback_to_version

    if args.versions_command == "list":
        versions = list_versions(args.graph, versions_dir=args.versions_dir)
        if args.json:
            print(json.dumps([v.to_dict() for v in versions], indent=2, ensure_ascii=False))
            return
        if not versions:
            print("No graph versions found.")
            return
        for version in versions:
            doc = f" document={version.document_id}" if version.document_id else ""
            print(
                f"{version.version_id} {version.operation} "
                f"{version.created_at}{doc} snapshot={version.snapshot_path}"
            )
        return

    if args.versions_command == "diff":
        try:
            diff = diff_versions(
                args.graph,
                args.old_version,
                args.new_version,
                versions_dir=args.versions_dir,
            )
        except KeyError as e:
            print(f"Error: Version not found: {e.args[0]}", file=sys.stderr)
            sys.exit(2)
        except FileNotFoundError as e:
            print(f"Error: Snapshot file not found: {e.filename}", file=sys.stderr)
            sys.exit(2)
        _print_diff_report(diff, as_json=args.json)
        if args.fail_on_change and diff.changed:
            sys.exit(1)
        return

    if args.versions_command == "rollback":
        try:
            rollback = rollback_to_version(
                args.graph,
                args.version,
                versions_dir=args.versions_dir,
            )
        except KeyError:
            print(f"Error: Version not found: {args.version}", file=sys.stderr)
            sys.exit(2)
        except FileNotFoundError as e:
            print(f"Error: Snapshot file not found: {e.filename}", file=sys.stderr)
            sys.exit(2)
        print(f"Rolled back {args.graph} to {args.version} ({rollback.version_id}).")
        return

    print("Error: Missing versions subcommand", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
