#!/usr/bin/env python3
"""CLI interface for DRG - Declarative Relationship Generation"""

import argparse
import json
import re
import sys
import os
from pathlib import Path

from .chunking import create_chunker
from .extract import extract_from_chunks, extract_typed, generate_schema_from_text
from .graph import KG
from .graph.builders import build_enhanced_kg
from .schema import DRGSchema, Entity, Relation, load_schema_from_json
from .utils.env_loader import load_dotenv


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
    )
    
    subparsers = parser.add_subparsers(dest="command")

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
    )
    extract_parser.add_argument(
        "--update-strategy",
        type=str,
        default="prefer_existing",
        choices=["prefer_existing", "prefer_new", "union"],
    )
    extract_parser.add_argument(
        "--update-document-id",
        type=str,
        default=None,
    )
    extract_parser.add_argument(
        "--infer",
        action="store_true",
    )
    extract_parser.add_argument(
        "--extract-events",
        action="store_true",
    )
    extract_parser.add_argument(
        "--events-registry",
        type=str,
        default=None,
    )
    extract_parser.add_argument(
        "--events-use-example",
        action="store_true",
    )
    extract_parser.add_argument(
        "--infer-min-confidence",
        type=float,
        default=0.5,
    )
    extract_parser.add_argument(
        "--infer-disable-rule",
        action="append",
        default=[],
    )
    extract_parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    extract_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
    )
    extract_parser.add_argument("--base-url", type=str, default=None)
    extract_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    # --- Eval command ---
    eval_parser = subparsers.add_parser("eval", help="Evaluation framework")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command")

    # eval run
    eval_run_parser = eval_subparsers.add_parser("run", help="Run benchmark")
    eval_run_parser.add_argument("dataset", type=str, help="Dataset JSON file or directory")
    eval_run_parser.add_argument("-o", "--output", type=str, help="Output report file")
    eval_run_parser.add_argument("--run-id", type=str, help="Run ID")
    eval_run_parser.add_argument("--retrieval-k", type=int, default=10)
    eval_run_parser.add_argument("--model", type=str, help="Model (overrides env)")
    eval_run_parser.add_argument("--api-key", type=str, help="API Key (overrides env)")

    # eval compare
    eval_comp_parser = eval_subparsers.add_parser("compare", help="Compare reports")
    eval_comp_parser.add_argument("baseline", type=str, help="Baseline JSON")
    eval_comp_parser.add_argument("candidate", type=str, help="Candidate JSON")
    eval_comp_parser.add_argument("-o", "--output", type=str, help="Output markdown")
    eval_comp_parser.add_argument("--threshold", type=float, default=0.01)

    # Legacy support: if no command, default to extract
    if len(sys.argv) > 1 and sys.argv[1] not in ["extract", "eval", "-h", "--help"]:
        sys.argv.insert(1, "extract")

    args = parser.parse_args()

    if args.command == "extract":
        _handle_extract(args)
    elif args.command == "eval":
        _handle_eval(args)
    else:
        parser.print_help()


def _handle_extract(args):
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
        schema = load_schema_from_json(args.schema)
    else:
        schema = create_default_schema()

    import os
    if args.no_hub_validation:
        os.environ["DRG_VALIDATE_HUB_DOMINANCE"] = "0"
    if args.model:
        os.environ["DRG_MODEL"] = args.model
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["DRG_BASE_URL"] = args.base_url
    if args.temperature != 0.0:
        os.environ["DRG_TEMPERATURE"] = str(args.temperature)

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
            entities_typed, triples = extract_typed(text, schema)

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
                document_id=effective_doc_id,
                events=extracted_events or None,
            )

            if args.update:
                from .graph import EnhancedKG, GraphMerger, MergeStrategy, NodeMergePolicy
                update_path = Path(args.update)
                base_kg = EnhancedKG.load_json(str(update_path)) if update_path.exists() else EnhancedKG()
                strategy = MergeStrategy(node_policy=NodeMergePolicy(args.update_strategy))
                GraphMerger(strategy).merge(base_kg, target_kg, document_id=effective_doc_id)
                target_kg = base_kg

            if args.infer:
                from .reasoning import MultiDocumentReasoner, ReasoningConfig
                infer_cfg = ReasoningConfig(
                    min_confidence=args.infer_min_confidence,
                    disabled_rules=frozenset(args.infer_disable_rule or []),
                )
                MultiDocumentReasoner(config=infer_cfg).reason(target_kg, document_id=effective_doc_id)

            output_json = target_kg.to_json()
        else:
            kg = KG.from_typed(entities_typed, triples)
            output_json = kg.to_json()

        if args.output == "-":
            print(output_json)
        else:
            args.output = args.update if args.update and args.output == "-" else args.output
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_json, encoding="utf-8")
            print(f"Knowledge graph written to: {output_path}", file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _handle_eval(args):
    from .evaluation import (
        BenchmarkRunner,
        PipelinePrediction,
        load_benchmark_datasets,
        save_json_report,
        save_markdown_report,
        render_markdown_report,
        compare_reports,
        render_regression_markdown,
    )
    
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.model:
        os.environ["DRG_MODEL"] = args.model

    if args.eval_command == "run":
        datasets = load_benchmark_datasets(args.dataset)
        runner = BenchmarkRunner(
            run_id=args.run_id,
            retrieval_k=args.retrieval_k,
            metadata={"model": os.getenv("DRG_MODEL")},
        )
        
        def extraction_runner(ds):
            # Real extraction runner for evaluation
            e, t = extract_typed(ds.text, create_default_schema())
            return PipelinePrediction(entities=e, relations=t)

        report = runner.evaluate(datasets, runner=extraction_runner)
        
        if args.output:
            if args.output.endswith(".md"):
                save_markdown_report(report, args.output)
            else:
                save_json_report(report, args.output)
        else:
            print(render_markdown_report(report))

    elif args.eval_command == "compare":
        with open(args.baseline) as f:
            base = json.load(f)
        with open(args.candidate) as f:
            cand = json.load(f)
        
        from .evaluation._types import EvaluationReport
        comparison = compare_reports(
            EvaluationReport.from_dict(base),
            EvaluationReport.from_dict(cand),
            regression_threshold=args.threshold,
        )
        output = render_regression_markdown(comparison)
        if args.output:
            Path(args.output).write_text(output)
        else:
            print(output)


if __name__ == "__main__":
    main()
