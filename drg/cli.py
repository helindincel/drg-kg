#!/usr/bin/env python3
"""CLI interface for DRG - Declarative Relationship Generation"""

import argparse
import json
import re
import sys
from pathlib import Path

from .chunking import create_chunker
from .extract import extract_from_chunks, extract_typed, generate_schema_from_text
from .graph import KG
from .graph.builders import build_enhanced_kg
from .schema import DRGSchema, Entity, Relation, load_schema_from_json
from .utils.env_loader import load_dotenv


def create_default_schema():
    """Varsayılan şema: Company -> Product"""
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
        epilog="""
Examples:
  drg extract input.txt -o output.json
  drg extract input.txt -o output.json --schema custom_schema.json
  echo "Apple released iPhone 16" | drg extract - -o output.json
        """,
    )

    parser.add_argument("input", type=str, help="Input text file or '-' for stdin")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="-",
        help="Output JSON file (default: stdout, or specify path like 'outputs/output.json')",
    )

    parser.add_argument(
        "--schema",
        type=str,
        help="Custom schema JSON file (optional, uses default Company->Product if not provided)",
    )

    parser.add_argument(
        "--auto-schema",
        action="store_true",
        help="Generate an EnhancedDRGSchema from the input text (recommended for richer, input-agnostic extraction). "
        "If provided, --schema is ignored.",
    )

    parser.add_argument(
        "--output-format",
        type=str,
        default=None,
        choices=["legacy", "enhancedkg"],
        help="Output format. 'legacy' matches CLI JSON (nodes/edges with edge key 'type'). "
        "'enhancedkg' writes EnhancedKG JSON (nodes/edges/clusters) for the UI. "
        "If omitted, inferred from output filename: '*_kg.json' -> enhancedkg else legacy.",
    )

    parser.add_argument(
        "--no-hub-validation",
        action="store_true",
        help="Disable hub-dominance validation gate (some documents are naturally hub-like).",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model identifier. Examples: 'openai/gpt-4o-mini' (cloud, needs API key), 'ollama_chat/llama3' (local, no API key). Default: from DRG_MODEL env or 'openai/gpt-4o-mini'",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for LLM (required for cloud models, not needed for local models like Ollama)",
    )

    parser.add_argument("--base-url", type=str, default=None, help="Custom API base URL (optional)")

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM generation (default: 0.0)",
    )

    args = parser.parse_args()

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
        schema = None  # lazy generate after model/env is set
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
            import traceback

            traceback.print_exc()
            sys.exit(1)
    else:
        schema = create_default_schema()

    # Set environment variables for automatic LLM configuration (DSPy otomatik okur)
    import os

    if args.no_hub_validation:
        os.environ["DRG_VALIDATE_HUB_DOMINANCE"] = "0"
    if args.model:
        os.environ["DRG_MODEL"] = args.model
    if args.api_key:
        # Set appropriate API key env var based on model
        model = args.model or os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
        if "gemini" in model.lower():
            # Different SDK/adapters use different env var names for Gemini.
            # Keep both to be robust (LiteLLM commonly reads GOOGLE_API_KEY).
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

    # Warn if using cloud model without API key
    api_key = (
        args.api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
    )
    model = args.model or os.getenv("DRG_MODEL", "openai/gpt-4o-mini")
    if not api_key and not model.startswith("ollama"):
        print("Warning: No API key found. Cloud models require an API key.", file=sys.stderr)
        print(
            "For local models, use: --model ollama_chat/llama3 --base-url http://localhost:11434",
            file=sys.stderr,
        )

    # Determine output format
    inferred_format = None
    if args.output != "-" and args.output.lower().endswith("_kg.json"):
        inferred_format = "enhancedkg"
    output_format = args.output_format or inferred_format or "legacy"

    # Extract
    try:
        # Generate schema (after env is set so LLM config can use the chosen model)
        if args.auto_schema:
            # Auto-schema typically needs a larger output budget to avoid truncation.
            # Keep this opt-in behind --auto-schema so default CLI stays cheap/safe.
            if not os.getenv("DRG_MAX_TOKENS"):
                os.environ["DRG_MAX_TOKENS"] = "4096"
            schema = generate_schema_from_text(text)

        # If schema is auto-generated (Enhanced) we use chunk-aware extraction for richer relations.
        if args.auto_schema:
            chunk_size = int(os.getenv("DRG_CHUNK_SIZE", "768"))
            overlap_ratio = float(os.getenv("DRG_OVERLAP_RATIO", "0.15"))
            strategy = os.getenv("DRG_CHUNKING_STRATEGY", "token_based")
            chunker = create_chunker(
                strategy=strategy, chunk_size=chunk_size, overlap_ratio=overlap_ratio
            )
            chunks = chunker.chunk(text, origin_dataset="cli", origin_file=args.input)
            entities_typed, triples = extract_from_chunks(
                chunks=[
                    {"text": c.text, "chunk_id": c.chunk_id, "metadata": c.metadata} for c in chunks
                ],
                schema=schema,
                enable_cross_chunk_relationships=True,
                enable_entity_resolution=True,
                enable_coreference_resolution=True,
                two_pass_extraction=True,
            )
        else:
            entities_typed, triples = extract_typed(text, schema)

        # Remove duplicates
        triples = list(dict.fromkeys(triples))

        if output_format == "enhancedkg":
            kg2 = build_enhanced_kg(
                entities_typed=entities_typed,
                triples=triples,
                schema=schema,
                source_text=text,
            )
            output_json = kg2.to_json()
        else:
            kg = KG.from_typed(entities_typed, triples)
            output_json = kg.to_json()
    except Exception as e:
        # Avoid leaking secrets (API keys can appear in URLs like ...?key=... in provider errors).
        import traceback

        raw_msg = f"{type(e).__name__}: {e}"
        raw_tb = traceback.format_exc()

        def _redact_secrets(s: str) -> str:
            if not s:
                return s
            # Redact URL query keys: key=XXXX
            s = re.sub(r"(?i)(key=)[^&\s]+", r"\1REDACTED", s)
            # Redact common Google API key shape if present
            s = re.sub(r"AIzaSy[0-9A-Za-z_-]{20,}", "REDACTED_GOOGLE_API_KEY", s)
            # Redact OpenRouter key shape if present
            s = re.sub(r"sk-or-v1-[0-9a-fA-F]{20,}", "REDACTED_OPENROUTER_KEY", s)
            return s

        print(f"Error during extraction: {_redact_secrets(raw_msg)}", file=sys.stderr)
        # Print full traceback only when explicitly requested
        import os

        if os.getenv("DRG_DEBUG", "").lower() in {"1", "true", "yes"}:
            print(_redact_secrets(raw_tb), file=sys.stderr)
        sys.exit(1)

    # Write output
    if args.output == "-":
        print(output_json)
    else:
        output_path = Path(args.output)
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json, encoding="utf-8")
        print(f"Knowledge graph written to: {output_path}", file=sys.stderr)

        # If schema was auto-generated, also persist it next to the output for UI/debugging.
        if args.auto_schema:
            try:
                schema_stem = output_path.stem.replace("_kg", "")
                schema_path = output_path.parent / f"{schema_stem}_schema.json"
                schema_path.write_text(
                    json.dumps(schema.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"Schema written to: {schema_path}", file=sys.stderr)
            except Exception:
                # Best-effort: schema saving should not fail the main extraction path.
                pass


if __name__ == "__main__":
    main()
