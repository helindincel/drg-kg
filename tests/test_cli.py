"""Smoke tests for ``drg.cli``.

The CLI is normally exercised by integration tests, but we want at least
the argparse wiring, file-not-found path, API-key routing, and the
secret-redaction logic to have unit coverage so regressions don't reach
PyPI without anyone noticing.

LLM calls are mocked, so these tests run offline.
"""

from __future__ import annotations

import json
import sys

import pytest

from drg import cli as cli_mod


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Reset env vars the CLI touches so tests are independent.

    The CLI calls ``load_dotenv(".env", override=False)`` on every invocation,
    which would otherwise re-populate API keys from a developer's local ``.env``
    file and break tests that assert *absence* of a key. We neutralise that
    side-effect here so the suite behaves identically on dev machines and CI.
    """
    for var in (
        "DRG_MODEL",
        "DRG_BASE_URL",
        "DRG_TEMPERATURE",
        "DRG_VALIDATE_HUB_DOMINANCE",
        "DRG_MAX_TOKENS",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "DRG_DEBUG",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cli_mod, "load_dotenv", lambda *a, **kw: None)


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    cli_mod.main()


class TestCreateDefaultSchema:
    def test_returns_company_to_product_schema(self):
        schema = cli_mod.create_default_schema()
        entity_names = {e.name for e in schema.entities}
        relation_names = {r.name for r in schema.relations}
        assert entity_names == {"Company", "Product"}
        assert "produces" in relation_names


class TestArgparseWiring:
    def test_help_exits_cleanly(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as excinfo:
            _run_main(monkeypatch, ["drg", "--help"])
        assert excinfo.value.code == 0
        captured = capsys.readouterr().out
        assert "DRG" in captured
        assert "--auto-schema" in captured

    def test_missing_required_input_arg_errors(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as excinfo:
            _run_main(monkeypatch, ["drg"])
        # argparse exits with code 2 when a required arg is missing.
        assert excinfo.value.code == 2
        assert "input" in capsys.readouterr().err.lower()

    def test_invalid_output_format_choice_errors(self, monkeypatch, capsys, tmp_path):
        input_file = tmp_path / "in.txt"
        input_file.write_text("Apple makes iPhones.")
        with pytest.raises(SystemExit) as excinfo:
            _run_main(
                monkeypatch,
                ["drg", str(input_file), "--output-format", "neo4j"],
            )
        assert excinfo.value.code == 2
        assert "invalid choice" in capsys.readouterr().err.lower()


class TestFileHandling:
    def test_missing_input_file_exits_with_clear_error(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as excinfo:
            _run_main(monkeypatch, ["drg", "/nonexistent/path/foo.txt"])
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        assert "not found" in err.lower()

    def test_missing_custom_schema_file_exits(self, monkeypatch, capsys, tmp_path):
        input_file = tmp_path / "in.txt"
        input_file.write_text("Apple makes iPhones.")
        with pytest.raises(SystemExit) as excinfo:
            _run_main(
                monkeypatch,
                [
                    "drg",
                    str(input_file),
                    "--schema",
                    str(tmp_path / "missing.json"),
                ],
            )
        assert excinfo.value.code == 1
        assert "error" in capsys.readouterr().err.lower()

    def test_malformed_schema_file_exits(self, monkeypatch, capsys, tmp_path):
        input_file = tmp_path / "in.txt"
        input_file.write_text("Apple makes iPhones.")
        bad_schema = tmp_path / "bad.json"
        bad_schema.write_text("{not really json")
        with pytest.raises(SystemExit) as excinfo:
            _run_main(
                monkeypatch,
                ["drg", str(input_file), "--schema", str(bad_schema)],
            )
        assert excinfo.value.code == 1
        assert "error" in capsys.readouterr().err.lower()


class TestApiKeyRouting:
    """``--api-key`` must populate the *provider-specific* env var."""

    @pytest.mark.parametrize(
        ("model", "expected_env_var"),
        [
            ("gemini/gemini-2.0-flash-exp", "GEMINI_API_KEY"),
            ("anthropic/claude-3-5-sonnet", "ANTHROPIC_API_KEY"),
            ("claude-3-opus", "ANTHROPIC_API_KEY"),
            ("openrouter/openai/gpt-4o", "OPENROUTER_API_KEY"),
            ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
        ],
    )
    def test_api_key_lands_in_provider_specific_env(
        self, monkeypatch, tmp_path, model, expected_env_var
    ):
        input_file = tmp_path / "in.txt"
        input_file.write_text("Apple makes iPhones.")

        # Stub extract to avoid actually calling an LLM.
        monkeypatch.setattr(
            cli_mod, "extract_typed", lambda text, schema: ([("Apple", "Company")], [])
        )

        _run_main(
            monkeypatch,
            [
                "drg",
                str(input_file),
                "--model",
                model,
                "--api-key",
                "secret-token-xyz",
                "-o",
                str(tmp_path / "out.json"),
            ],
        )

        import os

        assert os.environ.get(expected_env_var) == "secret-token-xyz"

    def test_gemini_also_populates_google_api_key(self, monkeypatch, tmp_path):
        input_file = tmp_path / "in.txt"
        input_file.write_text("Apple makes iPhones.")
        monkeypatch.setattr(cli_mod, "extract_typed", lambda text, schema: ([], []))
        _run_main(
            monkeypatch,
            [
                "drg",
                str(input_file),
                "--model",
                "gemini/gemini-2.0-flash-exp",
                "--api-key",
                "gemini-secret",
                "-o",
                str(tmp_path / "out.json"),
            ],
        )
        import os

        # LiteLLM commonly reads GOOGLE_API_KEY for Gemini, so both env
        # vars must be populated.
        assert os.environ["GEMINI_API_KEY"] == "gemini-secret"
        assert os.environ["GOOGLE_API_KEY"] == "gemini-secret"


class TestExtractionSuccess:
    def test_stdout_output_prints_json(self, monkeypatch, tmp_path, capsys):
        input_file = tmp_path / "in.txt"
        input_file.write_text("Apple produces iPhones.")
        monkeypatch.setattr(
            cli_mod,
            "extract_typed",
            lambda text, schema: (
                [("Apple", "Company"), ("iPhones", "Product")],
                [("Apple", "produces", "iPhones")],
            ),
        )
        _run_main(monkeypatch, ["drg", str(input_file)])
        captured = capsys.readouterr()
        payload = json.loads(captured.out.strip())
        assert "nodes" in payload
        assert any(n.get("id") == "Apple" for n in payload["nodes"])

    def test_writes_output_file_when_path_given(self, monkeypatch, tmp_path, capsys):
        input_file = tmp_path / "in.txt"
        input_file.write_text("Apple produces iPhones.")
        out_file = tmp_path / "nested" / "out.json"
        monkeypatch.setattr(
            cli_mod,
            "extract_typed",
            lambda text, schema: (
                [("Apple", "Company"), ("iPhones", "Product")],
                [("Apple", "produces", "iPhones")],
            ),
        )
        _run_main(monkeypatch, ["drg", str(input_file), "-o", str(out_file)])
        assert out_file.exists()
        payload = json.loads(out_file.read_text())
        assert "nodes" in payload
        assert "Knowledge graph written to" in capsys.readouterr().err

    def test_kg_suffix_in_output_path_triggers_enhanced_format(self, monkeypatch, tmp_path):
        input_file = tmp_path / "in.txt"
        input_file.write_text("Apple produces iPhones.")
        out_file = tmp_path / "result_kg.json"
        monkeypatch.setattr(
            cli_mod,
            "extract_typed",
            lambda text, schema: (
                [("Apple", "Company"), ("iPhones", "Product")],
                [("Apple", "produces", "iPhones")],
            ),
        )
        _run_main(monkeypatch, ["drg", str(input_file), "-o", str(out_file)])
        payload = json.loads(out_file.read_text())
        # EnhancedKG output includes nodes/edges/clusters; legacy is just nodes/edges.
        assert "nodes" in payload
        assert "edges" in payload


class TestStdinInput:
    def test_stdin_dash_argument_reads_from_stdin(self, monkeypatch, capsys):
        import io

        monkeypatch.setattr(sys, "stdin", io.StringIO("Apple makes iPhones."))
        monkeypatch.setattr(
            cli_mod,
            "extract_typed",
            lambda text, schema: ([("Apple", "Company")], []),
        )
        _run_main(monkeypatch, ["drg", "-"])
        captured = capsys.readouterr()
        payload = json.loads(captured.out.strip())
        assert any(n.get("id") == "Apple" for n in payload["nodes"])


class TestExtractionFailurePath:
    def test_extract_exception_redacts_google_key_in_error(self, monkeypatch, tmp_path, capsys):
        input_file = tmp_path / "in.txt"
        input_file.write_text("Apple makes iPhones.")

        def _boom(text, schema):
            raise RuntimeError(
                "401 unauthorized at https://api.example.com?key=AIzaSyABCDEFGHIJKLMNOPQRSTUVWX12345"
            )

        monkeypatch.setattr(cli_mod, "extract_typed", _boom)
        with pytest.raises(SystemExit) as excinfo:
            _run_main(monkeypatch, ["drg", str(input_file)])
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        # The raw API key MUST NOT appear in stderr.
        assert "AIzaSyABCDEFGHIJKLMNOPQRSTUVWX12345" not in err
        # And the redaction marker MUST be present.
        assert "REDACTED" in err

    def test_extract_exception_redacts_url_key_param(self, monkeypatch, tmp_path, capsys):
        input_file = tmp_path / "in.txt"
        input_file.write_text("Apple makes iPhones.")

        def _boom(text, schema):
            raise RuntimeError("HTTP 500: https://x/?key=supersecret123456789&other=ok")

        monkeypatch.setattr(cli_mod, "extract_typed", _boom)
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, ["drg", str(input_file)])
        err = capsys.readouterr().err
        assert "supersecret123456789" not in err
        assert "key=REDACTED" in err

    def test_debug_env_prints_traceback(self, monkeypatch, tmp_path, capsys):
        input_file = tmp_path / "in.txt"
        input_file.write_text("Apple makes iPhones.")

        def _boom(text, schema):
            raise RuntimeError("boom")

        monkeypatch.setenv("DRG_DEBUG", "1")
        monkeypatch.setattr(cli_mod, "extract_typed", _boom)
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, ["drg", str(input_file)])
        err = capsys.readouterr().err
        # Without DRG_DEBUG the traceback isn't printed; with it set, it is.
        assert "Traceback" in err


class TestWarningPath:
    def test_warns_when_no_api_key_and_cloud_model_selected(self, monkeypatch, tmp_path, capsys):
        input_file = tmp_path / "in.txt"
        input_file.write_text("Apple makes iPhones.")
        monkeypatch.setattr(cli_mod, "extract_typed", lambda text, schema: ([], []))
        _run_main(
            monkeypatch,
            [
                "drg",
                str(input_file),
                "--model",
                "openai/gpt-4o-mini",
                "-o",
                str(tmp_path / "out.json"),
            ],
        )
        err = capsys.readouterr().err
        assert "Warning" in err
        assert "API key" in err

    def test_ollama_model_does_not_warn_about_api_key(self, monkeypatch, tmp_path, capsys):
        input_file = tmp_path / "in.txt"
        input_file.write_text("Apple makes iPhones.")
        monkeypatch.setattr(cli_mod, "extract_typed", lambda text, schema: ([], []))
        _run_main(
            monkeypatch,
            [
                "drg",
                str(input_file),
                "--model",
                "ollama_chat/llama3",
                "-o",
                str(tmp_path / "out.json"),
            ],
        )
        err = capsys.readouterr().err
        assert "API key" not in err
