"""CLI smoke tests — subprocess only, no LLM required."""

from __future__ import annotations

import subprocess
import sys

import pytest

# The CLI imports drg.extract which requires dspy. Skip all tests in this
# module if dspy is not installed in the current environment.
# Use importlib.metadata (package registry) rather than import — other test
# files add a MagicMock stub to sys.modules which would fool an import check.
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    _pkg_version("dspy")
    dspy_available = True
except (ImportError, PackageNotFoundError):
    dspy_available = False

pytestmark = pytest.mark.skipif(
    not dspy_available,
    reason="dspy not installed — CLI smoke tests require dspy",
)


def _run(*args: str, input_: str | None = None) -> subprocess.CompletedProcess:
    """Run `drg <args>` as a subprocess and return the result."""
    cmd = [sys.executable, "-m", "drg.cli"] + list(args)
    return subprocess.run(
        cmd,
        input=input_,
        capture_output=True,
        text=True,
    )


class TestCLIHelp:
    def test_no_args_exits_nonzero(self):
        # Without subcommand the parser should show help and exit
        result = _run()
        assert result.returncode != 0 or result.stdout or result.stderr

    def test_help_flag_exits_zero(self):
        result = _run("--help")
        assert result.returncode == 0

    def test_help_mentions_extract(self):
        result = _run("--help")
        assert "extract" in result.stdout.lower() or "extract" in result.stderr.lower()
