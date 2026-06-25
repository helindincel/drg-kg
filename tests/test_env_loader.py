"""Tests for drg/utils/env_loader.py."""

from __future__ import annotations

import os
import textwrap

import pytest


@pytest.fixture()
def env_file(tmp_path):
    """Helper: write a .env file to a temp dir and return its path."""

    def _write(content: str):
        p = tmp_path / ".env"
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p

    return _write


def test_load_dotenv_basic(env_file, monkeypatch):
    """Happy path: KEY=VALUE lines are loaded into os.environ."""
    from drg.utils.env_loader import load_dotenv

    monkeypatch.delenv("_DRG_TEST_KEY", raising=False)
    p = env_file("_DRG_TEST_KEY=hello\n")
    result = load_dotenv(p)
    assert result is True
    assert os.environ["_DRG_TEST_KEY"] == "hello"


def test_load_dotenv_double_quotes(env_file, monkeypatch):
    """Double-quoted values have their quotes stripped."""
    from drg.utils.env_loader import load_dotenv

    monkeypatch.delenv("_DRG_TEST_QUOTED", raising=False)
    p = env_file('_DRG_TEST_QUOTED="world"\n')
    load_dotenv(p)
    assert os.environ["_DRG_TEST_QUOTED"] == "world"


def test_load_dotenv_single_quotes(env_file, monkeypatch):
    """Single-quoted values have their quotes stripped."""
    from drg.utils.env_loader import load_dotenv

    monkeypatch.delenv("_DRG_TEST_SQ", raising=False)
    p = env_file("_DRG_TEST_SQ='value'\n")
    load_dotenv(p)
    assert os.environ["_DRG_TEST_SQ"] == "value"


def test_load_dotenv_ignores_comments(env_file, monkeypatch):
    """Comment lines (# ...) are ignored."""
    from drg.utils.env_loader import load_dotenv

    monkeypatch.delenv("_DRG_COMMENT_KEY", raising=False)
    p = env_file("# this is a comment\n_DRG_COMMENT_KEY=yes\n")
    load_dotenv(p)
    assert os.environ["_DRG_COMMENT_KEY"] == "yes"


def test_load_dotenv_ignores_blank_lines(env_file, monkeypatch):
    """Blank lines do not cause errors."""
    from drg.utils.env_loader import load_dotenv

    monkeypatch.delenv("_DRG_BLANK_KEY", raising=False)
    p = env_file("\n\n_DRG_BLANK_KEY=present\n\n")
    load_dotenv(p)
    assert os.environ["_DRG_BLANK_KEY"] == "present"


def test_load_dotenv_override_false(env_file, monkeypatch):
    """With override=False, existing env vars are preserved."""
    from drg.utils.env_loader import load_dotenv

    monkeypatch.setenv("_DRG_NO_OVERRIDE", "original")
    p = env_file("_DRG_NO_OVERRIDE=new_value\n")
    load_dotenv(p, override=False)
    assert os.environ["_DRG_NO_OVERRIDE"] == "original"


def test_load_dotenv_override_true(env_file, monkeypatch):
    """With override=True, existing env vars ARE overwritten."""
    from drg.utils.env_loader import load_dotenv

    monkeypatch.setenv("_DRG_OVERRIDE", "original")
    p = env_file("_DRG_OVERRIDE=updated\n")
    load_dotenv(p, override=True)
    assert os.environ["_DRG_OVERRIDE"] == "updated"


def test_load_dotenv_missing_file(tmp_path):
    """Returns False when the .env file does not exist."""
    from drg.utils.env_loader import load_dotenv

    result = load_dotenv(tmp_path / "nonexistent.env")
    assert result is False


def test_load_dotenv_value_with_equals(env_file, monkeypatch):
    """Values that contain '=' are preserved correctly (only first = splits)."""
    from drg.utils.env_loader import load_dotenv

    monkeypatch.delenv("_DRG_EQ_VALUE", raising=False)
    p = env_file("_DRG_EQ_VALUE=a=b=c\n")
    load_dotenv(p)
    assert os.environ["_DRG_EQ_VALUE"] == "a=b=c"


def test_load_dotenv_no_value_key_ignored(env_file, monkeypatch):
    """Lines without '=' are silently skipped."""
    from drg.utils.env_loader import load_dotenv

    monkeypatch.delenv("_DRG_NOEQ", raising=False)
    p = env_file("NOEQUALSSIGN\n_DRG_NOEQ=valid\n")
    load_dotenv(p)
    assert os.environ.get("_DRG_NOEQ") == "valid"
    assert "NOEQUALSSIGN" not in os.environ
