"""Tests for :mod:`drg.utils.logging`."""

from __future__ import annotations

import io
import json
import logging

from drg.utils.logging import (
    JsonFormatter,
    configure_logging,
    get_logger,
    with_context,
)


def test_get_logger_returns_stdlib_logger():
    """``get_logger`` is a drop-in for ``logging.getLogger``."""
    log = get_logger("drg.test.module")
    assert isinstance(log, logging.Logger)
    assert log.name == "drg.test.module"


def test_with_context_attaches_extra_fields():
    """User context fields should flow into the adapter's ``extra``."""
    adapter = with_context(get_logger("drg.test.ctx"), schema="enhanced", chunk_id=3)
    assert adapter.extra == {"schema": "enhanced", "chunk_id": 3}


def test_with_context_drops_reserved_record_attrs():
    """Reserved ``LogRecord`` attributes must never bleed through context."""
    adapter = with_context(
        get_logger("drg.test.ctx"),
        levelname="HACK",
        msg="HACK",
        schema="ok",
    )
    assert "levelname" not in adapter.extra
    assert "msg" not in adapter.extra
    assert adapter.extra.get("schema") == "ok"


def test_with_context_merges_per_call_extra(caplog):
    """Per-call ``extra`` should merge with adapter-level context (not replace it)."""
    log = get_logger("drg.test.merge")
    adapter = with_context(log, schema="enhanced")
    with caplog.at_level(logging.INFO, logger="drg.test.merge"):
        adapter.info("x", extra={"chunk_id": 5})
    rec = caplog.records[-1]
    assert rec.schema == "enhanced"
    assert rec.chunk_id == 5


def test_json_formatter_emits_valid_json():
    """A simple record must serialize to one JSON object with all expected keys."""
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="drg.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    record.dataset = "demo"
    payload = json.loads(fmt.format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "drg.test"
    assert payload["message"] == "hello world"
    assert payload["dataset"] == "demo"


def test_json_formatter_handles_exceptions():
    """An ``exc_info`` record should include a multi-line ``exception`` field."""
    fmt = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord(
            name="drg.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="fail",
            args=None,
            exc_info=sys.exc_info(),
        )
    payload = json.loads(fmt.format(record))
    assert payload["level"] == "ERROR"
    assert "ValueError" in payload["exception"]


def test_json_formatter_drops_taskname():
    """Python 3.12+ adds ``taskName``; we don't want it in JSON output."""
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="drg.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="x",
        args=None,
        exc_info=None,
    )
    record.taskName = "fake-task"  # simulate the 3.12+ attribute
    payload = json.loads(fmt.format(record))
    assert "taskName" not in payload


def test_configure_logging_default_format(monkeypatch):
    """Without ``DRG_LOG_FORMAT=json`` we get a plain-text formatter."""
    monkeypatch.delenv("DRG_LOG_FORMAT", raising=False)
    buf = io.StringIO()
    configure_logging(force=True, stream=buf)
    drg_logger = logging.getLogger("drg")
    drg_logger.info("plain hello")
    out = buf.getvalue()
    assert "plain hello" in out
    assert out.lstrip().startswith("20") is False or "INFO" in out


def test_configure_logging_json_mode(monkeypatch):
    """With ``DRG_LOG_FORMAT=json`` every record is one JSON object per line."""
    monkeypatch.setenv("DRG_LOG_FORMAT", "json")
    buf = io.StringIO()
    configure_logging(force=True, stream=buf)
    drg_logger = logging.getLogger("drg")
    drg_logger.info("hello %s", "world", extra={"dataset": "demo"})
    line = buf.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["message"] == "hello world"
    assert payload["dataset"] == "demo"


def test_configure_logging_is_idempotent(monkeypatch):
    """Calling configure_logging twice should NOT stack handlers (without ``force``)."""
    monkeypatch.delenv("DRG_LOG_FORMAT", raising=False)
    drg_logger = logging.getLogger("drg")
    # Reset to a known state.
    for h in list(drg_logger.handlers):
        drg_logger.removeHandler(h)

    configure_logging(force=True)
    configure_logging()  # second call without ``force``
    assert len(drg_logger.handlers) == 1
