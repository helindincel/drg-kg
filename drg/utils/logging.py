"""Structured logging helpers for DRG.

Goals
=====

1. **One-line opt-in**: ``logger = get_logger(__name__)`` should be a drop-in
   replacement for the stdlib ``logging.getLogger(__name__)`` pattern that's
   already used everywhere in the codebase.
2. **No surprises by default**: when no env vars are set, ``get_logger``
   returns a plain stdlib ``Logger`` and DRG's logging stays exactly as it
   is today (modules still configure their own handlers, etc.).
3. **Optional JSON output**: setting ``DRG_LOG_FORMAT=json`` and calling
   :func:`configure_logging` once at process start switches the root DRG
   logger to a JSON formatter — useful for container logs / log shippers.
4. **Structured "extra" fields**: :func:`with_context` returns a
   :class:`logging.LoggerAdapter` that attaches user-supplied context fields
   (dataset, chunk_id, schema_name, …) to every record. These fields appear
   under ``extra`` for stdlib log records and as top-level keys in the JSON
   formatter.

Non-goals
=========

- Replacing every ``logging.getLogger(__name__)`` call site. That migration
  is incremental.
- Implementing OpenTelemetry / tracing. That's a separate sprint.
- Configuring log levels. Levels still come from stdlib config /
  environment (e.g. ``LOG_LEVEL`` if your entrypoint sets it).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from typing import Any

__all__ = [
    "DEFAULT_FORMAT",
    "JsonFormatter",
    "configure_logging",
    "get_logger",
    "with_context",
]


DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"


_TRUTHY = {"1", "true", "yes", "y", "on"}

# Reserved attribute names that stdlib's ``LogRecord`` already uses. Anything
# in here can't be safely overwritten by user-supplied context.
_RESERVED_RECORD_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        # Python 3.12+ auto-adds ``taskName`` for asyncio tasks. Drop it from
        # context output unless the caller explicitly sets it.
        "taskName",
    }
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Return a stdlib :class:`logging.Logger` for ``name``.

    Intentionally identical to ``logging.getLogger(name)`` so it can replace
    the existing pattern without behaviour changes. Provided as a single
    chokepoint in case we later want to enrich every DRG logger (e.g. attach
    a default context filter).
    """
    return logging.getLogger(name)


def with_context(logger: logging.Logger, /, **context: Any) -> logging.LoggerAdapter:
    """Wrap ``logger`` in a :class:`LoggerAdapter` that attaches ``**context``
    to every emitted record.

    Example::

        log = with_context(get_logger(__name__), schema="enhanced", chunk_id=3)
        log.info("Entity extraction done")  # extra: schema=enhanced, chunk_id=3

    Reserved record attributes (``name``, ``msg``, ``levelname``, …) are
    silently dropped from ``context`` to avoid clobbering log internals.
    """
    safe_context = {k: v for k, v in context.items() if k not in _RESERVED_RECORD_ATTRS}
    return _ContextAdapter(logger, safe_context)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    fmt: str | None = None,
    stream: Any = None,
    force: bool = False,
) -> None:
    """One-shot logging configuration for DRG entrypoints (CLI, API, scripts).

    This is **optional**. Libraries shouldn't call it on import; entrypoints
    can call it once at startup. When ``DRG_LOG_FORMAT=json`` is set, output
    is emitted as one JSON object per line regardless of ``fmt``.

    Args:
        level: Log level for the root ``drg`` logger.
        fmt: Custom format string (ignored when JSON mode is active).
            Defaults to :data:`DEFAULT_FORMAT`.
        stream: Output stream. Defaults to ``sys.stderr`` (stdlib default).
        force: When True, replace existing handlers on the ``drg`` logger.
            Otherwise the function is a no-op if handlers are already
            attached (so calling it twice is safe).
    """
    drg_logger = logging.getLogger("drg")
    if drg_logger.handlers and not force:
        drg_logger.setLevel(level)
        return

    if force:
        for h in list(drg_logger.handlers):
            drg_logger.removeHandler(h)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    if _is_json_mode():
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(fmt or DEFAULT_FORMAT))

    drg_logger.addHandler(handler)
    drg_logger.setLevel(level)
    # Don't propagate; we own DRG's log path once configured.
    drg_logger.propagate = False


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Why bespoke instead of pulling ``python-json-logger``? Zero extra deps,
    and we only need a thin formatter — the heavy lifting is already done
    by the stdlib ``LogRecord``.

    Fields:
        - ``time``: ISO-8601 timestamp.
        - ``level``: log level name (INFO, WARNING, …).
        - ``logger``: dotted logger name.
        - ``message``: rendered message.
        - ``exception``: traceback when ``exc_info`` is set.
        - **extras**: any attributes added via :func:`with_context` or
          ``logger.log(..., extra=...)`` appear at the top level.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Surface any extra fields (set via LoggerAdapter or ``extra=``).
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_ATTRS:
                continue
            if key.startswith("_"):
                continue
            payload.setdefault(key, value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        try:
            return json.dumps(payload, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            # Last-resort fallback so logging never crashes the host process.
            return json.dumps({"level": record.levelname, "message": record.getMessage()})


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _is_json_mode() -> bool:
    return os.getenv("DRG_LOG_FORMAT", "").strip().lower() == "json"


class _ContextAdapter(logging.LoggerAdapter):
    """LoggerAdapter that merges adapter ``extra`` into per-call ``extra``.

    The stdlib default *replaces* per-call ``extra`` with the adapter's
    ``extra``; we merge them so callers can pass per-call additions.
    """

    def process(self, msg: Any, kwargs: Mapping[str, Any]) -> tuple[Any, dict]:
        merged = dict(self.extra or {})
        per_call = kwargs.get("extra") or {}
        merged.update(per_call)
        new_kwargs = dict(kwargs)
        new_kwargs["extra"] = merged
        return msg, new_kwargs
