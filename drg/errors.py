"""Typed exception hierarchy for DRG.

Why a custom hierarchy?
=======================

Most of DRG currently raises ``RuntimeError`` / ``ValueError`` directly. That
works, but it forces callers to either catch broad built-ins (and risk
swallowing unrelated errors) or to inspect ``str(exc)`` to figure out what
went wrong. A small typed hierarchy fixes both problems:

- API consumers can ``except DRGError`` once at the boundary.
- Specific subclasses (``SchemaError``, ``ExtractionError``, …) let internal
  code branch on cause without string-matching.
- Each class subclasses an appropriate built-in (``ValueError`` for invalid
  input, ``RuntimeError`` for runtime failures) so existing ``except``
  clauses in user code keep working — this is a strictly additive change.

Strict mode interaction
=======================

Many internal call sites guard non-fatal failures with
``drg.utils.strict.is_strict``: in default mode they log a warning and
continue, in strict mode they re-raise. The exceptions defined here are the
canonical types that should bubble out of those strict re-raises.

Design notes
============

- Keep the hierarchy *shallow*. Two or three levels max; deeper trees are
  rarely useful and tempt people to over-classify.
- Every leaf exception has a one-line docstring that says *when* it's raised
  in the codebase. If you can't describe the trigger in a sentence, the leaf
  probably isn't the right granularity.
- We deliberately do NOT translate every ``raise ValueError(...)`` site at
  once — this module is the foundation; rolling the new types into call
  sites is an incremental follow-up done in the relevant modules.
"""

from __future__ import annotations

__all__ = [
    "ClusteringError",
    "ConfigError",
    "CoreferenceResolutionError",
    "DRGError",
    "EmbeddingError",
    "EntityResolutionError",
    "ExtractionError",
    "GraphError",
    "LLMConfigError",
    "ResolutionError",
    "SchemaError",
    "SchemaGenerationError",
]


class DRGError(Exception):
    """Base class for every DRG-specific error.

    Catching this at the outer boundary of an application gives a clean,
    library-wide bailout point without swallowing unrelated built-ins.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigError(DRGError, ValueError):
    """Invalid or missing configuration (env var, file, parameter)."""


class LLMConfigError(ConfigError):
    """The DSPy LM couldn't be configured (missing key, unknown model, etc.).

    Typically raised from ``drg.config.configure_lm`` or surfaced when an
    extraction call requires an LM but ``DRG_REQUIRE_LM`` is set and none is
    available.
    """


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class SchemaError(DRGError, ValueError):
    """User-provided schema is malformed or violates DRG invariants."""


class SchemaGenerationError(SchemaError, RuntimeError):
    """Auto-schema generation (``generate_schema_from_text``) failed.

    Distinct from :class:`SchemaError` because the *output* of the generator
    is at fault, not the user's input. Inherits from both so callers can
    catch either ``SchemaError`` or ``RuntimeError`` and still get this.
    """


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class ExtractionError(DRGError, RuntimeError):
    """Entity/relation extraction failed for reasons beyond bad input.

    Examples: TypedPredictor returned an unparseable shape, the LM raised
    mid-call, JSON fallback parsing failed.
    """


# ---------------------------------------------------------------------------
# Resolution (entity + coreference)
# ---------------------------------------------------------------------------


class ResolutionError(DRGError, RuntimeError):
    """Base for entity/coreference resolution failures."""


class EntityResolutionError(ResolutionError):
    """An :class:`~drg.entity_resolution.EntityResolver` step failed.

    Raised when strict mode is on and a hard failure occurs (e.g. embedding
    provider crash, dimensionality mismatch). The default non-strict path
    logs and continues with string similarity only.
    """


class CoreferenceResolutionError(ResolutionError):
    """A coreference strategy raised in strict mode.

    Heuristic strategies never raise; this typically surfaces an NLP backend
    failure when both spaCy and the neural coref plug-in are installed but
    misbehave.
    """


# ---------------------------------------------------------------------------
# Adjacent subsystems
# ---------------------------------------------------------------------------


class EmbeddingError(DRGError, RuntimeError):
    """An embedding provider couldn't fulfil an ``embed`` / ``embed_batch`` call."""


class ClusteringError(DRGError, RuntimeError):
    """A clustering algorithm failed on the given graph."""


class GraphError(DRGError, RuntimeError):
    """A KG operation (edge insertion, traversal, export) failed."""
