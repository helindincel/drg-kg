"""Relationship modelling package.

Public surface (kept stable for backward compatibility):

- :class:`RelationshipType` — taxonomy enum.
- :data:`RELATIONSHIP_CATEGORIES` — semantic groupings of the taxonomy.
- :class:`EnrichedRelationship` — dataclass for a fully-described relationship.
- :func:`create_enriched_relationship` — factory.
- :class:`RelationshipTypeClassifier` — rules + LLM dispatcher.

Internals (importable for advanced use / tests):

- ``_rule_based.classify_rule_based`` / ``apply_schema_constraints``
- ``_llm_based.is_available`` / ``create_relationship_classifier`` /
  ``classify_llm_based`` / ``build_classification_input``

Architecture
============

::

    drg/graph/relationship_model/
        __init__.py        # Public API + legacy module-level helpers
        _types.py          # RelationshipType + RELATIONSHIP_CATEGORIES
        _enriched.py       # EnrichedRelationship + create_enriched_relationship
        _rule_based.py     # keyword + schema-constraint classification
        _llm_based.py      # DSPy-conditional LLM classifier
        _classifier.py     # RelationshipTypeClassifier (dispatcher)
"""

from __future__ import annotations

from ._classifier import RelationshipTypeClassifier
from ._enriched import EnrichedRelationship, create_enriched_relationship
from ._llm_based import (
    build_classification_input as _build_classification_input,
)
from ._llm_based import (
    create_relationship_classifier as _create_relationship_classifier,
)
from ._llm_based import (
    is_available as _llm_is_available,
)
from ._types import RELATIONSHIP_CATEGORIES, RelationshipType

# DSPy availability — re-exported under the legacy module-level constant name
# so older code reading ``relationship_model.DSPY_AVAILABLE`` keeps working.
DSPY_AVAILABLE = _llm_is_available()

# Legacy module-level helpers some call sites used directly.
create_relationship_classifier = _create_relationship_classifier
build_classification_input = _build_classification_input

__all__ = [
    # Legacy module-level helpers (kept for back-compat)
    "DSPY_AVAILABLE",
    "RELATIONSHIP_CATEGORIES",
    "EnrichedRelationship",
    # Public surface
    "RelationshipType",
    "RelationshipTypeClassifier",
    "build_classification_input",
    "create_enriched_relationship",
    "create_relationship_classifier",
]
