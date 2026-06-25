"""LLM-based relationship classification (DSPy).

Optional path: if DSPy + pydantic aren't installable (sandboxed/headless
environments, etc.), :func:`is_available` returns ``False`` and the classifier
silently falls back to rule-based scoring only.

The DSPy signature is constructed lazily so that importing this module never
runs an LLM round-trip, and so that callers without DSPy installed can still
import the package.
"""

from __future__ import annotations

from typing import Any

from ...utils.logging import get_logger
from ._types import RELATIONSHIP_CATEGORIES, RelationshipType

logger = get_logger(__name__)

__all__ = [
    "build_classification_input",
    "classify_llm_based",
    "create_relationship_classifier",
    "is_available",
]


# Try to import DSPy. Catch *any* exception, not just ImportError — some
# sandboxed environments raise PermissionError when reading site-packages.
try:
    import dspy
    from pydantic import BaseModel

    DSPY_AVAILABLE = True
except Exception as e:
    DSPY_AVAILABLE = False
    dspy = None  # type: ignore[assignment]
    BaseModel = None  # type: ignore[assignment]
    logger.debug(
        "DSPy import unavailable; LLM-based relationship classification disabled: %s",
        e,
    )


def is_available() -> bool:
    """True when DSPy + pydantic are importable and LLM-based classification can run."""
    return DSPY_AVAILABLE


# ---------------------------------------------------------------------------
# DSPy-only definitions
# ---------------------------------------------------------------------------


if DSPY_AVAILABLE:  # pragma: no cover - exercised in environments with DSPy

    class RelationshipClassificationItem(BaseModel):
        """Single relationship classification result."""

        relationship_type: str
        confidence: float

    class RelationshipClassification(BaseModel):
        """Structured output for relationship classification."""

        classifications: list[RelationshipClassificationItem]

    def create_relationship_classifier():
        """Build the DSPy ``TypedPredictor`` for relationship classification.

        Constructed lazily by :class:`~drg.graph.relationship_model.RelationshipTypeClassifier`
        so we only pay the cost the first time the LLM path is exercised.
        """
        rel_types = [rt.value for rt in RelationshipType]
        rel_types_str = ", ".join(sorted(rel_types))

        categories_info = []
        for category, types in RELATIONSHIP_CATEGORIES.items():
            types_str = ", ".join([rt.value for rt in types])
            categories_info.append(f"{category}: {types_str}")
        categories_str = "\n".join(categories_info)

        class RelationshipClassificationSignature(dspy.Signature):
            """Classify the relationship type between two entities.

            Given the source entity, target entity, their types, and context,
            classify the most appropriate relationship type(s) from the taxonomy.

            Return the top 3 most likely relationship types with confidence scores.
            """

            source: str = dspy.InputField(desc="Source entity name")
            target: str = dspy.InputField(desc="Target entity name")
            source_type: str | None = dspy.InputField(
                desc="Source entity type (e.g., 'Person', 'Location')", default=None
            )
            target_type: str | None = dspy.InputField(
                desc="Target entity type (e.g., 'Person', 'Location')", default=None
            )
            raw_relation_text: str | None = dspy.InputField(
                desc="Raw text describing the relationship", default=None
            )
            context: str | None = dspy.InputField(
                desc="Contextual text where the relationship appears", default=None
            )

        # Python class docstrings are literals — format strings are NOT interpolated.
        # Set __doc__ dynamically so the taxonomy is actually visible to the LLM.
        RelationshipClassificationSignature.__doc__ = (
            "Classify the relationship type between two entities.\n\n"
            "Given the source entity, target entity, their types, and context, "
            "classify the most appropriate relationship type(s) from the taxonomy.\n\n"
            f"Available relationship types: {rel_types_str}\n\n"
            f"Categories:\n{categories_str}\n\n"
            "Return the top 3 most likely relationship types with confidence scores."
        )

        try:
            return dspy.Predict(RelationshipClassificationSignature)
        except Exception as e:
            logger.warning(f"Failed to create predictor: {e}")
            return None

    def build_classification_input(
        source: str,
        target: str,
        source_type: str | None = None,
        target_type: str | None = None,
        raw_relation_text: str | None = None,
        context: str | None = None,
    ) -> dict[str, Any]:
        return {
            "source": source,
            "target": target,
            "source_type": source_type,
            "target_type": target_type,
            "raw_relation_text": raw_relation_text,
            "context": context,
        }

else:  # DSPy unavailable — provide stubs that match the public API.
    RelationshipClassificationItem = None  # type: ignore[assignment]
    RelationshipClassification = None  # type: ignore[assignment]

    def create_relationship_classifier():
        """Stub: DSPy not available, returns ``None``."""
        return None

    def build_classification_input(*args, **kwargs) -> dict[str, Any]:
        """Stub: DSPy not available, returns empty input dict."""
        return {}


# ---------------------------------------------------------------------------
# Classifier entry point
# ---------------------------------------------------------------------------


def classify_llm_based(
    classifier,
    *,
    source: str,
    target: str,
    source_type: str | None = None,
    target_type: str | None = None,
    raw_relation_text: str | None = None,
    context: str | None = None,
) -> list[tuple[RelationshipType, float]]:
    """Run the LLM-backed classifier and return ``(RelationshipType, confidence)`` tuples.

    Never raises — degrades to an empty list when DSPy is unavailable or the
    LLM output can't be parsed. The dispatcher merges these results with the
    rule-based ones.
    """
    if not DSPY_AVAILABLE or classifier is None:
        return []

    try:
        inputs = build_classification_input(
            source=source,
            target=target,
            source_type=source_type,
            target_type=target_type,
            raw_relation_text=raw_relation_text,
            context=context,
        )
        result = classifier(**inputs)

        if RelationshipClassification is not None and isinstance(
            result, RelationshipClassification
        ):
            classifications = result.classifications
        else:
            classifications = getattr(result, "classifications", [])
            if not isinstance(classifications, list):
                logger.warning(f"Expected RelationshipClassification, got {type(result).__name__}")
                return []

        results: list[tuple[RelationshipType, float]] = []
        for item in classifications:
            if isinstance(item, dict):
                rel_type_str = item.get("relationship_type", "")
                confidence = item.get("confidence", 0.5)
            elif hasattr(item, "relationship_type") and hasattr(item, "confidence"):
                rel_type_str = item.relationship_type
                confidence = item.confidence
            else:
                continue

            try:
                rel_type = RelationshipType(rel_type_str.lower())
                results.append((rel_type, float(confidence)))
            except (ValueError, AttributeError):
                logger.debug(f"Invalid relationship type: {rel_type_str}")
                continue
        return results
    except Exception as e:
        logger.warning(f"LLM-based classification failed: {e}, returning empty results")
        return []
