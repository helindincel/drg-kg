"""Entity resolver — merges duplicate entity references.

Implementation lives here so the public ``__init__.py`` is small and
unambiguous. Behaviour is preserved from the legacy 610-line monolith;
only the seams changed:

- Similarity computation now goes through a :class:`SimilarityStrategy`
  (DI). The resolver no longer constructs an embedding provider on its
  own — the package's ``EntityResolver`` factory does that and hands a
  ready strategy down.
- Adaptive thresholding stays as a method here because it's resolver-level
  policy, not similarity policy.
"""

from __future__ import annotations

from ..utils.logging import get_logger
from ._normalize import normalize_entity_name
from ._strategy import SimilarityStrategy, StringSimilarity

logger = get_logger(__name__)


class EntityResolver:
    """Merge duplicate entity mentions while abstaining on ambiguous matches.

    The resolver is **conservative by design**: when two candidate canonical
    names tie within ``min_merge_margin`` it refuses to merge rather than
    guess. Coreference resolution (``drg.coreference_resolution``) is a
    separate concern and should run *before* entity resolution.

    Construction
    ------------
    Most callers use :func:`drg.entity_resolution.resolve_entities_and_relations`
    or instantiate ``EntityResolver`` via the package-level constructor —
    both of which wire up an appropriate :class:`SimilarityStrategy`
    automatically. You can also pass a custom strategy directly for full
    control (testing, custom backends).
    """

    def __init__(
        self,
        *,
        similarity_strategy: SimilarityStrategy,
        similarity_threshold: float = 0.65,
        adaptive_threshold: bool = True,
        use_normalization: bool = True,
        min_merge_margin: float = 0.08,
    ):
        self.similarity_strategy = similarity_strategy
        self.base_similarity_threshold = similarity_threshold
        # Mirrored for backward-compat introspection ("legacy field").
        self.similarity_threshold = similarity_threshold
        self.adaptive_threshold = adaptive_threshold
        self.use_normalization = use_normalization
        self.min_merge_margin = min_merge_margin

    # ------------------------------------------------------------------
    # Backward-compat introspection: legacy code reads these directly.
    # ------------------------------------------------------------------

    @property
    def embedding_provider(self):
        return getattr(self.similarity_strategy, "embedding_provider", None)

    @property
    def use_embedding(self) -> bool:
        return self.similarity_strategy.uses_embeddings

    @property
    def embedding_weight(self) -> float:
        return getattr(self.similarity_strategy, "embedding_weight", 0.0)

    # ------------------------------------------------------------------
    # Thresholding
    # ------------------------------------------------------------------

    def _get_adaptive_threshold(self, name1: str, name2: str) -> float:
        """Length- and substring-aware threshold.

        Shorter names get a more lenient threshold so we don't miss
        ``"Elena"`` ↔ ``"Dr. Elena Vasquez"``. Substring matches get the
        most lenient treatment because :func:`similarity_score` already
        boosts those.
        """
        if not self.adaptive_threshold:
            return self.base_similarity_threshold

        min_len = min(len(name1), len(name2))
        max_len = max(len(name1), len(name2))

        n1 = normalize_entity_name(name1) if self.use_normalization else name1.lower().strip()
        n2 = normalize_entity_name(name2) if self.use_normalization else name2.lower().strip()

        if (n1 in n2 or n2 in n1) and min_len >= 3:
            return 0.30

        if min_len < 5 and max_len > 10:
            return max(0.40, self.base_similarity_threshold - 0.25)
        if min_len < 8:
            return max(0.50, self.base_similarity_threshold - 0.15)
        if min_len < 10:
            return max(0.55, self.base_similarity_threshold - 0.10)

        return self.base_similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self, entities: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], dict[str, str]]:
        """Group duplicates within each entity type.

        Returns ``(resolved_entities, name_mapping)`` where ``name_mapping``
        sends every original mention to its canonical form (chosen as the
        longest variant in a group — usually the most complete name).
        """
        if not entities:
            return [], {}

        entities_by_type: dict[str, list[tuple[str, str]]] = {}
        for name, etype in entities:
            entities_by_type.setdefault(etype, []).append((name, etype))

        all_resolved: list[tuple[str, str]] = []
        name_mapping: dict[str, str] = {}
        for type_entities in entities_by_type.values():
            resolved, mapping = self._resolve_by_type(type_entities)
            all_resolved.extend(resolved)
            name_mapping.update(mapping)

        logger.info(
            f"Entity resolution: {len(entities)} entities -> {len(all_resolved)} unique entities "
            f"({len(entities) - len(all_resolved)} duplicates resolved)"
        )
        return all_resolved, name_mapping

    def resolve_relations(
        self,
        relations: list[tuple[str, str, str]],
        name_mapping: dict[str, str],
    ) -> list[tuple[str, str, str]]:
        """Rewrite relation endpoints to their canonical names and drop
        self-loops / duplicates introduced by the merge."""
        resolved: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for s, r, o in relations:
            canonical_s = name_mapping.get(s, s)
            canonical_o = name_mapping.get(o, o)
            if canonical_s == canonical_o:
                logger.debug(f"Skipping self-relation: {canonical_s} --{r}--> {canonical_s}")
                continue
            triple = (canonical_s, r, canonical_o)
            if triple in seen:
                continue
            seen.add(triple)
            resolved.append(triple)

        logger.info(
            f"Relation resolution: {len(relations)} relations -> {len(resolved)} unique relations "
            f"({len(relations) - len(resolved)} duplicates/self-relations removed)"
        )
        return resolved

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_by_type(
        self, entities: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], dict[str, str]]:
        """Merge logic for a single entity type.

        The pre-pass detects ambiguous single-token aliases ("elena" hits
        two multi-token Persons) and refuses to auto-merge them — a
        deliberate abstention that keeps the resolver safe across
        arbitrary inputs.
        """
        if len(entities) <= 1:
            return entities, {}

        normalized_names = [
            (
                name,
                normalize_entity_name(name) if self.use_normalization else name.lower().strip(),
            )
            for name, _ in entities
        ]
        multi_token_norms = [n for _, n in normalized_names if len(n.split()) >= 2]

        ambiguous_singletons: set[str] = set()
        for _, norm in normalized_names:
            if len(norm.split()) != 1:
                continue
            hits = 0
            for long_norm in multi_token_norms:
                if norm in set(long_norm.split()):
                    hits += 1
                    if hits >= 2:
                        ambiguous_singletons.add(norm)
                        break

        def _safe_to_merge(n1: str, n2: str, et: str) -> bool:
            """Conservative merge gating to keep entity resolution safe across arbitrary inputs."""
            a = normalize_entity_name(n1) if self.use_normalization else n1.lower().strip()
            b = normalize_entity_name(n2) if self.use_normalization else n2.lower().strip()
            if a == b:
                return True
            ta = a.split()
            tb = b.split()
            # Never merge two different single-token names.
            if len(ta) == 1 and len(tb) == 1:
                return False
            # Never auto-merge ambiguous single-token aliases into multi-token names.
            if et.lower() == "person":
                if (
                    len(ta) == 1
                    and len(tb) >= 2
                    and ta[0] in set(tb)
                    and ta[0] in ambiguous_singletons
                ):
                    return False
                if (
                    len(tb) == 1
                    and len(ta) >= 2
                    and tb[0] in set(ta)
                    and tb[0] in ambiguous_singletons
                ):
                    return False
            # Person names: require last-name agreement when both are multi-token.
            if et.lower() == "person" and len(ta) >= 2 and len(tb) >= 2:
                return ta[-1] == tb[-1]
            return True

        canonical_groups: dict[str, list[str]] = {}
        name_mapping: dict[str, str] = {}
        processed: set[str] = set()

        for i, (name, etype) in enumerate(entities):
            if name in processed:
                continue

            best: str | None = None
            best_sim = -1.0
            second_sim = -1.0
            for canonical in canonical_groups:
                if not _safe_to_merge(name, canonical, etype):
                    continue
                sim = self.similarity_strategy.score(name, canonical)
                thr = self._get_adaptive_threshold(name, canonical)
                if sim >= thr:
                    if sim > best_sim:
                        second_sim = best_sim
                        best_sim = sim
                        best = canonical
                    elif sim > second_sim:
                        second_sim = sim

            matched_canonical: str | None = None
            if best is not None and (
                second_sim < 0 or (best_sim - second_sim) >= self.min_merge_margin
            ):
                matched_canonical = best

            if matched_canonical:
                canonical_groups[matched_canonical].append(name)
                name_mapping[name] = matched_canonical
                processed.add(name)
                continue

            # New canonical group; choose the longest variant as canonical.
            similar_names: list[str] = [name]
            for other_name, _ in entities[i + 1 :]:
                if other_name in processed:
                    continue
                if not _safe_to_merge(name, other_name, etype):
                    continue
                sim = self.similarity_strategy.score(name, other_name)
                thr = self._get_adaptive_threshold(name, other_name)
                if sim >= thr:
                    similar_names.append(other_name)
                    processed.add(other_name)

            canonical = max(similar_names, key=len)
            canonical_groups[canonical] = similar_names
            for variant in similar_names:
                name_mapping[variant] = canonical
                processed.add(variant)

        etype = entities[0][1]  # all entities share a type at this layer
        resolved = [(canonical, etype) for canonical in canonical_groups]
        return resolved, name_mapping


__all__ = ["EntityResolver", "StringSimilarity"]
