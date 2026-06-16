"""NLP-backed coreference resolution.

Wraps spaCy plus optional neural coref plug-ins (``coreferee`` or ``neuralcoref``)
behind the :class:`CoreferenceStrategy` contract.

If spaCy isn't installed (or no English model is available), :meth:`is_available`
returns ``False`` and the dispatcher falls back to :class:`HeuristicCoreferenceStrategy`.
"""

from __future__ import annotations

import logging

from ._pronouns import build_pronoun_map
from ._scoring import (
    action_based_score,
    matches_svo_pattern,
    semantic_similarity_score,
)
from ._strategy import CoreferenceStrategy, ResolverConfig

logger = logging.getLogger(__name__)


class NLPCoreferenceStrategy(CoreferenceStrategy):
    """spaCy-based strategy with optional neural coref overlay."""

    def __init__(
        self,
        config: ResolverConfig,
        use_neural_coref: bool = True,
    ):
        super().__init__(config)
        self.use_neural_coref = use_neural_coref
        self.nlp = None
        self.neural_coref: str | None = None
        self._heuristic_fallback: CoreferenceStrategy | None = None

        self._load_pipeline()

    def _load_pipeline(self) -> None:
        try:
            import spacy
        except ImportError:
            logger.warning(
                "spaCy not available. Install with: pip install spacy "
                "or pip install drg[coreference]"
            )
            self.nlp = None
            return

        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model 'en_core_web_sm' loaded for coreference resolution")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_md")
                logger.info("spaCy model 'en_core_web_md' loaded for coreference resolution")
            except OSError:
                logger.warning(
                    "spaCy English model not found. "
                    "Install with: python -m spacy download en_core_web_sm"
                )
                self.nlp = None
                return

        if not self.use_neural_coref or self.nlp is None:
            return

        # Prefer coreferee (actively maintained); fall back to neuralcoref.
        try:
            import coreferee  # noqa: F401
            import spacy as _spacy

            _spacy_version = tuple(int(x) for x in _spacy.__version__.split(".")[:2])
            if _spacy_version >= (3, 7):
                import warnings

                warnings.warn(
                    f"coreferee is not actively maintained (last release 2023) and has known "
                    f"compatibility issues with spaCy {_spacy.__version__}. "
                    "Neural coreference may silently fail or produce incorrect results. "
                    "Consider switching to a HuggingFace-based coref model "
                    "(e.g. coreferee alternative: https://github.com/explosion/spacy-experimental).",
                    UserWarning,
                    stacklevel=2,
                )
            self.nlp.add_pipe("coreferee")
            self.neural_coref = "coreferee"
            logger.info("Neural coreference (coreferee) enabled")
            return
        except (ImportError, Exception) as e_cor:
            try:
                import neuralcoref

                self.nlp.add_pipe(neuralcoref.NeuralCoref(self.nlp.vocab), name="neuralcoref")
                self.neural_coref = "neuralcoref"
                logger.info("Neural coreference (neuralcoref) enabled")
            except (ImportError, Exception) as e_neural:
                logger.warning(
                    f"Neural coreference not available "
                    f"(coreferee: {e_cor}, neuralcoref: {e_neural}). "
                    "Install with: pip install coreferee or pip install neuralcoref. "
                    "Falling back to basic spaCy coreference."
                )
                self.neural_coref = None

    def is_available(self) -> bool:
        return self.nlp is not None

    def _get_heuristic_fallback(self) -> CoreferenceStrategy:
        # Lazy import to avoid a circular reference at module-load time.
        if self._heuristic_fallback is None:
            from ._heuristic_strategy import HeuristicCoreferenceStrategy

            self._heuristic_fallback = HeuristicCoreferenceStrategy(self.config)
        return self._heuristic_fallback

    def resolve(
        self,
        text: str,
        entities: list[tuple[str, str]],
        relations: list[tuple[str, str, str]],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
        if not self.nlp:
            return entities, relations

        try:
            doc = self.nlp(text)
            entity_names = {name for name, _ in entities}
            pronoun_to_entity: dict[str, str] = {}

            self._apply_neural_coref(doc, entity_names, pronoun_to_entity)

            # Fallback to structural scoring if neural coref didn't fire.
            if not pronoun_to_entity:
                self._apply_structural_scoring(doc, entities, entity_names, pronoun_to_entity)

            if not pronoun_to_entity:
                logger.debug("Coreference resolution: No pronouns found or resolved")
                return entities, relations

            resolved_relations: list[tuple[str, str, str]] = []
            for s, r, o in relations:
                resolved_relations.append(
                    (pronoun_to_entity.get(s, s), r, pronoun_to_entity.get(o, o))
                )
            coref_type = self.neural_coref or "basic"
            logger.info(
                f"Coreference resolution ({coref_type}): {len(pronoun_to_entity)} pronouns resolved"
            )
            return entities, resolved_relations

        except Exception as e:
            logger.warning(
                f"NLP-based coreference resolution failed: {e}, falling back to heuristics"
            )
            return self._get_heuristic_fallback().resolve(text, entities, relations)

    def _apply_neural_coref(
        self,
        doc,
        entity_names: set,
        pronoun_to_entity: dict[str, str],
    ) -> None:
        """Populate ``pronoun_to_entity`` from the active neural coref backend, if any."""
        person_pronouns = {"he", "she", "it", "they", "him", "her", "them"}

        if self.neural_coref == "coreferee":
            if not hasattr(doc._, "coref_chains"):
                return
            for chain in doc._.coref_chains:
                main_mention = chain.main
                main_text = doc[main_mention[0] : main_mention[1]].text
                if main_text not in entity_names:
                    continue
                for mention in chain:
                    mention_text = doc[mention[0] : mention[1]].text
                    if mention_text.lower() in person_pronouns:
                        pronoun_to_entity[mention_text] = main_text

        elif self.neural_coref == "neuralcoref":
            if not hasattr(doc._, "coref_clusters"):
                return
            for cluster in doc._.coref_clusters:
                main_mention = cluster.main.text
                if main_mention not in entity_names:
                    continue
                for mention in cluster.mentions:
                    mention_text = mention.text
                    if mention_text.lower() in person_pronouns:
                        pronoun_to_entity[mention_text] = main_mention

    def _apply_structural_scoring(
        self,
        doc,
        entities: list[tuple[str, str]],
        entity_names: set,
        pronoun_to_entity: dict[str, str],
    ) -> None:
        """Score candidate entities via sentence structure, type, distance, and
        optional embeddings/action context.

        Mirrors the legacy fallback path; abstains when the top score is below
        ``min_resolution_score`` or too close to the runner-up.
        """
        pronouns = build_pronoun_map(self.config.language)
        entity_type_map = dict(entities)
        sentences = list(doc.sents)

        for sent_idx, sent in enumerate(sentences):
            for token in sent:
                pronoun_lower = token.text.lower()
                if pronoun_lower not in pronouns:
                    continue

                pronoun_gender = pronouns[pronoun_lower]
                pronoun_pos = token.i
                best_match: str | None = None
                best_score = 0.0
                second_best_score = 0.0

                pronoun_sentence = next((s for s in sentences if token in s), None)
                pronoun_context = pronoun_sentence.text.lower() if pronoun_sentence else ""

                # Strategy 1: scan backwards within the same window.
                for prev_token in doc[max(0, pronoun_pos - 50) : pronoun_pos]:
                    for entity_name in entity_names:
                        entity_words = entity_name.lower().split()
                        if not any(
                            ew in prev_token.text.lower() or ew in prev_token.lemma_.lower()
                            for ew in entity_words
                        ):
                            continue

                        score = 1.0
                        entity_type = entity_type_map.get(entity_name, "")
                        if pronoun_gender in {"male", "female"} and entity_type == "Person":
                            score = 1.5

                        distance = pronoun_pos - prev_token.i
                        score *= 1.0 / (1.0 + distance * 0.1)

                        if self.config.embedding_provider and pronoun_context:
                            sem = semantic_similarity_score(
                                entity_name,
                                pronoun_pos,
                                doc,
                                self.config.embedding_provider,
                            )
                            if sem > 0:
                                score *= 1.0 + sem * 0.3

                        act = action_based_score(entity_name, pronoun_context, entity_type)
                        if act > 0:
                            score *= 1.0 + act * 0.2

                        if score > best_score:
                            second_best_score = best_score
                            best_match = entity_name
                            best_score = score
                        elif score > second_best_score:
                            second_best_score = score

                # Strategy 2: previous sentences (up to 2 back).
                if not best_match or best_score < 0.8:
                    for prev_sent_idx in range(max(0, sent_idx - 2), sent_idx):
                        prev_sent = sentences[prev_sent_idx]
                        prev_sent_text = prev_sent.text.lower()
                        for entity_name in entity_names:
                            entity_words = entity_name.lower().split()
                            if not any(word in prev_sent_text for word in entity_words):
                                continue
                            score = 0.7
                            entity_type = entity_type_map.get(entity_name, "")
                            if pronoun_gender in {"male", "female"} and entity_type == "Person":
                                score = 1.0

                            if self.config.embedding_provider:
                                combined_context = f"{prev_sent_text} {pronoun_context}"
                                sem = semantic_similarity_score(
                                    entity_name,
                                    pronoun_pos,
                                    doc,
                                    self.config.embedding_provider,
                                )
                                if sem > 0:
                                    score *= 1.0 + sem * 0.25
                                # touch combined_context to keep parity with legacy
                                _ = combined_context

                            if matches_svo_pattern(entity_name, prev_sent, pronoun_context):
                                score *= 1.3

                            if score > best_score:
                                second_best_score = best_score
                                best_match = entity_name
                                best_score = score
                            elif score > second_best_score:
                                second_best_score = score

                # Strategy 3 (conservative): single-Person fallback.
                if not best_match and pronoun_gender in {"male", "female"}:
                    persons = [e for e in entity_names if entity_type_map.get(e, "") == "Person"]
                    if len(persons) == 1:
                        best_match = persons[0]
                        best_score = max(best_score, 0.90)
                        second_best_score = max(second_best_score, 0.0)

                if best_match and (
                    best_score >= self.config.min_resolution_score
                    and (
                        second_best_score <= 0
                        or (best_score - second_best_score) >= self.config.min_resolution_margin
                    )
                ):
                    pronoun_to_entity[token.text] = best_match
                elif best_match:
                    logger.debug(
                        f"Pronoun '{token.text}' resolution ambiguous/low "
                        f"(best={best_score:.2f}, second={second_best_score:.2f}), skipping"
                    )
