"""Heuristic coreference resolution (no NLP dependency).

A pure-regex / Python fallback used when spaCy isn't installed. Behaviour is
copied verbatim from the legacy ``CoreferenceResolver._resolve_with_heuristics``
and ``_find_entity_for_pronoun`` so existing tests still pass; only the seams
have been cleaned up so it now plugs into the ``CoreferenceStrategy`` contract.
"""

from __future__ import annotations

import logging
import re

from ._pronouns import build_pronoun_map
from ._strategy import CoreferenceStrategy

logger = logging.getLogger(__name__)


class HeuristicCoreferenceStrategy(CoreferenceStrategy):
    """Sentence-structure + entity-type matching with conservative gating."""

    def is_available(self) -> bool:
        return True

    def resolve(
        self,
        text: str,
        entities: list[tuple[str, str]],
        relations: list[tuple[str, str, str]],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
        if not entities or not relations:
            return entities, relations

        sentences = re.split(r"[.!?]+\s+", text)
        entity_names = {name for name, _ in entities}
        entity_type_map = dict(entities)
        pronouns = build_pronoun_map(self.config.language)
        pronoun_to_entity: dict[str, str] = {}

        for s, _r, o in relations:
            if s.lower() in pronouns:
                match = self._find_entity_for_pronoun(
                    s, text, sentences, entity_names, entity_type_map, pronouns[s.lower()]
                )
                if match:
                    pronoun_to_entity[s] = match
            if o.lower() in pronouns:
                match = self._find_entity_for_pronoun(
                    o, text, sentences, entity_names, entity_type_map, pronouns[o.lower()]
                )
                if match:
                    pronoun_to_entity[o] = match

        if not pronoun_to_entity:
            logger.debug("Coreference resolution (heuristics): No pronouns resolved")
            return entities, relations

        resolved_relations: list[tuple[str, str, str]] = []
        for s, r, o in relations:
            resolved_relations.append((pronoun_to_entity.get(s, s), r, pronoun_to_entity.get(o, o)))
        logger.info(
            f"Coreference resolution (improved heuristics): {len(pronoun_to_entity)} pronouns resolved"
        )
        return entities, resolved_relations

    def _find_entity_for_pronoun(
        self,
        pronoun: str,
        text: str,
        sentences: list[str],
        entity_names: set[str],
        entity_type_map: dict[str, str],
        pronoun_gender: str,
    ) -> str | None:
        """Score candidate entities for a single pronoun occurrence.

        Strategy ordering: same-sentence > previous sentences > single-Person fallback.
        Returns ``None`` when the best candidate is below the confidence floor or
        too close to the runner-up (conservative abstention).
        """
        pronoun_lower = pronoun.lower()
        # Whole-word match — prevents 'o' from matching inside 'Elon'.
        m = re.search(rf"(?i)(?<!\w){re.escape(pronoun_lower)}(?!\w)", text)
        if not m:
            return None
        pronoun_index = m.start()

        def _sent_bounds_around(idx: int) -> tuple[int, int]:
            left = max(
                text.rfind(".", 0, idx),
                text.rfind("?", 0, idx),
                text.rfind("!", 0, idx),
            )
            start = 0 if left == -1 else left + 1
            right_candidates = [
                p
                for p in (
                    text.find(".", idx),
                    text.find("?", idx),
                    text.find("!", idx),
                )
                if p != -1
            ]
            end = (min(right_candidates) + 1) if right_candidates else len(text)
            return start, end

        cur_start, cur_end = _sent_bounds_around(pronoun_index)
        current_sent = text[cur_start:cur_end].strip().lower()

        prev_sents: list[str] = []
        prev_end = cur_start - 1
        for _ in range(2):
            if prev_end <= 0:
                break
            left = max(
                text.rfind(".", 0, prev_end),
                text.rfind("?", 0, prev_end),
                text.rfind("!", 0, prev_end),
            )
            start = 0 if left == -1 else left + 1
            prev_sents.append(text[start : prev_end + 1].strip().lower())
            prev_end = start - 2

        best_match: str | None = None
        best_score = 0.0
        second_best_score = 0.0

        if current_sent:
            sent_tokens = re.findall(r"\b\w+\b", current_sent)
            for entity_name in entity_names:
                entity_words = entity_name.lower().split()
                entity_mentions = [
                    i
                    for i, word in enumerate(sent_tokens)
                    if any(ew in word for ew in entity_words)
                ]
                pronoun_pos_in_sent = (
                    sent_tokens.index(pronoun_lower) if pronoun_lower in sent_tokens else -1
                )

                if entity_mentions and pronoun_pos_in_sent > 0:
                    before_pronoun = [pos for pos in entity_mentions if pos < pronoun_pos_in_sent]
                    if before_pronoun:
                        score = 1.0
                        distance = pronoun_pos_in_sent - max(before_pronoun)
                        score *= 1.0 / (1.0 + distance * 0.2)
                        entity_type = entity_type_map.get(entity_name, "")
                        if pronoun_gender in {"male", "female"} and entity_type == "Person":
                            score *= 1.5
                        elif pronoun_gender == "neutral" and entity_type != "Person":
                            score *= 1.5

                        if score > best_score:
                            second_best_score = best_score
                            best_match = entity_name
                            best_score = score
                        elif score > second_best_score:
                            second_best_score = score

        if not best_match or best_score < 0.7:
            for prev_sent in prev_sents:
                if not prev_sent:
                    continue
                # Abstain when multiple Persons exist in the same window.
                if pronoun_gender in {"male", "female", "ambiguous"}:
                    person_hits = [
                        e
                        for e in entity_names
                        if entity_type_map.get(e, "") == "Person" and e.lower() in prev_sent
                    ]
                    if len(person_hits) >= 2:
                        continue

                positions: list[tuple[int, str]] = []
                for entity_name in entity_names:
                    pos = prev_sent.find(entity_name.lower())
                    if pos != -1:
                        positions.append((pos, entity_name))
                positions.sort(key=lambda x: x[0])
                if not positions:
                    continue

                for rank, (_, entity_name) in enumerate(positions):
                    score = 0.9 if rank == 0 else 0.7
                    entity_type = entity_type_map.get(entity_name, "")
                    if pronoun_gender in {"male", "female"} and entity_type == "Person":
                        score *= 1.1
                    if pronoun_gender == "neutral" and entity_type and entity_type != "Person":
                        score *= 1.1
                    if score > best_score:
                        second_best_score = best_score
                        best_match = entity_name
                        best_score = score
                    elif score > second_best_score:
                        second_best_score = score

        if not best_match and pronoun_gender in {"male", "female"}:
            persons = [e for e in entity_names if entity_type_map.get(e, "") == "Person"]
            if len(persons) == 1:
                best_match = persons[0]
                best_score = max(best_score, 0.90)

        if best_match and (
            best_score >= self.config.min_resolution_score
            and (
                second_best_score <= 0
                or (best_score - second_best_score) >= self.config.min_resolution_margin
            )
        ):
            return best_match
        return None
