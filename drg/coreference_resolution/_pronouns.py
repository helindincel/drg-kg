"""Pronoun tables for coreference resolution.

Centralizes pronoun → gender/number tags so both NLP and heuristic strategies
share a single source of truth. Adding a new language means adding an entry
here, not touching strategy code.
"""

from __future__ import annotations

PronounGender = str  # "male" | "female" | "neutral" | "plural" | "ambiguous"

# English pronouns (subjective + objective + possessive)
_PRONOUNS_EN: dict[str, PronounGender] = {
    "he": "male",
    "she": "female",
    "him": "male",
    "her": "female",
    "his": "male",
    "hers": "female",
    "it": "neutral",
    "its": "neutral",
    "they": "plural",
    "them": "plural",
    "their": "plural",
    "theirs": "plural",
}

# Turkish pronouns are grammatically gender-neutral.
_PRONOUNS_TR: dict[str, PronounGender] = {
    "o": "ambiguous",
    "ona": "ambiguous",
    "onu": "ambiguous",
    "onun": "ambiguous",
    "onlar": "plural",
    "onlara": "plural",
    "onları": "plural",
    "onların": "plural",
}


def build_pronoun_map(language: str = "en") -> dict[str, PronounGender]:
    """Return the pronoun → gender map for the requested language.

    Args:
        language: Language hint. Falls back to English if unknown. Pass ``"tr"``
            to layer Turkish pronouns on top of the English set.

    Returns:
        Mapping from lowercase pronoun text to a gender/number tag.
    """
    pronouns: dict[str, PronounGender] = dict(_PRONOUNS_EN)
    lang = (language or "en").lower()
    if lang in {"tr", "turkish"}:
        pronouns.update(_PRONOUNS_TR)
    return pronouns
