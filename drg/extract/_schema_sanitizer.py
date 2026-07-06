"""Deterministic schema sanitizer for LLM-generated EnhancedDRGSchema objects.

``SchemaSanitizer`` is a pure post-processing layer that sits between the LLM
output and the rest of the pipeline.  It never calls an LLM; it enforces
structural rules (always correct regardless of domain) *and* a lightweight set
of semantic role-compatibility rules that catch the most common LLM errors.

Passes (run in order):
    0a. Canonical relation names        — renames synonyms (developed/created/
                                          produces → develops) and strips endpoint
                                          types from names (organization_monitors_
                                          person → monitors)
    0b. Canonical entity types          — renames/merges synonyms and artifact
                                          sub-types (Company → Organization;
                                          OperatingSystem/Hardware → Product)
    0c. Merge relation groups           — collapses fragmented families
                                          (four product groups → Product & Technology)
    1.  Primitive entity types          — removes Date, Year, Amount, …
    2.  Source/target type validity     — removes relations with undefined endpoints
    3.  Duplicate relation names        — dedup by (name, src, dst); one endpoint-
                                          free name may span several endpoint pairs
    4.  Duplicate semantic relations    — keeps first by (base_name, src, dst)
    5.  Duplicate relation groups       — keeps first by group name
    9.  Semantic type compatibility     — removes relations whose (src_role, dst_role)
                                          is incoherent for the relation's semantic family
                                          (e.g. valuation/criticism relations must target
                                          ACTORs, not ARTIFACTs or PLACEs; temporal
                                          relations must not target OCCURRENCEs; market
                                          expansion relations must not target ARTIFACTs)
    6.  Empty relation groups           — removes groups with no surviving relations
    7.  Orphan entity types             — removes entity types used in no relation
    8.  Example consistency             — removes examples that reference missing relations
    10. Relation genericity pruning     — removes relations that match patterns indicating
                                          they are document-specific one-off actions rather
                                          than reusable schema elements

The numbering of passes 6-8 is kept stable for backwards-compat; Pass 9 is
inserted between Pass 5 and Pass 6 so that groups emptied by it are cleaned up
automatically.

Usage::

    from drg.extract._schema_sanitizer import SchemaSanitizer

    sanitizer = SchemaSanitizer()
    clean_schema, report = sanitizer.sanitize(schema)
    print(report.summary())
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from ..schema import EnhancedDRGSchema, EntityType, Relation, RelationGroup

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    """Return a boolean environment flag.

    Unset → *default*.  Set to any of ``0/false/no/off`` (case-insensitive) or
    the empty string → ``False``.  Any other value → ``True``.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Entity type names that always represent literal scalar values and must
#: never appear as standalone entity nodes in a knowledge graph schema.
PRIMITIVE_ENTITY_TYPE_NAMES: frozenset[str] = frozenset(
    {
        # temporal literals
        "date",
        "year",
        "month",
        "day",
        "time",
        "timestamp",
        "datetime",
        "duration",
        "period",
        "interval",
        # numeric / monetary literals
        "amount",
        "price",
        "cost",
        "value",
        "number",
        "percentage",
        "percent",
        "quantity",
        "count",
        "version",
        # type-system primitives
        "string",
        "boolean",
        "integer",
        "float",
    }
)

#: Common noun suffixes appended by the coverage pass when it produces
#: semantic variants of existing relations (e.g. ``developed_product``
#: vs ``developed``).  Used for duplicate semantic detection.
_NOUN_SUFFIXES: tuple[str, ...] = (
    "_product",
    "_products",
    "_company",
    "_companies",
    "_person",
    "_people",
    "_os",
    "_software",
    "_hardware",
    "_service",
    "_event",
    "_location",
    "_organization",
    "_institution",
)

#: Directional/preposition suffixes that typically signal an inverse relation
#: (e.g. ``developed_by`` is the inverse of ``developed``).
#: Stripping these before deduplication allows Pass 4 to detect inverse
#: duplicates: if ``developed: Company→Product`` is already in the schema,
#: ``developed_by: Product→Company`` (same base name, swapped endpoints)
#: is redundant and will be dropped.
_DIRECTIONAL_SUFFIXES: tuple[str, ...] = (
    "_by",
    "_for",
    "_from",
    "_into",
    "_at",
    "_as",
)


# ---------------------------------------------------------------------------
# Canonical vocabulary  (Passes 0a / 0b / 0c)
# ---------------------------------------------------------------------------
#
# These maps push the schema toward a single canonical concept per
# entity/relation/group so that the ontology stays reusable across thousands of
# documents.  They run *before* the deduplication passes, so once two synonyms
# collapse to the same canonical name the existing dedup/orphan passes remove
# the redundancy automatically.
#
# Keys are always *normalized* names (see :meth:`SchemaSanitizer._normalize`):
# lower-cased, hyphens/spaces → underscores.

#: Synonymous relation names → one canonical relation name.
#: Only exact (whole-name) matches are canonicalized, so the false-positive risk
#: is very low.  Canonical verbs follow ``SCHEMA_GENERATION_PRINCIPLES`` and are
#: endpoint-free (``founded_by``, ``acquired``, ``partnered_with``, ``develops``).
_RELATION_NAME_CANONICAL: dict[str, str] = {
    # ── develops family (creation / production / manufacture of an artifact) ──
    "develop": "develops",
    "developed": "develops",
    "developing": "develops",
    "develops_product": "develops",
    "develops_products": "develops",
    "developed_product": "develops",
    "developed_products": "develops",
    "create": "develops",
    "creates": "develops",
    "created": "develops",
    "produce": "develops",
    "produces": "develops",
    "produced": "develops",
    "makes": "develops",
    "manufactures": "develops",
    "manufactured": "develops",
    "builds": "develops",
    "built": "develops",
    # ── partnership family ────────────────────────────────────────────────
    "formed_agreement_with": "partnered_with",
    "made_agreement_with": "partnered_with",
    "signed_agreement_with": "partnered_with",
    "entered_agreement_with": "partnered_with",
    "entered_into_agreement_with": "partnered_with",
    "formed_partnership_with": "partnered_with",
    "partnered": "partnered_with",
    "partners_with": "partnered_with",
    "collaborated_with": "partnered_with",
    # ── headquarters / location family ────────────────────────────────────
    "has_headquarters_in": "headquartered_in",
    "headquartered_at": "headquartered_in",
    "has_hq_in": "headquartered_in",
    # ── employment family ─────────────────────────────────────────────────
    "has_employee": "employs",
    "has_employees": "employs",
    "employ": "employs",
    "employed": "employs",
    "hires": "employs",
    "hired": "employs",
}

#: Synonymous / narrower entity-type names → one canonical actor type.
#: Always applied when name-canonicalization is enabled (very safe: these are
#: all clearly the same "collective body" concept).
_ENTITY_TYPE_CANONICAL: dict[str, str] = {
    "company": "Organization",
    "companies": "Organization",
    "corporation": "Organization",
    "corp": "Organization",
    "firm": "Organization",
    "business": "Organization",
    "enterprise": "Organization",
    "institution": "Organization",
    "organisation": "Organization",
}

#: Artifact sub-types → the canonical ``Product`` super-type.
#: Applied only when ``collapse_artifact_subtypes`` is enabled.  This removes the
#: "same instance classified under two types" problem (e.g. ``Windows`` showing
#: up as both ``Product`` and ``OperatingSystem``); the concrete kind belongs on
#: a ``category`` property, not on a separate entity type.
_ARTIFACT_SUBTYPE_CANONICAL: dict[str, str] = {
    "operatingsystem": "Product",
    "operating_system": "Product",
    "os": "Product",
    "softwaresuite": "Product",
    "software_suite": "Product",
    "software": "Product",
    "softwareapplication": "Product",
    "hardware": "Product",
    "application": "Product",
    "app": "Product",
    "platform": "Product",
    "tool": "Product",
    "framework": "Product",
    "library": "Product",
    "device": "Product",
    "service": "Product",
}

#: Fragmented relation-group names → one canonical group family.
#: Applied only when ``merge_relation_groups`` is enabled.  Keeps the ontology
#: modular instead of scattering product relations across four near-identical
#: groups.
_RELATION_GROUP_CANONICAL: dict[str, str] = {
    "product_and_technology_lifecycle": "Product & Technology",
    "product_and_technology": "Product & Technology",
    "products_and_technology": "Product & Technology",
    "product_lifecycle": "Product & Technology",
    "product_association": "Product & Technology",
    "product_development": "Product & Technology",
    "product_usage": "Product & Technology",
}

#: Tokens that denote an *entity role* rather than a semantic interaction.  When
#: one of these appears as the leading or trailing segment of a relation name,
#: it encodes an endpoint type into the name (anti-pattern, see principle #8).
#: :meth:`SchemaSanitizer._endpoint_free_name` strips one such leading and one
#: trailing token so ``organization_monitors_person`` becomes ``monitors``.
_ENDPOINT_ROLE_TOKENS: frozenset[str] = frozenset(
    {
        "organization", "organisation", "org", "company", "companies",
        "corporation", "corp", "firm", "business", "institution",
        "person", "people", "individual", "user", "human", "employee",
        "founder", "team",
        "product", "products", "software", "hardware", "operatingsystem",
        "os", "application", "app", "platform", "device",
        "technology", "technologies",
        "concept", "idea", "field", "topic", "discipline",
        "location", "place", "region", "country", "city",
        "event", "milestone", "incident",
        "government", "governmentbody", "agency", "authority",
        "fundinground", "legalcase", "regulation",
    }
)

#: Grammatical connectors that must never be left as the whole relation name
#: after stripping role tokens (prevents ``contributed_to`` collapsing to ``to``).
_CONNECTOR_TOKENS: frozenset[str] = frozenset(
    {"to", "of", "with", "in", "by", "for", "from", "at", "as", "on", "a", "an", "the"}
)

#: Light / auxiliary verbs that carry no interaction meaning on their own.  A
#: strip that would leave only these (e.g. ``has_employee`` → ``has``) is skipped
#: so the relation keeps a meaningful name.
_LIGHT_VERB_TOKENS: frozenset[str] = frozenset(
    {"has", "have", "had", "is", "are", "was", "were", "be", "been", "being", "get", "got", "gets"}
)


# ---------------------------------------------------------------------------
# Semantic role classification  (Pass 9)
# ---------------------------------------------------------------------------


class SemanticRole:
    """Broad semantic roles assigned to entity types by name heuristics.

    Roles are intentionally coarse-grained so that they remain stable across
    domains.  They are derived purely from the entity type's *name* — no LLM
    call is made.

    ACTOR      — autonomous agents that can decide, own things, be responsible
                 (Company, Person, Organization, Government, …)
    ARTIFACT   — created or manufactured objects, including software / hardware
                 (Product, Hardware, OperatingSystem, SoftwareSuite, Service, …)
    PLACE      — geographical or physical locations
                 (Location, Region, Country, City, …)
    OCCURRENCE — events, transactions, processes, states
                 (Event, Transaction, Merger, Acquisition, …)
    UNKNOWN    — fallback when no keyword matches
    """

    ACTOR = "ACTOR"
    ARTIFACT = "ARTIFACT"
    PLACE = "PLACE"
    OCCURRENCE = "OCCURRENCE"
    UNKNOWN = "UNKNOWN"


# Keyword substrings → role.  Listed in priority order; first match wins.
# Keywords are matched case-insensitively against the *full* entity type name.
_ROLE_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    # ACTOR — anything that can act, own, decide, be blamed
    (
        (
            "company", "corporation", "organization", "organisation",
            "institution", "government", "agency", "authority", "body",
            "ministry", "department", "firm", "entity", "party",
            "person", "individual", "human", "employee", "staff",
            "user", "founder", "ceo", "executive", "leader",
            "director", "manager", "officer", "team", "group", "committee",
        ),
        SemanticRole.ACTOR,
    ),
    # ARTIFACT — created / manufactured objects
    (
        (
            "product", "hardware", "software", "operatingsystem",
            "softwaresuite", "device", "tool", "system", "platform",
            "application", "service", "framework",
            "library", "module", "component", "infrastructure",
            "feature", "interface", "protocol", "format",
            "standard", "specification", "document", "report",
            "publication", "asset", "resource",
        ),
        SemanticRole.ARTIFACT,
    ),
    # PLACE — geographical / physical locations
    (
        (
            "location", "place", "region", "country", "city", "state",
            "area", "territory", "site", "address", "venue", "building",
            "campus", "headquarters", "office", "facility",
        ),
        SemanticRole.PLACE,
    ),
    # OCCURRENCE — events, processes, transactions
    (
        (
            "event", "incident", "transaction", "process", "action",
            "activity", "occurrence", "episode", "milestone", "meeting",
            "conference", "acquisition", "merger", "deal", "agreement",
            "phase", "stage", "period", "lawsuit", "trial",
        ),
        SemanticRole.OCCURRENCE,
    ),
]


def _classify_entity_role(entity_type_name: str) -> str:
    """Return the broad :class:`SemanticRole` for *entity_type_name*.

    Matching is done by substring search on the lower-cased name.  The first
    matching keyword group wins (ACTOR checked first, then ARTIFACT, PLACE,
    OCCURRENCE).  Returns ``SemanticRole.UNKNOWN`` when no keyword matches.
    """
    lower = entity_type_name.lower()
    for keywords, role in _ROLE_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return role
    return SemanticRole.UNKNOWN


# ---------------------------------------------------------------------------
# Semantic rules  (Pass 9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SemanticRule:
    """A declarative constraint on (source_role, target_role) for relation names.

    Attributes:
        pattern:            Compiled regex matched against the relation *name*
                            (re.search, case-insensitive).
        valid_source_roles: Set of acceptable source roles.  ``None`` → no constraint.
        valid_target_roles: Set of acceptable target roles.  ``None`` → no constraint.
        reason:             Human-readable explanation emitted in the log.
    """

    pattern: re.Pattern  # type: ignore[type-arg]
    valid_source_roles: frozenset[str] | None
    valid_target_roles: frozenset[str] | None
    reason: str


#: Rules checked in order; first matching rule is authoritative.
#: A rule "matches" when ``pattern.search(relation_name)`` is truthy.
#: Roles ACTOR and UNKNOWN are both accepted wherever ACTOR is required, so
#: that entity types the classifier cannot categorise do not generate false
#: positives.
_ACTOR_OR_UNKNOWN = frozenset({SemanticRole.ACTOR, SemanticRole.UNKNOWN})
_PLACE_OR_UNKNOWN = frozenset({SemanticRole.PLACE, SemanticRole.UNKNOWN})

_SEMANTIC_RULES: list[_SemanticRule] = [
    # ── Legal / regulatory ────────────────────────────────────────────────
    # Both sides must be actors.  A company cannot be "examined" or "fined"
    # in relation to a product — the *target* of the legal action is another
    # actor (regulator, court, government).
    _SemanticRule(
        pattern=re.compile(
            r"(legal_action|lawsuit|fine|penalt|sanction|scrutin|"
            r"investigat|examin|audit|prosecut|charg|regulat|"
            r"antitrust|compliance)",
            re.I,
        ),
        valid_source_roles=_ACTOR_OR_UNKNOWN,
        valid_target_roles=_ACTOR_OR_UNKNOWN,
        reason=(
            "Legal/regulatory relations must connect actors "
            "(Company, Person, Organization) — not artifacts or places."
        ),
    ),
    # ── Valuation / financial state ───────────────────────────────────────
    # A valuation relation like 'valued_at_over' or 'worth_over' expresses a
    # financial state of an *actor*.  The amount itself must be a property;
    # if a product appears as the target, the modelling is wrong.
    _SemanticRule(
        pattern=re.compile(
            r"(valued_at|valued_over|valued_above|value_over|value_above|"
            r"worth_over|worth_above|market_cap|valuat)",
            re.I,
        ),
        valid_source_roles=_ACTOR_OR_UNKNOWN,
        valid_target_roles=_ACTOR_OR_UNKNOWN,
        reason=(
            "Valuation relations must connect actors — monetary values are "
            "properties (not entity endpoints), so the target must also be an actor."
        ),
    ),
    # ── Criticism / blame / public reproach ───────────────────────────────
    # Criticism is directed at an *agent* (actor), never at a product or place.
    _SemanticRule(
        pattern=re.compile(
            r"(criticiz|criticis|reproach|blam|accus|condemn|censur|fault)",
            re.I,
        ),
        valid_source_roles=_ACTOR_OR_UNKNOWN,
        valid_target_roles=_ACTOR_OR_UNKNOWN,
        reason=(
            "Criticism/blame relations must target actors (Company, Person) — "
            "criticism is directed at agents, not at products or places."
        ),
    ),
    # ── Geographical location ─────────────────────────────────────────────
    # 'headquartered_in', 'located_in', 'based_in', 'incorporated_in', …
    # Source must be an actor; target must be a place.
    _SemanticRule(
        pattern=re.compile(
            r"(headquarter|located_in|based_in|registered_in|incorporated_in|"
            r"operates_in|presence_in|founded_in(?!_year))",
            re.I,
        ),
        valid_source_roles=_ACTOR_OR_UNKNOWN,
        valid_target_roles=_PLACE_OR_UNKNOWN,
        reason=(
            "Location relations must run from an actor to a place — "
            "the target must be a Location/Region/Country, not an artifact or actor."
        ),
    ),
    # ── Employment / membership ───────────────────────────────────────────
    # A person is employed *by* a company, not by a product.
    _SemanticRule(
        pattern=re.compile(
            r"(employ|works?_for|hired_by|staff_of|member_of_org|reports_to)",
            re.I,
        ),
        valid_source_roles=_ACTOR_OR_UNKNOWN,
        valid_target_roles=_ACTOR_OR_UNKNOWN,
        reason=(
            "Employment relations must connect actors — "
            "a person cannot be employed by an artifact."
        ),
    ),
    # ── Date hack: temporal relations must not target OCCURRENCE ──────────────
    # When Date entity is removed, the LLM sometimes reuses Event as a
    # surrogate for dates (e.g. ``founded_on: Company → Event``).  A temporal
    # relation's target must encode a point-in-time *as a property*, never as
    # an Event/Occurrence node.
    _SemanticRule(
        pattern=re.compile(
            r"(_on$|_when$|founded_on|started_on|launched_on|"
            r"established_on|ipo_on|went_public|listed_on|incorporated_on)",
            re.I,
        ),
        valid_source_roles=None,  # no constraint on source
        valid_target_roles=frozenset(
            {SemanticRole.ACTOR, SemanticRole.ARTIFACT, SemanticRole.PLACE, SemanticRole.UNKNOWN}
        ),
        reason=(
            "Temporal (date-encoding) relations must encode dates as properties — "
            "using an Event/Occurrence node as the target is a 'Date hack'; "
            "move the date to a property on the source entity instead."
        ),
    ),
    # ── Market / domain expansion: must not target ARTIFACT ──────────────────
    # 'expanded_into', 'entered_market', 'entered_domain', … describe a company
    # moving into a new market or sector.  The target should be a market
    # concept or location, *not* a Product, Hardware, or Software entity.
    _SemanticRule(
        pattern=re.compile(
            r"(expan.*into|expand_to|enter.*market|enter.*domain|"
            r"enter.*sector|enter.*industry|pivot.*to|moved_into|"
            r"diversif.*into)",
            re.I,
        ),
        valid_source_roles=_ACTOR_OR_UNKNOWN,
        valid_target_roles=frozenset({SemanticRole.PLACE, SemanticRole.UNKNOWN}),
        reason=(
            "Market/domain expansion relations target markets or industry sectors — "
            "the target must not be an artifact (Product, Software, Hardware, …); "
            "consider removing the relation or replacing the target with a "
            "Market/Industry entity type."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Relation genericity rules  (Pass 10)
# ---------------------------------------------------------------------------

#: Each entry is a ``(pattern, reason)`` pair.
#: Any relation whose normalised name matches one of these patterns is removed
#: by Pass 10 as too document-specific to be a reusable schema relation.
#:
#: Design principle — only add a pattern when ALL of the following hold:
#:   (a) the verb or phrase almost never generalises beyond the source document;
#:   (b) the interaction can be expressed by an existing, more generic relation;
#:   (c) the pattern is narrow enough that no valid generic relation name
#:       would accidentally match it.
_DOCUMENT_SPECIFIC_PATTERNS: list[tuple[re.Pattern, str]] = [  # type: ignore[type-arg]
    # ── Naming / suggestion ───────────────────────────────────────────────────
    # "suggested_name", "proposed_name", … encode a one-off historical event.
    (
        re.compile(r"suggest.*name|propos.*name|^named_by$", re.I),
        "Naming-suggestion relations encode a single one-off event; "
        "the name itself should be a property on the entity.",
    ),
    # ── Finance millionaire / wealth-creation ─────────────────────────────────
    # "created_millionaires", "made_millionaires", … are document-specific.
    (
        re.compile(r"creat.*millionaire|made.*millionaire|generat.*millionaire", re.I),
        "Millionaire-creation relations describe a single historical outcome; "
        "use a financial event with a property instead.",
    ),
    # ── Specific legal filing actions ─────────────────────────────────────────
    # "filed_motion_against", "filed_injunction_against", … are too narrow.
    # Generic legal interactions are covered by 'faced_legal_action_from' etc.
    (
        re.compile(r"filed.*(motion|injunct|complaint)|motion_against|injunction_against", re.I),
        "Specific legal-filing relations are too narrow; "
        "use 'faced_legal_action_from', 'sued', or a similar generic relation.",
    ),
    # ── Corporate division / subsidiary formation ─────────────────────────────
    # "formed_division", "created_division", … encode a one-off restructuring.
    (
        re.compile(r"^(formed|created|established)_division$", re.I),
        "Division-formation relations encode a single restructuring event; "
        "model the division as a Company entity and use 'acquired' or 'is_part_of'.",
    ),
    # ── Software extension / derivation (too granular) ───────────────────────
    # "is_extension_for" is inferred from a single sentence and is subsumed by
    # "is_based_on", "is_version_of", or "developed".
    (
        re.compile(r"^is_extension_(for|of)$", re.I),
        "'is_extension_for/of' is too granular; "
        "use 'is_based_on', 'is_version_of', or 'developed' instead.",
    ),
    # ── Bare 'marketed' ──────────────────────────────────────────────────────
    # "marketed" as a standalone relation duplicates "released", "developed",
    # or "produces" and rarely appears as a distinct interaction across documents.
    (
        re.compile(r"^marketed$", re.I),
        "'marketed' duplicates 'released' or 'produces'; "
        "remove it or merge into a more general production/release relation.",
    ),
    # ── Superlative / ranking claims ─────────────────────────────────────────
    # "one_of_most_valuable_*", "largest_by_*", "most_profitable_*" describe
    # a snapshot ranking from the source document, not a reusable interaction.
    (
        re.compile(r"^(one_of|among_the|ranked_as|considered_as|known_as)_", re.I),
        "Superlative/ranking relations encode a document-specific claim; "
        "model rankings as entity properties (e.g. rank, market_cap).",
    ),
    (
        re.compile(r"largest_by_|most_valuable|most_profitable|most_powerful|most_influential", re.I),
        "Superlative/ranking relations encode a snapshot ranking, not a reusable schema interaction.",
    ),
    # ── Single-event corporate narrative ─────────────────────────────────────
    # "rose_to_dominate_*", "grew_to_*" describe a historical narrative arc,
    # not a repeatable schema-level interaction.
    (
        re.compile(r"rose_to_(dominate|lead|control)|grew_to_(become|dominate|control)", re.I),
        "Narrative-arc relations describe a historical story beat; "
        "use 'competes_with' or model market share as a property.",
    ),
    # ── IPO / listing events pointing to Event node ──────────────────────────
    # "had_ipo_in", "went_public_in", "listed_in" with an Event target are
    # Date hacks (IPO date should be a property on the company).
    (
        re.compile(r"(had_ipo|went_public|ipo_in|listed_in|went_ipo)", re.I),
        "'had_ipo_in', 'went_public_in', etc. encode a date event; "
        "store the IPO date as a property on the Company instead.",
    ),
    # ── Wealth / billionaire creation ────────────────────────────────────────
    # "created_billionaires", "made_billionaires" describe a one-off outcome.
    (
        re.compile(r"creat.*billionaire|made.*billionaire|generat.*billionaire", re.I),
        "Billionaire-creation relations describe a single historical outcome; "
        "use a financial event with a property instead.",
    ),
    # ── is_a / generic type assertion ────────────────────────────────────────
    # "is_a: Company → TechnologyArea" is a type annotation, not a schema
    # relation.  Entity types already encode this information.
    (
        re.compile(r"^is_a$", re.I),
        "'is_a' is a type assertion, not a reusable schema relation; "
        "encode the category in the entity type description instead.",
    ),
    # ── Synonyms of existing generic relations ────────────────────────────────
    # "launched" overlaps entirely with "released"; "acquired_company" with
    # "acquired"; "announced" with "released" or "developed".  These near-
    # synonyms add noise without adding information.
    (
        re.compile(r"^launched$", re.I),
        "'launched' is a near-synonym of 'released'; "
        "use 'released' to keep the schema compact.",
    ),
    (
        re.compile(r"^(announced|unveiled|introduced)$", re.I),
        "'announced'/'unveiled'/'introduced' duplicate 'released' or 'developed'; "
        "use the more generic existing relation.",
    ),
    # ── Narrative / achievement relations ────────────────────────────────────
    # These describe a historical outcome or story beat, not a stable
    # ontological interaction.
    (
        re.compile(r"became_success|became_successful|became.*hit", re.I),
        "Success/achievement narrative relations are not stable ontological facts; "
        "model market performance as a property.",
    ),
    (
        re.compile(r"dominat.*market|rose.*dominat|captured.*market", re.I),
        "Market-dominance narrative relations describe a historical story beat; "
        "use 'competes_with' or encode market share as a property.",
    ),
    # ── Highly domain-specific hardware compatibility ─────────────────────────
    # "for_hardware: OperatingSystem → Hardware" is derived from a single
    # sentence in the source text and does not generalise.
    (
        re.compile(r"^for_hardware$", re.I),
        "'for_hardware' is derived from a single source sentence; "
        "use 'compatible_with' or model OS compatibility as a property.",
    ),
    # ── Edge-case bundling relations ─────────────────────────────────────────
    # "bundled_with" describes a specific distribution decision, not a
    # reusable ontological relationship.
    (
        re.compile(r"^bundled_with$|^bundled_together", re.I),
        "'bundled_with' encodes a specific distribution decision, not a reusable relation; "
        "model bundling as a property on the release event.",
    ),
    # ── Ownership when covered by produces/provides/is_brand_of ──────────────
    # A bare "owns: Company → Product" is redundant when those relations exist.
    (
        re.compile(r"^owns$", re.I),
        "'owns' (Company → Product) is redundant when 'produces', 'provides', or "
        "'is_brand_of' already express the ownership relationship.",
    ),
    # ── Vague product-to-product usage ───────────────────────────────────────
    # "used_for: Product → Product" is semantically underspecified;
    # it conflates compatibility, dependency, and operational context.
    (
        re.compile(r"^used_for$", re.I),
        "'used_for' between two products is too vague to be actionable; "
        "replace with a specific relation (e.g. 'compatible_with', 'requires', 'runs_on').",
    ),
    # ── Standalone 'features' ─────────────────────────────────────────────────
    # "features: Product → Product" conflates inclusion and endorsement;
    # use 'includes' or 'compatible_with' instead.
    (
        re.compile(r"^features$", re.I),
        "'features' (Product → Product) conflates inclusion, endorsement, and "
        "compatibility; use 'includes', 'compatible_with', or 'is_based_on'.",
    ),
    # ── Narrative influence / inspiration ─────────────────────────────────────
    # "inspired_by" encodes a one-off historical influence, not a reusable
    # ontological interaction.
    (
        re.compile(r"^inspired_by$|was_inspired_by|drew_inspiration", re.I),
        "'inspired_by' encodes a one-off narrative influence, not a reusable "
        "schema interaction; remove it.",
    ),
    # ── One-off operational deployment ────────────────────────────────────────
    # "used_in_operation: Product → Location" describes a single narrative
    # deployment (e.g. a product used in one military/government operation).
    (
        re.compile(r"used_in_operation|used_in_military|deployed_in_operation|used_in_conflict", re.I),
        "Operation-usage relations describe a single narrative deployment; "
        "model usage context as a property or remove.",
    ),
    # ── Headquarters relocation event ─────────────────────────────────────────
    # "moved_headquarters_to" is a temporal event, not a stable relation;
    # 'headquartered_in' plus a date property already covers it.
    (
        re.compile(r"^moved_headquarters|^relocated_to$|^moved_hq|^moved_.*_to$", re.I),
        "Headquarters-move relations encode a one-off relocation event; "
        "use 'headquartered_in' with a date property instead.",
    ),
    # ── Leadership succession events ──────────────────────────────────────────
    # "replaced_as_ceo", "succeeded_as_ceo" describe a single leadership change.
    # Tenure should be modelled with start/end date properties on a role relation.
    (
        re.compile(r"^replaced_as_|^succeeded_as_|_as_ceo$|^became_ceo$|^resigned_as_", re.I),
        "Leadership-succession relations encode a one-off change event; model "
        "tenure with start/end date properties on a 'holds_position' relation.",
    ),
    # ── Redundant inverse of develops/produces ────────────────────────────────
    # "is_brand_of: Product → Company" is the inverse of the forward
    # Organization → Product production relation and adds no new information.
    (
        re.compile(r"^is_brand_of$", re.I),
        "'is_brand_of' (Product → Organization) is a redundant inverse of "
        "'develops_product'/'produces'; keep the forward relation only.",
    ),
]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class SanitizationReport:
    """Structured record of every change made by :class:`SchemaSanitizer`.

    Attributes:
        removed_entity_types: Names of entity types that were removed.
        removed_relations:    ``(group_name, relation_name)`` pairs removed.
        removed_groups:       Names of relation groups that were dropped.
        removed_examples:     ``(group_name, example_type)`` pairs removed.
    """

    removed_entity_types: list[str] = field(default_factory=list)
    removed_relations: list[tuple[str, str]] = field(default_factory=list)
    removed_groups: list[str] = field(default_factory=list)
    removed_examples: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return (
            len(self.removed_entity_types)
            + len(self.removed_relations)
            + len(self.removed_groups)
            + len(self.removed_examples)
        )

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        if not self.total_changes:
            return "SchemaSanitizer: no changes needed."
        parts: list[str] = []
        if self.removed_entity_types:
            parts.append(
                f"{len(self.removed_entity_types)} entity type(s) removed "
                f"({', '.join(self.removed_entity_types)})"
            )
        if self.removed_relations:
            parts.append(f"{len(self.removed_relations)} relation(s) removed")
        if self.removed_groups:
            parts.append(
                f"{len(self.removed_groups)} group(s) removed "
                f"({', '.join(self.removed_groups)})"
            )
        if self.removed_examples:
            parts.append(f"{len(self.removed_examples)} example(s) removed")
        return "SchemaSanitizer: " + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------


class SchemaSanitizer:
    """Deterministic post-processor for LLM-generated :class:`EnhancedDRGSchema` objects.

    All passes are stateless and idempotent — calling :meth:`sanitize` twice
    on the same schema produces the same result as calling it once.

    The sanitizer applies two categories of checks:

    **Structural checks** (Passes 1–5, 6–8):
        Rules that are *always* correct regardless of domain.  These include
        removing primitive entity types, fixing invalid relation endpoints,
        and deduplicating names.

    **Semantic role-compatibility check** (Pass 9):
        Classifies each entity type into a broad semantic role (ACTOR,
        ARTIFACT, PLACE, OCCURRENCE) based on its name, then verifies that
        each relation's (source_role, target_role) pair is compatible with
        the relation's semantic family (legal, valuation, criticism, …).
        Relations whose type pair violates a declared rule are removed.
        This pass is intentionally conservative: it only fires when *both*
        a name-pattern match and a role-mismatch occur simultaneously, so
        the false-positive rate is very low.

    **Canonical vocabulary normalization** (Passes 0a/0b/0c):
        Runs *before* the deduplication passes.  Collapses synonymous relation
        names, synonymous/narrower entity types, and fragmented relation groups
        onto single canonical concepts so the ontology stays reusable across
        documents.  Controlled by constructor flags / environment variables.
    """

    def __init__(
        self,
        *,
        canonicalize_names: bool | None = None,
        collapse_artifact_subtypes: bool | None = None,
        merge_relation_groups: bool | None = None,
    ) -> None:
        """Configure the canonical-vocabulary passes.

        Each flag defaults to an environment variable, which in turn defaults to
        ``True`` (canonicalization is on by default):

        * ``canonicalize_names``          — ``DRG_SCHEMA_CANONICALIZE``
        * ``collapse_artifact_subtypes``  — ``DRG_SCHEMA_COLLAPSE_ARTIFACT_SUBTYPES``
        * ``merge_relation_groups``       — ``DRG_SCHEMA_MERGE_RELATION_GROUPS``

        Passing an explicit ``bool`` overrides the environment variable.  All
        structural passes (1–11) always run regardless of these flags.
        """
        self.canonicalize_names = (
            _env_flag("DRG_SCHEMA_CANONICALIZE", True)
            if canonicalize_names is None
            else canonicalize_names
        )
        self.collapse_artifact_subtypes = (
            _env_flag("DRG_SCHEMA_COLLAPSE_ARTIFACT_SUBTYPES", True)
            if collapse_artifact_subtypes is None
            else collapse_artifact_subtypes
        )
        self.merge_relation_groups = (
            _env_flag("DRG_SCHEMA_MERGE_RELATION_GROUPS", True)
            if merge_relation_groups is None
            else merge_relation_groups
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sanitize(
        self, schema: EnhancedDRGSchema
    ) -> tuple[EnhancedDRGSchema, SanitizationReport]:
        """Run all sanitization passes and return the cleaned schema + report.

        Passes are applied in a fixed order so that later passes operate on
        the already-cleaned output of earlier ones.

        Pass order::

            P0a → P0b → P0c → P1 → P2 → P3 → P4 → P5 → P9 → P10 → P11 → P6 → P7 → P8

        Passes 0a-0c (canonical vocabulary) run first so downstream dedup passes
        remove the redundancy they create.

        Pass 9 (semantic type compatibility) is inserted before the empty-group
        and orphan-type cleanup passes (P6, P7) so that any groups or entity
        types invalidated by P9 are automatically removed by those passes.

        Args:
            schema: The raw ``EnhancedDRGSchema`` returned by the LLM.

        Returns:
            A ``(cleaned_schema, report)`` tuple where ``report`` describes
            every change that was made.
        """
        report = SanitizationReport()

        # P0a-P0c: canonical-vocabulary normalization runs first so that the
        # deduplication/orphan passes clean up any redundancy the collapse
        # creates (e.g. two 'develops_product' relations, a merged entity type).
        schema = self._pass0a_canonicalize_relation_names(schema, report)
        schema = self._pass0b_canonicalize_entity_types(schema, report)
        schema = self._pass0c_merge_relation_groups(schema, report)

        schema = self._pass1_remove_primitive_entity_types(schema, report)
        schema = self._pass2_validate_source_target_types(schema, report)
        schema = self._pass3_deduplicate_relation_names(schema, report)
        schema = self._pass4_deduplicate_semantic_relations(schema, report)
        schema = self._pass5_deduplicate_relation_groups(schema, report)
        # P9 must run before P6/P7 so that groups emptied by P9 are purged.
        schema = self._pass9_validate_semantic_types(schema, report)
        # P10 must also run before P6/P7 for the same reason.
        schema = self._pass10_prune_document_specific_relations(schema, report)
        # P11: rescue orphan entity types before P7 deletes them.
        schema = self._pass11_rescue_orphan_entity_types(schema, report)
        schema = self._pass6_remove_empty_groups(schema, report)
        schema = self._pass7_remove_orphan_entity_types(schema, report)
        schema = self._pass8_validate_examples(schema, report)

        if report.total_changes:
            logger.info(report.summary())
        else:
            logger.debug("SchemaSanitizer: schema is already clean, no changes needed.")

        return schema, report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(name: str) -> str:
        """Lowercase, strip, and replace hyphens/spaces with underscores."""
        return (name or "").strip().lower().replace("-", "_").replace(" ", "_")

    @classmethod
    def _base_name(cls, name: str) -> str:
        """Strip known noun and directional suffixes from a normalized relation name.

        ``developed_product`` → ``developed``
        ``acquired_company``  → ``acquired``
        ``developed_by``      → ``developed``
        ``expanded_into``     → ``expanded``
        """
        n = cls._normalize(name)
        for suffix in (*_NOUN_SUFFIXES, *_DIRECTIONAL_SUFFIXES):
            if n.endswith(suffix) and len(n) > len(suffix):
                return n[: -len(suffix)]
        return n

    @classmethod
    def _endpoint_free_name(cls, name: str) -> str:
        """Strip endpoint-type tokens from a relation name (principle #8).

        Removes at most one leading and one trailing :data:`_ENDPOINT_ROLE_TOKENS`
        segment so the name describes the interaction, not its endpoints::

            organization_monitors_person   -> monitors
            person_contributed_to_field    -> contributed_to
            develops_product               -> develops
            event_caused_concept           -> caused

        Names that carry no role token (``founded_by``, ``works_at``,
        ``is_version_of``, ``has_facility_in``) are returned unchanged.  A strip
        is skipped when it would leave only role/connector tokens behind, so the
        verb of the interaction is always preserved.
        """
        n = cls._normalize(name)
        parts = n.split("_")
        if len(parts) < 2:
            return n

        def _has_verb(segs: list[str]) -> bool:
            return any(
                s not in _ENDPOINT_ROLE_TOKENS
                and s not in _CONNECTOR_TOKENS
                and s not in _LIGHT_VERB_TOKENS
                for s in segs
            )

        # Strip one leading role token (e.g. "organization_monitors_person").
        if parts[0] in _ENDPOINT_ROLE_TOKENS and _has_verb(parts[1:]):
            parts = parts[1:]
        # Strip one trailing role token (e.g. "monitors_person", "develops_product").
        if len(parts) > 1 and parts[-1] in _ENDPOINT_ROLE_TOKENS and _has_verb(parts[:-1]):
            parts = parts[:-1]

        return "_".join(parts) or n

    @staticmethod
    def _rebuild(
        schema: EnhancedDRGSchema,
        *,
        entity_types: list[EntityType] | None = None,
        relation_groups: list[RelationGroup] | None = None,
    ) -> EnhancedDRGSchema:
        """Return a new schema with optionally replaced fields."""
        return EnhancedDRGSchema(
            entity_types=entity_types if entity_types is not None else list(schema.entity_types),
            relation_groups=(
                relation_groups if relation_groups is not None else list(schema.relation_groups)
            ),
            entity_groups=list(schema.entity_groups),
            property_groups=list(schema.property_groups),
            auto_discovery=schema.auto_discovery,
        )

    @staticmethod
    def _rebuild_group(rg: RelationGroup, relations: list[Relation]) -> RelationGroup:
        """Return a new RelationGroup with replaced relations (keeps other fields)."""
        return RelationGroup(
            name=rg.name,
            description=rg.description,
            relations=relations,
            examples=list(rg.examples),
        )

    @staticmethod
    def _rebuild_group_with_examples(
        rg: RelationGroup,
        examples: list[dict[str, Any]],
    ) -> RelationGroup:
        """Return a new RelationGroup with replaced examples (keeps other fields)."""
        return RelationGroup(
            name=rg.name,
            description=rg.description,
            relations=list(rg.relations),
            examples=examples,
        )

    # ------------------------------------------------------------------
    # Pass 0a — canonical relation names
    # ------------------------------------------------------------------

    def _pass0a_canonicalize_relation_names(
        self, schema: EnhancedDRGSchema, report: SanitizationReport
    ) -> EnhancedDRGSchema:
        """Rename synonymous / endpoint-encoded relations to their canonical name.

        Two normalizations are applied, in order:

        1. :data:`_RELATION_NAME_CANONICAL` — exact synonym lookup
           (``developed`` / ``created`` / ``produces`` → ``develops``).
        2. :meth:`_endpoint_free_name` — strips endpoint-type tokens from the
           name (``organization_monitors_person`` → ``monitors``), per
           principle #8.

        No relations are removed here — renaming makes equivalent relations
        collide on a single name, which Pass 3/Pass 4 then deduplicate.  When
        an endpoint-free name is shared across *different* endpoint types (e.g.
        ``monitors: Org→Person`` and ``monitors: Org→Concept``) the endpoint-aware
        Pass 3 keeps both.
        """
        if not self.canonicalize_names:
            return schema

        new_groups: list[RelationGroup] = []
        changed = False

        for rg in schema.relation_groups:
            kept: list[Relation] = []
            group_changed = False

            for rel in rg.relations:
                norm = self._normalize(rel.name)
                canon = _RELATION_NAME_CANONICAL.get(norm, norm)
                canon = self._endpoint_free_name(canon)
                if canon != norm:
                    logger.info(
                        "SchemaSanitizer [P0a]: canonicalizing relation %r -> %r"
                        " in group %r.",
                        rel.name,
                        canon,
                        rg.name,
                    )
                    rel = Relation(
                        name=canon,
                        src=rel.src,
                        dst=rel.dst,
                        description=rel.description,
                        detail=rel.detail,
                        properties=dict(rel.properties),
                    )
                    group_changed = True
                kept.append(rel)

            if group_changed:
                new_groups.append(self._rebuild_group(rg, kept))
                changed = True
            else:
                new_groups.append(rg)

        return self._rebuild(schema, relation_groups=new_groups) if changed else schema

    # ------------------------------------------------------------------
    # Pass 0b — canonical entity types
    # ------------------------------------------------------------------

    def _pass0b_canonicalize_entity_types(
        self, schema: EnhancedDRGSchema, report: SanitizationReport
    ) -> EnhancedDRGSchema:
        """Rename/merge synonymous or sub-typed entity types onto canonical types.

        Two maps are consulted per entity type (normalized name lookup):

        * :data:`_ENTITY_TYPE_CANONICAL` — always applied (Company → Organization).
        * :data:`_ARTIFACT_SUBTYPE_CANONICAL` — applied only when
          ``collapse_artifact_subtypes`` is set (OperatingSystem/Hardware → Product).

        When two entity types collapse onto the same canonical name their
        definitions are merged (examples and properties are unioned; the
        description of the entity whose *own* name equals the canonical name
        wins).  All relations are rewired to the canonical endpoints, and the
        merged-away original names are recorded in the report.
        """
        if not self.canonicalize_names:
            return schema

        rename: dict[str, str] = {}
        for et in schema.entity_types:
            norm = self._normalize(et.name)
            canon = _ENTITY_TYPE_CANONICAL.get(norm)
            if canon is None and self.collapse_artifact_subtypes:
                canon = _ARTIFACT_SUBTYPE_CANONICAL.get(norm)
            if canon and canon != et.name:
                rename[et.name] = canon

        if not rename:
            return schema

        merged: dict[str, EntityType] = {}
        order: list[str] = []
        canonical_def: set[str] = set()  # targets whose description is authoritative

        for et in schema.entity_types:
            target = rename.get(et.name, et.name)
            is_canonical_named = et.name == target

            if target not in merged:
                merged[target] = EntityType(
                    name=target,
                    description=et.description,
                    examples=list(et.examples),
                    properties=dict(et.properties),
                )
                order.append(target)
                if is_canonical_named:
                    canonical_def.add(target)
                elif target != et.name:
                    logger.info(
                        "SchemaSanitizer [P0b]: canonicalizing entity type %r -> %r.",
                        et.name,
                        target,
                    )
            else:
                existing = merged[target]
                examples = list(existing.examples)
                for ex in et.examples:
                    if ex not in examples:
                        examples.append(ex)

                if is_canonical_named and target not in canonical_def:
                    description = et.description
                    props = dict(et.properties)
                    for k, v in existing.properties.items():
                        props.setdefault(k, v)
                    canonical_def.add(target)
                else:
                    description = existing.description
                    props = dict(existing.properties)
                    for k, v in et.properties.items():
                        props.setdefault(k, v)

                merged[target] = EntityType(
                    name=target,
                    description=description,
                    examples=examples,
                    properties=props,
                )
                logger.info(
                    "SchemaSanitizer [P0b]: merging entity type %r into %r.",
                    et.name,
                    target,
                )

        surviving = set(order)
        for et in schema.entity_types:
            if et.name not in surviving:
                report.removed_entity_types.append(et.name)

        new_entity_types = [merged[name] for name in order]

        new_groups: list[RelationGroup] = []
        for rg in schema.relation_groups:
            rels: list[Relation] = []
            for rel in rg.relations:
                new_src = rename.get(rel.src or "", rel.src)
                new_dst = rename.get(rel.dst or "", rel.dst)
                if new_src != rel.src or new_dst != rel.dst:
                    rel = Relation(
                        name=rel.name,
                        src=new_src,
                        dst=new_dst,
                        description=rel.description,
                        detail=rel.detail,
                        properties=dict(rel.properties),
                    )
                rels.append(rel)
            new_groups.append(self._rebuild_group(rg, rels))

        return self._rebuild(
            schema, entity_types=new_entity_types, relation_groups=new_groups
        )

    # ------------------------------------------------------------------
    # Pass 0c — merge fragmented relation groups
    # ------------------------------------------------------------------

    def _pass0c_merge_relation_groups(
        self, schema: EnhancedDRGSchema, report: SanitizationReport
    ) -> EnhancedDRGSchema:
        """Merge fragmented relation groups onto canonical group families.

        Uses :data:`_RELATION_GROUP_CANONICAL` (normalized group-name lookup).
        Groups that map to the same canonical family are concatenated (relations
        and examples) into the first occurrence; duplicate relations produced by
        the merge are cleaned up by Pass 3/Pass 4.
        """
        if not self.merge_relation_groups:
            return schema

        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        changed = False

        for rg in schema.relation_groups:
            canon = _RELATION_GROUP_CANONICAL.get(self._normalize(rg.name))
            display = canon or rg.name
            key = self._normalize(display)

            if key not in merged:
                merged[key] = {
                    "name": display,
                    "description": rg.description,
                    "relations": list(rg.relations),
                    "examples": list(rg.examples),
                }
                order.append(key)
                if canon and canon != rg.name:
                    logger.info(
                        "SchemaSanitizer [P0c]: renaming group %r -> %r.",
                        rg.name,
                        display,
                    )
                    changed = True
            else:
                merged[key]["relations"].extend(rg.relations)
                merged[key]["examples"].extend(rg.examples)
                logger.info(
                    "SchemaSanitizer [P0c]: merging group %r into %r.",
                    rg.name,
                    merged[key]["name"],
                )
                report.removed_groups.append(rg.name)
                changed = True

        if not changed:
            return schema

        new_groups = [
            RelationGroup(
                name=merged[key]["name"],
                description=merged[key]["description"],
                relations=merged[key]["relations"],
                examples=merged[key]["examples"],
            )
            for key in order
        ]
        return self._rebuild(schema, relation_groups=new_groups)

    # ------------------------------------------------------------------
    # Pass 1 — primitive entity types
    # ------------------------------------------------------------------

    def _pass1_remove_primitive_entity_types(
        self, schema: EnhancedDRGSchema, report: SanitizationReport
    ) -> EnhancedDRGSchema:
        """Remove entity types whose names represent literal scalar values.

        Dates, years, amounts, percentages, versions, etc. must always be
        modeled as properties on entity types or relations — never as standalone
        entity nodes.  This pass removes them and cascades to relations.
        """
        removed: set[str] = set()
        kept_entity_types: list[EntityType] = []

        for et in schema.entity_types:
            if self._normalize(et.name) in PRIMITIVE_ENTITY_TYPE_NAMES:
                logger.info(
                    "SchemaSanitizer [P1]: removing primitive entity type %r"
                    " — use a property instead.",
                    et.name,
                )
                removed.add(et.name)
                report.removed_entity_types.append(et.name)
            else:
                kept_entity_types.append(et)

        if not removed:
            return schema

        # Cascade: remove relations that touch a removed type.
        new_groups: list[RelationGroup] = []
        for rg in schema.relation_groups:
            kept: list[Relation] = []
            for rel in rg.relations:
                if rel.src in removed or rel.dst in removed:
                    logger.info(
                        "SchemaSanitizer [P1]: dropping relation %r in group %r"
                        " (endpoint %r is primitive).",
                        rel.name,
                        rg.name,
                        rel.src if rel.src in removed else rel.dst,
                    )
                    report.removed_relations.append((rg.name, rel.name))
                else:
                    kept.append(rel)
            if not kept:
                logger.info(
                    "SchemaSanitizer [P1]: dropping empty relation group %r.",
                    rg.name,
                )
                report.removed_groups.append(rg.name)
            elif kept != list(rg.relations):
                new_groups.append(self._rebuild_group(rg, kept))
            else:
                new_groups.append(rg)

        return self._rebuild(schema, entity_types=kept_entity_types, relation_groups=new_groups)

    # ------------------------------------------------------------------
    # Pass 2 — source/target type validity
    # ------------------------------------------------------------------

    def _pass2_validate_source_target_types(
        self, schema: EnhancedDRGSchema, report: SanitizationReport
    ) -> EnhancedDRGSchema:
        """Remove relations whose source or target is not a defined entity type.

        Also silently fixes case mismatches (e.g. ``company`` → ``Company``).
        """
        valid_types: set[str] = {et.name for et in schema.entity_types}
        # Build a case-insensitive lookup for auto-correction.
        lower_to_canonical: dict[str, str] = {
            et.name.lower(): et.name for et in schema.entity_types
        }

        new_groups: list[RelationGroup] = []
        changed = False

        for rg in schema.relation_groups:
            kept: list[Relation] = []
            group_changed = False

            for rel in rg.relations:
                src = rel.src or ""
                dst = rel.dst or ""

                resolved_src = src if src in valid_types else lower_to_canonical.get(src.lower())
                resolved_dst = dst if dst in valid_types else lower_to_canonical.get(dst.lower())

                if resolved_src is None or resolved_dst is None:
                    missing = []
                    if resolved_src is None:
                        missing.append(f"source={src!r}")
                    if resolved_dst is None:
                        missing.append(f"target={dst!r}")
                    logger.info(
                        "SchemaSanitizer [P2]: dropping relation %r in group %r"
                        " — undefined type(s): %s.",
                        rel.name,
                        rg.name,
                        ", ".join(missing),
                    )
                    report.removed_relations.append((rg.name, rel.name))
                    group_changed = True
                    continue

                # Fix case mismatch silently.
                if resolved_src != src or resolved_dst != dst:
                    rel = Relation(
                        name=rel.name,
                        src=resolved_src,
                        dst=resolved_dst,
                        description=rel.description,
                        detail=rel.detail,
                        properties=dict(rel.properties),
                    )
                    group_changed = True

                kept.append(rel)

            if group_changed:
                new_groups.append(self._rebuild_group(rg, kept))
                changed = True
            else:
                new_groups.append(rg)

        return self._rebuild(schema, relation_groups=new_groups) if changed else schema

    # ------------------------------------------------------------------
    # Pass 3 — duplicate relation names (exact)
    # ------------------------------------------------------------------

    def _pass3_deduplicate_relation_names(
        self, schema: EnhancedDRGSchema, report: SanitizationReport
    ) -> EnhancedDRGSchema:
        """Keep the first occurrence of each ``(name, src, dst)`` triple.

        Traverses relation groups in order; within each group, relations are
        also processed in order.  The first occurrence always wins.

        The dedup key includes the endpoints (not just the name) so that a
        single canonical, endpoint-free relation name may legitimately connect
        different endpoint pairs — e.g. ``monitors: Org→Person`` and
        ``monitors: Org→Concept`` both survive.  Exact ``(name, src, dst)``
        duplicates are still removed.  ``EnhancedDRGSchema`` already indexes a
        relation name to a list of endpoint pairs, so shared names are supported
        downstream.
        """
        seen_keys: set[tuple[str, str, str]] = set()
        new_groups: list[RelationGroup] = []
        changed = False

        for rg in schema.relation_groups:
            kept: list[Relation] = []
            group_changed = False

            for rel in rg.relations:
                key = (
                    self._normalize(rel.name),
                    self._normalize(rel.src or ""),
                    self._normalize(rel.dst or ""),
                )
                if key in seen_keys:
                    logger.info(
                        "SchemaSanitizer [P3]: dropping duplicate relation %r "
                        "(%s→%s) in group %r.",
                        rel.name,
                        rel.src,
                        rel.dst,
                        rg.name,
                    )
                    report.removed_relations.append((rg.name, rel.name))
                    group_changed = True
                else:
                    seen_keys.add(key)
                    kept.append(rel)

            if group_changed:
                new_groups.append(self._rebuild_group(rg, kept))
                changed = True
            else:
                new_groups.append(rg)

        return self._rebuild(schema, relation_groups=new_groups) if changed else schema

    # ------------------------------------------------------------------
    # Pass 4 — duplicate semantic relations
    # ------------------------------------------------------------------

    def _pass4_deduplicate_semantic_relations(
        self, schema: EnhancedDRGSchema, report: SanitizationReport
    ) -> EnhancedDRGSchema:
        """Remove relations whose semantic key ``(base_name, src, dst)`` duplicates
        an earlier one, including inverse duplicates.

        Two sub-checks are applied in order:

        **Forward duplicate** — same ``(base_name, src, dst)``:
            ``developed_product: Company→Product`` is a duplicate of
            ``developed: Company→Product``.

        **Inverse duplicate** — same ``base_name`` but ``(src, dst)`` swapped:
            ``developed_by: Product→Company`` is an inverse duplicate of
            ``developed: Company→Product``.  Only applies when the base name
            matches *and* the endpoint types are exactly swapped, so symmetric
            relations (e.g. ``partnered_with: Company→Company``) are unaffected.
        """
        seen_keys: set[tuple[str, str, str]] = set()
        new_groups: list[RelationGroup] = []
        changed = False

        for rg in schema.relation_groups:
            kept: list[Relation] = []
            group_changed = False

            for rel in rg.relations:
                base = self._base_name(rel.name)
                src_n = self._normalize(rel.src or "")
                dst_n = self._normalize(rel.dst or "")
                key = (base, src_n, dst_n)
                inverse_key = (base, dst_n, src_n)
                is_symmetric = src_n == dst_n

                if key in seen_keys:
                    logger.info(
                        "SchemaSanitizer [P4]: dropping forward duplicate %r in group %r"
                        " (key=%s).",
                        rel.name,
                        rg.name,
                        key,
                    )
                    report.removed_relations.append((rg.name, rel.name))
                    group_changed = True
                elif not is_symmetric and inverse_key in seen_keys:
                    logger.info(
                        "SchemaSanitizer [P4]: dropping inverse duplicate %r in group %r"
                        " (inverse of key=%s).",
                        rel.name,
                        rg.name,
                        inverse_key,
                    )
                    report.removed_relations.append((rg.name, rel.name))
                    group_changed = True
                else:
                    seen_keys.add(key)
                    kept.append(rel)

            if group_changed:
                new_groups.append(self._rebuild_group(rg, kept))
                changed = True
            else:
                new_groups.append(rg)

        return self._rebuild(schema, relation_groups=new_groups) if changed else schema

    # ------------------------------------------------------------------
    # Pass 5 — duplicate relation groups
    # ------------------------------------------------------------------

    def _pass5_deduplicate_relation_groups(
        self, schema: EnhancedDRGSchema, report: SanitizationReport
    ) -> EnhancedDRGSchema:
        """Keep the first relation group with each normalized name.

        When the LLM or the coverage pass emits two groups with the same name
        (e.g. two "Product and Technology Lifecycle" groups), the second is
        dropped entirely.  Any surviving relations from duplicate groups were
        already handled by the name/semantic deduplication passes.
        """
        seen_group_names: set[str] = set()
        new_groups: list[RelationGroup] = []
        changed = False

        for rg in schema.relation_groups:
            norm = self._normalize(rg.name)
            if norm in seen_group_names:
                logger.info(
                    "SchemaSanitizer [P5]: dropping duplicate relation group %r.",
                    rg.name,
                )
                report.removed_groups.append(rg.name)
                changed = True
            else:
                seen_group_names.add(norm)
                new_groups.append(rg)

        return self._rebuild(schema, relation_groups=new_groups) if changed else schema

    # ------------------------------------------------------------------
    # Pass 6 — empty relation groups
    # ------------------------------------------------------------------

    def _pass6_remove_empty_groups(
        self, schema: EnhancedDRGSchema, report: SanitizationReport
    ) -> EnhancedDRGSchema:
        """Drop relation groups with no surviving relations.

        ``RelationGroup.__post_init__`` raises ``SchemaError`` on empty groups,
        so we must never attempt to reconstruct one.  This pass ensures that
        any group emptied by earlier passes is silently removed rather than
        causing a hard error downstream.
        """
        new_groups: list[RelationGroup] = []
        changed = False

        for rg in schema.relation_groups:
            if not rg.relations:
                logger.info(
                    "SchemaSanitizer [P6]: dropping empty relation group %r.",
                    rg.name,
                )
                report.removed_groups.append(rg.name)
                changed = True
            else:
                new_groups.append(rg)

        return self._rebuild(schema, relation_groups=new_groups) if changed else schema

    # ------------------------------------------------------------------
    # Pass 7 — orphan entity types
    # ------------------------------------------------------------------

    def _pass7_remove_orphan_entity_types(
        self, schema: EnhancedDRGSchema, report: SanitizationReport
    ) -> EnhancedDRGSchema:
        """Remove entity types that are not used as source or target in any relation.

        Orphan entity types inflate the LLM prompt on future extraction calls
        and mislead the extractor into producing entities that can never
        participate in a valid relation.
        """
        used: set[str] = set()
        for rg in schema.relation_groups:
            for rel in rg.relations:
                if rel.src:
                    used.add(rel.src)
                if rel.dst:
                    used.add(rel.dst)

        kept: list[EntityType] = []
        changed = False

        for et in schema.entity_types:
            if et.name not in used:
                logger.warning(
                    "SchemaSanitizer [P7]: removing orphan entity type %r — "
                    "it was defined but never used as source or target in any "
                    "relation.  This usually means the LLM defined the type but "
                    "then used a generic type (e.g. Organization→Organization) "
                    "instead.  Consider adding a relation that uses %r as an "
                    "endpoint (e.g. 'filed_against: Organization → %s').",
                    et.name,
                    et.name,
                    et.name,
                )
                report.removed_entity_types.append(et.name)
                changed = True
            else:
                kept.append(et)

        return self._rebuild(schema, entity_types=kept) if changed else schema

    # ------------------------------------------------------------------
    # Pass 8 — example consistency
    # ------------------------------------------------------------------

    @staticmethod
    def _example_relation_name(ex: dict[str, Any]) -> str:
        """Extract the relation name from an example dict.

        Examples may store the relation name under different keys depending on
        the model/adapter: ``"relation"`` (most common), ``"type"``, or
        ``"rel"``.  The first non-empty value wins.
        """
        for key in ("relation", "type", "rel"):
            value = str(ex.get(key, "")).strip()
            if value:
                return value
        return ""

    def _build_instance_type_index(
        self, schema: EnhancedDRGSchema
    ) -> dict[str, set[str]]:
        """Map each entity-type example instance (lowercased) to its type name(s).

        Built from the ``examples`` field of every entity type, e.g.
        ``Organization: ["Anthropic", ...]`` yields ``{"anthropic": {"Organization"}}``.
        An instance may map to multiple types if it is listed under several.
        Used to best-effort validate that a relation example's head/tail is an
        instance of the relation's declared source/target type.
        """
        index: dict[str, set[str]] = {}
        for et in schema.entity_types:
            for inst in et.examples:
                key = self._normalize(str(inst))
                if key:
                    index.setdefault(key, set()).add(et.name)
        return index

    def _pass8_validate_examples(
        self, schema: EnhancedDRGSchema, report: SanitizationReport
    ) -> EnhancedDRGSchema:
        """Validate relation-group examples against their relation definitions.

        Each example has the shape
        ``{"head": "...", "relation": "<relation_name>", "tail": "..."}``
        (the relation name may also be under ``"type"`` or ``"rel"``).

        An example is removed when either check fails:

        1. **Relation existence** — its relation name does not match any relation
           defined in the same group (case-insensitive).  This catches examples
           left behind after a relation was removed or renamed by an earlier pass
           (e.g. ``released_on`` removed by P1, ``sued_by`` renamed by P11).

        2. **Endpoint type consistency** — when the head/tail instance is a known
           example of some entity type(s) (via :meth:`_build_instance_type_index`),
           and that type set does *not* include the relation's declared source
           (for head) or target (for tail), the example violates the relation
           definition and is removed.  Unknown instances are kept (conservative).
        """
        instance_types = self._build_instance_type_index(schema)
        new_groups: list[RelationGroup] = []
        changed = False

        for rg in schema.relation_groups:
            # relation name -> (src, dst) for endpoint validation
            rel_by_name: dict[str, Relation] = {
                self._normalize(rel.name): rel for rel in rg.relations
            }
            kept_examples: list[dict[str, Any]] = []
            group_changed = False

            for ex in rg.examples:
                if not isinstance(ex, dict):
                    kept_examples.append(ex)
                    continue

                ex_rel = self._example_relation_name(ex)

                # Examples with no relation name at all are kept as-is.
                if not ex_rel:
                    kept_examples.append(ex)
                    continue

                norm_rel = self._normalize(ex_rel)

                # Check 1: relation must exist in this group.
                if norm_rel not in rel_by_name:
                    logger.info(
                        "SchemaSanitizer [P8]: removing example %r from group %r"
                        " — relation not defined in this group.",
                        ex_rel,
                        rg.name,
                    )
                    report.removed_examples.append((rg.name, ex_rel))
                    group_changed = True
                    continue

                # Check 2: head/tail must be consistent with declared endpoints
                # when the instance's type is known.
                rel = rel_by_name[norm_rel]
                head_key = self._normalize(str(ex.get("head", "")))
                tail_key = self._normalize(str(ex.get("tail", "")))

                head_types = instance_types.get(head_key)
                tail_types = instance_types.get(tail_key)

                violation: str | None = None
                if head_types is not None and rel.src not in head_types:
                    violation = (
                        f"head {ex.get('head')!r} is a known {sorted(head_types)} "
                        f"but relation source is {rel.src!r}"
                    )
                elif tail_types is not None and rel.dst not in tail_types:
                    violation = (
                        f"tail {ex.get('tail')!r} is a known {sorted(tail_types)} "
                        f"but relation target is {rel.dst!r}"
                    )

                if violation:
                    logger.info(
                        "SchemaSanitizer [P8]: removing example %r from group %r"
                        " — endpoint type mismatch: %s.",
                        ex_rel,
                        rg.name,
                        violation,
                    )
                    report.removed_examples.append((rg.name, ex_rel))
                    group_changed = True
                    continue

                kept_examples.append(ex)

            if group_changed:
                new_groups.append(self._rebuild_group_with_examples(rg, kept_examples))
                changed = True
            else:
                new_groups.append(rg)

        return self._rebuild(schema, relation_groups=new_groups) if changed else schema

    # ------------------------------------------------------------------
    # Pass 9 — semantic source/target type compatibility
    # ------------------------------------------------------------------

    def _pass9_validate_semantic_types(
        self, schema: EnhancedDRGSchema, report: SanitizationReport
    ) -> EnhancedDRGSchema:
        """Remove relations whose (source_role, target_role) pair is semantically
        incoherent for the relation's declared semantic family.

        Algorithm
        ---------
        1. Classify every entity type in the schema into a broad semantic role
           (ACTOR, ARTIFACT, PLACE, OCCURRENCE, UNKNOWN) using
           :func:`_classify_entity_role`.
        2. For each relation, retrieve the roles of its source and target types.
        3. Iterate over :data:`_SEMANTIC_RULES` in order; the first rule whose
           ``pattern`` matches the relation name (``re.search``) is authoritative.
        4. If the rule's ``valid_source_roles`` or ``valid_target_roles`` constraint
           is violated, the relation is removed and the violation is logged at INFO.

        Design choices
        --------------
        * UNKNOWN is always accepted wherever any other role is accepted, so that
          entity types the classifier cannot categorise do not generate false positives.
        * The check is purely name-based — no LLM call is made.
        * Rules are intentionally conservative: they only fire on *clear* semantic
          mismatches (e.g. a valuation relation whose target is an ARTIFACT, or a
          legal relation whose target is a PLACE).
        * After this pass, Pass 6 removes any groups that became empty.
        """
        role_map: dict[str, str] = {
            et.name: _classify_entity_role(et.name) for et in schema.entity_types
        }

        new_groups: list[RelationGroup] = []
        changed = False

        for rg in schema.relation_groups:
            kept: list[Relation] = []
            group_changed = False

            for rel in rg.relations:
                src_role = role_map.get(rel.src or "", SemanticRole.UNKNOWN)
                dst_role = role_map.get(rel.dst or "", SemanticRole.UNKNOWN)

                violation: str | None = None
                for rule in _SEMANTIC_RULES:
                    if not rule.pattern.search(rel.name):
                        continue
                    # First matching rule is authoritative.
                    if rule.valid_source_roles and src_role not in rule.valid_source_roles:
                        violation = (
                            f"source type {rel.src!r} has role {src_role!r}, "
                            f"expected one of "
                            f"{sorted(rule.valid_source_roles - {SemanticRole.UNKNOWN})}. "
                            f"{rule.reason}"
                        )
                    elif rule.valid_target_roles and dst_role not in rule.valid_target_roles:
                        violation = (
                            f"target type {rel.dst!r} has role {dst_role!r}, "
                            f"expected one of "
                            f"{sorted(rule.valid_target_roles - {SemanticRole.UNKNOWN})}. "
                            f"{rule.reason}"
                        )
                    break  # only the first matching rule is applied

                if violation:
                    logger.info(
                        "SchemaSanitizer [P9]: dropping semantically invalid relation"
                        " %r in group %r — %s",
                        rel.name,
                        rg.name,
                        violation,
                    )
                    report.removed_relations.append((rg.name, rel.name))
                    group_changed = True
                else:
                    kept.append(rel)

            if group_changed:
                # Guard: if all relations were removed, don't call _rebuild_group
                # with an empty list (RelationGroup.__post_init__ rejects it).
                # P6 will remove the now-empty group.
                if kept:
                    new_groups.append(self._rebuild_group(rg, kept))
                changed = True
            else:
                new_groups.append(rg)

        return self._rebuild(schema, relation_groups=new_groups) if changed else schema

    # ------------------------------------------------------------------
    # Pass 10 — relation genericity / document-specificity pruning
    # ------------------------------------------------------------------

    def _pass10_prune_document_specific_relations(
        self, schema: EnhancedDRGSchema, report: SanitizationReport
    ) -> EnhancedDRGSchema:
        """Remove relations whose names match known document-specific patterns.

        A relation is *document-specific* when:

        * Its verb or phrase describes a one-off historical event that rarely
          generalises across different documents (e.g. ``suggested_name``,
          ``created_millionaires``, ``filed_motion_against``).
        * A more generic relation already in the schema can express the same
          interaction (e.g. ``marketed`` ≈ ``released`` or ``produces``).
        * The phrase was almost certainly extracted from a single sentence in
          the source text rather than from a repeating, multi-instance pattern.

        Matching is done against :data:`_DOCUMENT_SPECIFIC_PATTERNS`, a curated
        list of ``(regex, reason)`` pairs.  The normalised relation name is tested
        with ``re.search``; the first matching pattern is authoritative.

        This pass is intentionally narrow — patterns are only added when the
        risk of false positives is very low.  After this pass, Pass 6 removes
        any groups that became empty.
        """
        new_groups: list[RelationGroup] = []
        changed = False

        for rg in schema.relation_groups:
            kept: list[Relation] = []
            group_changed = False

            for rel in rg.relations:
                pruned_reason: str | None = None
                norm = self._normalize(rel.name)
                for pattern, reason in _DOCUMENT_SPECIFIC_PATTERNS:
                    if pattern.search(norm):
                        pruned_reason = reason
                        break

                if pruned_reason:
                    logger.info(
                        "SchemaSanitizer [P10]: pruning document-specific relation"
                        " %r in group %r — %s",
                        rel.name,
                        rg.name,
                        pruned_reason,
                    )
                    report.removed_relations.append((rg.name, rel.name))
                    group_changed = True
                else:
                    kept.append(rel)

            if group_changed:
                # Guard: if all relations were removed, don't call _rebuild_group
                # with an empty list (RelationGroup.__post_init__ rejects it).
                # P6 will remove the now-empty group.
                if kept:
                    new_groups.append(self._rebuild_group(rg, kept))
                changed = True
            else:
                new_groups.append(rg)

        return self._rebuild(schema, relation_groups=new_groups) if changed else schema

    # ------------------------------------------------------------------
    # Pass 11 — orphan entity type rescue
    # ------------------------------------------------------------------

    # Each entry: (type_name_pattern, [(rel_name_pattern, new_rel_name, swap_endpoints), ...])
    # When an entity type matching type_name_pattern is orphaned AND a relation
    # matching rel_name_pattern exists, the relation is rewritten so the orphaned
    # type becomes an endpoint — rescuing it from P7 deletion.
    # swap_endpoints=False → src unchanged, dst = orphan_type
    # swap_endpoints=True  → src = orphan_type, dst unchanged
    _ORPHAN_RESCUE_RULES: list = [
        (
            __import__("re").compile(r"legalcase|lawsuit|litigation", __import__("re").I),
            [
                (__import__("re").compile(r"^sued_by$|^filed_against$|^litigated_against$", __import__("re").I),
                 "filed_against", False),
                (__import__("re").compile(r"^settled_with$|^reached_settlement$", __import__("re").I),
                 "settled_case", False),
            ],
        ),
        (
            __import__("re").compile(r"^regulation$|^policy$|^legislation$|^directive$", __import__("re").I),
            [
                (__import__("re").compile(r"^designated_as$|^regulated_by$|^subject_to_regulation$", __import__("re").I),
                 "subject_to", False),
            ],
        ),
        (
            __import__("re").compile(r"fundinground|funding_round|investment_round", __import__("re").I),
            [
                (__import__("re").compile(r"^had_valuation_of$|^valued_at$|^raised_in_round$", __import__("re").I),
                 "raised_in", False),
            ],
        ),
    ]

    def _pass11_rescue_orphan_entity_types(
        self, schema: EnhancedDRGSchema, report: SanitizationReport
    ) -> EnhancedDRGSchema:
        """Rescue orphan entity types by rewiring generic relations to use them.

        When the LLM defines a semantically rich entity type (e.g. ``LegalCase``,
        ``Regulation``, ``FundingRound``) but then uses a broad type like
        ``Organization`` as both source and target in the corresponding relation,
        the entity type ends up orphaned and P7 would silently delete it.

        This pass runs *before* P7.  For each orphaned entity type it checks
        whether an existing relation matches a known "rescue rule".  If a match
        is found, the generic relation is rewritten to use the orphaned type as
        its endpoint — making the entity type non-orphan so P7 keeps it.

        Example
        -------
        Before::
            entity_types: [..., LegalCase (unused)]
            sued_by: Organization → Organization

        After P11::
            entity_types: [..., LegalCase]
            filed_against: Organization → LegalCase
        """
        used: set[str] = set()
        for rg in schema.relation_groups:
            for rel in rg.relations:
                if rel.src:
                    used.add(rel.src)
                if rel.dst:
                    used.add(rel.dst)

        orphans = [et for et in schema.entity_types if et.name not in used]
        if not orphans:
            return schema

        new_groups: list[RelationGroup] = [
            self._rebuild_group(rg, list(rg.relations)) for rg in schema.relation_groups
        ]
        changed = False

        for orphan in orphans:
            rescued = False
            for type_pattern, rel_rules in self._ORPHAN_RESCUE_RULES:
                if not type_pattern.search(orphan.name):
                    continue
                for rel_pattern, new_rel_name, swap in rel_rules:
                    for gi, rg in enumerate(new_groups):
                        for ri, rel in enumerate(rg.relations):
                            if not rel_pattern.search(self._normalize(rel.name)):
                                continue
                            new_src = orphan.name if swap else rel.src
                            new_dst = rel.dst if swap else orphan.name
                            new_rel = Relation(
                                name=new_rel_name,
                                src=new_src,
                                dst=new_dst,
                                description=rel.description,
                                detail=rel.detail,
                                properties=dict(rel.properties),
                            )
                            rels = list(new_groups[gi].relations)
                            rels[ri] = new_rel
                            new_groups[gi] = self._rebuild_group(new_groups[gi], rels)
                            logger.info(
                                "SchemaSanitizer [P11]: rescued orphan type %r — "
                                "rewrote %r -> %r (%s -> %s) in group %r.",
                                orphan.name,
                                rel.name,
                                new_rel_name,
                                new_src,
                                new_dst,
                                rg.name,
                            )
                            changed = True
                            rescued = True
                            break
                        if rescued:
                            break
                    if rescued:
                        break
                if rescued:
                    break

        return self._rebuild(schema, relation_groups=new_groups) if changed else schema
