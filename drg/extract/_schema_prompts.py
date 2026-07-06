"""Prompt text for schema-generation DSPy signatures.

Structure:
    SCHEMA_CORE_RULES — shared domain-ontology rules (included once per composed prompt)
    SCHEMA_*_TASK — pass-specific instructions only
    SCHEMA_*_INSTRUCTIONS — CORE + TASK, used at call time

Structural cleanup (primitives, duplicates, orphans, invalid endpoints) is handled
by ``SchemaSanitizer`` after each LLM pass — prompts focus on semantic modeling.
"""

from __future__ import annotations

SCHEMA_INTERACTION_FAMILIES = """
Interaction families to consider (derive relation names from the text, do not copy these labels blindly):
- organizational structure: founding, leadership, employment, division membership, board roles
- corporate transactions: investment, acquisition, partnership, ownership stakes
- product and technology lifecycle: development, release, launch, versioning, availability
- legal and regulatory: lawsuits, settlements, designations, compliance actions
- government and policy: orders, bans, contracts, restrictions involving institutions or officials
- events and milestones: IPO filings, funding rounds, announcements, operational incidents
- research and methods: techniques, frameworks, published findings attributed to actors
- education and programs: initiatives, advisory roles, institutional collaborations
""".strip()

SCHEMA_CORE_RULES = """
Your objective is NOT to model only the current document. Infer the narrowest reusable
domain that explains the text and build a compact, reusable, domain-agnostic extraction
ontology that would remain useful for future documents from the same domain. Avoid
domains like "Technology" or "Science" when a more specific recurring document family
exists (e.g. technology company profiles, scientific biographies). Generalize recurring
concepts instead of describing document-specific facts. Derive every type and relation
from observed semantic patterns — no hard-coded domain lists. Obey ontology_budget.

This ontology will drive downstream knowledge extraction. Optimize for future extraction
quality rather than perfectly describing the current document.

Domain modeling over document coverage:
- The schema is an ontology, not a summary of the input text. Model the domain itself —
  stable semantic structure expected to recur across many independent documents.
- This is ontology design, not information extraction. Do not encode document facts as
  schema structure. Relations must name recurring semantic interactions between concepts,
  not one-off narrative events or story beats from the current text.
- Before adding an entity type or relation, ask: "Would this represent a recurring
  concept of the inferred domain, or is it merely a document-specific fact?" Do not
  remove important domain concepts simply because they appear only once. One-off
  narrative facts belong in downstream extraction, not the reusable ontology.
- Prefer reusable semantics over document-specific coverage.

Ontology stability:
- Prefer stable semantic concepts over transient document concepts.
- The ontology should evolve slowly as new documents from the same domain are added.

Semantic modeling:
- Allocate entity types according to semantic importance, not frequency. Core domain
  concepts deserve dedicated entity types even if mentioned only a few times.
- Every entity type should improve extraction on unseen documents from the same domain.
- Keep all entity types at a consistent semantic abstraction level. Do not mix coarse
  categories with specialized subtypes (e.g. Organization alongside Restaurant;
  TaxonomicRank alongside conservation status; LiteraryWork alongside Character).
- One entity type = one semantic concept. Never reuse a type as a proxy for another
  concept or overload one type to represent multiple unrelated ideas.
- Prefer precise entity types over wrong generics (LegalCase, Regulation, FundingRound,
  Platform — not Organization/Product catch-alls for specialized concepts).
- Model recurring events as entity types when they anchor multiple participants or facts.
- Do not create multiple relations expressing the same semantic interaction. Prefer one
  reusable canonical relation whenever possible (develops, not develops_product).
- Prefer the most semantically specific source and target entity types available. Avoid
  broad endpoints when a more precise type exists (e.g. LegalCase over Organization).
- Literals (dates, amounts, roles, versions) → entity or relation properties, not nodes.
- Entity examples: canonical instance names, not pronouns or vague phrases.

Budget discipline:
- If ontology_budget is tight, omit lower-priority concepts. Never sacrifice semantic
  correctness: do not reuse an existing type as a placeholder, overload one type for
  multiple concepts, or merge unrelated concepts simply to save slots.

Relations (downstream extraction uses ONLY declared relations):
- Survey interaction_families; add a domain-level relation for each fundamental family
  with extractable text evidence (not single-sentence one-offs).
- A relation does not need to appear many times in this document. If it represents a
  fundamental interaction of the inferred domain, include it.
- Relation names must be canonical and reusable across future documents. Reject
  document-specific verbs, time qualifiers, or narrative hooks (opened_in, received_status,
  documented_in, published_this_year). Prefer stable interaction names (founded_in,
  has_status, mentioned_in, published_in).
- Every relation must be a concrete semantic edge extractable between two entity instances
  in future documents. Avoid conceptual, explanatory, or ontology-only relations that
  would rarely appear as explicit facts in text.
- Every relation: non-empty detail (verbatim cue phrase).
- Each relation group: 1–3 example triples with head/tail matching declared endpoints.

Post-processing removes scalar entity types, invalid endpoints, duplicates, orphans,
and document-specific one-off relations automatically — prioritize semantic correctness.
""".strip()

SCHEMA_GENERATION_TASK = """
Task: generate a draft EnhancedDRGSchema from the input text.

Before generating the schema:
1. Infer the domain.
2. Identify recurring concepts.
3. Generate the ontology.
4. Before finalizing, verify: each entity type = one semantic concept; each relation is
   reusable outside this document; relation names are canonical, not document-specific;
   endpoints use the most specific available entity types; example triples conform to
   declared source and target types.

Use the text as evidence of the underlying domain, not as a checklist of facts to
encode. Focus on reusable domain concepts and recurring interaction patterns within
ontology_budget. Prefer empty properties {} unless essential. Property values must be
plain short strings — never JSON Schema objects inside properties.
""".strip()

SCHEMA_REVIEW_TASK = """
Task: review and correct a draft schema (already structurally sanitized).

Fix semantic issues only — preserve what is already correct:
1. Remove or generalize anything that models this document instead of the domain.
2. Split overloaded entity types; introduce precise types (Platform, LegalCase, …)
   where Organization/Product is semantically wrong. Align entity types to one
   consistent abstraction level; remove types reused as proxies for other concepts.
3. Correct relation source/target to the most semantically specific available entity
   type; avoid broad endpoints (Organization→Organization) when a precise type exists;
   introduce intermediary event/legal types when broad X→X relations hide meaning.
4. Convert attribute-encoding relations (dates, roles, amounts as edges) to properties.
5. Replace document-specific relation names with canonical, domain-agnostic ones.
6. Re-check: one concept per entity type; reusable relations only; example triples
   match declared endpoints. If budget pressure caused overload, drop lower-priority
   concepts instead of forcing incorrect reuse.
""".strip()

SCHEMA_COVERAGE_AUDIT_TASK = """
Task: propose ONLY missing entity types and relations for domain-level coverage gaps.

Imagine another unseen document from the same domain. Only add entity types and
relations that would improve extraction across future documents.

Entity types: add when a recurring domain concept cannot be modeled with existing
types; each new type must improve extraction on unseen documents from the same domain
and appear in at least one proposed relation. Do not rename existing types. Do not add
types for document-specific instances. Obey additional_entity_budget.

Relations: add when no existing relation (name + endpoints) covers a fundamental domain
interaction, the pattern would recur across future documents with different entity pairs,
and the name is domain-agnostic. Do not propose relations that are semantic variants of
existing ones — only add genuinely new interaction patterns. No synonyms, narrative, or
vague relations. Every proposal needs detail (verbatim cue). Prefer empty lists over
low-quality additions. Obey additional_relation_budget.
""".strip()


def _compose_prompt(task: str) -> str:
    return f"{SCHEMA_CORE_RULES}\n\n{task}".strip()


SCHEMA_GENERATION_INSTRUCTIONS = _compose_prompt(SCHEMA_GENERATION_TASK)
SCHEMA_REVIEW_INSTRUCTIONS = _compose_prompt(SCHEMA_REVIEW_TASK)
SCHEMA_COVERAGE_AUDIT_INSTRUCTIONS = _compose_prompt(SCHEMA_COVERAGE_AUDIT_TASK)

SCHEMA_RETRY_GUIDANCE_EMPTY = (
    "No prior attempt failed. Infer the narrowest reusable domain and follow core "
    "ontology rules on the first pass — generalize recurring concepts, do not model "
    "only this document."
)

SCHEMA_RETRY_GUIDANCE_TEMPLATE = """
Prior attempt failed. Obey ontology_budget and core rules.

Failure: {reason}

If truncated/empty: fewer groups, shorter descriptions, stay within budget.
If semantics were wrong: use precise entity types (LegalCase, Regulation, …) instead
of forcing Organization/Product; remove document-specific types and relations; use
canonical relation names; keep one abstraction level; omit concepts rather than
overloading types. Do not repeat the same mistake.
""".strip()

__all__ = [
    "SCHEMA_CORE_RULES",
    "SCHEMA_COVERAGE_AUDIT_INSTRUCTIONS",
    "SCHEMA_COVERAGE_AUDIT_TASK",
    "SCHEMA_GENERATION_INSTRUCTIONS",
    "SCHEMA_GENERATION_TASK",
    "SCHEMA_INTERACTION_FAMILIES",
    "SCHEMA_RETRY_GUIDANCE_EMPTY",
    "SCHEMA_RETRY_GUIDANCE_TEMPLATE",
    "SCHEMA_REVIEW_INSTRUCTIONS",
    "SCHEMA_REVIEW_TASK",
]
