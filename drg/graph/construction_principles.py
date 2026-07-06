"""Shared quality principles for schema generation and knowledge-graph construction.

These guidelines are injected into DSPy signatures so ontology design and
extraction optimize for semantic correctness, information density, and
cross-document reusability — not raw node or edge count.
"""

from __future__ import annotations

SCHEMA_GENERATION_PRINCIPLES = """
Ontology Design Principles (critical):

The schema is an ontology, NOT a summary of the current document. Produce the
smallest declarative ontology that preserves all reusable semantics needed for
downstream extraction. Model the domain itself, not a particular document. Do
not optimize for the number of entity types or relations alone, and never
sacrifice reusable semantics for compactness.

1. Prefer canonical entity types.
- Do not introduce multiple entity types for the same conceptual category.
  Prefer ONE reusable type (e.g. Organization) instead of overlapping variants
  (Company, Corporation, Firm) unless a distinction will consistently recur
  across many documents.
- Avoid unnecessary fragmentation of artifacts (Product vs Software vs Hardware
  vs OperatingSystem vs SoftwareSuite). Prefer the broader reusable type
  (Product) and record the specific kind as a property (e.g. category) unless
  the distinction is essential to downstream reasoning.
- When in doubt, choose the broader reusable entity type.

2. Avoid relation synonyms.
- Each semantic interaction has exactly ONE canonical relation. Never emit
  several relations that mean the same thing (developed / develops / created /
  produces; collaborated_with / partnered_with).
- Do NOT fragment one interaction across endpoint types: monitors_person,
  monitors_technology, and monitors_concept are the SAME relation `monitors`
  reused with different targets — declare a single `monitors` relation, not
  three. Relation names capture the semantic interaction, not the endpoints.

3. Keep relations domain-level, not document-level.
- A relation must represent a reusable semantic relationship expected to recur
  across many documents in the domain. Do not mint a relation just because one
  sentence used a distinctive verb.
- Avoid one-off narrative/event relations such as resigned_from,
  moved_headquarters_to, abandoned_after, ordered_by, revealed_identity_to.
  Ask: "Would this relation reasonably appear across many documents in the same
  domain?" If not, it belongs in the extracted graph, not the ontology.

4. Prefer properties for literal information.
- Model literals as properties, never as entity types or relation targets:
  dates, years, versions, roles, titles, status values, identifiers, monetary
  values, percentages, counts.
- Never define an entity type whose primary purpose is to store a literal value
  (Date, Amount, Version, a funding round that only holds a valuation, …).

5. Keep relation granularity consistent.
- All relations should sit at roughly the same semantic abstraction level. Do
  not mix broad reusable relations with narrow procedural or narrative ones.
  Describe stable semantic structure, not individual story events.

6. Generate canonical examples.
- Entity examples must be canonical entity names — reusable instances, not
  textual mentions.
- Never use pronouns, descriptions, aliases, shortened mentions, or generic
  noun phrases as examples ("her work", "the company", "major cities",
  "test subjects"). Use the canonical name the instance carries across
  documents.

7. Maintain type consistency.
- For every relation, the source type, target type, description, detail, and
  examples must all describe exactly the same semantic relationship.
- A relation description must NEVER contradict its declared source or target
  types (e.g. do not write "Organization sued Organization" on a relation
  declared Organization -> LegalCase).

8. Do not encode endpoint types into relation names.
- The endpoint types are already defined by the schema, so the name must carry
  only the semantic interaction. Prefer `develops` over `develops_product`,
  `monitors` over `organization_monitors_person`, `contributed_to` over
  `person_contributed_to_field`, `located_in` over `organization_located_in`.

9. Minimize ontology complexity.
- Prefer a smaller ontology of reusable concepts over a large one of narrowly
  specialized relations. Introduce a new entity type or relation ONLY when it
  is a genuinely distinct concept expected to recur across many documents.

10. Maintain a stable canonical vocabulary.
- Reuse existing ontology terminology across documents instead of inventing
  synonyms. Consistently prefer one canonical form per concept so the ontology
  stays consistent across datasets over the long term.

11. Organize relation groups by semantic cohesion.
- Relation groups must partition the ontology into coherent semantic areas. Do
  not create multiple groups describing the same concept; merge naturally
  related interactions into one broader semantic group.

12. Optimize for reusability.
- Heuristic: if a relation or entity type is unlikely to appear in many
  independent documents within the domain, it should NOT become part of the
  reusable ontology — model it as an extracted fact instead.

Naming quick-reference:
- Prefer: founded_by, acquired, partnered_with, develops, monitors, located_in.
- Avoid: develops_product, organization_monitors_person, moved_headquarters_to,
  became_market_leader, resigned_from, abandoned_after.

Final validation (verify before returning the schema):
- every entity type is reusable and canonical (not a synonym of another type)
- every relation is a single canonical relation, with an endpoint-free name
- no duplicate or synonymous relations exist
- no relation is document-specific or a one-off narrative event
- literals are modeled as properties, not entity types or relation endpoints
- examples are canonical entity names, not textual mentions
- descriptions do not contradict declared source/target types
- relation groups are semantically cohesive and non-overlapping
- every entity type participates in at least one relation
- the ontology remains compact, declarative, reusable, and domain-agnostic

Prefer reusable semantics over document-specific coverage.
""".strip()

KG_CONSTRUCTION_PRINCIPLES = """
Knowledge graph construction principles (optimize for quality and density):

Before creating any relationship, answer:
  1. Does this edge represent a unique semantic fact explicitly stated in the source?
  2. Is this fact already represented through another node or relationship?
  3. Does this edge add new information?
If (2) is yes and (3) is no, do not create the edge.
Prefer semantic precision over graph density.

1. Semantic correctness — each edge states one unique fact explicitly
   supported by the source text; do not collapse distinct facts into one
   generic relation.
2. Real-world modeling — use the most specific schema type that fits; do not
   force different concepts into the same generic entity type.
3. Properties over entities — store dates, amounts, percentages, roles,
   versions, and similar literals on entity or relation properties, not as
   standalone nodes.
4. Event intermediaries — when multiple facts describe the same occurrence
   (funding round, acquisition, lawsuit, launch), route them through an
   event entity instead of duplicating parallel binary edges.
5. No shortcut edges — do not add redundant or inferred edges solely to
   increase connectivity, reduce orphans, or inflate graph size.
6. Canonical entities — merge aliases (e.g. Pentagon / Department of Defense);
   use real names, not pronouns or vague fragments like "the company".
7. Meaningful structure — connect entities directly when the text supports
   peer relationships; avoid hub-and-spoke graphs where everything links only
   to one central node.
8. Preserve hierarchy — product versions, subsidiaries, divisions, model
   families via version_of, has_division, and related relations.
9. Semantic predicates — founded_by, develops, released, invested_in;
   avoid related_to / associated_with unless no better relation exists.
10. Temporal fidelity — retain dates on relations and properties when the text
    specifies time for funding, employment, releases, or legal events.
11. Graph consistency — respect relation direction and schema endpoint types;
    avoid duplicate or contradictory edges.
12. Information density — maximize extractable meaning per node and edge;
    prefer a compact, queryable graph over high node or edge counts.
""".strip()

RELATION_PREFLIGHT_CHECKLIST = """
Before creating a relationship, ask:
1. Does this edge represent a unique semantic fact explicitly stated in the source?
2. Is this fact already represented through another node or relationship?
3. Does this edge add new information?
If (2) is yes and (3) is no, do not create the edge.
Prefer semantic precision over graph density.
""".strip()

__all__ = [
    "KG_CONSTRUCTION_PRINCIPLES",
    "RELATION_PREFLIGHT_CHECKLIST",
    "SCHEMA_GENERATION_PRINCIPLES",
]
