"""Provenance helpers for graph nodes and edges.

The public JSON surface remains backward-compatible: legacy metadata keys
(``source_ref``, ``source_documents`` and ``evidence``) keep working, while new
graphs also get a structured ``metadata["provenance"]`` block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "ProvenanceRecord",
    "attach_provenance",
    "find_text_provenance",
    "provenance_from_metadata",
]


_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ProvenanceRecord:
    """Structured source metadata for a graph element."""

    document_id: str | None = None
    sentence_id: str | None = None
    chunk_id: str | None = None
    source_span: tuple[int, int] | None = None
    snippet: str | None = None
    extracted_at: str | None = None
    extractor_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.document_id:
            out["document_id"] = self.document_id
        if self.sentence_id:
            out["sentence_id"] = self.sentence_id
        if self.chunk_id:
            out["chunk_id"] = self.chunk_id
        if self.source_span is not None:
            out["source_span"] = [self.source_span[0], self.source_span[1]]
        if self.snippet:
            out["snippet"] = self.snippet
        if self.extracted_at:
            out["extracted_at"] = self.extracted_at
        if self.extractor_version:
            out["extractor_version"] = self.extractor_version
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ProvenanceRecord:
        if not isinstance(data, dict):
            return cls()
        span_raw = data.get("source_span")
        span: tuple[int, int] | None = None
        if (
            isinstance(span_raw, (list, tuple))
            and len(span_raw) == 2
            and all(isinstance(v, int) for v in span_raw)
        ):
            span = (span_raw[0], span_raw[1])
        return cls(
            document_id=data.get("document_id")
            if isinstance(data.get("document_id"), str)
            else None,
            sentence_id=data.get("sentence_id")
            if isinstance(data.get("sentence_id"), str)
            else None,
            chunk_id=data.get("chunk_id") if isinstance(data.get("chunk_id"), str) else None,
            source_span=span,
            snippet=data.get("snippet") if isinstance(data.get("snippet"), str) else None,
            extracted_at=data.get("extracted_at")
            if isinstance(data.get("extracted_at"), str)
            else None,
            extractor_version=(
                data.get("extractor_version")
                if isinstance(data.get("extractor_version"), str)
                else None
            ),
        )


def _sentences(text: str) -> list[tuple[str, int, int, str]]:
    out: list[tuple[str, int, int, str]] = []
    for idx, match in enumerate(_SENTENCE_RE.finditer(text or "")):
        snippet = re.sub(r"\s+", " ", match.group(0)).strip()
        if snippet:
            out.append((f"s{idx}", match.start(), match.end(), snippet))
    return out


def find_text_provenance(
    text: str | None,
    terms: tuple[str, ...],
    *,
    document_id: str | None = None,
    chunk_id: str | None = None,
    extractor_version: str | None = None,
) -> ProvenanceRecord:
    """Return best-effort sentence/span provenance for terms in ``text``."""

    base = {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "extracted_at": _utc_now_iso(),
        "extractor_version": extractor_version,
    }
    if not text:
        return ProvenanceRecord(**base)

    lowered_terms = [t.lower().strip() for t in terms if t and t.strip()]
    best: tuple[str, int, int, str] | None = None
    for sent in _sentences(text):
        sent_l = sent[3].lower()
        if lowered_terms and all(term in sent_l for term in lowered_terms):
            best = sent
            break
        if best is None and any(term in sent_l for term in lowered_terms):
            best = sent

    if best is None:
        return ProvenanceRecord(**base)

    sentence_id, start, end, snippet = best
    return ProvenanceRecord(
        **base,
        sentence_id=sentence_id,
        source_span=(start, end),
        snippet=snippet,
    )


def provenance_from_metadata(metadata: dict[str, Any] | None) -> ProvenanceRecord:
    """Read structured provenance, falling back to legacy metadata keys."""

    meta = metadata or {}
    record = ProvenanceRecord.from_dict(meta.get("provenance"))
    if record.to_dict():
        return record
    source_ref = meta.get("source_ref")
    docs = meta.get("source_documents")
    document_id = source_ref if isinstance(source_ref, str) and source_ref else None
    if document_id is None and isinstance(docs, list):
        document_id = next((d for d in docs if isinstance(d, str) and d), None)
    evidence = meta.get("evidence")
    return ProvenanceRecord(
        document_id=document_id,
        snippet=evidence if isinstance(evidence, str) and evidence else None,
    )


def attach_provenance(
    metadata: dict[str, Any],
    record: ProvenanceRecord,
    *,
    preserve_legacy: bool = True,
) -> dict[str, Any]:
    """Return a copy of ``metadata`` with a structured provenance block."""

    out = dict(metadata)
    prov = record.to_dict()
    if prov:
        existing = out.get("provenance")
        if isinstance(existing, dict):
            merged = dict(existing)
            merged.update(prov)
            prov = merged
        out["provenance"] = prov

    if preserve_legacy and record.document_id:
        out.setdefault("source_ref", record.document_id)
        docs = list(out.get("source_documents", []) or [])
        if record.document_id not in docs:
            docs.append(record.document_id)
        out["source_documents"] = docs
    if preserve_legacy and record.snippet:
        out.setdefault("evidence", record.snippet)
    return out
