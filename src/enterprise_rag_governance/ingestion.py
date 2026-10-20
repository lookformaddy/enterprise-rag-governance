"""Ingestion: redact PII, chunk, embed, and store — in that order.

Redaction happens *before* chunking and embedding, so no chunk, vector,
citation, or answer can ever contain the original PII. The stored
document text is the redacted text, which keeps chunk offsets and
citations consistent with what a reader is allowed to see.
"""

from __future__ import annotations

from dataclasses import dataclass

from .audit import AuditLog, RedactionAuditEntry
from .chunking import chunk_text
from .embedder import HashingEmbedder
from .models import Document
from .pii import redact
from .store import DocumentStore


@dataclass
class IngestResult:
    doc_id: str
    chunk_count: int
    redactions: dict[str, int]
    redaction_total: int

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "chunk_count": self.chunk_count,
                "redactions": dict(self.redactions),
                "redaction_total": self.redaction_total}


class IngestionPipeline:
    def __init__(
        self,
        store: DocumentStore,
        embedder: HashingEmbedder | None = None,
        audit: AuditLog | None = None,
        *,
        chunk_chars: int = 600,
        overlap: int = 80,
    ) -> None:
        self.store = store
        self.embedder = embedder or HashingEmbedder()
        self.audit = audit or AuditLog()
        self.chunk_chars = chunk_chars
        self.overlap = overlap

    def ingest(self, doc: Document) -> IngestResult:
        result = redact(doc.text)
        stored = Document(
            doc_id=doc.doc_id, title=doc.title, text=result.text,
            owner=doc.owner, sensitivity=doc.sensitivity,
            groups=tuple(doc.groups), allowed_users=tuple(doc.allowed_users),
            metadata=dict(doc.metadata),
        )
        chunks = chunk_text(stored.doc_id, stored.text,
                            chunk_chars=self.chunk_chars, overlap=self.overlap)
        for chunk in chunks:
            chunk.embedding = self.embedder.embed(chunk.text)
        self.store.add(stored, chunks)
        counts = result.counts()
        if result.total:
            self.audit.record_redaction(RedactionAuditEntry(
                doc_id=doc.doc_id, findings=counts, total=result.total))
        return IngestResult(doc_id=doc.doc_id, chunk_count=len(chunks),
                            redactions=counts, redaction_total=result.total)
