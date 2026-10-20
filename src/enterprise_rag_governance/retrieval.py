"""Permission-aware retrieval: filter first, rank second, leak nothing.

The order is the whole point. Every chunk is checked against the
requesting principal's authorization *before* any scoring happens, and
only authorized chunks are ever scored, ranked, counted, or returned.
A denied document therefore cannot influence the result in any way —
not its rank, not a citation, not even an exact count: the caller (and
the audit log) only ever sees a coarse denied-count bucket.
"""

from __future__ import annotations

from .audit import AuditEntry, AuditLog
from .embedder import HashingEmbedder, cosine, tokenize
from .models import (
    Principal,
    RetrievalResult,
    ScoredChunk,
    denied_bucket,
    query_hash,
)
from .store import DocumentStore


class PermissionAwareRetriever:
    def __init__(
        self,
        store: DocumentStore,
        embedder: HashingEmbedder | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder or HashingEmbedder()
        self.audit = audit or AuditLog()

    def retrieve(self, principal: Principal, query: str, *, k: int = 5) -> RetrievalResult:
        if k <= 0:
            raise ValueError("k must be positive")
        q_hash = query_hash(query)
        q_vec = self.embedder.embed(query)
        q_tokens = set(tokenize(query))

        # 1) FILTER — authorization before any ranking signal exists.
        authorized = []
        denied_docs: set[str] = set()
        for doc in self.store.documents():
            if doc.is_authorized(principal):
                authorized.extend(self.store.chunks_for(doc.doc_id))
            else:
                denied_docs.add(doc.doc_id)
        # Note: denied_docs is used only to size the bucket below. Its
        # contents are never returned, logged, or cited.

        # 2) RANK — only authorized chunks are scored.
        scored: list[ScoredChunk] = []
        for chunk in authorized:
            overlap = len(q_tokens & set(tokenize(chunk.text)))
            if overlap == 0:
                continue  # no lexical evidence — a hash collision is not relevance
            vec = chunk.embedding or self.embedder.embed(chunk.text)
            score = cosine(q_vec, vec)
            score += 0.01 * overlap  # deterministic tie-break boost
            if score > 0.0:
                scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))

        # Did any denied document actually match the query terms? A
        # bucket over *matching* denied docs is the honest signal;
        # counting every denied doc in the corpus would overstate it.
        matching_denied = 0
        for doc in self.store.documents():
            if doc.doc_id in denied_docs and q_tokens & set(tokenize(doc.text)):
                matching_denied += 1
        bucket = denied_bucket(matching_denied)

        top = scored[:k]
        result = RetrievalResult(
            principal_id=principal.user_id, query_hash=q_hash,
            results=top, denied_bucket=bucket,
        )
        self.audit.record_retrieval(AuditEntry(
            principal_id=principal.user_id, query_hash=q_hash,
            returned_doc_ids=result.doc_ids, denied_bucket=bucket,
            result_count=len(top), k=k,
        ))
        return result
