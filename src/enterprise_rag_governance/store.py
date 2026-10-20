"""In-memory document + chunk store. The ACL lives on the document."""

from __future__ import annotations

import threading

from .errors import DocumentNotFoundError, DuplicateDocumentError
from .models import Chunk, Document, Principal


class DocumentStore:
    """Thread-safe store keyed by document id, with chunk indexes."""

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
        self._chunks: dict[str, list[Chunk]] = {}
        self._lock = threading.Lock()

    def add(self, doc: Document, chunks: list[Chunk]) -> None:
        with self._lock:
            if doc.doc_id in self._docs:
                raise DuplicateDocumentError(f"document already exists: {doc.doc_id}")
            self._docs[doc.doc_id] = doc
            self._chunks[doc.doc_id] = list(chunks)

    def get(self, doc_id: str) -> Document:
        try:
            return self._docs[doc_id]
        except KeyError as exc:
            raise DocumentNotFoundError(f"no such document: {doc_id}") from exc

    def chunks_for(self, doc_id: str) -> list[Chunk]:
        return list(self._chunks.get(doc_id, []))

    def all_chunks(self) -> list[Chunk]:
        out: list[Chunk] = []
        for doc_id in self._docs:
            out.extend(self._chunks.get(doc_id, []))
        return out

    def documents(self) -> list[Document]:
        return list(self._docs.values())

    def authorized_documents(self, principal: Principal) -> list[Document]:
        return [d for d in self._docs.values() if d.is_authorized(principal)]

    def is_authorized(self, doc_id: str, principal: Principal) -> bool:
        doc = self._docs.get(doc_id)
        return bool(doc and doc.is_authorized(principal))

    def __len__(self) -> int:
        return len(self._docs)
