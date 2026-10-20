"""Core typed models: principals, documents, chunks, citations, answers."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any


class Sensitivity(IntEnum):
    """Ordered sensitivity levels; a higher value is more sensitive."""

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3

    @classmethod
    def parse(cls, value: str | Sensitivity) -> Sensitivity:
        if isinstance(value, cls):
            return value
        try:
            return cls[value.strip().upper()]
        except KeyError as exc:
            raise ValueError(f"unknown sensitivity level: {value!r}") from exc


@dataclass(frozen=True)
class Principal:
    """Who is asking: a user, their groups, and their clearance level."""

    user_id: str
    groups: tuple[str, ...] = ()
    clearance: Sensitivity = Sensitivity.INTERNAL

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Principal:
        return cls(
            user_id=str(data["user_id"]),
            groups=tuple(str(g) for g in data.get("groups", ())),
            clearance=Sensitivity.parse(data.get("clearance", "internal")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"user_id": self.user_id, "groups": list(self.groups),
                "clearance": self.clearance.name.lower()}


@dataclass
class Document:
    """A stored source document with its access-control list."""

    doc_id: str
    title: str
    text: str
    owner: str
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    groups: tuple[str, ...] = ()
    allowed_users: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_authorized(self, principal: Principal) -> bool:
        """The single authorization rule for the whole platform.

        1. Clearance is necessary: the principal's clearance must be at
           or above the document's sensitivity.
        2. ``public`` and ``internal`` documents then need nothing more
           (any principal at that clearance may read them).
        3. ``confidential`` and ``restricted`` documents additionally
           require ownership, an explicit user grant, or a shared group.
        """
        if principal.clearance < self.sensitivity:
            return False
        if self.sensitivity <= Sensitivity.INTERNAL:
            return True
        return (
            principal.user_id == self.owner
            or principal.user_id in self.allowed_users
            or bool(set(principal.groups) & set(self.groups))
        )

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "doc_id": self.doc_id, "title": self.title, "owner": self.owner,
            "sensitivity": self.sensitivity.name.lower(),
            "groups": list(self.groups),
            "allowed_users": list(self.allowed_users),
            "metadata": dict(self.metadata),
        }
        if include_text:
            d["text"] = self.text
        return d


@dataclass
class Chunk:
    """One retrievable slice of a (redacted) document."""

    chunk_id: str
    doc_id: str
    index: int
    text: str
    start_char: int
    end_char: int
    embedding: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("embedding", None)
        return d


@dataclass(frozen=True)
class Citation:
    """A pointer from an answer sentence back to its source chunk."""

    doc_id: str
    chunk_id: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"chunk": self.chunk.to_dict(), "score": round(self.score, 6)}


@dataclass
class Answer:
    """An extractive answer, or an abstention when evidence is absent."""

    text: str
    abstained: bool
    citations: list[Citation] = field(default_factory=list)
    query_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "abstained": self.abstained,
                "citations": [c.to_dict() for c in self.citations],
                "query_hash": self.query_hash}


def query_hash(query: str) -> str:
    """Stable hash of a query — audit logs store this, never the text."""
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:16]


def denied_bucket(count: int) -> str:
    """Bucket a denied count so its exact value cannot leak information.

    Knowing that *exactly one* restricted document matched a query is
    itself a disclosure; a coarse bucket (``"1-3"``) is not.
    """
    if count <= 0:
        return "0"
    if count <= 3:
        return "1-3"
    if count <= 10:
        return "4-10"
    return "11+"


@dataclass
class RetrievalResult:
    """What retrieval may expose: authorized chunks only, plus a bucket."""

    principal_id: str
    query_hash: str
    results: list[ScoredChunk] = field(default_factory=list)
    denied_bucket: str = "0"
    created_at: float = field(default_factory=time.time)

    @property
    def doc_ids(self) -> list[str]:
        seen: list[str] = []
        for r in self.results:
            if r.chunk.doc_id not in seen:
                seen.append(r.chunk.doc_id)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "principal_id": self.principal_id,
            "query_hash": self.query_hash,
            "results": [r.to_dict() for r in self.results],
            "denied_bucket": self.denied_bucket,
        }
