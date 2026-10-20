"""Audit logging for retrievals and redactions.

Two invariants are enforced by construction:

- Query *text* is never stored — only its hash.
- Denied documents are never named — only a coarse count bucket is
  recorded, for retrievals and in the entries themselves.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AuditEntry:
    principal_id: str
    query_hash: str
    returned_doc_ids: list[str]
    denied_bucket: str
    result_count: int
    k: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RedactionAuditEntry:
    doc_id: str
    findings: dict[str, int]
    total: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLog:
    """Append-only, thread-safe audit log (optionally JSONL-backed)."""

    def __init__(self, path: str | None = None) -> None:
        self._entries: list[AuditEntry] = []
        self._redactions: list[RedactionAuditEntry] = []
        self._lock = threading.Lock()
        self._path = path

    def record_retrieval(self, entry: AuditEntry) -> AuditEntry:
        with self._lock:
            self._entries.append(entry)
            self._append({"type": "retrieval", **entry.to_dict()})
        return entry

    def record_redaction(self, entry: RedactionAuditEntry) -> RedactionAuditEntry:
        with self._lock:
            self._redactions.append(entry)
            self._append({"type": "redaction", **entry.to_dict()})
        return entry

    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    def redactions(self) -> list[RedactionAuditEntry]:
        return list(self._redactions)

    def for_principal(self, principal_id: str) -> list[AuditEntry]:
        return [e for e in self._entries if e.principal_id == principal_id]

    def _append(self, payload: dict[str, Any]) -> None:
        if self._path:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, sort_keys=True) + "\n")
