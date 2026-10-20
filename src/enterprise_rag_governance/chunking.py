"""Deterministic text chunking with character offsets.

Chunks are windows over the document text, snapped forward to a
sentence/paragraph boundary where one is nearby, so a chunk rarely
starts mid-sentence. Offsets always refer to the stored (redacted)
document text — that is what citations point at.
"""

from __future__ import annotations

from .models import Chunk

_BOUNDARIES = ("\n\n", ". ", ".\n", "\n", " ")


def chunk_text(
    doc_id: str,
    text: str,
    *,
    chunk_chars: int = 600,
    overlap: int = 80,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks with exact offsets."""
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap < 0 or overlap >= chunk_chars:
        raise ValueError("overlap must be in [0, chunk_chars)")
    chunks: list[Chunk] = []
    n = len(text)
    start = 0
    index = 0
    while start < n:
        end = min(n, start + chunk_chars)
        if end < n:
            end = _snap(text, start, end)
        piece = text[start:end]
        if piece.strip():
            chunks.append(Chunk(
                chunk_id=f"{doc_id}#c{index}", doc_id=doc_id, index=index,
                text=piece, start_char=start, end_char=end,
            ))
            index += 1
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _snap(text: str, start: int, end: int) -> int:
    """Move ``end`` to the latest boundary in the second half of the window."""
    floor = start + (end - start) // 2
    best = -1
    for sep in _BOUNDARIES:
        pos = text.rfind(sep, floor, end)
        if pos > best:
            best = pos + (2 if sep in ("\n\n", ". ") else 1)
    return best if best > floor else end
