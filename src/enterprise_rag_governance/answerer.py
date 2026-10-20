"""A deterministic extractive answerer over authorized chunks only.

There is no LLM. The answerer selects, from the retrieved chunks, the
sentences that best cover the query terms and composes the answer from
those sentences verbatim — so every sentence in the answer is traceable
to exactly one citation, and nothing can be "generated" that was not in
an authorized chunk. With no usable evidence it abstains instead of
guessing; that refusal is a feature, not a failure mode.
"""

from __future__ import annotations

import re

from .embedder import tokenize
from .models import Answer, Citation, RetrievalResult

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")

ABSTAIN_TEXT = (
    "I don't have authorized evidence to answer that question. "
    "No accessible source supports an answer, so I am abstaining "
    "rather than guessing."
)


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Return ``(sentence, start, end)`` triples with offsets in ``text``."""
    out: list[tuple[str, int, int]] = []
    for m in _SENTENCE_RE.finditer(text):
        raw = m.group(0)
        sentence = raw.strip()
        if not sentence:
            continue
        start = m.start() + (len(raw) - len(raw.lstrip()))
        out.append((sentence, start, start + len(sentence)))
    return out


class ExtractiveAnswerer:
    def __init__(self, *, min_score: float = 0.05, max_sentences: int = 3) -> None:
        self.min_score = min_score
        self.max_sentences = max_sentences

    def answer(self, query: str, retrieval: RetrievalResult) -> Answer:
        usable = [r for r in retrieval.results if r.score >= self.min_score]
        if not usable:
            return Answer(text=ABSTAIN_TEXT, abstained=True,
                          query_hash=retrieval.query_hash)
        q_tokens = set(tokenize(query))
        candidates: list[tuple[int, int, str, Citation]] = []
        seen: set[str] = set()
        for rank, scored in enumerate(usable):
            chunk = scored.chunk
            for sentence, s_off, e_off in split_sentences(chunk.text):
                key = sentence.lower()
                if key in seen:
                    continue
                seen.add(key)
                overlap = len(q_tokens & set(tokenize(sentence)))
                candidates.append((overlap, -rank, sentence, Citation(
                    doc_id=chunk.doc_id, chunk_id=chunk.chunk_id,
                    start_char=chunk.start_char + s_off,
                    end_char=chunk.start_char + e_off,
                )))
        if not candidates:
            return Answer(text=ABSTAIN_TEXT, abstained=True,
                          query_hash=retrieval.query_hash)
        # Prefer query-covering sentences; keep at least the top-ranked
        # chunk's best sentence even with zero token overlap.
        candidates.sort(key=lambda c: (-c[0], c[1]))
        chosen = [c for c in candidates if c[0] > 0][: self.max_sentences]
        if not chosen:
            chosen = candidates[:1]
        text = " ".join(c[2] for c in chosen)
        return Answer(text=text, abstained=False,
                      citations=[c[3] for c in chosen],
                      query_hash=retrieval.query_hash)
