"""A deterministic, offline hashing embedder.

Tokens are hashed into a fixed-width vector (with a bigram component so
word order contributes weakly) and L2-normalised. There is no model, no
network, and no API key: the same text always produces the same vector,
which is exactly what governance tests and evals need — relevance is
reproducible, and a regression is a code change, not model drift.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Stopwords are dropped so shared function words cannot manufacture a
# relevance score (or an answer) out of an unrelated document.
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "what", "who", "whom", "how", "many", "much", "do", "does", "did",
    "in", "on", "of", "for", "to", "and", "or", "at", "by", "with",
    "from", "that", "this", "it", "its", "as", "an", "when", "where",
})


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _bucket(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % dim


class HashingEmbedder:
    """Bag-of-words hashing embedder with cosine-ready output."""

    def __init__(self, dim: int = 256) -> None:
        if dim < 16:
            raise ValueError("dim must be >= 16")
        self.dim = dim

    def embed(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        for tok in tokens:
            vec[_bucket(tok, self.dim)] += 1.0
        for a, b in zip(tokens, tokens[1:]):
            vec[_bucket(f"{a}#{b}", self.dim)] += 0.5
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return tuple(vec)
        return tuple(v / norm for v in vec)


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))
