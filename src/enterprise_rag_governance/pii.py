"""PII detection and redaction with an audit trail.

Patterns are deliberately conservative (they favour catching a likely
secret over preserving prose). The audit records *what kind* of PII was
found and how many times — never the matched values themselves, so the
audit log does not become a second copy of the data it protects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Order matters: more specific patterns run first so a credit-card
# number is not partially consumed by the phone pattern.
_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[REDACTED:email]"),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED:ssn]"),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[REDACTED:credit_card]"),
    ("phone", re.compile(r"\b(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b"),
     "[REDACTED:phone]"),
]


@dataclass
class RedactionFinding:
    kind: str
    count: int


@dataclass
class RedactionResult:
    text: str
    findings: list[RedactionFinding] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(f.count for f in self.findings)

    def counts(self) -> dict[str, int]:
        return {f.kind: f.count for f in self.findings}


def _looks_like_card(match: re.Match[str]) -> bool:
    digits = re.sub(r"\D", "", match.group(0))
    return 13 <= len(digits) <= 16


def redact(text: str) -> RedactionResult:
    """Return ``text`` with PII replaced, plus per-kind finding counts."""
    findings: list[RedactionFinding] = []
    out = text
    for kind, pattern, replacement in _PATTERNS:
        if kind == "credit_card":
            matches = [m for m in pattern.finditer(out) if _looks_like_card(m)]
            if matches:
                findings.append(RedactionFinding(kind, len(matches)))
                # Replace right-to-left so earlier offsets stay valid.
                for m in reversed(matches):
                    out = out[: m.start()] + replacement + out[m.end():]
            continue
        out, n = pattern.subn(replacement, out)
        if n:
            findings.append(RedactionFinding(kind, n))
    return RedactionResult(text=out, findings=findings)


def detect(text: str) -> dict[str, int]:
    """Count PII by kind without modifying the text."""
    return redact(text).counts()
