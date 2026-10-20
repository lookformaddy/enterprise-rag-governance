"""Golden-question evaluation for governed retrieval.

Metrics, all computed per golden question and averaged:

- **recall@k** — fraction of the expected (authorized) documents that
  retrieval returned in the top k.
- **unauthorized-leak rate** — fraction of questions where *any*
  returned chunk belongs to a document the principal may not read.
  Governance bar: exactly 0. Anything else is a release blocker.
- **citation precision** — fraction of answer citations that point at
  a chunk that was actually retrieved and is authorized.
- **abstention correctness** — whether the answerer abstained exactly
  when the golden question says it should have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .answerer import ExtractiveAnswerer
from .models import Principal
from .retrieval import PermissionAwareRetriever
from .store import DocumentStore


@dataclass
class GoldenQuestion:
    question_id: str
    query: str
    principal: Principal
    expected_doc_ids: set[str] = field(default_factory=set)
    should_abstain: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenQuestion:
        return cls(
            question_id=str(data["question_id"]),
            query=str(data["query"]),
            principal=Principal.from_dict(data["principal"]),
            expected_doc_ids=set(data.get("expected_doc_ids", [])),
            should_abstain=bool(data.get("should_abstain", False)),
        )


@dataclass
class QuestionResult:
    question_id: str
    recall_at_k: float
    leaked: bool
    citation_precision: float
    abstention_correct: bool
    retrieved_doc_ids: list[str]
    abstained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "recall_at_k": round(self.recall_at_k, 6),
            "leaked": self.leaked,
            "citation_precision": round(self.citation_precision, 6),
            "abstention_correct": self.abstention_correct,
            "retrieved_doc_ids": list(self.retrieved_doc_ids),
            "abstained": self.abstained,
        }


@dataclass
class EvalReport:
    k: int
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def recall_at_k(self) -> float:
        return _mean([r.recall_at_k for r in self.results])

    @property
    def unauthorized_leak_rate(self) -> float:
        return _mean([1.0 if r.leaked else 0.0 for r in self.results])

    @property
    def citation_precision(self) -> float:
        return _mean([r.citation_precision for r in self.results])

    @property
    def abstention_accuracy(self) -> float:
        return _mean([1.0 if r.abstention_correct else 0.0 for r in self.results])

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "question_count": len(self.results),
            "recall_at_k": round(self.recall_at_k, 6),
            "unauthorized_leak_rate": round(self.unauthorized_leak_rate, 6),
            "citation_precision": round(self.citation_precision, 6),
            "abstention_accuracy": round(self.abstention_accuracy, 6),
            "results": [r.to_dict() for r in self.results],
        }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class Evaluator:
    def __init__(
        self,
        store: DocumentStore,
        retriever: PermissionAwareRetriever,
        answerer: ExtractiveAnswerer | None = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.answerer = answerer or ExtractiveAnswerer()

    def evaluate(self, questions: list[GoldenQuestion], *, k: int = 5) -> EvalReport:
        report = EvalReport(k=k)
        for q in questions:
            retrieval = self.retriever.retrieve(q.principal, q.query, k=k)
            answer = self.answerer.answer(q.query, retrieval)
            retrieved = retrieval.doc_ids
            if q.expected_doc_ids:
                recall = len(set(retrieved) & q.expected_doc_ids) / len(q.expected_doc_ids)
            else:
                recall = 1.0 if not retrieved else 0.0
            leaked = any(
                not self.store.is_authorized(r.chunk.doc_id, q.principal)
                for r in retrieval.results
            )
            retrieved_chunk_ids = {r.chunk.chunk_id for r in retrieval.results}
            if answer.citations:
                good = sum(
                    1 for c in answer.citations
                    if c.chunk_id in retrieved_chunk_ids
                    and self.store.is_authorized(c.doc_id, q.principal)
                )
                precision = good / len(answer.citations)
            else:
                precision = 1.0  # an abstention makes no citation claims
            report.results.append(QuestionResult(
                question_id=q.question_id, recall_at_k=recall, leaked=leaked,
                citation_precision=precision,
                abstention_correct=(answer.abstained == q.should_abstain),
                retrieved_doc_ids=retrieved, abstained=answer.abstained,
            ))
        return report
