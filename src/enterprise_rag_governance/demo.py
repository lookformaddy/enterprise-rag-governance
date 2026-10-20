"""End-to-end offline demo: governed ingestion, retrieval, and eval.

Synthetic corpus, four principals with different clearances/groups.
The demo proves the governance claims in order: PII is redacted at
ingestion, an authorized reader gets a cited answer, an unauthorized
reader gets an abstention and no leak, and the eval suite reports a
zero unauthorized-leak rate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .answerer import ExtractiveAnswerer
from .audit import AuditLog
from .embedder import HashingEmbedder
from .evaluation import Evaluator, GoldenQuestion
from .ingestion import IngestionPipeline
from .models import Document, Principal, Sensitivity
from .retrieval import PermissionAwareRetriever
from .store import DocumentStore

ALICE = Principal("alice", groups=("engineering",), clearance=Sensitivity.INTERNAL)
BOB = Principal("bob", groups=("finance",), clearance=Sensitivity.CONFIDENTIAL)
CAROL = Principal("carol", groups=("executives",), clearance=Sensitivity.RESTRICTED)
DAVE = Principal("dave", groups=(), clearance=Sensitivity.PUBLIC)

DOCUMENTS = [
    Document(
        doc_id="handbook", title="Employee Handbook", owner="hr",
        sensitivity=Sensitivity.PUBLIC,
        text=("The employee handbook covers vacation policy and holidays. "
              "Employees receive twenty days of vacation each year. "
              "The office closes for public holidays listed in the handbook."),
    ),
    Document(
        doc_id="eng-runbook", title="Platform Runbook", owner="alice",
        sensitivity=Sensitivity.INTERNAL,
        text=("The platform runbook describes incident response for the "
              "data platform. The on-call engineer restarts the ingestion "
              "service first during an incident. Escalation goes to the "
              "platform team after thirty minutes."),
    ),
    Document(
        doc_id="finance-q3", title="Q3 Finance Summary", owner="bob",
        sensitivity=Sensitivity.CONFIDENTIAL, groups=("finance",),
        text=("The quarterly finance summary reports revenue growth of "
              "twelve percent. Operating margin improved because cloud "
              "spend fell. The finance team forecasts steady revenue "
              "growth next quarter."),
    ),
    Document(
        doc_id="exec-comp", title="Executive Compensation Notes", owner="carol",
        sensitivity=Sensitivity.RESTRICTED, groups=("executives",),
        text=("Executive compensation notes describe bonus targets for "
              "the leadership group. Bonus targets depend on company "
              "revenue and strategic milestones for the year."),
    ),
    Document(
        doc_id="support-note", title="Support Contact Note", owner="alice",
        sensitivity=Sensitivity.INTERNAL,
        text=("For urgent support, contact the desk at helpdesk@example.com "
              "or call 415-555-0132. A legacy record lists card "
              "4111 1111 1111 1111 and reference 123-45-6789 for a "
              "synthetic test account."),
    ),
]

GOLDEN = [
    GoldenQuestion("g-vacation", "How many days of vacation do employees receive?",
                   ALICE, {"handbook"}),
    GoldenQuestion("g-incident", "What does the on-call engineer restart during an incident?",
                   ALICE, {"eng-runbook"}),
    GoldenQuestion("g-revenue", "What revenue growth did the finance summary report?",
                   BOB, {"finance-q3"}),
    GoldenQuestion("g-revenue-denied", "What revenue growth did the finance summary report?",
                   ALICE, set(), should_abstain=True),
    GoldenQuestion("g-comp-denied", "What are the executive bonus targets?",
                   BOB, set(), should_abstain=True),
    GoldenQuestion("g-comp", "What do executive bonus targets depend on?",
                   CAROL, {"exec-comp"}),
]


@dataclass
class RagSystem:
    store: DocumentStore
    audit: AuditLog
    ingestion: IngestionPipeline
    retriever: PermissionAwareRetriever
    answerer: ExtractiveAnswerer
    evaluator: Evaluator


def build_system() -> RagSystem:
    store = DocumentStore()
    audit = AuditLog()
    embedder = HashingEmbedder()
    ingestion = IngestionPipeline(store, embedder, audit)
    retriever = PermissionAwareRetriever(store, embedder, audit)
    answerer = ExtractiveAnswerer()
    evaluator = Evaluator(store, retriever, answerer)
    return RagSystem(store, audit, ingestion, retriever, answerer, evaluator)


def build_demo_system() -> RagSystem:
    system = build_system()
    for doc in DOCUMENTS:
        system.ingestion.ingest(doc)
    return system


def run_demo(verbose: bool = True) -> dict:
    system = build_demo_system()

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"ingested {len(system.store)} documents")
    for entry in system.audit.redactions():
        log(f"  redaction audit: doc={entry.doc_id} findings={entry.findings}")

    stored_support = system.store.get("support-note")
    assert "4111" not in stored_support.text and "123-45-6789" not in stored_support.text
    log("PII check: stored support-note text contains no card / SSN values")

    q_bob = "What revenue growth did the finance summary report?"
    ans_bob = system.answerer.answer(q_bob, system.retriever.retrieve(BOB, q_bob))
    log(f"bob (finance, confidential): abstained={ans_bob.abstained}")
    log(f"  answer: {ans_bob.text}")
    log(f"  citations: {[c.to_dict() for c in ans_bob.citations]}")
    assert not ans_bob.abstained and ans_bob.citations
    assert all(c.doc_id == "finance-q3" for c in ans_bob.citations)

    ans_alice = system.answerer.answer(q_bob, system.retriever.retrieve(ALICE, q_bob))
    log(f"alice (engineering, internal) same question: abstained={ans_alice.abstained}")
    assert ans_alice.abstained and not ans_alice.citations
    assert "finance" not in ans_alice.text.lower() or "authorized" in ans_alice.text.lower()

    report = system.evaluator.evaluate(GOLDEN, k=5)
    summary = report.to_dict()
    log(f"eval: recall@5={summary['recall_at_k']} "
        f"leak_rate={summary['unauthorized_leak_rate']} "
        f"citation_precision={summary['citation_precision']} "
        f"abstention_accuracy={summary['abstention_accuracy']}")
    assert summary["unauthorized_leak_rate"] == 0.0
    assert summary["citation_precision"] == 1.0

    log(f"audit entries: {len(system.audit.entries())} retrievals logged "
        f"(query hashes only, denied counts bucketed)")
    return {"eval": summary, "audit_entries": len(system.audit.entries()),
            "bob_answer": ans_bob.to_dict(), "alice_answer": ans_alice.to_dict()}


if __name__ == "__main__":
    run_demo()
