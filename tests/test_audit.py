"""Audit log: complete for retrievals, silent about everything denied."""

from enterprise_rag_governance.models import query_hash

Q = "What revenue growth did the finance summary report?"


def test_retrieval_is_audited_with_hash_not_text(system, bob):
    system.retriever.retrieve(bob, Q, k=5)
    entries = system.audit.for_principal("bob")
    assert entries
    entry = entries[-1]
    assert entry.query_hash == query_hash(Q)
    assert entry.returned_doc_ids == ["finance-q3"]
    payload = str(entry.to_dict())
    assert Q not in payload  # raw query text is never stored


def test_denied_retrieval_audit_leaks_nothing(system, alice):
    system.retriever.retrieve(alice, Q, k=5)
    entry = system.audit.for_principal("alice")[-1]
    payload = str(entry.to_dict())
    assert "finance-q3" not in payload
    assert "Finance" not in payload
    assert entry.denied_bucket in {"0", "1-3", "4-10", "11+"}
    assert "finance-q3" not in entry.returned_doc_ids


def test_every_retrieval_appends_one_entry(system, alice, bob):
    before = len(system.audit.entries())
    system.retriever.retrieve(alice, "vacation policy")
    system.retriever.retrieve(bob, "revenue growth")
    assert len(system.audit.entries()) == before + 2
