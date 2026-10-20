"""Retrieval: ranking quality and the filter-before-rank guarantees."""

DENIED_QUERY = "What revenue growth did the finance summary report?"


def test_authorized_query_ranks_expected_doc_first(system, bob):
    r = system.retriever.retrieve(bob, DENIED_QUERY, k=5)
    assert r.results
    assert r.results[0].chunk.doc_id == "finance-q3"
    scores = [x.score for x in r.results]
    assert scores == sorted(scores, reverse=True)


def test_denied_doc_never_returned(system, alice):
    r = system.retriever.retrieve(alice, DENIED_QUERY, k=5)
    assert all(x.chunk.doc_id != "finance-q3" for x in r.results)
    assert "finance-q3" not in r.doc_ids


def test_denied_doc_does_not_leak_via_counts_or_payload(system, alice):
    r = system.retriever.retrieve(alice, DENIED_QUERY, k=5)
    payload = str(r.to_dict())
    assert "finance-q3" not in payload
    assert "Q3 Finance Summary" not in payload
    # Only a bucket is exposed — never an exact denied count field.
    assert r.denied_bucket in {"0", "1-3", "4-10", "11+"}
    assert "denied_count" not in payload


def test_retrieval_is_deterministic(system, bob):
    a = system.retriever.retrieve(bob, DENIED_QUERY, k=5)
    b = system.retriever.retrieve(bob, DENIED_QUERY, k=5)
    assert [(x.chunk.chunk_id, x.score) for x in a.results] == \
           [(x.chunk.chunk_id, x.score) for x in b.results]


def test_k_limits_results(system, carol):
    r = system.retriever.retrieve(carol, "executive bonus targets revenue", k=1)
    assert len(r.results) <= 1


def test_unrelated_query_returns_nothing(system, alice):
    r = system.retriever.retrieve(alice, "photosynthesis chlorophyll zebra quantum", k=5)
    assert r.results == []


def test_public_principal_only_gets_public_docs(system, dave):
    r = system.retriever.retrieve(dave, "vacation holidays handbook engineer incident", k=10)
    assert {x.chunk.doc_id for x in r.results} <= {"handbook"}
