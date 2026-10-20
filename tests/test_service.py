"""FastAPI surface: health, ingest, query, audit, eval."""

from fastapi.testclient import TestClient

from enterprise_rag_governance.service import create_app

BOB = {"user_id": "bob", "groups": ["finance"], "clearance": "confidential"}
ALICE = {"user_id": "alice", "groups": ["engineering"], "clearance": "internal"}
Q = "What revenue growth did the finance summary report?"


def make_client():
    return TestClient(create_app())


def _ingest(client, **over):
    body = {"doc_id": "finance-q3", "title": "Q3 Finance Summary",
            "text": ("The quarterly finance summary reports revenue growth "
                     "of twelve percent. Operating margin improved."),
            "owner": "bob", "sensitivity": "confidential", "groups": ["finance"]}
    body.update(over)
    return client.post("/documents", json=body)


def test_health():
    assert make_client().get("/health").json()["status"] == "ok"


def test_ingest_and_authorized_query():
    client = make_client()
    resp = _ingest(client)
    assert resp.status_code == 201 and resp.json()["chunk_count"] >= 1
    out = client.post("/query", json={"principal": BOB, "query": Q}).json()
    assert out["answer"]["abstained"] is False
    assert out["answer"]["citations"][0]["doc_id"] == "finance-q3"


def test_unauthorized_query_abstains_via_api():
    client = make_client()
    _ingest(client)
    out = client.post("/query", json={"principal": ALICE, "query": Q}).json()
    assert out["answer"]["abstained"] is True
    assert out["answer"]["citations"] == []
    assert "finance-q3" not in str(out["retrieval"])


def test_duplicate_ingest_is_409():
    client = make_client()
    assert _ingest(client).status_code == 201
    assert _ingest(client).status_code == 409


def test_ingest_redacts_pii_via_api():
    client = make_client()
    resp = _ingest(client, doc_id="note", sensitivity="internal", groups=[],
                   owner="alice", text="Contact helpdesk@example.com or 415-555-0132.")
    assert resp.status_code == 201
    assert resp.json()["redaction_total"] >= 2


def test_audit_and_eval_endpoints():
    client = make_client()
    _ingest(client)
    client.post("/query", json={"principal": BOB, "query": Q})
    audit = client.get("/audit").json()
    assert len(audit["retrievals"]) == 1
    golden = [{"question_id": "g1", "query": Q, "principal": BOB,
               "expected_doc_ids": ["finance-q3"]},
              {"question_id": "g2", "query": Q, "principal": ALICE,
               "expected_doc_ids": [], "should_abstain": True}]
    report = client.post("/eval", json={"golden": golden}).json()
    assert report["unauthorized_leak_rate"] == 0.0
    assert report["question_count"] == 2
