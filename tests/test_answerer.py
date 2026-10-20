"""Citations, extractive answering, and abstention."""

Q = "What revenue growth did the finance summary report?"


def _answer(system, principal, query):
    return system.answerer.answer(query, system.retriever.retrieve(principal, query))


def test_authorized_answer_has_correct_citations(system, bob):
    ans = _answer(system, bob, Q)
    assert not ans.abstained
    assert ans.citations
    for cit in ans.citations:
        stored = system.store.get(cit.doc_id)
        # The citation offsets must address real text in the stored doc,
        # and that text must appear verbatim in the answer.
        assert stored.text[cit.start_char:cit.end_char] in ans.text
        chunk = [c for c in system.store.chunks_for(cit.doc_id)
                 if c.chunk_id == cit.chunk_id]
        assert chunk, f"citation points at unknown chunk {cit.chunk_id}"


def test_answer_text_comes_only_from_authorized_chunks(system, bob):
    ans = _answer(system, bob, Q)
    assert all(c.doc_id == "finance-q3" for c in ans.citations)
    assert "revenue" in ans.text.lower()


def test_abstains_for_unauthorized_evidence(system, alice):
    ans = _answer(system, alice, Q)
    assert ans.abstained
    assert ans.citations == []
    assert "twelve" not in ans.text  # the denied fact must not appear


def test_abstains_when_no_evidence_exists(system, carol):
    ans = _answer(system, carol, "photosynthesis chlorophyll zebra quantum")
    assert ans.abstained and ans.citations == []


def test_abstention_identical_for_denied_and_absent(system, alice, carol):
    denied = _answer(system, alice, Q)
    absent = _answer(system, carol, "photosynthesis chlorophyll zebra quantum")
    assert denied.text == absent.text  # no oracle via refusal wording


def test_vacation_answer_for_internal_user(system, alice):
    ans = _answer(system, alice, "How many days of vacation do employees receive?")
    assert not ans.abstained
    assert any(c.doc_id == "handbook" for c in ans.citations)
    assert "twenty" in ans.text.lower()
