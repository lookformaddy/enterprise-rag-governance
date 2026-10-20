"""PII detection and redaction, including the ingestion-time guarantee."""

from enterprise_rag_governance.pii import detect, redact


def test_email_redacted():
    r = redact("Reach the desk at helpdesk@example.com today.")
    assert "helpdesk@example.com" not in r.text
    assert "[REDACTED:email]" in r.text
    assert r.counts() == {"email": 1}


def test_phone_redacted():
    r = redact("Call 415-555-0132 for support.")
    assert "415-555-0132" not in r.text
    assert r.counts().get("phone") == 1


def test_ssn_like_redacted():
    r = redact("Reference 123-45-6789 on file.")
    assert "123-45-6789" not in r.text
    assert r.counts().get("ssn") == 1


def test_credit_card_redacted_spaced_and_plain():
    r = redact("Card 4111 1111 1111 1111 and 5500005555555559.")
    assert "4111" not in r.text and "5500005555555559" not in r.text
    assert r.counts().get("credit_card") == 2


def test_clean_text_unchanged():
    text = "The handbook covers vacation policy and holidays."
    r = redact(text)
    assert r.text == text and r.total == 0


def test_detect_counts_without_modifying():
    counts = detect("mail a@b.com or call 415-555-0132")
    assert counts == {"email": 1, "phone": 1}


def test_ingestion_stores_only_redacted_text(system):
    stored = system.store.get("support-note")
    assert "4111" not in stored.text
    assert "123-45-6789" not in stored.text
    assert "helpdesk@example.com" not in stored.text
    for chunk in system.store.chunks_for("support-note"):
        assert "4111" not in chunk.text


def test_redaction_audit_records_kinds_not_values(system):
    entries = system.audit.redactions()
    assert entries, "expected a redaction audit entry for support-note"
    entry = [e for e in entries if e.doc_id == "support-note"][0]
    assert entry.total >= 3
    assert set(entry.findings) <= {"email", "phone", "ssn", "credit_card"}
    assert "4111" not in str(entry.to_dict())
