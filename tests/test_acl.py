"""ACL filtering: clearance, groups, ownership, and explicit grants."""

from enterprise_rag_governance.demo import build_system
from enterprise_rag_governance.models import Document, Principal, Sensitivity


def _doc(**kw):
    base = dict(doc_id="d1", title="T", text="alpha beta gamma", owner="owner1")
    base.update(kw)
    return Document(**base)


def test_public_doc_readable_by_public_principal(dave):
    assert _doc(sensitivity=Sensitivity.PUBLIC).is_authorized(dave)


def test_internal_doc_denied_to_public_principal(dave, alice):
    doc = _doc(sensitivity=Sensitivity.INTERNAL)
    assert not doc.is_authorized(dave)
    assert doc.is_authorized(alice)


def test_confidential_requires_group_even_with_clearance():
    doc = _doc(sensitivity=Sensitivity.CONFIDENTIAL, groups=("finance",))
    outsider = Principal("mallory", groups=("legal",), clearance=Sensitivity.CONFIDENTIAL)
    insider = Principal("bob", groups=("finance",), clearance=Sensitivity.CONFIDENTIAL)
    assert not doc.is_authorized(outsider)
    assert doc.is_authorized(insider)


def test_confidential_group_without_clearance_is_denied():
    doc = _doc(sensitivity=Sensitivity.CONFIDENTIAL, groups=("finance",))
    low = Principal("intern", groups=("finance",), clearance=Sensitivity.INTERNAL)
    assert not doc.is_authorized(low)


def test_owner_and_named_user_grants():
    doc = _doc(sensitivity=Sensitivity.RESTRICTED, groups=("executives",))
    owner = Principal("owner1", clearance=Sensitivity.RESTRICTED)
    granted = Principal("delegate", clearance=Sensitivity.RESTRICTED)
    doc.allowed_users = ("delegate",)
    assert doc.is_authorized(owner)
    assert doc.is_authorized(granted)


def test_restricted_denied_to_confidential_finance(bob):
    doc = _doc(sensitivity=Sensitivity.RESTRICTED, groups=("executives",))
    assert not doc.is_authorized(bob)


def test_store_authorized_documents(system, alice, bob):
    alice_docs = {d.doc_id for d in system.store.authorized_documents(alice)}
    bob_docs = {d.doc_id for d in system.store.authorized_documents(bob)}
    assert "finance-q3" not in alice_docs and "finance-q3" in bob_docs
    assert "exec-comp" not in bob_docs
    assert "handbook" in alice_docs and "handbook" in bob_docs
