"""Embedder determinism — the property eval reproducibility rests on."""

from enterprise_rag_governance.embedder import HashingEmbedder, cosine, tokenize


def test_same_text_same_vector():
    e = HashingEmbedder()
    assert e.embed("revenue growth summary") == e.embed("revenue growth summary")


def test_similar_text_scores_higher_than_unrelated():
    e = HashingEmbedder()
    q = e.embed("revenue growth finance summary")
    near = e.embed("the finance summary reports revenue growth")
    far = e.embed("vacation handbook holidays office")
    assert cosine(q, near) > cosine(q, far)


def test_stopwords_do_not_create_similarity():
    e = HashingEmbedder()
    a = e.embed("what is the revenue")
    b = e.embed("what is the vacation")
    assert cosine(a, b) < 0.5
    assert tokenize("What is the revenue?") == ["revenue"]
