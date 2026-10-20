"""Eval suite: leak rate is zero and the other metrics are meaningful."""

from enterprise_rag_governance.demo import GOLDEN


def test_eval_zero_leak_and_perfect_citations(system):
    report = system.evaluator.evaluate(GOLDEN, k=5).to_dict()
    assert report["unauthorized_leak_rate"] == 0.0
    assert report["citation_precision"] == 1.0
    assert report["question_count"] == len(GOLDEN)


def test_eval_recall_and_abstention(system):
    report = system.evaluator.evaluate(GOLDEN, k=5).to_dict()
    assert report["recall_at_k"] >= 0.8
    assert report["abstention_accuracy"] >= 0.8
    per_q = {r["question_id"]: r for r in report["results"]}
    assert per_q["g-revenue-denied"]["abstained"] is True
    assert per_q["g-revenue-denied"]["leaked"] is False
    assert per_q["g-revenue"]["recall_at_k"] == 1.0


def test_eval_empty_golden_set(system):
    report = system.evaluator.evaluate([], k=5).to_dict()
    assert report["question_count"] == 0
    assert report["unauthorized_leak_rate"] == 0.0
