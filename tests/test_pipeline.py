"""
Tests for End-to-End Orchestrator Pipeline and Evaluation Metrics.
"""

import pytest

from nlp_engine.pipeline.orchestrator import NLPOrchestrator
from nlp_engine.utils.metrics import (
    calculate_classification_metrics,
    calculate_perplexity,
    calculate_bleu_score
)


def test_nlp_orchestrator(sample_tickets_df):
    orchestrator = NLPOrchestrator()
    orchestrator.fit(sample_tickets_df)

    test_ticket = "URGENT: Billed $49 twice for order ORD-12345 on 2026-03-22. Terrible customer support!"
    res = orchestrator.process_ticket(test_ticket)

    assert "cleaned_text" in res
    assert "triage" in res
    assert "sentiment" in res
    assert "entities" in res
    assert "bot_resolution" in res
    assert res["triage"]["urgency"] in ["Critical", "High", "Medium", "Low"]


def test_evaluation_metrics():
    y_true = ["A", "B", "A", "C"]
    y_pred = ["A", "B", "C", "C"]
    metrics = calculate_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 3

    ppl = calculate_perplexity(0.0)
    assert ppl == 1.0

    bleu = calculate_bleu_score(["a", "b", "c"], ["a", "b", "c"])
    assert abs(bleu - 1.0) < 1e-4
