"""
Tests for Probabilistic Models & Sequence Tagging (N-gram, HMM, Naive Bayes).
"""

import numpy as np
import pytest

from nlp_engine.probabilistic.ngram import NGramLanguageModel
from nlp_engine.probabilistic.hmm import HiddenMarkovModelTagger
from nlp_engine.probabilistic.classifier import TicketNaiveBayes


def test_ngram_language_model():
    corpus = [
        ["please", "refund", "my", "charge"],
        ["cancel", "my", "account", "please"],
        ["help", "with", "my", "order"]
    ]
    lm = NGramLanguageModel(n=2, alpha=1.0)
    lm.fit(corpus)

    prob = lm.conditional_prob("refund", ("please",))
    assert 0.0 < prob <= 1.0

    ppl = lm.calculate_sentence_perplexity(["please", "refund", "my", "charge"])
    assert ppl > 0.0
    assert not np.isnan(ppl)

    gen = lm.generate(prompt_tokens=["please"], max_length=5)
    assert isinstance(gen, list)


def test_hmm_pos_tagger(sample_tagged_corpus):
    hmm = HiddenMarkovModelTagger()
    hmm.fit(sample_tagged_corpus)

    test_tokens = ["please", "cancel", "my", "order"]
    tagged = hmm.viterbi(test_tokens)
    assert len(tagged) == len(test_tokens)
    for word, tag in tagged:
        assert isinstance(word, str)
        assert isinstance(tag, str)


def test_naive_bayes_classifier():
    X = np.array([
        [2, 1, 0, 0],
        [3, 2, 0, 0],
        [0, 0, 3, 2],
        [0, 0, 2, 4]
    ])
    y = np.array(["Billing", "Billing", "Tech", "Tech"])

    nb = TicketNaiveBayes(alpha=1.0)
    nb.fit(X, y)

    preds = nb.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset(set(y))

    probs = nb.predict_proba(X)
    assert probs.shape == (4, 2)
    # Probabilities must sum to 1
    assert np.allclose(np.sum(probs, axis=1), 1.0)
