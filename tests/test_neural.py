"""
Tests for Deep Learning in NLP (MLP, RNN, Attention, Transformer).
"""

import numpy as np
import pytest

from nlp_engine.neural.mlp import TextMLPClassifier
from nlp_engine.neural.rnn import SequenceRNN
from nlp_engine.neural.attention import BahdanauAttention
from nlp_engine.neural.transformer import TransformerClassifier


def test_text_mlp_classifier():
    X = np.random.randn(20, 10).astype("float32")
    y = np.random.randint(0, 3, size=20)

    mlp = TextMLPClassifier(input_dim=10, hidden_dims=[16], num_classes=3)
    losses = mlp.fit(X, y, epochs=2, batch_size=10)
    assert len(losses) == 2

    preds = mlp.predict(X)
    assert len(preds) == 20


def test_sequence_rnn_vanishing_gradient():
    seq_rnn = SequenceRNN(vocab_size=200, hidden_dim=32, num_classes=3)
    diag = seq_rnn.diagnose_vanishing_gradient(sequence_length=15)
    assert "vanilla_rnn_grad_norm" in diag
    assert "lstm_grad_norm" in diag
    assert diag["lstm_grad_norm"] >= 0.0


def test_transformer_classifier():
    try:
        import torch
        transformer = TransformerClassifier(vocab_size=100, d_model=32, nhead=2, num_classes=3)
        dummy_input = torch.randint(1, 100, (4, 10))
        logits = transformer(dummy_input)
        assert logits.shape == (4, 3)
    except ImportError:
        pass
