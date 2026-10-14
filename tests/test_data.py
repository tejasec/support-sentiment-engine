"""
Tests for data generation, ingestion, and splitting.
"""

import numpy as np
import pytest

from nlp_engine.data.generator import generate_support_tickets
from nlp_engine.data.dataset import split_tickets, TicketDataset


def test_generate_support_tickets():
    df = generate_support_tickets(n_samples=40, random_state=42, spam_ratio=0.1)
    assert len(df) == 40
    assert "ticket_id" in df.columns
    assert "category" in df.columns
    assert "urgency" in df.columns
    assert "sentiment" in df.columns
    assert "is_spam" in df.columns
    assert df["is_spam"].sum() > 0
    assert not df["text"].isnull().any()


def test_split_tickets(sample_tickets_df):
    train_df, val_df, test_df = split_tickets(sample_tickets_df, train_size=0.7, val_size=0.15, test_size=0.15)
    total = len(sample_tickets_df)
    assert len(train_df) == int(total * 0.7)
    assert len(val_df) == int(total * 0.15)
    assert len(test_df) == total - len(train_df) - len(val_df)


def test_ticket_dataset():
    x = np.random.randn(10, 5)
    y = np.array([0, 1] * 5)
    ds = TicketDataset(x, y)
    assert len(ds) == 10
    item = ds[0]
    assert "features" in item and "target" in item
    assert np.array_equal(item["features"], x[0])
