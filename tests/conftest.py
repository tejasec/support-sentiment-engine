"""
Pytest fixtures for nlp_engine tests.
"""

from typing import List, Tuple
import pytest
import pandas as pd

from nlp_engine.data.generator import generate_support_tickets


@pytest.fixture
def sample_corpus() -> List[str]:
    return [
        "Please refund the unexpected billing charge for order ORD-1002.",
        "The system crashed with error 503 and high memory lag on the dashboard.",
        "Where is my shipping package? Delivery is delayed since Monday.",
        "How can I reset my account password and update 2FA credentials?",
        "CLAIM YOUR FREE $1000 PRIZE NOW at http://spam.net/win"
    ]


@pytest.fixture
def sample_tickets_df() -> pd.DataFrame:
    return generate_support_tickets(n_samples=50, random_state=42)


@pytest.fixture
def sample_tagged_corpus() -> List[List[Tuple[str, str]]]:
    return [
        [("refund", "VERB"), ("the", "DET"), ("charge", "NOUN")],
        [("the", "DET"), ("charge", "NOUN"), ("failed", "VERB")],
        [("please", "ADV"), ("cancel", "VERB"), ("my", "PRON"), ("order", "NOUN")]
    ]
