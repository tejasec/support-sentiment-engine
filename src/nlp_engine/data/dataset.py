"""
Dataset loading and train/validation/test splitting utilities.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


def load_ticket_dataframe(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load ticket corpus from CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Ticket file not found: {path}")
    return pd.read_csv(path)


def split_tickets(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test subsets with reproducibility.
    """
    assert np.isclose(train_size + val_size + test_size, 1.0), "Splits must sum to 1.0"

    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * train_size)
    n_val = int(n * val_size)

    train_df = shuffled.iloc[:n_train].copy()
    val_df = shuffled.iloc[n_train:n_train + n_val].copy()
    test_df = shuffled.iloc[n_train + n_val:].copy()

    return train_df, val_df, test_df


class TicketDataset:
    """
    Generic PyTorch Dataset wrapper for ticket tokens or dense features.
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {
            "features": self.features[idx],
            "target": self.targets[idx]
        }
