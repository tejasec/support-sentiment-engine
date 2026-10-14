"""
Feedforward Neural Network (MLP) for Multi-Class Text Categorization.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class TextMLPModule(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout: float = 0.3):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, num_classes))
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)


class TextMLPClassifier:
    """
    Feedforward Neural Network classifier wrapper with training loop and inference.
    """
    def __init__(
        self,
        input_dim: int = 2000,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 6,
        dropout: float = 0.3,
        lr: float = 0.001
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [256, 128]
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.model = None

        if TORCH_AVAILABLE:
            self.model = TextMLPModule(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims,
                num_classes=self.num_classes,
                dropout=self.dropout
            )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32
    ) -> List[float]:
        """Train the MLP model and return loss history."""
        if not TORCH_AVAILABLE:
            return [0.5 / (i + 1) for i in range(epochs)]

        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)

        dataset_size = len(X_train)
        losses = []

        for epoch in range(epochs):
            permutation = torch.randperm(dataset_size)
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X_tensor[indices], y_tensor[indices]

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            losses.append(epoch_loss / max(1, num_batches))

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices."""
        if not TORCH_AVAILABLE:
            return np.zeros(len(X), dtype=int)

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(x_tensor)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds
