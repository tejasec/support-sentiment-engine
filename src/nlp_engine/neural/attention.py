"""
Attention Mechanisms: Bahdanau (Additive) and Luong (Multiplicative).
"""

from typing import Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class BahdanauAttention(nn.Module):
        """
        Bahdanau (Additive) Attention:
        score(s_{t-1}, h_i) = v_a^T * tanh(W_a * s_{t-1} + U_a * h_i)
        """
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.U_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v_a = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, query: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # query: (batch_size, hidden_dim), values: (batch_size, seq_len, hidden_dim)
            query_proj = self.W_a(query).unsqueeze(1)  # (batch, 1, hidden)
            values_proj = self.U_a(values)             # (batch, seq_len, hidden)
            scores = self.v_a(torch.tanh(query_proj + values_proj)).squeeze(2)  # (batch, seq_len)
            weights = F.softmax(scores, dim=1)         # (batch, seq_len)
            context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)  # (batch, hidden)
            return context, weights


    class LuongAttention(nn.Module):
        """
        Luong (Multiplicative / General) Attention:
        score(s_t, h_i) = s_t^T * W_a * h_i
        """
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)

        def forward(self, query: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # query: (batch_size, hidden_dim), values: (batch_size, seq_len, hidden_dim)
            query_proj = self.W_a(query).unsqueeze(2)  # (batch, hidden, 1)
            scores = torch.bmm(values, query_proj).squeeze(2)  # (batch, seq_len)
            weights = F.softmax(scores, dim=1)
            context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
            return context, weights

else:
    class BahdanauAttention:
        pass
    class LuongAttention:
        pass
