"""
Transformer Architecture: Multi-Head Self-Attention, Positional Encoding, and Encoder Block.
"""

import math
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
    class PositionalEncoding(nn.Module):
        """
        Sinusoidal Positional Encoding (Vaswani et al., 2017).
        """
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch_size, seq_len, d_model)
            return x + self.pe[:, :x.size(1), :]


    class MultiHeadAttention(nn.Module):
        """
        Multi-Head Self-Attention Mechanism with scaled dot-product.
        """
        def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
            super().__init__()
            assert d_model % nhead == 0, "d_model must be divisible by nhead"
            self.d_model = d_model
            self.nhead = nhead
            self.d_k = d_model // nhead

            self.q_linear = nn.Linear(d_model, d_model)
            self.k_linear = nn.Linear(d_model, d_model)
            self.v_linear = nn.Linear(d_model, d_model)
            self.out_linear = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            batch_size = q.size(0)

            # Linear projections and reshape to (batch, nhead, seq_len, d_k)
            Q = self.q_linear(q).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
            K = self.k_linear(k).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
            V = self.v_linear(v).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

            # Scaled Dot-Product Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context = torch.matmul(attn_weights, V)
            context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            output = self.out_linear(context)
            return output, attn_weights


    class TransformerClassifier(nn.Module):
        """
        Transformer Encoder Classifier with self-attention, feedforward, and pooling.
        """
        def __init__(
            self,
            vocab_size: int = 5000,
            d_model: int = 128,
            nhead: int = 4,
            dim_feedforward: int = 256,
            num_classes: int = 6,
            dropout: float = 0.1
        ):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos_encoder = PositionalEncoding(d_model)
            self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

            self.ff = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model)
            )
            self.classifier = nn.Linear(d_model, num_classes)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            embedded = self.dropout(self.pos_encoder(self.embedding(x)))
            attn_out, _ = self.self_attn(embedded, embedded, embedded)
            x_norm = self.norm1(embedded + attn_out)
            ff_out = self.ff(x_norm)
            out = self.norm2(x_norm + ff_out)
            # Global Average Pooling over sequence length
            pooled = torch.mean(out, dim=1)
            logits = self.classifier(pooled)
            return logits

else:
    class PositionalEncoding:
        pass
    class MultiHeadAttention:
        pass
    class TransformerClassifier:
        pass
