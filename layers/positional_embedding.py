import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, max_length: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, hidden_dim)

    def forward(self, x: int) -> torch.Tensor:
        return self.embedding(x)
