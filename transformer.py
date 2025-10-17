import torch
import torch.nn as nn

from layers.positional_embedding import PositionalEmbedding
from layers.swiglu import SwiGLU
from layers.attention import Attention
from layers.rmsnorm import RMSNorm


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        vocab_size: int,
        key_dim: int,
        num_heads: int,
        intermediate_dim: int,
        max_length: int,
        eps: float = 1e-6,
    ):

        super().__init__()
        self.num_layers = num_layers
        self.RMSNorm = RMSNorm(eps)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_embedding = PositionalEmbedding(hidden_dim, max_length)
        self.unembedding = nn.Linear(hidden_dim, vocab_size, bias=False)
        layers = []
        for _ in range(num_layers):
            layers.append(Attention(hidden_dim, key_dim, num_heads, eps))
            layers.append(SwiGLU(hidden_dim, intermediate_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: int) -> torch.Tensor:
        x = self.embedding(x) + self.positional_embedding(x)
        x = self.model(x)
        x = self.unembedding(x)
        return x
