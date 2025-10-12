import torch
import torch.nn as nn

from layers.linear import Linear
from layers.positional_embedding import PositionalEmbedding
from layers.swiglu import SwiGLU
from layers.attention import Attention
from layers.layernorm import LayerNorm
from layers.moe import MoE
from layers.positional_embedding import PositionalEmbedding


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        vocab_size: int,
        key_dim: int,
        num_heads: int,
        intermediate_dim: int,
        num_selected: int,
        num_experts: int,
        max_length: int,
        eps: float = 1e-6,
    ):

        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.RMSNorm = LayerNorm(eps)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_embedding = nn.Embedding(max_length, hidden_dim)
        self.unembedding = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.num_selected = num_selected
        self.num_experts = num_experts
        layers = []
        for i in range(num_layers):
            layers.append(Attention(hidden_dim, key_dim, num_heads, eps))
            layers.append(SwiGLU(hidden_dim, intermediate_dim))
            if i % 3 == 0:
                layers.append(
                    MoE(num_experts, hidden_dim, intermediate_dim, num_selected)
                )
        self.model = nn.Sequential(*layers)

    def forward(self, x: int) -> torch.Tensor:
        x = self.embedding(x) + self.positional_embedding(
            torch.arange(x.size(1), device=x.device)
        )
        x = self.model(x)
        x = self.unembedding(x)
        return x
