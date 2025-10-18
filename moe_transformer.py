import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Optional

from layers.swiglu import SwiGLU
from layers.attention import Attention
from layers.rmsnorm import RMSNorm
from layers.moe import MoE


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
        moe_layers: List[int],
        use_cache: bool,
        eps: float = 1e-6,
    ):

        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.rms_norm = RMSNorm(hidden_dim,eps)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.unembedding = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.num_selected = num_selected
        self.num_experts = num_experts
        self.layers = nn.ModuleList()
        self.use_cache = use_cache
        for i in range(num_layers):
            self.layers.append(Attention(hidden_dim, key_dim, num_heads, use_cache=use_cache))
            self.layers.append(SwiGLU(hidden_dim, intermediate_dim))
            if i in moe_layers:
                self.layers.append(
                    MoE(num_experts, hidden_dim, intermediate_dim, num_selected)
                )

    def forward(self, x: int, key_cache: Optional[Tensor] = None, value_cache: Optional[Tensor] = None,) -> torch.Tensor:
        #cache dim: l b s k
        if self.use_cache:
            output_key_cache = []
            output_value_cache = []
        x = self.embedding(x)
        layer_num = -1
        for layer in self.layers:
            if isinstance(layer, Attention):
                if self.use_cache:
                    layer_num += 1
                    x, (new_key_cache, new_value_cache) = layer(x,key_cache[layer_num],value_cache[layer_num])
                    output_key_cache.append(new_key_cache)
                    output_value_cache.append(new_value_cache)
                else:
                    x, _ = layer(x)
            else:
                x = layer(x)
        x = self.unembedding(x)
        return x, (output_key_cache, output_value_cache) if self.use_cache else None