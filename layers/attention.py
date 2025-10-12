import torch
import torch.nn as nn
from layers.layernorm import LayerNorm

class Attention(nn.Module):
    def __init__(
        self, 
        hidden_dim: int,
        key_dim: int,
        num_heads: int,
        eps: float=1e-5
    ): 
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.eps = eps
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.qkv = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(hidden_dim, key_dim*3))
        )
        self.o_proj = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(key_dim,hidden_dim))
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor: 
        residual = x  # b s h

        x_norm = self.layer_norm(x)
        q = x_norm @ self.qkv[:, :self.key_dim] # b s k
        k = x_norm @ self.qkv[:, self.key_dim:2*self.key_dim] # b s k
        v = x_norm @ self.qkv[:, 2*self.key_dim:] # b s k
        attention_scores = torch.einsum("bsk,bkt->bst", q, k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.key_dim))) # b s t
        S = residual.size(1)
        attention_mask = torch.tril(
            torch.ones(S, S, dtype=torch.bool, device=x.device)
        ).unsqueeze(0) # 1 s t
        attention_scores = attention_scores.masked_fill(~attention_mask, float("-inf"))
        attention_scores = attention_scores - attention_scores.max(dim=-1, keepdim=True).values
        softmax_scores = torch.softmax(attention_scores, dim=-1)
        output = softmax_scores @ v
        output = output @ self.o_proj
        return output + residual
