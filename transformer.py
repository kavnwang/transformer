import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.proj = nn.Parameter(torch.zeros(in_dim, out_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...h,hd->...d", x, self.proj) + self.bias

class ReLU(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.where(x > 0, x, torch.zeros_like(x))
        return x

class RMSNorm(nn.Module):
    def __init__(
        self,
        eps: float=1e-6
    ):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_std = (x.var(dim=-1, keepdim=True) + self.eps).sqrt()
        residual_mean = x.mean(dim=-1, keepdim=True)
        return (x - residual_mean)/ residual_std

class Attention(nn.Module):
    def __init__(
        self, 
        hidden_dim: int,
        key_dim: int,
        num_heads: int,
        eps: float=1e-6
    ): 
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.eps = eps
        self.RMSNorm = RMSNorm(eps)
        self.qkv = nn.Parameter(
            torch.zeros(hidden_dim, key_dim*3)
        )
        self.o_proj = nn.Parameter(
            torch.zeros(key_dim,hidden_dim)
        )
    def forward(self,x: torch.Tensor) -> torch.Tensor: 
        h = x  # b s h
        q = torch.einsum("bsh,hk->bsk", x, self.qkv[:, :self.key_dim])
        k = torch.einsum("bsh,hk->bsk", x, self.qkv[:, self.key_dim:2*self.key_dim])
        v = torch.einsum("bsh,hk->bsk", x, self.qkv[:, 2*self.key_dim:])
        attention_scores = torch.einsum("bsk,btk->bst", q, k) / self.key_dim**0.5
        S = h.size(1)
        attention_mask = torch.tril(
            torch.ones(S, S, dtype=torch.bool, device=x.device)
        ).unsqueeze(0)
        attention_scores = attention_scores.masked_fill(~attention_mask, float("-inf"))
        softmax_scores = torch.softmax(attention_scores, dim=-1)
        x = torch.einsum("bst,btk->bsk", softmax_scores, v)
        x = torch.einsum("bsk,kh->bsh", x, self.o_proj)  # b s h
        return x + h

class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.up_proj = Linear(hidden_dim,intermediate_dim)
        self.down_proj = Linear(intermediate_dim,hidden_dim)
        self.relu = ReLU()
        self.mlp = nn.Sequential(self.up_proj, self.relu, self.down_proj)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)

class SwiGLU(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.up_proj = Linear(hidden_dim,intermediate_dim // 2)
        self.gate_proj = Linear(hidden_dim,intermediate_dim // 2)
        self.down_proj = Linear(intermediate_dim,hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class MoE(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim



class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        vocab_size: int, 
        key_dim: int,
        num_heads: int,
        intermediate_dim: int,
        eps: float=1e-6
    ): 

        super().__init__()
        self.num_layers = num_layers
        self.RMSNorm = RMSNorm()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.unembedding = Linear(hidden_dim, vocab_size)
        layers = []
        for _ in range(num_layers):
            layers.append(Attention(hidden_dim, key_dim, num_heads, eps))
            layers.append(MLP(hidden_dim, intermediate_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: int) -> torch.Tensor:
        x = self.embedding(x)
        x = self.model(x)
        x = self.unembedding(x)
        return x