import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        eps: float=1e-5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.eps = eps
        self.up_proj = nn.Linear(hidden_dim,intermediate_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_dim,intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim,hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.layer_norm(x)
        up = self.up_proj(x_norm) #b s i
        gate = self.gate_proj(x_norm)
        gate = self.silu(gate) #b s i
        product = up * gate #b s i
        return x + self.down_proj(product)



