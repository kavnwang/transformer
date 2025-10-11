import torch
import torch.nn as nn
from layers.silu import SiLU
from layers.layernorm import LayerNorm
class SwiGLU(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        eps: float=1e-6
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.eps = eps
        self.up_proj = nn.Linear(hidden_dim,intermediate_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_dim,intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim,hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.layer_norm = LayerNorm(eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("swiglu", torch.max(abs(x)), torch.min(abs(x)))
        print(x.dtype)
        x_norm = self.layer_norm(x)
        up = self.up_proj(x_norm) #b s i
        gate = self.gate_proj(x_norm)
        print("swiglu norm", torch.max(abs(x_norm)),torch.min(abs(x_norm)))
        print("swiglu up", torch.max(abs(up)),torch.min(abs(gate)))
        print("swiglu gate", torch.max(abs(gate)), torch.min(abs(gate)))
        gate = self.silu(gate) #b s i
        print(gate.dtype)
        print("swiglu gate silu", torch.max(abs(gate)), torch.min(abs(gate)))
        product = up * gate #b s i
        print("swiglu", torch.max(abs(product)), torch.min(abs(product)))
        return x + self.down_proj(product)



