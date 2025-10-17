import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        eps: float=1e-5
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x ** 2, dim=-1) + self.eps)[:,:,None] * self.weight #b s h / ()
