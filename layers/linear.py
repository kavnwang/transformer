import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(out_dim)))
        self.proj = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(in_dim, out_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.proj + self.bias
