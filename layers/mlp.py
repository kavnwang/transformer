import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.relu = nn.ReLU()
        self.mlp = nn.Sequential(self.up_proj, self.relu, self.down_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)
