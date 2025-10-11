import torch
import torch.nn as nn

class ReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.where(x > 0, x, 0) # Maybe use scalar 0
        return x
