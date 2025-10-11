import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(
        self,
        eps: float=1e-5
    ):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_std = (x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt() #B S 1
        residual_mean = x.mean(dim=-1, keepdim=True) # B S 1
        return (x - residual_mean) / residual_std
