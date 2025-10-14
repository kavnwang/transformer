import torch
import torch.nn as nn
import einops


class MoESwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, num_experts: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.up_proj = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(num_experts, hidden_dim, intermediate_dim)
            )
        )
        self.gate_proj = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(num_experts, hidden_dim, intermediate_dim)
            )
        )
        self.down_proj = nn.Parameter(
            nn.init.kaiming_uniform_(
                torch.empty(num_experts, intermediate_dim, hidden_dim)
            )
        )
        self.silu = nn.SiLU()
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = torch.einsum("e h i, b s e h -> b s e i", self.up_proj, x)
        gate = torch.einsum("e h i, b s e h -> b s e i", self.gate_proj, x)
        gate = self.silu(gate)  # b s e i
        product = up * gate
        output = torch.einsum("e i h, b s e i -> b s e h", self.down_proj, product)
        return output


class MoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        num_selected: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_selected = num_selected
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)  # inefficient
        self.mlp = MoESwiGLU(hidden_dim, intermediate_dim, num_experts)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b s h
        x_norm = self.layer_norm(x)
        scores = self.router(x_norm)  # b s e
        _, indices = torch.topk(scores, k=self.num_selected, dim=-1)  # b s n
        probs = torch.ones_like(scores) * float("-inf")
        probs[:, torch.arange(probs.shape[1]).unsqueeze(1), indices] = scores[
            :, torch.arange(probs.shape[1]).unsqueeze(1), indices
        ]
        probs = torch.softmax(probs, dim=-1)  # b s e
        x_copy = x_norm.unsqueeze(-2).tile((1, 1, self.num_experts, 1))
        output = torch.einsum(
            "b s e, b s e h -> b s e h", probs, self.mlp(x_copy)
        )  # b s e h
        return x + einops.reduce(output, "b s e h -> b s h", "sum")
