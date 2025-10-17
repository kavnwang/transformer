import torch
import torch.nn as nn
from layers.rmsnorm import RMSNorm
from einops import rearrange

class Attention(nn.Module):
    def __init__(
        self, 
        hidden_dim: int,
        key_dim: int,
        num_heads: int,
        qk_norm: bool = False,
        eps: float=1e-5
    ): 
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.eps = eps
        self.rms_norm = RMSNorm(hidden_dim, eps=1e-5)
        self.q_proj = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(hidden_dim, key_dim))
        )
        self.k_proj = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(hidden_dim, key_dim))
        )
        self.v_proj = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(hidden_dim, key_dim))
        )
        self.o_proj = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(key_dim,hidden_dim))
        )
        if qk_norm:
            self.q_norm = RMSNorm(key_dim // self.num_heads, eps)
            self.k_norm = RMSNorm(key_dim // self.num_heads, eps)

    def forward(self,x: torch.Tensor) -> torch.Tensor: 
        residual = x  # b s h
        x_norm = self.rms_norm(x)
        q = rearrange(x_norm @ self.q_proj,"... (h d) -> ... h d",h=self.num_heads) # b s h d
        k = rearrange(x_norm @ self.k_proj, "... (h d) -> ... h d",h=self.num_heads) # b t h d
        v = rearrange(x_norm @ self.v_proj, "... (h d) -> ... h d",h=self.num_heads) # b t h d
        if self.q_norm:
            self.q_norm(q)
        if self.k_norm:
            self.k_norm(k)
        attention_scores = torch.einsum("bshd,bthd->bhst", q, k) * (1.0 / torch.sqrt(torch.tensor(self.key_dim))) #b h s t
        S = residual.size(1)
        attention_mask = torch.tril(
            torch.ones(S, S, dtype=torch.bool, device=x.device)
        ).unsqueeze(0) # 1 s t
        attention_scores = attention_scores.masked_fill(~attention_mask, float("-inf"))
        attention_scores = attention_scores - attention_scores.max(dim=-1, keepdim=True).values
        softmax_scores = torch.softmax(attention_scores, dim=-1)
        output = torch.einsum("bhst,bthd->bshd", softmax_scores, v) #(b h s t) (b t h d) -> (b s h d)
        output = rearrange(output, "... h d -> ... (h d)",h=self.num_heads) @ self.o_proj
        return output + residual
