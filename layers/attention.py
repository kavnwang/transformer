import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional
from layers.rmsnorm import RMSNorm
from einops import rearrange, repeat
import math

class Attention(nn.Module):
    def __init__(
        self, 
        hidden_dim: int,
        key_dim: int,
        num_heads: int,
        qk_norm: bool = False,
        rope_theta: float = 10000.0,
        use_cache: bool = False,
        eps: float=1e-5,
    ): 
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.eps = eps
        self.rms_norm = RMSNorm(hidden_dim, eps=1e-5)
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(hidden_dim,key_dim)
        self.k_proj = nn.Linear(hidden_dim,key_dim)
        self.v_proj = nn.Linear(hidden_dim,key_dim)
        self.o_proj = nn.Linear(key_dim,hidden_dim)
        self.qk_norm = qk_norm
        self.use_cache = use_cache
        if qk_norm:
            self.q_norm = RMSNorm(key_dim, eps)
            self.k_norm = RMSNorm(key_dim, eps)
    
    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        x1 = repeat(x[...,0::2], "b s n d -> b s n (2 d)")
        x2 = repeat(x[...,1::2], "b s n d -> b s n (2 d)")
        d = x.shape[-1]
        theta = repeat(self.rope_theta ** (-2 * torch.arange(d // 2).to(x.device) / d)[None, None, None, :],"b s n d -> b s n (2 d)")
        theta = theta * torch.arange(x.shape[1]).to(x.device)[None,:,None,None]
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        return torch.where(
            (torch.arange(d) % 2 == 0).to(x.device) [None, None, None, :], 
            cos * x1 - sin * x2,
            sin * x1 + cos * x2
        )

    def forward(self,x: torch.Tensor, key_cache: Optional[Tensor] = None, value_cache: Optional[Tensor] = None,) -> torch.Tensor: 
        residual = x  # b s h
        x_norm = self.rms_norm(x)
        q = self.q_proj(x_norm)
        if self.use_cache and key_cache is not None:
            k = key_cache
            k_new = self.k_proj(x[:,-1:,:])
            k = torch.cat((k,k_new),dim=1)
        else:
            k = self.k_proj(x_norm)
            if self.qk_norm:
                k = self.k_norm(k)
        if self.use_cache:
            key_cache = k
        if self.use_cache and value_cache is not None:
            v = value_cache
            v_new = self.v_proj(x[:,-1:,:])
            v = torch.cat((v,v_new),dim=1)
        else:
            v = self.v_proj(x_norm)
        if self.use_cache:
            value_cache = v
        if self.qk_norm:
            q = self.q_norm(q)

        q = rearrange(q, "... (h d) -> ... h d",h=self.num_heads) # b s h d
        k = rearrange(k, "... (h d) -> ... h d",h=self.num_heads) # b t h d
        v = rearrange(v, "... (h d) -> ... h d",h=self.num_heads) # b t h d

        q = self.apply_rope(q)
        k = self.apply_rope(k)
        
        attention_scores = torch.einsum("bshd,bthd->bhst", q, k) / math.sqrt(self.key_dim // self.num_heads) #b h s t
        S = residual.size(1)
        attention_mask = torch.tril(
            torch.ones(S, S, dtype=torch.bool, device=x.device)
        ).unsqueeze(0) # 1 s t
        attention_scores = attention_scores.masked_fill(~attention_mask, float("-inf"))
        softmax_scores = torch.softmax(attention_scores, dim=-1)
        output = torch.einsum("bhst,bthd->bshd", softmax_scores, v) #(b h s t) (b t h d) -> (b s h d)
        output = self.o_proj(rearrange(output, "... h d -> ... (h d)",h=self.num_heads))
        hidden_states = output + residual
        return hidden_states, (key_cache, value_cache) if self.use_cache is True else None