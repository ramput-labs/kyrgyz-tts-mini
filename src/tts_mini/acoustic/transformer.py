"""Parameter names here must match the checkpoints, so do not rename layers."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeBeta(nn.Module):
    """Snake activation with separate magnitude: x + 1/b * sin^2(x * a)."""

    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.in_features = out_features if isinstance(out_features, list) else [out_features]
        self.proj = nn.Linear(in_features, out_features)

        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta
        return x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def forward(self, x):
        return F.gelu(self.proj(x), approximate=self.approximate)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


ACTIVATIONS = {
    "gelu": GELU,
    "gelu-approximate": lambda dim_in, dim_out: GELU(dim_in, dim_out, approximate="tanh"),
    "geglu": GEGLU,
    "snakebeta": SnakeBeta,
}


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0, activation_fn: str = "geglu"):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.ModuleList(
            [ACTIVATIONS[activation_fn](dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim)]
        )

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, query_dim: int, heads: int, dim_head: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)])

    def forward(self, x, attention_mask: Optional[torch.Tensor] = None):
        b, t, _ = x.shape
        q, k, v = (proj(x).view(b, t, self.heads, -1).transpose(1, 2) for proj in (self.to_q, self.to_k, self.to_v))
        if attention_mask is not None:
            # The 0/1 float mask is *added* to the scores (not used as a boolean mask).
            # The checkpoints were trained this way, so keep it.
            attention_mask = attention_mask.view(b, 1, 1, t).to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.to_out[1](self.to_out[0](out))


class BasicTransformerBlock(nn.Module):
    """Pre-norm self-attention + feed-forward block."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout, bias=attention_bias
        )
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)

    def forward(self, hidden_states, attention_mask=None, timestep=None):
        hidden_states = self.attn1(self.norm1(hidden_states), attention_mask=attention_mask) + hidden_states
        return self.ff(self.norm3(hidden_states)) + hidden_states
