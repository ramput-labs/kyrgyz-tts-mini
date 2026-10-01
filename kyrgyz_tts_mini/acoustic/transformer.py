import torch
import torch.nn.functional as F
from torch import nn


class SnakeBeta(nn.Module):
    """Snake activation with separate magnitude: x + 1/b * sin^2(x * a)."""

    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)
        self.alpha_logscale = alpha_logscale
        init = torch.zeros(out_features) if alpha_logscale else torch.ones(out_features)
        self.alpha = nn.Parameter(init * alpha, requires_grad=alpha_trainable)
        self.beta = nn.Parameter(init * alpha, requires_grad=alpha_trainable)

    def forward(self, x):
        x = self.proj(x)
        alpha, beta = (self.alpha.exp(), self.beta.exp()) if self.alpha_logscale else (self.alpha, self.beta)
        return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


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
        self.net = nn.Sequential(
            ACTIVATIONS[activation_fn](dim, inner_dim), nn.Dropout(dropout), nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim: int, heads: int, dim_head: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, attention_mask: torch.Tensor | None = None):
        b, t, _ = x.shape
        q, k, v = (proj(x).view(b, t, self.heads, -1).transpose(1, 2) for proj in (self.to_q, self.to_k, self.to_v))
        if attention_mask is not None:
            # The 0/1 mask is *added* to the scores, not used as a boolean mask: the checkpoints were trained this way.
            attention_mask = attention_mask.view(b, 1, 1, t).to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        return self.to_out(out.transpose(1, 2).reshape(b, t, -1))


class BasicTransformerBlock(nn.Module):
    """Pre-norm self-attention + feed-forward block."""

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, dropout=0.0, activation_fn="geglu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, num_attention_heads, attention_head_dim, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)

    def forward(self, x, attention_mask=None):
        x = self.attn1(self.norm1(x), attention_mask=attention_mask) + x
        return self.ff(self.norm3(x)) + x
