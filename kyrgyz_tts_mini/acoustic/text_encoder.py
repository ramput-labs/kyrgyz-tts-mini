import math

import torch
from einops import rearrange
from torch import nn

from kyrgyz_tts_mini.acoustic.utils import sequence_mask


class LayerNorm(nn.Module):
    """LayerNorm over the channel axis of (batch, channels, time)."""

    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        shape = [1, -1] + [1] * (x.dim() - 2)
        return x * self.gamma.view(*shape) + self.beta.view(*shape)


class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2
            )
            for i in range(n_layers)
        )
        self.norm_layers = nn.ModuleList(LayerNorm(hidden_channels) for _ in range(n_layers))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_mask):
        residual = x
        for conv, norm in zip(self.conv_layers, self.norm_layers, strict=True):
            x = self.relu_drop(norm(conv(x * x_mask)))
        return (residual + self.proj(x)) * x_mask


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.drop(self.norm_1(torch.relu(self.conv_1(x * x_mask))))
        x = self.drop(self.norm_2(torch.relu(self.conv_2(x * x_mask))))
        return self.proj(x * x_mask) * x_mask


class RotaryPositionalEmbeddings(nn.Module):
    """Rotary position embeddings; only the first `d` features are rotated."""

    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.d = int(d)
        self.cos_cached = self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)
        seq_idx = torch.arange(x.shape[0], device=x.device).float()
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def forward(self, x: torch.Tensor):
        """x: (batch, heads, time, features)."""
        x = rearrange(x, "b h t d -> t b h d")
        self._build_cache(x)
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]
        half = self.d // 2
        neg_half_x = torch.cat([-x_rope[..., half:], x_rope[..., :half]], dim=-1)
        x_rope = x_rope * self.cos_cached[: x.shape[0]] + neg_half_x * self.sin_cached[: x.shape[0]]
        return rearrange(torch.cat((x_rope, x_pass), dim=-1), "t b h d -> b h t d")


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.query_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
        self.key_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, attn_mask):
        b, d, t = x.shape
        q, k, v = (
            rearrange(conv(x), "b (h c) t -> b h t c", h=self.n_heads)
            for conv in (self.conv_q, self.conv_k, self.conv_v)
        )
        q, k = self.query_rotary_pe(q), self.key_rotary_pe(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.k_channels)
        scores = scores.masked_fill(attn_mask == 0, -1e4)
        out = torch.matmul(self.drop(torch.softmax(scores, dim=-1)), v)
        return self.conv_o(out.transpose(2, 3).contiguous().view(b, d, t))


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.drop(torch.relu(self.conv_1(x * x_mask)))
        return self.conv_2(x * x_mask) * x_mask


class Encoder(nn.Module):
    """Post-norm transformer encoder with rotary attention and convolutional feed-forward layers."""

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList(
            MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout) for _ in range(n_layers)
        )
        self.norm_layers_1 = nn.ModuleList(LayerNorm(hidden_channels) for _ in range(n_layers))
        self.ffn_layers = nn.ModuleList(
            FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout) for _ in range(n_layers)
        )
        self.norm_layers_2 = nn.ModuleList(LayerNorm(hidden_channels) for _ in range(n_layers))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        layers = zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2, strict=True)
        for attn, norm_1, ffn, norm_2 in layers:
            x = x * x_mask
            x = norm_1(x + self.drop(attn(x, attn_mask)))
            x = norm_2(x + self.drop(ffn(x, x_mask)))
        return x * x_mask


class TextEncoder(nn.Module):
    """Token ids → mel-space means `mu` and log-durations `logw` per token."""

    def __init__(self, encoder_params, duration_predictor_params, n_vocab):
        super().__init__()
        self.n_channels = encoder_params.n_channels
        self.emb = nn.Embedding(n_vocab, self.n_channels)
        self.prenet = (
            ConvReluNorm(self.n_channels, self.n_channels, self.n_channels, kernel_size=5, n_layers=3, p_dropout=0.5)
            if encoder_params.prenet
            else None
        )
        self.encoder = Encoder(
            self.n_channels,
            encoder_params.filter_channels,
            encoder_params.n_heads,
            encoder_params.n_layers,
            encoder_params.kernel_size,
            encoder_params.p_dropout,
        )
        self.proj_m = nn.Conv1d(self.n_channels, encoder_params.n_feats, 1)
        self.proj_w = DurationPredictor(
            self.n_channels,
            duration_predictor_params.filter_channels_dp,
            duration_predictor_params.kernel_size,
            duration_predictor_params.p_dropout,
        )

    def forward(self, x, x_lengths):
        """Returns mu (batch, n_feats, tokens), logw (batch, 1, tokens) and x_mask (batch, 1, tokens)."""
        x = self.emb(x).transpose(1, -1) * math.sqrt(self.n_channels)
        x_mask = sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        if self.prenet is not None:
            x = self.prenet(x, x_mask)
        x = self.encoder(x, x_mask)
        return self.proj_m(x) * x_mask, self.proj_w(x.detach(), x_mask), x_mask
