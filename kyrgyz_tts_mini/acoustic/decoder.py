"""1D U-Net with transformer blocks: the vector field estimator of the flow-matching decoder."""

import math
from itertools import pairwise

import torch
from einops import pack, rearrange
from torch import nn

from kyrgyz_tts_mini.acoustic.transformer import BasicTransformerBlock


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "SinusoidalPosEmb requires an even dim"
        self.dim = dim

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -(math.log(10000) / (half_dim - 1)))
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Block1D(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(dim, dim_out, 3, padding=1), nn.GroupNorm(groups, dim_out), nn.Mish())

    def forward(self, x, mask):
        return self.block(x * mask) * mask


class ResnetBlock1D(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask) + self.mlp(time_emb).unsqueeze(-1)
        return self.block2(h, mask) + self.res_conv(x * mask)


class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        return self.linear_2(self.act(self.linear_1(sample)))


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
        down_block_type="transformer",
        mid_block_type="transformer",
        up_block_type="transformer",
    ):
        super().__init__()
        if {down_block_type, mid_block_type, up_block_type} != {"transformer"}:
            raise ValueError("only 'transformer' blocks are supported")
        channels = tuple(channels)
        time_embed_dim = channels[0] * 4
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        self.time_mlp = TimestepEmbedding(in_channels, time_embed_dim)

        def transformers(dim):
            return nn.ModuleList(
                BasicTransformerBlock(dim, num_heads, attention_head_dim, dropout, act_fn) for _ in range(n_blocks)
            )

        self.down_blocks = nn.ModuleList()
        dims = (in_channels, *channels)
        for i, (dim_in, dim_out) in enumerate(pairwise(dims)):
            is_last = i == len(channels) - 1
            downsample = nn.Conv1d(dim_out, dim_out, 3, padding=1) if is_last else Downsample1D(dim_out)
            self.down_blocks.append(
                nn.ModuleList([ResnetBlock1D(dim_in, dim_out, time_embed_dim), transformers(dim_out), downsample])
            )

        mid = channels[-1]
        self.mid_blocks = nn.ModuleList(
            nn.ModuleList([ResnetBlock1D(mid, mid, time_embed_dim), transformers(mid)]) for _ in range(num_mid_blocks)
        )

        self.up_blocks = nn.ModuleList()
        dims = (*channels[::-1], channels[0])
        for i, (dim_in, dim_out) in enumerate(pairwise(dims)):
            is_last = i == len(channels) - 1
            upsample = nn.Conv1d(dim_out, dim_out, 3, padding=1) if is_last else Upsample1D(dim_out)
            self.up_blocks.append(
                nn.ModuleList([ResnetBlock1D(2 * dim_in, dim_out, time_embed_dim), transformers(dim_out), upsample])
            )

        self.final_block = Block1D(channels[0], channels[0])
        self.final_proj = nn.Conv1d(channels[0], out_channels, 1)

    @staticmethod
    def _attend(x, blocks, mask):
        x = rearrange(x, "b c t -> b t c")
        mask = rearrange(mask, "b 1 t -> b t")
        for block in blocks:
            x = block(x, attention_mask=mask)
        return rearrange(x, "b t c -> b c t")

    def forward(self, x, mask, mu, t):
        """x, mu: (batch, channels, time); mask: (batch, 1, time); t: flow time in [0, 1]."""
        t = self.time_mlp(self.time_embeddings(t))
        x = pack([x, mu], "b * t")[0]

        hiddens, masks = [], [mask]
        for resnet, blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = self._attend(resnet(x, mask_down, t), blocks, mask_down)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks.pop()
        mask_mid = masks[-1]
        for resnet, blocks in self.mid_blocks:
            x = self._attend(resnet(x, mask_mid, t), blocks, mask_mid)

        for resnet, blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            x = resnet(pack([x, hiddens.pop()], "b * t")[0], mask_up, t)
            x = upsample(self._attend(x, blocks, mask_up) * mask_up)

        x = self.final_block(x, mask_up)
        return self.final_proj(x * mask_up) * mask
