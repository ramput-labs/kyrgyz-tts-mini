"""HiFi-GAN generator (V1 config): mel-spectrogram → waveform."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1
N_MELS = 80
UPSAMPLE_RATES = (8, 8, 2, 2)
UPSAMPLE_KERNEL_SIZES = (16, 16, 4, 4)
UPSAMPLE_INITIAL_CHANNEL = 512
RESBLOCK_KERNEL_SIZES = (3, 7, 11)
RESBLOCK_DILATIONS = ((1, 3, 5), (1, 3, 5), (1, 3, 5))


def conv(channels: int, kernel_size: int, dilation: int = 1) -> nn.Module:
    padding = (kernel_size * dilation - dilation) // 2
    return weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding))


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...]):
        super().__init__()
        self.convs1 = nn.ModuleList(conv(channels, kernel_size, d) for d in dilations)
        self.convs2 = nn.ModuleList(conv(channels, kernel_size) for _ in dilations)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))
        return x

    def remove_weight_norm(self):
        for layer in (*self.convs1, *self.convs2):
            remove_weight_norm(layer)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pre = weight_norm(nn.Conv1d(N_MELS, UPSAMPLE_INITIAL_CHANNEL, 7, 1, padding=3))
        channels = [UPSAMPLE_INITIAL_CHANNEL // 2**i for i in range(len(UPSAMPLE_RATES) + 1)]
        self.ups = nn.ModuleList(
            weight_norm(nn.ConvTranspose1d(channels[i], channels[i + 1], k, u, padding=(k - u) // 2))
            for i, (u, k) in enumerate(zip(UPSAMPLE_RATES, UPSAMPLE_KERNEL_SIZES, strict=True))
        )
        self.resblocks = nn.ModuleList(
            ResBlock(ch, k, d)
            for ch in channels[1:]
            for k, d in zip(RESBLOCK_KERNEL_SIZES, RESBLOCK_DILATIONS, strict=True)
        )
        self.conv_post = weight_norm(nn.Conv1d(channels[-1], 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        n = len(RESBLOCK_KERNEL_SIZES)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            x = sum(block(x) for block in self.resblocks[i * n : (i + 1) * n]) / n
        return torch.tanh(self.conv_post(F.leaky_relu(x)))

    def remove_weight_norm(self):
        for layer in (self.conv_pre, *self.ups, self.conv_post):
            remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
