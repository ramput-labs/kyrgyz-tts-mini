"""Matcha-TTS acoustic model: Kyrgyz token ids → mel-spectrogram."""

import collections
import functools
import typing
from pathlib import Path

import omegaconf
import torch
from torch import nn

from kyrgyz_tts_mini.acoustic.flow_matching import CFM
from kyrgyz_tts_mini.acoustic.text_encoder import TextEncoder
from kyrgyz_tts_mini.acoustic.utils import fix_len_compatibility, generate_path, sequence_mask

SAMPLE_RATE = 22050

# torch>=2.6 loads with weights_only=True; allowlist the objects pickled in the checkpoints.
CHECKPOINT_SAFE_GLOBALS = [
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    omegaconf.base.Metadata,
    omegaconf.nodes.AnyNode,
    torch.optim.Adam,
    functools.partial,
    collections.defaultdict,
    typing.Any,
    list,
    dict,
    int,
]


class AcousticModel(nn.Module):
    def __init__(self, n_vocab, encoder, decoder, data_statistics, **_unused_hparams):
        super().__init__()
        n_feats = encoder.encoder_params.n_feats
        self.encoder = TextEncoder(encoder.encoder_params, encoder.duration_predictor_params, n_vocab)
        self.decoder = CFM(n_feats, decoder)

        data_statistics = data_statistics or {"mel_mean": 0.0, "mel_std": 1.0}
        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: torch.device | str = "cpu") -> "AcousticModel":
        with torch.serialization.safe_globals(CHECKPOINT_SAFE_GLOBALS):
            checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["hyper_parameters"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device).eval()

    @torch.inference_mode()
    def synthesize(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        n_timesteps: int,
        temperature: float = 1.0,
        length_scale: float = 1.0,
    ) -> torch.Tensor:
        """Token ids (batch, tokens) → mel-spectrogram (batch, n_feats, frames)."""
        mu_x, logw, x_mask = self.encoder(x, x_lengths)

        w_ceil = torch.ceil(torch.exp(logw) * x_mask) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()

        y_mask = sequence_mask(y_lengths, fix_len_compatibility(y_max_length)).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1))
        mu_y = torch.matmul(attn.transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)

        mel = self.decoder(mu_y, y_mask, n_timesteps, temperature)[:, :, :y_max_length]
        return mel * self.mel_std.unsqueeze(-1) + self.mel_mean.unsqueeze(-1)
