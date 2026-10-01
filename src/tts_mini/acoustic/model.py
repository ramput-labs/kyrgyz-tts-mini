import collections
import functools
import time
import typing
from pathlib import Path

import omegaconf
import torch

from tts_mini.acoustic.flow_matching import CFM
from tts_mini.acoustic.text_encoder import TextEncoder
from tts_mini.acoustic.utils import denormalize, fix_len_compatibility, generate_path, sequence_mask

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

SAMPLE_RATE = 22050
HOP_LENGTH = 256


class AcousticModel(torch.nn.Module):
    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_feats, encoder, decoder, cfm, data_statistics, **_training_only):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)

        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim,
        )
        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

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
    def synthesise(self, x, x_lengths, n_timesteps, temperature=1.0, spks=None, length_scale=1.0):
        """Token ids (B, T) → {"mel": (B, n_feats, frames), "mel_lengths", "rtf"}."""
        start = time.perf_counter()

        if self.n_spks > 1:
            spks = self.spk_emb(spks.long())

        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)

        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature, spks)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        elapsed = time.perf_counter() - start
        return {
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths,
            "rtf": elapsed * SAMPLE_RATE / (decoder_outputs.shape[-1] * HOP_LENGTH),
        }
