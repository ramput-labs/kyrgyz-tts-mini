import torch
from torch import nn

from kyrgyz_tts_mini.acoustic.decoder import Decoder


class CFM(nn.Module):
    """Conditional flow matching: integrates the decoder's vector field from noise to a mel-spectrogram."""

    def __init__(self, n_feats: int, decoder_params):
        super().__init__()
        self.estimator = Decoder(in_channels=2 * n_feats, out_channels=n_feats, **decoder_params)

    def forward(self, mu: torch.Tensor, mask: torch.Tensor, n_timesteps: int, temperature: float = 1.0) -> torch.Tensor:
        x = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        t, dt = t_span[0], t_span[1] - t_span[0]
        for step in range(1, len(t_span)):
            x = x + dt * self.estimator(x, mask, mu, t)
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x
