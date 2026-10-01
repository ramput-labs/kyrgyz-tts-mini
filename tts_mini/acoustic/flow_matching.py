
import torch

from tts_mini.acoustic.decoder import Decoder


class CFM(torch.nn.Module):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__()
        self.n_feats = in_channels
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.sigma_min = getattr(cfm_params, "sigma_min", 1e-4)

        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Sample a mel-spectrogram (batch, n_feats, frames) conditioned on the encoder output `mu`."""
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        t, dt = t_span[0], t_span[1] - t_span[0]
        for step in range(1, len(t_span)):
            x = x + dt * self.estimator(x, mask, mu, t, spks, cond)
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x
