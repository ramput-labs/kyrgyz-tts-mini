import torch
from torch import nn


class Denoiser(nn.Module):
    """Subtracts the vocoder's bias spectrum (its output for a silent mel) to reduce background hiss."""

    def __init__(self, vocoder: nn.Module, filter_length: int = 1024, n_overlap: int = 4, win_length: int = 1024):
        super().__init__()
        self.n_fft = filter_length
        self.hop_length = filter_length // n_overlap
        self.win_length = win_length
        param = next(vocoder.parameters())
        self.register_buffer("window", torch.hann_window(win_length, device=param.device))

        with torch.no_grad():
            bias_audio = vocoder(torch.zeros((1, 80, 88), dtype=param.dtype, device=param.device)).float().squeeze(0)
            bias_spec, _ = self.stft(bias_audio)
        self.register_buffer("bias_spec", bias_spec[:, :, :1])

    def stft(self, audio):
        spec = torch.view_as_real(
            torch.stft(audio, self.n_fft, self.hop_length, self.win_length, self.window, return_complex=True)
        )
        return torch.sqrt(spec.pow(2).sum(-1)), torch.atan2(spec[..., -1], spec[..., 0])

    def istft(self, magnitude, phase):
        spec = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
        return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window)

    @torch.inference_mode()
    def forward(self, audio, strength=0.0005):
        magnitude, phase = self.stft(audio)
        return self.istft(torch.clamp(magnitude - self.bias_spec * strength, 0.0), phase)
