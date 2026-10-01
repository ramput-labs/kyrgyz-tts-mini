import torch
import torch.nn.functional as F


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fix_len_compatibility(length: torch.Tensor, num_downsamplings: int = 2) -> int:
    """Round up to a multiple of the U-Net's total downsampling factor."""
    factor = 2**num_downsamplings
    return int(torch.ceil(length / factor).item()) * factor


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Hard monotonic alignment (batch, text, frames) from per-token durations."""
    b, t_x, t_y = mask.shape
    path = sequence_mask(torch.cumsum(duration, 1).view(b * t_x), t_y).to(mask.dtype).view(b, t_x, t_y)
    path = path - F.pad(path, (0, 0, 1, 0))[:, :-1]
    return path * mask
