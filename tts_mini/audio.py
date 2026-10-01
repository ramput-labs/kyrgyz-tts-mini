from datetime import datetime
from pathlib import Path

import numpy as np


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save(audio: np.ndarray, sr: int, path: str | Path) -> Path:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_16")
    return path


def timestamped(folder: Path, suffix: str = "") -> Path:
    return folder / f"{datetime.now():%Y%m%d-%H%M%S-%f}{suffix}.wav"


def play(audio: np.ndarray, sr: int) -> None:
    import sounddevice as sd

    sd.play(audio, sr)
    sd.wait()
