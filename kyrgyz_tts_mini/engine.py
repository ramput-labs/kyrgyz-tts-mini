"""Text → speech: the acoustic model turns text into a mel-spectrogram, the vocoder turns that into audio."""

import threading
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import torch

from kyrgyz_tts_mini import config
from kyrgyz_tts_mini.acoustic.model import SAMPLE_RATE, AcousticModel
from kyrgyz_tts_mini.config import Settings
from kyrgyz_tts_mini.text import has_letters, to_ids
from kyrgyz_tts_mini.vocoder.denoiser import Denoiser
from kyrgyz_tts_mini.vocoder.hifigan import Generator

DEFAULTS = Settings()


@dataclass
class Speech:
    audio: np.ndarray  # mono float32 in [-1, 1]
    sample_rate: int
    elapsed: float  # synthesis time in seconds

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    @classmethod
    def join(cls, parts: list["Speech"], pause: float = 0.3) -> "Speech":
        sr = parts[0].sample_rate
        silence = np.zeros(int(pause * sr), dtype=np.float32)
        audio = np.concatenate([chunk for part in parts for chunk in (part.audio, silence)][:-1])
        return cls(audio, sr, sum(part.elapsed for part in parts))

    def save(self, path: str | Path) -> Path:
        import soundfile as sf

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, self.audio, self.sample_rate, subtype="PCM_16")
        return path

    def play(self) -> None:
        import sounddevice as sd

        sd.play(self.audio, self.sample_rate)
        sd.wait()


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make setup` (or `python -m kyrgyz_tts_mini download`) first.")
    return path


class TTS:
    sample_rate = SAMPLE_RATE

    def __init__(self, device: str | None = None):
        self.device = torch.device(device or pick_device())
        self._voices: dict[str, AcousticModel] = {}
        self._vocoder: tuple[Generator, Denoiser] | None = None
        self._lock = threading.Lock()

    def voice(self, name: str) -> AcousticModel:
        if name not in self._voices:
            self._voices[name] = AcousticModel.from_checkpoint(_require(config.VOICES[name]), self.device)
        return self._voices[name]

    def vocoder(self) -> tuple[Generator, Denoiser]:
        if self._vocoder is None:
            with warnings.catch_warnings():  # torch's deprecated weight_norm, needed to match the checkpoint keys
                warnings.simplefilter("ignore", FutureWarning)
                vocoder = Generator().to(self.device)
            state = torch.load(_require(config.VOCODER), map_location=self.device)["generator"]
            vocoder.load_state_dict(state)
            vocoder.eval().remove_weight_norm()
            self._vocoder = vocoder, Denoiser(vocoder)
        return self._vocoder

    def warm_up(self, voice: str) -> None:
        self.voice(voice)
        self.vocoder()

    def synthesize(self, text: str, voice: str = config.DEFAULT_VOICE, settings: Settings = DEFAULTS) -> Speech:
        if not has_letters(text):
            raise ValueError("Nothing to say: the text has no Kyrgyz Cyrillic letters.")
        model = self.voice(voice)
        vocoder, denoiser = self.vocoder()

        start = time.perf_counter()
        with self._lock, torch.inference_mode():
            x = torch.tensor([to_ids(text)], dtype=torch.long, device=self.device)
            x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=self.device)
            mel = model.synthesize(x, x_lengths, settings.steps, settings.temperature, settings.rate)
            audio = vocoder(mel).clamp(-1, 1)
            if settings.denoise > 0:
                audio = denoiser(audio.squeeze(), strength=settings.denoise)
            audio = audio.float().cpu().squeeze().numpy()
        return Speech(audio, SAMPLE_RATE, time.perf_counter() - start)

    def synthesize_lines(
        self, lines: Iterable[str], voice: str = config.DEFAULT_VOICE, settings: Settings = DEFAULTS
    ) -> Speech:
        """Speak each non-empty line and join them with a short pause."""
        texts = [line.strip() for line in lines if line.strip()]
        if not texts:
            raise ValueError("Nothing to say: the text is empty.")
        return Speech.join([self.synthesize(text, voice, settings) for text in texts])


@cache
def get_tts(device: str | None = None) -> TTS:
    return TTS(device)
