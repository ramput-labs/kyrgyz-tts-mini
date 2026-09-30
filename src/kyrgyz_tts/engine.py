"""kyrgyz-tts engine: Matcha-TTS voices and a HiFi-GAN vocoder."""

import threading
import time
import warnings
from dataclasses import dataclass
from functools import cache

import numpy as np
import torch

from kyrgyz_tts import config
from kyrgyz_tts.audio import pick_device
from kyrgyz_tts.hifigan.config import v1 as HIFIGAN_V1
from kyrgyz_tts.hifigan.denoiser import Denoiser
from kyrgyz_tts.hifigan.models import Generator as HiFiGAN
from kyrgyz_tts.matcha.model import SAMPLE_RATE, MatchaTTS
from kyrgyz_tts.text import has_letters, intersperse, text_to_sequence

VOICES = tuple(config.VOICES)


class _AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def _require(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make setup` (or `kyrgyz-tts download`) first.")
    return path


@dataclass
class Speech:
    audio: np.ndarray  # mono float32 in [-1, 1]
    sample_rate: int
    seconds: float  # processing time

    @property
    def audio_seconds(self) -> float:
        return len(self.audio) / self.sample_rate


class TTS:
    """Loads voices lazily and the vocoder once; inference is serialized with a lock."""

    sample_rate = SAMPLE_RATE

    def __init__(self, device: str | None = None):
        self.device = torch.device(device or pick_device())
        self._voices: dict[str, MatchaTTS] = {}
        self._vocoder = self._denoiser = None
        self._lock = threading.Lock()

    def voice(self, name: str) -> MatchaTTS:
        if name not in self._voices:
            self._voices[name] = MatchaTTS.from_checkpoint(_require(config.VOICES[name]), self.device)
        return self._voices[name]

    def vocoder(self) -> tuple[HiFiGAN, Denoiser]:
        if self._vocoder is None:
            path = _require(config.VOCODER)
            with warnings.catch_warnings():  # HiFi-GAN uses the older (deprecated) weight_norm API
                warnings.simplefilter("ignore", FutureWarning)
                vocoder = HiFiGAN(_AttrDict(HIFIGAN_V1)).to(self.device)
            vocoder.load_state_dict(torch.load(path, map_location=self.device)["generator"])
            vocoder.eval()
            vocoder.remove_weight_norm()
            self._vocoder, self._denoiser = vocoder, Denoiser(vocoder, mode="zeros")
        return self._vocoder, self._denoiser

    def warm_up(self, voice: str) -> None:
        self.voice(voice)
        self.vocoder()

    def synthesize(
        self,
        text: str,
        voice: str = "man",
        *,
        temperature: float = 0.667,
        speaking_rate: float = 1.0,
        steps: int = 10,
        denoiser_strength: float = 0.00025,
    ) -> Speech:
        """Synthesize `text`. speaking_rate is a length scale: higher is slower."""
        if not has_letters(text):
            raise ValueError("Nothing to say: the text has no Kyrgyz Cyrillic letters.")
        ids = intersperse(text_to_sequence(text), 0)
        model = self.voice(voice)
        vocoder, denoiser = self.vocoder()

        start = time.perf_counter()
        with self._lock, torch.inference_mode():
            x = torch.tensor(ids, dtype=torch.long, device=self.device)[None]
            x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=self.device)
            output = model.synthesise(
                x, x_lengths, n_timesteps=steps, temperature=temperature, length_scale=speaking_rate
            )
            audio = vocoder(output["mel"]).clamp(-1, 1)
            if denoiser_strength > 0:
                audio = denoiser(audio.squeeze(), strength=denoiser_strength)
            audio = audio.float().cpu().squeeze().numpy()

        return Speech(audio=audio, sample_rate=SAMPLE_RATE, seconds=time.perf_counter() - start)


@cache
def get_tts(device: str | None = None) -> TTS:
    return TTS(device)
