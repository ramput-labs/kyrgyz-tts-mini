import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HF_REPO = os.environ.get("KYRGYZ_TTS_MINI_HF_REPO", "ramput-labs/kyrgyz-tts-mini")
MODELS_DIR = Path(os.environ.get("KYRGYZ_TTS_MINI_MODELS", ROOT / "models"))
OUTPUTS_DIR = Path(os.environ.get("KYRGYZ_TTS_MINI_OUTPUTS", ROOT / "outputs"))
SAMPLES = ROOT / "samples" / "texts.txt"

VOICES = {"woman": MODELS_DIR / "woman.ckpt", "man": MODELS_DIR / "man.ckpt"}
DEFAULT_VOICE = "woman"
VOCODER = MODELS_DIR / "vocoder.pt"


@dataclass(frozen=True)
class Settings:
    temperature: float = 0.667  # sampling variation; 0 = deterministic
    rate: float = 1.0  # length scale; higher is slower
    steps: int = 10  # ODE solver steps; more is slower and slightly cleaner
    denoise: float = 0.00025  # vocoder denoiser strength; 0 = off
