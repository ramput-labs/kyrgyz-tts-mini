"""Paths, overridable through environment variables."""

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_IN_CHECKOUT = (_REPO / "pyproject.toml").exists()

MODELS_DIR = Path(
    os.environ.get("KYRGYZ_TTS_MODELS", _REPO / "models" if _IN_CHECKOUT else Path.home() / ".cache" / "kyrgyz-tts")
)
OUTPUTS_DIR = Path(os.environ.get("KYRGYZ_TTS_OUTPUTS", _REPO / "outputs" if _IN_CHECKOUT else "outputs"))

# kyrgyz-tts voices (Matcha-TTS) and the universal HiFi-GAN vocoder. Weights: see README credits.
VOICES = {"woman": MODELS_DIR / "woman.ckpt", "man": MODELS_DIR / "man.ckpt"}
VOCODER = MODELS_DIR / "hifigan_univ_v1"
