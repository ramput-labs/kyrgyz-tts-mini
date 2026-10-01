import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

MODELS_DIR = Path(os.environ.get("TTS_MINI_MODELS", _REPO / "models"))
OUTPUTS_DIR = Path(os.environ.get("TTS_MINI_OUTPUTS", _REPO / "outputs"))

VOICES = {"woman": MODELS_DIR / "woman.ckpt", "man": MODELS_DIR / "man.ckpt"}
VOCODER = MODELS_DIR / "hifigan_univ_v1"
