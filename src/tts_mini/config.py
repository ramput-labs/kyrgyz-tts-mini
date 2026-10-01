import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_IN_CHECKOUT = (_REPO / "pyproject.toml").exists()

MODELS_DIR = Path(
    os.environ.get("TTS_MINI_MODELS", _REPO / "models" if _IN_CHECKOUT else Path.home() / ".cache" / "tts-mini")
)
OUTPUTS_DIR = Path(os.environ.get("TTS_MINI_OUTPUTS", _REPO / "outputs" if _IN_CHECKOUT else "outputs"))

VOICES = {"woman": MODELS_DIR / "woman.ckpt", "man": MODELS_DIR / "man.ckpt"}
VOCODER = MODELS_DIR / "hifigan_univ_v1"
