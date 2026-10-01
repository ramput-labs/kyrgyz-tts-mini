import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

HF_REPO = os.environ.get("KYRGYZ_TTS_MINI_HF_REPO", "ramput-labs/kyrgyz-tts-mini")
MODELS_DIR = Path(os.environ.get("KYRGYZ_TTS_MINI_MODELS", _REPO / "models"))
OUTPUTS_DIR = Path(os.environ.get("KYRGYZ_TTS_MINI_OUTPUTS", _REPO / "outputs"))

VOICES = {"woman": MODELS_DIR / "woman.ckpt", "man": MODELS_DIR / "man.ckpt"}
VOCODER = MODELS_DIR / "vocoder.pt"
