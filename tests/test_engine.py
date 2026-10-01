"""Synthesis with the real models; skipped when they are not downloaded."""

import numpy as np
import pytest

from kyrgyz_tts_mini.models import MODELS, is_installed

pytestmark = pytest.mark.skipif(not all(map(is_installed, MODELS)), reason="models not downloaded (run `make setup`)")


@pytest.fixture(scope="module")
def tts():
    from kyrgyz_tts_mini.engine import get_tts

    return get_tts()


@pytest.mark.parametrize("voice", ["woman", "man"])
def test_synthesizes_speech(tts, voice):
    speech = tts.synthesize("Саламатсызбы!", voice)
    assert speech.sample_rate == 22050
    assert 0.3 < speech.audio_seconds < 5
    assert speech.audio.dtype == np.float32 and np.abs(speech.audio).max() > 0.05


def test_temperature_zero_is_deterministic(tts):
    a = tts.synthesize("Бишкек", "woman", temperature=0).audio
    b = tts.synthesize("Бишкек", "woman", temperature=0).audio
    assert np.array_equal(a, b)


def test_text_without_kyrgyz_letters_is_rejected(tts):
    with pytest.raises(ValueError):
        tts.synthesize("hello 123")
