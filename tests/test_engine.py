import numpy as np
import pytest

from kyrgyz_tts_mini.config import Settings
from kyrgyz_tts_mini.download import MODELS, is_installed
from kyrgyz_tts_mini.engine import Speech

needs_models = pytest.mark.skipif(not all(map(is_installed, MODELS)), reason="models not downloaded (run `make setup`)")


def test_join_inserts_pauses_between_parts():
    parts = [Speech(np.ones(10, np.float32), 100, 1.0), Speech(np.ones(5, np.float32), 100, 0.5)]
    joined = Speech.join(parts, pause=0.1)
    assert len(joined.audio) == 10 + 10 + 5
    assert joined.elapsed == 1.5 and joined.audio[10:20].sum() == 0


def test_save_writes_a_wav(tmp_path):
    path = Speech(np.zeros(2205, np.float32), 22050, 0.0).save(tmp_path / "out" / "a.wav")
    assert path.read_bytes()[:4] == b"RIFF"


@pytest.fixture(scope="module")
def tts():
    from kyrgyz_tts_mini.engine import get_tts

    return get_tts()


@needs_models
@pytest.mark.parametrize("voice", ["woman", "man"])
def test_synthesizes_speech(tts, voice):
    speech = tts.synthesize("Саламатсызбы!", voice)
    assert speech.sample_rate == 22050
    assert 0.3 < speech.duration < 5
    assert speech.audio.dtype == np.float32 and np.abs(speech.audio).max() > 0.05


@needs_models
def test_temperature_zero_is_deterministic(tts):
    a = tts.synthesize("Бишкек", settings=Settings(temperature=0)).audio
    b = tts.synthesize("Бишкек", settings=Settings(temperature=0)).audio
    assert np.array_equal(a, b)


@needs_models
def test_text_without_kyrgyz_letters_is_rejected(tts):
    with pytest.raises(ValueError):
        tts.synthesize("hello 123")
    with pytest.raises(ValueError):
        tts.synthesize_lines(["", "  "])
