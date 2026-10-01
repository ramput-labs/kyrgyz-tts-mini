# kyrgyz-tts-mini

**English** · [Кыргызча](README.ky.md)

A small, fast way to run Kyrgyz text-to-speech on your own machine.

kyrgyz-tts-mini is an **inference toolkit** for the pretrained Kyrgyz voices made by
[Ulutsoft LLC](https://huggingface.co/UlutSoftLLC/kyrgyz-tts) for the National Commission on the State Language
under the President of the Kyrgyz Republic (Mamtil). We did not train these voices. This project packages them
with a clean command line, a Python API and a web UI, and runs them on CPU, CUDA or Apple Silicon.

- Two voices: `woman` and `man`
- About 20× faster than real time on an Apple M-series GPU
- One-command setup; models download from [Hugging Face](https://huggingface.co/ramput-labs/kyrgyz-tts-mini) and are checked against their SHA-256

## How it works

```
text ─► tokens ─► acoustic model ─► mel-spectrogram ─► vocoder ─► 22.05 kHz WAV
                  Ulutsoft voice                       HiFi-GAN
                  (Matcha-TTS)
```

1. **Text**: lowercase Kyrgyz Cyrillic is mapped to token ids. No phonemizer is used.
2. **Acoustic model**: Ulutsoft's [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) checkpoint
   (`woman.ckpt` or `man.ckpt`) predicts a mel-spectrogram with conditional flow matching.
3. **Vocoder**: the universal [HiFi-GAN](https://github.com/jik876/hifi-gan) checkpoint (`vocoder.pt`)
   turns the mel-spectrogram into audio, and a light denoiser removes background hiss.

## Quick start

Requires macOS or Linux (on Windows, use WSL2), Python 3.11+ and about 4 GB of free disk.

```bash
git clone https://github.com/ramput-labs/kyrgyz-tts-mini.git && cd kyrgyz-tts-mini
make setup    # venv + dependencies + models + health check
make run      # one sentence in both voices → outputs/
```

<details>
<summary>Without Make</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m kyrgyz_tts_mini download    # models → models/
python -m kyrgyz_tts_mini doctor      # check that everything works
```

</details>

## Usage

### Command line

| Make | Without Make |
| --- | --- |
| `make speak TEXT="Кош келиңиз!"` | `python -m kyrgyz_tts_mini speak "Кош келиңиз!"` |
| `make speak TEXT="Салам" VOICE=man ARGS=--play` | `python -m kyrgyz_tts_mini speak "Салам" -v man --play` |
| `make speak-file FILE=samples/texts.txt` | `python -m kyrgyz_tts_mini speak -f samples/texts.txt` |
| `make say` (type a line, hear it) | `python -m kyrgyz_tts_mini speak` |
| `make web` (web UI) | `python -m kyrgyz_tts_mini web` |
| `make demo` (web UI, opens the browser) | `python -m kyrgyz_tts_mini web --open` |

Audio is saved to `outputs/` unless you pass `-o file.wav`. With `-f`, every line of the file is spoken and the
results are joined into one WAV with short pauses. The web UI runs at http://127.0.0.1:7860.

| `speak` option | Default | |
| --- | --- | --- |
| `-v, --voice` | `woman` | `woman` or `man` |
| `-o, --output` | `outputs/<time>-<voice>.wav` | output file |
| `-p, --play` | off | play the result |
| `--rate` | `1.0` | speaking pace; higher is slower |
| `--temperature` | `0.667` | variation; `0` gives the same output every time |
| `--steps` | `10` | solver steps; more is slower and slightly cleaner |
| `--denoise` | `0.00025` | denoiser strength; `0` turns it off |
| `--device` | auto | `cuda`, `mps` or `cpu` |

### Python

```python
from kyrgyz_tts_mini.config import Settings
from kyrgyz_tts_mini.engine import get_tts

tts = get_tts()
tts.synthesize("Саламатсызбы!", "woman").save("hello.wav")
tts.synthesize("Саламатсызбы!", "man", Settings(rate=1.2, temperature=0)).play()
```

`synthesize` returns a `Speech` with `audio` (mono float32), `sample_rate`, `duration`, and `save()` / `play()`.
`synthesize_lines` speaks several lines and joins them with pauses.

### Writing text for the voices

- Only Kyrgyz Cyrillic is spoken (`а–я`, `ң`, `ө`, `ү` and basic punctuation).
- Digits and Latin letters are skipped, so write numbers as words: `2024` → `эки миң жыйырма төрт`.
- For long text, put one sentence per line. Each line is synthesized separately, which sounds better.

## Models

`make download` (or `python -m kyrgyz_tts_mini download`) puts these files in `models/`. Each one is checked
against its SHA-256. `make check` verifies them again later.

| File | Size | Role | Original source |
| --- | --- | --- | --- |
| `woman.ckpt` | 219 MB | female voice | [UlutSoftLLC/kyrgyz-tts](https://huggingface.co/UlutSoftLLC/kyrgyz-tts) · `checkpoint_epoch=479.ckpt` |
| `man.ckpt` | 219 MB | male voice | [UlutSoftLLC/kyrgyz-tts](https://huggingface.co/UlutSoftLLC/kyrgyz-tts) · `checkpoint_epoch=279.ckpt` |
| `vocoder.pt` | 56 MB | vocoder | [Matcha-TTS checkpoints](https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/tag/v1.0) · HiFi-GAN universal `g_02500000` |

They are unmodified copies, mirrored at [ramput-labs/kyrgyz-tts-mini](https://huggingface.co/ramput-labs/kyrgyz-tts-mini)
with a model card and license.

## Project layout

```
kyrgyz_tts_mini/
├── cli.py          command line: speak, web, doctor, download
├── web.py          Gradio web UI
├── engine.py       TTS engine: text → Speech
├── text.py         Kyrgyz text → token ids
├── download.py     model download and checksum verification
├── config.py       paths, voices, default settings
├── acoustic/       Matcha-TTS acoustic model (text → mel-spectrogram)
└── vocoder/        HiFi-GAN vocoder and denoiser (mel-spectrogram → audio)
tests/              pytest suite (model tests are skipped if the models are missing)
samples/texts.txt   example sentences
```

## Development

```bash
make test       # all tests
make lint       # ruff lint and format check (make format to fix)
make help       # every command
```

| Environment variable | Default | |
| --- | --- | --- |
| `KYRGYZ_TTS_MINI_MODELS` | `models/` | where the models live |
| `KYRGYZ_TTS_MINI_OUTPUTS` | `outputs/` | where audio is saved |
| `KYRGYZ_TTS_MINI_HF_REPO` | `ramput-labs/kyrgyz-tts-mini` | Hugging Face repo to download from |

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Python 3.11+ not found` | `make setup PYTHON=/path/to/python3.12` |
| download fails or `404` | check your connection to huggingface.co, then run `make download` again (finished files are skipped) |
| `--play` does nothing | no audio device (SSH or server); open the file in `outputs/` |
| anything else | `make doctor`, or start over with `make clean-all && make setup` |

## Credits

- **Voices**: [Ulutsoft LLC](https://huggingface.co/UlutSoftLLC) for Mamtil, the National Commission on the State
  Language under the President of the Kyrgyz Republic ([MamtilTTS](https://github.com/UlutSoftLLC/MamtilTTS)).
- **Acoustic model**: [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) by Shivam Mehta et al.
- **Vocoder**: [HiFi-GAN](https://github.com/jik876/hifi-gan) by Jungil Kong et al.

## License

The code is [MIT](LICENSE). The model weights are licensed separately; see the
[model license](https://huggingface.co/ramput-labs/kyrgyz-tts-mini/blob/main/LICENSE):

- **Voices** (`woman.ckpt`, `man.ckpt`): all rights stay with Ulutsoft LLC and Mamtil. Contact them about commercial use.
- **Vocoder** (`vocoder.pt`): MIT, © 2020 Jungil Kong.

Do not use these voices to impersonate real people or to pass off synthetic speech as a real recording.
