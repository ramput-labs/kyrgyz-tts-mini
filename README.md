# kyrgyz-tts-mini

**English** · [Кыргызча](README.ky.md)

Small, fast Kyrgyz text-to-speech. Two voices (`woman`, `man`), runs on CPU, CUDA and Apple Silicon.
Use it from the terminal, from Python, or in a web UI.

## Requirements

- macOS or Linux (Windows: WSL2)
- Python 3.11+
- ~4 GB free disk (models are ~500 MB, downloaded on setup)

## Setup with Make

```bash
git clone https://github.com/ramput-labs/kyrgyz-tts-mini.git && cd kyrgyz-tts-mini
make setup        # venv + install + download models + health check
make run          # say a sentence in both voices → outputs/
```

## Setup without Make

```bash
git clone https://github.com/ramput-labs/kyrgyz-tts-mini.git && cd kyrgyz-tts-mini
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m kyrgyz_tts_mini download   # fetch models into models/
python -m kyrgyz_tts_mini doctor     # check everything works
```

## Examples

| Make | Without Make |
| --- | --- |
| `make speak TEXT="Кош келиңиз!"` | `python -m kyrgyz_tts_mini speak "Кош келиңиз!"` |
| `make speak TEXT="Салам" VOICE=man ARGS=--play` | `python -m kyrgyz_tts_mini speak "Салам" -v man --play` |
| `make speak-file FILE=samples/texts.txt` | `python -m kyrgyz_tts_mini speak -f samples/texts.txt` |
| `make say` (type → listen) | `python -m kyrgyz_tts_mini speak` |
| `make web` (web UI) | `python -m kyrgyz_tts_mini web` |
| `make demo` (Gradio playground, opens the browser) | `python -m kyrgyz_tts_mini web --open` |

Output WAVs go to `outputs/` unless you pass `-o file.wav`. The web UI runs at http://127.0.0.1:7860.

**Python:**

```python
from kyrgyz_tts_mini.audio import save
from kyrgyz_tts_mini.engine import get_tts

speech = get_tts().synthesize("Саламатсызбы!", "woman")
save(speech.audio, speech.sample_rate, "hello.wav")
```

**Options** for `python -m kyrgyz_tts_mini speak`:

| Option | Default | |
| --- | --- | --- |
| `-v, --voice` | `woman` | `woman` or `man` |
| `-o, --output` | `outputs/<time>-<voice>.wav` | output file |
| `-p, --play` | off | play the result |
| `--rate` | `1.0` | higher = slower |
| `--temperature` | `0.667` | variation; `0` = same output every time |
| `--steps` | `10` | quality vs. speed |
| `--device` | auto | `cuda`, `mps` or `cpu` |

## Development

```bash
make test         # all tests (model tests skip if models are missing)
make lint         # or: make format
make help         # every command
```

Without Make: `pytest`.

Env vars: `KYRGYZ_TTS_MINI_MODELS` (models folder), `KYRGYZ_TTS_MINI_OUTPUTS` (output folder), `KYRGYZ_TTS_MINI_HF_REPO` (model repo).

## Models

`make download` fetches the models from [huggingface.co/ramput-labs/kyrgyz-tts-mini](https://huggingface.co/ramput-labs/kyrgyz-tts-mini)
into `models/` and checks every file's SHA-256.

| File | Size | |
| --- | --- | --- |
| `woman.ckpt` | 219 MB | female voice |
| `man.ckpt` | 219 MB | male voice |
| `vocoder.pt` | 56 MB | vocoder |

**Uploading** (maintainers): put the three files in `models/`, then:

```bash
.venv/bin/hf auth login     # token with write access: https://huggingface.co/settings/tokens
make upload                 # private repo; make upload ARGS=--public to publish
```

Without Make: `python -m kyrgyz_tts_mini upload [--public]`. It verifies the checksums, creates the repo if needed and
uploads the files with a model card.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Python 3.11+ not found` | `make setup PYTHON=/path/to/python3.12` |
| `401` / `404` on download | the Hugging Face repo is private or not uploaded yet: `hf auth login`, or set `HF_TOKEN` |
| `--play` does nothing | no audio device (SSH/server); open the file in `outputs/` |
| Anything else | `make doctor`, or `make clean-all && make setup` |

Only Kyrgyz Cyrillic is spoken; digits and Latin letters are skipped. Write numbers as words.

## License

Code: [MIT](LICENSE).
Voice weights: trained by the National Commission on the State Language under the President of the
Kyrgyz Republic (Mamtil) / Ulutsoft LLC. No license is published, so ask them before redistributing or
using commercially.
