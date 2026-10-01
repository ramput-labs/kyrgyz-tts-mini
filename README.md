# tts-mini

**English** · [Кыргызча](README.ky.md)

Small, fast Kyrgyz text-to-speech. Two voices (`woman`, `man`), runs on CPU, CUDA and Apple Silicon.
Use it from the terminal, from Python, or in a web UI.

## Requirements

- macOS or Linux (Windows: WSL2)
- Python 3.11+
- ~4 GB free disk (models are ~500 MB, downloaded on setup)

## Setup with Make

```bash
git clone <repo-url> tts-mini && cd tts-mini
make setup        # venv + install + download models + health check
make run          # say a sentence in both voices → outputs/
```

## Setup without Make

```bash
git clone <repo-url> tts-mini && cd tts-mini
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m tts_mini download   # fetch models into models/
python -m tts_mini doctor     # check everything works
```

## Examples

| Make | Without Make |
| --- | --- |
| `make speak TEXT="Кош келиңиз!"` | `python -m tts_mini speak "Кош келиңиз!"` |
| `make speak TEXT="Салам" VOICE=man ARGS=--play` | `python -m tts_mini speak "Салам" -v man --play` |
| `make speak-file FILE=samples/texts.txt` | `python -m tts_mini speak -f samples/texts.txt` |
| `make say` (type → listen) | `python -m tts_mini speak` |
| `make web` (web UI) | `python -m tts_mini web` |

Output WAVs go to `outputs/` unless you pass `-o file.wav`. The web UI runs at http://127.0.0.1:7860.

**Python:**

```python
from tts_mini.audio import save
from tts_mini.engine import get_tts

speech = get_tts().synthesize("Саламатсызбы!", "woman")
save(speech.audio, speech.sample_rate, "hello.wav")
```

**Options** for `python -m tts_mini speak`:

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

Env vars: `TTS_MINI_MODELS` (models folder), `TTS_MINI_OUTPUTS` (output folder).

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Python 3.11+ not found` | `make setup PYTHON=/path/to/python3.12` |
| Google Drive refused the download | wait a bit, run `make download` again (it resumes) |
| `--play` does nothing | no audio device (SSH/server); open the file in `outputs/` |
| Anything else | `make doctor`, or `make clean-all && make setup` |

Only Kyrgyz Cyrillic is spoken; digits and Latin letters are skipped. Write numbers as words.

## License

Code: [MIT](LICENSE).
Voice weights: trained by the National Commission on the State Language under the President of the
Kyrgyz Republic (Mamtil) / Ulutsoft LLC. No license is published, so ask them before redistributing or
using commercially.
