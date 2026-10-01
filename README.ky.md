# tts-mini

[English](README.md) · **Кыргызча**

Кыргызча текстти үнгө айландыруучу кичинекей, ылдам курал. Эки үн бар (`woman` — аял, `man` — эркек).
CPU, CUDA жана Apple Silicon'до иштейт. Терминалдан, Python'дон же веб-интерфейстен колдонсо болот.

## Талаптар

- macOS же Linux (Windows'то WSL2)
- Python 3.11+
- ~4 ГБ бош орун (моделдер ~500 МБ, орнотууда жүктөлөт)

## Make менен орнотуу

```bash
git clone <repo-url> tts-mini && cd tts-mini
make setup        # venv + орнотуу + моделдерди жүктөө + текшерүү
make run          # бир сүйлөмдү эки үн менен окуйт → outputs/
```

## Make'сиз орнотуу

```bash
git clone <repo-url> tts-mini && cd tts-mini
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
tts-mini download     # моделдерди models/ папкасына жүктөө
tts-mini doctor       # баары иштеп жатканын текшерүү
```

## Мисалдар

| Make | Make'сиз |
| --- | --- |
| `make speak TEXT="Кош келиңиз!"` | `tts-mini speak "Кош келиңиз!"` |
| `make speak TEXT="Салам" VOICE=man ARGS=--play` | `tts-mini speak "Салам" -v man --play` |
| `make speak-file FILE=samples/texts.txt` | `tts-mini speak -f samples/texts.txt` |
| `make say` (жаз → ук) | `tts-mini speak` |
| `make demo` (веб-интерфейс) | `pip install -e ".[demo]" && python scripts/gradio_demo.py` |

WAV файлдар `outputs/` папкасына сакталат (же `-o file.wav` менен башка жерге). Веб-интерфейс:
http://127.0.0.1:7860.

**Python'до:**

```python
from tts_mini.audio import save
from tts_mini.engine import get_tts

speech = get_tts().synthesize("Саламатсызбы!", "woman")
save(speech.audio, speech.sample_rate, "salam.wav")
```

**`tts-mini speak` параметрлери:**

| Параметр | Демейки | |
| --- | --- | --- |
| `-v, --voice` | `woman` | `woman` же `man` |
| `-o, --output` | `outputs/<убакыт>-<үн>.wav` | натыйжа файлы |
| `-p, --play` | өчүк | натыйжаны угуу |
| `--rate` | `1.0` | чоң сан — жайыраак |
| `--temperature` | `0.667` | ар түрдүүлүк; `0` — ар дайым бирдей |
| `--steps` | `10` | сапат менен ылдамдык |
| `--device` | авто | `cuda`, `mps` же `cpu` |

## Иштеп чыгуу

```bash
make test         # бардык тесттер (моделдер жок болсо, моделдик тесттер өткөрүлөт)
make lint         # же: make format
make help         # бардык буйруктар
```

Make'сиз: `pytest`, `ruff check src tests scripts`.

Чөйрө өзгөрмөлөрү: `TTS_MINI_MODELS` (моделдердин папкасы), `TTS_MINI_OUTPUTS` (натыйжалардын папкасы).

## Көйгөйлөрдү чечүү

| Көйгөй | Чечими |
| --- | --- |
| `Python 3.11+ not found` | `make setup PYTHON=/path/to/python3.12` |
| Google Drive жүктөөнү четке какты | бир аз күтүп, `make download` кайра иштетиңиз (калган жеринен уланат) |
| `--play` иштебейт | үн түзмөгү жок (SSH/сервер); `outputs/` ичиндеги файлды угуңуз |
| Башка көйгөй | `make doctor`, же `make clean-all && make setup` |

Кыргыз кирилл тамгалары гана окулат; сандар жана латын тамгалары өткөрүлөт. Сандарды сөз менен жазыңыз.

## Лицензия

Код: [MIT](LICENSE). Моделдин кээ бир коду ачык булактуу долбоорлордон алынган, караңыз:
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
Үн моделдери: КР Президентине караштуу Мамлекеттик тил боюнча улуттук комиссия (Мамтил) / Ulutsoft LLC
үйрөткөн. Лицензиясы жарыяланган эмес, ошондуктан кайра таратуудан же коммерциялык колдонуудан мурун
алардан уруксат сураңыз.
