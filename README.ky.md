# kyrgyz-tts-mini

[English](README.md) · **Кыргызча**

Кыргызча текстти өз компьютериңизде үнгө айландыруучу кичинекей, ылдам курал.

kyrgyz-tts-mini — бул КР Президентине караштуу Мамлекеттик тил боюнча улуттук комиссия (Мамтил) үчүн
[Ulutsoft LLC](https://huggingface.co/UlutSoftLLC/kyrgyz-tts) даярдаган кыргызча үн моделдерин **иштетүүчү
(inference) курал**. Бул үндөрдү биз үйрөткөн жокпуз. Долбоор аларды ыңгайлуу буйрук сабы, Python API жана
веб-интерфейс менен бириктирип, CPU, CUDA же Apple Silicon'до иштетет.

- Эки үн: `woman` (аял) жана `man` (эркек)
- Apple M-сериясынын GPU'сунда реалдуу убакыттан ~20 эсе ылдам
- Бир буйрук менен орнотулат; моделдер [Hugging Face'тен](https://huggingface.co/ramput-labs/kyrgyz-tts-mini) жүктөлүп, SHA-256 менен текшерилет

## Кантип иштейт

```
текст ─► токендер ─► акустикалык модель ─► мел-спектрограмма ─► вокодер ─► 22.05 кГц WAV
                     Ulutsoft үнү                              HiFi-GAN
                     (Matcha-TTS)
```

1. **Текст**: кичине тамгалуу кыргыз кирилл тексти токен id'лерине айланат. Фонемайзер колдонулбайт.
2. **Акустикалык модель**: Ulutsoft'тун [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) модели
   (`woman.ckpt` же `man.ckpt`) мел-спектрограмманы түзөт.
3. **Вокодер**: универсалдуу [HiFi-GAN](https://github.com/jik876/hifi-gan) модели (`vocoder.pt`)
   мел-спектрограмманы аудиого айлантат, ал эми денойзер фондогу ызы-чууну азайтат.

## Тез баштоо

Керектүүлөр: macOS же Linux (Windows'то WSL2), Python 3.11+ жана ~4 ГБ бош орун.

```bash
git clone https://github.com/ramput-labs/kyrgyz-tts-mini.git && cd kyrgyz-tts-mini
make setup    # venv + көз карандылыктар + моделдер + текшерүү
make run      # бир сүйлөмдү эки үн менен окуйт → outputs/
```

<details>
<summary>Make'сиз</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m kyrgyz_tts_mini download    # моделдер → models/
python -m kyrgyz_tts_mini doctor      # баары иштеп жатканын текшерүү
```

</details>

## Колдонуу

### Буйрук сабы

| Make | Make'сиз |
| --- | --- |
| `make speak TEXT="Кош келиңиз!"` | `python -m kyrgyz_tts_mini speak "Кош келиңиз!"` |
| `make speak TEXT="Салам" VOICE=man ARGS=--play` | `python -m kyrgyz_tts_mini speak "Салам" -v man --play` |
| `make speak-file FILE=samples/texts.txt` | `python -m kyrgyz_tts_mini speak -f samples/texts.txt` |
| `make say` (жаз → ук) | `python -m kyrgyz_tts_mini speak` |
| `make web` (веб-интерфейс) | `python -m kyrgyz_tts_mini web` |
| `make demo` (веб-интерфейс, браузерде ачылат) | `python -m kyrgyz_tts_mini web --open` |

Аудио `outputs/` папкасына сакталат (же `-o file.wav` менен башка жерге). `-f` менен файлдын ар бир сабы
окулуп, кыска тыныгуулар менен бир WAV файлга бириктирилет. Веб-интерфейс: http://127.0.0.1:7860.

| `speak` параметри | Демейки | |
| --- | --- | --- |
| `-v, --voice` | `woman` | `woman` же `man` |
| `-o, --output` | `outputs/<убакыт>-<үн>.wav` | натыйжа файлы |
| `-p, --play` | өчүк | натыйжаны угуу |
| `--rate` | `1.0` | сүйлөө ылдамдыгы; чоң сан — жайыраак |
| `--temperature` | `0.667` | ар түрдүүлүк; `0` — ар дайым бирдей |
| `--steps` | `10` | кадамдар саны; көп болсо — жайыраак, бир аз тазараак |
| `--denoise` | `0.00025` | денойзердин күчү; `0` — өчүк |
| `--device` | авто | `cuda`, `mps` же `cpu` |

### Python

```python
from kyrgyz_tts_mini.config import Settings
from kyrgyz_tts_mini.engine import get_tts

tts = get_tts()
tts.synthesize("Саламатсызбы!", "woman").save("salam.wav")
tts.synthesize("Саламатсызбы!", "man", Settings(rate=1.2, temperature=0)).play()
```

`synthesize` — `Speech` объектин кайтарат: `audio` (моно float32), `sample_rate`, `duration`, `save()` / `play()`.
`synthesize_lines` бир нече сапты окуп, тыныгуулар менен бириктирет.

### Текстти кантип жазуу керек

- Кыргыз кирилл тамгалары гана окулат (`а–я`, `ң`, `ө`, `ү` жана негизги тыныш белгилери).
- Сандар жана латын тамгалары өткөрүлөт, ошондуктан сандарды сөз менен жазыңыз: `2024` → `эки миң жыйырма төрт`.
- Узун текстти ар бир сапка бир сүйлөмдөн бөлүңүз — ар бир сап өзүнчө окулуп, жакшыраак угулат.

## Моделдер

`make download` (же `python -m kyrgyz_tts_mini download`) бул файлдарды `models/` папкасына жүктөп, ар
бирин SHA-256 менен текшерет. Кийин `make check` менен кайра текшерсе болот.

| Файл | Көлөмү | Милдети | Түпнуска булагы |
| --- | --- | --- | --- |
| `woman.ckpt` | 219 МБ | аял үнү | [UlutSoftLLC/kyrgyz-tts](https://huggingface.co/UlutSoftLLC/kyrgyz-tts) · `checkpoint_epoch=479.ckpt` |
| `man.ckpt` | 219 МБ | эркек үнү | [UlutSoftLLC/kyrgyz-tts](https://huggingface.co/UlutSoftLLC/kyrgyz-tts) · `checkpoint_epoch=279.ckpt` |
| `vocoder.pt` | 56 МБ | вокодер | [Matcha-TTS checkpoints](https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/tag/v1.0) · HiFi-GAN universal `g_02500000` |

Булар — түпнуска файлдардын өзгөртүлбөгөн көчүрмөлөрү. Алар модель картасы жана лицензиясы менен
[ramput-labs/kyrgyz-tts-mini](https://huggingface.co/ramput-labs/kyrgyz-tts-mini) репозиторийинде сакталат.

## Долбоордун түзүлүшү

```
kyrgyz_tts_mini/
├── cli.py          буйрук сабы: speak, web, doctor, download
├── web.py          Gradio веб-интерфейси
├── engine.py       TTS кыймылдаткычы: текст → Speech
├── text.py         кыргызча текст → токен id'лери
├── download.py     моделдерди жүктөө жана текшерүү
├── config.py       жолдор, үндөр, демейки жөндөөлөр
├── acoustic/       Matcha-TTS акустикалык модели (текст → мел-спектрограмма)
└── vocoder/        HiFi-GAN вокодери жана денойзер (мел-спектрограмма → аудио)
tests/              pytest тесттери (моделдер жок болсо, моделдик тесттер өткөрүлөт)
samples/texts.txt   мисал сүйлөмдөр
```

## Иштеп чыгуу

```bash
make test       # бардык тесттер
make lint       # ruff текшерүүсү (оңдоо үчүн: make format)
make help       # бардык буйруктар
```

| Чөйрө өзгөрмөсү | Демейки | |
| --- | --- | --- |
| `KYRGYZ_TTS_MINI_MODELS` | `models/` | моделдердин папкасы |
| `KYRGYZ_TTS_MINI_OUTPUTS` | `outputs/` | аудио сакталуучу папка |
| `KYRGYZ_TTS_MINI_HF_REPO` | `ramput-labs/kyrgyz-tts-mini` | моделдер жүктөлүүчү Hugging Face репозиторийи |

## Көйгөйлөрдү чечүү

| Көйгөй | Чечими |
| --- | --- |
| `Python 3.11+ not found` | `make setup PYTHON=/path/to/python3.12` |
| жүктөө иштебейт же `404` | huggingface.co менен байланышты текшерип, `make download` кайра иштетиңиз (жүктөлгөн файлдар өткөрүлөт) |
| `--play` иштебейт | үн түзмөгү жок (SSH же сервер); `outputs/` ичиндеги файлды угуңуз |
| башка көйгөй | `make doctor`, же башынан баштаңыз: `make clean-all && make setup` |

## Ыраазычылык

- **Үндөр**: КР Президентине караштуу Мамлекеттик тил боюнча улуттук комиссия (Мамтил) үчүн
  [Ulutsoft LLC](https://huggingface.co/UlutSoftLLC) ([MamtilTTS](https://github.com/UlutSoftLLC/MamtilTTS)).
- **Акустикалык модель**: [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS), Shivam Mehta ж.б.
- **Вокодер**: [HiFi-GAN](https://github.com/jik876/hifi-gan), Jungil Kong ж.б.

## Лицензия

Код: [MIT](LICENSE). Моделдердин салмактары өзүнчө лицензияланат —
[моделдин лицензиясын](https://huggingface.co/ramput-labs/kyrgyz-tts-mini/blob/main/LICENSE) караңыз:

- **Үндөр** (`woman.ckpt`, `man.ckpt`): бардык укуктар Ulutsoft LLC жана Мамтилде калат. Коммерциялык колдонуу боюнча алар менен байланышыңыз.
- **Вокодер** (`vocoder.pt`): MIT, © 2020 Jungil Kong.

Бул үндөрдү чыныгы адамдарды туурап сүйлөтүү же синтезделген кепти чыныгы жазуу катары көрсөтүү үчүн колдонбоңуз.
