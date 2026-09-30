# kyrgyz-tts

**Кыргызча текстти үнгө айландыруу (text-to-speech).** Эки даяр үн бар: аял жана эркек. Үндөр
[Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) архитектурасында үйрөтүлгөн, үндү HiFi-GAN вокодери
түзөт. Долбоор буйрук сабынан (CLI), Python'дон жана Gradio веб-интерфейсинен колдонулат.

- ⚡ Ылдам: Apple Silicon'до 4 секунддук сүйлөм болжол менен 0,15 секундда түзүлөт. CUDA жана CPU'да да иштейт.
- 🧰 Бир буйрук менен орнотулат: `make setup` чөйрөнү түзүп, моделдерди жүктөп, баарын текшерет.
- 🔒 Ишенимдүү жүктөө: ар бир файл SHA-256 менен текшерилет. Бир булак иштебесе, кийинкиси колдонулат.

Кепти текстке айландыруу керек болсо, шериктеш долбоорду караңыз: [kyrgyz-asr](../kyrgyz-asr).

## Тез баштоо

**Талаптар:** macOS же Linux (Windows'то WSL2), Python 3.11+, ~4 ГБ бош орун жана интернет.

```bash
git clone <бул-репозиторий> kyrgyz-tts
cd kyrgyz-tts
make setup      # бир жолу: чөйрө + моделдер (~500 МБ) + текшерүү
make run        # демо: бир эле сүйлөм эки үн менен → outputs/
```

`make setup` бүткөндө «Ready» деген жазуу чыгат. Угуп көрүү үчүн: `make run PLAY=1`.

### `make setup` эмне кылат

1. **Машинаны текшерет** (`scripts/check_env.py`): Python'дун версиясын, `venv` модулун жана дисктеги бош
   орунду. Бир нерсе жетпесе, аны кантип оңдоону айтып берет.
2. **`.venv` чөйрөсүн түзүп**, долбоорду ага орнотот. `pyproject.toml` өзгөргөндө гана кайра орнотот.
3. **Моделдерди жүктөйт** (`models/`): ар бир файл адегенде убактылуу `.part` файлга түшөт, SHA-256
   текшерилет, андан кийин гана ордуна коюлат. Үзүлүп калса, кайра иштеткенде уланат.
4. **Текшерүү жүргүзөт** (`kyrgyz-tts doctor`): түзмөктү жана моделдерди текшерип, чыныгы сүйлөм түзүп көрөт.

Аны каалаган убакта кайра иштетсеңиз болот: бүткөн кадамдар өткөрүлүп жиберилет.

## Колдонуу

```bash
make speak TEXT="Кош келиңиз!" VOICE=man ARGS=--play   # бир сүйлөм
make say                                              # интерактивдүү: жазасыз — угасыз
make speak-file FILE=samples/texts.txt                # файлдагы ар бир сап → бир WAV
make help                                             # бардык буйруктар
```

### Веб-интерфейс (Gradio)

```bash
make demo                  # http://127.0.0.1:7860 — текст жазып, үн тандап, угуп, WAV жүктөп алыңыз
make demo ARGS=--share     # коомдук gradio.live шилтемеси
```

`make demo` Gradio'ну (`.[demo]` кошумчасы) өзү орнотот. Ар бир сап кыска тыныгуу менен окулат;
ылдамдык, temperature, ODE кадамдары жана denoise «Settings» бөлүгүндө.

Же түз эле `.venv/bin/kyrgyz-tts` (же `python -m kyrgyz_tts`):

```bash
kyrgyz-tts speak "Саламатсызбы!" -v woman -o salam.wav --play
kyrgyz-tts speak -f story.txt --rate 1.2        # жайыраак
kyrgyz-tts speak                                # интерактивдүү режим
kyrgyz-tts doctor                               # абалды текшерүү
```

| Параметр | Демейки | Мааниси |
| --- | --- | --- |
| `-v, --voice` | `woman` | `woman` (аял) же `man` (эркек) |
| `-o, --output` | `outputs/<убакыт>-<үн>.wav` | натыйжа сакталуучу файл |
| `-p, --play` | өчүк | натыйжаны дароо угуу |
| `--rate` | `1.0` | узундук коэффициенти: чоң сан — жайыраак |
| `--temperature` | `0.667` | ар түрдүүлүк; `0` — ар дайым бирдей натыйжа |
| `--steps` | `10` | ODE кадамдары (сапат менен ылдамдыктын тең салмагы) |
| `--denoise` | `0.00025` | вокодердин ызы-чуусун басуу; `0` — өчүк |
| `--device` | эң ылайыктуусу | `cuda`, `mps` же `cpu` |

### Python'до

```python
from kyrgyz_tts.audio import save
from kyrgyz_tts.engine import get_tts

speech = get_tts().synthesize("Саламатсызбы!", "woman")   # .audio (float32), .sample_rate (22050)
save(speech.audio, speech.sample_rate, "salam.wav")
```

## Моделдер

Моделдер git'те **сакталбайт**. `make setup` (же `make download`) аларды `models/` папкасына жүктөйт:

| Файл | Көлөмү | Эмне |
| --- | --- | --- |
| `models/woman.ckpt` | 219 МБ | аял үнү |
| `models/man.ckpt` | 219 МБ | эркек үнү |
| `models/hifigan_univ_v1` | 56 МБ | HiFi-GAN вокодери |

```bash
make download    # жетпегендерин жүктөө (бар болсо, өткөрүп жиберет)
make check       # SHA-256 менен кайра текшерүү
```

Ар бир файлдын бир нече булагы (mirror) бар: алгач долбоордун Google Drive'ы, андан кийин баштапкы
шилтемелер. Ар бир булак 3 жолу аракет кылынат. Контролдук суммасы туура келбеген файл эч качан орнотулбайт.
Булактардын тизмеси [`src/kyrgyz_tts/download.py`](src/kyrgyz_tts/download.py) файлындагы `MODELS` бөлүгүндө.

**Моделдерди өз Drive'ыңызга жайгаштыруу:** `kyrgyz-tts download --pack upload/` → `upload/` ичиндеги
файлдарды Drive'га жүктөп, «Anyone with the link» кылып бөлүшүңүз → ар бир файлдын ID'син (шилтемедеги `/d/`
менен `/view` ортосундагы бөлүк) `MODELS` тизмесиндеги `gdrive` катарынын башына кошуңуз.

## Долбоордун түзүлүшү

```
Makefile              setup / run / test ж.б. буйруктар (make help)
scripts/check_env.py  орнотуудан мурунку текшерүү (стандарттык китепкана гана)
scripts/gradio_demo.py  веб-интерфейс (make demo)
src/kyrgyz_tts/
  cli.py              kyrgyz-tts буйругу: speak, doctor, download
  engine.py           TTS: үндөрдү жана вокодерди жүктөйт, synthesize() → Speech
  text.py             кыргызча текст → белгилердин номерлери
  download.py         моделдерди жүктөө, текшерүү, --pack
  config.py           жолдор (чөйрө өзгөрмөлөрү аркылуу өзгөртүлөт)
  audio.py            түзмөк тандоо, сактоо, ойнотуу
  matcha/, hifigan/   Matcha-TTS жана HiFi-GAN (угуу үчүн гана керектүү бөлүгү)
samples/texts.txt     мисал текст
tests/                тесттер
models/, outputs/     моделдер жана натыйжалар (git'ке кирбейт)
```

## Иштеп чыгуу

```bash
make test        # бардык тесттер (моделдер жок болсо, моделдик тесттер өткөрүлүп жиберилет)
make test-fast   # моделсиз тесттер
make lint        # ruff;  make format — автоматтык оңдоо
make clean       # кэштерди тазалоо;  make clean-all — .venv да (models/ калат)
make clean-models CONFIRM=yes   # моделдерди өчүрүү (ырастоо талап кылынат)
```

Чөйрө өзгөрмөлөрү: `KYRGYZ_TTS_MODELS` (моделдердин папкасы), `KYRGYZ_TTS_OUTPUTS` (натыйжалардын папкасы).

## Көйгөйлөрдү чечүү

| Көйгөй | Чечими |
| --- | --- |
| `Python 3.11+ not found` | Python 3.12 орнотуңуз (`brew install python@3.12` же `sudo apt install python3.12 python3.12-venv`) же жолун көрсөтүңүз: `make setup PYTHON=/path/to/python3.12` |
| `venv module is missing` | Debian/Ubuntu: `sudo apt install python3-venv` |
| Google Drive жүктөөнү четке какты | Бир аздан кийин `make download` кайра иштетиңиз, жүктөө калган жеринен уланат. Drive бир файлды көп жолу жүктөсө, убактылуу бөгөйт. |
| `another download is already running` | Башка терминалда жүктөө жүрүп жатат. Ал бүткүчө күтүңүз. |
| `--play` иштебейт | Үн түзмөгү жок (SSH, сервер). Файл `outputs/` папкасына сакталат, ошону угуңуз. |
| Кандайдыр бир нерсе бузулду | `make doctor` эмне туура эмес экенин көрсөтөт. `make clean-all && make setup` баарын кайра түзөт (моделдер кайра жүктөлбөйт). |

**Чектөө:** үндөр кыргыз кирилл тамгаларын жана негизги тыныш белгилерин гана окуйт. Сандар жана латын
тамгалары өткөрүлүп жиберилет, программа аларды эскертет. Сандарды сөз менен жазыңыз: «2024» ордуна «эки миң
жыйырма төрт».

## Лицензия жана ыраазычылык

- Код: [MIT](LICENSE). Matcha-TTS жана HiFi-GAN коду да MIT ([THIRD_PARTY_LICENSE-Matcha-TTS](THIRD_PARTY_LICENSE-Matcha-TTS)).
- **Үн моделдеринин салмактары (Мамтил):** КР Президентине караштуу Мамлекеттик тил боюнча улуттук комиссия, Ulutsoft LLC үйрөткөн.
  Лицензиясы көрсөтүлгөн эмес, ошондуктан кайра таратуудан же коммерциялык колдонуудан мурун ээлеринен уруксат
  сураңыз.
- HiFi-GAN вокодери: Jungil Kong ж.б. (MIT).
