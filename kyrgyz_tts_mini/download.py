"""Download the models from Hugging Face and verify them against their SHA-256.

The files are unmodified copies of the originals: the voices by Ulutsoft LLC / Mamtil
(huggingface.co/UlutSoftLLC/kyrgyz-tts) and the HiFi-GAN universal vocoder from Matcha-TTS.
"""

import fcntl
import hashlib
import shutil
import sys
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from kyrgyz_tts_mini import config

HUB = f"https://huggingface.co/{config.HF_REPO}/resolve/main"
ATTEMPTS = 3
RETRY_DELAY = 3  # seconds, doubled per retry


@dataclass(frozen=True)
class Model:
    name: str
    path: Path
    size: int
    sha256: str

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def url(self) -> str:
        return f"{HUB}/{self.filename}"


MODELS = [
    Model(
        "woman",
        config.VOICES["woman"],
        size=218_842_178,
        sha256="e951355ab0e5d07aac026c38dd96ce0712251ca74360625d94a85b7d3170fd3d",
    ),
    Model(
        "man",
        config.VOICES["man"],
        size=218_841_284,
        sha256="f31782d348be5177911193f3f322bd6e076ff5b89ecd027880029089fb24e7f7",
    ),
    Model(
        "vocoder",
        config.VOCODER,
        size=55_788_858,
        sha256="771eaf4876485a35e25577563d390c262e23c2421e4a8c929eacfde34a5b7a60",
    ),
]


class DownloadError(RuntimeError):
    pass


class ChecksumError(DownloadError):
    pass


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def human(size: float) -> str:
    return f"{size / 1e9:.1f} GB" if size >= 1e9 else f"{size / 1e6:.0f} MB"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def is_installed(model: Model) -> bool:
    return model.path.is_file() and model.path.stat().st_size == model.size


def problem(model: Model) -> str | None:
    if not is_installed(model):
        return "missing"
    return None if sha256(model.path) == model.sha256 else "checksum mismatch"


@contextmanager
def download_lock() -> Iterator[None]:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with (config.MODELS_DIR / ".download.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DownloadError("another download is already running for this models folder") from None
        yield


def ensure_space(model: Model) -> None:
    need = model.size * 1.05
    free = shutil.disk_usage(model.path.parent).free
    if free < need:
        raise DownloadError(f"not enough disk space for {model.name}: need {human(need)}, have {human(free)}")


def transfer(model: Model) -> None:
    part = model.path.with_name(model.filename + ".part")
    with urllib.request.urlopen(model.url, timeout=60) as response, part.open("wb") as f:
        shutil.copyfileobj(response, f, 1 << 20)
    part.replace(model.path)


def fetch(model: Model) -> None:
    model.path.parent.mkdir(parents=True, exist_ok=True)
    ensure_space(model)
    delay = RETRY_DELAY
    for attempt in range(1, ATTEMPTS + 1):
        log(f"↓ {model.name} from {urlsplit(model.url).netloc} ({human(model.size)}), attempt {attempt}/{ATTEMPTS}")
        try:
            transfer(model)
            if problem(model) is not None:
                model.path.unlink(missing_ok=True)
                raise ChecksumError(f"{model.name}: downloaded file does not match the expected SHA-256")
            log(f"✓ {model.name} → {model.path}")
            return
        except ChecksumError:
            raise
        except Exception as e:
            if attempt == ATTEMPTS:
                raise DownloadError(f"{model.name}: download failed: {e}") from e
            log(f"  failed ({e}); retrying in {delay}s")
            time.sleep(delay)
            delay *= 2


def download(names: list[str] | None = None, force: bool = False) -> None:
    with download_lock():
        for model in MODELS:
            if names and model.name not in names:
                continue
            if is_installed(model) and not force:
                log(f"✓ {model.name} already installed")
                continue
            fetch(model)


def check() -> bool:
    ok = True
    for model in MODELS:
        issue = problem(model)
        ok &= issue is None
        print(f"{model.name:<10} {'ok' if issue is None else issue.upper():<20} {model.path}")
    return ok
