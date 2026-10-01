"""Download, verify and upload the models (hosted on Hugging Face).

python -m kyrgyz_tts_mini download            fetch missing models (safe to re-run)
python -m kyrgyz_tts_mini download --check    verify installed files against their SHA-256
python -m kyrgyz_tts_mini upload [--public]   push the local models to the Hugging Face repo
"""

import argparse
import fcntl
import hashlib
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from kyrgyz_tts_mini import config


@dataclass(frozen=True)
class Model:
    name: str
    path: Path
    size: int
    sha256: str

    @property
    def filename(self) -> str:
        return self.path.name


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

ATTEMPTS = 3
RETRY_DELAY = 3  # seconds, doubled per retry

MODEL_CARD = """---
language: ky
pipeline_tag: text-to-speech
license: other
tags: [kyrgyz, tts, kyrgyz-tts-mini]
---

# kyrgyz-tts-mini

Model files for [kyrgyz-tts-mini](https://github.com/ramput-labs/kyrgyz-tts-mini), a small Kyrgyz text-to-speech tool.

| File | |
| --- | --- |
| `woman.ckpt` | female voice |
| `man.ckpt` | male voice |
| `vocoder.pt` | vocoder (mel → audio) |

Download them with `make download` (or `python -m kyrgyz_tts_mini download`) from the GitHub repo.

Voice weights were trained by the National Commission on the State Language under the President of the
Kyrgyz Republic (Mamtil) / Ulutsoft LLC. No license is published; ask them before redistributing or
using commercially.
"""


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


def hub_download(model: Model, force: bool = False) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(config.HF_REPO, model.filename, local_dir=model.path.parent, force_download=force)


def fetch(model: Model, force: bool = False) -> None:
    model.path.parent.mkdir(parents=True, exist_ok=True)
    ensure_space(model)
    delay = RETRY_DELAY
    for attempt in range(1, ATTEMPTS + 1):
        log(f"↓ {model.name} from huggingface.co/{config.HF_REPO} ({human(model.size)}), attempt {attempt}/{ATTEMPTS}")
        try:
            hub_download(model, force=force or attempt > 1)
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
    selected = [m for m in MODELS if not names or m.name in names]
    with download_lock():
        for model in selected:
            if is_installed(model) and not force:
                log(f"✓ {model.name} already installed")
                continue
            fetch(model, force)


def check() -> bool:
    ok = True
    for model in MODELS:
        issue = problem(model)
        ok &= issue is None
        print(f"{model.name:<10} {'ok' if issue is None else issue.upper():<20} {model.path}")
    return ok


def upload(public: bool = False) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    for model in MODELS:
        if issue := problem(model):
            raise DownloadError(f"{model.name}: {issue} ({model.path}); run `make download` or fix it before uploading")

    api = HfApi()
    url = api.create_repo(config.HF_REPO, private=not public, exist_ok=True)
    log(f"↑ uploading {len(MODELS)} models to {url} ({'public' if public else 'private'})")
    operations = [CommitOperationAdd(m.filename, str(m.path)) for m in MODELS]
    operations.append(CommitOperationAdd("README.md", MODEL_CARD.encode()))
    api.create_commit(config.HF_REPO, operations=operations, commit_message="Upload kyrgyz-tts-mini models")
    log(f"✓ done: {url}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m kyrgyz_tts_mini download",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    names = [m.name for m in MODELS]
    parser.add_argument("names", nargs="*", metavar="MODEL", help=f"any of: {', '.join(names)} (default: all)")
    parser.add_argument("--force", action="store_true", help="download again even if installed")
    parser.add_argument("--check", action="store_true", help="verify installed models and exit")
    args = parser.parse_args(argv)
    if unknown := set(args.names) - set(names):
        parser.error(f"unknown model {', '.join(sorted(unknown))}; choose from {', '.join(names)}")

    try:
        if args.check:
            sys.exit(0 if check() else 1)
        download(args.names, args.force)
    except (DownloadError, OSError) as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit("\ninterrupted: run the same command again to resume")
