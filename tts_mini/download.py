"""Download and verify the models.

  python -m tts_mini download              everything that is missing (safe to re-run)
  python -m tts_mini download --check      verify installed files against their SHA-256
  python -m tts_mini download --pack DIR   copy installed models to DIR to host your own mirror

Downloads land in a .part file and are moved into models/ only after the SHA-256 matches.
To self-host: run --pack, upload to Google Drive ("Anyone with the link"), and put the file ids first in `gdrive`.
"""

import argparse
import fcntl
import hashlib
import os
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from tts_mini import config

PROG = "python -m tts_mini"


@dataclass(frozen=True)
class Model:
    name: str
    path: Path
    size: int
    sha256: str
    gdrive: tuple[str, ...]  # Google Drive file ids, tried in order


MODELS = [
    Model(
        "woman",
        config.VOICES["woman"],
        size=218_842_178,
        sha256="e951355ab0e5d07aac026c38dd96ce0712251ca74360625d94a85b7d3170fd3d",
        gdrive=("1Jqf7lDXgbBqClfQ0xhw_mQOPwL0UgEPi", "1aaaspyNOYZGdoyWhkhUa6RMsCzIzu7fC"),
    ),
    Model(
        "man",
        config.VOICES["man"],
        size=218_841_284,
        sha256="f31782d348be5177911193f3f322bd6e076ff5b89ecd027880029089fb24e7f7",
        gdrive=("1s1TXC3K1UqAFG_PY0Doo8U0UtpkjKDRw", "1tBs7ZVuGq3_jlsHCLT1zs_A4u-Lv7ggl"),
    ),
    Model(
        "vocoder",
        config.VOCODER,
        size=55_788_858,
        sha256="771eaf4876485a35e25577563d390c262e23c2421e4a8c929eacfde34a5b7a60",
        gdrive=("1RqfYM-yfFh6mdrkuWz8Ve8QV6PkHRYvZ", "1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW"),
    ),
]

ATTEMPTS = 3
RETRY_DELAY = 3  # seconds, doubled per retry


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


def gdrive_download(file_id: str, output: Path) -> None:
    import gdown

    if gdown.download(id=file_id, output=str(output), quiet=False, resume=True) is None:
        raise DownloadError("Google Drive refused the download (is it shared as 'Anyone with the link'?)")


def from_gdrive(model: Model, file_id: str) -> None:
    part = model.path.with_name(model.path.name + ".part")
    gdrive_download(file_id, part)
    if part.stat().st_size != model.size or sha256(part) != model.sha256:
        part.unlink()
        raise ChecksumError(f"{model.name}: downloaded file does not match the expected SHA-256")
    os.replace(part, model.path)


def fetch(model: Model) -> None:
    if not model.gdrive:
        raise DownloadError(f"{model.name}: no mirror configured")
    model.path.parent.mkdir(parents=True, exist_ok=True)
    ensure_space(model)
    failures = []
    for mirror, file_id in enumerate(model.gdrive, 1):
        delay = RETRY_DELAY
        for attempt in range(1, ATTEMPTS + 1):
            log(f"↓ {model.name} from mirror {mirror} ({human(model.size)}), attempt {attempt}/{ATTEMPTS}")
            try:
                from_gdrive(model, file_id)
                log(f"✓ {model.name} → {model.path}")
                return
            except ChecksumError as e:
                failures.append(f"mirror {mirror}: {e}")
                break  # this mirror serves a different file: retrying will not help
            except Exception as e:
                failures.append(f"mirror {mirror}: {e}")
                if attempt < ATTEMPTS:
                    log(f"  failed ({e}); retrying in {delay}s")
                    time.sleep(delay)
                    delay *= 2
    raise DownloadError(f"{model.name}: every mirror failed:\n  " + "\n  ".join(failures))


def download(names: list[str] | None = None, force: bool = False) -> None:
    selected = [m for m in MODELS if not names or m.name in names]
    with download_lock():
        for model in selected:
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


def pack(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        if not is_installed(model):
            log(f"skip {model.name}: not installed")
            continue
        target = out / model.path.name
        shutil.copyfile(model.path, target)
        print(f'{model.name}: upload {target.name}; size={target.stat().st_size:_}, sha256="{sha256(target)}"')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=f"{PROG} download", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    names = [m.name for m in MODELS]
    parser.add_argument("names", nargs="*", metavar="MODEL", help=f"any of: {', '.join(names)} (default: all)")
    parser.add_argument("--force", action="store_true", help="download again even if installed")
    parser.add_argument("--check", action="store_true", help="verify installed models and exit")
    parser.add_argument("--pack", metavar="DIR", type=Path, help="copy installed models to DIR for uploading")
    args = parser.parse_args(argv)
    if unknown := set(args.names) - set(names):
        parser.error(f"unknown model {', '.join(sorted(unknown))}; choose from {', '.join(names)}")

    try:
        if args.check:
            sys.exit(0 if check() else 1)
        if args.pack:
            pack(args.pack)
            return
        download(args.names, args.force)
    except (DownloadError, OSError) as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit("\ninterrupted: run the same command again to resume")


if __name__ == "__main__":
    main()
