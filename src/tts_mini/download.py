"""Download and verify the model files (they are not stored in git).

  tts-mini download              everything that is missing (safe to re-run)
  tts-mini download --check      verify installed files against their SHA-256
  tts-mini download --pack DIR   copy installed models to DIR for uploading to your own mirror

Every model lists its mirrors in order. A download goes to a temporary file, is checked against its
SHA-256 and only then moved into place, so an interrupted or corrupted download never lands in models/.
To host the files yourself: run --pack, upload the files, share them as "Anyone with the link", and put
the Google Drive file ids (the part between /d/ and /view in the link) first in `gdrive` below.
"""

import argparse
import fcntl
import hashlib
import os
import shutil
import sys
import tempfile
import time
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from tts_mini import config

PROG = "tts-mini"


@dataclass(frozen=True)
class Model:
    name: str
    path: Path
    size: int
    sha256: str  # of the downloaded file (the .zip for folder models)
    gdrive: tuple[str, ...] = ()
    folder: bool = False
    hf_repo: str | None = None
    hf_revision: str | None = None
    weights: tuple[tuple[str, str], ...] = ()  # (file, sha256) checked after unzipping


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
    if model.folder:
        return (model.path / "config.json").exists() and all((model.path / f).exists() for f, _ in model.weights)
    return model.path.is_file() and model.path.stat().st_size == model.size


def problem(model: Model) -> str | None:
    if not is_installed(model):
        return "missing"
    if model.folder:
        for file, expected in model.weights:
            if sha256(model.path / file) != expected:
                return f"{file}: checksum mismatch"
        return None
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
    need = model.size * (2.1 if model.folder else 1.05)
    free = shutil.disk_usage(model.path.parent).free
    if free < need:
        raise DownloadError(f"not enough disk space for {model.name}: need {human(need)}, have {human(free)}")


def gdrive_download(file_id: str, output: Path) -> None:
    import gdown

    if gdown.download(id=file_id, output=str(output), quiet=False, resume=True) is None:
        raise DownloadError("Google Drive refused the download (is it shared as 'Anyone with the link'?)")


def verify_download(model: Model, file: Path) -> None:
    if file.stat().st_size != model.size or sha256(file) != model.sha256:
        file.unlink()
        raise ChecksumError(f"{model.name}: downloaded file does not match the expected SHA-256")


def verify_weights(model: Model, folder: Path) -> None:
    if not (folder / "config.json").exists():
        raise ChecksumError(f"{model.name}: no config.json in the downloaded model")
    for file, expected in model.weights:
        if not (folder / file).exists() or sha256(folder / file) != expected:
            raise ChecksumError(f"{model.name}: {file} does not match the expected SHA-256")


def install_folder(model: Model, source: Path) -> None:
    old = model.path.with_name(model.path.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if model.path.exists():
        model.path.rename(old)
    shutil.move(str(source), model.path)
    shutil.rmtree(old, ignore_errors=True)


def extract(archive: Path, workdir: Path) -> Path:
    with zipfile.ZipFile(archive) as z:
        z.extractall(workdir)
    entries = [p for p in workdir.iterdir() if p.name != "__MACOSX"]
    return entries[0] if len(entries) == 1 and entries[0].is_dir() else workdir


def from_gdrive(model: Model, file_id: str) -> None:
    part = model.path.with_name(model.path.name + (".zip" if model.folder else "") + ".part")
    gdrive_download(file_id, part)
    verify_download(model, part)
    if not model.folder:
        os.replace(part, model.path)
        return
    with tempfile.TemporaryDirectory(dir=model.path.parent, prefix=".extract-") as tmp:
        folder = extract(part, Path(tmp) / "unzipped")
        verify_weights(model, folder)
        install_folder(model, folder)
    part.unlink()


def from_hub(model: Model) -> None:
    from huggingface_hub import snapshot_download

    with tempfile.TemporaryDirectory(dir=model.path.parent, prefix=".hub-") as tmp:
        folder = Path(tmp) / model.path.name
        snapshot_download(model.hf_repo, revision=model.hf_revision, local_dir=folder)
        shutil.rmtree(folder / ".cache", ignore_errors=True)
        verify_weights(model, folder)
        install_folder(model, folder)


def fetch(model: Model, source: str = "auto") -> None:
    mirrors: list[tuple[str, str | None]] = []
    if source in ("auto", "gdrive"):
        mirrors += [("Google Drive", file_id) for file_id in model.gdrive]
    if source in ("auto", "hf") and model.hf_repo:
        mirrors.append(("Hugging Face", None))
    if not mirrors:
        raise DownloadError(f"{model.name}: no mirror for --source {source}")

    model.path.parent.mkdir(parents=True, exist_ok=True)
    ensure_space(model)
    failures = []
    for label, file_id in mirrors:
        delay = RETRY_DELAY
        for attempt in range(1, ATTEMPTS + 1):
            log(f"↓ {model.name} from {label} ({human(model.size)}), attempt {attempt}/{ATTEMPTS}")
            try:
                if file_id:
                    from_gdrive(model, file_id)
                else:
                    from_hub(model)
                log(f"✓ {model.name} → {model.path}")
                return
            except ChecksumError as e:
                failures.append(f"{label}: {e}")
                break  # this mirror serves a different file: retrying will not help
            except Exception as e:
                failures.append(f"{label}: {e}")
                if attempt < ATTEMPTS:
                    log(f"  failed ({e}); retrying in {delay}s")
                    time.sleep(delay)
                    delay *= 2
    raise DownloadError(f"{model.name}: every mirror failed:\n  " + "\n  ".join(failures))


def download(names: list[str] | None = None, source: str = "auto", force: bool = False) -> None:
    selected = [m for m in MODELS if not names or m.name in names]
    with download_lock():
        for model in selected:
            if is_installed(model) and not force:
                log(f"✓ {model.name} already installed")
                continue
            fetch(model, source)


def check() -> bool:
    ok = True
    for model in MODELS:
        issue = problem(model)
        ok &= issue is None
        print(f"{model.name:<22} {'ok' if issue is None else issue.upper():<24} {model.path}")
    return ok


def pack(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        if not is_installed(model):
            log(f"skip {model.name}: not installed")
            continue
        if model.folder:
            target = out / f"{model.path.name}.zip"
            with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as z:
                for file in sorted(p for p in model.path.rglob("*") if p.is_file()):
                    z.write(file, Path(model.path.name) / file.relative_to(model.path))
        else:
            target = out / model.path.name
            shutil.copyfile(model.path, target)
        print(f'{model.name}: upload {target.name}; size={target.stat().st_size:_}, sha256="{sha256(target)}"')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=f"{PROG} download", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    names = [m.name for m in MODELS]
    parser.add_argument("names", nargs="*", metavar="MODEL", help=f"any of: {', '.join(names)} (default: all)")
    parser.add_argument("--source", choices=["auto", "gdrive", "hf"], default="auto", help="mirrors to use")
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
        download(args.names, args.source, args.force)
    except (DownloadError, OSError) as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit("\ninterrupted: run the same command again to resume")


if __name__ == "__main__":
    main()
