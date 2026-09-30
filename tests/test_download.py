"""Downloader tests with small fake models and fake mirrors: no network, no real weights."""

import hashlib
import shutil
import zipfile

import pytest

import kyrgyz_tts.download as d


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeDrive:
    """`drive / "file-id"` is where that file lives; `calls` lists the requested ids."""

    def __init__(self, folder):
        self.folder = folder
        self.calls = []

    def __truediv__(self, file_id):
        return self.folder / file_id


@pytest.fixture
def drive(monkeypatch, tmp_path):
    """A folder standing in for Google Drive (file id → file); records every request."""
    fake_drive = FakeDrive(tmp_path / "drive")
    fake_drive.folder.mkdir()

    def fake(file_id, out):
        fake_drive.calls.append(file_id)
        if not (fake_drive / file_id).exists():
            raise ConnectionError(f"no such file {file_id}")
        shutil.copyfile(fake_drive / file_id, out)

    monkeypatch.setattr(d, "gdrive_download", fake)
    monkeypatch.setattr(d, "RETRY_DELAY", 0)
    monkeypatch.setattr(d.config, "MODELS_DIR", tmp_path / "models")
    return fake_drive


def file_model(tmp_path, data=b"weights", **overrides):
    fields = dict(name="voice", path=tmp_path / "models" / "voice.ckpt", size=len(data), sha256=digest(data))
    return d.Model(**{**fields, "gdrive": ("primary", "backup"), **overrides})


def folder_model(tmp_path, zip_bytes: bytes, weights=b"w" * 100, **overrides):
    fields = dict(
        name="asr",
        path=tmp_path / "models" / "whisper",
        size=len(zip_bytes),
        sha256=digest(zip_bytes),
        folder=True,
        gdrive=("zip-id",),
        weights=(("model.safetensors", digest(weights)),),
    )
    return d.Model(**{**fields, **overrides})


def make_zip(path, files: dict[str, bytes]) -> bytes:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return path.read_bytes()


def test_file_is_verified_and_installed_atomically(tmp_path, drive):
    (drive / "primary").write_bytes(b"weights")
    model = file_model(tmp_path)

    d.fetch(model)

    assert model.path.read_bytes() == b"weights"
    assert d.problem(model) is None
    assert not list(model.path.parent.glob("*.part"))


def test_falls_back_to_the_next_mirror(tmp_path, drive):
    (drive / "backup").write_bytes(b"weights")  # primary is gone
    model = file_model(tmp_path)

    d.fetch(model)

    assert model.path.read_bytes() == b"weights"
    assert drive.calls == ["primary"] * d.ATTEMPTS + ["backup"]


def test_tampered_mirror_is_skipped_without_retrying(tmp_path, drive):
    (drive / "primary").write_bytes(b"tampered")
    (drive / "backup").write_bytes(b"weights")
    model = file_model(tmp_path)

    d.fetch(model)

    assert model.path.read_bytes() == b"weights"
    assert drive.calls == ["primary", "backup"]


def test_every_mirror_failing_raises_and_leaves_nothing(tmp_path, drive):
    (drive / "primary").write_bytes(b"tampered")
    model = file_model(tmp_path)

    with pytest.raises(d.DownloadError, match="every mirror failed"):
        d.fetch(model)
    assert not model.path.exists()
    assert not list(model.path.parent.glob("*.part"))


def test_transient_error_is_retried(tmp_path, drive, monkeypatch):
    (drive / "primary").write_bytes(b"weights")
    real = d.gdrive_download
    failures = iter([ConnectionError("reset")])

    def flaky(file_id, out):
        if (error := next(failures, None)) is not None:
            raise error
        real(file_id, out)

    monkeypatch.setattr(d, "gdrive_download", flaky)
    d.fetch(file_model(tmp_path))
    assert drive.calls == ["primary"]  # the second attempt reached the mirror


def test_folder_zip_with_or_without_top_folder(tmp_path, drive):
    for i, prefix in enumerate(["whisper/", ""]):
        data = make_zip(drive / "zip-id", {f"{prefix}config.json": b"{}", f"{prefix}model.safetensors": b"w" * 100})
        model = folder_model(tmp_path / str(i), data)

        d.fetch(model)

        assert sorted(p.name for p in model.path.iterdir()) == ["config.json", "model.safetensors"]
        assert d.problem(model) is None
        assert not list(model.path.parent.glob(".extract-*"))


def test_folder_with_wrong_weights_is_rejected(tmp_path, drive):
    data = make_zip(drive / "zip-id", {"config.json": b"{}", "model.safetensors": b"other"})

    with pytest.raises(d.DownloadError):
        d.fetch(folder_model(tmp_path, data))
    assert not (tmp_path / "models" / "whisper").exists()


def test_pack_then_install_roundtrip(tmp_path, drive, monkeypatch):
    source = tmp_path / "installed" / "whisper"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"w" * 100)
    monkeypatch.setattr(d, "MODELS", [folder_model(tmp_path, b"", path=source)])

    d.pack(tmp_path / "upload")
    archive = tmp_path / "upload" / "whisper.zip"
    shutil.copyfile(archive, drive / "zip-id")
    target = folder_model(tmp_path, archive.read_bytes())

    d.fetch(target)
    assert (target.path / "model.safetensors").read_bytes() == b"w" * 100


def test_not_enough_disk_space(tmp_path, drive, monkeypatch):
    monkeypatch.setattr(d.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(10, 10, 0))
    with pytest.raises(d.DownloadError, match="disk space"):
        d.fetch(file_model(tmp_path))


def test_only_one_download_at_a_time(tmp_path, drive):
    with d.download_lock():
        with pytest.raises(d.DownloadError, match="already running"), d.download_lock():
            pass


def test_check_reports_each_model(tmp_path, drive, monkeypatch, capsys):
    good = file_model(tmp_path, name="good", path=tmp_path / "good.ckpt")
    good.path.write_bytes(b"weights")
    corrupt = file_model(tmp_path, name="corrupt", path=tmp_path / "corrupt.ckpt", sha256="0" * 64)
    corrupt.path.write_bytes(b"weights")
    missing = file_model(tmp_path, name="missing", path=tmp_path / "missing.ckpt")
    monkeypatch.setattr(d, "MODELS", [good, corrupt, missing])

    assert d.check() is False
    out = capsys.readouterr().out
    assert "good" in out and "CHECKSUM MISMATCH" in out and "MISSING" in out


def test_manifest_is_consistent():
    names = [m.name for m in d.MODELS]
    assert len(names) == len(set(names))
    for m in d.MODELS:
        assert len(m.sha256) == 64 and m.size > 0
        assert m.gdrive or m.hf_repo, f"{m.name} has no mirror"
