"""Downloader tests with small fake models and fake mirrors: no network, no real weights."""

import hashlib
import shutil

import pytest

import tts_mini.download as d


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeDrive:
    def __init__(self, folder):
        self.folder = folder
        self.calls = []

    def __truediv__(self, file_id):
        return self.folder / file_id


@pytest.fixture
def drive(monkeypatch, tmp_path):
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
    fields = dict(
        name="voice",
        path=tmp_path / "models" / "voice.ckpt",
        size=len(data),
        sha256=digest(data),
        gdrive=("primary", "backup"),
    )
    return d.Model(**{**fields, **overrides})


def test_file_is_verified_and_installed_atomically(tmp_path, drive):
    (drive / "primary").write_bytes(b"weights")
    model = file_model(tmp_path)

    d.fetch(model)

    assert model.path.read_bytes() == b"weights"
    assert d.problem(model) is None
    assert not list(model.path.parent.glob("*.part"))


def test_falls_back_to_the_next_mirror(tmp_path, drive):
    (drive / "backup").write_bytes(b"weights")
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
    assert drive.calls == ["primary"]


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
        assert m.gdrive, f"{m.name} has no mirror"


def test_pack_copies_installed_models(tmp_path, monkeypatch, capsys):
    model = file_model(tmp_path, path=tmp_path / "voice.ckpt")
    model.path.write_bytes(b"weights")
    monkeypatch.setattr(d, "MODELS", [model])

    d.pack(tmp_path / "upload")

    assert (tmp_path / "upload" / "voice.ckpt").read_bytes() == b"weights"
    assert digest(b"weights") in capsys.readouterr().out
