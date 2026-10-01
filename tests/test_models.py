"""Model download tests with a fake Hugging Face repo: no network, no real weights."""

import hashlib
import shutil

import pytest

import kyrgyz_tts_mini.models as m


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeHub:
    """A folder standing in for the Hugging Face repo; `calls` lists the requested files."""

    def __init__(self, folder):
        self.folder = folder
        self.calls = []

    def __truediv__(self, filename):
        return self.folder / filename


@pytest.fixture
def hub(monkeypatch, tmp_path):
    folder = FakeHub(tmp_path / "hub")
    folder.folder.mkdir()

    def fake(model, force=False):
        folder.calls.append(model.filename)
        if not (folder / model.filename).exists():
            raise ConnectionError(f"404 {model.filename}")
        shutil.copyfile(folder / model.filename, model.path)

    monkeypatch.setattr(m, "hub_download", fake)
    monkeypatch.setattr(m, "RETRY_DELAY", 0)
    monkeypatch.setattr(m.config, "MODELS_DIR", tmp_path / "models")
    return folder


def model(tmp_path, data=b"weights", **overrides):
    fields = dict(name="voice", path=tmp_path / "models" / "voice.ckpt", size=len(data), sha256=digest(data))
    return m.Model(**{**fields, **overrides})


def test_downloads_and_verifies(tmp_path, hub):
    (hub / "voice.ckpt").write_bytes(b"weights")
    voice = model(tmp_path)

    m.fetch(voice)

    assert voice.path.read_bytes() == b"weights"
    assert m.problem(voice) is None


def test_tampered_file_is_rejected_and_removed(tmp_path, hub):
    (hub / "voice.ckpt").write_bytes(b"tampered")
    voice = model(tmp_path, size=len(b"tampered"))

    with pytest.raises(m.ChecksumError):
        m.fetch(voice)
    assert not voice.path.exists()
    assert hub.calls == ["voice.ckpt"]


def test_network_errors_are_retried_then_reported(tmp_path, hub):
    with pytest.raises(m.DownloadError, match="download failed"):
        m.fetch(model(tmp_path))
    assert hub.calls == ["voice.ckpt"] * m.ATTEMPTS


def test_transient_error_is_retried(tmp_path, hub, monkeypatch):
    (hub / "voice.ckpt").write_bytes(b"weights")
    real = m.hub_download
    failures = iter([ConnectionError("reset")])

    def flaky(model, force=False):
        if (error := next(failures, None)) is not None:
            raise error
        real(model, force)

    monkeypatch.setattr(m, "hub_download", flaky)
    m.fetch(model(tmp_path))
    assert hub.calls == ["voice.ckpt"]


def test_installed_models_are_skipped(tmp_path, hub, monkeypatch):
    voice = model(tmp_path)
    voice.path.parent.mkdir(parents=True)
    voice.path.write_bytes(b"weights")
    monkeypatch.setattr(m, "MODELS", [voice])

    m.download()
    assert hub.calls == []


def test_not_enough_disk_space(tmp_path, hub, monkeypatch):
    monkeypatch.setattr(m.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(10, 10, 0))
    with pytest.raises(m.DownloadError, match="disk space"):
        m.fetch(model(tmp_path))


def test_only_one_download_at_a_time(tmp_path, hub):
    with m.download_lock():
        with pytest.raises(m.DownloadError, match="already running"), m.download_lock():
            pass


def test_check_reports_each_model(tmp_path, monkeypatch, capsys):
    good = model(tmp_path, name="good", path=tmp_path / "good.ckpt")
    good.path.write_bytes(b"weights")
    corrupt = model(tmp_path, name="corrupt", path=tmp_path / "corrupt.ckpt", sha256="0" * 64)
    corrupt.path.write_bytes(b"weights")
    missing = model(tmp_path, name="missing", path=tmp_path / "missing.ckpt")
    monkeypatch.setattr(m, "MODELS", [good, corrupt, missing])

    assert m.check() is False
    out = capsys.readouterr().out
    assert "good" in out and "CHECKSUM MISMATCH" in out and "MISSING" in out


def test_upload_refuses_missing_or_corrupt_models(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "MODELS", [model(tmp_path)])
    with pytest.raises(m.DownloadError, match="missing"):
        m.upload()


def test_manifest_is_consistent():
    names = [x.name for x in m.MODELS]
    files = [x.filename for x in m.MODELS]
    assert len(set(names)) == len(names) and len(set(files)) == len(files)
    for x in m.MODELS:
        assert len(x.sha256) == 64 and x.size > 0
