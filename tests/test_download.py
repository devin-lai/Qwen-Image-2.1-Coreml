"""Download retries must not treat existing directories as complete packages."""

import sys
from types import SimpleNamespace

import pytest

import download_models


def _write_package(dest, name):
    for filename in (
        "Manifest.json",
        "Data/com.apple.CoreML/model.mlmodel",
        "Data/com.apple.CoreML/weights/weight.bin",
    ):
        path = dest / name / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"downloaded")


@pytest.mark.parametrize("complete", [False, True])
def test_existing_packages_still_check_requested_revision(tmp_path, monkeypatch, complete):
    name = download_models.PACKAGES["vae"][0]
    (tmp_path / name).mkdir()
    if complete:
        _write_package(tmp_path, name)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        _write_package(tmp_path, name)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setattr(sys, "argv", [
        "download_models.py", "--dest", str(tmp_path), "--only", "vae", "--revision", "release-v2",
    ])
    download_models.main()
    assert len(calls) == 1
    assert calls[0]["revision"] == "release-v2"
    assert calls[0]["allow_patterns"] == [f"{name}/**", "LICENSE", "Notice", "SHA256SUMS"]


def test_incomplete_download_exits_with_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda **kwargs: None))
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--dest", str(tmp_path), "--only", "vae"])
    with pytest.raises(SystemExit, match="incomplete package"):
        download_models.main()
