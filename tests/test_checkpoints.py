import hashlib
from pathlib import Path

import pytest

from local_jev import checkpoints
from local_jev.errors import ChecksumMismatch, NotDownloaded

REPO = "convaiinnovations/laya"
REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"


def make_files(root: Path, name: str, weights: bytes = b"weights") -> None:
    folder = checkpoints.checkpoint_dir(root, name)
    for item in checkpoints.REQUIRED:
        path = folder / item
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(weights if item == "model.safetensors" else b"{}")


def test_patterns_are_scoped_to_the_checkpoint_subfolder():
    assert checkpoints.patterns("english")[0] == "model.safetensors"
    assert all(
        pattern.startswith("multilingual/") for pattern in checkpoints.patterns("multilingual")
    )


def test_snapshot_root_follows_the_hugging_face_cache_layout(tmp_path):
    assert checkpoints.snapshot_root(REPO, REVISION, tmp_path) == (
        tmp_path / "models--convaiinnovations--laya" / "snapshots" / REVISION
    )


def test_fetch_uses_the_cache_without_downloading(tmp_path, monkeypatch):
    root = checkpoints.snapshot_root(REPO, REVISION, tmp_path)
    make_files(root, "english")
    monkeypatch.setattr(
        checkpoints, "snapshot_download", lambda *a, **k: pytest.fail("must not download")
    )
    assert (
        checkpoints.fetch(REPO, REVISION, "english", allow_download=False, cache_dir=tmp_path)
        == root
    )


def test_fetch_without_permission_to_download_raises(tmp_path):
    with pytest.raises(NotDownloaded, match="multilingual"):
        checkpoints.fetch(REPO, REVISION, "multilingual", allow_download=False, cache_dir=tmp_path)


def test_fetch_downloads_the_pinned_revision(tmp_path, monkeypatch):
    calls = []

    def fake_download(repo_id, *, revision, allow_patterns, cache_dir):
        calls.append((repo_id, revision, allow_patterns))
        root = checkpoints.snapshot_root(repo_id, revision, cache_dir)
        make_files(root, "multilingual")
        return str(root)

    monkeypatch.setattr(checkpoints, "snapshot_download", fake_download)
    root = checkpoints.fetch(
        REPO, REVISION, "multilingual", allow_download=True, cache_dir=tmp_path
    )
    assert calls == [(REPO, REVISION, checkpoints.patterns("multilingual"))]
    assert checkpoints.is_complete(root, "multilingual")


def test_verify_accepts_matching_weights_and_rejects_others(tmp_path):
    make_files(tmp_path, "english", b"pinned weights")
    good = checkpoints.Checkpoint(
        "english", None, hashlib.sha256(b"pinned weights").hexdigest(), len(b"pinned weights")
    )
    checkpoints.verify(tmp_path, "english", expected=good)
    with pytest.raises(ChecksumMismatch, match="bytes"):
        checkpoints.verify(
            tmp_path, "english", expected=checkpoints.Checkpoint("english", None, good.sha256, 1)
        )
    with pytest.raises(ChecksumMismatch, match="SHA-256"):
        checkpoints.verify(
            tmp_path,
            "english",
            expected=checkpoints.Checkpoint("english", None, "0" * 64, good.size),
        )
