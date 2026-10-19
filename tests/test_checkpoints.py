import hashlib
import os
import re
import shutil
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


def move_weights_behind_links(root: Path, name: str, *, shared_store: bool) -> Path:
    """Lay the weights out like huggingface_hub and return the file that holds the bytes.

    The snapshot has a relative symlink to blobs/<etag>. With the cache-wide shared blob store of
    huggingface_hub 1.33.0, blobs/<etag> is itself a symlink to <cache>/blobs/<xx>/<hash>.
    """
    link = checkpoints.checkpoint_dir(root, name) / "model.safetensors"
    blob = root.parents[1] / "blobs" / "etag"
    holder = root.parents[2] / "blobs" / "xx" / "hash" if shared_store else blob
    blob.parent.mkdir(parents=True, exist_ok=True)
    holder.parent.mkdir(parents=True, exist_ok=True)
    link.rename(holder)
    if shared_store:
        blob.symlink_to(os.path.relpath(holder, blob.parent))
    link.symlink_to(os.path.relpath(blob, link.parent))
    return holder


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


@pytest.mark.parametrize("shared_store", [False, True], ids=["blob", "shared-store"])
def test_the_checksum_hint_names_the_file_that_holds_the_bytes(tmp_path, shared_store):
    root = checkpoints.snapshot_root(REPO, REVISION, tmp_path)
    make_files(root, "english", b"corrupt weights")
    holder = move_weights_behind_links(root, "english", shared_store=shared_store)
    size = len(b"corrupt weights")
    for expected in (
        checkpoints.Checkpoint("english", None, "0" * 64, size),
        checkpoints.Checkpoint("english", None, "0" * 64, size + 1),
    ):
        with pytest.raises(ChecksumMismatch) as info:
            checkpoints.verify(root, "english", expected=expected)
        assert info.value.hint == f"delete {holder.resolve()} and run `local-jev pull`"


@pytest.mark.parametrize("shared_store", [False, True], ids=["blob", "shared-store"])
def test_following_the_checksum_hint_lets_fetch_download_the_weights_again(
    tmp_path, monkeypatch, shared_store
):
    root = checkpoints.snapshot_root(REPO, REVISION, tmp_path)
    make_files(root, "english", b"corrupt weights")
    holder = move_weights_behind_links(root, "english", shared_store=shared_store)
    corrupt = checkpoints.Checkpoint("english", None, "0" * 64, len(b"corrupt weights"))
    with pytest.raises(ChecksumMismatch) as info:
        checkpoints.verify(root, "english", expected=corrupt)

    match = re.fullmatch(r"delete (.+) and run `local-jev pull`", info.value.hint)
    assert match, info.value.hint
    named = Path(match.group(1))
    if named.is_dir():
        shutil.rmtree(named)
    else:
        named.unlink()

    assert not holder.exists(), (
        "the corrupt bytes survive, so huggingface_hub would link them again"
    )
    assert not checkpoints.is_complete(root, "english")

    calls = []

    def fake_download(repo_id, *, revision, allow_patterns, cache_dir):
        calls.append(allow_patterns)
        holder.write_bytes(b"good weights")
        return str(root)

    monkeypatch.setattr(checkpoints, "snapshot_download", fake_download)
    assert (
        checkpoints.fetch(REPO, REVISION, "english", allow_download=True, cache_dir=tmp_path)
        == root
    )
    assert calls == [checkpoints.patterns("english")]
    good = checkpoints.Checkpoint(
        "english", None, hashlib.sha256(b"good weights").hexdigest(), len(b"good weights")
    )
    checkpoints.verify(root, "english", expected=good)
