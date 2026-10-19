"""Pinned Laya checkpoints: where they live in the Hugging Face cache, download and verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import constants, snapshot_download

from local_jev.errors import ChecksumMismatch, NotDownloaded

FILES = (
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/*",
    "mlx_config.json",
)
REQUIRED = (
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
)
PULL_HINT = "run `local-jev pull` (add --all for typed-decisions)"


@dataclass(frozen=True)
class Checkpoint:
    name: str
    subfolder: str | None
    sha256: str
    size: int


CHECKPOINTS = {
    "english": Checkpoint(
        "english",
        None,
        "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
        842_609_210,
    ),
    "multilingual": Checkpoint(
        "multilingual",
        "multilingual",
        "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
        643_835_514,
    ),
    "typed-decisions": Checkpoint(
        "typed-decisions",
        "typed-decisions",
        "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e",
        842_609_220,
    ),
}


def patterns(name: str) -> list[str]:
    subfolder = CHECKPOINTS[name].subfolder
    prefix = f"{subfolder}/" if subfolder else ""
    return [prefix + item for item in FILES]


def snapshot_root(repo: str, revision: str, cache_dir: Path | None = None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else Path(constants.HF_HUB_CACHE)
    return base / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


def checkpoint_dir(root: Path, name: str) -> Path:
    subfolder = CHECKPOINTS[name].subfolder
    return root / subfolder if subfolder else root


def is_complete(root: Path, name: str) -> bool:
    folder = checkpoint_dir(root, name)
    return all((folder / item).is_file() for item in REQUIRED)


def fetch(
    repo: str, revision: str, name: str, *, allow_download: bool, cache_dir: Path | None = None
) -> Path:
    """Return the snapshot root holding `name`, downloading it first when allowed."""
    root = snapshot_root(repo, revision, cache_dir)
    if is_complete(root, name):
        return root
    if not allow_download:
        raise NotDownloaded(f"the {name} checkpoint is not downloaded", PULL_HINT)
    downloaded = Path(
        snapshot_download(
            repo, revision=revision, allow_patterns=patterns(name), cache_dir=cache_dir
        )
    )
    if not is_complete(downloaded, name):
        raise NotDownloaded(f"the download of the {name} checkpoint is incomplete", PULL_HINT)
    return downloaded


def verify(root: Path, name: str, *, expected: Checkpoint | None = None) -> None:
    """Check the weight file against its pinned size and SHA-256."""
    expected = expected or CHECKPOINTS[name]
    path = checkpoint_dir(root, name) / "model.safetensors"
    # In the Hugging Face cache this path is a symlink and the bytes sit behind it. huggingface_hub
    # relinks a deleted symlink to a blob that is still there without downloading, so the hint
    # names the file the link resolves to. Deleting that leaves a dangling link, and pull downloads.
    hint = f"delete {path.resolve()} and run `local-jev pull`"
    size = path.stat().st_size
    if size != expected.size:
        raise ChecksumMismatch(f"{path} is {size} bytes, expected {expected.size}", hint)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected.sha256:
        raise ChecksumMismatch(f"{path} does not match its pinned SHA-256", hint)
