"""Start a throwaway local-jev server for the evaluation and benchmark scripts."""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import httpx2


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def temporary_server(**env_overrides: str) -> Iterator[tuple[str, subprocess.Popen[bytes]]]:
    port = free_port()
    workdir = Path(tempfile.mkdtemp(prefix="local-jev-"))
    env = {
        **os.environ,
        "LOCAL_JEV_CONFIG": str(workdir / "none.toml"),
        "LOCAL_JEV_PORT": str(port),
        "LOCAL_JEV_NOUL_MODE": "native",
        "LOCAL_JEV_LOG_DIR": str(workdir / "logs"),
        "LOCAL_JEV_RESULTS_DIR": str(workdir / "results"),
        **env_overrides,
    }
    env.pop("LOCAL_JEV_TOKEN", None)
    stderr = (workdir / "stderr.log").open("wb")
    process = subprocess.Popen(
        [sys.executable, "-m", "local_jev", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=stderr,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                if httpx2.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx2.HTTPError:
                pass
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"the server did not start; see {workdir / 'stderr.log'}")
            time.sleep(0.2)
        yield base, process
    finally:
        process.terminate()
        process.wait(timeout=30)
        stderr.close()


def wait_until_ready(
    base: str, names: tuple[str, ...] = ("english", "multilingual"), timeout: float = 900
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = httpx2.get(f"{base}/v1/status", timeout=5).json()
        if set(names) <= set(status["engine"].get("ready", [])):
            return status
        time.sleep(1)
    raise RuntimeError(f"{names} were not ready within {timeout:.0f} s; run `local-jev pull` first")
