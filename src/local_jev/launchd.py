"""Start local-jev at login with a per-user launchd agent."""

from __future__ import annotations

import os
import plistlib
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

LABEL = "io.github.andresrubio.local-jev"


def plist_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def render_plist(program: list[str], log_dir: Path, env: dict[str, str]) -> bytes:
    data: dict[str, Any] = {
        "Label": LABEL,
        "ProgramArguments": list(program),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "StandardOutPath": str(log_dir / "launchd.out.log"),
        "StandardErrorPath": str(log_dir / "launchd.err.log"),
    }
    if env:
        data["EnvironmentVariables"] = dict(env)
    return plistlib.dumps(data, sort_keys=True)


def install(
    program: list[str],
    log_dir: Path,
    env: dict[str, str],
    *,
    home: Path | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> Path:
    path = plist_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    domain = f"gui/{os.getuid()}"
    run(["launchctl", "bootout", f"{domain}/{LABEL}"], check=False, capture_output=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(render_plist(program, log_dir, env))
    path.chmod(0o600)
    run(["launchctl", "bootstrap", domain, str(path)], check=True)
    return path


def uninstall(*, home: Path | None = None, run: Callable[..., Any] = subprocess.run) -> bool:
    path = plist_path(home)
    run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"], check=False, capture_output=True)
    if path.exists():
        path.unlink()
        return True
    return False
