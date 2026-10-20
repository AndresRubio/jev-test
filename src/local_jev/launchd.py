"""Start local-jev at login with a per-user launchd agent."""

from __future__ import annotations

import os
import plistlib
import subprocess
import tempfile
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
    """Write the plist, boot out any running agent, then bootstrap the new one.

    Nothing running is touched until the plist is safely on disk, so a failed write leaves the
    agent that is running now, and the plist it was started from, as they were. A failed bootstrap
    raises CalledProcessError after the old agent was stopped and the new plist was written.
    """
    path = plist_path(home)
    data = render_plist(program, log_dir, env)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    _write_private(path, data)
    domain = f"gui/{os.getuid()}"
    run(["launchctl", "bootout", f"{domain}/{LABEL}"], check=False, capture_output=True)
    run(["launchctl", "bootstrap", domain, str(path)], check=True)
    return path


def _write_private(path: Path, data: bytes) -> None:
    """Put `data` at `path` in one step, readable only by this user.

    The plist holds the token. It is written to a temporary file in the same folder that is private
    from the moment it exists, and then renamed over the target, so no reader sees a half-written
    plist and a failure leaves the old one in place.
    """
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            os.fchmod(fh.fileno(), 0o600)
            fh.write(data)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def uninstall(*, home: Path | None = None, run: Callable[..., Any] = subprocess.run) -> bool:
    path = plist_path(home)
    run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"], check=False, capture_output=True)
    if path.exists():
        path.unlink()
        return True
    return False
