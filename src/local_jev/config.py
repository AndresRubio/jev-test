"""Settings: defaults, then ~/.config/local-jev/config.toml, then LOCAL_JEV_* variables."""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx2

from local_jev.security import fits_in_a_header

DEFAULT_CONFIG_PATH = Path("~/.config/local-jev/config.toml")
LAYA_REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"
CHECKPOINT_NAMES = ("english", "multilingual", "typed-decisions")
CONFIG_ENV = "LOCAL_JEV_CONFIG"
REMOTE_URL_ENV = "LOCAL_JEV_REMOTE_URL"
REMOTE_URL_EXAMPLE = "http://127.0.0.1:8009/v1/systemone"


class ConfigError(ValueError):
    """The configuration file or the environment holds an invalid value."""


@dataclass(frozen=True)
class LayaSettings:
    repo: str = "convaiinnovations/laya"
    revision: str = LAYA_REVISION
    dtype: str = "float16"
    prefetch: tuple[str, ...] = ("english", "multilingual")


@dataclass(frozen=True)
class BatchSettings:
    allowed_roots: tuple[Path, ...] = (Path.home(),)
    max_items: int = 5000
    max_file_bytes: int = 50 * 1024 * 1024
    max_evaluations: int = 20_000
    group_size: int = 32
    timeout_seconds: float = 300.0


@dataclass(frozen=True)
class RemoteSettings:
    url: str = ""
    api_key_env: str = ""


@dataclass(frozen=True)
class Settings:
    port: int = 7711
    backend: str = "laya-mlx"
    idle_unload_minutes: float = 15.0
    noul_mode: str = "ab_labels"
    log_inputs: bool = False
    token: str | None = field(default=None, repr=False)
    request_timeout_seconds: float = 30.0
    queue_limit: int = 64
    log_dir: Path = Path.home() / "Library" / "Logs" / "local-jev"
    results_dir: Path = Path.home() / "Library" / "Caches" / "local-jev" / "results"
    laya: LayaSettings = field(default_factory=LayaSettings)
    batch: BatchSettings = field(default_factory=BatchSettings)
    remote: RemoteSettings = field(default_factory=RemoteSettings)


_TOP: dict[str, Any] = {
    "port": int,
    "backend": str,
    "idle_unload_minutes": float,
    "noul_mode": str,
    "log_inputs": bool,
    "request_timeout_seconds": float,
    "queue_limit": int,
    "log_dir": Path,
    "results_dir": Path,
}
_SECTIONS: dict[str, tuple[type, dict[str, Any]]] = {
    "laya": (LayaSettings, {"repo": str, "revision": str, "dtype": str, "prefetch": "names"}),
    "batch": (
        BatchSettings,
        {
            "allowed_roots": "paths",
            "max_items": int,
            "max_file_bytes": int,
            "max_evaluations": int,
            "group_size": int,
            "timeout_seconds": float,
        },
    ),
    "remote": (RemoteSettings, {"url": str, "api_key_env": str}),
}
_ENV: dict[str, tuple[str, Any]] = {
    "LOCAL_JEV_PORT": ("port", int),
    "LOCAL_JEV_BACKEND": ("backend", str),
    "LOCAL_JEV_IDLE_UNLOAD_MINUTES": ("idle_unload_minutes", float),
    "LOCAL_JEV_NOUL_MODE": ("noul_mode", str),
    "LOCAL_JEV_LOG_INPUTS": ("log_inputs", bool),
    "LOCAL_JEV_TOKEN": ("token", str),
    "LOCAL_JEV_LOG_DIR": ("log_dir", Path),
    "LOCAL_JEV_RESULTS_DIR": ("results_dir", Path),
}
# Every environment variable load_settings reads, and the ones among them that hold a path.
ENV_VARS = (CONFIG_ENV, *_ENV, REMOTE_URL_ENV)
PATH_ENV_VARS = (CONFIG_ENV, *(var for var, (_, kind) in _ENV.items() if kind is Path))


def load_settings(path: Path | None = None, env: Mapping[str, str] | None = None) -> Settings:
    """Defaults, overlaid by the TOML file when it exists, overlaid by LOCAL_JEV_* variables."""
    env = os.environ if env is None else env
    if path is None:
        path = Path(env.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH)
    path = _expanded(str(path), CONFIG_ENV)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    settings = _apply_env(_from_mapping(data, path), env)
    _validate(settings)
    return settings


def _from_mapping(data: dict[str, Any], source: Path) -> Settings:
    top: dict[str, Any] = {}
    sections: dict[str, Any] = {}
    for key, value in data.items():
        if key in _SECTIONS:
            cls, types = _SECTIONS[key]
            if not isinstance(value, dict):
                raise ConfigError(f"{source}: [{key}] must be a table")
            kwargs = {}
            for sub, sub_value in value.items():
                if sub not in types:
                    raise ConfigError(f"{source}: unknown setting {key}.{sub}")
                kwargs[sub] = _from_file(types[sub], sub_value, f"{key}.{sub}")
            sections[key] = cls(**kwargs)
        elif key in _TOP:
            top[key] = _from_file(_TOP[key], value, key)
        else:
            raise ConfigError(f"{source}: unknown setting {key!r}")
    return Settings(**top, **sections)


def _from_file(kind: Any, value: Any, where: str) -> Any:
    """A value from the config file, whose paths must be absolute.

    The login agent starts in /, so a relative path would mean something else there than in a
    terminal: "." as an allowed root would open the whole disk to decide_batch.
    """
    coerced = _coerce(kind, value, where)
    paths = coerced if kind == "paths" else (coerced,) if kind is Path else ()
    if not all(path.is_absolute() for path in paths):
        raise ConfigError(f"{where} must be an absolute path (a leading ~ is allowed)")
    return coerced


def _apply_env(settings: Settings, env: Mapping[str, str]) -> Settings:
    changes = {
        name: _from_env(kind, env[var], var) for var, (name, kind) in _ENV.items() if env.get(var)
    }
    settings = replace(settings, **changes)
    if env.get(REMOTE_URL_ENV):
        settings = replace(settings, remote=replace(settings.remote, url=env[REMOTE_URL_ENV]))
    return settings


def _from_env(kind: Any, value: str, where: str) -> Any:
    """A value from the environment, where a relative path means one under the current folder."""
    coerced = _coerce(kind, value, where)
    return coerced.resolve() if kind is Path and not coerced.is_absolute() else coerced


def _coerce(kind: Any, value: Any, where: str) -> Any:
    if kind is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("1", "true", "yes", "on"):
            return True
        if isinstance(value, str) and value.lower() in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"{where} must be true or false")
    if kind in (int, float):
        if isinstance(value, bool):
            raise ConfigError(f"{where} must be a number")
        try:
            return kind(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{where} must be a number") from exc
    if kind is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string")
        return value
    if kind is Path:
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{where} must be a path")
        return _expanded(value, where)
    if kind in ("names", "paths"):
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            raise ConfigError(f"{where} must be a list of strings")
        return tuple(_expanded(v, where) for v in value) if kind == "paths" else tuple(value)
    raise AssertionError(f"unhandled setting type for {where}")


def _expanded(value: str, where: str) -> Path:
    """The path with a leading ~ or ~user replaced by that home folder.

    Path.expanduser raises RuntimeError for a user the Mac does not have.
    """
    try:
        return Path(value).expanduser()
    except RuntimeError as exc:
        user = value.split("/", 1)[0]
        raise ConfigError(f"{where} starts with {user}, a user this Mac does not have") from exc


def _validate(s: Settings) -> None:
    if not 1024 <= s.port <= 65535:
        raise ConfigError("port must be between 1024 and 65535")
    if s.backend not in ("laya-mlx", "remote"):
        raise ConfigError("backend must be 'laya-mlx' or 'remote'")
    if s.noul_mode not in ("native", "ab_labels"):
        raise ConfigError("noul_mode must be 'native' or 'ab_labels'")
    if s.idle_unload_minutes < 0:
        raise ConfigError("idle_unload_minutes must be 0 (never unload) or more")
    if s.request_timeout_seconds <= 0 or s.batch.timeout_seconds <= 0:
        raise ConfigError("timeouts must be positive")
    if s.queue_limit < 1 or s.batch.group_size < 1 or s.batch.max_items < 1:
        raise ConfigError("queue_limit, batch.group_size and batch.max_items must be at least 1")
    if s.laya.dtype not in ("float16", "float32"):
        raise ConfigError("laya.dtype must be 'float16' or 'float32'")
    if not re.fullmatch(r"[0-9a-f]{40}", s.laya.revision):
        raise ConfigError("laya.revision must be a full 40-character commit hash")
    unknown = sorted(set(s.laya.prefetch) - set(CHECKPOINT_NAMES))
    if unknown:
        raise ConfigError(f"laya.prefetch has unknown checkpoints: {unknown}")
    if s.backend == "remote" and not s.remote.url:
        raise ConfigError("backend 'remote' needs remote.url (or LOCAL_JEV_REMOTE_URL)")
    if s.remote.url:
        _check_remote_url(s.remote.url)
    if s.token is not None and not fits_in_a_header(s.token):
        # Every request would get a 401. The message never repeats the token.
        raise ConfigError(
            "LOCAL_JEV_TOKEN must be printable ASCII with no whitespace at either end"
        )


def _check_remote_url(url: str) -> None:
    """Refuse a URL the remote engine cannot post to, or would send a user name and password to.

    httpx2 raises InvalidURL on every call for a malformed URL, and sends a URL's userinfo as
    Basic auth. The messages never repeat the URL, because it may hold a password.
    """
    where = f"remote.url (or {REMOTE_URL_ENV})"
    try:
        parsed: httpx2.URL | None = httpx2.URL(url)
    except httpx2.InvalidURL:
        parsed = None  # raised below, outside this block, so the error does not chain the URL
    if parsed is None or parsed.scheme not in ("http", "https") or not parsed.host:
        raise ConfigError(
            f"{where} must be an http or https URL with a host, such as {REMOTE_URL_EXAMPLE}"
        )
    if parsed.userinfo:
        raise ConfigError(
            f"{where} must not hold a user name or password; "
            "put the API key in the variable that remote.api_key_env names"
        )
