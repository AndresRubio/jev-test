"""Loopback-only access, an optional bearer token, and the file policy for decide_batch."""

from __future__ import annotations

import hmac
import json
import re
from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Any

from local_jev.errors import invalid, too_large

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

LOOPBACK_ORIGIN = re.compile(
    r"http://(?:127\.0\.0\.1|localhost|\[::1\])(?::[0-9]{1,5})?", re.ASCII | re.IGNORECASE
)


def fits_in_a_header(value: str) -> bool:
    """Whether the HTTP stack can carry the value, a token or an API key, in a header.

    The stack quotes a header value it refuses (trailing whitespace, control characters) in its
    error text, and non-ASCII text raises a bare UnicodeEncodeError. A secret that fails this
    check is never sent, so it cannot reach an exception message or a log.
    """
    return value.isascii() and value.isprintable() and value == value.strip()


def loopback_hosts(port: int) -> frozenset[str]:
    return frozenset({f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"})


def is_loopback_origin(origin: str) -> bool:
    """True only for http://127.0.0.1, http://localhost or http://[::1], with or without a port."""
    return LOOPBACK_ORIGIN.fullmatch(origin) is not None


class SecurityMiddleware:
    """Answer only requests addressed to this machine's loopback, not sent from web pages."""

    def __init__(self, app: ASGIApp, *, port: int, token: str | None = None) -> None:
        self.app = app
        self._port = port
        self._hosts = loopback_hosts(port)
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = _header_values(scope)
        hosts = headers.get("host", [])
        if len(hosts) != 1 or hosts[0].lower() not in self._hosts:
            await _reject(
                send,
                421,
                "misdirected",
                "this server only answers requests addressed to its loopback address",
                f"use http://127.0.0.1:{self._port}",
            )
            return
        if not all(is_loopback_origin(origin) for origin in headers.get("origin", [])):
            await _reject(
                send,
                403,
                "forbidden_origin",
                "requests from web pages are not allowed",
                "call local-jev from a local program, not a browser",
            )
            return
        if self._token and scope.get("path") != "/health":
            # Joined, so a request that carries two Authorization headers never matches.
            supplied = ", ".join(headers.get("authorization", [])).encode()
            # surrogatepass: a token holding non-UTF-8 bytes fails closed instead of raising.
            expected = f"Bearer {self._token}".encode("utf-8", "surrogatepass")
            if not hmac.compare_digest(supplied, expected):
                await _reject(
                    send,
                    401,
                    "unauthorized",
                    "missing or wrong bearer token",
                    "send Authorization: Bearer $LOCAL_JEV_TOKEN",
                )
                return
        await self.app(scope, receive, send)


def _header_values(scope: Scope) -> dict[str, list[str]]:
    """Every value of every header, so one Host or Origin cannot hide behind another."""
    values: dict[str, list[str]] = {}
    for key, value in scope.get("headers", []):
        values.setdefault(key.decode("latin-1").lower(), []).append(value.decode("latin-1"))
    return values


async def _reject(send: Send, status: int, code: str, message: str, hint: str) -> None:
    body = json.dumps({"error": {"code": code, "message": message, "hint": hint}}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class FilePolicy:
    """Which local files decide_batch may read."""

    def __init__(self, allowed_roots: tuple[Path, ...], max_bytes: int) -> None:
        self._roots = tuple(Path(root).expanduser().resolve() for root in allowed_roots)
        self._max_bytes = max_bytes

    def check(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            raise invalid(
                f"'path' must be absolute: {path!r}",
                "pass the full path, e.g. /Users/you/data/items.jsonl",
            )
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise invalid(f"no such file: {path}", "check the path") from exc
        if not any(resolved.is_relative_to(root) for root in self._roots):
            allowed = ", ".join(str(root) for root in self._roots)
            raise invalid(
                f"{path} is outside the allowed folders",
                f"allowed: {allowed}; change batch.allowed_roots in the config",
            )
        if not resolved.is_file():
            raise invalid(
                f"{path} is not a regular file", "point at a file, not a folder or a device"
            )
        size = resolved.stat().st_size
        if size > self._max_bytes:
            raise too_large(
                f"{path} is {size} bytes; the limit is {self._max_bytes}", "split the file"
            )
        return resolved
