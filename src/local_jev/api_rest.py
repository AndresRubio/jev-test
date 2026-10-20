"""Jev-compatible REST routes: /v1/systemone, /v1/batch, /v1/status and /health."""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from local_jev.errors import ServiceError, invalid, too_large

MAX_BODY_BYTES = 5 * 1024 * 1024


class _AsciiJSONResponse(JSONResponse):
    """A JSONResponse that writes every non-ASCII character as an escape.

    Starlette writes the characters themselves and then encodes them as UTF-8, which raises for a
    lone surrogate. An error message repeats the caller's path and a batch snippet repeats an inline
    item, so a caller who sends the JSON escape for U+D800 gets that text back in the response.
    """

    def render(self, content: Any) -> bytes:
        text = json.dumps(content, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        return text.encode("ascii")


def build_rest_routes(service: Any) -> list[Route]:
    async def health(request: Request) -> Response:
        return _AsciiJSONResponse({"ok": True})

    async def systemone(request: Request) -> Response:
        return await _guarded(request, lambda body: service.decide(body, interface="rest"))

    async def batch(request: Request) -> Response:
        return await _guarded(request, lambda body: service.decide_batch(body, interface="rest"))

    async def status(request: Request) -> Response:
        return _AsciiJSONResponse(service.status())

    return [
        Route("/health", health, methods=["GET"]),
        Route("/v1/systemone", systemone, methods=["POST"]),
        Route("/v1/batch", batch, methods=["POST"]),
        Route("/v1/status", status, methods=["GET"]),
    ]


async def _guarded(
    request: Request, handler: Callable[[Any], Awaitable[dict[str, Any]]]
) -> Response:
    try:
        body = await _read_json(request)
        return _AsciiJSONResponse(await handler(body))
    except ServiceError as exc:
        return _AsciiJSONResponse(exc.to_dict(), status_code=exc.status)


async def _read_json(request: Request) -> Any:
    body = await _read_body(request)
    try:
        return json.loads(body, parse_constant=_refuse_constant, parse_float=_finite_float)
    except (ValueError, RecursionError) as exc:
        # JSONDecodeError and UnicodeDecodeError are ValueErrors. An integer of more than 4,300
        # digits raises a plain ValueError, and deeply nested arrays raise RecursionError.
        # The hooks above raise ValueError for NaN, the infinities and numbers that overflow.
        raise invalid(f"the body is not valid JSON: {exc}", "send a JSON object") from exc


async def _read_body(request: Request) -> bytearray:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise too_large(
            f"the body is {declared} bytes; the limit is {MAX_BODY_BYTES}",
            "use decide_batch with a path",
        )
    # A body sent in chunks has no length to check first, so it is counted as it arrives. It is
    # refused as soon as it passes the limit, without reading the rest.
    body = bytearray()
    async with aclosing(request.stream()) as chunks:
        async for chunk in chunks:
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                raise too_large(
                    f"the body is over {MAX_BODY_BYTES} bytes; the limit is {MAX_BODY_BYTES}",
                    "use decide_batch with a path",
                )
            body.extend(chunk)
    return body


def _refuse_constant(name: str) -> Any:
    """NaN and the infinities are not JSON. json.loads reads them, but no response can hold them."""
    raise ValueError(f"{name} is not allowed")


def _finite_float(text: str) -> float:
    """float(text), refusing a literal such as 1e400 that overflows to an infinity."""
    number = float(text)
    if not math.isfinite(number):
        shown = text if len(text) <= 32 else f"{text[:29]}..."
        raise ValueError(f"the number {shown} is out of range")
    return number
