"""Jev-compatible REST routes: /v1/systemone, /v1/batch, /v1/status and /health."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
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
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise too_large(
            f"the body is {declared} bytes; the limit is {MAX_BODY_BYTES}",
            "use decide_batch with a path",
        )
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise too_large(
            f"the body is {len(raw)} bytes; the limit is {MAX_BODY_BYTES}",
            "use decide_batch with a path",
        )
    try:
        return json.loads(raw)
    except (ValueError, RecursionError) as exc:
        # JSONDecodeError and UnicodeDecodeError are ValueErrors. An integer of more than 4,300 digits
        # raises a plain ValueError, and deeply nested arrays raise RecursionError.
        raise invalid(f"the body is not valid JSON: {exc}", "send a JSON object") from exc
