"""Assemble the Starlette app: security middleware, REST routes, the MCP endpoint and the lifespan."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Mount

from local_jev.api_mcp import build_mcp_server
from local_jev.api_rest import MAX_BODY_BYTES, build_rest_routes
from local_jev.config import Settings
from local_jev.engines.base import Engine
from local_jev.security import SecurityMiddleware, loopback_hosts
from local_jev.service import DecisionService

LOOPBACK_ORIGINS = ("http://127.0.0.1", "http://localhost", "http://[::1]")


def mcp_transport_security(port: int) -> TransportSecuritySettings:
    """The SDK's own Host and Origin check, set to the loopback allowlist of SecurityMiddleware.

    For host="127.0.0.1" the SDK would accept any port in a Host and refuse an Origin without a
    port, such as http://localhost. This list takes the three loopback hosts on the configured port
    and the three loopback origins with or without a port, as the middleware does.

    The SDK cannot match the middleware exactly. It compares case-sensitively, so a Host or Origin in
    capitals passes the middleware and fails here. It also reads "<origin>:*" as the origin, a colon
    and any text, so a malformed Origin such as http://localhost:abc passes here. The middleware
    runs first on every request and refuses it.
    """
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=sorted(loopback_hosts(port)),
        allowed_origins=[*LOOPBACK_ORIGINS, *(f"{origin}:*" for origin in LOOPBACK_ORIGINS)],
    )


def build_engine(settings: Settings) -> Engine:
    if settings.backend == "remote":
        from local_jev.engines.remote import RemoteJevEngine

        variable = settings.remote.api_key_env
        api_key = os.environ.get(variable, "").strip() if variable else ""
        return RemoteJevEngine(
            settings.remote.url, api_key=api_key or None, timeout=settings.request_timeout_seconds
        )
    from local_jev.engines.laya_mlx import LayaMLXEngine

    return LayaMLXEngine(settings.laya)


def create_app(settings: Settings, *, service: DecisionService | None = None) -> Starlette:
    service = service or DecisionService(settings, build_engine(settings))
    mcp = build_mcp_server(service)
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        host="127.0.0.1",
        max_request_body_size=MAX_BODY_BYTES,
        transport_security=mcp_transport_security(settings.port),
    )

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        await service.start()
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            await service.stop()

    return Starlette(
        routes=[*build_rest_routes(service), Mount("/", app=mcp_app)],
        middleware=[Middleware(SecurityMiddleware, port=settings.port, token=settings.token)],
        lifespan=lifespan,
    )
