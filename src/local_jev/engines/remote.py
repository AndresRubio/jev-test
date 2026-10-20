"""Forward decisions to any Jev-compatible /v1/systemone endpoint: Kev, upstream laya-serve or hosted Jev."""

from __future__ import annotations

import math
from typing import Any

import httpx2

from local_jev.engines.base import EngineCapabilities, EngineResult
from local_jev.errors import ModelUnavailable
from local_jev.security import fits_in_a_header

ENDPOINT_HINT = "is remote.url a Jev-compatible /v1/systemone endpoint?"


class RemoteJevEngine:
    name = "remote"
    capabilities = EngineCapabilities(laya_guards=False)

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._client = httpx2.Client(timeout=timeout, transport=transport)

    def decide(self, state: Any, questions: dict[str, dict[str, Any]], model: str) -> EngineResult:
        payload = {
            "model": "jev-latest" if model == "auto" else model,
            "state": state,
            "questions": questions,
        }
        if self._api_key and not fits_in_a_header(self._api_key):
            raise ModelUnavailable(
                "the API key for the remote backend cannot be sent as an HTTP header",
                "check the key for stray whitespace or non-ASCII characters (remote.api_key_env)",
            )
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            response = self._client.post(self._url, json=payload, headers=headers)
        except httpx2.HTTPError as exc:
            raise ModelUnavailable(
                f"the remote backend is unreachable: {exc}",
                "check remote.url and that the server is running",
            ) from exc
        if response.status_code == 422:
            raise ValueError(f"the remote backend rejected the request: {response.text[:500]}")
        if response.status_code in (401, 403):
            raise ModelUnavailable(
                "the remote backend refused the API key", "check remote.api_key_env"
            )
        if response.status_code >= 400:
            raise ModelUnavailable(
                f"the remote backend returned HTTP {response.status_code}", "retry later"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ModelUnavailable(
                "the remote backend returned an unexpected payload", ENDPOINT_HINT
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise ModelUnavailable(
                "the remote backend returned an unexpected payload", ENDPOINT_HINT
            )
        usage = data.get("usage")
        if usage is not None and not isinstance(usage, dict):
            raise ModelUnavailable(
                "the remote backend returned an unexpected payload", ENDPOINT_HINT
            )
        return EngineResult(
            answers=data["answers"], usage=usage or {}, model=str(data.get("model", "remote"))
        )

    def count_tokens(self, text: str, model: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    def prefetch(self, names: tuple[str, ...]) -> None:
        return None

    def is_loaded(self) -> bool:
        return False

    def unload(self) -> None:
        return None

    def status(self) -> dict[str, Any]:
        return {"backend": self.name, "url": self._url, "api_key": bool(self._api_key)}
