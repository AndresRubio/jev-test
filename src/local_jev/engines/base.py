"""The interface every backend implements.

decide(), count_tokens(), prefetch() and unload() may block. decide() and unload() always run
on the single inference worker thread. status() and is_loaded() must be cheap and thread-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class EngineCapabilities:
    laya_guards: bool  # understands Laya's noul labels and reports truncation in usage


@dataclass
class EngineResult:
    answers: dict[str, dict[str, Any]]
    usage: dict[str, Any]
    model: str
    checkpoint: str | None = None
    routing_reason: str | None = None
    shortlist: dict[str, dict[str, Any]] = field(default_factory=dict)


class Engine(Protocol):
    name: str
    capabilities: EngineCapabilities

    def decide(
        self, state: Any, questions: dict[str, dict[str, Any]], model: str
    ) -> EngineResult: ...

    def count_tokens(self, text: str, model: str) -> int: ...

    def prefetch(self, names: tuple[str, ...]) -> None: ...

    def is_loaded(self) -> bool: ...

    def unload(self) -> None: ...

    def status(self) -> dict[str, Any]: ...
