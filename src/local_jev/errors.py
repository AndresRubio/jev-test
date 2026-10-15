"""Errors shared by the service, the engines and the runtime."""

from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    """An error the REST and MCP front ends report to the caller."""

    def __init__(self, status: int, code: str, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "hint": self.hint}}

    def for_model(self) -> str:
        text = f"{self.code}: {self.message}"
        return f"{text} Hint: {self.hint}" if self.hint else text


def invalid(message: str, hint: str = "") -> ServiceError:
    return ServiceError(422, "invalid_request", message, hint)


def too_large(message: str, hint: str = "") -> ServiceError:
    return ServiceError(413, "too_large", message, hint)


class ModelUnavailable(Exception):
    """The engine cannot answer right now: weights missing, backend down or bad outputs."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class NotDownloaded(ModelUnavailable):
    """A checkpoint is not in the local Hugging Face cache."""


class ChecksumMismatch(ModelUnavailable):
    """A weight file does not match its pinned size or SHA-256."""


class BusyError(Exception):
    """The inference queue is full."""


class RuntimeTimeout(Exception):
    """An inference job did not finish within its timeout."""
