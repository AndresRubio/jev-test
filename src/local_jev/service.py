"""DecisionService: the one place REST and MCP requests are handled."""

from __future__ import annotations

import asyncio
import logging
import os
import resource
import threading
import time
from collections.abc import Callable
from typing import Any, NoReturn

from local_jev import __version__
from local_jev.batch import (
    Item,
    aggregate_warnings,
    cleanup_results,
    load_items,
    parse_batch_params,
    public_record,
    select_matches,
    snippet,
    summarize,
    write_results,
)
from local_jev.config import Settings
from local_jev.decision_log import DecisionLog
from local_jev.engines.base import Engine, EngineResult
from local_jev.errors import BusyError, ModelUnavailable, RuntimeTimeout, ServiceError, too_large
from local_jev.guard import finalize, prepare_questions
from local_jev.runtime import Runtime
from local_jev.schema import parse_decide_request
from local_jev.security import FilePolicy

logger = logging.getLogger(__name__)
TOKEN_BUDGET_HINT = "shorten option descriptions or split into two questions"


def translate(exc: BaseException) -> ServiceError | None:
    """Map engine and runtime exceptions onto ServiceError; None for anything unexpected."""
    if isinstance(exc, ServiceError):
        return exc
    if isinstance(exc, BusyError):
        return ServiceError(529, "busy", str(exc), "retry with exponential backoff")
    if isinstance(exc, RuntimeTimeout):
        return ServiceError(
            504, "timeout", str(exc), "retry; the first call after an idle period loads the model"
        )
    if isinstance(exc, ModelUnavailable):
        return ServiceError(503, "model_unavailable", exc.message, exc.hint)
    if isinstance(exc, ValueError):
        message = str(exc)
        hint = (
            TOKEN_BUDGET_HINT
            if "token budget" in message
            else "check the questions against the Jev schema"
        )
        return ServiceError(422, "invalid_request", message, hint)
    return None


def _raise_translated(exc: BaseException) -> NoReturn:
    translated = translate(exc)
    if translated is None or translated is exc:
        raise exc
    raise translated from exc


class DecisionService:
    def __init__(
        self,
        settings: Settings,
        engine: Engine,
        *,
        runtime: Runtime | None = None,
        log: DecisionLog | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.runtime = runtime or Runtime(
            engine,
            queue_limit=settings.queue_limit,
            idle_unload_seconds=settings.idle_unload_minutes * 60,
        )
        self.log = log or DecisionLog(settings.log_dir, log_inputs=settings.log_inputs)
        self.prefetch_thread: threading.Thread | None = None
        self._files = FilePolicy(settings.batch.allowed_roots, settings.batch.max_file_bytes)
        self._clock = clock
        self._janitor: asyncio.Task[None] | None = None

    async def start(self) -> None:
        removed = cleanup_results(self.settings.results_dir)
        if removed:
            logger.info("removed %d result files older than 7 days", removed)
        self._janitor = asyncio.create_task(self._janitor_loop())
        names = self.settings.laya.prefetch
        if names:
            self.prefetch_thread = threading.Thread(
                target=self._prefetch, args=(names,), name="local-jev-prefetch", daemon=True
            )
            self.prefetch_thread.start()

    async def stop(self) -> None:
        """Cancel the janitor, stop the worker and close the log.

        Call it once no request is in flight. A request still running is answered but its decision
        is not logged, because a closed log drops records; requests still queued are cancelled.
        """
        if self._janitor is not None:
            self._janitor.cancel()
        self.runtime.shutdown()
        self.log.close()

    async def decide(self, body: Any, *, interface: str) -> dict[str, Any]:
        request = parse_decide_request(body)
        prepared = self._prepare(request.questions)
        start = self._clock()
        result = await self._run(
            lambda engine: engine.decide(request.state, prepared, request.model),
            timeout=self.settings.request_timeout_seconds,
            kind="decide",
        )
        latency_ms = (self._clock() - start) * 1000
        response = self._finalize(result, request.questions, latency_ms)
        self.log.record(
            interface=interface,
            kind="decide",
            response=response,
            questions=request.questions,
            latency_ms=latency_ms,
            state=request.state,
        )
        return response

    async def decide_batch(self, body: Any, *, interface: str) -> dict[str, Any]:
        cfg = self.settings.batch
        params = parse_batch_params(body, max_items=cfg.max_items)
        path = self._files.check(params.path) if params.path is not None else None
        try:
            items = await asyncio.to_thread(
                load_items,
                params,
                path=path,
                count_tokens=lambda text: self.engine.count_tokens(text, params.model),
                max_items=cfg.max_items,
            )
        except Exception as exc:
            _raise_translated(exc)
        evaluations = len(items) * len(params.questions)
        if evaluations > cfg.max_evaluations:
            raise too_large(
                f"{len(items)} items x {len(params.questions)} questions = {evaluations} evaluations; "
                f"the limit is {cfg.max_evaluations}",
                "ask fewer questions or split the file",
            )
        prepared = self._prepare(params.questions)
        start = self._clock()
        deadline = start + cfg.timeout_seconds
        records: list[dict[str, Any]] = []
        for offset in range(0, len(items), cfg.group_size):
            group = items[offset : offset + cfg.group_size]
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise self._deadline_error(len(records), len(items))
            try:
                results = await self._run(
                    lambda engine, group=group: [
                        engine.decide(item.state, prepared, params.model) for item in group
                    ],
                    timeout=remaining,
                    kind="batch",
                )
            except ServiceError as exc:
                # A group that outlives the deadline is told apart from other failures by its code.
                if exc.code == "timeout":
                    raise self._deadline_error(len(records), len(items)) from exc
                raise
            for item, result in zip(group, results, strict=True):
                records.append(self._record(item, self._finalize(result, params.questions, 0.0)))
        latency_ms = (self._clock() - start) * 1000
        output_path = write_results(self.settings.results_dir, records)
        matches = [
            public_record(record) for record in select_matches(records, params.filter, params.top_k)
        ]
        out = {
            "count": len(records),
            "summary": summarize(records, params.questions),
            "matches": matches,
            "output_path": str(output_path),
            "warnings": aggregate_warnings(records),
            "latency_ms": round(latency_ms, 1),
        }
        self.log.record(
            interface=interface,
            kind="batch",
            response=None,
            questions=params.questions,
            latency_ms=latency_ms,
            state=params.path or "(inline items)",
            extra={
                "items": len(records),
                "matches": len(matches),
                "output_path": str(output_path),
                "backend": self.engine.name,
            },
        )
        return out

    def status(self) -> dict[str, Any]:
        s = self.settings
        return {
            "version": __version__,
            "pid": os.getpid(),
            "backend": self.engine.name,
            "engine": self.engine.status(),
            "runtime": self.runtime.stats(),
            "memory": {
                "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)
            },
            "config": {
                "port": s.port,
                "noul_mode": s.noul_mode,
                "idle_unload_minutes": s.idle_unload_minutes,
                "log_inputs": s.log_inputs,
                "token_required": s.token is not None,
            },
        }

    def _deadline_error(self, done: int, total: int) -> ServiceError:
        return ServiceError(
            504,
            "timeout",
            f"the batch ran past {self.settings.batch.timeout_seconds:g} s after {done} of {total} items",
            "split the batch or ask fewer questions",
        )

    def _prepare(self, questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return prepare_questions(
            questions,
            noul_mode=self.settings.noul_mode,
            laya_guards=self.engine.capabilities.laya_guards,
        )

    def _finalize(
        self, result: EngineResult, questions: dict[str, dict[str, Any]], latency_ms: float
    ) -> dict[str, Any]:
        try:
            return finalize(result, questions, backend=self.engine.name, latency_ms=latency_ms)
        except ModelUnavailable as exc:
            _raise_translated(exc)

    @staticmethod
    def _record(item: Item, response: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item.id,
            "n": item.n,
            "start_line": item.start_line,
            "end_line": item.end_line,
            "snippet": snippet(item.state),
            "checkpoint": response["local"]["checkpoint"],
            "answers": response["answers"],
        }

    async def _run(self, job: Callable[[Engine], Any], *, timeout: float, kind: str) -> Any:
        try:
            return await self.runtime.run(job, timeout=timeout, kind=kind)
        except Exception as exc:
            _raise_translated(exc)

    def _prefetch(self, names: tuple[str, ...]) -> None:
        try:
            self.engine.prefetch(names)
        except Exception:
            logger.exception(
                "prefetching %s failed; run `local-jev pull` to retry", ", ".join(names)
            )

    async def _janitor_loop(self) -> None:
        seconds = self.runtime.idle_unload_seconds
        interval = min(30.0, max(1.0, seconds / 2)) if seconds > 0 else 30.0
        while True:
            await asyncio.sleep(interval)
            self.runtime.check_idle()
