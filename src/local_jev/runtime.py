"""One inference worker thread: jobs never overlap, the queue is bounded, idle models unload."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

from local_jev.engines.base import Engine
from local_jev.errors import BusyError, RuntimeTimeout

logger = logging.getLogger(__name__)
T = TypeVar("T")


class Runtime:
    def __init__(
        self,
        engine: Engine,
        *,
        queue_limit: int = 64,
        idle_unload_seconds: float = 900.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.idle_unload_seconds = idle_unload_seconds
        self._engine = engine
        self._queue_limit = queue_limit
        self._clock = clock
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-jev-inference")
        self._lock = threading.Lock()
        self._pending = 0
        self._last_used = clock()
        self._unload_scheduled = False
        self._latencies: dict[str, deque[float]] = {}

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    async def run(self, job: Callable[[Engine], T], *, timeout: float, kind: str = "decide") -> T:
        """Run job(engine) on the worker thread; never more than one job at a time."""
        with self._lock:
            if self._pending >= self._queue_limit:
                raise BusyError(f"{self._pending} requests are already queued")
            self._pending += 1
        try:
            future = self._executor.submit(self._execute, job, kind)
        except BaseException:
            with self._lock:
                self._pending -= 1  # nothing was queued, e.g. after shutdown()
            raise
        future.add_done_callback(self._release)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout)
        except TimeoutError as exc:
            future.add_done_callback(partial(_log_late_failure, kind))
            raise RuntimeTimeout(f"no answer within {timeout:g} s") from exc
        except asyncio.CancelledError:
            future.add_done_callback(partial(_log_late_failure, kind))
            raise

    def check_idle(self) -> bool:
        """Schedule an unload when nothing ran for idle_unload_seconds. Returns True if scheduled."""
        if self.idle_unload_seconds <= 0:
            return False
        with self._lock:
            idle = self._clock() - self._last_used
            if self._pending or self._unload_scheduled or idle < self.idle_unload_seconds:
                return False
            self._unload_scheduled = True
        if not self._engine.is_loaded():
            with self._lock:
                self._unload_scheduled = False
            return False
        self._executor.submit(self._unload)
        return True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            pending = self._pending
            idle = self._clock() - self._last_used
            samples = list(self._latencies.get("decide", ()))
        return {
            "queue_depth": pending,
            "idle_seconds": round(idle, 1),
            "idle_unload_seconds": self.idle_unload_seconds,
            "decide_latency_ms": _percentiles(samples),
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _execute(self, job: Callable[[Engine], T], kind: str) -> T:
        start = self._clock()
        try:
            return job(self._engine)
        finally:
            elapsed_ms = (self._clock() - start) * 1000
            with self._lock:
                self._latencies.setdefault(kind, deque(maxlen=1000)).append(elapsed_ms)
                self._last_used = self._clock()

    def _release(self, _future: Future[Any]) -> None:
        with self._lock:
            self._pending -= 1
            self._last_used = self._clock()

    def _unload(self) -> None:
        try:
            self._engine.unload()
        except Exception:
            logger.exception("unloading the idle model failed")
        finally:
            with self._lock:
                self._unload_scheduled = False


def _log_late_failure(kind: str, future: Future[Any]) -> None:
    """Log the error of a job whose caller got a timeout or a cancellation instead of it."""
    if future.cancelled():
        return
    exc = future.exception()
    if exc is not None:
        logger.warning(
            "a %s job failed after its caller timed out or was cancelled", kind, exc_info=exc
        )


def _percentiles(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"count": 0, "p50": None, "p95": None}
    ordered = sorted(samples)

    def pick(q: float) -> float:
        return round(ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))], 1)

    return {"count": len(ordered), "p50": pick(0.5), "p95": pick(0.95)}
