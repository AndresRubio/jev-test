import asyncio
import threading

import pytest

from local_jev.errors import BusyError, RuntimeTimeout
from local_jev.runtime import Runtime
from tests.fakes import FakeEngine

QUESTIONS = {"q": {"type": "noul", "instructions": "?"}}


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


async def wait_for(condition, attempts=200):
    for _ in range(attempts):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


@pytest.mark.anyio
async def test_run_returns_the_job_result():
    runtime = Runtime(FakeEngine(), idle_unload_seconds=0)
    try:
        assert await runtime.run(lambda engine: engine.name, timeout=5) == "fake"
    finally:
        runtime.shutdown()


@pytest.mark.anyio
async def test_jobs_never_overlap():
    engine = FakeEngine()
    runtime = Runtime(engine, idle_unload_seconds=0)
    try:
        await asyncio.gather(
            *(
                runtime.run(lambda e: e.decide("yes", QUESTIONS, "auto"), timeout=5)
                for _ in range(8)
            )
        )
    finally:
        runtime.shutdown()
    assert engine.max_active == 1
    assert len(engine.calls) == 8


@pytest.mark.anyio
async def test_a_full_queue_raises_busy():
    gate = threading.Event()
    runtime = Runtime(FakeEngine(block=gate), queue_limit=2, idle_unload_seconds=0)
    try:
        first = asyncio.create_task(
            runtime.run(lambda e: e.decide("a", QUESTIONS, "auto"), timeout=5)
        )
        second = asyncio.create_task(
            runtime.run(lambda e: e.decide("b", QUESTIONS, "auto"), timeout=5)
        )
        await wait_for(lambda: runtime.pending == 2)
        with pytest.raises(BusyError):
            await runtime.run(lambda e: e.decide("c", QUESTIONS, "auto"), timeout=5)
        gate.set()
        await asyncio.gather(first, second)
        await wait_for(lambda: runtime.pending == 0)
    finally:
        gate.set()
        runtime.shutdown()


@pytest.mark.anyio
async def test_a_slow_job_times_out_and_frees_its_slot():
    gate = threading.Event()
    runtime = Runtime(FakeEngine(block=gate), idle_unload_seconds=0)
    try:
        with pytest.raises(RuntimeTimeout):
            await runtime.run(lambda e: e.decide("a", QUESTIONS, "auto"), timeout=0.05)
        gate.set()
        await wait_for(lambda: runtime.pending == 0)
    finally:
        gate.set()
        runtime.shutdown()


@pytest.mark.anyio
async def test_an_idle_engine_is_unloaded_once():
    clock = FakeClock()
    engine = FakeEngine()
    runtime = Runtime(engine, idle_unload_seconds=60, clock=clock)
    try:
        await runtime.run(lambda e: e.decide("a", QUESTIONS, "auto"), timeout=5)
        await wait_for(lambda: runtime.pending == 0)
        assert engine.is_loaded()
        clock.now += 30
        assert runtime.check_idle() is False
        clock.now += 31
        assert runtime.check_idle() is True
        await wait_for(lambda: engine.unload_calls == 1)
        assert not engine.is_loaded()
        clock.now += 120
        assert runtime.check_idle() is False
    finally:
        runtime.shutdown()


def test_zero_disables_idle_unload():
    clock = FakeClock()
    engine = FakeEngine()
    engine.loaded = True
    runtime = Runtime(engine, idle_unload_seconds=0, clock=clock)
    clock.now += 10_000
    assert runtime.check_idle() is False
    runtime.shutdown()


@pytest.mark.anyio
async def test_stats_report_decide_latency_percentiles():
    runtime = Runtime(FakeEngine(), idle_unload_seconds=0)
    try:
        for _ in range(5):
            await runtime.run(lambda e: None, timeout=5)
        await runtime.run(lambda e: None, timeout=5, kind="batch")
        await wait_for(lambda: runtime.pending == 0)
        stats = runtime.stats()
    finally:
        runtime.shutdown()
    assert stats["decide_latency_ms"]["count"] == 5
    assert stats["decide_latency_ms"]["p50"] is not None
    assert stats["queue_depth"] == 0
    assert stats["idle_unload_seconds"] == 0


def logged(caplog, message):
    return any(record.exc_info and str(record.exc_info[1]) == message for record in caplog.records)


@pytest.mark.anyio
async def test_a_job_that_fails_after_its_timeout_is_logged(caplog):
    gate = threading.Event()
    engine = FakeEngine(block=gate, fail_with=RuntimeError("late failure"))
    runtime = Runtime(engine, idle_unload_seconds=0)
    try:
        caller = asyncio.create_task(
            runtime.run(lambda e: e.decide("a", QUESTIONS, "auto"), timeout=0.2)
        )
        await wait_for(lambda: engine.active == 1)
        with pytest.raises(RuntimeTimeout):
            await caller
        gate.set()
        await wait_for(lambda: logged(caplog, "late failure"))
    finally:
        gate.set()
        runtime.shutdown()


@pytest.mark.anyio
async def test_a_job_that_fails_after_its_caller_is_cancelled_is_logged(caplog):
    gate = threading.Event()
    engine = FakeEngine(block=gate, fail_with=RuntimeError("late failure"))
    runtime = Runtime(engine, idle_unload_seconds=0)
    try:
        caller = asyncio.create_task(
            runtime.run(lambda e: e.decide("a", QUESTIONS, "auto"), timeout=5)
        )
        await wait_for(lambda: engine.active == 1)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        gate.set()
        await wait_for(lambda: logged(caplog, "late failure"))
    finally:
        gate.set()
        runtime.shutdown()


class UnloadFails(FakeEngine):
    def unload(self) -> None:
        self.unload_calls += 1
        raise RuntimeError("unload failed")


@pytest.mark.anyio
async def test_a_failed_unload_is_logged(caplog):
    clock = FakeClock()
    engine = UnloadFails()
    engine.loaded = True
    runtime = Runtime(engine, idle_unload_seconds=60, clock=clock)
    try:
        clock.now += 61
        assert runtime.check_idle() is True
        await wait_for(lambda: logged(caplog, "unload failed"))
    finally:
        runtime.shutdown()


@pytest.mark.anyio
async def test_a_job_refused_after_shutdown_frees_its_slot():
    runtime = Runtime(FakeEngine(), idle_unload_seconds=0)
    runtime.shutdown()
    with pytest.raises(RuntimeError):
        await runtime.run(lambda e: None, timeout=5)
    assert runtime.pending == 0
