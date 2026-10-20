import asyncio
import dataclasses
import json
import os
import threading
import time
from pathlib import Path

import pytest

from local_jev.engines.base import EngineCapabilities
from local_jev.errors import BusyError, ModelUnavailable, RuntimeTimeout, ServiceError
from local_jev.service import DecisionService, translate
from tests.fakes import FakeEngine

Q = {
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
    "dept": {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "tech"]},
}


def make(settings, engine=None, **changes):
    return DecisionService(dataclasses.replace(settings, **changes), engine or FakeEngine())


def read_log(service):
    return [json.loads(line) for line in service.log.path.read_text().splitlines()]


@pytest.mark.anyio
async def test_decide_returns_a_jev_response_and_logs_metadata(settings):
    service = make(settings)
    try:
        response = await service.decide({"state": "yes, billing", "questions": Q}, interface="test")
    finally:
        await service.stop()
    assert response["answers"]["dept"]["choice"] == "billing"
    assert response["local"]["backend"] == "fake"
    [entry] = read_log(service)
    assert entry["interface"] == "test"
    assert entry["kind"] == "decide"
    assert entry["answers"]["refund"] == {"noul": 0.9, "answer_confidence": 0.9}
    assert "state" not in entry


@pytest.mark.anyio
async def test_ab_labels_mode_reaches_the_engine(settings):
    engine = FakeEngine()
    service = make(settings, engine, noul_mode="ab_labels")
    try:
        response = await service.decide({"state": "x", "questions": Q}, interface="test")
    finally:
        await service.stop()
    assert engine.calls[0][1]["refund"]["labels"] == {"true": "A", "false": "B"}
    assert set(response["answers"]) == {"refund", "dept"}


@pytest.mark.anyio
async def test_an_invalid_request_is_422(settings):
    service = make(settings)
    try:
        with pytest.raises(ServiceError) as info:
            await service.decide({"state": "x"}, interface="test")
    finally:
        await service.stop()
    assert info.value.status == 422


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "status", "code", "hint"),
    [
        (
            ValueError("Question 'dept' has too many options for the token budget"),
            422,
            "invalid_request",
            "shorten option descriptions",
        ),
        (
            ModelUnavailable("the english checkpoint is not downloaded", "run `local-jev pull`"),
            503,
            "model_unavailable",
            "local-jev pull",
        ),
    ],
)
async def test_engine_errors_are_translated(settings, error, status, code, hint):
    service = make(settings, FakeEngine(fail_with=error))
    try:
        with pytest.raises(ServiceError) as info:
            await service.decide({"state": "x", "questions": Q}, interface="test")
    finally:
        await service.stop()
    assert (info.value.status, info.value.code) == (status, code)
    assert hint in info.value.hint


@pytest.mark.anyio
async def test_a_full_queue_is_529(settings):
    gate = threading.Event()
    service = make(settings, FakeEngine(block=gate), queue_limit=1)
    try:
        first = asyncio.create_task(
            service.decide({"state": "x", "questions": Q}, interface="test")
        )
        for _ in range(200):
            if service.runtime.pending == 1:
                break
            await asyncio.sleep(0.01)
        with pytest.raises(ServiceError) as info:
            await service.decide({"state": "y", "questions": Q}, interface="test")
        assert info.value.status == 529
        gate.set()
        await first
    finally:
        gate.set()
        await service.stop()


@pytest.mark.anyio
async def test_a_slow_answer_is_504(settings):
    gate = threading.Event()
    service = make(settings, FakeEngine(block=gate), request_timeout_seconds=0.05)
    try:
        with pytest.raises(ServiceError) as info:
            await service.decide({"state": "x", "questions": Q}, interface="test")
        assert info.value.status == 504
    finally:
        gate.set()
        await service.stop()


@pytest.mark.anyio
async def test_batch_over_inline_items(settings):
    service = make(settings)
    body = {
        "questions": Q,
        "items": ["yes billing", "no tech", {"id": "x9", "text": "yes tech"}],
        "filter": {"question": "refund", "op": ">=", "value": 0.5},
    }
    try:
        out = await service.decide_batch(body, interface="test")
    finally:
        await service.stop()
    assert out["count"] == 3
    assert out["summary"]["dept"] == {"type": "choice", "counts": {"billing": 1, "tech": 2}}
    assert out["summary"]["refund"]["yes"] == 2
    assert [match["id"] for match in out["matches"]] == [1, "x9"]
    lines = Path(out["output_path"]).read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[2])["id"] == "x9"


@pytest.mark.anyio
async def test_batch_reads_a_jsonl_file(settings, tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text(
        '{"id": "a", "body": {"text": "yes billing"}}\n{"id": "b", "body": {"text": "tech"}}\n'
    )
    service = make(settings)
    try:
        out = await service.decide_batch(
            {"questions": Q, "path": str(path), "field": "body.text"}, interface="test"
        )
    finally:
        await service.stop()
    assert out["count"] == 2
    assert out["matches"][0]["snippet"] == "yes billing"
    [entry] = read_log(service)
    assert entry["kind"] == "batch"
    assert entry["items"] == 2


@pytest.mark.anyio
async def test_batch_paths_outside_the_allowed_roots_are_rejected(settings, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "x.txt"
    outside.write_text("hello\n")
    service = make(settings)
    try:
        with pytest.raises(ServiceError, match="outside the allowed folders"):
            await service.decide_batch({"questions": Q, "path": str(outside)}, interface="test")
    finally:
        await service.stop()


@pytest.mark.anyio
async def test_the_batch_evaluation_limit_is_413(settings):
    service = make(settings, batch=dataclasses.replace(settings.batch, max_evaluations=3))
    try:
        with pytest.raises(ServiceError) as info:
            await service.decide_batch({"questions": Q, "items": ["a", "b"]}, interface="test")
    finally:
        await service.stop()
    assert info.value.status == 413


@pytest.mark.anyio
async def test_a_batch_past_its_deadline_is_504(settings):
    batch = dataclasses.replace(settings.batch, timeout_seconds=0.05, group_size=1)
    service = make(settings, FakeEngine(delay=0.03), batch=batch)
    try:
        with pytest.raises(ServiceError) as info:
            await service.decide_batch(
                {"questions": Q, "items": [str(i) for i in range(10)]}, interface="test"
            )
    finally:
        await service.stop()
    assert info.value.status == 504
    assert info.value.code == "timeout"


@pytest.mark.anyio
async def test_start_prefetches_and_cleans_old_results(settings):
    old = settings.results_dir / "old.jsonl"
    old.parent.mkdir(parents=True)
    old.write_text("{}\n")
    os.utime(old, (time.time() - 8 * 24 * 3600,) * 2)
    engine = FakeEngine()
    service = make(settings, engine, laya=dataclasses.replace(settings.laya, prefetch=("english",)))
    await service.start()
    service.prefetch_thread.join(timeout=5)
    await service.stop()
    assert engine.prefetched == ["english"]
    assert not old.exists()


def test_status_reports_engine_runtime_and_config(settings):
    service = make(settings, token="s3cret")
    try:
        status = service.status()
    finally:
        service.runtime.shutdown()
        service.log.close()
    assert status["backend"] == "fake"
    assert status["engine"]["backend"] == "fake"
    assert status["runtime"]["queue_depth"] == 0
    assert status["config"]["token_required"] is True
    assert status["memory"]["peak_rss_mb"] > 0
    assert "s3cret" not in json.dumps(status)


# Tests beyond the brief: they pin behaviour the brief's tests never reach.


class EmptyAnswers(FakeEngine):
    """A backend that returns no answer for any question."""

    def decide(self, state, questions, model):
        return dataclasses.replace(super().decide(state, questions, model), answers={})


class StallsAfter(FakeEngine):
    """Answers the first `answered` states at once, then blocks until the gate opens."""

    def __init__(self, answered, gate):
        super().__init__()
        self.answered = answered
        self.gate = gate

    def decide(self, state, questions, model):
        if len(self.calls) >= self.answered:
            self.gate.wait(5)
        return super().decide(state, questions, model)


class RemoteLike(FakeEngine):
    """What RemoteJevEngine reports about itself: another name, and no Laya guards."""

    name = "remote"
    capabilities = EngineCapabilities(laya_guards=False)


def scripted_clock(*readings):
    values = iter(readings)
    return lambda: next(values)


def record_runs(service):
    """The `kind` of every job the service hands to its runtime, in order."""
    kinds = []
    run = service.runtime.run

    async def spy(job, *, timeout, kind="decide"):
        kinds.append(kind)
        return await run(job, timeout=timeout, kind=kind)

    service.runtime.run = spy
    return kinds


@pytest.mark.parametrize(
    ("error", "status", "code", "hint"),
    [
        (BusyError("3 requests are already queued"), 529, "busy", "retry with exponential backoff"),
        (
            RuntimeTimeout("no answer within 30 s"),
            504,
            "timeout",
            "first call after an idle period",
        ),
        (
            ValueError("unknown model 'gpt'"),
            422,
            "invalid_request",
            "check the questions against the Jev schema",
        ),
    ],
)
def test_translate_maps_runtime_and_value_errors(error, status, code, hint):
    translated = translate(error)
    assert (translated.status, translated.code, translated.message) == (status, code, str(error))
    assert hint in translated.hint


def test_translate_passes_service_errors_through_and_maps_nothing_else():
    error = ServiceError(413, "too_large", "too big", "split it")
    assert translate(error) is error
    assert translate(RuntimeError("boom")) is None
    assert translate(KeyError("x")) is None


@pytest.mark.anyio
async def test_an_unexpected_engine_error_leaves_unchanged(settings):
    service = make(settings, FakeEngine(fail_with=RuntimeError("boom")))
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await service.decide({"state": "x", "questions": Q}, interface="test")
    finally:
        await service.stop()


@pytest.mark.anyio
@pytest.mark.parametrize("kind", ["decide", "batch"])
async def test_a_malformed_backend_answer_is_503(settings, kind):
    service = make(settings, EmptyAnswers())
    try:
        with pytest.raises(ServiceError) as info:
            if kind == "decide":
                await service.decide({"state": "x", "questions": Q}, interface="test")
            else:
                await service.decide_batch({"questions": Q, "items": ["x"]}, interface="test")
    finally:
        await service.stop()
    assert (info.value.status, info.value.code) == (503, "model_unavailable")
    assert info.value.hint


@pytest.mark.anyio
async def test_the_batch_evaluation_limit_is_inclusive(settings):
    service = make(settings, batch=dataclasses.replace(settings.batch, max_evaluations=4))
    try:
        out = await service.decide_batch({"questions": Q, "items": ["a", "b"]}, interface="test")
    finally:
        await service.stop()
    assert out["count"] == 2


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("third_reading", "groups_run"),
    [(400.0, 1), (299.95, 2)],
    ids=["between-groups", "inside-a-group"],
)
async def test_a_batch_deadline_says_how_many_items_finished(settings, third_reading, groups_run):
    gate = threading.Event()
    batch = dataclasses.replace(settings.batch, group_size=2)
    # The service reads the clock at the start of the batch, before group 1 and before group 2.
    # The deadline is 300 s in: at 400 s group 2 is never submitted, at 299.95 s it gets 0.05 s
    # and stalls on the gate (the engine answers the first two items, then waits).
    service = DecisionService(
        dataclasses.replace(settings, batch=batch),
        StallsAfter(2, gate),
        clock=scripted_clock(0.0, 0.0, third_reading),
    )
    kinds = record_runs(service)
    try:
        with pytest.raises(ServiceError) as info:
            await service.decide_batch(
                {"questions": Q, "items": ["a", "b", "c", "d", "e"]}, interface="test"
            )
    finally:
        gate.set()
        await service.stop()
    assert (info.value.status, info.value.code) == (504, "timeout")
    assert info.value.message == "the batch ran past 300 s after 2 of 5 items"
    assert len(kinds) == groups_run  # a group is not even submitted once the deadline has passed


@pytest.mark.anyio
async def test_a_batch_goes_to_the_worker_one_group_at_a_time(settings):
    engine = FakeEngine()
    batch = dataclasses.replace(settings.batch, group_size=2)
    service = make(settings, engine, batch=batch)
    kinds = record_runs(service)
    try:
        out = await service.decide_batch({"questions": Q, "items": list("abcde")}, interface="test")
    finally:
        await service.stop()
    assert kinds == ["batch", "batch", "batch"]
    assert out["count"] == len(engine.calls) == 5


@pytest.mark.anyio
async def test_a_remote_engine_gets_no_label_tweak_and_is_tagged_in_the_log(settings):
    engine = RemoteLike()
    service = make(settings, engine, noul_mode="ab_labels")
    try:
        await service.decide({"state": "x", "questions": Q}, interface="test")
        await service.decide_batch({"questions": Q, "items": ["x"]}, interface="test")
    finally:
        await service.stop()
    assert "labels" not in engine.calls[0][1]["refund"]
    decide_entry, batch_entry = read_log(service)
    assert decide_entry["backend"] == batch_entry["backend"] == "remote"


@pytest.mark.anyio
async def test_idle_unload_minutes_reach_the_runtime_as_seconds(settings):
    service = make(settings, idle_unload_minutes=2)
    try:
        assert service.runtime.idle_unload_seconds == 120
    finally:
        await service.stop()


@pytest.mark.anyio
async def test_batch_reads_through_a_symlink_that_stays_inside_the_allowed_roots(
    settings, tmp_path
):
    real = tmp_path / "real.jsonl"
    real.write_text('{"id": "a", "text": "yes billing"}\n')
    link = tmp_path / "link.jsonl"
    link.symlink_to(real)
    service = make(settings)
    try:
        out = await service.decide_batch(
            {"questions": Q, "path": str(link), "field": "text"}, interface="test"
        )
    finally:
        await service.stop()
    assert out["count"] == 1
    assert out["matches"][0]["snippet"] == "yes billing"
