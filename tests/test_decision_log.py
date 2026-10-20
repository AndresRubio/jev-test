import json

from local_jev import decision_log
from local_jev.decision_log import DecisionLog
from local_jev.guard import finalize
from tests.fakes import FakeEngine

QUESTIONS = {"refund": {"type": "noul", "instructions": "Money back?"}}


def response():
    return finalize(
        FakeEngine().decide("yes", QUESTIONS, "auto"),
        QUESTIONS,
        backend="fake",
        latency_ms=5.0,
        laya_guards=True,
    )


def entries(log):
    return [json.loads(line) for line in log.path.read_text().splitlines()]


def test_records_hold_metadata_only_by_default(tmp_path):
    log = DecisionLog(tmp_path, log_inputs=False)
    log.record(
        interface="mcp",
        kind="decide",
        response=response(),
        questions=QUESTIONS,
        latency_ms=5.04,
        state="yes please",
    )
    log.close()
    [entry] = entries(log)
    assert entry["interface"] == "mcp"
    assert entry["kind"] == "decide"
    assert (entry["model"], entry["backend"], entry["checkpoint"]) == (
        "fake-model",
        "fake",
        "english",
    )
    assert entry["questions"] == [{"id": "refund", "type": "noul"}]
    assert entry["answers"] == {"refund": {"noul": 0.9, "answer_confidence": 0.9}}
    assert entry["input_tokens"] == 10
    assert entry["latency_ms"] == 5.0
    assert "state" not in entry
    assert "question_definitions" not in entry
    assert entry["ts"].endswith("+00:00")


def test_inputs_are_logged_when_enabled(tmp_path):
    log = DecisionLog(tmp_path, log_inputs=True)
    log.record(
        interface="rest",
        kind="decide",
        response=response(),
        questions=QUESTIONS,
        latency_ms=1.0,
        state="yes please",
    )
    log.close()
    [entry] = entries(log)
    assert entry["state"] == "yes please"
    assert entry["question_definitions"] == QUESTIONS


def test_batch_records_carry_extra_fields(tmp_path):
    log = DecisionLog(tmp_path, log_inputs=False)
    log.record(
        interface="rest",
        kind="batch",
        response=None,
        questions=QUESTIONS,
        latency_ms=99.0,
        extra={"items": 3, "output_path": "/x.jsonl"},
    )
    log.close()
    [entry] = entries(log)
    assert entry["kind"] == "batch"
    assert entry["items"] == 3
    assert "answers" not in entry


def test_the_log_rotates(tmp_path):
    log = DecisionLog(tmp_path, log_inputs=False, max_bytes=400, backups=2)
    for _ in range(20):
        log.record(
            interface="rest",
            kind="decide",
            response=response(),
            questions=QUESTIONS,
            latency_ms=1.0,
        )
    log.close()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "decisions.jsonl",
        "decisions.jsonl.1",
        "decisions.jsonl.2",
    ]


def test_no_state_or_question_text_reaches_the_file_by_default(tmp_path):
    questions = {
        "dept": {
            "type": "choice",
            "instructions": "SECRET instructions",
            "criteria": {"billing": "SECRET billing criterion", "tech": "SECRET tech criterion"},
        }
    }
    state = "SECRET state: my tech problem"
    result = finalize(
        FakeEngine().decide(state, questions, "auto"),
        questions,
        backend="fake",
        latency_ms=1.0,
        laya_guards=True,
    )
    log = DecisionLog(tmp_path, log_inputs=False)
    log.record(
        interface="rest",
        kind="decide",
        response=result,
        questions=questions,
        latency_ms=1.0,
        state=state,
    )
    log.close()
    text = log.path.read_text()
    assert "SECRET" not in text
    assert json.loads(text)["answers"] == {"dept": {"choice": "tech", "answer_confidence": 0.8}}


def test_two_logs_never_write_into_each_others_files(tmp_path, monkeypatch):
    # Pin ids to one value so both logs would share a name if the logger were named after id(self).
    monkeypatch.setattr(decision_log, "id", lambda _obj: 1, raising=False)
    stale = DecisionLog(tmp_path / "stale", log_inputs=False)
    fresh = DecisionLog(tmp_path / "fresh", log_inputs=False)
    fresh.record(interface="rest", kind="batch", response=None, questions=QUESTIONS, latency_ms=1.0)
    fresh.close()
    stale.close()
    assert stale.path.read_text() == ""
    assert len(fresh.path.read_text().splitlines()) == 1
