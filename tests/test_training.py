"""The training export turns decisions.jsonl into rows in the format Laya, Clef and Unsloth train on."""

import json

from local_jev.training import export_training, training_row

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "money", "technical": "bugs"},
    },
    "refund": {"type": "noul", "instructions": "Money back?"},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
}


def entry(**overrides):
    base = {
        "ts": "2026-11-24T10:00:00.000+00:00",
        "interface": "mcp",
        "kind": "decide",
        "latency_ms": 5.0,
        "questions": [{"id": qid, "type": q["type"]} for qid, q in QUESTIONS.items()],
        "model": "english",
        "backend": "laya-mlx",
        "checkpoint": "english",
        "answers": {
            "team": {"choice": "billing", "answer_confidence": 0.9566},
            "refund": {"noul": 0.8, "answer_confidence": 0.8},
            "urgency": {"score": 1.9005, "answer_confidence": 0.9045},
        },
        "input_tokens": 40,
        "warnings": [],
        "state": "Charged twice, refund today please.",
        "question_definitions": QUESTIONS,
    }
    return {**base, **overrides}


def test_a_logged_decision_becomes_a_row_with_the_model_answers_as_gold():
    row = training_row(entry())
    assert row == {
        "state": "Charged twice, refund today please.",
        "questions": QUESTIONS,
        "gold": {"team": "billing", "refund": True, "urgency": 2},
        "confidence": {"team": 0.9566, "refund": 0.8, "urgency": 0.9045},
    }


def test_a_noul_below_one_half_is_false():
    answers = {**entry()["answers"], "refund": {"noul": 0.3, "answer_confidence": 0.7}}
    assert training_row(entry(answers=answers))["gold"]["refund"] is False


def test_a_decision_logged_without_inputs_gives_no_row():
    without = entry()
    del without["state"]
    del without["question_definitions"]
    assert training_row(without) is None


def test_a_batch_record_gives_no_row():
    assert training_row(entry(kind="batch", items=3, output_path="/x.jsonl")) is None


def test_a_failed_decision_gives_no_row():
    failed = entry()
    del failed["answers"]
    assert training_row(failed) is None


def write(path, entries):
    path.write_text("".join(json.dumps(item) + "\n" for item in entries), encoding="utf-8")


def test_export_reads_the_rotated_files_oldest_first_and_counts_what_it_skips(tmp_path):
    without = entry(state=None)
    del without["state"]
    del without["question_definitions"]
    write(tmp_path / "decisions.jsonl.2", [entry(state="first")])
    write(tmp_path / "decisions.jsonl.1", [entry(state="second"), without])
    write(tmp_path / "decisions.jsonl", [entry(kind="batch"), entry(state="third")])
    export = export_training(tmp_path)
    assert [row["state"] for row in export.rows] == ["first", "second", "third"]
    assert export.without_inputs == 1
    assert export.ignored == 1


def test_export_of_a_missing_log_is_empty(tmp_path):
    export = export_training(tmp_path / "nowhere")
    assert (export.rows, export.without_inputs, export.ignored) == ([], 0, 0)
