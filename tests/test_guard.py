import pytest

from local_jev.engines.base import EngineResult
from local_jev.errors import ModelUnavailable
from local_jev.guard import (
    AB_LABELS,
    answer_confidence,
    compact_answers,
    finalize,
    prepare_questions,
    warning_code,
)
from tests.fakes import FakeEngine

QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
    "dept": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "refunds", "tech": "bugs"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["low", "medium", "high"],
    },
}


def run(engine=None, questions=QUESTIONS, state="yes, billing please"):
    engine = engine or FakeEngine()
    return finalize(
        engine.decide(state, questions, "auto"), questions, backend="fake", latency_ms=12.34
    )


def test_native_mode_returns_an_unchanged_deep_copy():
    prepared = prepare_questions(QUESTIONS, noul_mode="native", laya_guards=True)
    assert prepared == QUESTIONS and prepared is not QUESTIONS
    prepared["refund"]["instructions"] = "changed"
    assert QUESTIONS["refund"]["instructions"] == "Does the customer ask for money back?"


def test_ab_labels_mode_adds_neutral_labels_to_noul_questions_only():
    custom = {
        **QUESTIONS,
        "own": {"type": "noul", "instructions": "?", "labels": {"true": "Y", "false": "N"}},
    }
    prepared = prepare_questions(custom, noul_mode="ab_labels", laya_guards=True)
    assert prepared["refund"]["labels"] == AB_LABELS
    assert prepared["own"]["labels"] == {"true": "Y", "false": "N"}
    assert "labels" not in prepared["dept"]
    assert "labels" not in custom["refund"]


def test_ab_labels_mode_is_ignored_for_engines_without_laya_guards():
    prepared = prepare_questions(QUESTIONS, noul_mode="ab_labels", laya_guards=False)
    assert "labels" not in prepared["refund"]


def test_finalize_keeps_jev_fields_and_adds_the_local_block():
    response = run()
    assert set(response) == {"model", "answers", "usage", "local"}
    assert response["model"] == "fake-model"
    assert response["usage"] == {"input_tokens": 30, "output_tokens": 0}
    dept = response["answers"]["dept"]
    assert set(dept) == {
        "type",
        "choice",
        "probabilities",
        "confidence",
        "answer_confidence",
        "warnings",
    }
    assert dept["choice"] == "billing"
    assert dept["warnings"] == []
    assert set(response["answers"]["refund"]) == {
        "type",
        "noul",
        "confidence",
        "answer_confidence",
        "warnings",
    }
    assert set(response["answers"]["urgency"]) == {
        "type",
        "score",
        "legend",
        "probabilities",
        "confidence",
        "answer_confidence",
        "warnings",
    }
    assert response["local"] == {
        "backend": "fake",
        "checkpoint": "english",
        "routing_reason": "fake routing",
        "latency_ms": 12.3,
        "warnings": [],
    }


def test_truncation_and_collapsed_options_become_warnings():
    engine = FakeEngine(
        usage_extra={
            "state_tokens": 900,
            "state_tokens_dropped": 412,
            "truncated": True,
            "truncated_questions": ["refund"],
            "options": {"dept": {"total": 2, "distinct": 1, "tokens_per_option": 1}},
        }
    )
    response = run(engine)
    assert response["answers"]["refund"]["warnings"] == [
        "state_truncated: 412 of 900 input tokens did not fit and were ignored; "
        "use decide_batch with format=chunks, or model=multilingual"
    ]
    assert warning_code(response["answers"]["dept"]["warnings"][0]) == "options_collapsed"
    assert response["answers"]["urgency"]["warnings"] == []
    assert response["local"]["warnings"][0].startswith("refund: state_truncated:")
    assert response["local"]["usage"]["state_tokens_dropped"] == 412
    assert response["usage"] == {"input_tokens": 30, "output_tokens": 0}


def test_shortlist_metadata_is_reported():
    engine = FakeEngine(
        shortlist={
            "dept": {
                "labels": ["billing", "tech"],
                "scores": [0.9, 0.5],
                "k": 2,
                "n": 30,
                "passthrough": False,
            }
        }
    )
    dept = run(engine)["answers"]["dept"]
    assert dept["shortlisted"] == {"kept": ["billing", "tech"], "dropped": 28}
    assert dept["warnings"][0].startswith("shortlisted: 30 options were narrowed to the 2")


def test_a_passthrough_shortlist_is_silent():
    engine = FakeEngine(
        shortlist={
            "dept": {
                "labels": ["billing", "tech"],
                "scores": None,
                "k": 20,
                "n": 2,
                "passthrough": True,
            }
        }
    )
    dept = run(engine)["answers"]["dept"]
    assert "shortlisted" not in dept
    assert dept["warnings"] == []


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"type": "noul", "noul": 0.3}, 0.7),
        ({"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.4}}, 0.6),
        ({"type": "score", "score": 1.2, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}, 0.6),
        ({"type": "noul", "noul": 0.3, "answer_confidence": 0.91}, 0.91),
    ],
)
def test_answer_confidence(answer, expected):
    assert answer_confidence(answer) == expected


def test_a_missing_answer_is_model_unavailable():
    result = EngineResult(answers={}, usage={}, model="remote")
    with pytest.raises(ModelUnavailable, match="no answer for question 'refund'"):
        finalize(result, {"refund": QUESTIONS["refund"]}, backend="remote", latency_ms=1.0)


def test_compact_answers():
    compact = compact_answers(run()["answers"])
    assert compact["dept"] == {"choice": "billing", "answer_confidence": 0.8}
    assert compact["refund"] == {"noul": 0.9, "answer_confidence": 0.9}
    assert set(compact["urgency"]) == {"score", "answer_confidence"}
