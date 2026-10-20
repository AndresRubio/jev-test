import pytest

from local_jev.engines.base import EngineResult
from local_jev.errors import ModelUnavailable
from local_jev.guard import (
    AB_LABELS,
    BACKEND_HINT,
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


def run(engine=None, questions=QUESTIONS, state="yes, billing please", laya_guards=True):
    engine = engine or FakeEngine()
    return finalize(
        engine.decide(state, questions, "auto"),
        questions,
        backend="fake",
        latency_ms=12.34,
        laya_guards=laya_guards,
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


LAYA_USAGE = {
    "state_tokens": 900,
    "state_tokens_dropped": 412,
    "truncated": True,
    "truncated_questions": ["refund"],
    "options": {"dept": {"total": 2, "distinct": 1, "tokens_per_option": 1}},
}


def test_truncation_and_collapsed_options_become_warnings():
    engine = FakeEngine(usage_extra=LAYA_USAGE)
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


def test_an_engine_without_laya_guards_gets_no_truncation_or_collapse_warnings():
    # Spec section 7 rule 8: a remote backend gets rules 1 and 6 only, so rule 5 is skipped.
    response = run(FakeEngine(usage_extra=LAYA_USAGE), laya_guards=False)
    assert [answer["warnings"] for answer in response["answers"].values()] == [[], [], []]
    assert response["local"]["warnings"] == []


def test_shortlist_metadata_is_reported():
    engine = FakeEngine(
        shortlist={
            "dept": {"labels": ["billing", "tech"], "n": 30, "passthrough": False, "rounds": 1}
        }
    )
    dept = run(engine)["answers"]["dept"]
    assert dept["shortlisted"] == {"kept": ["billing", "tech"], "dropped": 28}
    assert dept["warnings"] == [
        "shortlisted: 30 options were narrowed to 2 finalists by elimination rounds; "
        "probabilities cover only those"
    ]


def test_a_passthrough_shortlist_is_silent():
    engine = FakeEngine(
        shortlist={"dept": {"labels": ["billing", "tech"], "n": 2, "passthrough": True}}
    )
    dept = run(engine)["answers"]["dept"]
    assert "shortlisted" not in dept
    assert dept["warnings"] == []


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"type": "noul", "noul": 0.3}, 0.7),
        ({"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.4}}, 0.6),
        ({"type": "choice", "choice": "b", "probabilities": {"a": 0.6, "b": 0.4}}, 0.4),
        ({"type": "score", "score": 1.2, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}, 0.6),
        ({"type": "noul", "noul": 0.3, "answer_confidence": 0.91}, 0.91),
    ],
)
def test_answer_confidence(answer, expected):
    assert answer_confidence(answer) == expected


def test_a_missing_answer_is_model_unavailable():
    result = EngineResult(answers={}, usage={}, model="remote")
    with pytest.raises(ModelUnavailable, match="no answer for question 'refund'"):
        finalize(
            result,
            {"refund": QUESTIONS["refund"]},
            backend="remote",
            latency_ms=1.0,
            laya_guards=False,
        )


def test_compact_answers():
    compact = compact_answers(run()["answers"])
    assert compact["dept"] == {"choice": "billing", "answer_confidence": 0.8}
    assert compact["refund"] == {"noul": 0.9, "answer_confidence": 0.9}
    assert set(compact["urgency"]) == {"score", "answer_confidence"}


NAN = float("nan")
INF = float("inf")
CHOICE_PROBABILITIES = {"billing": 0.8, "tech": 0.2}
SCORE_PROBABILITIES = {"0": 0.15, "1": 0.15, "2": 0.7}


def from_backend(answers, questions, usage=None):
    result = EngineResult(answers=answers, usage=usage or {}, model="remote")
    return finalize(result, questions, backend="remote", latency_ms=1.0, laya_guards=False)


def test_a_remote_choice_that_is_not_the_most_likely_label_gets_its_own_probability():
    # Spec section 7 rule 6: for a choice, answer_confidence is the probability of the chosen label.
    answers = {
        "dept": {"type": "choice", "choice": "tech", "probabilities": {"billing": 0.7, "tech": 0.3}}
    }
    response = from_backend(answers, {"dept": QUESTIONS["dept"]})
    assert response["answers"]["dept"]["answer_confidence"] == 0.3


def test_an_answer_without_a_type_takes_the_questions_type_and_a_computed_confidence():
    answers = {
        "refund": {"noul": 0.25, "confidence": 0.4},
        "dept": {"choice": "tech", "probabilities": {"billing": 0.3, "tech": 0.7}},
    }
    questions = {"refund": QUESTIONS["refund"], "dept": QUESTIONS["dept"]}
    response = from_backend(answers, questions)
    assert response["answers"]["refund"] == {
        "type": "noul",
        "noul": 0.25,
        "confidence": 0.4,
        "answer_confidence": 0.75,
        "warnings": [],
    }
    assert response["answers"]["dept"]["type"] == "choice"
    assert response["answers"]["dept"]["answer_confidence"] == 0.7


def test_a_choice_answer_without_probabilities_is_model_unavailable():
    answers = {"dept": {"type": "choice", "choice": "billing", "confidence": 0.3}}
    with pytest.raises(ModelUnavailable, match="'dept' is missing 'probabilities'") as caught:
        from_backend(answers, {"dept": QUESTIONS["dept"]})
    assert caught.value.hint == BACKEND_HINT


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param({"type": "choice", "choice": "a", "confidence": 0.3}, id="no probabilities"),
        pytest.param(
            {"type": "score", "score": 1.0, "probabilities": {}, "confidence": 0.3},
            id="empty probabilities",
        ),
        pytest.param({}, id="nothing to go on"),
    ],
)
def test_answer_confidence_never_borrows_the_entropy_confidence(answer):
    with pytest.raises(ModelUnavailable) as caught:
        answer_confidence(answer)
    assert caught.value.hint == BACKEND_HINT


MALFORMED_ANSWERS = [
    # not an answer
    pytest.param("refund", None, "no answer for question 'refund'", id="null"),
    pytest.param("refund", "yes", "'refund' that is not an object", id="text"),
    pytest.param("refund", [0.9], "'refund' that is not an object", id="list"),
    # a type that does not fit
    pytest.param("refund", {"type": ["noul"], "noul": 0.9}, "unknown answer type", id="list type"),
    pytest.param(
        "refund", {"type": "maybe", "noul": 0.9}, "unknown answer type 'maybe'", id="unknown type"
    ),
    pytest.param(
        "refund",
        {"type": "choice", "choice": "billing", "probabilities": CHOICE_PROBABILITIES},
        "'choice' answer for 'refund', which is a 'noul' question",
        id="choice answer to a noul question",
    ),
    pytest.param(
        "dept",
        {"type": "score", "score": 1.0, "probabilities": SCORE_PROBABILITIES},
        "'score' answer for 'dept', which is a 'choice' question",
        id="score answer to a choice question",
    ),
    pytest.param(
        "urgency",
        {"type": "noul", "noul": 0.9},
        "'noul' answer for 'urgency', which is a 'score' question",
        id="noul answer to a score question",
    ),
    # a missing value
    pytest.param("refund", {}, "'refund' is missing 'noul'", id="empty noul answer"),
    pytest.param(
        "refund",
        {"type": "noul", "confidence": 0.9},
        "'refund' is missing 'noul'",
        id="noul without noul",
    ),
    pytest.param(
        "dept",
        {"type": "choice", "probabilities": CHOICE_PROBABILITIES},
        "'dept' is missing 'choice'",
        id="choice without choice",
    ),
    pytest.param(
        "dept",
        {"type": "choice", "choice": "billing"},
        "'dept' is missing 'probabilities'",
        id="choice without probabilities",
    ),
    pytest.param(
        "dept", {}, "'dept' is missing 'choice', 'probabilities'", id="empty choice answer"
    ),
    pytest.param(
        "urgency",
        {"type": "score", "probabilities": SCORE_PROBABILITIES},
        "'urgency' is missing 'score'",
        id="score without score",
    ),
    pytest.param(
        "urgency",
        {"type": "score", "score": 1.0},
        "'urgency' is missing 'probabilities'",
        id="score without probabilities",
    ),
    pytest.param(
        "dept",
        {"type": "choice", "choice": "billing", "probabilities": {}},
        "'dept' is missing 'probabilities'",
        id="empty probabilities",
    ),
    pytest.param(
        "dept",
        {"type": "choice", "choice": "billing", "probabilities": {}, "answer_confidence": 0.9},
        "'dept' is missing 'probabilities'",
        id="empty probabilities with an answer_confidence",
    ),
    pytest.param(
        "dept",
        {"type": "choice", "choice": "sales", "probabilities": CHOICE_PROBABILITIES},
        "the chosen label has no probability",
        id="a choice outside its probabilities",
    ),
    pytest.param(
        "dept",
        {"type": "choice", "choice": ["billing"], "probabilities": CHOICE_PROBABILITIES},
        "the chosen label has no probability",
        id="a choice that is not a label",
    ),
    # a number that is missing, not a number, or not finite
    pytest.param("refund", {"type": "noul", "noul": None}, "not finite numbers", id="noul null"),
    pytest.param("refund", {"type": "noul", "noul": "abc"}, "not finite numbers", id="noul text"),
    pytest.param("refund", {"type": "noul", "noul": NAN}, "not finite numbers", id="noul nan"),
    pytest.param("refund", {"type": "noul", "noul": INF}, "not finite numbers", id="noul infinity"),
    pytest.param(
        "refund", {"type": "noul", "noul": -INF}, "not finite numbers", id="noul negative infinity"
    ),
    pytest.param(
        "refund",
        {"type": "noul", "noul": 10**400},
        "not finite numbers",
        id="noul too large for a float",
    ),
    pytest.param(
        "refund",
        {"type": "noul", "noul": 0.9, "answer_confidence": None},
        "not finite numbers",
        id="answer_confidence null",
    ),
    pytest.param(
        "refund",
        {"type": "noul", "noul": 0.9, "answer_confidence": "abc"},
        "not finite numbers",
        id="answer_confidence text",
    ),
    pytest.param(
        "refund",
        {"type": "noul", "noul": 0.9, "answer_confidence": NAN},
        "not finite numbers",
        id="answer_confidence nan",
    ),
    pytest.param(
        "refund",
        {"type": "noul", "noul": 0.9, "answer_confidence": INF},
        "not finite numbers",
        id="answer_confidence infinity",
    ),
    pytest.param(
        "urgency",
        {"type": "score", "score": NAN, "probabilities": SCORE_PROBABILITIES},
        "not finite numbers",
        id="score nan",
    ),
    pytest.param(
        "urgency",
        {"type": "score", "score": INF, "probabilities": SCORE_PROBABILITIES},
        "not finite numbers",
        id="score infinity",
    ),
    pytest.param(
        "urgency",
        {"type": "score", "score": "high", "probabilities": SCORE_PROBABILITIES},
        "not finite numbers",
        id="score text",
    ),
    pytest.param(
        "dept",
        {"type": "choice", "choice": "billing", "probabilities": ["billing", "tech"]},
        "not finite numbers",
        id="probabilities as a list",
    ),
    pytest.param(
        "dept",
        {"type": "choice", "choice": "billing", "probabilities": {"billing": None, "tech": 0.2}},
        "not finite numbers",
        id="probability null",
    ),
    pytest.param(
        "dept",
        {"type": "choice", "choice": "billing", "probabilities": {"billing": "x", "tech": 0.2}},
        "not finite numbers",
        id="probability text",
    ),
    pytest.param(
        "dept",
        {"type": "choice", "choice": "billing", "probabilities": {"billing": NAN, "tech": 0.2}},
        "not finite numbers",
        id="probability nan",
    ),
    pytest.param(
        "dept",
        {"type": "choice", "choice": "billing", "probabilities": {"billing": INF, "tech": 0.2}},
        "not finite numbers",
        id="probability infinity",
    ),
    pytest.param(
        "dept",
        {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8, "tech": NAN}},
        "not finite numbers",
        id="probability nan after a larger one",
    ),
    pytest.param(
        "dept",
        {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8, "tech": -INF}},
        "not finite numbers",
        id="probability negative infinity after a larger one",
    ),
]


@pytest.mark.parametrize(("qid", "answer", "problem"), MALFORMED_ANSWERS)
def test_a_malformed_backend_answer_is_model_unavailable(qid, answer, problem):
    with pytest.raises(ModelUnavailable, match=problem) as caught:
        from_backend({qid: answer}, {qid: QUESTIONS[qid]})
    assert caught.value.hint == BACKEND_HINT


@pytest.mark.parametrize(
    ("usage", "field"),
    [
        pytest.param({"input_tokens": None}, "usage.input_tokens", id="input tokens null"),
        pytest.param({"input_tokens": "many"}, "usage.input_tokens", id="input tokens text"),
        pytest.param({"input_tokens": NAN}, "usage.input_tokens", id="input tokens nan"),
        pytest.param({"input_tokens": INF}, "usage.input_tokens", id="input tokens infinity"),
        pytest.param(
            {"input_tokens": -INF}, "usage.input_tokens", id="input tokens negative infinity"
        ),
        pytest.param({"output_tokens": None}, "usage.output_tokens", id="output tokens null"),
        pytest.param({"output_tokens": NAN}, "usage.output_tokens", id="output tokens nan"),
    ],
)
def test_unusable_token_counts_are_model_unavailable(usage, field):
    answers = {"refund": {"type": "noul", "noul": 0.9}}
    with pytest.raises(ModelUnavailable, match=field) as caught:
        from_backend(answers, {"refund": QUESTIONS["refund"]}, usage=usage)
    assert caught.value.hint == BACKEND_HINT


def test_numbers_sent_as_text_are_coerced():
    answers = {
        "refund": {"type": "noul", "noul": "0.25"},
        "dept": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": "0.8", "tech": "0.2"},
        },
    }
    questions = {"refund": QUESTIONS["refund"], "dept": QUESTIONS["dept"]}
    usage = {"input_tokens": "12", "output_tokens": 3.0}
    response = from_backend(answers, questions, usage=usage)
    assert response["answers"]["refund"]["noul"] == 0.25
    assert response["answers"]["refund"]["answer_confidence"] == 0.75
    assert response["answers"]["dept"]["probabilities"] == {"billing": 0.8, "tech": 0.2}
    assert response["answers"]["dept"]["answer_confidence"] == 0.8
    assert response["usage"] == {"input_tokens": 12, "output_tokens": 3}
