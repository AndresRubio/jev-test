import pytest

from local_jev.errors import ServiceError
from local_jev.schema import DecideRequest, parse_decide_request
from tests.samples import JEV_DOC_EXAMPLE

NOUL = {"type": "noul", "instructions": "?"}


def test_the_documented_jev_request_parses():
    request = parse_decide_request(JEV_DOC_EXAMPLE)
    assert isinstance(request, DecideRequest)
    assert request.model == "auto"
    assert request.state == {"body": "billed twice, refund please or we cancel"}
    assert list(request.questions) == ["dept", "refund", "urgency"]
    assert request.questions["dept"]["criteria"] == {"billing": "refunds", "tech": "bugs"}


def test_choice_lists_and_null_descriptions_are_accepted():
    request = parse_decide_request(
        {
            "state": "x",
            "questions": {
                "a": {"type": "choice", "instructions": "pick", "criteria": ["red", "green"]},
                "b": {
                    "type": "choice",
                    "instructions": "pick",
                    "criteria": {"red": None, "green": "grass"},
                },
            },
        }
    )
    assert request.questions["a"]["criteria"] == ["red", "green"]
    assert request.questions["b"]["criteria"] == {"red": None, "green": "grass"}


def test_noul_labels_pass_through():
    request = parse_decide_request(
        {"state": "x", "questions": {"q": {**NOUL, "labels": {"true": "A", "false": "B"}}}}
    )
    assert request.questions["q"]["labels"] == {"true": "A", "false": "B"}


def test_unknown_question_fields_are_dropped():
    request = parse_decide_request({"state": "x", "questions": {"q": {**NOUL, "colour": "blue"}}})
    assert request.questions["q"] == {"type": "noul", "instructions": "?"}


def test_explicit_model_is_kept():
    body = {"state": "x", "model": "multilingual", "questions": {"q": NOUL}}
    assert parse_decide_request(body).model == "multilingual"


def choice(criteria, **extra):
    return {
        "state": "x",
        "questions": {"q": {"type": "choice", "instructions": "?", "criteria": criteria, **extra}},
    }


def score(criteria):
    return {
        "state": "x",
        "questions": {"q": {"type": "score", "instructions": "?", "criteria": criteria}},
    }


@pytest.mark.parametrize(
    ("body", "status", "fragment"),
    [
        ([], 422, "JSON object"),
        ({"questions": {"q": NOUL}}, 422, "missing 'state'"),
        ({"state": 5, "questions": {"q": NOUL}}, 422, "'state' must be"),
        ({"state": "   ", "questions": {"q": NOUL}}, 422, "'state' is empty"),
        ({"state": "x", "questions": {}}, 422, "non-empty object"),
        ({"state": "x", "questions": {"q": {"type": "maybe", "instructions": "?"}}}, 422, "type"),
        ({"state": "x", "questions": {"q": {"type": "noul"}}}, 422, "no instructions"),
        (
            {"state": "x", "questions": {"q": {**NOUL, "criteria": {"yes": "y"}}}},
            422,
            "'true' and 'false'",
        ),
        (
            {"state": "x", "questions": {"q": {**NOUL, "labels": {"true": "A", "false": "A"}}}},
            422,
            "labels",
        ),
        (choice(["only"]), 422, "at least 2"),
        (choice(["a", "a"]), 422, "unique"),
        (choice("a,b"), 422, "list of labels"),
        (choice(["a", "b"], labels={"true": "A", "false": "B"}), 422, "only supported on noul"),
        (score(["only"]), 422, "2 to 10"),
        (score(["low", None]), 422, "level 1"),
        ({"state": "x" * 200_001, "questions": {"q": NOUL}}, 413, "200000"),
        ({"state": "x", "questions": {f"q{i}": NOUL for i in range(51)}}, 413, "limit is 50"),
        ({"state": "x", "model": "", "questions": {"q": NOUL}}, 422, "'model'"),
    ],
)
def test_invalid_requests_explain_how_to_fix_them(body, status, fragment):
    with pytest.raises(ServiceError) as info:
        parse_decide_request(body)
    assert info.value.status == status
    assert fragment in info.value.message
    assert info.value.hint
