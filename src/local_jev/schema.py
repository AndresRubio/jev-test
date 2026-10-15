"""Validate Jev /v1/systemone requests (https://docs.typesafe.ai/api.md) with fix-it hints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from local_jev.errors import invalid, too_large

QUESTION_TYPES = ("noul", "choice", "score")
MAX_QUESTIONS = 50
MAX_STATE_CHARS = 200_000
MAX_CHOICE_OPTIONS = 255
MODEL_HINT = "use auto, english, multilingual or typed-decisions"
CHOICE_EXAMPLE = 'e.g. {"billing": "invoices, refunds", "technical": "bugs, outages"}'


@dataclass(frozen=True)
class DecideRequest:
    state: Any
    questions: dict[str, dict[str, Any]]
    model: str = "auto"


def parse_decide_request(body: Any, *, max_questions: int = MAX_QUESTIONS) -> DecideRequest:
    if not isinstance(body, dict):
        raise invalid(
            "the request body must be a JSON object", 'send {"state": ..., "questions": {...}}'
        )
    if "state" not in body:
        raise invalid(
            "missing 'state'", "state is the text, object or array the questions are about"
        )
    state = parse_state(body["state"])
    questions = parse_questions(body.get("questions"), max_questions=max_questions)
    return DecideRequest(state=state, questions=questions, model=parse_model(body.get("model")))


def parse_state(state: Any) -> Any:
    if not isinstance(state, (str, dict, list)):
        raise invalid(
            "'state' must be a string, object or array", "wrap numbers and booleans in an object"
        )
    if isinstance(state, str) and not state.strip():
        raise invalid("'state' is empty", "send the text the questions are about")
    size = len(state) if isinstance(state, str) else len(json.dumps(state, ensure_ascii=False))
    if size > MAX_STATE_CHARS:
        raise too_large(
            f"'state' is {size} characters; the limit is {MAX_STATE_CHARS}",
            "send only the evidence the questions need, or use decide_batch with format=chunks",
        )
    return state


def parse_model(model: Any) -> str:
    if model is None:
        return "auto"
    if not isinstance(model, str) or not model.strip():
        raise invalid("'model' must be a non-empty string", MODEL_HINT)
    model = model.strip()
    return "auto" if model in ("auto", "jev-latest") else model


def parse_questions(
    questions: Any, *, max_questions: int = MAX_QUESTIONS
) -> dict[str, dict[str, Any]]:
    if not isinstance(questions, dict) or not questions:
        raise invalid(
            "'questions' must be a non-empty object keyed by question id",
            'e.g. {"refund": {"type": "noul", "instructions": "Does the customer ask for money back?"}}',
        )
    if len(questions) > max_questions:
        raise too_large(
            f"{len(questions)} questions; the limit is {max_questions}",
            "split them across several calls",
        )
    parsed: dict[str, dict[str, Any]] = {}
    for qid, question in questions.items():
        if not qid.strip():
            raise invalid(
                "question ids must be non-empty strings", 'use short ids such as "refund" or "dept"'
            )
        parsed[qid] = _parse_question(qid, question)
    return parsed


def _parse_question(qid: str, question: Any) -> dict[str, Any]:
    if not isinstance(question, dict):
        raise invalid(
            f"question {qid!r} must be an object",
            '{"type": ..., "instructions": ..., "criteria": ...}',
        )
    qtype = question.get("type")
    if qtype not in QUESTION_TYPES:
        raise invalid(
            f"question {qid!r} has type {qtype!r}",
            "type must be 'noul' (yes/no), 'choice' or 'score'",
        )
    if not _non_empty(question.get("instructions")):
        raise invalid(
            f"question {qid!r} has no instructions",
            "say exactly what to decide, phrased positively",
        )
    if qtype != "noul" and "labels" in question:
        raise invalid(
            f"question {qid!r}: labels are only supported on noul questions", "remove 'labels'"
        )
    parsed: dict[str, Any] = {"type": qtype, "instructions": question["instructions"]}
    criteria = question.get("criteria")
    if qtype == "noul":
        if criteria is not None:
            if (
                not isinstance(criteria, dict)
                or not criteria
                or not set(criteria) <= {"true", "false"}
            ):
                raise invalid(
                    f"question {qid!r}: noul criteria may only use the keys 'true' and 'false'",
                    'e.g. {"true": "the customer asks for a refund", "false": "anything else"}',
                )
            parsed["criteria"] = criteria
        if "labels" in question:
            parsed["labels"] = _parse_labels(qid, question["labels"])
    elif qtype == "choice":
        options = _choice_options(qid, criteria)
        if len(options) < 2:
            raise invalid(f"question {qid!r} needs at least 2 options", CHOICE_EXAMPLE)
        if len(options) > MAX_CHOICE_OPTIONS:
            raise too_large(
                f"question {qid!r} has {len(options)} options; the limit is {MAX_CHOICE_OPTIONS}",
                "split it into a coarse question and a fine question",
            )
        parsed["criteria"] = criteria
    else:
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise invalid(
                f"question {qid!r}: score criteria must be a list of 2 to 10 levels",
                'e.g. ["not urgent", "soon", "blocking"]',
            )
        for index, level in enumerate(criteria):
            if not _non_empty(level):
                raise invalid(
                    f"question {qid!r}: score level {index} has no description",
                    "every level needs a description",
                )
        parsed["criteria"] = criteria
    return parsed


def _parse_labels(qid: str, labels: Any) -> dict[str, str]:
    valid = (
        isinstance(labels, dict)
        and set(labels) == {"true", "false"}
        and all(isinstance(value, str) and value.strip() for value in labels.values())
        and labels["true"].strip() != labels["false"].strip()
    )
    if not valid:
        raise invalid(
            f"question {qid!r}: labels must map 'true' and 'false' to two different non-empty strings",
            'e.g. {"true": "A", "false": "B"}',
        )
    return labels


def _choice_options(qid: str, criteria: Any) -> list[str]:
    if isinstance(criteria, list):
        if not all(isinstance(option, str) and option.strip() for option in criteria):
            raise invalid(
                f"question {qid!r}: choice options must be non-empty strings", CHOICE_EXAMPLE
            )
        if len(set(criteria)) != len(criteria):
            raise invalid(
                f"question {qid!r}: choice options must be unique", "remove the duplicate labels"
            )
        return list(criteria)
    if isinstance(criteria, dict):
        for label, description in criteria.items():
            if not label.strip():
                raise invalid(f"question {qid!r}: choice labels must be non-empty", CHOICE_EXAMPLE)
            if description is not None and not isinstance(description, (str, dict, list)):
                raise invalid(
                    f"question {qid!r}: option {label!r} must be described by a string, object, array or null",
                    CHOICE_EXAMPLE,
                )
        return list(criteria)
    raise invalid(
        f"question {qid!r}: choice criteria must be a list of labels or an object of label -> description",
        CHOICE_EXAMPLE,
    )


def _non_empty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, (dict, list)) and bool(value)
