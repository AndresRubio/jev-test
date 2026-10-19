"""Engine-agnostic request tweaks and response shaping (spec §7)."""

from __future__ import annotations

import copy
import math
from typing import Any

from local_jev.engines.base import EngineResult
from local_jev.errors import ModelUnavailable

AB_LABELS = {"true": "A", "false": "B"}
JEV_FIELDS = {
    "noul": ("type", "noul", "confidence"),
    "choice": ("type", "choice", "probabilities", "confidence"),
    "score": ("type", "score", "legend", "probabilities", "confidence"),
}
REQUIRED_FIELDS = {
    "noul": ("noul",),
    "choice": ("choice", "probabilities"),
    "score": ("score", "probabilities"),
}
TRUNCATION_HINT = "use decide_batch with format=chunks, or model=multilingual"
BACKEND_HINT = "check that the backend speaks Jev's /v1/systemone format"


def prepare_questions(
    questions: dict[str, dict[str, Any]], *, noul_mode: str, laya_guards: bool
) -> dict[str, dict[str, Any]]:
    """A deep copy of the questions with the model-facing tweaks applied."""
    prepared = copy.deepcopy(questions)
    if laya_guards and noul_mode == "ab_labels":
        for question in prepared.values():
            if question["type"] == "noul" and "labels" not in question:
                question["labels"] = dict(AB_LABELS)
    return prepared


def answer_confidence(answer: dict[str, Any]) -> float:
    """Probability of the reported answer: max(p) for choice/score, max(p, 1 - p) for noul.

    The entropy-based `confidence` is a different quantity, so it is never used in its place.
    Raises ModelUnavailable when the answer has no probabilities to compute the value from.
    """
    if "answer_confidence" in answer:
        return round(float(answer["answer_confidence"]), 4)
    if answer.get("type") == "noul":
        p = float(answer["noul"])
        return round(max(p, 1.0 - p), 4)
    probabilities = answer.get("probabilities") or {}
    if probabilities:
        return round(max(float(value) for value in probabilities.values()), 4)
    raise ModelUnavailable(
        "the answer has no probabilities to take an answer_confidence from", BACKEND_HINT
    )


def finalize(
    result: EngineResult, questions: dict[str, dict[str, Any]], *, backend: str, latency_ms: float
) -> dict[str, Any]:
    """Turn an engine result into the public, Jev-compatible response.

    Raises ModelUnavailable if the backend's answers or token counts are missing or malformed.
    """
    per_question = _usage_warnings(result.usage)
    answers: dict[str, dict[str, Any]] = {}
    local_warnings: list[str] = []
    for qid, question in questions.items():
        answer = _shape_answer(qid, question, result.answers.get(qid))
        warnings = list(per_question.get(qid, []))
        meta = result.shortlist.get(qid)
        if meta and not meta.get("passthrough", True):
            kept = list(meta["labels"])
            answer["shortlisted"] = {"kept": kept, "dropped": int(meta["n"]) - len(kept)}
            warnings.append(
                f"shortlisted: {meta['n']} options were narrowed to the {len(kept)} most similar to the input; "
                "probabilities cover only those"
            )
        answer["warnings"] = warnings
        local_warnings.extend(f"{qid}: {warning}" for warning in warnings)
        answers[qid] = answer
    usage = {
        "input_tokens": _token_count(result.usage, "input_tokens"),
        "output_tokens": _token_count(result.usage, "output_tokens"),
    }
    local: dict[str, Any] = {
        "backend": backend,
        "checkpoint": result.checkpoint,
        "routing_reason": result.routing_reason,
        "latency_ms": round(latency_ms, 1),
        "warnings": local_warnings,
    }
    extra = {key: value for key, value in result.usage.items() if key not in usage}
    if extra:
        local["usage"] = extra
    return {"model": result.model, "answers": answers, "usage": usage, "local": local}


def compact_answers(answers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Just the answer value and its confidence, for logs and batch results."""
    return {
        qid: {
            answer["type"]: answer[answer["type"]],
            "answer_confidence": answer["answer_confidence"],
        }
        for qid, answer in answers.items()
    }


def warning_code(warning: str) -> str:
    return warning.split(":", 1)[0]


def _usage_warnings(usage: dict[str, Any]) -> dict[str, list[str]]:
    warnings: dict[str, list[str]] = {}
    if usage.get("truncated"):
        message = (
            f"state_truncated: {usage.get('state_tokens_dropped', 0)} of {usage.get('state_tokens', 0)} "
            f"input tokens did not fit and were ignored; {TRUNCATION_HINT}"
        )
        for qid in usage.get("truncated_questions") or []:
            warnings.setdefault(qid, []).append(message)
    for qid, info in (usage.get("options") or {}).items():
        warnings.setdefault(qid, []).append(
            f"options_collapsed: only {info.get('distinct')} of {info.get('total')} options stayed distinct; "
            "shorten option descriptions or split the question"
        )
    return warnings


def _shape_answer(qid: str, question: dict[str, Any], raw: Any) -> dict[str, Any]:
    """One backend answer in its public form, or ModelUnavailable when it cannot be trusted."""
    if raw is None:
        raise ModelUnavailable(f"the backend returned no answer for question {qid!r}", BACKEND_HINT)
    if not isinstance(raw, dict):
        raise ModelUnavailable(
            f"the backend returned an answer for {qid!r} that is not an object", BACKEND_HINT
        )
    expected = question["type"]
    qtype = raw.get("type", expected)
    if not isinstance(qtype, str) or qtype not in JEV_FIELDS:
        raise ModelUnavailable(
            f"the backend returned an unknown answer type {qtype!r} for {qid!r}", BACKEND_HINT
        )
    if qtype != expected:
        raise ModelUnavailable(
            f"the backend returned a {qtype!r} answer for {qid!r}, which is a {expected!r} question",
            BACKEND_HINT,
        )
    missing = [
        key
        for key in REQUIRED_FIELDS[qtype]
        if key not in raw or (key == "probabilities" and not raw[key])
    ]
    if missing:
        raise ModelUnavailable(
            f"the backend's answer for {qid!r} is missing {', '.join(map(repr, missing))}",
            BACKEND_HINT,
        )
    try:
        answer: dict[str, Any] = {"type": qtype}
        answer.update({key: raw[key] for key in JEV_FIELDS[qtype] if key != "type" and key in raw})
        for key in ("noul", "score"):
            if key in answer:
                answer[key] = _finite(answer[key])
        if "probabilities" in answer:
            answer["probabilities"] = {
                label: _finite(p) for label, p in answer["probabilities"].items()
            }
        answer["answer_confidence"] = _finite(answer_confidence({**raw, **answer}))
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        raise ModelUnavailable(
            f"the backend returned an answer for {qid!r} with values that are not finite numbers",
            BACKEND_HINT,
        ) from exc
    return answer


def _token_count(usage: dict[str, Any], key: str) -> int:
    """A token count from the backend's usage as an int; absent counts as 0.

    int() itself rejects NaN (ValueError) and the infinities (OverflowError).
    """
    try:
        return int(usage.get(key, 0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelUnavailable(
            f"the backend returned a usage.{key} that is not a finite number", BACKEND_HINT
        ) from exc


def _finite(value: Any) -> float:
    """float(value), rejecting NaN and the infinities."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{number} is not a finite number")
    return number
