"""A deterministic stand-in for a real engine: no weights, no MLX."""

from __future__ import annotations

import threading
import time
from typing import Any

from local_jev.engines.base import EngineCapabilities, EngineResult


class FakeEngine:
    """Answers from keywords: noul is 0.9 when the state says "yes"; choice picks a mentioned label."""

    name = "fake"
    capabilities = EngineCapabilities(laya_guards=True)

    def __init__(
        self,
        *,
        usage_extra: dict[str, Any] | None = None,
        shortlist: dict[str, dict[str, Any]] | None = None,
        fail_with: BaseException | None = None,
        block: threading.Event | None = None,
        delay: float = 0.0,
    ) -> None:
        self.usage_extra = usage_extra or {}
        self.shortlist = shortlist or {}
        self.fail_with = fail_with
        self.block = block
        self.delay = delay
        self.calls: list[tuple[Any, dict[str, Any], str]] = []
        self.loaded = False
        self.unload_calls = 0
        self.prefetched: list[str] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def decide(self, state: Any, questions: dict[str, dict[str, Any]], model: str) -> EngineResult:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.block is not None:
                self.block.wait(5)
            if self.delay:
                time.sleep(self.delay)
            self.calls.append((state, questions, model))
            if self.fail_with is not None:
                raise self.fail_with
            self.loaded = True
            text = str(state).lower()
            answers = {qid: _answer(question, text) for qid, question in questions.items()}
            usage = {"input_tokens": 10 * len(questions), "output_tokens": 0, **self.usage_extra}
            return EngineResult(
                answers=answers,
                usage=usage,
                model="fake-model",
                checkpoint="english",
                routing_reason="fake routing",
                shortlist=dict(self.shortlist),
            )
        finally:
            with self._lock:
                self.active -= 1

    def count_tokens(self, text: str, model: str) -> int:
        return len(text.split())

    def prefetch(self, names: tuple[str, ...]) -> None:
        self.prefetched.extend(names)

    def is_loaded(self) -> bool:
        return self.loaded

    def unload(self) -> None:
        self.unload_calls += 1
        self.loaded = False

    def status(self) -> dict[str, Any]:
        return {
            "backend": "fake",
            "loaded": ["english"] if self.loaded else [],
            "ready": ["english"],
        }


def _answer(question: dict[str, Any], text: str) -> dict[str, Any]:
    action = {"act_probability": 1.0}
    if question["type"] == "noul":
        p = 0.9 if "yes" in text else 0.1
        confidence = max(p, 1 - p)
        return {
            "type": "noul",
            "noul": p,
            "confidence": confidence,
            "answer_confidence": confidence,
            "action": action,
        }
    if question["type"] == "choice":
        labels = list(question["criteria"])
        chosen = next((label for label in labels if label.lower() in text), labels[0])
        rest = round(0.2 / (len(labels) - 1), 4)
        probabilities = {label: (0.8 if label == chosen else rest) for label in labels}
        return {
            "type": "choice",
            "choice": chosen,
            "probabilities": probabilities,
            "confidence": 0.5,
            "answer_confidence": 0.8,
            "action": action,
        }
    levels = question["criteria"]
    top = len(levels) - 1
    rest = round(0.3 / top, 4)
    probabilities = {str(i): (0.7 if i == top else rest) for i in range(len(levels))}
    score = round(sum(i * p for i, p in enumerate(probabilities.values())), 4)
    return {
        "type": "score",
        "score": score,
        "legend": {str(i): level for i, level in enumerate(levels)},
        "probabilities": probabilities,
        "confidence": 0.4,
        "answer_confidence": 0.7,
        "action": action,
    }
