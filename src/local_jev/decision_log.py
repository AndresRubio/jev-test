"""A rotating JSONL log with one line per decision: metadata only, unless log_inputs is on."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from local_jev.guard import compact_answers


class DecisionLog:
    def __init__(
        self,
        log_dir: Path,
        *,
        log_inputs: bool,
        max_bytes: int = 10 * 1024 * 1024,
        backups: int = 5,
    ) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / "decisions.jsonl"
        self._log_inputs = log_inputs
        self._handler = RotatingFileHandler(
            self.path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
        )
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        # A random name, not id(self): ids are reused once an unclosed log is garbage-collected,
        # and two logs sharing a logger would write each record into both files.
        self._logger = logging.getLogger(f"local_jev.decisions.{uuid.uuid4().hex}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.addHandler(self._handler)

    def record(
        self,
        *,
        interface: str,
        kind: str,
        response: dict[str, Any] | None,
        questions: dict[str, dict[str, Any]],
        latency_ms: float,
        state: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "interface": interface,
            "kind": kind,
            "latency_ms": round(latency_ms, 1),
            "questions": [
                {"id": qid, "type": question["type"]} for qid, question in questions.items()
            ],
        }
        if response is not None:
            local = response.get("local", {})
            entry.update(
                model=response.get("model"),
                backend=local.get("backend"),
                checkpoint=local.get("checkpoint"),
                answers=compact_answers(response["answers"]),
                input_tokens=response["usage"]["input_tokens"],
                warnings=local.get("warnings", []),
            )
        if extra:
            entry.update(extra)
        if self._log_inputs:
            entry["state"] = state
            entry["question_definitions"] = questions
        self._logger.info(json.dumps(entry, ensure_ascii=False, default=str))

    def close(self) -> None:
        self._logger.removeHandler(self._handler)
        self._handler.close()
