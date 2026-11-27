"""Turn decisions.jsonl into training rows: state, questions and gold, the format Laya, Clef and
Unsloth train on.

Gold holds the model's own answers, so a row is a draft to correct before training. The log keeps
only each answer's value and confidence, so a score's gold is its expected level rounded to the
nearest level, halfway up, and a noul's gold is true when P(yes) is at least one half."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROTATED = re.compile(r"^decisions\.jsonl\.(\d+)$")


@dataclass
class TrainingExport:
    rows: list[dict[str, Any]] = field(default_factory=list)
    without_inputs: int = 0  # decisions logged while log_inputs was off
    ignored: int = 0  # batch records and decisions that failed
    malformed: int = 0  # lines that are not JSON, such as one the server never finished writing


def training_row(entry: dict[str, Any]) -> dict[str, Any] | None:
    """The training row of one log entry, or None when the entry has no decision or no inputs."""
    if entry.get("kind") != "decide" or "answers" not in entry:
        return None
    if "state" not in entry or "question_definitions" not in entry:
        return None
    questions = entry["question_definitions"]
    gold = {qid: _gold(questions[qid]["type"], answer) for qid, answer in entry["answers"].items()}
    confidence = {qid: answer["answer_confidence"] for qid, answer in entry["answers"].items()}
    return {
        "state": entry["state"],
        "questions": questions,
        "gold": gold,
        "confidence": confidence,
    }


def _gold(kind: str, answer: dict[str, Any]) -> Any:
    if kind == "noul":
        return float(answer["noul"]) >= 0.5
    if kind == "score":
        # Not round(): it sends a half to the even level, so 0.5 would give 0 and 2.5 would give 2.
        return math.floor(float(answer["score"]) + 0.5)
    return answer["choice"]


def log_files(log_dir: Path) -> list[Path]:
    """decisions.jsonl and its rotated copies, oldest first."""
    if not log_dir.is_dir():
        return []
    rotated = []
    for path in log_dir.iterdir():
        match = ROTATED.match(path.name)
        if match:
            rotated.append((int(match.group(1)), path))
    files = [path for _, path in sorted(rotated, reverse=True)]
    current = log_dir / "decisions.jsonl"
    if current.is_file():
        files.append(current)
    return files


def export_training(log_dir: Path) -> TrainingExport:
    export = TrainingExport()
    for path in log_files(log_dir):
        with path.open(encoding="utf-8") as lines:
            for line in lines:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    export.malformed += 1
                    continue
                row = training_row(entry)
                if row is not None:
                    export.rows.append(row)
                elif entry.get("kind") == "decide" and "answers" in entry:
                    export.without_inputs += 1
                else:
                    export.ignored += 1
    return export
