"""The hand-labelled items of eval/items.jsonl as training rows, written to eval/seed-training.jsonl.

Run `uv run python eval/seed_training.py` after changing the items. Each row has the shape
`local-jev export-training` writes, state, questions and gold, under the question id `q` that
eval/run_eval.py sends. Gold is the expected answer of the item, so the rows need no correcting
before training. There is no `confidence` field, since no model answered these items.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ITEMS = Path(__file__).with_name("items.jsonl")
SEED = Path(__file__).with_name("seed-training.jsonl")
QUESTION_ID = "q"


def seed_row(item: dict[str, Any]) -> dict[str, Any]:
    """The training row of one evaluation item: its state, its question under the id `q`, and
    its expected answer as gold."""
    return {
        "state": item["state"],
        "questions": {QUESTION_ID: item["question"]},
        "gold": {QUESTION_ID: item["expected"]},
    }


def seed_rows(items_path: Path) -> list[dict[str, Any]]:
    """One row per non-blank line of the items file, in file order."""
    lines = items_path.read_text(encoding="utf-8").splitlines()
    return [seed_row(json.loads(line)) for line in lines if line.strip()]


def render(rows: list[dict[str, Any]]) -> str:
    """The rows as JSON lines, one row per line, with a trailing newline."""
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main() -> int:
    rows = seed_rows(ITEMS)
    SEED.write_text(render(rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {SEED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
