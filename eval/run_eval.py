"""Accuracy on eval/items.jsonl for both yes/no label modes; writes eval/results.md.

Run `uv run python eval/run_eval.py` after `local-jev pull`. It starts its own temporary server.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import httpx2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _server import temporary_server, wait_until_ready  # noqa: E402

ITEMS = Path(__file__).with_name("items.jsonl")
RESULTS = Path(__file__).with_name("results.md")
AB_LABELS = {"true": "A", "false": "B"}


def predict(base: str, item: dict[str, Any], ab_labels: bool) -> tuple[Any, str]:
    question = dict(item["question"])
    if ab_labels:
        question["labels"] = AB_LABELS
    response = httpx2.post(
        f"{base}/v1/systemone",
        json={"state": item["state"], "questions": {"q": question}},
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    answer = body["answers"]["q"]
    if question["type"] == "noul":
        guess: Any = answer["noul"] >= 0.5
    elif question["type"] == "choice":
        guess = answer["choice"]
    else:
        guess = int(max(answer["probabilities"], key=answer["probabilities"].get))
    return guess, body["local"]["checkpoint"]


def evaluate(base: str, items: list[dict[str, Any]]) -> dict[str, dict[str, list[int]]]:
    table: dict[str, dict[str, list[int]]] = {}
    for mode in ("native", "ab_labels"):
        stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for item in items:
            kind = item["question"]["type"]
            if mode == "ab_labels" and kind != "noul":
                continue
            guess, checkpoint = predict(base, item, ab_labels=mode == "ab_labels")
            correct = int(guess == item["expected"])
            keys = [kind, f"{kind} · {checkpoint}"] + (
                ["noul · negations"] if item.get("negation") else []
            )
            for key in keys:
                stats[key][0] += correct
                stats[key][1] += 1
        table[mode] = dict(stats)
    return table


def render(table: dict[str, dict[str, list[int]]]) -> tuple[str, str]:
    native, ab = table["native"]["noul"], table["ab_labels"]["noul"]
    winner = "ab_labels" if ab[0] > native[0] else "native"
    lines = [
        f"# Evaluation ({date.today().isoformat()})",
        "",
        "60 hand-labelled items (`eval/items.jsonl`), 12 of them in Spanish, one question per request.",
        "",
    ]
    for mode, stats in table.items():
        lines += [f"## noul_mode = {mode}", "", "| Slice | Correct | Accuracy |", "|---|---|---|"]
        for key in sorted(stats):
            ok, total = stats[key]
            lines.append(f"| {key} | {ok}/{total} | {ok / total:.0%} |")
        lines.append("")
    lines.append(
        f"The default `noul_mode` is `{winner}`, the mode with more correct yes/no answers: "
        f"native {native[0]}/{native[1]}, ab_labels {ab[0]}/{ab[1]}. A tie goes to `native`."
    )
    return "\n".join(lines) + "\n", winner


def main() -> int:
    items = [
        json.loads(line) for line in ITEMS.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    with temporary_server() as (base, _process):
        wait_until_ready(base)
        table = evaluate(base, items)
    text, winner = render(table)
    RESULTS.write_text(text, encoding="utf-8")
    print(text)
    print(f"winner: {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
