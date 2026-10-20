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


def predict(base: str, item: dict[str, Any], ab_labels: bool) -> tuple[Any, str, dict[str, Any]]:
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
    return guess, body["local"]["checkpoint"], answer


def evaluate(
    base: str, items: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, list[int]]], list[dict[str, Any]]]:
    table: dict[str, dict[str, list[int]]] = {}
    misses: list[dict[str, Any]] = []
    for mode in ("native", "ab_labels"):
        stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for item in items:
            kind = item["question"]["type"]
            if mode == "ab_labels" and kind != "noul":
                continue
            guess, checkpoint, answer = predict(base, item, ab_labels=mode == "ab_labels")
            correct = int(guess == item["expected"])
            if not correct:
                misses.append({"mode": mode, "item": item, "guess": guess, "answer": answer})
            keys = [kind, f"{kind} · {checkpoint}"] + (
                ["noul · negations"] if item.get("negation") else []
            )
            for key in keys:
                stats[key][0] += correct
                stats[key][1] += 1
        table[mode] = dict(stats)
    return table, misses


def shown(item: dict[str, Any], value: Any) -> str:
    """An expected value or an answer in words: yes or no, a label, or a level and its name."""
    question = item["question"]
    if question["type"] == "noul":
        return "yes" if value else "no"
    if question["type"] == "choice":
        return str(value)
    return f"{value} ({str(question['criteria'][value]).split(':')[0]})"


def probability(miss: dict[str, Any]) -> str:
    """P(yes) for a yes/no item, else the probability of the answer the model gave."""
    answer = miss["answer"]
    if miss["item"]["question"]["type"] == "noul":
        return f"{answer['noul']:.4f}"
    return f"{answer['probabilities'][str(miss['guess'])]:.4f}"


def cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def render(table: dict[str, dict[str, list[int]]], misses: list[dict[str, Any]]) -> tuple[str, str]:
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
    lines += [
        "",
        "## Misses",
        "",
        "Probability is P(yes) for a yes/no item, and the probability of the given answer for a "
        "choice or a score.",
        "",
        "| Mode | Item | Text | Question | Expected | Answer | Probability |",
        "|---|---|---|---|---|---|---|",
    ]
    for miss in misses:
        item = miss["item"]
        name = item["id"] + (" (negation)" if item.get("negation") else "")
        row = [
            miss["mode"],
            name,
            item["state"],
            item["question"]["instructions"],
            shown(item, item["expected"]),
            shown(item, miss["guess"]),
            probability(miss),
        ]
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines) + "\n", winner


def main() -> int:
    items = [
        json.loads(line) for line in ITEMS.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    with temporary_server() as (base, _process):
        wait_until_ready(base)
        table, misses = evaluate(base, items)
    text, winner = render(table, misses)
    RESULTS.write_text(text, encoding="utf-8")
    print(text)
    print(f"winner: {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
