"""Latency, load time, memory and batch throughput of a temporary local-jev server.

Run `uv run python scripts/bench.py` after `local-jev pull`. It writes docs/benchmarks.md.
"""

from __future__ import annotations

import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path

import httpx2
from _server import temporary_server, wait_until_ready

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "benchmarks.md"
ENGLISH = "I was billed twice for March, please refund one."
SPANISH = "Me han cobrado dos veces este mes, quiero que me devuelvan uno."
ONE = {"q": {"type": "noul", "instructions": "Does the customer ask for money back?"}}
TEN = {
    f"q{i}": {"type": "noul", "instructions": f"Does sentence {i} mention money?"}
    for i in range(10)
}
UNITS_MB = {"B": 1 / 1024 / 1024, "K": 1 / 1024, "M": 1.0, "G": 1024.0}
SHORT_MESSAGES = (
    "I was billed twice for March, please refund one.",
    "The export button returns a 500 error since yesterday.",
    "Do you offer a discount for 50 seats on the annual plan?",
    "Please send me the signed contract by Friday.",
    "Thanks, it works great now!",
)
BATCH_ITEMS = [f"{SHORT_MESSAGES[i % len(SHORT_MESSAGES)]} (ticket {i})" for i in range(200)]
TWO = {**ONE, "q2": {"type": "noul", "instructions": "Does the customer report a bug?"}}
LONG_TEXT = (
    "The customer wrote a long message about the invoice, the delivery and the account settings. "
)
LONG_ITEMS = [f"Ticket {i}. " + LONG_TEXT * 25 for i in range(200)]


def timed(base: str, state: str, questions: dict) -> float:
    start = time.perf_counter()
    response = httpx2.post(
        f"{base}/v1/systemone", json={"state": state, "questions": questions}, timeout=120
    )
    elapsed = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    return elapsed


def batch_rate(base: str, items: list[str], questions: dict) -> float:
    """Items per second for one POST /v1/batch of `items` with `questions`."""
    start = time.perf_counter()
    response = httpx2.post(
        f"{base}/v1/batch", json={"questions": questions, "items": items}, timeout=600
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    return len(items) / elapsed


def state_tokens(base: str, state: str) -> int:
    """The length of `state` in tokens, as the checkpoint that answers it counts them."""
    response = httpx2.post(
        f"{base}/v1/systemone", json={"state": state, "questions": ONE}, timeout=120
    )
    response.raise_for_status()
    return response.json()["local"]["usage"]["state_tokens"]


def footprint_mb(pid: int) -> float:
    """Physical footprint as `top` reports it; this includes the GPU buffers MLX allocated."""
    out = subprocess.check_output(["top", "-l", "1", "-pid", str(pid), "-stats", "mem"], text=True)
    value = out.strip().splitlines()[-1].strip().rstrip("+-")
    number = float(re.match(r"[\d.]+", value).group())
    return number * UNITS_MB.get(value[-1].upper(), 1.0)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def machine() -> str:
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    memory_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 2**30
    return f"{chip}, {memory_gb:.0f} GB, macOS {platform.mac_ver()[0]}"


def main() -> int:
    with temporary_server(LOCAL_JEV_IDLE_UNLOAD_MINUTES="0.1") as (base, process):
        wait_until_ready(base)
        cold = timed(base, ENGLISH, ONE)
        one = [timed(base, ENGLISH, ONE) for _ in range(100)]
        ten = [timed(base, ENGLISH, TEN) for _ in range(50)]
        english_mb = footprint_mb(process.pid)
        timed(base, SPANISH, ONE)
        both_mb = footprint_mb(process.pid)
        time.sleep(15)  # idle unload after 6 s; the janitor checks every 3 s
        unloaded = httpx2.get(f"{base}/v1/status", timeout=5).json()["engine"]["loaded"] == []
        idle_mb = footprint_mb(process.pid)
        reload_ms = timed(base, ENGLISH, ONE)
        # The batches run last, so they cannot move the figures above.
        short_one = [batch_rate(base, BATCH_ITEMS, ONE) for _ in range(3)]
        short_two = [batch_rate(base, BATCH_ITEMS, TWO) for _ in range(3)]
        long_tokens = state_tokens(base, LONG_ITEMS[0])
        long_one = [batch_rate(base, LONG_ITEMS, ONE) for _ in range(3)]
    one_median, ten_median = statistics.median(one), statistics.median(ten)
    rows = [
        ("One short question, median", f"{one_median:.1f} ms", "≤ 100 ms", one_median <= 100),
        ("One short question, 95th percentile", f"{percentile(one, 0.95):.1f} ms", "", None),
        ("10 questions in one call, median", f"{ten_median:.1f} ms", "≤ 400 ms", ten_median <= 400),
        ("First request (loads English)", f"{cold:.0f} ms", "≤ 10,000 ms", cold <= 10_000),
        (
            "First request after idle unload",
            f"{reload_ms:.0f} ms",
            "≤ 10,000 ms",
            reload_ms <= 10_000,
        ),
        ("Memory with English loaded", f"{english_mb:.0f} MB", "", None),
        ("Memory with both checkpoints loaded", f"{both_mb:.0f} MB", "≤ 2,500 MB", both_mb <= 2500),
        ("Memory after idle unload", f"{idle_mb:.0f} MB", "≤ 400 MB", idle_mb <= 400),
        ("Models unloaded after idle", str(unloaded), "True", unloaded),
        (
            "Batch of 200 short items, one yes/no question, median of 3",
            f"{statistics.median(short_one):.0f} items/s",
            "",
            None,
        ),
        (
            "Batch of 200 short items, two yes/no questions, median of 3",
            f"{statistics.median(short_two):.0f} items/s",
            "",
            None,
        ),
        (
            f"Batch of 200 items of about {long_tokens} tokens, one yes/no question, median of 3",
            f"{statistics.median(long_one):.0f} items/s",
            "",
            None,
        ),
    ]
    lines = [
        f"# Benchmarks ({date.today().isoformat()})",
        "",
        f"Machine: {machine()}. laya-mlx {version('laya-mlx')}, local-jev {version('local-jev')}.",
        "Latency is end to end over HTTP from a local client. Memory is the process's physical footprint "
        "from `top`, which counts a MB as 1,048,576 bytes.",
        "",
        "| Measure | Result | Target | Pass |",
        "|---|---|---|---|",
    ]
    for name, result, target, ok in rows:
        mark = "" if ok is None else ("yes" if ok else "no")
        lines.append(f"| {name} | {result} | {target} | {mark} |")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all(ok is not False for *_, ok in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
