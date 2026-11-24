"""Calibration measures for the evaluation: a reliability table, ECE and the Brier score.

Each record pairs the confidence of one answer, the probability the server reported for the
answer it gave, with whether that answer was right. A calibrated server is right about 90% of
the time when it reports 0.9.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

EDGE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Record:
    """One answered question: how sure the server was and whether it was right."""

    confidence: float
    correct: bool


@dataclass(frozen=True)
class Bucket:
    """The records whose confidence falls in [low, high), or [low, 1.0] for the last bucket."""

    low: float
    high: float
    items: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Accuracy minus mean confidence: positive when the server is underconfident."""
        return self.accuracy - self.mean_confidence


def _require(records: Sequence[Record]) -> None:
    if not records:
        raise ValueError("no records")


def reliability(records: Sequence[Record], width: float = 0.1) -> list[Bucket]:
    """The non-empty buckets of `width` between 0 and 1, in ascending order."""
    _require(records)
    buckets = round(1 / width)
    grouped: dict[int, list[Record]] = {}
    for record in records:
        index = min(int((record.confidence + EDGE_TOLERANCE) / width), buckets - 1)
        grouped.setdefault(index, []).append(record)
    table = []
    for index in sorted(grouped):
        members = grouped[index]
        low = index * width
        table.append(
            Bucket(
                low=round(low, 6),
                high=round(low + width, 6),
                items=len(members),
                mean_confidence=round(sum(r.confidence for r in members) / len(members), 4),
                accuracy=round(sum(r.correct for r in members) / len(members), 4),
            )
        )
    return table


def ece(records: Sequence[Record], width: float = 0.1) -> float:
    """Expected calibration error: each bucket's absolute gap weighted by its share of records."""
    table = reliability(records, width)
    return sum(bucket.items / len(records) * abs(bucket.gap) for bucket in table)


def brier(records: Sequence[Record]) -> float:
    """Mean squared distance between the confidence and the outcome (1 right, 0 wrong)."""
    _require(records)
    return sum((r.confidence - (1.0 if r.correct else 0.0)) ** 2 for r in records) / len(records)
