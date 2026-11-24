"""Calibration measures for the evaluation: a reliability table, ECE and the Brier score.

Each record pairs the confidence of one answer with whether that answer was right. The
confidence is the probability of the answer the model gave: max(P(yes), 1 - P(yes)) for a
yes/no item, and the probability of the chosen label or level otherwise. A calibrated server is
right about 90% of the time when it reports 0.9.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

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


def _buckets(records: Sequence[Record], width: float) -> list[Bucket]:
    """The non-empty buckets with exact, unrounded values."""
    _require(records)
    buckets = round(1 / width)
    grouped: dict[int, list[Record]] = {}
    for record in records:
        index = min(int((record.confidence + EDGE_TOLERANCE) / width), buckets - 1)
        grouped.setdefault(index, []).append(record)
    return [
        Bucket(
            low=index * width,
            high=index * width + width,
            items=len(members),
            mean_confidence=sum(r.confidence for r in members) / len(members),
            accuracy=sum(r.correct for r in members) / len(members),
        )
        for index, members in sorted(grouped.items())
    ]


def reliability(records: Sequence[Record], width: float = 0.1) -> list[Bucket]:
    """The non-empty buckets of `width` between 0 and 1, in ascending order, rounded to print."""
    return [
        replace(
            bucket,
            low=round(bucket.low, 6),
            high=round(bucket.high, 6),
            mean_confidence=round(bucket.mean_confidence, 4),
            accuracy=round(bucket.accuracy, 4),
        )
        for bucket in _buckets(records, width)
    ]


def ece(records: Sequence[Record], width: float = 0.1) -> float:
    """Expected calibration error: each bucket's absolute gap weighted by its share of records."""
    return sum(b.items / len(records) * abs(b.gap) for b in _buckets(records, width))


def brier(records: Sequence[Record]) -> float:
    """Mean squared distance between the confidence and the outcome (1 right, 0 wrong)."""
    _require(records)
    return sum((r.confidence - (1.0 if r.correct else 0.0)) ** 2 for r in records) / len(records)
