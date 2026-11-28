"""eval/seed_training.py turns the hand-labelled evaluation items into training rows, and the
committed eval/seed-training.jsonl is what it generates."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[1] / "eval"

NOUL = {
    "id": "n01",
    "lang": "en",
    "negation": False,
    "state": "Please refund the duplicate charge on my March invoice.",
    "question": {"type": "noul", "instructions": "Does the customer ask for money back?"},
    "expected": True,
}
CHOICE = {
    "id": "c01",
    "lang": "en",
    "state": "I was billed twice for March, please refund one.",
    "question": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "invoices, payments, refunds", "other": "anything else"},
    },
    "expected": "billing",
}
SCORE = {
    "id": "s01",
    "lang": "en",
    "state": "Production is down for all customers since 10 minutes ago.",
    "question": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent: can wait a week", "soon: within a day", "urgent: right now"],
    },
    "expected": 2,
}


@pytest.fixture(scope="module")
def seed_training():
    spec = importlib.util.spec_from_file_location("seed_training", EVAL / "seed_training.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_noul_item_gives_a_row_with_its_expected_answer_as_gold(seed_training):
    assert seed_training.seed_row(NOUL) == {
        "state": "Please refund the duplicate charge on my March invoice.",
        "questions": {"q": NOUL["question"]},
        "gold": {"q": True},
    }


def test_a_choice_item_gives_a_row_with_its_label_as_gold(seed_training):
    assert seed_training.seed_row(CHOICE) == {
        "state": "I was billed twice for March, please refund one.",
        "questions": {"q": CHOICE["question"]},
        "gold": {"q": "billing"},
    }


def test_a_score_item_gives_a_row_with_its_level_as_gold(seed_training):
    assert seed_training.seed_row(SCORE) == {
        "state": "Production is down for all customers since 10 minutes ago.",
        "questions": {"q": SCORE["question"]},
        "gold": {"q": 2},
    }


def test_the_evaluation_items_give_sixty_distinct_rows(seed_training):
    rows = seed_training.rows_from_items(seed_training.ITEMS)
    assert len(rows) == 60
    keys = {(row["state"], json.dumps(row["questions"], sort_keys=True)) for row in rows}
    assert len(keys) == 60


def test_the_committed_seed_file_is_what_the_script_generates(seed_training):
    expected = seed_training.render(seed_training.rows_from_items(seed_training.ITEMS))
    assert (EVAL / "seed-training.jsonl").read_text(encoding="utf-8") == expected
