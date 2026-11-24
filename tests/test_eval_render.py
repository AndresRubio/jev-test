"""The calibration section of eval/results.md, rendered from synthetic data."""

import importlib.util
import sys
from pathlib import Path

import pytest

from local_jev.calibration import Record

EVAL = Path(__file__).resolve().parents[1] / "eval"


@pytest.fixture(scope="module")
def run_eval():
    spec = importlib.util.spec_from_file_location("run_eval", EVAL / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_writes_a_calibration_section_per_mode(run_eval):
    table = {"native": {"noul": [1, 2]}, "ab_labels": {"noul": [1, 1]}}
    records = {
        "native": [Record(0.95, True), Record(0.55, False)],
        "ab_labels": [Record(0.7, True)],
    }
    text, _winner = run_eval.render(table, [], records)
    assert text.index("The default `noul_mode`") < text.index("## Calibration")
    assert text.index("## Calibration") < text.index("## Misses")
    calibration = text.split("## Calibration")[1].split("## Misses")[0]
    native, ab_labels = calibration.split("### noul_mode = ab_labels")
    assert "### noul_mode = native (2 items)" in native
    assert "| 0.9 to 1.0 | 1 | 0.9500 | 100% | +0.0500 |" in native
    assert "| 0.5 to 0.6 | 1 | 0.5500 | 0% | -0.5500 |" in native
    assert "Mean confidence 0.7500, accuracy 50%, ECE 0.3000, Brier 0.1525." in native
    assert ab_labels.startswith(" (1 item)")
    assert "| 0.7 to 0.8 | 1 | 0.7000 | 100% | +0.3000 |" in ab_labels
    assert "Mean confidence 0.7000, accuracy 100%, ECE 0.3000, Brier 0.0900." in ab_labels


@pytest.mark.parametrize(
    ("value", "shown"), [(0.9, "0.9"), (1.0, "1.0"), (0.75, "0.75"), (0.0, "0.0")]
)
def test_a_bucket_edge_keeps_the_decimals_it_needs(run_eval, value, shown):
    assert run_eval.edge(value) == shown
