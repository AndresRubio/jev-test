# Calibration in the evaluation: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `eval/results.md` reports how well Laya's confidence matches its accuracy, with the maths in the package and under test.

**Architecture:** A new module `src/local_jev/calibration.py` holds pure functions over `(confidence, correct)` records: a reliability table by bucket, the expected calibration error and the Brier score. `eval/run_eval.py` collects one record per prediction and renders a "Calibration" section. Nothing the server returns changes.

**Tech Stack:** Python 3.12, dataclasses, pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-24-eval-calibration-design.md`

## Global Constraints

- Repo prose is English and follows the humanizer rules (no dashes in prose, no bold labels, plain sentences).
- Unit tests run with `uv run pytest` and need no network and no weights.
- `ruff check` and `ruff format --check` pass at a line length of 100.
- Numbers in `results.md` use four decimals, like the misses table; accuracy is a percentage with no decimals, like the accuracy tables.

## Review Focus

- A confidence of exactly 1.0 must land in the last bucket, never in a bucket of its own (Task 1 test).
- A confidence on a bucket edge, such as 0.7, must land in the bucket that starts there, despite float division (Task 1 test).
- A choice with four options can give a confidence under 0.5; the table must start at the lowest bucket seen, not at 0.5 (Task 1 test).
- An empty record list must raise, not return a silent 0 (Task 1 test).
- In `ab_labels` mode only yes/no items are predicted, so its section covers 30 records, and the table must say how many (Task 2 test).

---

### Task 1: The calibration module

**Files:**
- Create: `src/local_jev/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Produces:
  - `Record(confidence: float, correct: bool)`, a frozen dataclass.
  - `Bucket(low: float, high: float, items: int, mean_confidence: float, accuracy: float)`, frozen, with a `gap` property equal to `accuracy - mean_confidence`.
  - `reliability(records: Sequence[Record], width: float = 0.1) -> list[Bucket]`: buckets of `width` from 0 to 1, the last one closed at 1.0, only the non-empty ones, in ascending order.
  - `ece(records: Sequence[Record], width: float = 0.1) -> float`: sum over buckets of `items / len(records) * abs(gap)`.
  - `brier(records: Sequence[Record]) -> float`: mean of `(confidence - (1.0 if correct else 0.0)) ** 2`.
  - `reliability`, `ece` and `brier` raise `ValueError("no records")` on an empty sequence.

- [ ] **Step 1: Write the failing tests**

```python
from local_jev.calibration import Bucket, Record, brier, ece, reliability


def test_reliability_groups_records_by_bucket_and_skips_empty_ones():
    records = [Record(0.95, True), Record(0.91, False), Record(0.55, True)]
    assert reliability(records) == [
        Bucket(low=0.5, high=0.6, items=1, mean_confidence=0.55, accuracy=1.0),
        Bucket(low=0.9, high=1.0, items=2, mean_confidence=0.93, accuracy=0.5),
    ]


def test_a_confidence_of_one_lands_in_the_last_bucket():
    assert reliability([Record(1.0, True)]) == [Bucket(0.9, 1.0, 1, 1.0, 1.0)]


def test_a_confidence_on_a_bucket_edge_starts_that_bucket():
    assert reliability([Record(0.7, True)])[0].low == 0.7


def test_the_table_starts_at_the_lowest_confidence_seen():
    assert reliability([Record(0.35, False)])[0] == Bucket(0.3, 0.4, 1, 0.35, 0.0)


def test_gap_is_accuracy_minus_mean_confidence():
    assert Bucket(0.9, 1.0, 2, 0.93, 0.5).gap == pytest.approx(-0.43)


def test_ece_weights_each_bucket_gap_by_its_share_of_the_records():
    records = [Record(0.95, True), Record(0.91, False), Record(0.55, True)]
    # bucket 0.5: 1/3 * |1.0 - 0.55|; bucket 0.9: 2/3 * |0.5 - 0.93|
    assert ece(records) == pytest.approx(1 / 3 * 0.45 + 2 / 3 * 0.43)


def test_brier_is_the_mean_squared_error_against_a_right_answer():
    assert brier([Record(0.8, True), Record(0.6, False)]) == pytest.approx((0.04 + 0.36) / 2)


@pytest.mark.parametrize("function", [reliability, ece, brier])
def test_an_empty_record_list_raises(function):
    with pytest.raises(ValueError, match="no records"):
        function([])
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_calibration.py -v`
Expected: FAIL at import, `No module named 'local_jev.calibration'`.

- [ ] **Step 3: Implement `src/local_jev/calibration.py`**

Module docstring: how the eval measures calibration, with the one-line definition of confidence from the spec. Bucket index: `min(int((confidence + 1e-9) / width), buckets - 1)` with `buckets = round(1 / width)`, so an edge value starts its bucket and 1.0 stays in the last one. `low = index * width` and `high = low + width`, both rounded to 6 decimals so the dataclass compares equal to literals. `mean_confidence` and `accuracy` are rounded to 4 decimals.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_calibration.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src tests && uv run ruff format --check src tests
git add src/local_jev/calibration.py tests/test_calibration.py
git commit -m "feat: reliability table, ECE and Brier score for the evaluation"
```

### Task 2: The eval script renders a Calibration section

**Files:**
- Modify: `eval/run_eval.py` (`evaluate`, `render`, `main`)
- Test: `tests/test_eval_render.py`

**Interfaces:**
- Consumes: `Record`, `reliability`, `ece`, `brier` from Task 1.
- Produces: `evaluate(base, items) -> tuple[table, misses, records]` where `records: dict[str, list[Record]]` keyed by mode; `render(table, misses, records) -> tuple[str, str]`.

- [ ] **Step 1: Write the failing test**

Load the script the way `tests/test_examples.py` loads the examples (`importlib.util.spec_from_file_location` on `ROOT / "eval" / "run_eval.py"`). Build a `table` with one `noul` slice per mode, an empty `misses` list and records `{"native": [Record(0.95, True), Record(0.55, False)], "ab_labels": [Record(0.7, True)]}`.

```python
def test_render_writes_a_calibration_section_per_mode(run_eval):
    text, _winner = run_eval.render(table, [], records)
    calibration = text.split("## Calibration")[1].split("## Misses")[0]
    assert "### noul_mode = native (2 items)" in calibration
    assert "### noul_mode = ab_labels (1 item)" in calibration
    assert "| 0.9 to 1.0 | 1 | 0.9500 | 100% | +0.0500 |" in calibration
    assert "| 0.5 to 0.6 | 1 | 0.5500 | 0% | -0.5500 |" in calibration
    assert "ECE 0.3000, Brier 0.1525." in calibration
```

Check the two numbers by hand before writing them down: ECE is 0.5 × 0.05 + 0.5 × 0.55 and Brier is (0.0025 + 0.3025) / 2.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_eval_render.py -v`
Expected: FAIL, `render() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Change `eval/run_eval.py`**

`evaluate` appends `Record(answer["answer_confidence"], bool(correct))` to `records[mode]` for every prediction and returns it as a third value. `render` takes `records` and, after the accuracy tables and the `noul_mode` sentence, writes `## Calibration`, a paragraph defining confidence, ECE and Brier (copy the spec's definitions), then per mode a `### noul_mode = <mode> (<n> items)` heading, a table with the columns `Confidence | Items | Mean confidence | Accuracy | Gap` where the first column reads `0.9 to 1.0` and the gap carries its sign, and a line `ECE <ece:.4f>, Brier <brier:.4f>.`. `main` passes `records` through. Update the module docstring to mention calibration.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_eval_render.py tests/test_calibration.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check eval tests && uv run ruff format --check eval tests
git add eval/run_eval.py tests/test_eval_render.py
git commit -m "feat: the evaluation reports calibration per noul_mode"
```

### Task 3: Run the evaluation and cite the numbers

**Files:**
- Modify: `eval/results.md` (regenerated), `README.md` "What to expect", `docs/report.md` "Known limits of the model"

**Interfaces:**
- Consumes: the script from Task 2.

- [ ] **Step 1: Make sure the weights are present**

Run: `uv run local-jev pull`
Expected: the English and multilingual checkpoints download (about 1.5 GB) or are reported as already present. This step needs the owner's go-ahead for the download.

- [ ] **Step 2: Run the evaluation**

Run: `uv run python eval/run_eval.py`
Expected: it prints the report and rewrites `eval/results.md` with a `## Calibration` section. Check that the accuracy tables still match the previous run (native 22/30, ab_labels 24/30 on noul) and note any drift in the commit message.

- [ ] **Step 3: Cite the numbers**

In `README.md`, replace the bullet "Its probabilities are over-confident. Use them to rank answers, and remember that confidence is not permission." with one that gives the native-mode mean confidence against the accuracy and the ECE, and keeps the sentence about ranking and permission. In `docs/report.md`, give the same numbers in the first bullet of "Known limits of the model". Spell out ECE on first use in each file.

- [ ] **Step 4: Run the whole unit suite and the docs check**

Run: `uv run pytest -q`
Expected: all PASS (`tests/test_docs.py` checks what the docs cite).

- [ ] **Step 5: Commit**

```bash
git add eval/results.md README.md docs/report.md
git commit -m "docs: measure Laya's calibration in the evaluation"
```
