# Calibration in the evaluation (2026-11-24)

## Intent

The README says Laya's probabilities are over-confident, and it rests that on a few misses with a high probability. The evaluation should measure it: when the model gives an answer with a confidence of 0.9, how often is that answer right. The same measurement is the first step of the `local-jev calibrate` command that section 14 of the design spec leaves for later.

Source of the idea: Hightouch's post on Jev as an identity-resolution judge (2026-11-04), which found that Jev's errors cluster near 50% confidence while a competing decision model is over-confident, and built a cheap-model-plus-Opus hybrid on that.

## Goals

- `eval/results.md` gets a "Calibration" section per `noul_mode`: a reliability table by confidence bucket and two summary numbers.
- The maths lives in the package and has unit tests that need no server and no weights.
- The README and the build report cite the numbers once the evaluation has run.

## Non-goals

- Fitting temperatures or changing any probability the server returns.
- A threshold sweep. With 30 yes/no items it would overfit; it waits for more items.
- An escalation band in `decide_batch`. That is a separate design.

## Definitions

Every item contributes one record: the confidence in the answer the model gave (`answer_confidence`, which is max(P(yes), 1 - P(yes)) for a yes/no item and the probability of the chosen label or level otherwise) and whether that answer was right. One definition for the three question types, so the table can mix them.

- Reliability table: buckets of width 0.1 on the confidence, from the lowest confidence seen up to 1.0, the last bucket closed at 1.0. Each row shows the bucket, the number of items, the mean confidence, the accuracy and the gap (accuracy minus mean confidence). Empty buckets are left out.
- ECE, the expected calibration error: the sum over buckets of (items in the bucket / all items) times the absolute gap.
- Brier score: the mean of (confidence minus 1 if right, confidence minus 0 if wrong) squared.

## Approaches considered

1. Keep the maths inside `eval/run_eval.py`. Shortest, but the script is not importable from the tests without a path hack, and the future `calibrate` command could not reuse it.
2. A module `src/local_jev/calibration.py` with pure functions, used by the eval script and tested in `tests/test_calibration.py`. Chosen.
3. A third-party metrics library. The maths is twenty lines; a dependency buys nothing.

## The change

- `src/local_jev/calibration.py`: `Record(confidence, correct)`, `reliability(records, width=0.1) -> list[Bucket]`, `ece(records, width=0.1) -> float`, `brier(records) -> float`.
- `eval/run_eval.py`: `evaluate` also collects the records per mode; `render` writes the section after the accuracy tables and before the misses.
- `tests/test_calibration.py`: hand-worked cases, including an empty input and a confidence of exactly 1.0.
- `README.md` "What to expect" and `docs/report.md` "Known limits": one sentence each with the ECE and the mean confidence against the accuracy, filled in from a real run.

## Left to the builder

- The exact wording of the section and the column order.
- Rounding (four decimals, like the misses table).
