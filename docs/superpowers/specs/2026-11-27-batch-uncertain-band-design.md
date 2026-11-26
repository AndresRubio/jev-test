# An uncertainty band in decide_batch (2026-11-27)

## Intent

`decide_batch` scores every item in a file with the small local model. The agent that calls it is a far stronger judge, and far more expensive per item. Hightouch's post on Jev as an identity-resolution judge (2026-11-04) scores every pair with the cheap decision model and sends only the pairs whose confidence falls in a band to a strong model, and matches the strong model's accuracy at about a quarter of the cost. In local-jev the strong model is the calling agent. This change lets a batch call hand the agent only the items the model is unsure about, least confident first, so the agent re-judges those and trusts the rest.

The calibration section of `eval/results.md` (2026-11-25) shows that the model's gaps are concentrated in some confidence buckets, which is what makes a band on confidence a useful selector.

## Goals

- A batch call can return the items whose `answer_confidence` for one question lies inside a band, least confident first, capped by `top_k`.
- The summary still counts every item, and the response says how many items were in the band before the cap.
- The tool description tells the agent when to use it and that it is the cheap-model-plus-agent pattern.
- Same parameter on MCP, `POST /v1/batch` and `local-jev batch`.

## Non-goals and anti-goals

- No change to how any probability or `answer_confidence` is computed.
- No change under `eval/`.
- No threshold sweep or default band. The 0.3 to 0.9 band in the post was tuned for Jev on one task; the agent passes its own bounds.
- The band must not change the results file or the summary: the agent reads the uncertain items, and the file still holds every item.

## Constraints

- Tool descriptions are part of the interface, and `tests/test_api_mcp.py` pins their phrases.
- `api_mcp.py` holds no logic. Parsing and selection live in `batch.py`, as for `filter`.
- Tests run with `uv run pytest`, with no network and no weights.

## Approaches considered

1. A new filter op `between` on `filter`: `{"question": "relevant", "op": "between", "value": [0.3, 0.9]}`. No new parameter, but `filter` compares the answer value (P(true), the expected level, the label), not the confidence, so one op would silently compare a different quantity, and the order of the matches would have to flip with the op. For a choice question it makes no sense without also changing what is compared.
2. A generalized `filter` with an `on` key (`"answer"` or `"answer_confidence"`) and an `order` parameter. The most general shape, but the agent would compose three settings for the one case this change is for, and the description would grow for everyone.
3. A separate `uncertain` parameter that selects by confidence and sets the order, combinable with `filter`. Chosen: the pattern has a name, the description can point at it in one sentence, and the existing `filter` keeps its meaning.

## The change

### Parameter

`uncertain` (optional): `{"question": "<id>", "max_confidence": <number>, "min_confidence": <number, optional>}`.

- `question` must be one of the batch's questions, of any type.
- `max_confidence` is required, from 0 to 1. `min_confidence` defaults to 0 and must not exceed `max_confidence`. Both bounds are inclusive. Any other key is refused, as for `filter`.
- An item is in the band when its `answer_confidence` for that question lies between the bounds. For a yes/no question `answer_confidence` is max(P(true), 1 - P(true)), so it is never under 0.5, and a `max_confidence` of 0.9 keeps the items whose P(true) lies between 0.1 and 0.9. For a choice or score question it is the probability of the chosen label or level.

### Selection and order

`select_matches` applies `filter` and `uncertain` together: an item is a match when it passes the filter, if there is one, and lies in the band, if there is one. With `uncertain` the matches are ordered by that question's `answer_confidence`, least confident first, ties in input order. Without it the order is as today: most confident first by the filter question with a `filter`, input order without one. `top_k` caps the list after the sort.

### Response

A new field `matched`: the number of items that passed the filter and the band before the `top_k` cap. Without either it equals `count`. The agent reads it to know whether `top_k` cut the band. `count`, `summary`, `output_path` and `warnings` cover every item, as before.

### Interfaces

- MCP: `decide_batch` gains `uncertain: dict | None = None` and passes it in the body like `filter`. `BATCH_INTRO` gains a paragraph that says what the band keeps, that the matches come least confident first, that `summary` still counts every item, that `matched` says how many were in the band, and when to use it: the model answers every item cheaply and the agent re-judges only the uncertain ones. `tests/test_api_mcp.py` pins "least confident first", "judge" and "uncertain".
- REST: `POST /v1/batch` hands the body to the same service, so it gains the parameter with no code change. The README says so.
- CLI: `local-jev batch --uncertain '<json>'`, mirroring `--filter`.

### Tests

- `tests/test_batch.py`: the parser accepts a band, defaults `min_confidence`, and refuses an unknown question, a missing or non-numeric bound, a bound outside 0 to 1, a crossed band and a stray key; `select_matches` keeps only the band, orders least confident first, applies `filter` and `uncertain` together and caps at `top_k`; `matched` is counted before the cap.
- `tests/test_api_mcp.py`: the pinned phrases and the parameter reaching the service.
- `tests/test_service.py`: a batch over inline items with `uncertain` returns the uncertain items least confident first, a summary over every item and `matched`.
- `tests/test_cli.py`: `--uncertain` reaches the payload.

## Left to the builder

- Where `matched` is computed so that `select_matches` keeps one responsibility.
- The exact wording of the description paragraph and the README sentence.
