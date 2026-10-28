# local-jev

Local, open-weight typed decisions in the style of TypeSafe's Jev. It runs [Laya](https://github.com/NandhaKishorM/laya) on Apple Silicon through [laya-mlx](https://github.com/mizorewww/laya-mlx). Agents reach it over MCP, a Jev-compatible REST API (`POST /v1/systemone`) or a CLI, and with the default backend the text you send it stays on your Mac.

It answers three kinds of question about a piece of text or JSON, with probabilities and without generating text:

| Type | Answer |
|---|---|
| `noul` | P(yes) for a yes/no question |
| `choice` | The most likely option, with a probability for every option |
| `score` | The expected level on a rubric of 2 to 10 levels |

## Requirements

- A Mac with Apple Silicon, running macOS 14 or later.
- [uv](https://docs.astral.sh/uv/).
- About 1.5 GB of disk for the English and multilingual checkpoints, and about 2.5 GB of memory while both are loaded.

## Install

```bash
uv tool install --editable .
local-jev pull
```

`local-jev pull` downloads the English and multilingual weights (about 1.5 GB) and checks each weight file against its pinned size and SHA-256. `local-jev pull --all` also fetches the typed-decisions checkpoint. The weights come from `convaiinnovations/laya` at the revision pinned in `src/local_jev/config.py`, and the checksums are in `src/local_jev/checkpoints.py`. If you skip `pull`, `local-jev serve` downloads the English and multilingual checkpoints in the background when it starts, and a request that needs one gets a 503 until it is ready.

## Run

```bash
local-jev serve
```

`serve` runs in the foreground on http://127.0.0.1:7711. To use another port, set `port` in the config file or `LOCAL_JEV_PORT`, because `status`, `connect`, `decide` and `batch` take the port from there; `serve --port` moves only the server.

```bash
local-jev install-agent
```

`install-agent` runs local-jev at login instead, through a launchd agent, and also starts it now. It shows the plist it will write and asks before installing. The login agent and a `local-jev serve` in a terminal cannot both bind the port, and while the terminal server holds it, launchd restarts the agent every 10 seconds.

```bash
local-jev status
```

`status` prints what the running server is doing, or says that it is not running.

The server listens only on 127.0.0.1. It answers only requests addressed to 127.0.0.1, localhost or [::1] on its own port, and it refuses requests from web pages unless the page itself comes from 127.0.0.1, localhost or [::1] over plain http, on any port. Models load on first use and unload after 15 idle minutes.

## Connect an agent

```bash
local-jev connect
```

This prints `claude mcp add --transport http --scope user local-jev http://127.0.0.1:7711/mcp`. Add `--apply` to run it after a confirmation, or use `--harness codex` (also `cursor` or `json`) to print the configuration for other agents.

The MCP server has three tools:

- `decide(state, questions, model="auto")` answers questions about one state, in Jev's request format.
- `decide_batch(questions, path=None, items=None, format="auto", field=None, chunk_tokens=256, filter=None, top_k=20, model="auto")` asks the same questions, at most 8, about many items. The items come from a local file at an absolute `path` (`jsonl`, `csv`, `lines`, or `chunks` of a long text) or from an inline `items` list. It returns a summary per question, up to `top_k` matches and the path of a file with every item's answers, so the items never enter the agent's context. With a `filter`, the matches are the items that pass it, most confident first; without one, they are the first `top_k` items.
- `status()` reports what the server is doing and never loads a model.

Agents that only run shell commands can use the CLI: `local-jev decide --state "..." --questions-file q.json` and `local-jev batch --questions-file q.json --path items.jsonl`. Both send the request to the running server and print its JSON answer.

## REST API

`POST /v1/systemone` accepts TypeSafe's documented request and returns the same answer fields, so an existing Jev client works when pointed at `http://127.0.0.1:7711`.

Differences from Jev:

- Each answer also has `answer_confidence` (the probability of the reported answer) and `warnings`. An answer whose options were narrowed also has `shortlisted`.
- A `local` block reports the backend, the checkpoint, the routing reason, the latency, the warnings and Laya's token diagnostics.
- `model` is optional: `auto`, `english`, `multilingual` or `typed-decisions`. `jev-latest` means `auto`.
- Errors are `{"error": {"code", "message", "hint"}}`.

`POST /v1/batch` takes the same parameters as `decide_batch`, and `local-jev batch` calls it. `GET /v1/status` returns what `status()` returns, and `GET /health` answers `{"ok": true}` without loading a model.

## What to expect

[docs/benchmarks.md](docs/benchmarks.md) has the speed and memory figures for this Mac, an M1 Pro with 16 GB, and [eval/results.md](eval/results.md) has the accuracy on 60 hand-labelled items in English and Spanish.

On that Mac one short question takes about 24 ms end to end over HTTP, and 10 questions in one call about 110 ms. The first request, which loads the English checkpoint, takes under a second. The process's footprint, as `top` reports it, is about 2,400 MB with both checkpoints loaded and under 300 MB after the idle unload.

`decide_batch` gets through about 47 short items per second with one yes/no question, about 30 with two questions and about 8 when each item is about 430 tokens long. A batch has 300 seconds (`batch.timeout_seconds`) for all of its model calls, and the time it spends waiting behind other requests counts too, so split files of more than about 2,000 of those long items. A batch that runs out of time fails with a `timeout` error (a 504 over REST) that says how many items were done, and it writes no results file.

In the evaluation the model answered 22 of the 30 yes/no questions correctly with its own true/false labels and 24 with neutral A/B labels, so `ab_labels` is the default. It got 16 of the 20 choice questions right, and 7 of the 10 score questions when its most likely level counts as the answer.

Laya is a small encoder (322M to 421M parameters), and it has limits:

- It is good at simple classification and yes/no reading, and weak at nuanced judgement unless it is fine-tuned on your own decisions.
- Its probabilities are over-confident. Use them to rank answers, and remember that confidence is not permission.
- The English checkpoint keeps about 320 to 480 tokens of the state, depending on how long the question and its options are, and the multilingual one about 760 to 990. An answer carries a `state_truncated` warning when the state was cut.
- Negations in the text ("don't cancel") are a weak spot. Every negation item in the evaluation has its negation in the text, and the model got only 5 of those 10 right with true/false labels and 8 of 10 with A/B labels; some misses in both modes had a P(yes) above 0.9.
- Phrase questions positively; the tool descriptions ask agents to do the same.
- The three score misses were all "not urgent" items of the urgency rubric, rated one or two levels too urgent.
- Quality drops with more than about 20 options, so local-jev narrows a larger choice in elimination rounds. It splits the options into groups of at most 20, sends each group's winner on, and asks the final question over the finalists. That answer has `shortlisted: {"kept": [...], "dropped": n}` and a `shortlisted` warning, and its probabilities cover only the finalists. A round that cuts the state or merges options adds the same `state_truncated` or `options_collapsed` warning as an ordinary question.
- On the English checkpoint, treat the probabilities of a choice with 11 to 20 options as uncalibrated. The checkpoint ships a temperature of 0.1006 for that size. laya-mlx raises it to 0.5 and prints a `RuntimeWarning` when it loads the checkpoint, but 0.5 still doubles the logits. A narrowed choice is affected only when 11 or more finalists reach the final question, which takes more than 200 options.
- A strict xfail test (`test_an_upgrade_question_goes_to_sales` in `tests/integration/test_laya_real.py`) pins one known miss: the English checkpoint sends "I would like to upgrade to the premium plan. What does it cost?" to billing (0.6868) and gives sales 0.255.
- A catch-all option such as "other: anything else" is rarely chosen. In the Claude Code check below, 40 of 50 messages written for "other" went to a concrete department (a privacy-policy question and an address change to billing, a thank-you and a hiring question to technical), and a question about the price difference between two plans went to billing, like the upgrade case above.
- Text in a state can steer the answer (prompt injection), so never use local-jev as the only check on untrusted input.

Verified with Claude Code on 2026-10-20: a new headless session (`claude -p`) listed `decide`, `decide_batch` and `status` over MCP and called each one. `decide` sent the double-billing message to billing (0.9566), and `decide_batch` read a 200-line JSONL file of synthetic support messages, returned the five billing matches it was asked for and wrote its results file. It gave 150 of the 200 messages the department they were written for: all 50 billing, all 50 technical, 40 of 50 sales and 10 of 50 other.

## Configuration

`~/.config/local-jev/config.toml`, where every key is optional:

```toml
port = 7711
backend = "laya-mlx"            # or "remote": any Jev-compatible /v1/systemone URL
idle_unload_minutes = 15        # 0 = never unload
noul_mode = "ab_labels"         # or "native": the model's default true/false labels
log_inputs = false              # true also logs the inputs, for training later

[laya]
dtype = "float16"
prefetch = ["english", "multilingual"]

[batch]
allowed_roots = ["~"]           # folders decide_batch may read

[remote]
url = ""                        # e.g. "http://127.0.0.1:8009/v1/systemone" for a Kev server
api_key_env = ""                # the environment variable that holds the remote API key
```

The file also takes `log_dir`, `results_dir`, `request_timeout_seconds` (30) and `queue_limit` (64) at the top, `repo` and `revision` under `[laya]`, and `max_items` (5000), `max_file_bytes` (50 MB), `max_evaluations` (20000), `group_size` (32) and `timeout_seconds` (300) under `[batch]`. An unknown key or a bad value stops every command except `uninstall-agent` with `config error: ...`.

`LOCAL_JEV_CONFIG` points at another config file, and these environment variables override the file: `LOCAL_JEV_PORT`, `LOCAL_JEV_BACKEND`, `LOCAL_JEV_REMOTE_URL`, `LOCAL_JEV_IDLE_UNLOAD_MINUTES`, `LOCAL_JEV_NOUL_MODE`, `LOCAL_JEV_LOG_INPUTS`, `LOCAL_JEV_LOG_DIR` and `LOCAL_JEV_RESULTS_DIR`. launchd does not read your shell profile, so `install-agent` copies the `LOCAL_JEV_*` variables that are set, and `HF_HOME` and `HF_HUB_CACHE`, into the login agent, with paths made absolute. With `backend = "remote"` it also copies the variable that `remote.api_key_env` names, shows its value masked in the preview as it does the token's, and prints a warning when that variable is not set. Run `install-agent` again after you change one; it replaces the agent and restarts it.

Set `LOCAL_JEV_TOKEN` to require a bearer token on every request except `GET /health`. Set one on a Mac with other user accounts: every account can reach 127.0.0.1, so without a token another account can ask the server to read files in your home folder through `decide_batch` and get snippets back. The CLI sends the token, `install-agent` copies it into the plist (mode 600), and `connect` prints the `claude mcp add` command with `--header "Authorization: Bearer $LOCAL_JEV_TOKEN"` at the end. Whether your shell fills in the variable or `connect --apply` passes the token, `claude mcp add` gets the token on its command line, so it shows in the process list (`ps`) while that command runs.

## Files

The paths are the defaults; `log_dir` and `results_dir` move the logs and the result files.

| Path | What |
|---|---|
| `~/Library/Logs/local-jev/server.log` | Server log, rotated at 10 MB with 5 old files kept |
| `~/Library/Logs/local-jev/decisions.jsonl` | One line per decision, rotated the same way; metadata only unless `log_inputs` is on |
| `~/Library/Logs/local-jev/launchd.out.log` and `launchd.err.log` | Output of the login agent; the error log gets what the server writes outside `server.log`, such as a config error, and is not rotated |
| `~/Library/Caches/local-jev/results/` | `decide_batch` result files; the server deletes the ones older than 7 days when it starts |
| `~/Library/LaunchAgents/io.github.andresrubio.local-jev.plist` | The login agent, written by `install-agent` |
| `~/.cache/huggingface/hub/models--convaiinnovations--laya/` | Model weights; `HF_HOME` or `HF_HUB_CACHE` moves them |

## Uninstall

```bash
local-jev uninstall-agent
claude mcp remove --scope user local-jev
uv tool uninstall local-jev
```

`uninstall-agent` asks before it stops and removes the login agent. The logs, the result files and the weights stay; delete their folders from the table above to remove them too.

## Development

```bash
uv sync
uv run pytest                    # unit tests: no network, no weights
uv run pytest -m integration     # real weights (about 1.5 GB), downloaded on first run
uv run python eval/run_eval.py   # accuracy; writes eval/results.md
uv run python scripts/bench.py   # speed and memory; writes docs/benchmarks.md
```

[GitHub Actions](.github/workflows/ci.yml) runs `ruff check`, `ruff format --check` and the unit tests on every push to `main` and every pull request. It uses a macOS runner because mlx ships wheels for Apple Silicon only.

The evaluation and the benchmark each start a throwaway server on a free port. It ignores your config file, and its logs and results go to a temporary folder.

Design: [docs/superpowers/specs/2026-09-25-local-jev-design.md](docs/superpowers/specs/2026-09-25-local-jev-design.md).

## Learning MCP from this repo

[docs/mcp-study-guide.md](docs/mcp-study-guide.md) explains the Model Context Protocol through this code: the JSON-RPC messages a client sends to `/mcp`, where each part of the protocol lives in `src/`, the design choices behind the server, and a set of interview questions with answers. [tests/test_mcp_wire.py](tests/test_mcp_wire.py) sends those raw messages through the whole app, and [examples/minimal_mcp_server.py](examples/minimal_mcp_server.py) is a stdio server with a tool, resources and a prompt, the primitives local-jev does not use.

## Licence

Laya's code and weights are Apache-2.0, by Convai Innovations. laya-mlx is Apache-2.0.
