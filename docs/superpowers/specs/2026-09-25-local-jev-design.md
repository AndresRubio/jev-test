# local-jev: design spec

- **Date:** 2026-09-25
- **Status:** Approved 2026-10-14, amended the same day for laya-mlx 0.4.0 and MCP SDK 2.3.0 (§15)
- **Owner:** Andres Rubio

## 1. Summary

`local-jev` is a local decision service for this Mac. It answers typed questions in the style of TypeSafe's Jev: yes/no (`noul`), `choice` and `score`, each with probabilities. It uses open-weight models, Laya by default, and agents reach it in three ways:

- **MCP**, for Claude Code or any other agent harness. This is the primary interface.
- **A Jev-compatible REST API** (`POST /v1/systemone`), so an existing Jev client works by changing its base URL.
- **A command-line tool**, for agents that can only run shell commands.

It runs as one background process on `127.0.0.1`. It loads the model once, serves every client from that copy, and unloads the model when idle.

## 2. Background

- **Jev is proprietary.** It is available only through TypeSafe's API: about 100 ms per call, $0.042 per million input tokens, and inputs leave the machine. The user wants a local-first, open-weight equivalent.
- **Laya fits the role.** It comes from Convai Innovations, with Apache-2.0 code and weights, and uses the same question types as Jev.
  - It is an encoder: it reads the question and the input together in one forward pass and scores each option. It generates no text and produces 0 output tokens.
  - There are three checkpoints:

    | Checkpoint | Encoder | Parameters | Context |
    |---|---|---|---|
    | English (`convaiinnovations/laya`) | ModernBERT-large | 421M | 512 tokens |
    | Multilingual (`convaiinnovations/laya`, subfolder `multilingual`) | mmBERT-base | 322M | 1,024 tokens, up to 8,192 |
    | Typed decisions (subfolder `typed-decisions`) | ModernBERT-large | 421M | 1,024 tokens |

- **laya-mlx is a native Apple Silicon port.** It is an independent port to MLX, Apache-2.0. Version 0.4.0, released 2026-10-10, natively reports state truncation and collapsed options, returns `answer_confidence`, supports neutral `labels` on yes/no questions, and frees MLX memory on unload. Its own M3 Max measurements of version 0.2.0, 16-bit, for one short question:
  - English: 17.8 ms median, with a peak of 944 MiB.
  - Multilingual: 10.9 ms median, with a peak of 688 MiB.
  - Its answers match upstream Laya on 63 of 63 validation questions.
- **Laya's own README lists known limits**, and this design must respect them:
  - **Untrained accuracy is near chance on agent-style decisions:** 0.36, against 0.32 for random guessing and 0.46 for always picking the most common answer. After fine-tuning it reaches 0.77.
  - **Simple classification is strong:** AG News news categories 0.95; BoolQ yes/no reading questions 0.83.
  - Probabilities are over-confident until calibrated.
  - Yes/no answers can follow the `false`/`true` label text instead of the input.
  - Negated requests are sometimes misread with high confidence.
  - Quality drops sharply with more than about 20 options.
  - Input that exceeds the context limit is cut off silently.
- **The usage pattern favours batch triage.** In this user's sessions, most token cost comes from re-reading context. A tool that triages many items from a file, without the items entering the agent's context, saves more than single decisions do.

## 3. Goals and non-goals

**Goals (v1)**

- **G1.** Claude Code and other MCP clients connect over MCP Streamable HTTP and call `decide`, `decide_batch` and `status`.
- **G2.** `POST /v1/systemone` accepts TypeSafe's documented request format and returns a response that is a superset of Jev's.
- **G3.** A `local-jev` CLI covers serving, deciding, batching, status, connecting harnesses and the login agent.
- **G4.** A guard layer handles Laya's documented pitfalls for every caller and reports problems as warnings.
- **G5.** It is fast: at most 100 ms median for one short question on the M1 Pro (16 GB) once loaded.
- **G6.** It is private by default: loopback only, no outbound network calls except model downloads, and metadata-only logging.
- **G7.** Idle models are unloaded to free memory.

**Non-goals (v1)**

- Claude Code hooks, such as a command-safety gate or a large-output relevance filter.
- Calibration (temperature fitting), fine-tuning, and training classification heads.
- Running Kev in the same process. It is reachable through the remote backend instead.
- The Apple Neural Engine, a GUI, multi-user or remote access, and Linux or Windows.

## 4. Architecture

```text
  Claude Code · Codex · Cursor · Hermes · any MCP client        Jev clients / SDKs        shell-only agents
                  │  MCP (Streamable HTTP, /mcp)                        │ POST /v1/systemone          │ local-jev CLI
                  ▼                                                     ▼                              ▼ (HTTP)
  ┌───────────────────────────── local-jev serve  (one process, 127.0.0.1:7711) ─────────────────────────────┐
  │  uvicorn (ASGI) → Starlette app                                                                          │
  │     security middleware: Host/Origin allowlist (loopback) · optional bearer token                        │
  │     /mcp  (MCP Python SDK v2, MCPServer.streamable_http_app)   /v1/systemone  /v1/batch  /v1/status  /health │
  │                                   │                                                                      │
  │                           guard layer (validate · rewrite · warn · answer_confidence)                    │
  │                                   │                                                                      │
  │                           runtime: single inference worker · bounded queue · timeouts · idle unload      │
  │                                   │                                                                      │
  │                 Engine interface ─┴─ LayaMLXEngine (default)  |  RemoteJevEngine (opt-in, any Jev URL)   │
  │                                         laya_mlx.Router → MLX on the Metal GPU, 16-bit                   │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Inference runtime

There is no separate inference server. `LayaMLXEngine` runs Laya in-process through `laya_mlx`, and MLX executes on the Metal GPU. uvicorn only handles HTTP.

The alternatives were rejected for these reasons:

| Option | Why not |
|---|---|
| Ollama, llama.cpp, LM Studio | They run generative GGUF language models. Laya is an encoder with a custom decision head, so they cannot run it. |
| vLLM, Hugging Face TEI | They support standard classifier and pooling heads, not Laya's decision head. They are also impractical on macOS. |
| Upstream `laya-serve` | It is Jev-compatible, but runs on PyTorch/MPS: heavier, and about 25% slower than MLX on laya-mlx's own M3 Max measurements. It also has no guard layer. It can still be used through `RemoteJevEngine`. |
| Kev's `kev.serve` | It suits Kev models and can also be used through `RemoteJevEngine`. |

Inference is serialized on one worker thread. This is safe with MLX, avoids GPU contention, and is fast enough because each request takes tens of milliseconds.

## 5. Components

Each unit has one purpose and a narrow interface. The package is `local_jev` in a `src/` layout.

| Module | Purpose | Depends on |
|---|---|---|
| `errors.py` | `ServiceError` (HTTP status, code, message, hint) and the engine and runtime exceptions that map onto it. | none |
| `config.py` | Load `~/.config/local-jev/config.toml`, apply `LOCAL_JEV_*` environment overrides, and supply defaults (§12). | `errors` |
| `schema.py` | Jev request validation (plain Python, no pydantic), with errors that include how to fix the request. | `errors` |
| `checkpoints.py` | Pinned checkpoint table, download through `huggingface_hub`, and SHA-256 verification. | `huggingface-hub` |
| `service.py` | `DecisionService`, shared by REST and MCP: parse, guard, run, finalize, log, batch orchestration, status. | everything below `api_*` |
| `guard.py` | Changes applied before and after inference (§7). It is engine-agnostic, except for flags an engine declares. | `schema`, engine capabilities |
| `engines/base.py` | The `Engine` protocol: `decide(state, questions, model) -> dict`, `embed(texts)`, `load()`, `unload()`, `status()`, `capabilities`. | none |
| `engines/laya_mlx.py` | Wraps `laya_mlx.Router` (§8). Also handles truncation measurement, shortlisting, and retrying in 32-bit when outputs are not finite. | `laya-mlx` |
| `engines/remote.py` | Forwards to a Jev-compatible URL, with an optional bearer token taken from an environment variable. | `httpx` |
| `runtime.py` | The inference worker thread, a bounded queue, per-request timeouts, the idle-unload timer, and a latency histogram. | an engine |
| `batch.py` | Item readers (inline, JSONL, CSV, lines, chunks), filtering, top-k selection, summaries, and the JSONL result writer. | `runtime`, `guard` |
| `security.py` | Host/Origin allowlist middleware, the optional bearer token, and the file-access policy. | `config` |
| `api_rest.py` | `/v1/systemone`, `/v1/batch`, `/v1/status`, `/health`. | `runtime`, `batch` |
| `api_mcp.py` | `MCPServer` with the `decide`, `decide_batch` and `status` tools, and their descriptions (§6.2). | `runtime`, `batch` |
| `app.py` | Assembles the Starlette app: middleware, REST routes, MCP mount, and a lifespan that runs `mcp.session_manager.run()`. | all of the above |
| `decision_log.py` | The decision log in JSONL, with size-based rotation. | `config` |
| `cli.py` | The `local-jev` command (§6.3). | `httpx`, `app`, `launchd` |
| `launchd.py` | Renders, installs and removes the login agent plist. | none |

## 6. Interfaces

### 6.1 REST

**`POST /v1/systemone`** is compatible with Jev.

The request:

```json
{
  "model": "auto",
  "state": "string | object | array",
  "questions": {
    "<id>": {"type": "noul",   "instructions": "...", "criteria": {"true": "...", "false": "..."}},
    "<id>": {"type": "choice", "instructions": "...", "criteria": {"billing": "refunds", "tech": null}},
    "<id>": {"type": "score",  "instructions": "...", "criteria": ["not urgent", "soon", "blocking"]}
  }
}
```

- **`model`** is optional; Jev requires it. Allowed values:
  - `auto`, or Jev's `jev-latest`, routes by language.
  - `english`, `multilingual` or `typed-decisions` selects a checkpoint.
  - Any other value is passed through unchanged by the remote backend.
- **Limits:** at most 50 questions per request and at most 200,000 characters of serialized state. Anything the model window cannot hold is truncated and reported with a warning.

The response is a superset of Jev's.

- **The first three fields match Jev exactly:**
  - `model` is the Laya checkpoint that answered, for example `laya-english`.
  - `answers` holds one entry per question, each with Jev's fields.
  - `usage` reports `input_tokens`, and `output_tokens` is always 0.
- **Local additions inside each answer:**
  - `answer_confidence`: the probability of the reported answer.
  - `warnings`: a list of strings, each starting with its code, for example `"state_truncated: 412 of 900 input tokens did not fit and were ignored; …"`. Empty when there are none.
  - `shortlisted`: present only when the options were narrowed.
- **Fields not passed through:** Laya's `action` block, because its `act_probability` carries no usable signal. `usage` holds only Jev's two fields.
- **The top-level `local` block** holds `backend`, `checkpoint`, `routing_reason`, `latency_ms` and `warnings`, each warning prefixed with its question id. It also holds `usage`, with Laya's extra diagnostics (`state_tokens`, `state_tokens_dropped`, `truncated`, `truncated_questions`, `options`), when present.

```json
{
  "model": "laya-english",
  "answers": {
    "dept":   {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.91, "tech": 0.09},
               "confidence": 0.56, "answer_confidence": 0.91, "warnings": []},
    "refund": {"type": "noul", "noul": 0.88, "confidence": 0.88, "answer_confidence": 0.88, "warnings": []},
    "urgency": {"type": "score", "score": 1.7, "legend": {"0": "not urgent", "1": "soon", "2": "blocking"},
                "probabilities": {"0": 0.05, "1": 0.2, "2": 0.75}, "confidence": 0.4,
                "answer_confidence": 0.75, "warnings": []}
  },
  "usage": {"input_tokens": 212, "output_tokens": 0},
  "local": {"backend": "laya-mlx", "checkpoint": "english", "latency_ms": 41.2, "warnings": []}
}
```

**`POST /v1/batch`** is a local extension that the CLI uses. It takes the same parameters and returns the same result as the MCP `decide_batch` tool (§6.2).

**`GET /v1/status`** returns the same data as the MCP `status` tool. **`GET /health`** returns `{"ok": true}` without loading a model.

Errors are returned as `{"error": {"code", "message", "hint"}}`:

| Status | Code | When |
|---|---|---|
| 401 | `unauthorized` | A token is configured and the request's token is missing or wrong. |
| 403 | `forbidden_origin` | The Origin header is not on the loopback allowlist. |
| 421 | `misdirected` | The Host header is not a loopback host. |
| 413 | `too_large` | The body exceeds 5 MB, or a limit in §6.1 or §6.2 is exceeded. |
| 422 | `invalid_request` | Validation failed. The `hint` explains the fix. |
| 503 | `model_unavailable` | The model failed to load, or its outputs were not finite after the 32-bit retry. |
| 504 | `timeout` | The request took more than 30 s (300 s for batches). |
| 529 | `busy` | The queue is full; retry with backoff. |

### 6.2 MCP tools

The endpoint is `http://127.0.0.1:7711/mcp` over Streamable HTTP. It is registered in Claude Code with:

```bash
claude mcp add --transport http --scope user local-jev http://127.0.0.1:7711/mcp
```

**`decide(state, questions, model="auto")`** takes the same input and returns the same output as `POST /v1/systemone`.

**`decide_batch(questions, items=None, path=None, format="auto", field=None, chunk_tokens=256, filter=None, top_k=20, model="auto")`**

- **Input.** The caller supplies exactly one of `items` (an inline list of states) or `path` (an absolute path to a local file).
- **`format`** is one of the following. `auto` picks by file extension: `.jsonl` and `.ndjson` become `jsonl`, `.csv` becomes `csv`, and anything else becomes `lines`. Inline `items` are always treated as a list of states.

  | Format | Items |
  |---|---|
  | `jsonl` | Each line is a JSON object. `field` is an optional dotted path to the text; default is the whole object. |
  | `csv` | Each row. `field` is the column name; default is the whole row as an object. |
  | `lines` | Each non-empty line. |
  | `chunks` | The text split at line boundaries into pieces of about `chunk_tokens` tokens, counted with the tokenizer of the checkpoint that will answer. A single line longer than `chunk_tokens` is split by tokens. Each item records its `start_line` and `end_line`. |

- **Item id.** Each item gets an `id`: the item's own `id` field if it has one, otherwise its line or row number.
- **`filter`** is optional: `{"question": "<id>", "op": ">=" | "<=" | "==", "value": <number | label>}`.
  - A `noul` compares its P(true).
  - A `choice` checks whether the chosen label equals the value.
  - A `score` compares the expected score.
- **Limits:** at most 8 questions, 5,000 items, 20,000 question evaluations and 50 MB of file.
- **Execution:** laya-mlx has no API for scoring several states in one forward pass, so each item is one model call. Items go to the inference worker in groups of 32, one job per group, so single `decide` requests can run between groups. If the 300 s deadline passes, the batch fails with a 504 that says how many items were done.
- **The result contains:**
  - **`count`:** the number of items.
  - **`summary`:** for each question, either counts per label (`choice`); the count at or above 0.5 plus the mean P(true) (`noul`); or a histogram of the most likely level plus the mean score (`score`).
  - **`matches`:** up to `top_k` items that pass the filter, or all items when there is no filter. They are sorted by the filter question's `answer_confidence`, or by input order when there is no filter. Each match has its `id`, a snippet of the first 200 characters, its line range when the format is `chunks`, and compact answers.
  - **`output_path`:** the full per-item results are always written to `~/Library/Caches/local-jev/results/<UTC timestamp>-<8 hex>.jsonl`, and this is that path.
  - **`warnings`** and **`latency_ms`**.

**`status()`** reports the backend, the loaded checkpoints, approximate memory, the idle-unload countdown, latency percentiles (median and 95th), the queue depth, the version, and a summary of the configuration. It never loads a model.

**Tool descriptions** are part of the interface and must state these rules for the calling agent:

- Questions are read literally, so say exactly what you mean.
- There is no arithmetic or counting, and dates are read as text.
- Keep the state short and factual: send evidence, not summaries. The English model sees about 320 tokens of it.
- Phrase questions positively; negations are unreliable.
- At most 20 options; more are shortlisted.
- Probabilities rank answers but are not calibrated, and confidence is not permission.
- The state is not treated as hostile, so do not use this as the only check on untrusted text.
- Prefer `decide_batch` with `path` for many items, so the items stay out of your context.

### 6.3 CLI

| Command | What it does |
|---|---|
| `local-jev serve [--port 7711]` | Runs the server in the foreground. It always binds to 127.0.0.1. |
| `local-jev pull [--all]` | Downloads the pinned English and multilingual checkpoints; `--all` adds typed-decisions. |
| `local-jev decide (--state TEXT \| --state-file F) --questions-file Q [--model M]` | Posts to `/v1/systemone` and prints JSON. |
| `local-jev batch --questions-file Q --path FILE [--format …] [--field …] [--filter …] [--top-k N]` | Posts to `/v1/batch`. |
| `local-jev status` | Prints `/v1/status`, or says the server is not running. |
| `local-jev connect [--harness claude-code\|codex\|cursor\|json] [--apply]` | Prints the connection command or configuration for a harness. `--apply` runs the `claude mcp add` command for Claude Code after asking for confirmation. |
| `local-jev install-agent` / `uninstall-agent` | Adds or removes the macOS login agent that runs `local-jev serve` at login: label `io.github.andresrubio.local-jev`, file `~/Library/LaunchAgents/io.github.andresrubio.local-jev.plist`, with `RunAtLoad` and `KeepAlive` on and logs in `~/Library/Logs/local-jev/`. It shows the plist path and contents and asks before writing, then loads the agent with `launchctl bootstrap gui/$UID`. |

## 7. Guard layer

1. **Validation** follows Jev's schema:
   - `noul` criteria are optional and use the keys `true` and `false`.
   - `choice` needs at least 2 unique labels, given as a list or a dict. Jev allows up to 255.
   - `score` needs 2 to 10 levels, each with a non-empty description; upstream Laya rejects null levels.
   - Instructions may be a string, an object or an array. Objects and arrays are serialized to JSON.
2. **The yes/no mode** is set by the `noul_mode` configuration option:
   - `native` sends the question as a Laya `noul`, with the default `false`/`true` option labels.
   - `ab_labels` adds `"labels": {"true": "A", "false": "B"}` to every `noul` question that has no `labels` of its own. laya-mlx 0.4.0 shows these neutral labels to the model and still returns `noul` = P(true). This is Laya's documented workaround for label bias.
   - Callers may always pass their own `labels`.
   - The default is set by the evaluation in §10.3: the mode with higher accuracy on the yes/no set wins, and a tie goes to `native`.
3. **More than 20 options** are narrowed by `laya_mlx.predict_tournament` in elimination rounds. Each round cuts the options into groups of at most 20, answers every group in one shared forward pass, and sends each group's winner to the next round. One final call then answers every question, with each narrowed choice cut to its finalists. A choice with 20 or fewer options is never narrowed, even when it shares a request with a larger one. The answer gains `shortlisted: {"kept": [...], "dropped": n}`, and its probabilities cover only the finalists. When a round truncates the state or collapses options, the narrowed question gets the same warnings as in rule 5.
4. **Option-budget overflow.** When laya-mlx raises "too many options for the token budget", the service returns a 422 with the hint "shorten option descriptions or split into two questions".
5. **Truncation and collapsed options.** laya-mlx 0.4.0 reports both in `usage`:
   - `truncated_questions` and the token counts become the warning `state_truncated` on each affected answer, with the hint "use decide_batch with format=chunks, or model=multilingual".
   - `usage.options` becomes the warning `options_collapsed` when some options no longer had a distinct token span.
6. **`answer_confidence`** gives every answer type one comparable number:
   - For `choice`, the probability of the chosen label.
   - For `score`, the probability of the most likely level.
   - For `noul`, the larger of p and 1 − p.

   laya-mlx 0.4.0 returns this value itself; the guard computes it the same way for backends that do not. Laya's `confidence` field, which is 1 minus the normalized entropy, is passed through unchanged.
7. **Non-finite outputs** get one retry on a 32-bit copy of the checkpoint, loaded lazily. If that also fails, the service returns a 503.
8. **Remote backends** get rules 1 and 6 only. Laya-specific changes are skipped, because they depend on flags the engine declares.

## 8. Model lifecycle and performance

- **Checkpoints** come from the original publisher on Hugging Face, `convaiinnovations/laya`:
  - English is the repository root: 842 MB of weights plus a 3 MB tokenizer.
  - Multilingual is the `multilingual` subfolder: 643 MB plus a 34 MB tokenizer.
  - Typed decisions is the `typed-decisions` subfolder: 842 MB plus 3 MB. It loads only when a request asks for it by name.
  - laya-mlx downloads only the weight, configuration and tokenizer files, into the standard Hugging Face cache.
  - **Pinned revision: `c5d78730f3493e4fe16d61507ef4b78eef7318cf`.** This is the revision laya-mlx 0.2.0 validated its port against. The weight files it serves (SHA-256 prefixes: English `891102d37268`, multilingual `9d628fd971b7`, typed decisions `4fa56de72383`) are byte-identical to the latest revision (`55cf4c4…`, 2026-09-24) and to the separate `laya-multilingual` and `laya-typed-decisions` repositories that laya-mlx tested. The revision is stored in the configuration defaults. An integration test checks the downloaded weight hashes against these prefixes.
- **Routing** uses `laya_mlx.Router(dtype="float16", max_loaded=2, default="english")`.
  - The `models` mapping points at the pinned snapshot in the Hugging Face cache, because the Router itself cannot take a revision.
  - It detects the language and script and sends non-English text to the multilingual checkpoint.
  - `default="english"` keeps the pre-0.4.0 behaviour, in which input whose language cannot be decided (code, short commands) goes to the English checkpoint.
  - An explicit `model` always wins.
- **Downloads.**
  - `local-jev serve` starts a background thread that downloads and verifies the English and multilingual checkpoints when they are missing. This is `laya.prefetch` in the configuration.
  - A request that needs a checkpoint that is not ready yet gets a 503, `model_unavailable`, with a hint to retry, or to run `local-jev pull` (`--all` for typed decisions). It never hangs waiting for a download.
  - Weight hashes are checked once per process.
- **Loading** happens on the first request that needs a checkpoint. There is no preload option.
- **Idle unload.** After `idle_unload_minutes` (default 15; 0 means never), the service calls `Router.unload()` and then `mx.clear_cache()`. The next request loads the model again.
- **Worker.** One inference thread handles requests from a queue of up to 64. Engine calls run 16 questions per forward pass.
- **Targets**, measured on this M1 Pro with 16 GB and reported in `docs/benchmarks.md`:

  | Measure | Target |
  |---|---|
  | One short question, once loaded | median ≤ 100 ms |
  | 10 questions, once loaded | median ≤ 400 ms |
  | Loading from disk | ≤ 10 s |
  | Process memory with both checkpoints loaded | ≤ 2.5 GB |
  | Process memory after idle unload | ≤ 400 MB |

## 9. Security and privacy

- **Binding.** The server binds to `127.0.0.1` only; there is no option to bind other addresses in v1.
- **Host and Origin checks.** Middleware on every route rejects a Host that is not `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>`, with a 421. It rejects, with a 403, any Origin that is present and is not a loopback origin (`http://127.0.0.1`, `http://localhost` or `http://[::1]`, on any port). This includes `Origin: null`. The MCP SDK's own transport security is also configured with the same loopback allowlist.
- **Optional bearer token** (`LOCAL_JEV_TOKEN`), checked on every route except `/health`. `connect` includes the header when a token is set. `install-agent` copies a set token into the plist's `EnvironmentVariables` and writes the plist with mode 600.
- **File policy for `decide_batch`.**
  - The path must be absolute and resolve, after following symlinks, to a regular file under `batch.allowed_roots` (default: the home directory).
  - The file is at most 50 MB, is decoded as UTF-8 with replacement characters for bad bytes, and is opened read-only.
  - Results are written only under `~/Library/Caches/local-jev/results/`. Files there older than 7 days are deleted when the server starts.
- **Nothing is executed.** No part of the state or a file is ever run. The only outbound network calls are Hugging Face downloads, during `pull` or on first use, and the opt-in remote backend.
- **Hosted Jev** requires `backend = "remote"`, a URL, and the environment variable named in `remote.api_key_env`. Log records from the remote backend are tagged `remote`.
- **Logs** are kept in `~/Library/Logs/local-jev/`:
  - `server.log`.
  - `decisions.jsonl`: by default, metadata only (time, interface, backend and checkpoint, question ids and types, answers with `answer_confidence`, latency, token counts, warnings). `log_inputs = true` also stores states and questions, for training later.
  - Both rotate at 10 MB, keeping 5 files.

## 10. Testing and evaluation

### 10.1 Unit tests (pytest, fake engine, no weights; run in CI-like `uv run pytest`)

- **Schema:** TypeSafe's documented request and response examples parse and serialize, and each invalid-question case gets the right code and hint.
- **Guard layer:** yes/no questions sent as A/B map back correctly, the shortlist triggers above 20 options, truncation is detected (using a stub tokenizer), `answer_confidence` is right for each type, and remote backends get only rules 1 and 6.
- **Batch:** every reader (including JSONL `field` paths, CSV columns, and chunk line ranges), every filter operator, top-k ordering, summaries, the result-file location, and the limits.
- **Runtime** (with a fake clock): ordering through the single worker, the full queue returning 529, the timeout returning 504, and idle unload and reload.
- **Security:** a bad Host returns 421, a bad Origin returns 403, the token check works, and file-policy violations are rejected (relative path, symlink escaping the allowed roots, oversize file).
- **CLI:** argument parsing, `connect` output for each harness, and plist rendering with a golden file.

### 10.2 Integration tests (`pytest -m integration`, real weights)

- **Loading:** English and multilingual load, and fixed sanity cases give the expected answers, including an English billing-routing example and a Spanish one.
- **Jev compatibility:** a request built to TypeSafe's documented schema, sent to `localhost/v1/systemone`, gets a response in the same schema.
- **MCP:** a client from the MCP Python SDK connects over Streamable HTTP, lists the three tools, and calls each one.
- **Performance:** latency, load time and memory are measured against the targets in §8 and written to `docs/benchmarks.md`.

### 10.3 Mini-evaluation (`eval/`)

- **The set:** 60 hand-labelled items, about 20% in Spanish.
  - 30 yes/no, including 10 negations.
  - 20 choice, with 3 to 6 options.
  - 10 score.
- **Output:** accuracy per type and checkpoint, for both yes/no modes. The result sets the `noul_mode` default (§7.2) and becomes a "What to expect" section in the README, with the known limits.

### 10.4 Manual acceptance

1. `local-jev connect --apply`.
2. In a new Claude Code session, call `decide`.
3. Then call `decide_batch` with `path` on a sample JSONL of 200 items.

## 11. Acceptance criteria

1. Claude Code, registered with `claude mcp add --transport http --scope user local-jev http://127.0.0.1:7711/mcp`, lists and calls `decide`, `decide_batch` and `status`.
2. TypeSafe's documented `POST /v1/systemone` request returns a response in the same schema from localhost.
3. On this M1 Pro, one short question has a median of at most 100 ms once loaded, as recorded in `docs/benchmarks.md`.
4. After the idle timeout, process memory falls to 400 MB or less, and the next request succeeds.
5. Requests with a non-loopback Host or Origin are rejected.
6. `uv run pytest` passes, and `uv run pytest -m integration` passes on this Mac.

## 12. Project layout and tooling

- **Python 3.12**, managed with uv. The package is `local_jev` in a `src/` layout, installed with `uv tool install .` so `local-jev` is on the PATH.
- **Pinned dependencies:**
  - `laya-mlx==0.4.0`, released 2026-10-10.
  - `mcp==2.3.0`, the Python SDK v2 (`from mcp.server import MCPServer`), released 2026-10-02.
  - `uvicorn`, `starlette`, `huggingface-hub`, `tokenizers`.
  - `httpx2`, the maintained successor of httpx. The MCP SDK already depends on it, and Starlette's test client prefers it.
  - Development only: `pytest`, `ruff`.
- **Configuration defaults** in `~/.config/local-jev/config.toml`:

  ```toml
  port = 7711
  backend = "laya-mlx"           # or "remote"
  idle_unload_minutes = 15
  noul_mode = "native"           # "native" or "ab_labels"; final default set by §10.3
  log_inputs = false

  [laya]
  revision = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"
  dtype = "float16"
  prefetch = ["english", "multilingual"]   # downloaded and verified in the background at serve start

  [batch]
  allowed_roots = ["~"]

  [remote]
  url = ""
  api_key_env = ""                # e.g. "TYPESAFE_API_KEY"
  ```

- **Layout:**

  ```text
  local-jev/
    pyproject.toml   README.md   docs/benchmarks.md   docs/superpowers/specs/
    src/local_jev/{__init__,config,schema,guard,runtime,batch,security,api_rest,api_mcp,app,decision_log,cli,launchd}.py
    src/local_jev/engines/{__init__,base,laya_mlx,remote}.py
    tests/            eval/
  ```

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Laya is about a week old and laya-mlx about six days. | Pin the package version and the model revision. The `Engine` interface lets the backend change to upstream `laya-serve`, Kev or hosted Jev through `RemoteJevEngine`. |
| Untrained accuracy on nuanced decisions is weak. | Warnings and `answer_confidence` on every answer; the §10.3 evaluation is published in the README; tool descriptions tell callers what the model is good for. Calibration and fine-tuning are future work. |
| MLX thread safety and GPU contention. | A single inference worker, with MLX used only on that thread. |
| Memory pressure on 16 GB. | At most 2 loaded checkpoints, typed decisions loaded on demand, idle unload, and memory targets in §8. |
| MCP SDK v2 API changes. | An exact version pin, plus the MCP integration test. |
| Prompt injection inside a state can move answers. | Documented in the tool descriptions. v1 is not a security boundary for untrusted text. |

## 14. Future work (post-v1)

- **Claude Code hooks:** a command-safety gate on `PreToolUse`, and a relevance filter for large outputs on `PostToolUse` that uses `decide_batch` with the `chunks` format.
- **`local-jev calibrate`:** fit temperatures from labelled decision logs.
- **Trained heads:** decision heads trained on the model's own embeddings, from the user's labelled examples.
- **Apple Neural Engine:** use it through laya-apple.
- **Kev in-process:** add Kev as an in-process engine.

## 15. Amendments (2026-10-14)

New releases appeared between drafting and approval. The plan follows these changes, which are already applied in the sections above:

| What changed | Why |
|---|---|
| `laya-mlx` pin 0.2.0 → 0.4.0 | It natively reports truncation and collapsed options, returns `answer_confidence`, supports neutral yes/no `labels`, and frees MLX memory on unload. That removes custom guard code. Its 0.4.0 validation reports 378/378 answer agreement with upstream. |
| `mcp` pin 2.2.0 → 2.3.0 | Fixes only; it adds the in-process `Client(server)` that the tests use. |
| `Router(default="english")` | laya-mlx 0.4.0 changed the default for input whose language cannot be decided to the multilingual checkpoint. This keeps the routing this spec describes. |
| `noul_mode` values `native` / `ab_labels` (was `ab_choice`) | Neutral labels are now native; no rewrite into a choice question is needed. |
| Background prefetch at `serve` start; 503 while downloading; no `preload` option | A first request must not hang on an 845 MB download. |
| Model revision stays `c5d7873…` | The newer Hugging Face revision `7b928d8…` (2026-10-03) has byte-identical weights, configurations and tokenizers. |
| `decide_batch` runs one model call per item, in groups of 32 | laya-mlx has no multi-state batching API. |
| New modules `errors.py`, `checkpoints.py`, `service.py` | Clearer unit boundaries. REST and MCP share one service. |
| `warnings` are strings that start with a code; `usage` holds only Jev's fields; Laya's `action` is dropped | Jev compatibility, and readable messages for agents. |
| `httpx2` instead of `httpx` | It is already a dependency of the MCP SDK, and Starlette's test client prefers it. |
| 2026-10-20: more than 20 options are narrowed by `laya_mlx.predict_tournament` in groups of at most 20 (was `predict_shortlist` with `embed_fn_from_agent`) | On the real weights, the embedding shortlist dropped the right label in 3 of 3 cases with 30 options, and the engine gave a wrong answer at 0.82 to 0.98 each time. The tournament picked the right label in all three (0.93 to 0.96). |
