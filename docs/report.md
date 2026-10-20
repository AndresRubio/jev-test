# local-jev build report

local-jev works as its design spec describes on the Apple M1 Pro it was built for. It meets the spec's speed and memory targets, as long as memory is counted in the binary units that `top` uses. On 2026-10-20 a new Claude Code session listed its three MCP tools and called each one. The main limit is the Laya model's accuracy on harder questions, and some hardening work is still open.

local-jev answers typed questions about a piece of text. It runs the open-weight Laya checkpoints on Apple Silicon through laya-mlx 0.4.0. A question can be a yes/no question (`noul`), a choice among options or a score on a rubric. Agents reach it over MCP at `http://127.0.0.1:7711/mcp`, through a REST API whose `POST /v1/systemone` follows TypeSafe's Jev, or through the `local-jev` command line. A launchd login agent can keep it running.

The work followed the design spec (`docs/superpowers/specs/2026-09-25-local-jev-design.md`) and a 15-task plan (`docs/superpowers/plans/2026-10-15-local-jev.md`), plus one task the project owner approved during the build. Claude agents wrote the code one task at a time, and each task had its own review and fix loop. A final review of the whole branch led to one more round of fixes. On 2026-10-20 the branch's 43 commits were merged into `main` and pushed to https://github.com/AndresRubio/jev-test.

## Conclusions

### Speed and memory meet the targets

All figures are from `docs/benchmarks.md`, measured on the M1 Pro with 16 GB of memory and macOS 26.6.

| Measure | Result | Target |
|---|---|---|
| One short question, median | 24.0 ms | 100 ms or less |
| 10 questions in one call, median | 107.7 ms | 400 ms or less |
| First request, which loads the English checkpoint | 808 ms | 10 s or less |
| First request after the idle unload | 348 ms | 10 s or less |
| Memory with both checkpoints loaded | 2,402 MB | 2.5 GB or less |
| Memory after the idle unload | 281 MB | 400 MB or less |

`top` counts a MB as 1,048,576 bytes, so the 2,402 MB with both checkpoints loaded are 2.52 GB in decimal units. That is under the spec's limit of 2.5 GB only if the limit is read in binary units too.

A batch of short items runs at 47 items per second with one yes/no question and at 30 with two. Items of about 430 tokens bring it down to 8 per second. A batch gets 300 seconds for all of its model calls, so split any file with more than about 2,000 long items.

### The acceptance check with Claude Code passed

A non-interactive Claude Code session (`claude -p`) saw `decide`, `decide_batch` and `status` and called each one. `decide` sent a double-billing message to billing with a probability of 0.9566. `decide_batch` read a file of 200 synthetic support messages in 7.3 seconds, returned the five billing matches it was asked for and wrote its results file.

### The model is the weak part

The evaluation (`eval/results.md`) has 60 hand-labelled items, 12 of them in Spanish. On its 30 yes/no questions the model got 22 right with its own true/false labels and 24 right with neutral A/B labels. Under the spec, the mode with more right answers becomes the default, and that was `ab_labels`. The model also got 16 of 20 choice questions and 7 of 10 score questions right. Items whose text has a negation were the hardest group, with 5 of 10 right under true/false labels and 8 of 10 under A/B labels.

In the Claude Code check, 150 of the 200 support messages went to the department they were written for: all 50 billing messages, all 50 technical ones, 40 of 50 sales messages and 10 of the 50 meant for the catch-all option `other`. A question about the price difference between two plans went to billing all 10 times it appeared. An integration test records a similar miss as a strict expected failure: the English checkpoint sends a question about the cost of an upgrade to billing.

### Large choices go through elimination rounds

The spec first cut a choice with more than 20 options down to the 20 labels whose embeddings were closest to the input. With the real weights, that shortlist dropped the right label in all three 30-option cases tested, and the engine then gave a wrong answer with probabilities of 0.82 to 0.98. `laya_mlx.predict_tournament` splits the options into groups of at most 20 and runs elimination rounds. It picked the right label in all three cases, with probabilities of 0.93 to 0.96. The project owner chose the tournament on 2026-10-20, and the amendments table in section 15 of the spec records the change.

### Only this Mac can reach the server

The server listens on 127.0.0.1 only. It refuses requests addressed to any other host name and requests from web pages whose origin is not 127.0.0.1, localhost or [::1], and it can require a bearer token. Without a token any account on the Mac can reach it, so another user could have `decide_batch` read files in the owner's home folder and get snippets back. The owner chose to keep the token optional, and the README tells users to set `LOCAL_JEV_TOKEN` on a Mac with other accounts.

### The test suites pass, and nothing checks types

The 615 unit tests pass without network access or model weights. 8 integration tests pass with the real weights, and the ninth, the upgrade test, fails as expected. ruff reports no problems in `src`, `tests`, `scripts` or `eval`. No static type checker has run on the code, and `Any` appears on 158 lines of `src/`.

## Decisions made during the build

- 2026-10-19: a choice with more than 255 options gets a 422 `invalid_request` error, as the spec says. The plan had a 413.
- 2026-10-19: a review finding that came from the plan's own text was fixed without asking the owner when the fix was small, followed the spec and changed no name, interface or behaviour that the plan or spec defines. Anything else went to the owner.
- 2026-10-19: the owner approved downloading the English and multilingual weights, about 1.5 GB, for the tests that use the real model. The typed-decisions checkpoint was never downloaded.
- 2026-10-20: a choice with more than 20 options goes through the tournament.
- 2026-10-20, after the final review: a remote backend on a loopback address ignores the system proxies, `install-agent` copies the remote API key variable into the login agent, and the bearer token stays optional.
- 2026-10-20: for the acceptance check, the CLI was installed with `uv tool install --editable .`, the server ran in a terminal, and `local-jev` was registered in Claude Code at user scope.

The plan's "Recorded deviations" section lists every place where the code departs from the plan's text.

## Known limits of the model

The README's "What to expect" section has the details.

- Probabilities are over-confident. They rank answers well enough, but a high probability alone should not trigger an action.
- English choices with 11 to 20 options are uncalibrated. The checkpoint ships a temperature of 0.1006 for that size, and laya-mlx raises it to 0.5.
- The English checkpoint reads about 320 to 480 tokens of the input and the multilingual one about 760 to 990. Longer input is cut, and the answer carries a `state_truncated` warning.
- Negations in the text, pricing questions and catch-all options are weak spots.
- Text in the input can steer the answer, so local-jev should never be the only check on untrusted text.

## Shortcuts and open risks

- The plan had no type-checking step, and none was added during the build.
- `test_an_upgrade_question_goes_to_sales` (`tests/integration/test_laya_real.py:98`) records the upgrade miss as an expected failure. Its marker has no `raises=AssertionError`, so an unrelated exception would also count as the expected failure. `test_rest_and_mcp_with_the_real_model` sends the same message through the server and only checks that the answer is one of the three labels.
- The session that ran the acceptance check also wrote the 200 support messages and their expected departments, so the 150 of 200 figure is self-graded. The server for that check ran with `HF_HUB_OFFLINE=1`, which a user would not normally set.
- During the build, the agents that did the work reported the test counts and measurements, and review agents checked them. The coordinating session ran the unit and integration suites itself before the merge, and the unit suite again on `main`. It ran neither after merging GitHub's initial `LICENSE` commit.
- `_guarded` in `src/local_jev/api_rest.py` catches every unexpected exception, logs it and returns a 500 with a generic `internal_error` body, so a REST client never sees the cause.
- About 118 minor review findings were deferred during the build. Their list was in a working file that was deleted after the merge. The decisions made during the build are still recorded in the spec's amendments table (section 15) and in the plan's "Recorded deviations".
- Commits written by Sonnet and Haiku agents carry a `Co-Authored-By: Claude Opus 5.5` line, because the project's commit rule asks for that line on every commit.

## Next steps

1. Put local-jev into daily use and test the login agent. The server is not running now, and `local-jev install-agent` has never run on this Mac. Test a fresh install and a reinstall, and check whether macOS privacy protection stops the agent from reading `~/Desktop`, `~/Documents` and `~/Downloads`. This repository is under `~/Desktop`, and the editable install runs the code from it.
2. Have `install-agent` generate a bearer token by default. The server and the CLI would read it from a file only the owner can read (mode 0600) whenever `LOCAL_JEV_TOKEN` is unset. Section 9 of the spec, on security and privacy, calls the token optional and would need to change.
3. Add continuous integration and type checking. A GitHub Actions job on an Apple Silicon macOS runner can run ruff and the unit tests. Then add pyright or mypy and replace `Any` where the types are known.
4. Fix the findings that the final review parked, and the smaller deferred ones that are still known:
   - `src/local_jev/cli.py:371` calls `sys.stderr.isatty()`, which fails when `serve` starts with stderr closed.
   - An inline MCP item whose id holds a nested NaN or infinity, such as `[NaN]`, still writes invalid JSON to the results file (`src/local_jev/batch.py:262`).
   - A `~name` path for an unknown user stops every command with a traceback (`src/local_jev/config.py:202` and `:206`).
   - Warnings from tournament rounds can go to the wrong question when two choice questions with more than 20 options share their instructions. Mapping the groups by position fixes it, and the docstring at `src/local_jev/engines/laya_mlx.py:252` should say how the mapping works.
   - The components table in section 5 of the spec lists `config` as a dependency of `security.py` and leaves `security` out of the `remote.py` row.
   - The bad-token check in `cli._call` can no longer be reached from `main()`. It needs a comment saying why it stays, or it can go.
   - `repr(app.user_middleware)` shows the token (`src/local_jev/app.py:78`). Nothing logs it today.
   - The upgrade test's expected-failure marker needs `raises=AssertionError`.
   - The `decide_batch` tool description has 300 seconds written into its text (`src/local_jev/api_mcp.py:56`), so it becomes wrong if `batch.timeout_seconds` changes.
   - With proxies off for a loopback remote backend, httpx2 also stops reading `SSL_CERT_FILE` and `SSL_CERT_DIR`, so a loopback https backend that relies on them fails its certificate check. Document it or pass the certificate settings explicitly.
5. Grow the evaluation. The `ab_labels` default rests on 24 against 22 right answers out of 30. More items on pricing, catch-all options and negations, plus real examples from daily use, would show whether that margin holds. The spec lists decision heads trained on the owner's own labelled examples as future work, and they are the likely fix for these weak spots.
6. Watch for upstream changes: a laya-mlx or checkpoint release that changes how the English temperature for 11 or more options is handled, a fix for the Codex bug where `bearer_token_env_var` is not read even when the shell sets it (openai/codex#41378), and Cursor support for newer MCP protocol versions.
