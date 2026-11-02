# Coding standards

Read during review, with the diff in hand. `CLAUDE.md` holds the pointers; this file holds the judgement calls.

## Claims about the SDK

A sentence in `docs/` that states how the `mcp` SDK behaves names the test that pins it, in backticks, and that test exercises the behaviour the sentence describes. A sentence that names an SDK function gives its dotted path (`mcp.shared.inbound.classify_inbound_request`), never a line number. `tests/test_docs.py` checks that the names exist; the reviewer checks that the test and the sentence say the same thing, and asks for the test first when there is none.

## Prose

Repo text is English and follows the writing rules of the humanizer hook: plain sentences, no dashes in prose, no bold labels on list items, no stock AI words. The hook checks staged `.md` files at commit time; docstrings and comments in `.py` files are the reviewer's.
