"""decide_batch building blocks: parameters, item readers, filters, summaries and result files."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import secrets
import stat
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

from local_jev.errors import ServiceError, invalid, too_large
from local_jev.guard import compact_answers, warning_code
from local_jev.schema import (
    NonFiniteNumber,
    finite_float,
    parse_model,
    parse_questions,
    parse_state,
    refuse_constant,
)

FORMATS = ("auto", "jsonl", "csv", "lines", "chunks")
FILTER_OPS = (">=", "<=", "==")
MAX_BATCH_QUESTIONS = 8
SNIPPET_CHARS = 200
RESULT_MAX_AGE_SECONDS = 7 * 24 * 3600
FILTER_EXAMPLE = '{"question": "relevant", "op": ">=", "value": 0.8}'
READ_HINT = "check the path and its permissions"
JSONL_HINT = "each line must hold one JSON value"
READ_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class Filter:
    question: str
    op: str
    value: Any


@dataclass(frozen=True)
class BatchParams:
    questions: dict[str, dict[str, Any]]
    items: list[Any] | None
    path: str | None
    format: str
    field: str | None
    chunk_tokens: int
    filter: Filter | None
    top_k: int
    model: str


@dataclass(frozen=True)
class Item:
    id: Any
    n: int
    state: Any
    start_line: int | None = None
    end_line: int | None = None


def parse_batch_params(body: Any, *, max_items: int) -> BatchParams:
    if not isinstance(body, dict):
        raise invalid(
            "the request body must be a JSON object",
            '{"questions": {...}, "path": "/abs/file.jsonl"}',
        )
    questions = parse_questions(body.get("questions"), max_questions=MAX_BATCH_QUESTIONS)
    items, path = body.get("items"), body.get("path")
    if (items is None) == (path is None):
        raise invalid(
            "pass exactly one of 'items' or 'path'",
            "items for a short inline list, path for a local file",
        )
    if items is not None:
        if not isinstance(items, list) or not items:
            raise invalid("'items' must be a non-empty array", "or pass 'path' to read a file")
        if len(items) > max_items:
            raise too_large(f"{len(items)} items; the limit is {max_items}", "split the batch")
    elif not isinstance(path, str) or not path:
        raise invalid("'path' must be a non-empty string", "pass an absolute path to a local file")
    fmt = body.get("format", "auto")
    if fmt not in FORMATS:
        raise invalid(f"unknown format {fmt!r}", f"use one of {', '.join(FORMATS)}")
    field = body.get("field")
    if field is not None and (not isinstance(field, str) or not field.strip()):
        raise invalid(
            "'field' must be a non-empty string",
            "a dotted JSON path such as body.text, or a CSV column",
        )
    return BatchParams(
        questions=questions,
        items=items,
        path=path,
        format=fmt,
        field=field,
        chunk_tokens=_int_in_range(body.get("chunk_tokens", 256), "chunk_tokens", 32, 1024),
        filter=_parse_filter(body.get("filter"), questions),
        top_k=_int_in_range(body.get("top_k", 20), "top_k", 1, 200),
        model=parse_model(body.get("model")),
    )


def _int_in_range(value: Any, name: str, low: int, high: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise invalid(f"'{name}' must be an integer from {low} to {high}", f"got {value!r}")
    return value


def _parse_filter(raw: Any, questions: dict[str, dict[str, Any]]) -> Filter | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {"question", "op", "value"}:
        raise invalid("'filter' must have exactly the keys question, op and value", FILTER_EXAMPLE)
    qid, op, value = raw["question"], raw["op"], raw["value"]
    if not isinstance(qid, str) or qid not in questions:
        raise invalid(f"filter question {qid!r} is not one of the questions", FILTER_EXAMPLE)
    if op not in FILTER_OPS:
        raise invalid(f"filter op {op!r} is not supported", "use >=, <= or ==")
    question = questions[qid]
    if question["type"] == "choice":
        if op != "==" or value not in list(question["criteria"]):
            raise invalid(
                "a choice filter needs op '==' and one of the question's labels",
                '{"question": "dept", "op": "==", "value": "billing"}',
            )
    elif not isinstance(value, (int, float)) or isinstance(value, bool):
        raise invalid(
            f"a {question['type']} filter needs a numeric value",
            "noul compares P(true), score the expected level",
        )
    return Filter(question=qid, op=op, value=value)


def resolve_format(fmt: str, path: Path) -> str:
    if fmt != "auto":
        return fmt
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    return "csv" if suffix == ".csv" else "lines"


def load_items(
    params: BatchParams, *, path: Path | None, count_tokens: Callable[[str], int], max_items: int
) -> list[Item]:
    if params.items is not None:
        return _inline_items(params.items, max_items)
    if path is None:
        raise invalid("'path' was not checked", "this is a bug; please report it")
    text = _read_text(path)
    # auto goes by the name the caller gave; path is the resolved file and may be named otherwise.
    fmt = resolve_format(params.format, Path(params.path) if params.path else path)
    if fmt == "jsonl":
        found = _jsonl_items(text, params.field)
    elif fmt == "csv":
        found = _csv_items(text, params.field)
    elif fmt == "lines":
        found = (
            Item(id=n, n=n, state=line)
            for n, line in enumerate(_lines(text), start=1)
            if line.strip()
        )
    else:
        found = _chunk_items(text, params.chunk_tokens, count_tokens)
    # The readers are lazy, so one item past the limit is as far as any of them goes.
    items = list(islice(found, max_items + 1))
    if not items:
        raise invalid(f"{path} has no items", "check the format and that the file is not empty")
    if len(items) > max_items:
        raise too_large(
            f"{path} has more than {max_items} items; the limit is {max_items}", "split the file"
        )
    return items


def _inline_items(items: list[Any], max_items: int) -> list[Item]:
    if len(items) > max_items:
        raise too_large(f"{len(items)} items; the limit is {max_items}", "split the batch")
    return [
        Item(id=_own_id(state, n), n=n, state=_checked_state(n, state))
        for n, state in enumerate(items, start=1)
    ]


def _checked_state(n: int, state: Any) -> Any:
    """The item if it is a valid state, as for a single decide; else that error, naming the item."""
    try:
        return parse_state(state)
    except ServiceError as exc:
        raise ServiceError(exc.status, exc.code, f"item {n}: {exc.message}", exc.hint) from exc


def _read_text(path: Path) -> str:
    """The text of a regular file, read through one descriptor opened without following symlinks.

    FilePolicy.check approved this path earlier, but the file may have changed since. O_NOFOLLOW
    refuses a file that is now a symlink, O_NONBLOCK keeps a pipe from blocking the open, and fstat
    on the open descriptor refuses anything that is not a regular file. Only the bytes counted
    when the file was opened are read, so a file that keeps growing cannot grow the read.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise invalid(f"{path} could not be read: it is not a regular file", READ_HINT)
            chunks: list[bytes] = []
            remaining = info.st_size
            while remaining > 0:
                chunk = os.read(fd, min(remaining, READ_CHUNK_BYTES))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
        finally:
            os.close(fd)
    except OSError as exc:
        raise invalid(
            f"{path} could not be read: {exc.strerror or type(exc).__name__}", READ_HINT
        ) from exc
    # The wrapper is what Path.read_text uses: bad bytes become U+FFFD and \r\n and \r become \n.
    # A leading byte order mark, which Excel's CSV UTF-8 writes, is dropped from the text. The
    # utf-8-sig codec would do the same but loses a file that is only half a mark, with no U+FFFD.
    raw = io.BytesIO(b"".join(chunks))
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace").read().removeprefix("\ufeff")


def _lines(text: str) -> Iterator[str]:
    """The lines of the text, ended by a newline and by nothing else.

    str.splitlines also splits at form feeds, U+2028, U+0085 and a few more characters. That moves
    line numbers away from what grep -n and an editor show, and cuts valid JSON strings in two.
    By now every line ending is a newline. Like splitlines, a final newline starts no new line.
    It is lazy, so a reader that stops early does not split the rest of the file.
    """
    start = 0
    while start < len(text):
        end = text.find("\n", start)
        if end < 0:
            end = len(text)
        yield text[start:end]
        start = end + 1


def _own_id(value: Any, fallback: int) -> Any:
    """The item's own id, or its number when it has none or one that no JSON response can hold.

    MCP reads requests with pydantic_core.from_json, which turns NaN and the infinities into floats.
    """
    own = value["id"] if isinstance(value, dict) and "id" in value else fallback
    return fallback if isinstance(own, float) and not math.isfinite(own) else own


def _as_state(value: Any) -> Any:
    if isinstance(value, (str, dict, list)) and value != "":
        return value
    return json.dumps(value)


def _jsonl_items(text: str, field: str | None) -> Iterator[Item]:
    for line_no, line in enumerate(_lines(text), start=1):
        if not line.strip():
            continue
        try:
            # The hooks of the REST body parser: an id of NaN would make the response unwritable.
            value = json.loads(line, parse_constant=refuse_constant, parse_float=finite_float)
        except RecursionError as exc:
            raise invalid(
                f"line {line_no} is not valid JSON: it is nested too deeply", JSONL_HINT
            ) from exc
        except json.JSONDecodeError as exc:
            raise invalid(f"line {line_no} is not valid JSON: {exc.msg}", JSONL_HINT) from exc
        except NonFiniteNumber as exc:
            raise invalid(f"line {line_no} is not valid JSON: {exc}", JSONL_HINT) from exc
        except ValueError as exc:
            raise invalid(
                f"line {line_no} is not valid JSON: a number has too many digits", JSONL_HINT
            ) from exc
        state = _lookup(value, field, line_no) if field else value
        yield Item(id=_own_id(value, line_no), n=line_no, state=_as_state(state))


def _lookup(value: Any, dotted: str, line_no: int) -> Any:
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise invalid(f"line {line_no} has no field {dotted!r}", "check the 'field' path")
        value = value[part]
    return value


def _csv_items(text: str, field: str | None) -> Iterator[Item]:
    reader = csv.DictReader(io.StringIO(text))
    try:
        columns = reader.fieldnames or []
        if not columns:
            raise invalid("the CSV file has no header row", "the first line must name the columns")
        if field is not None and field not in columns:
            raise invalid(f"column {field!r} not found", f"columns: {', '.join(columns)}")
        for n, row in enumerate(reader, start=1):
            state = (
                row[field]
                if field is not None
                else {key: value for key, value in row.items() if key is not None}
            )
            yield Item(id=row.get("id") or n, n=n, state=_as_state(state))
    except csv.Error as exc:
        # Raised for a cell over the csv module's field size limit.
        hint = (
            f"close every quoted cell, and keep each cell under {csv.field_size_limit()} characters"
        )
        raise invalid(f"the CSV file could not be parsed: {exc}", hint) from exc


def _chunk_items(
    text: str, chunk_tokens: int, count_tokens: Callable[[str], int]
) -> Iterator[Item]:
    buffer: list[str] = []
    buffer_tokens = 0
    start: int | None = None
    built = 0

    def flush(end_line: int) -> list[Item]:
        nonlocal buffer, buffer_tokens, start, built
        body = "\n".join(buffer)
        chunks = []
        if body.strip() and start is not None:
            built += 1
            chunks.append(
                Item(
                    id=f"{start}-{end_line}",
                    n=built,
                    state=body,
                    start_line=start,
                    end_line=end_line,
                )
            )
        buffer, buffer_tokens, start = [], 0, None
        return chunks

    line_no = 0
    for line_no, line in enumerate(_lines(text), start=1):
        tokens = count_tokens(line) if line.strip() else 0
        if tokens > chunk_tokens:
            yield from flush(line_no - 1)
            pieces = math.ceil(tokens / chunk_tokens)
            size = math.ceil(len(line) / pieces)
            for index in range(pieces):
                piece = line[index * size : (index + 1) * size]
                if piece.strip():
                    built += 1
                    yield Item(
                        id=f"{line_no}-{line_no}#{index + 1}",
                        n=built,
                        state=piece,
                        start_line=line_no,
                        end_line=line_no,
                    )
            continue
        if buffer and buffer_tokens + tokens > chunk_tokens:
            yield from flush(line_no - 1)
        if start is None:
            start = line_no
        buffer.append(line)
        buffer_tokens += tokens
    yield from flush(line_no)


def passes(flt: Filter | None, answers: dict[str, dict[str, Any]]) -> bool:
    if flt is None:
        return True
    answer = answers[flt.question]
    if answer["type"] == "choice":
        return answer["choice"] == flt.value
    value = answer["noul"] if answer["type"] == "noul" else answer["score"]
    if flt.op == ">=":
        return value >= flt.value
    if flt.op == "<=":
        return value <= flt.value
    return value == flt.value


def select_matches(
    records: list[dict[str, Any]], flt: Filter | None, top_k: int
) -> list[dict[str, Any]]:
    matching = [record for record in records if passes(flt, record["answers"])]
    if flt is not None:
        matching.sort(
            key=lambda record: record["answers"][flt.question]["answer_confidence"], reverse=True
        )
    return matching[:top_k]


def summarize(
    records: list[dict[str, Any]], questions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for qid, question in questions.items():
        answers = [record["answers"][qid] for record in records]
        if question["type"] == "choice":
            counts = dict.fromkeys(question["criteria"], 0)
            for answer in answers:
                counts[answer["choice"]] = counts.get(answer["choice"], 0) + 1
            summary[qid] = {"type": "choice", "counts": counts}
        elif question["type"] == "noul":
            values = [answer["noul"] for answer in answers]
            summary[qid] = {
                "type": "noul",
                "yes": sum(value >= 0.5 for value in values),
                "no": sum(value < 0.5 for value in values),
                "mean": round(sum(values) / len(values), 4),
            }
        else:
            histogram: dict[str, int] = {}
            for answer in answers:
                probabilities = answer.get("probabilities") or {}
                level = (
                    max(probabilities, key=probabilities.get)
                    if probabilities
                    else str(round(answer["score"]))
                )
                histogram[level] = histogram.get(level, 0) + 1
            scores = [answer["score"] for answer in answers]
            summary[qid] = {
                "type": "score",
                "histogram": dict(sorted(histogram.items(), key=lambda kv: int(kv[0]))),
                "mean": round(sum(scores) / len(scores), 4),
            }
    return summary


def aggregate_warnings(records: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for record in records:
        keys = {
            f"{qid}: {warning_code(warning)}"
            for qid, answer in record["answers"].items()
            for warning in answer.get("warnings", [])
        }
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
    return [f"{key} in {count} of {len(records)} items" for key, count in sorted(counts.items())]


def snippet(state: Any) -> str:
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return text[:SNIPPET_CHARS]


def public_record(record: dict[str, Any], *, include_warnings: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": record["id"],
        "n": record["n"],
        "snippet": record["snippet"],
        "answers": compact_answers(record["answers"]),
    }
    if record.get("start_line") is not None:
        out["start_line"] = record["start_line"]
        out["end_line"] = record["end_line"]
    if include_warnings:
        out["checkpoint"] = record.get("checkpoint")
        out["warnings"] = [
            f"{qid}: {warning}"
            for qid, answer in record["answers"].items()
            for warning in answer.get("warnings", [])
        ]
    return out


# Exactly the names write_results gives its files: a UTC time, then 4 random bytes in lowercase hex.
RESULT_NAME = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}\.jsonl")


def write_results(results_dir: Path, records: list[dict[str, Any]]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}.jsonl"
    # A lone surrogate is written as a JSON escape (\udXXX) instead of failing the whole batch.
    with path.open("w", encoding="utf-8", errors="backslashreplace") as fh:
        for record in records:
            fh.write(
                json.dumps(
                    public_record(record, include_warnings=True), ensure_ascii=False, default=str
                )
                + "\n"
            )
    return path


def cleanup_results(
    results_dir: Path, *, max_age_seconds: float = RESULT_MAX_AGE_SECONDS, now: float | None = None
) -> int:
    """Delete the result files older than max_age_seconds; return how many went.

    results_dir may point at a folder that also holds the user's own files, so only names that
    write_results gives are deleted.
    """
    if not results_dir.exists():
        return 0
    cutoff = (time.time() if now is None else now) - max_age_seconds
    removed = 0
    for path in results_dir.glob("*.jsonl"):
        if not RESULT_NAME.fullmatch(path.name):
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            pass  # a broken link, a folder or a file that vanished: leave it alone
    return removed
