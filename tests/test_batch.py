import csv
import json
import os
import signal
import time

import pytest

from local_jev.batch import (
    Filter,
    aggregate_warnings,
    cleanup_results,
    load_items,
    parse_batch_params,
    passes,
    public_record,
    resolve_format,
    select_matches,
    snippet,
    summarize,
    write_results,
)
from local_jev.errors import ServiceError

NOUL = {"type": "noul", "instructions": "Is it relevant?"}
CHOICE = {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "tech"]}
SCORE = {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}
QUESTIONS = {"relevant": NOUL, "dept": CHOICE, "urgency": SCORE}


def params(**body):
    body.setdefault("questions", {"relevant": NOUL})
    if "items" not in body and "path" not in body:
        body["items"] = ["a"]
    return parse_batch_params(body, max_items=5)


def words(text):
    return len(text.split())


def test_defaults():
    p = params()
    assert (p.format, p.chunk_tokens, p.top_k, p.model, p.filter) == ("auto", 256, 20, "auto", None)


@pytest.mark.parametrize(
    ("body", "status", "fragment"),
    [
        ({"questions": {"relevant": NOUL}}, 422, "exactly one"),
        ({"questions": {"relevant": NOUL}, "items": ["a"], "path": "/x"}, 422, "exactly one"),
        ({"questions": {"relevant": NOUL}, "items": []}, 422, "non-empty array"),
        ({"questions": {"relevant": NOUL}, "items": ["a"] * 6}, 413, "limit is 5"),
        ({"questions": {f"q{i}": NOUL for i in range(9)}, "items": ["a"]}, 413, "limit is 8"),
        ({"questions": {"relevant": NOUL}, "items": ["a"], "format": "xml"}, 422, "unknown format"),
        ({"questions": {"relevant": NOUL}, "items": ["a"], "chunk_tokens": 8}, 422, "chunk_tokens"),
        ({"questions": {"relevant": NOUL}, "items": ["a"], "top_k": 0}, 422, "top_k"),
        (
            {
                "questions": {"relevant": NOUL},
                "items": ["a"],
                "filter": {"question": "other", "op": ">=", "value": 1},
            },
            422,
            "not one of the questions",
        ),
        (
            {
                "questions": {"relevant": NOUL},
                "items": ["a"],
                "filter": {"question": "relevant", "op": ">", "value": 1},
            },
            422,
            "not supported",
        ),
        (
            {
                "questions": {"relevant": NOUL},
                "items": ["a"],
                "filter": {"question": "relevant", "op": ">=", "value": "high"},
            },
            422,
            "numeric",
        ),
        (
            {
                "questions": {"dept": CHOICE},
                "items": ["a"],
                "filter": {"question": "dept", "op": ">=", "value": "billing"},
            },
            422,
            "op '=='",
        ),
        (
            {
                "questions": {"dept": CHOICE},
                "items": ["a"],
                "filter": {"question": "dept", "op": "==", "value": "sales"},
            },
            422,
            "op '=='",
        ),
    ],
)
def test_invalid_parameters(body, status, fragment):
    with pytest.raises(ServiceError) as info:
        parse_batch_params(body, max_items=5)
    assert info.value.status == status
    assert fragment in info.value.message


def test_a_choice_filter_is_parsed():
    p = params(questions={"dept": CHOICE}, filter={"question": "dept", "op": "==", "value": "tech"})
    assert p.filter == Filter("dept", "==", "tech")


@pytest.mark.parametrize("question", [[], {}, ["relevant"], {"relevant": 1}, 1, None], ids=repr)
def test_a_filter_question_that_is_not_a_string_is_refused(question):
    body = {
        "questions": {"relevant": NOUL},
        "items": ["a"],
        "filter": {"question": question, "op": ">=", "value": 1},
    }
    with pytest.raises(ServiceError) as info:
        parse_batch_params(body, max_items=5)
    assert info.value.status == 422
    assert "not one of the questions" in info.value.message


def test_inline_items_use_their_own_id_or_their_position():
    items = load_items(
        params(items=["a", {"id": "x", "text": "b"}]), path=None, count_tokens=words, max_items=5
    )
    assert [(item.id, item.n) for item in items] == [(1, 1), ("x", 2)]


def test_jsonl_with_a_dotted_field(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text('{"id": "a", "body": {"text": "first"}}\n\n{"body": {"text": "second"}}\n')
    items = load_items(
        params(path=str(path), field="body.text"), path=path, count_tokens=words, max_items=5
    )
    assert [(item.id, item.n, item.state) for item in items] == [
        ("a", 1, "first"),
        (3, 3, "second"),
    ]


def test_jsonl_errors_name_the_line(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text('{"text": "ok"}\nnot json\n')
    with pytest.raises(ServiceError, match="line 2"):
        load_items(params(path=str(path)), path=path, count_tokens=words, max_items=5)
    path.write_text('{"text": "ok"}\n')
    with pytest.raises(ServiceError, match="no field 'body'"):
        load_items(params(path=str(path), field="body"), path=path, count_tokens=words, max_items=5)


@pytest.mark.parametrize(
    "line",
    ["[" * 100_000 + "]" * 100_000, "1" * 5_000],
    ids=["nested too deeply", "number too long"],
)
def test_jsonl_lines_that_json_cannot_load_are_refused(tmp_path, line):
    path = tmp_path / "items.jsonl"
    path.write_text('{"text": "ok"}\n' + line + "\n")
    with pytest.raises(ServiceError, match="line 2 is not valid JSON") as info:
        load_items(params(path=str(path)), path=path, count_tokens=words, max_items=5)
    assert info.value.status == 422


def test_csv_with_a_column(tmp_path):
    path = tmp_path / "items.csv"
    path.write_text("id,text\nr1,hello\n,world\n")
    items = load_items(
        params(path=str(path), field="text"), path=path, count_tokens=words, max_items=5
    )
    assert [(item.id, item.state) for item in items] == [("r1", "hello"), (2, "world")]
    with pytest.raises(ServiceError) as info:
        load_items(params(path=str(path), field="body"), path=path, count_tokens=words, max_items=5)
    assert "column 'body' not found" in info.value.message
    assert info.value.hint == "columns: id, text"


@pytest.fixture
def csv_cell_limit():
    """The csv module's own limit is process-wide, so pin it to its default for the test."""
    previous = csv.field_size_limit(131072)
    yield 131072
    csv.field_size_limit(previous)


def test_a_csv_cell_longer_than_the_csv_module_allows_is_refused(tmp_path, csv_cell_limit):
    path = tmp_path / "items.csv"
    path.write_text("id,text\nr1," + "x" * 200_000 + "\n")
    with pytest.raises(ServiceError, match="the CSV file could not be parsed") as info:
        load_items(params(path=str(path)), path=path, count_tokens=words, max_items=5)
    assert info.value.status == 422
    assert str(csv_cell_limit) in info.value.hint


def test_lines_skip_blank_lines(tmp_path):
    path = tmp_path / "items.txt"
    path.write_text("one\n\nthree\n")
    items = load_items(params(path=str(path)), path=path, count_tokens=words, max_items=5)
    assert [(item.id, item.state) for item in items] == [(1, "one"), (3, "three")]


def test_chunks_group_lines_and_split_long_ones(tmp_path):
    path = tmp_path / "doc.md"
    long_line = " ".join(f"w{i}" for i in range(70))
    path.write_text("a b c\nd e f\ng h i\n" + long_line + "\nj k\n")
    items = load_items(
        params(path=str(path), format="chunks", chunk_tokens=32),
        path=path,
        count_tokens=words,
        max_items=10,
    )
    assert items[0].state == "a b c\nd e f\ng h i"
    assert (items[0].start_line, items[0].end_line) == (1, 3)
    long_pieces = [item for item in items if item.start_line == 4]
    assert len(long_pieces) == 3
    assert len({item.id for item in long_pieces}) == 3
    assert items[-1].state == "j k"
    assert items[-1].start_line == 5


@pytest.mark.parametrize(
    ("name", "expected"),
    [("a.jsonl", "jsonl"), ("a.NDJSON", "jsonl"), ("a.csv", "csv"), ("a.md", "lines")],
)
def test_auto_format_follows_the_extension(tmp_path, name, expected):
    assert resolve_format("auto", tmp_path / name) == expected


def test_empty_and_oversized_files(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n")
    with pytest.raises(ServiceError, match="has no items"):
        load_items(params(path=str(empty)), path=empty, count_tokens=words, max_items=5)
    many = tmp_path / "many.txt"
    many.write_text("\n".join("x" for _ in range(6)))
    with pytest.raises(ServiceError) as info:
        load_items(params(path=str(many)), path=many, count_tokens=words, max_items=5)
    assert info.value.status == 413


def record(n, *, noul=0.9, choice="billing", score=1.0, warnings=()):
    return {
        "id": n,
        "n": n,
        "start_line": None,
        "end_line": None,
        "snippet": f"item {n}",
        "checkpoint": "english",
        "answers": {
            "relevant": {
                "type": "noul",
                "noul": noul,
                "answer_confidence": max(noul, 1 - noul),
                "warnings": list(warnings),
            },
            "dept": {
                "type": "choice",
                "choice": choice,
                "probabilities": {"billing": 0.8, "tech": 0.2},
                "answer_confidence": 0.8,
                "warnings": [],
            },
            "urgency": {
                "type": "score",
                "score": score,
                "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
                "answer_confidence": 0.8,
                "warnings": [],
            },
        },
    }


def test_passes():
    answers = record(1, noul=0.7, choice="tech", score=1.6)["answers"]
    assert passes(None, answers)
    assert passes(Filter("relevant", ">=", 0.5), answers)
    assert not passes(Filter("relevant", "<=", 0.5), answers)
    assert passes(Filter("dept", "==", "tech"), answers)
    assert passes(Filter("urgency", ">=", 1.5), answers)


def test_select_matches_sorts_by_confidence_and_keeps_order_without_a_filter():
    records = [record(1, noul=0.6), record(2, noul=0.95), record(3, noul=0.2), record(4, noul=0.8)]
    assert [m["id"] for m in select_matches(records, Filter("relevant", ">=", 0.5), top_k=2)] == [
        2,
        4,
    ]
    assert [m["id"] for m in select_matches(records, None, top_k=3)] == [1, 2, 3]


def test_summarize():
    records = [
        record(1, noul=0.9, choice="billing", score=1.0),
        record(2, noul=0.2, choice="tech", score=2.0),
    ]
    summary = summarize(records, QUESTIONS)
    assert summary["relevant"] == {"type": "noul", "yes": 1, "no": 1, "mean": 0.55}
    assert summary["dept"] == {"type": "choice", "counts": {"billing": 1, "tech": 1}}
    assert summary["urgency"] == {"type": "score", "histogram": {"1": 2}, "mean": 1.5}


def test_aggregate_warnings_counts_items_per_code():
    warning = "state_truncated: 10 of 400 input tokens did not fit"
    records = [record(1, warnings=[warning]), record(2), record(3, warnings=[warning, warning])]
    assert aggregate_warnings(records) == ["relevant: state_truncated in 2 of 3 items"]


def test_snippet_and_public_record():
    assert snippet("x" * 300) == "x" * 200
    assert snippet({"a": "ñ"}) == '{"a": "ñ"}'
    item = record(7)
    item["start_line"], item["end_line"] = 10, 12
    assert public_record(item) == {
        "id": 7,
        "n": 7,
        "snippet": "item 7",
        "start_line": 10,
        "end_line": 12,
        "answers": {
            "relevant": {"noul": 0.9, "answer_confidence": 0.9},
            "dept": {"choice": "billing", "answer_confidence": 0.8},
            "urgency": {"score": 1.0, "answer_confidence": 0.8},
        },
    }


def test_write_and_clean_up_results(tmp_path):
    path = write_results(tmp_path / "results", [record(1, warnings=["shortlisted: x"]), record(2)])
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert path.parent == tmp_path / "results"
    assert path.suffix == ".jsonl"
    assert lines[0]["warnings"] == ["relevant: shortlisted: x"]
    assert lines[1]["checkpoint"] == "english"
    old = tmp_path / "results" / "old.jsonl"
    old.write_text("{}\n")
    os.utime(old, (time.time() - 8 * 24 * 3600,) * 2)
    assert cleanup_results(tmp_path / "results") == 1
    assert path.exists()
    assert not old.exists()


def test_results_keep_text_that_is_not_valid_unicode_as_json_escapes(tmp_path):
    item = record(1, warnings=["shortlisted: \ud83d"])
    item["id"], item["snippet"] = "\ud800", "lone \udcff surrogate"
    path = write_results(tmp_path / "results", [item])
    line = json.loads(path.read_text(encoding="utf-8"))
    assert (line["id"], line["snippet"]) == ("\ud800", "lone \udcff surrogate")
    assert line["warnings"] == ["relevant: shortlisted: \ud83d"]


def test_cleanup_skips_entries_it_cannot_remove(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    old = results / "old.jsonl"
    old.write_text("{}\n")
    folder = results / "folder.jsonl"
    folder.mkdir()
    os.symlink(results / "missing", results / "broken.jsonl")
    ancient = time.time() - 8 * 24 * 3600
    for entry in (old, folder):
        os.utime(entry, (ancient, ancient))
    assert cleanup_results(results) == 1
    assert not old.exists()
    assert folder.exists()


# The tests below pin how load_items reads a file after FilePolicy.check has approved the path.


@pytest.fixture
def deadline():
    """Fail a test that blocks for 5 seconds instead of hanging the suite (main thread only)."""

    def expire(signum, frame):
        # Not a TimeoutError: that is an OSError, which the code under test turns into a 422.
        pytest.fail("blocked for 5 seconds")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(5)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


def refused(path):
    with pytest.raises(ServiceError) as info:
        load_items(params(path=str(path)), path=path, count_tokens=words, max_items=5)
    assert info.value.status == 422
    assert info.value.code == "invalid_request"
    assert "could not be read" in info.value.message
    assert "permissions" in info.value.hint
    return info.value


def test_a_symlink_is_not_followed(tmp_path):
    target = tmp_path / "real.txt"
    target.write_text("secret\n")
    link = tmp_path / "link.txt"
    os.symlink(target, link)
    refused(link)


def test_a_fifo_is_refused_without_blocking(tmp_path, deadline):
    fifo = tmp_path / "pipe.txt"
    os.mkfifo(fifo)
    refused(fifo)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a file with no permissions")
def test_an_unreadable_file_is_refused(tmp_path):
    path = tmp_path / "locked.txt"
    path.write_text("hello\n")
    path.chmod(0)
    refused(path)


def test_a_directory_is_refused(tmp_path):
    refused(tmp_path)


def test_a_bad_byte_becomes_a_replacement_character(tmp_path):
    path = tmp_path / "items.txt"
    path.write_bytes(b"caf\xe9 au lait\nplain\n")
    items = load_items(params(path=str(path)), path=path, count_tokens=words, max_items=5)
    assert [item.state for item in items] == ["caf\ufffd au lait", "plain"]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_every_line_ending_reads_the_same(tmp_path, newline):
    files = {
        "items.csv": (
            ["id,text", "r1,hello", ',"two', 'lines"'],
            "text",
            [("r1", "hello"), (2, "two\nlines")],
        ),
        "items.jsonl": (['{"text": "a"}', "", '{"text": "b"}'], "text", [(1, "a"), (3, "b")]),
        "items.txt": (["one", "", "three"], None, [(1, "one"), (3, "three")]),
    }
    for name, (rows, field, expected) in files.items():
        path = tmp_path / name
        path.write_bytes((newline.join(rows) + newline).encode())
        items = load_items(
            params(path=str(path), field=field), path=path, count_tokens=words, max_items=5
        )
        assert [(item.id, item.state) for item in items] == expected, name


def test_only_the_bytes_counted_when_the_file_was_opened_are_read(tmp_path, monkeypatch):
    path = tmp_path / "items.txt"
    path.write_text("one\ntwo\nthree\nfour\n")
    real_fstat, inode = os.fstat, path.stat().st_ino

    def fstat_before_growth(fd):
        info = real_fstat(fd)
        if info.st_ino != inode:
            return info
        return type("Stat", (), {"st_mode": info.st_mode, "st_size": len("one\ntwo\n")})()

    monkeypatch.setattr(os, "fstat", fstat_before_growth)
    items = load_items(params(path=str(path)), path=path, count_tokens=words, max_items=5)
    assert [item.state for item in items] == ["one", "two"]
