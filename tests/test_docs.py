"""What the docs cite must exist: the tests they name, and the SDK symbols they name.

docs/mcp-sdk-notes.md pins each fact to a test and names SDK functions by their dotted path.
A cited test that is gone, or a dotted path the pinned SDK no longer has, fails here."""

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted(ROOT.glob("docs/*.md")) + [ROOT / "CODING_STANDARDS.md", ROOT / "README.md"]
TESTS = ROOT / "tests"

CITED_TEST = re.compile(r"`(test_[a-z0-9_]+)`")
SDK_PATH = re.compile(r"`((?:mcp|mcp_types)(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`")


def cited(pattern):
    return sorted({(doc.name, name) for doc in DOCS for name in pattern.findall(doc.read_text())})


def defined_tests():
    names = set()
    for path in TESTS.rglob("test_*.py"):
        names.update(re.findall(r"^(?:async )?def (test_[a-z0-9_]+)\(", path.read_text(), re.M))
    return names


@pytest.mark.parametrize(("doc", "name"), cited(CITED_TEST), ids=lambda value: value)
def test_every_test_a_doc_cites_exists(doc, name):
    assert name in defined_tests(), f"{doc} cites {name}, which no test file defines"


def resolve(dotted):
    """Import the longest module prefix, then walk attributes for the rest."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            value = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attribute in parts[cut:]:
            value = getattr(value, attribute)
        return value
    raise ImportError(dotted)


@pytest.mark.parametrize(("doc", "dotted"), cited(SDK_PATH), ids=lambda value: value)
def test_every_sdk_path_a_doc_names_resolves(doc, dotted):
    try:
        resolve(dotted)
    except (ImportError, AttributeError) as missing:
        pytest.fail(f"{doc} names {dotted}, which the pinned SDK does not have: {missing}")


def test_the_docs_cite_something():
    """The two checks above are empty, and pass, when the regexes stop matching the docs."""
    assert len(cited(CITED_TEST)) >= 9
    assert len(cited(SDK_PATH)) >= 3
