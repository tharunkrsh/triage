"""Milestone 3 Jinja2 audit report: Markdown + self-contained HTML."""

import re
import tomllib
from html.parser import HTMLParser
from pathlib import Path

import pytest

from tests.conftest import AUDIT_FIXTURES
from triage.audit import run_audit
from triage.ingest import load
from triage.models import UsageRecord
from triage.report.builder import render_html, render_markdown

TEMPLATES = Path(__file__).resolve().parents[1] / "src" / "triage" / "report" / "templates"
_VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "wbr",
}


class _TagBalance(HTMLParser):
    """Minimal well-formedness check: every non-void element is closed in order."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.seen: set[str] = set()

    def handle_starttag(self, tag, attrs):
        self.seen.add(tag)
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
        else:
            self.stack.pop()


def assert_valid_html(html: str) -> _TagBalance:
    assert html.lstrip().lower().startswith("<!doctype html>")
    parser = _TagBalance()
    parser.feed(html)
    parser.close()
    assert parser.errors == []
    assert parser.stack == []
    assert {"html", "head", "body", "title", "style"} <= parser.seen
    return parser


def assert_self_contained(html: str) -> None:
    assert not re.search(r"<link\b", html, re.I)
    assert not re.search(r"<script[^>]*\bsrc=", html, re.I)
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", html, re.I)
    assert "@import" not in html


def assert_valid_markdown_tables(md: str) -> int:
    """Each table: header row, `|---|` separator, rows with a matching column count."""
    tables = 0
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if not re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*", line):
            continue
        tables += 1
        width = lines[i - 1].strip().strip("|").count("|")
        j = i + 1
        while j < len(lines) and lines[j].strip().startswith("|"):
            assert lines[j].strip().strip("|").count("|") == width, lines[j]
            j += 1
    return tables


@pytest.fixture
def bleed_result(m3_table):
    return run_audit(load(AUDIT_FIXTURES / "bleed.jsonl").records, m3_table)


@pytest.fixture
def empty_result(m3_table):
    return run_audit([], m3_table)


# ---------------------------------------------------------------- packaging


def test_templates_exist():
    assert (TEMPLATES / "report.html.j2").is_file()
    assert (TEMPLATES / "report.md.j2").is_file()


def test_dependencies_declared():
    pyproject = tomllib.loads((TEMPLATES.parents[3] / "pyproject.toml").read_text("utf-8"))
    deps = " ".join(pyproject["project"]["dependencies"]).lower()
    assert "duckdb" in deps
    assert "jinja2" in deps


# ---------------------------------------------------------------- markdown


def test_markdown_contents(bleed_result):
    md = render_markdown(bleed_result)
    assert md.lstrip().startswith("# ")
    assert "Total spend" in md
    assert "$1.75" in md
    # Priority Tier callout with token count.
    assert "Priority Tier" in md
    assert "550,000" in md
    # Each finding with USD savings.
    assert "sha256:repeat-a" in md and "$0.52" in md
    assert "req_cliff_over" in md and "$0.44" in md
    assert "req_bloated" in md
    assert "$0.96" in md  # total estimated savings
    # Concentration + offenders.
    assert "Top" in md and "offenders" in md.lower()
    assert "key_a" in md and "ws_2" in md and "summarise" in md
    assert assert_valid_markdown_tables(md) >= 2


def test_markdown_empty_is_clean(empty_result):
    md = render_markdown(empty_result)
    assert "$0.00" in md
    assert "No bleed findings" in md
    assert not re.search(r"(nan|inf|infinity)", md, re.I)  # no 0/0 artefacts
    assert_valid_markdown_tables(md)


def test_markdown_escapes_pipes_in_keys(m3_table):
    result = run_audit([UsageRecord(model="test-flat", input_tokens=1, feature="a|b")], m3_table)
    assert_valid_markdown_tables(render_markdown(result))


# ---------------------------------------------------------------- html


def test_html_contents(bleed_result):
    html = render_html(bleed_result)
    assert_valid_html(html)
    assert_self_contained(html)
    for needle in (
        "Total spend",
        "$1.75",
        "Priority Tier",
        "550,000",
        "sha256:repeat-a",
        "$0.52",
        "req_cliff_over",
        "$0.44",
        "req_bloated",
        "$0.96",
        "key_a",
        "ws_2",
    ):
        assert needle in html, needle
    assert "<table" in html


def test_html_empty_is_clean(empty_result):
    html = render_html(empty_result)
    assert_valid_html(html)
    assert_self_contained(html)
    assert "No bleed findings" in html
    assert "$0.00" in html


def test_html_autoescapes_user_values(m3_table):
    r = UsageRecord(model="test-flat", input_tokens=1_000, feature="<script>alert(1)</script>")
    html = render_html(run_audit([r], m3_table))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert_valid_html(html)
