"""Milestone 3 `tokenize audit <file>` command."""

import pytest
from typer.testing import CliRunner

from tests.conftest import AUDIT_FIXTURES, M3_PRICING
from tokenize_cli.cli import app
from tokenize_cli.pricing import PRICING_ENV_VAR

runner = CliRunner()
MD = "tokenize-audit.md"
HTML = "tokenize-audit.html"


@pytest.fixture(autouse=True)
def _m3_pricing(monkeypatch):
    monkeypatch.setenv(PRICING_ENV_VAR, str(M3_PRICING))


def audit(*args: str):
    return runner.invoke(app, ["audit", *args])


def test_audit_writes_markdown_and_html(tmp_path):
    result = audit(str(AUDIT_FIXTURES / "bleed.jsonl"), "--output-dir", str(tmp_path))
    assert result.exit_code == 0, result.output
    md = (tmp_path / MD).read_text(encoding="utf-8")
    html = (tmp_path / HTML).read_text(encoding="utf-8")
    for text in (md, html):
        assert "$1.75" in text
        assert "550,000" in text
        assert "sha256:repeat-a" in text
        assert "req_cliff_over" in text
    # Terminal summary.
    assert "$1.75" in result.output
    assert MD in result.output and HTML in result.output


def test_audit_defaults_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = audit(str(AUDIT_FIXTURES / "clean.jsonl"))
    assert result.exit_code == 0, result.output
    assert (tmp_path / MD).is_file()
    assert (tmp_path / HTML).is_file()
    assert "No bleed findings" in (tmp_path / MD).read_text(encoding="utf-8")


def test_audit_console_export(tmp_path):
    result = audit(str(AUDIT_FIXTURES / "console_aggregated.csv"), "--output-dir", str(tmp_path))
    assert result.exit_code == 0, result.output
    md = (tmp_path / MD).read_text(encoding="utf-8")
    assert "$17.25" in md
    assert "1,100,000" in md  # priority tokens
    assert "key_fast" in md


def test_audit_empty_file(tmp_path):
    result = audit(str(AUDIT_FIXTURES / "empty.jsonl"), "--output-dir", str(tmp_path))
    assert result.exit_code == 0, result.output
    md = (tmp_path / MD).read_text(encoding="utf-8")
    assert "$0.00" in md
    assert "No bleed findings" in md
    assert (tmp_path / HTML).is_file()


def test_audit_missing_file(tmp_path):
    result = audit("nope.jsonl", "--output-dir", str(tmp_path))
    assert result.exit_code == 1
    assert "no such file" in result.output
    assert not (tmp_path / MD).exists()


def test_audit_listed_in_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "audit" in result.output
