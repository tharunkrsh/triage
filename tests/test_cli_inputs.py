"""CLI behaviour for Claude Code logs and folder or glob inputs."""

import shutil

import pytest
from typer.testing import CliRunner

from tests.conftest import FIXTURES
from triage import cli
from triage.pricing import PRICING_ENV_VAR

SESSIONS = FIXTURES / "ingest" / "claude_code"
REPLAY = FIXTURES / "simulate" / "replay.jsonl"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _offline(tmp_path, monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", PRICING_ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(tmp_path)
    shutil.copytree(SESSIONS, tmp_path / ".claude" / "projects")


def _run(*args):
    return runner.invoke(cli.app, list(args))


@pytest.mark.parametrize("spec", ["~/.claude/projects", "~/.claude/projects/**/*.jsonl"])
def test_ingest_previews_a_folder_or_glob(spec):
    result = _run("ingest", spec)
    assert result.exit_code == 0, result.output
    assert "claude-code" in result.output
    assert "Files" in result.output and "claude-haiku-5-5" in result.output


def test_ingest_accepts_the_format_flag():
    result = _run("ingest", "~/.claude/projects", "--format", "claude-code")
    assert result.exit_code == 0, result.output


def test_audit_writes_reports_without_full_paths(tmp_path):
    result = _run("audit", "~/.claude/projects", "-o", "~/out")
    assert result.exit_code == 0, result.output
    assert "Audit: projects" in result.output
    report = (tmp_path / "out" / "triage-audit.md").read_text(encoding="utf-8")
    assert "demo-app" in report and "other-tool" in report
    assert str(tmp_path) not in report and "C:\\work" not in report


def test_audit_finds_the_cliff_request_from_a_subagent(tmp_path):
    result = _run("audit", "~/.claude/projects/**/agent-*.jsonl")
    assert result.exit_code == 0, result.output
    row = next(x for x in result.output.splitlines() if "cliff requests" in x)
    assert "1" in row.split("requests")[1]


def test_audit_reports_a_pattern_with_no_matches():
    result = _run("audit", "~/nothing/*.jsonl")
    assert result.exit_code == 1 and "no files match" in result.output


def test_simulate_dry_run_over_a_folder(tmp_path):
    (tmp_path / "replays").mkdir()
    shutil.copy(REPLAY, tmp_path / "replays" / "one.jsonl")
    result = _run("simulate", "replays", "-t", "haiku-5-5", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "Estimated Batch API spend" in result.output
    assert "one.jsonl: line 6" in result.output


def test_simulate_rejects_transcripts_with_a_reason():
    result = _run("simulate", "~/.claude/projects", "-t", "haiku-5-5", "--dry-run")
    assert result.exit_code == 1
    assert "transcripts can't be replayed" in result.output
    assert "no replayable tasks in projects" in result.output


def test_entry_point_leaves_globs_for_triage_to_expand(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "app", lambda **kwargs: calls.append(kwargs))
    cli.run()
    assert calls == [{"windows_expand_args": False}]
