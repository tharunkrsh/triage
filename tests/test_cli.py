from typer.testing import CliRunner

from tests.conftest import FIXTURES
from triage import __version__
from triage.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Audit Claude API spend" in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


CSV = FIXTURES / "ingest" / "console_usage.csv"


def test_ingest_preview():
    result = runner.invoke(app, ["ingest", str(CSV)])
    assert result.exit_code == 0
    assert "console-csv" in result.output
    assert "row 4" in result.output


def test_ingest_strict_fails():
    result = runner.invoke(app, ["ingest", str(CSV), "--strict"])
    assert result.exit_code == 1
    assert "row 4" in result.output


def test_ingest_missing_file():
    result = runner.invoke(app, ["ingest", "nope.jsonl"])
    assert result.exit_code == 1
    assert "no such file" in result.output
