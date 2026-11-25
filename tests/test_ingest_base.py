import pytest

from tests.conftest import FIXTURES
from tokenize_cli.ingest import IngestError, detect_format, load
from tokenize_cli.ingest.base import flatten

DIR = FIXTURES / "ingest"


@pytest.mark.parametrize(
    "name, fmt",
    [
        ("sample.jsonl", "jsonl"),
        ("console_usage.csv", "console-csv"),
        ("console_usage_analytics.json", "console-json"),
        ("console_usage_legacy.json", "console-json"),
    ],
)
def test_detect_by_extension(name, fmt):
    assert detect_format(DIR / name) == fmt


def test_detect_by_content(tmp_path):
    jsonl = tmp_path / "log.txt"
    jsonl.write_text('{"model": "m"}\n')
    report = tmp_path / "report.txt"
    report.write_text('{\n  "data": []\n}\n')
    assert detect_format(jsonl) == "jsonl"
    assert detect_format(report) == "console-json"


def test_undetectable_raises(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    with pytest.raises(IngestError, match="--format"):
        load(p)


def test_format_override(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text('{"model": "m"}\n')
    assert load(p, fmt="jsonl").records[0].model == "m"


def test_unknown_format_and_missing_file(tmp_path):
    with pytest.raises(IngestError, match="unknown format"):
        load(DIR / "sample.jsonl", fmt="xml")
    with pytest.raises(IngestError, match="no such file"):
        load(tmp_path / "nope.jsonl")


def test_time_range_spans_period_end():
    span = load(DIR / "console_usage_analytics.json").time_range
    assert span is not None and span[0] < span[1]


def test_flatten():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}
