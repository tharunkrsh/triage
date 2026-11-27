import json
import shutil

import pytest

from tests.conftest import FIXTURES
from triage.ingest import IngestError, detect_format, load, resolve_inputs
from triage.ingest.claude_code import ClaudeCodeAdapter

SAMPLE = FIXTURES / "ingest" / "sample.jsonl"
CONSOLE = FIXTURES / "ingest" / "console_usage.csv"
SESSIONS = FIXTURES / "ingest" / "claude_code"


def _touch(path, text='{"model": "claude-sonnet-5-5", "input_tokens": 1}\n'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _names(inputs):
    return [inputs.display(p) for p in inputs.files]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def test_a_file_is_returned_as_is_even_with_brackets(tmp_path):
    path = _touch(tmp_path / "log[1].jsonl")
    inputs = resolve_inputs(path)
    assert inputs.single and inputs.files == [path] and inputs.label == "log[1].jsonl"


def test_directory_is_searched_recursively_for_jsonl_only(tmp_path):
    for name in ("b.jsonl", "a/deep/c.NDJSON", ".hidden/d.jsonl", "e.csv", "f.txt"):
        _touch(tmp_path / "logs" / name)
    inputs = resolve_inputs(tmp_path / "logs")
    assert inputs.label == "logs" and not inputs.single
    assert _names(inputs) == [".hidden/d.jsonl", "a/deep/c.NDJSON", "b.jsonl"]


def test_glob_expands_double_star_through_dot_folders_and_takes_any_type(tmp_path):
    for name in ("x.csv", ".dot/y.csv", "z/z.csv", "z/skip.jsonl"):
        _touch(tmp_path / name)
    inputs = resolve_inputs(f"{tmp_path}/**/*.csv")
    assert _names(inputs) == [".dot/y.csv", "x.csv", "z/z.csv"]
    assert inputs.label == "*.csv"


def test_overlapping_matches_are_listed_once(tmp_path):
    _touch(tmp_path / "a.jsonl")
    assert len(resolve_inputs(f"{tmp_path}/**/**/*.jsonl").files) == 1


@pytest.mark.parametrize(
    "spec", ["~/.claude/projects", "~/.claude/projects/**/*.jsonl", "~/.claude/**/a.jsonl"]
)
def test_tilde_is_expanded_before_matching(home, spec):
    _touch(home / ".claude" / "projects" / "p" / "a.jsonl")
    assert [p.name for p in resolve_inputs(spec).files] == ["a.jsonl"]


@pytest.mark.parametrize(
    ("spec", "message"),
    [("nope.jsonl", "no such file"), ("*.nope", "no files match"), (".", "no .jsonl")],
)
def test_nothing_to_read_is_an_error(tmp_path, monkeypatch, spec, message):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(IngestError, match=message):
        resolve_inputs(spec)


def test_per_request_formats_combine_with_labelled_issues(tmp_path):
    shutil.copy(SAMPLE, tmp_path / "sample.jsonl")
    shutil.copytree(SESSIONS, tmp_path / "sessions")
    _touch(tmp_path / "bad.jsonl", '{"model": "claude-sonnet-5-5"}\nnot json\n')
    result = load(tmp_path)
    assert result.source_format == "claude-code, jsonl" and result.file_count == 5
    assert len(result.records) == len(load(SAMPLE).records) + 4 + 1
    assert "bad.jsonl: line 2" in [i.location for i in result.issues]


def test_unreadable_files_warn_unless_strict(tmp_path):
    shutil.copy(SAMPLE, tmp_path / "good.jsonl")
    (tmp_path / "binary.jsonl").write_bytes(b"\xff\xfe\x00\x81\n")
    result = load(tmp_path)
    assert result.file_count == 1 and result.records
    assert [i.location for i in result.issues] == ["binary.jsonl"]
    with pytest.raises(IngestError, match="binary.jsonl"):
        load(tmp_path, strict=True)


def test_all_files_unreadable_is_an_error(tmp_path):
    for name in ("a.bin", "b.bin"):
        _touch(tmp_path / name, "plain text\n")
    with pytest.raises(IngestError, match="none of the 2 files could be read"):
        load(f"{tmp_path}/*.bin")


def test_console_exports_cannot_be_mixed_with_request_logs(tmp_path):
    shutil.copy(SAMPLE, tmp_path / "sample.jsonl")
    shutil.copy(CONSOLE, tmp_path / "usage.csv")
    with pytest.raises(IngestError, match="counted twice"):
        load(f"{tmp_path}/*")
    assert load(f"{tmp_path}/*.csv").source_format == "console-csv"


def test_two_console_exports_combine(tmp_path):
    shutil.copy(CONSOLE, tmp_path / "a.csv")
    shutil.copy(CONSOLE, tmp_path / "b.csv")
    assert len(load(f"{tmp_path}/*.csv").records) == 2 * len(load(CONSOLE).records)


def test_format_applies_to_every_file(tmp_path):
    _touch(tmp_path / "a.txt", json.dumps({"model": "claude-sonnet-5-5"}) + "\n")
    _touch(tmp_path / "b.txt", json.dumps({"model": "claude-haiku-5-5"}) + "\n")
    assert len(load(f"{tmp_path}/*.txt", fmt="jsonl").records) == 2
    with pytest.raises(IngestError, match="unknown format"):
        load(tmp_path, fmt="nope")


def test_request_log_mentioning_data_is_not_a_console_export(tmp_path):
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}}
    line = {"model": "claude-sonnet-5-5", "input_tokens": 10, "output_tokens": 5}
    log = _touch(tmp_path / "img.jsonl", json.dumps({**line, "messages": [image]}) + "\n")
    assert detect_format(log) == "jsonl"
    assert len(load(log).records) == 1


def test_results_name_their_source_without_a_second_resolve(tmp_path):
    _touch(tmp_path / "logs" / "a.jsonl")
    assert load(tmp_path / "logs" / "a.jsonl").source == "a.jsonl"
    assert load(tmp_path / "logs").source == "logs"
    assert load(f"{tmp_path}/logs/*.jsonl").source == "*.jsonl"


def test_unreadable_session_log_is_not_counted(tmp_path):
    shutil.copytree(SESSIONS, tmp_path / "sessions")
    inputs = resolve_inputs(tmp_path / "sessions")
    missing = inputs.subset([*inputs.files, tmp_path / "sessions" / "gone.jsonl"])
    result = ClaudeCodeAdapter().read_set(missing)
    assert result.file_count == len(inputs.files)
    assert "gone.jsonl" in [i.location for i in result.issues]
    with pytest.raises(IngestError, match="gone.jsonl"):
        ClaudeCodeAdapter().read_set(missing, strict=True)


def test_a_single_unreadable_file_is_a_clean_error(tmp_path):
    log = tmp_path / "binary.jsonl"
    log.write_bytes(b"\xff\xfe\x00\x81\n")
    with pytest.raises(IngestError, match=r"can't read binary.jsonl: unreadable \(Unicode"):
        load(log, fmt="jsonl")
