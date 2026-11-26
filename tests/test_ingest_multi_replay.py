import json
import shutil

import pytest

from tests.conftest import FIXTURES
from triage.ingest import IngestError
from triage.ingest.jsonl import read_replays
from triage.ingest.replay import load_replays
from triage.simulate.batch import check_unique_task_ids

REPLAY = FIXTURES / "simulate" / "replay.jsonl"
SESSIONS = FIXTURES / "ingest" / "claude_code"


def _task(**extra):
    return {
        "model": "claude-sonnet-5-5",
        "messages": [{"role": "user", "content": "synthetic"}],
        "response": "ok",
        **extra,
    }


def _write(path, *objects):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(o) + "\n" for o in objects), encoding="utf-8")
    return path


def test_a_single_file_loads_as_before():
    assert load_replays(REPLAY) == read_replays(REPLAY)


def test_generated_ids_get_a_file_prefix_and_explicit_ids_do_not(tmp_path):
    _write(tmp_path / "a.jsonl", _task(), _task(task_id="named"))
    _write(tmp_path / "sub" / "a.jsonl", _task())
    tasks = load_replays(tmp_path).tasks
    assert [t.task_id for t in tasks] == ["a.jsonl:line-1", "named", "sub/a.jsonl:line-1"]
    check_unique_task_ids(tasks)


def test_duplicate_explicit_ids_are_rejected(tmp_path):
    _write(tmp_path / "a.jsonl", _task(task_id="same"))
    _write(tmp_path / "b.jsonl", _task(task_id="same"))
    with pytest.raises(ValueError, match="same"):
        check_unique_task_ids(load_replays(tmp_path).tasks)


def test_transcripts_and_non_jsonl_matches_are_skipped(tmp_path):
    shutil.copytree(SESSIONS, tmp_path / "sessions")
    _write(tmp_path / "a.jsonl", _task(), {"model": "m"})
    _write(tmp_path / "notes.txt", {})
    result = load_replays(f"{tmp_path}/**/*.*")
    assert [t.task_id for t in result.tasks] == ["a.jsonl:line-1"]
    messages = [i.message for i in result.issues]
    assert "skipped 3 files. Claude Code transcripts can't be replayed" in messages[0]
    locations = {i.location for i in result.issues}
    assert {"a.jsonl: line 2", "notes.txt"} <= locations


def test_a_single_transcript_yields_no_tasks_and_says_why():
    transcript = next(SESSIONS.rglob("*.jsonl"))
    result = load_replays(transcript)
    assert result.tasks == [] and "skipped 1 file." in result.issues[0].message


def test_unreadable_files(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(b"\xff\xfe\x00\x81\n")
    with pytest.raises(IngestError, match="bad.jsonl"):
        load_replays(bad)
    _write(tmp_path / "good.jsonl", _task())
    result = load_replays(tmp_path)
    assert len(result.tasks) == 1 and result.issues[0].location == "bad.jsonl"
