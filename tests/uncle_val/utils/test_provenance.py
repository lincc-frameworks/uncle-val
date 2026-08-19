import json
import re
import sys

from uncle_val.utils.provenance import run_info, write_run_info

EXPECTED_KEYS = {"argv", "cwd", "started_at", "git_commit", "git_dirty"}


def test_run_info_keys_and_types():
    """run_info() reports the command line and the git state it ran from"""
    info = run_info()
    assert set(info) == EXPECTED_KEYS
    assert info["argv"] == sys.argv
    if info["git_commit"] is not None:
        assert re.fullmatch(r"[0-9a-f]{40}", info["git_commit"]), info["git_commit"]
        assert isinstance(info["git_dirty"], bool)


def test_write_run_info_round_trips(tmp_path):
    """write_run_info() writes the same mapping it returns, as readable JSON"""
    path = tmp_path / "run_info.json"
    written = write_run_info(path)
    assert json.loads(path.read_text()) == written
