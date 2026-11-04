"""The differential CLI must fail on a disagreement in any input class."""
from __future__ import annotations

import json

import pytest

from tools import fuzz_epsilon_checkers as fuzz


@pytest.mark.parametrize("case_class,disagree,expected_exit", [
    ("base", False, 0),
    ("base", True, 1),
    ("id:boolean_in_correction", True, 1),
])
def test_cli_exit_status_tracks_all_reported_disagreements(
        monkeypatch, tmp_path, case_class, disagree, expected_exit):
    case = {"test_case": "controlled CLI sensitivity specimen"}
    correct = ("EXACT_OPTIMAL", "0", "1")
    changed = ("NOT_PROVEN", "1", "0")
    monkeypatch.setattr(fuzz, "build_cases", lambda *_args: [(case_class, case)])
    monkeypatch.setattr(fuzz, "add_eps_boundary", lambda cases: cases)
    monkeypatch.setattr(fuzz, "py_replay", lambda _case: (correct, correct))
    monkeypatch.setattr(fuzz, "py_raw", lambda _case: correct)
    monkeypatch.setattr(fuzz, "js_batch", lambda _cases: [changed if disagree else correct])
    output = tmp_path / "fuzz.json"

    result = fuzz.main(["--cases", "1", "--out", str(output)])

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cases"] == 1
    assert report["disagreeing_classes"] == ([case_class] if disagree else [])
    assert result == expected_exit
    assert report["valid_domain_disagreements"] == int(disagree and case_class == "base")
