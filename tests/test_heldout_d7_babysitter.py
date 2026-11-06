"""The held-out d=7 run's restart-on-death supervisor (experiments/m0_prior_art/heldout_d7_babysitter.py), the bytes
that ran: it exits on the stop record, on completion, and past the deadline; it reads index lines only."""
from __future__ import annotations

import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

import heldout_d7_babysitter as bs  # noqa: E402


def _index(root, arm, seeds, with_status=False):
    p = root / "evidence" / "epsilon_rerun" / arm / "chunks.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"seed": s, **({"status": {"NOT_PROVEN": 50}} if with_status else {})} for s in seeds]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_completion_means_forty_indexed_chunks_in_every_arm(tmp_path):
    assert bs.complete(tmp_path) is False
    for arm in bs.ARMS:
        _index(tmp_path, arm, range(1, 41), with_status=True)   # statuses present and NOT read
    assert bs.complete(tmp_path) is True
    _index(tmp_path, bs.ARMS[-1], range(1, 40))
    assert bs.complete(tmp_path) is False


def test_it_exits_on_the_stop_record_without_touching_the_driver(tmp_path, monkeypatch):
    (tmp_path / "d7_stop_record.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["bs", str(tmp_path), "999999", "0"])
    assert bs.main() == 0
    log = (tmp_path / "d7_babysitter.log").read_text(encoding="utf-8")
    assert "stopping rule fired" in log and "restart" not in log


def test_it_exits_past_the_deadline_leaving_the_stop_to_the_watchdog(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bs", str(tmp_path), "999999", "0", "--budget-core-hours", "0"])
    assert bs.main() == 0
    assert "watchdog owns the stop" in (tmp_path / "d7_babysitter.log").read_text(encoding="utf-8")


def test_the_committed_bytes_are_the_ones_the_deviations_file_records():
    import hashlib
    import re
    doc = (ROOT / "docs" / "PREREG_heldout_d7_2026-11-03_DEVIATIONS.md").read_text(encoding="utf-8")
    m = re.search(r"heldout_d7_babysitter\.py`.{0,200}?sha256\s+([0-9a-f]{64})", doc, flags=re.S)
    assert m, "the deviations file does not record the babysitter's sha256"
    src = (ROOT / "experiments" / "m0_prior_art" / "heldout_d7_babysitter.py").read_bytes()
    assert hashlib.sha256(src.replace(b"\r\n", b"\n")).hexdigest() == m.group(1)
    # it never reads a verdict: no "status" access anywhere in its source
    tree = ast.parse(src)
    assert not any(isinstance(n, ast.Constant) and n.value == "status" for n in ast.walk(tree))
