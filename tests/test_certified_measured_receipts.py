"""Paper 1 Sec. VI E's receipts, kept (2026-11-03): the artifact is what it says, and its receipts replay.

External review 1 found the section's certificates "not kept, not replayed, not attacked". The re-run
(`experiments/dual_observable/certified_measured_receipts.py`) keeps them; this holds the artifact to its own claims:
it reproduces the lane's counts, its receipts hash to what it records, its instances hash to their `edges_sha256`, and
a sample of receipts re-decides identically under both Python verifiers.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys
from fractions import Fraction

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
DIR = ROOT / "evidence" / "certified_measured_receipts"
if not (DIR / "summary.json").is_file():
    pytest.skip("the Sec. VI E receipts have not been produced in this checkout", allow_module_level=True)
DOC = json.loads((DIR / "summary.json").read_text(encoding="utf-8"))
LANE = {r["record"]: r["models"]["measured"]
        for r in json.loads((ROOT / "evidence" / "certified_measured_lane.json").read_text(encoding="utf-8"))["runs"]}


def test_it_reproduces_the_lane_and_attacks_every_shot():
    assert len(DOC["runs"]) == 6
    for r in DOC["runs"]:
        assert (r["nontrivial"], r["exact_checker_accepted"]) == (LANE[r["record"]]["nontrivial"],
                                                                 LANE[r["record"]]["certified"])
        assert r["verifiers_agree_on"] == r["nontrivial"]
        assert r["forgeries_accepted"] == 0 and r["battery_attacks_not_false"] == 0
        assert r["forgeries_by_class"].get("D_spurious_cycle", 0) > 0     # the battery's heavier-edge-set class ran
    assert DOC["totals"]["nontrivial"] == sum(r["nontrivial"] for r in DOC["runs"])


@pytest.mark.parametrize("i", range(6))
def test_receipts_and_instance_hash_to_their_records(i):
    r = DOC["runs"][i]
    body = gzip.decompress((ROOT / r["receipts"]).read_bytes())
    assert hashlib.sha256(body).hexdigest() == r["receipts_sha256"]
    inst = json.loads((ROOT / r["instance"]).read_text(encoding="utf-8"))
    assert hashlib.sha256(json.dumps(inst["edges"], separators=(",", ":")).encode()).hexdigest() == \
        inst["edges_sha256"]
    lines = body.decode().splitlines()
    assert len(lines) == r["nontrivial"]


def test_a_sample_of_receipts_re_decides_identically():
    import replay_epsilon_receipts as rep
    checked = 0
    for r in DOC["runs"]:
        inst = json.loads((ROOT / r["instance"]).read_text(encoding="utf-8"))
        edges = [(int(u), int(v), float.fromhex(w)) for u, v, w in inst["edges"]]
        lines = gzip.decompress((ROOT / r["receipts"]).read_bytes()).decode().splitlines()
        for line in lines[:: max(1, len(lines) // 40)]:
            rec = json.loads(line)
            a, b = rep._decide_both(edges, rec, rep._receipt_duals(rec), Fraction(1, 10**6))
            assert a.status == b[0] == rec["status"]
            assert str(a.epsilon) == rec["epsilon"] and str(b[1]) == rec["epsilon"]
            checked += 1
    assert checked >= 200
