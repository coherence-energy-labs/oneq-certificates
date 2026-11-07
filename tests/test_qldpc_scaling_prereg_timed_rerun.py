"""Paper 2's production-cost re-run of the preregistered scaling campaign (2026-11-06, external audit 3, Paper A).

experiments/m0_prior_art/qldpc_scaling_prereg_timed_rerun.py re-ran the campaign under the bytes of record with
per-stage timers and wrote evidence/qldpc_scaling_prereg/costs/production_<dataset>.json. The paper's cost table is
read from those records, so these tests hold each record's summary to its own per-shot times by the stated quantile
rule, and its reproduction claim to the committed record file's bytes.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))
import qldpc_scaling_prereg_timed_rerun as tr  # noqa: E402

COSTS = ROOT / "evidence" / "qldpc_scaling_prereg" / "costs"


def test_nearest_rank_is_the_ceil_qn_th_smallest_and_the_lower_middle_median():
    q = tr.nearest_rank([10.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0])
    assert q == {"median": 5.0, "p90": 9.0, "p99": 10.0, "max": 10.0, "n": 10}
    assert tr.nearest_rank([3.0])["median"] == 3.0 and tr.nearest_rank([1.0, 2.0])["median"] == 1.0
    one_tail = tr.nearest_rank([1.0] * 99 + [100.0])
    assert one_tail["p99"] == 1.0 and one_tail["max"] == 100.0          # the 99th of 100 is not the outlier


def test_every_shipped_production_record_summarises_its_own_shots_and_reproduced_the_record_of_campaign():
    files = sorted(COSTS.glob("production_*.json"))
    assert len(files) == 8, [f.name for f in files]                       # one per preregistered dataset
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        assert d["schema"] == tr.SCHEMA, f.name
        assert d["summary_s"]["production"] == tr.nearest_rank(d["shots"]["production_wall_s"]), f.name
        assert d["summary_s"]["total"] == tr.nearest_rank(d["shots"]["total_wall_s"]), f.name
        rep = d["reproduction"]
        rec = ROOT / "evidence" / "qldpc_scaling_prereg" / f"{d['dataset']}.jsonl.gz"
        assert rep["records_byte_identical"] and rep["analysis_identical"] and rep["unresolved"] == 0, f.name
        assert rep["records_sha256_rerun"] == rep["records_sha256_of_record"] == \
            hashlib.sha256(rec.read_bytes()).hexdigest(), f.name
        lines = [x for x in gzip.decompress(rec.read_bytes()).decode("utf-8").splitlines() if x.strip()]
        assert rep["records"] == len(lines) == len(d["shots"]["production_wall_s"]), f.name
        assert set(d["limits"]) >= {"bnb_max_nodes", "tree_checker_b_timeout_s", "milp_time_limit_s"}, f.name


def test_a_record_serialises_deterministically_and_round_trips():
    recs = [{"b": 2, "a": [1, 2]}, {"c": None}]
    blob = tr._serialise(recs)
    assert blob == tr._serialise(recs)                                    # mtime 0: no timestamp in the gzip header
    assert [json.loads(x) for x in gzip.decompress(blob).decode().splitlines()] == recs
    assert tr._serialise(recs[::-1]) != blob
