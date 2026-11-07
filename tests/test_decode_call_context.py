"""decode_call_context.py (paper 1 Sec. IX, external audit 4, item 1.1): its arithmetic, and its record is its own.

The record explains why the cost paragraph's per-call decode (132 us mean) and the native-dual paragraph's (17 us
median) differ: the same call is timed in three contexts on the same shots. Held here: the edge-set comparison the
run refuses on, the summary's statistics and ratios on known lists, and the shipped record's internal arithmetic.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "m0_prior_art"))

import decode_call_context as dcc  # noqa: E402

REC = ROOT / "evidence" / "decode_call_context_d5_r25.json"


def test_edge_set_ignores_orientation_and_order_but_not_content():
    assert dcc.edge_set([(3, 1), (-1, 2)]) == dcc.edge_set([(2, -1), (1, 3)])
    assert dcc.edge_set([(3, 1)]) != dcc.edge_set([(3, 1), (2, -1)])
    assert dcc.edge_set([(3, 1)]) != dcc.edge_set([(3, 2)])


def test_summary_statistics_and_ratios_on_known_lists():
    us = {"after_certification": [100.0, 120.0, 110.0], "repeated": [30.0, 30.0, 30.0], "binding": [20.0, 10.0, 15.0]}
    s = dcc.summarise(us, certify_ms=[50.0, 60.0, 40.0], batch_us=[10.0, 20.0])
    d = s["decode_per_call_us"]
    assert (d["after_certification"]["median"], d["after_certification"]["mean"]) == (110.0, 110.0)
    assert d["binding"]["median"] == 15.0 and d["binding"]["max"] == 20.0
    assert s["ratio_mean_certify_over_mean_decode"]["repeated"] == pytest.approx(50_000.0 / 30.0)
    assert s["ratio_mean_certify_over_mean_batch_decode"] == pytest.approx(50_000.0 / 15.0)
    assert s["ratio_median_after_over_median_repeated"] == pytest.approx(110.0 / 30.0)


def test_the_shipped_record_is_its_own_arithmetic():
    assert REC.is_file(), "missing: evidence/decode_call_context_d5_r25.json"
    r = json.loads(REC.read_text(encoding="utf-8"))
    assert r["schema"] == "oneq-decode-call-context/1" and r["config"] == "sim_d5_r25"
    cost = json.loads((ROOT / "evidence" / "certify_cost_d5_r25.json").read_text(encoding="utf-8"))
    # the cost run's shots: its seed plan, graph and nontrivial count
    assert (r["seeds"], r["shots"], r["nontrivial"], r["edges"], r["detectors"]) == \
        (cost["seeds"], cost["shots"], cost["nontrivial"], cost["edges"], cost["detectors"])
    assert r["corrections_identical_across_arms"] == r["nontrivial"] == r["certify_ms"]["n"]
    for k in dcc.ARMS:
        s = r["decode_per_call_us"][k]
        assert s["n"] == r["nontrivial"] and 0 < s["median"] <= s["p90"] <= s["p99"] <= s["max"]
        assert r["ratio_mean_certify_over_mean_decode"][k] == pytest.approx(1e3 * r["certify_ms"]["mean"] / s["mean"])
    assert r["ratio_mean_certify_over_mean_batch_decode"] == pytest.approx(
        1e3 * r["certify_ms"]["mean"] / r["decode_batch_us_per_shot"])
    assert set(r["head"]["sha256"]) == set(dcc.JUDGED)
