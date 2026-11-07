"""Two of paper 4's evidence writers that no test exercised (tests/test_no_new_dark_evidence_writer.py named them
before the v2026.11.05 mint; fixed 2026-11-07).

* experiments/superactivation/general_codes/verify_reduction_dense.py, the dense cross-check of the group reduction
  (paper 4, Lemma 2): run here on a five-qubit two-group code over every flag pattern, and its negative controls must
  each be caught.
* experiments/superactivation/ppt_relaxation_exact.py, the first exact check of Proposition 11's feasible point (in
  the flag-block form; superseded by ppt_relaxation_fullspace.py, which assumes no block structure, but its record still
  ships): the feasible point at one use for both channels, and its control, must behave as the record says.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "superactivation" / "general_codes"))
sys.path.insert(0, str(ROOT / "experiments" / "superactivation"))


def test_the_dense_construction_agrees_with_the_group_reduction_on_every_flag_pattern_of_a_small_code():
    vrd = pytest.importorskip("verify_reduction_dense")
    out = vrd.case_all_patterns(2, 3, seed=7)
    assert out["flag_patterns_checked"] == 2 ** 5
    for key in ("max_identity_deviation", "max_certifier_selection_deviation", "max_residual_outside_symmetric",
                "max_state_deviation", "fidelity_abs_difference"):
        assert out[key] < 1e-12, (key, out[key])
    assert out["max_certifier_selection_radius"] < vrd.RADIUS_CAP
    assert 0.25 <= out["fidelity_reduction"] <= 1.0


def test_each_comparator_of_the_dense_check_catches_a_relative_one_in_a_million_error():
    vrd = pytest.importorskip("verify_reduction_dense")
    nc = vrd.negative_controls(3, 4, seed=11)
    assert nc["relative_size"] == 1e-6
    caught = {k: v for k, v in nc.items() if k != "relative_size"}
    assert len(caught) == 4
    for k, v in caught.items():
        assert v > 1e-10, (k, v)                 # each perturbation is seen, far above the 1e-12 agreement above


def test_the_shipped_dense_check_record_is_a_pass_with_its_controls_caught():
    rec = json.loads((ROOT / "evidence" / "superactivation" / "general_codes" / "reduction_dense_check_v2.json")
                     .read_text(encoding="utf-8"))
    assert rec["verdict"] == "AGREE" and rec["tolerance_exact_quantities"] == 1e-12
    every = [c for c in rec["cases"] if c.get("flag_patterns_checked") == 2 ** c.get("n", -1)]
    assert every and all(c["max_state_deviation"] < rec["tolerance_exact_quantities"] for c in every)
    nc = {k: v for k, v in rec["negative_controls"].items() if k != "relative_size"}
    assert len(nc) == 4 and all(v > rec["tolerance_exact_quantities"] for v in nc.values()), nc


def test_the_flag_block_feasible_point_has_value_three_quarters_at_one_use_for_both_channels():
    pre = pytest.importorskip("ppt_relaxation_exact")
    for choi in (pre.choi_noisy, pre.choi_erased):
        r = pre.check(1, choi)
        assert r["feasible"] and r["value_independent_of_p"] and r["equals_1_minus_2^-(n+1)"], r
        assert r["value"] == "3/4"
    bad = pre.check(1, pre.choi_noisy, halve=False)        # the kept block without its 1/2 is not feasible
    assert not bad["feasible"] and bad["first_violation"] is not None


def test_the_shipped_flag_block_record_says_what_the_check_gives():
    pre = pytest.importorskip("ppt_relaxation_exact")
    rec = json.loads((ROOT / "evidence" / "superactivation" / "ppt_relaxation_exact.json").read_text(encoding="utf-8"))
    assert rec["verdict"] == "EXACT" and rec["control_kept_block_without_one_half"]["feasible"] is False
    one_use = [r for r in rec["rows"] if r["n"] == 1]
    assert len(one_use) == 2
    for r in one_use:
        choi = pre.choi_noisy if r["channel"].startswith("flagged") else pre.choi_erased
        assert pre.check(1, choi)["value"] == r["value"] == "3/4"


# ---------------------------------------------------------------------------------------------------------------------
# phase_diagram.py's `symmetric` and `add-symmetric` modes (external audit 4, item 4.4, 2026-11-07): the d = 5 row of
# Table III had no n >= 5 entry because the original scan never reached it. The row was filled by the same function and
# parameters as the carried-forward symmetric rows, after re-running that function on every one of them.

SF_EV = ROOT / "evidence" / "superactivation" / "shield_family"


def _phase_diagram():
    sys.path.insert(0, str(ROOT / "experiments" / "superactivation" / "shield_family"))
    import phase_diagram as pd_mod
    return pd_mod


def test_the_d5_symmetric_row_is_bound_to_the_record_that_reproduced_every_carried_forward_row():
    pd_mod = _phase_diagram()
    table = json.loads((SF_EV / "phase_diagram.json").read_text(encoding="utf-8"))
    rec_path = SF_EV / "onset_search" / "symmetric_d5.json"
    raw = rec_path.read_bytes()
    rec = json.loads(raw)
    import hashlib
    sha = hashlib.sha256(raw).hexdigest()
    row = table["5"]
    sym = {n: e for n, e in row["entries"].items() if int(n) >= 5}
    assert sym and set(sym) == set(rec["F"])
    for n, e in sym.items():
        assert e["record"] == rec_path.name and e["record_sha256"] == sha and e["search"] == pd_mod.SYMMETRIC_SEARCH
        assert row["F"][n] == rec["F"][n]
    assert row["onset_upper_bound"] == rec["onset_upper_bound"]
    # the reproduction covered EVERY symmetric entry the table carries forward, and matched each to the last bit
    carried = {(d, n) for d in table for n in pd_mod.carried_symmetric(table, int(d))}
    done = {(d, int(n)) for d, rows in rec["reproduction"]["rows"].items() for n in rows}
    assert carried == done and len(done) >= 20
    assert rec["reproduction"]["reproduces"] and rec["reproduction"]["covers_every_carried_row"]
    assert rec["reproduction"]["max_abs_difference"] == 0.0
    for d, rows in rec["reproduction"]["rows"].items():
        for n, r in rows.items():
            assert r["now"] == r["recorded"] == table[d]["F"][n]
    assert rec["parameters"] == {"restarts": 3, "steps": 300, "seed": 13, "nmin": 5, "nmax": 17,
                                 "stops_at_first_crossing": True}


def test_add_symmetric_refuses_an_unreproduced_record_and_a_filled_row_and_leaves_other_rows_alone(tmp_path):
    pd_mod = _phase_diagram()
    table = json.loads((SF_EV / "phase_diagram.json").read_text(encoding="utf-8"))
    # a copy of the table with d = 5 as it was before item 4.4 (n <= 4 only, no onset)
    t5 = table["5"]
    t5["F"] = {n: v for n, v in t5["F"].items() if int(n) <= 4}
    t5["entries"] = {n: v for n, v in t5["entries"].items() if int(n) <= 4}
    t5.pop("family", None), t5.pop("onset_class", None)
    t5["onset_upper_bound"] = None
    tpath = tmp_path / "phase_diagram.json"
    tpath.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8", newline="\n")
    rec = {"schema": "oneq.superactivation.symmetric-onset/1", "d": 5, "search": pd_mod.SYMMETRIC_SEARCH,
           "F": {"5": 0.74, "6": 0.7501}, "reproduction": {"reproduces": False}}
    (tmp_path / "symmetric_d5.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(SystemExit, match="did not reproduce"):
        pd_mod.add_symmetric(5, tmp_path, tpath)
    rec["reproduction"]["reproduces"] = True
    (tmp_path / "symmetric_d5.json").write_text(json.dumps(rec), encoding="utf-8")
    before = {d: json.dumps(r) for d, r in json.loads(tpath.read_text(encoding="utf-8")).items() if d != "5"}
    row = pd_mod.add_symmetric(5, tmp_path, tpath)
    assert row["onset_upper_bound"] == 6 and row["family"] == "symmetric" and row["onset_class"] == "MEASURED-float"
    after = json.loads(tpath.read_text(encoding="utf-8"))
    assert {d: json.dumps(r) for d, r in after.items() if d != "5"} == before
    with pytest.raises(SystemExit, match="already holds"):         # a second insertion, or any filled row
        pd_mod.add_symmetric(5, tmp_path, tpath)
    rec["d"] = 6
    (tmp_path / "symmetric_d6.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(SystemExit, match="already holds"):
        pd_mod.add_symmetric(6, tmp_path, tpath)


def test_run_symmetric_reproduces_before_it_searches_and_stops_at_the_first_crossing(tmp_path, monkeypatch):
    pd_mod = _phase_diagram()
    table = {"6": {"F": {"5": 0.7498, "6": 0.7540}, "entries": {
        "5": {"search": pd_mod.SYMMETRIC_SEARCH, "provenance": "carried forward as recorded at R142"},
        "6": {"search": pd_mod.SYMMETRIC_SEARCH, "provenance": "carried forward as recorded at R142"}}}}
    tpath = tmp_path / "t.json"
    tpath.write_text(json.dumps(table), encoding="utf-8")
    calls = []
    truth = dict(table["6"]["F"])                        # what the function gives at d = 6

    def fake(n, p):                                      # d = 6 reproduces exactly; d = 5 crosses at n = 7
        calls.append((n, round(p, 6)))
        return truth[str(n)] if abs(p - 1 / (6 ** 0.5 + 1)) < 1e-12 else 0.7455 + 0.0025 * (n - 5)
    monkeypatch.setattr(pd_mod, "symmetric_best", fake)
    rec = pd_mod.run_symmetric(5, tmp_path, tpath, nmax=17)          # default: every row with carried entries
    assert rec["reproduction"]["reproduces"] and rec["reproduction"]["covers_every_carried_row"]
    assert rec["onset_upper_bound"] == 7 and list(rec["F"]) == ["5", "6", "7"]
    assert [c[0] for c in calls] == [5, 6, 5, 6, 7]       # the reproduction runs first
    table["7"] = {"F": {"5": 0.7600}, "entries": {"5": dict(table["6"]["entries"]["5"])}}
    tpath.write_text(json.dumps(table), encoding="utf-8")
    rec = pd_mod.run_symmetric(5, tmp_path, tpath, nmax=17, check=(6,))    # a list that leaves d = 7 out
    assert rec["reproduction"]["max_abs_difference"] == 0.0              # what it did run, it reproduced
    assert not rec["reproduction"]["covers_every_carried_row"] and not rec["reproduction"]["reproduces"]
    del table["7"]
    table["6"]["F"]["6"] = 0.7541                         # a carried value the function no longer gives
    tpath.write_text(json.dumps(table), encoding="utf-8")
    rec = pd_mod.run_symmetric(5, tmp_path, tpath, nmax=17)
    assert not rec["reproduction"]["reproduces"] and rec["reproduction"]["covers_every_carried_row"]
    with pytest.raises(SystemExit, match="no symmetric"):
        pd_mod.run_symmetric(5, tmp_path, tpath, check=(9,))
