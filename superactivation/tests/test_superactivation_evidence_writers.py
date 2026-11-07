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
