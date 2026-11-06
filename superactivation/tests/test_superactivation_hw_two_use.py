"""The two-use superactivation protocol as hardware circuits (shield_family/hw_two_use.py), the first client of
oneq.hw. Checked exactly, each check able to fail:
  - the hardware design (cardinal inputs, Bob's classical frame, beta selection) reproduces the closed form
    F_e = (7 - 6p + 2p^2)/8 branch by branch, and a frame with a sign dropped does not;
  - the score stays in its declared range; the strata weights are the design's;
  - the ISA circuits the device would run, scored by oneq.hw.engine with the noise switched off, equal the
    independent statevector ideal of the logical circuits.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "superactivation" / "shield_family"))
import hw_two_use as H  # noqa: E402

from oneq.hw import engine as E  # noqa: E402
from oneq.hw import noise as N  # noqa: E402
from oneq.hw import snapshot as S  # noqa: E402


@pytest.mark.parametrize("m", [1, 2, 3])
def test_the_hardware_design_reproduces_the_closed_form_branch_by_branch(m):
    """against the closed form AND, branch by branch, the independent reference-qubit implementation"""
    import lean_protocol as lp
    F, per = H.ideal_f_e(m)
    assert F == pytest.approx(H.f_e_closed(m), abs=1e-12)
    p = H.p_flip(m)
    assert per["kept_kept"] == pytest.approx(1.0, abs=1e-12)
    _, ref = lp.simulate(m)
    for branch in ("kept_kept", "kept_erased", "erased_erased"):
        assert per[branch] == pytest.approx(ref[branch], abs=1e-12), branch
    assert 0.25 * (per["kept_kept"] + 2 * per["kept_erased"] + per["erased_erased"]) == pytest.approx(F, abs=1e-12)
    assert F == pytest.approx((7 - 6 * p + 2 * p * p) / 8, abs=1e-12)


@pytest.mark.parametrize("cmp", [H.Compile(fanout=True), H.Compile(fanin=True), H.Compile(fanout=True, fanin=True)],
                         ids=lambda c: c.name)
def test_every_compile_variant_is_an_exact_identity(cmp):
    """fan-out (CZs from the key's Z-copy) and fan-in (the parity of the pair bits computed coherently, 4 bits read)
    reproduce the closed form exactly; m = 2 exercises both (shield 1 takes the copy, one CNOT computes the parity)"""
    F, per = H.ideal_f_e(2, cmp)
    assert F == pytest.approx(H.f_e_closed(2), abs=1e-12)
    _, ref = H.ideal_f_e(2)
    for b in ref:
        assert per[b] == pytest.approx(ref[b], abs=1e-12), b
    u = H.Unit("kept_kept", ("s1", (0, 1)), 1, ("X", 1), (0, 0))
    assert H.circuit(2, u, cmp).num_clbits == (4 if cmp.fanin else 5)


def test_the_lean_reference_qubit_protocol_it_was_derived_from_is_exact_too():
    import lean_protocol as lp
    F, per = lp.simulate(1)
    assert F == pytest.approx(H.f_e_closed(1), abs=1e-12)
    assert per["kept_kept"] == pytest.approx(H.ideal_f_e(1)[1]["kept_kept"], abs=1e-12)


def test_a_frame_with_its_z_correction_dropped_is_caught(monkeypatch):
    real = H.score

    def no_parity(m, u, cmp=H.BASELINE):
        if u.branch == "erased_erased":
            return real(m, u, cmp)
        meas_basis, sign = H.BOB_BASIS[u.inp[0]]

        def kept(bits):
            ell, b1, o = bits[0], bits[m + 1], bits[m + 2]
            if ell != u.beta:
                return 0.0
            want = sign * u.inp[1] * (-1) ** ((ell ^ b1) if meas_basis in "ZY" else 0)
            return 2.0 if (1 - 2 * o) == want else 0.0
        return kept
    monkeypatch.setattr(H, "score", no_parity)
    F, _ = H.ideal_f_e(1)
    assert abs(F - H.f_e_closed(1)) > 0.05


def test_the_label_enumeration_is_p_unitaries_without_its_matrices():
    import hardware_design as hd
    for m in (1, 2, 3):
        mat = [(w, ch) for w, ch, _ in hd.p_unitaries(m)]
        lab = list(H.p_labels(m))
        assert [ch for _, ch in lab] == [ch for _, ch in mat]
        assert np.allclose([w for w, _ in lab], [w for w, _ in mat], atol=1e-15)
    # m = 9 would build 262,656 matrices of 1024 x 1024 through p_unitaries; the labels alone are instant
    assert sum(w for w, _ in H.p_labels(9)) == pytest.approx(1.0, abs=1e-12)
    assert len(list(H.erased_erased_units(9))) == 24


def test_design_weights_and_strata_weights_are_the_designs():
    for m in (1, 2, 5):
        assert sum(w for _, w in H.strata(m)) + H.BRANCH_WEIGHT["erased_erased"] == pytest.approx(1.0, abs=1e-12)
    assert sum(w for w, _ in H.design_units(2)) == pytest.approx(1.0, abs=1e-12)
    names = [u.name for _, u in H.design_units(2)]
    assert len(names) == len(set(names))


def test_scores_stay_in_their_declared_ranges():
    rng = np.random.default_rng(0)
    m = 3
    for key, _ in H.strata(m):
        u = H.sample_in_stratum(m, key, rng)
        sc = H.score(m, u)
        for x in range(2 ** (m + 3)):
            assert sc(tuple((x >> j) & 1 for j in range(m + 3))) in (0.0, 2.0)
    u = H.Unit("erased_erased", None, 0, ("Z", 1), (0, 0))
    assert {H.score(m, u)(b) for b in ((0, 0), (0, 1), (1, 1))} == {0.5}
    u = H.Unit("erased_erased", None, 0, ("X", -1), (0, 0))
    assert {H.score(m, u)(b) for b in ((0, 0), (0, 1))} == {0.0, 1.0}


def test_the_isa_circuits_scored_noiselessly_by_the_engine_equal_the_logical_ideal():
    from qiskit import transpile
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    be = FakeKingston()
    snap = S.from_backend(be)
    rng = np.random.default_rng(4)
    m = 2
    checked = 0
    for key, _ in H.strata(m)[:6]:
        u = H.sample_in_stratum(m, key, rng)
        tqc = transpile(H.circuit(m, u), be, optimization_level=3, seed_transpiler=3,
                        initial_layout=[2, 3, 4, 5, 6, 16, 23, 22][: 4 + 2 * m])
        r = E.expectation(tqc, H.score(m, u), snap, N.NOISELESS, 0, np.random.default_rng(0))
        assert r.exact
        assert r.mean == pytest.approx(H.ideal_expectation(m, u), abs=1e-9), u.name
        checked += 1
    assert checked == 6


def test_the_end_to_end_plan_freezes_verifies_and_is_refused_by_its_own_prediction():
    """hw_run.plan on FakeKingston: one circuit per stratum, frozen, re-loaded by hash, and the PREDICTED guard
    refuses it on the committed prediction for that same calibration"""
    import argparse
    import hw_run as R
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    from oneq.hw import freeze as F
    be = FakeKingston()
    snap = S.from_backend(be)
    P = R.plan(argparse.Namespace(m=5, C=1, S=100), be, snap)
    out = P["out"]
    assert out["n_circuits"] == len(R.all_strata(5)) == 13 and out["total_shots"] == 1300
    loaded = F.load(P["rec"], P["blob"], check_sources=False)
    assert list(loaded) == [u.name for u in P["stat_units"]]
    assert any(g.startswith("PREDICTED") for g in out["guards"])
    assert out["predicted_pass_probability"] < 1e-6 and out["verdict_if_submitted"].startswith("NO-GO")
    assert abs(sum(P["weights"].values()) - 1) < 1e-12


def test_the_prediction_is_the_ideal_minus_the_weighted_loss_and_refuses_a_partial_design():
    """hw_predict.combine: the arithmetic every committed hw_predict_*.json was produced by"""
    import hw_predict as HP
    m = 5
    strata = [("|".join(map(str, k)), w) for k, w in H.strata(m)] + [("erased_erased", 0.25)]
    zero = {n: dict(weight=w, mean_loss=0.0, se_loss=0.0, traj_se=0.0) for n, w in strata}
    out = HP.combine(m, zero)
    assert out["F_e_pred"] == pytest.approx(H.f_e_closed(m), abs=1e-12) and out["F_e_se"] == 0.0
    lossy = {n: dict(r, mean_loss=0.1, se_loss=0.02) for n, r in zero.items()}
    out = HP.combine(m, lossy)
    assert out["F_avg_pred"] == pytest.approx(out["F_avg_ideal"] - 0.1, abs=1e-12)   # weights sum to 1
    assert out["F_e_pred"] == pytest.approx(H.f_e_closed(m) - 0.15, abs=1e-12)       # F_e = (3 F_avg - 1)/2
    assert out["F_e_se"] == pytest.approx(1.5 * 0.02 * math.sqrt(sum(w * w for _, w in strata)), rel=1e-12)
    assert out["margin_pred"] == pytest.approx(out["F_e_pred"] - 0.75)
    del lossy["erased_erased"]
    with pytest.raises(ValueError):
        HP.combine(m, lossy)
    # the committed live prediction is reproduced from its own strata by this arithmetic
    import json
    rec = json.loads((ROOT / "evidence/superactivation/shield_family/hw_predict_m5_ibm_kingston_cz2.29.json")
                     .read_text(encoding="utf-8"))
    again = HP.combine(rec["m"], rec["strata"])
    assert again["F_e_pred"] == pytest.approx(rec["F_e_pred"], abs=1e-12)
    assert again["F_e_se"] == pytest.approx(rec["F_e_se"], abs=1e-12)


def test_selection_weight_two_makes_the_estimator_unbiased_for_any_device():
    """E[2 1[l = beta] c] over beta uniform = sum_l P(l) c(l): the device's l law can be anything"""
    for p_l1 in (0.0, 0.3, 1.0):
        c = {0: 0.9, 1: 0.6}
        est = sum(0.5 * 2 * ((1 - p_l1) if beta == 0 else p_l1) * c[beta] for beta in (0, 1))
        assert est == pytest.approx((1 - p_l1) * c[0] + p_l1 * c[1])
    # the ideal margin the hardware must keep at d = 32: 0.018 of F_e
    assert math.isclose(H.f_e_closed(5) - 0.75, 0.0179758, abs_tol=1e-7)
