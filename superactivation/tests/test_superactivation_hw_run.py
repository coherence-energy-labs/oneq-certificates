"""The real-hardware superactivation run's analysis (shield_family/hw_superact_run.py), checked on synthetic counts
BEFORE any submission -- the way R133's estimator bias was caught:
  - counts drawn from each unit's EXACT ideal outcome law recover the design's ideal F_e and the per-branch values,
    and agree with the 'predicted' column set to the ideal (delta ~ 0, model verdict holds);
  - a reversed bit order (the classic counts-key mistake) is caught: the estimate moves far off;
  - the ladder fit recovers a planted per-CZ depolarizing rate and converts it in process-infidelity units;
  - the copy bit's agreement with P's sector is measured from the counts.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "superactivation" / "shield_family"))
import hw_superact_run as R  # noqa: E402
import hw_two_use as H  # noqa: E402

M_TEST = 2


def ideal_counts(m, u, n, reverse=False):
    """the unit's exact ideal outcome law over its clbits, as qiskit count keys (clbit 0 rightmost)"""
    from qiskit.quantum_info import Statevector
    qc = H.circuit(m, u)
    bare = qc.remove_final_measurements(inplace=False)
    probs = Statevector(bare).probabilities()
    meas = {qc.find_bit(i.clbits[0]).index: qc.find_bit(i.qubits[0]).index for i in qc.data
            if i.operation.name == "measure"}
    nb = qc.num_clbits
    law: dict = {}
    for idx, p in enumerate(probs):
        if p < 1e-15:
            continue
        bits = [(idx >> meas[j]) & 1 for j in range(nb)]
        key = "".join(map(str, bits if reverse else reversed(bits)))
        law[key] = law.get(key, 0.0) + p
    return {k: int(round(p * n)) for k, p in law.items() if round(p * n) > 0}


def synthetic_record(n_per=3, shots=200_000, seed=1):
    rng = np.random.default_rng(seed)
    units, counts = [], {}
    for key, W in H.strata(M_TEST):
        name_s = R.stratum_name(key)
        for c in range(n_per):
            u = H.sample_in_stratum(M_TEST, key, rng)
            units.append((name_s, W / n_per, u))
    for w, u in H.erased_erased_units(M_TEST):
        units.append(("erased_erased", w, u))
    meta = []
    for i, (s, w, u) in enumerate(units):
        name = f"u{i:03d}__{s}"
        cnt = ideal_counts(M_TEST, u, shots)
        tot = sum(cnt.values())
        counts[name] = cnt
        ideal = H.ideal_expectation(M_TEST, u)
        meta.append(dict(name=name, stratum=s, weight=w, unit=R.unit_record(u), ideal=ideal, predicted=ideal,
                         predicted_se=0.0, exact=True, cz=0, n_bits=H.circuit(M_TEST, u).num_clbits,
                         hi=1.0 if u.branch == "erased_erased" else 2.0, shots=tot))
    lam0, eps = 0.97, 0.008
    ladder = []
    for d in R.LADDER_DEPTHS:
        lam = lam0 * (1 - eps) ** d
        for b in R.BASES:
            corr = -lam if b == "YY" else lam
            n = 10 ** 7
            even = int(round((1 + corr) / 2 * n))
            counts[f"ladder_d{d}_{b}"] = {"00": even, "01": n - even}
            ladder.append(dict(name=f"ladder_d{d}_{b}", depth=d, basis=b, predicted_correlator=corr))
    return dict(m=M_TEST, units=meta, shots=None, ladder=ladder, ladder_edge=[0, 1], ladder_edge_vendor_cz=2.0e-3,
                model_tolerance_F_e=R.MODEL_TOLERANCE_FE), counts, eps


def _analyse(rec, counts):
    return R.analyse(rec, counts)          # each synthetic unit registers its own (rounded) shot count


def test_ideal_counts_recover_the_ideal_design_and_the_model_verdict_holds():
    rec, counts, _ = synthetic_record()
    out = _analyse(rec, counts)
    want = sum(u["weight"] * u["ideal"] for u in rec["units"])
    assert out["measured"]["F_avg"] == pytest.approx(want, abs=2e-5)
    assert out["verdicts"]["delta_F_e"] == pytest.approx(0.0, abs=3e-5)
    assert out["verdicts"]["model_holds_at_scale"]
    assert out["measured"]["branches"]["kept_kept"] == pytest.approx(1.0, abs=1e-4)
    assert out["copy_bit_agreement"] == pytest.approx(1.0, abs=1e-9)        # ideal: l is the sector, always
    assert out["measured"]["sd_F_e"] > 0 and out["measured"]["shot_sd_F_e"] < 0.01


def test_a_reversed_bit_order_is_caught():
    rec, counts, _ = synthetic_record()
    bad = {k: ({kk[::-1]: v for kk, v in c.items()} if k.startswith("u") else c) for k, c in counts.items()}
    good = _analyse(rec, counts)["measured"]["F_e"]
    wrong = _analyse(rec, bad)["measured"]["F_e"]
    assert abs(good - wrong) > 0.05


def test_the_ladder_recovers_a_planted_cz_rate_in_process_infidelity_units():
    rec, counts, eps = synthetic_record(n_per=1)
    lad = _analyse(rec, counts)["ladder"]["measured"]
    assert lad["eps_per_cz"] == pytest.approx(eps, rel=2e-3)
    assert lad["cz_factor"] == pytest.approx((15 / 16 * eps) / (5 / 4 * 2.0e-3), rel=2e-3)


def test_the_allocation_is_neyman_with_a_floor_and_covers_every_stratum():
    import json
    prior = json.loads(R.PRIOR.read_text(encoding="utf-8"))
    alloc = R.allocation(prior)
    assert set(alloc) == {R.stratum_name(k) for k, _ in H.strata(R.M)}
    assert min(alloc.values()) >= R.MIN_PER_STRATUM
    assert abs(sum(alloc.values()) - R.N_KEPT) <= len(alloc) * R.MIN_PER_STRATUM
    units = R.design_units(prior)
    assert sum(w for _, w, _ in units) == pytest.approx(1.0, abs=1e-12)
    assert sum(1 for s, _, _ in units if s == "erased_erased") == 24
    assert math.isclose(sum(w for s, w, _ in units if s == "erased_erased"), 0.25, abs_tol=1e-12)
