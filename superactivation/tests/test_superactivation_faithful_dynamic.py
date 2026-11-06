"""R147: channel-faithful circuits, the one-CZ pair decoder, and P_d's sector 0 by measure-and-correct
(shield_family/hw_faithful.py, hw_dynamic.py, hw_levers.py). Each identity checked exactly, each able to fail:
  - faithful circuits (both uses encoded, P physical on both, erasure by RESET) reproduce the closed form, with either
    pair decoder, and a physical reset equals an untouched register;
  - the one-CZ decoders measure exactly the R143 decoder's observables, with one CZ; a dropped gate breaks it;
  - the measure-and-correct sector 0 has P_d's Choi state; a wrong correction (no Z on the key) does not;
  - the dynamic circuits (real mid-circuit measurement + if_test feed-forward) score the exact ideal -- computed
    exactly by branching (oneq.hw.dynamic_exact), not sampled under Aer: the suite's interpreter has no qiskit-aer,
    so these four checks were SKIPPED until 2026-10-16, dark on the circuits the IBM model test would run;
  - the exact stratum ideals and the both-erased branch sum to the closed form (the control variate's anchor).
"""
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "superactivation" / "shield_family"))
import hw_faithful as F  # noqa: E402
import hw_two_use as H  # noqa: E402


def test_faithful_circuits_reproduce_the_closed_form_with_both_decoders():
    for dec in ("cz2", "cz1", "mix"):
        fe, per = F.ideal_f_e(1, dec)
        assert fe == pytest.approx(H.f_e_closed(1), abs=1e-12), dec
        assert per["kept_kept"] == pytest.approx(1.0, abs=1e-12)


def _reset_gap(m, n_units, seed=4):
    """hw_faithful.reset_check's question, answered by exact branching instead of Aer's density matrix: the largest
    |E[score] with a physical RESET of the erased register - E[score] with the register left untouched|"""
    from oneq.hw import dynamic_exact as DX
    rng = np.random.default_rng(seed)
    worst, resets = 0.0, 0
    for branch in ("kept_erased", "erased_erased"):
        for _ in range(n_units):
            u = F.FUnit(branch, (F.hd.sample_choice(m, rng), F.hd.sample_choice(m, rng)), int(rng.integers(2)),
                        F.INPUTS[int(rng.integers(6))])
            for decoder in ("cz1", "cz2"):
                qc = F.circuit(m, u, decoder, erase="reset")
                resets += sum(i.operation.name == "reset" for i in qc.data)
                val = DX.distribution(qc).expectation(F.score(m, u, decoder))
                worst = max(worst, abs(val - F.ideal_expectation(m, u, decoder)))
    return worst, resets


def test_physical_reset_equals_an_untouched_register():
    worst, resets = _reset_gap(1, n_units=3)
    assert resets > 0                                   # the erasures really are physical resets
    assert worst < 1e-12


def _observable(qc_builder, beta, read):
    """U^dag (read) U for the pair circuit on (a, s), qubit order (a, s) with a most significant"""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(2)
    qc_builder(qc, 0, 1, beta)
    U = Operator(qc.reverse_bits()).data                    # a (qubit 0) most significant
    Z, I = np.diag([1.0, -1]), np.eye(2)
    D = {"a": np.kron(Z, I), "as": np.kron(Z, Z)}[read]
    return U.conj().T @ D @ U, qc.count_ops().get("cz", 0) + qc.count_ops().get("cx", 0) + qc.count_ops().get("ch", 0)


@pytest.mark.parametrize("beta", [0, 1])
def test_the_one_cz_decoder_measures_the_r143_observable(beta):
    ref, n_ref = _observable(lambda qc, a, s, b: F.pair_decode(qc, a, s, b, "cz2"), beta, "a")
    new, n_new = _observable(lambda qc, a, s, b: F.pair_decode(qc, a, s, b, "cz1"), beta, "as" if beta == 0 else "a")
    assert np.abs(new - ref).max() < 1e-12
    assert (n_ref, n_new) == (2, 1)

    def broken(qc, a, s, b):                                  # control: drop the decoder's first rotation
        F.pair_decode(qc, a, s, b, "cz1")
        del qc.data[0]
    bad, _ = _observable(broken, beta, "as" if beta == 0 else "a")
    assert np.abs(bad - ref).max() > 0.1


def test_no_local_unitary_mixture_gives_sector_zero():
    """Lam_t maps X_key (x) 1 to X_key Z_s / 2; any local-unitary mixture keeps the shield factor at the identity"""
    I2, X2, Y2, Z2 = np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1])
    CZ = np.diag([1, 1, 1, -1.0])
    W = [CZ, CZ @ np.kron(I2, Z2), np.kron(I2, X2), np.kron(Z2, Y2)]
    out = sum(w @ np.kron(X2, I2) @ w.conj().T for w in W) / 4
    assert np.abs(out - np.kron(X2, Z2) / 2).max() < 1e-12


@pytest.mark.parametrize("m", [1, 2])
def test_measure_and_correct_sector_zero_has_p_d_choi(m):
    import hw_dynamic as D
    assert D.locc_choi_error(m) < 1e-14


def test_measure_without_the_correction_is_not_p_d(monkeypatch):
    """control: the same measure-and-correct map with the key's Z dropped is a different channel"""
    import hw_dynamic as D
    real_kron = np.kron
    Z2 = np.diag([1.0, -1])

    def no_correction(a, b):
        if a.shape == (2, 2) and np.allclose(a, Z2) and np.allclose(b, np.diag([0.0, 1])):
            return real_kron(np.eye(2), b)
        return real_kron(a, b)
    monkeypatch.setattr(D.np, "kron", no_correction)
    assert D.locc_choi_error(1) > 1e-3


def test_dynamic_circuits_score_the_exact_ideal_exactly():
    """hw_dynamic.aer_check's question without sampling: each unit's dynamic circuit, on both schedules, scores the
    exact ideal of its unitary circuit to rounding -- where Aer at 4000 shots could only bound it by a z-score"""
    import hw_dynamic as D
    from oneq.hw import dynamic_exact as DX
    rng = np.random.default_rng(2)
    m, feed_forward = 2, 0
    for _ in range(6):
        key = H.strata(m)[int(rng.integers(len(H.strata(m))))][0]
        u = H.sample_in_stratum(m, ("kept_kept" if key[0] == "kept_kept" else "kept_erased", "s0", key[2], key[3]),
                                rng)
        for early in (True, False):
            qc = D.lean_dynamic(m, u, "cz2", early=early)
            feed_forward += sum(i.operation.name == "if_else" for i in qc.data)
            ncl = F.n_pair_bits(m, u, "cz2") + 3          # the protocol's bits; the mid-circuit register trails them
            got = DX.distribution(qc).expectation(F.score(m, u, "cz2"), nbits=ncl)
            assert got == pytest.approx(F.lean_ideal(m, u, "cz2"), abs=1e-12)
    assert feed_forward > 0


@pytest.mark.parametrize("m", [3, 5])
def test_exact_stratum_ideals_and_the_both_erased_branch_sum_to_the_closed_form(m):
    import hw_levers as V
    tot = sum(W * V.stratum_exact(m, k) for k, W in H.strata(m))
    ee = sum(w * H.ideal_expectation(m, u) for w, u in H.erased_erased_units(m))
    assert tot + ee == pytest.approx((2 * H.f_e_closed(m) + 1) / 3, abs=1e-12)


# ---------------------------------------------------------------- feed-forward: a test that can see it (review, R147)

def _exact_score(qc, score):
    """E[score] over every classical bit of qc, exactly (oneq.hw.dynamic_exact branches at each mid-circuit read)"""
    from oneq.hw import dynamic_exact as DX
    return DX.distribution(qc).expectation(score)


def _matched_dynamic_units(m, n, seed):
    """sector-0 units with beta = 0 (Bob's basis matches the copy bit, so the score is 0 or 2, never trivially 0), at
    least one shield read mid-circuit, and key0 read in X or Y -- the units where the feed-forward decides the outcome"""
    import hw_dynamic as D
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        branch = ("kept_kept", "kept_erased")[len(out) % 2]
        u = H.sample_in_stratum(m, (branch, "s0", 0, 0), rng)
        # Bob reads key0 in X or Y (inputs Z, Y): a missing Z on the key is visible there; in Z it commutes away
        if D.measure_type(u.choice) and H.BOB_BASIS[u.inp[0]][0] in "XY":
            out.append(u)
    return out


@pytest.mark.parametrize("early", [True, False])
def test_dynamic_circuits_score_their_ideal_and_dropping_the_feed_forward_is_caught(early):
    """real mid-circuit reads + if_test, both schedules: every matched unit scores its ideal 2 exactly (a matched
    noiseless unit is deterministic), and with the conditional Z removed every such unit scores ~1 (the reads' parity
    is a fair coin, and a Z on the key flips an X- or Y-basis read-out)"""
    import hw_dynamic as D
    caught = 0
    for u in _matched_dynamic_units(3, 4, seed=11):
        qc = D.lean_dynamic(3, u, "cz2", early=early, one_register=True)
        assert F.lean_ideal(3, u, "cz2") == pytest.approx(2.0, abs=1e-12)
        assert _exact_score(qc, F.score(3, u, "cz2")) == pytest.approx(2.0, abs=1e-12)
        broken = qc.copy()
        broken.data = [ins for ins in broken.data if ins.operation.name != "if_else"]
        if _exact_score(broken, F.score(3, u, "cz2")) < 1.9:
            caught += 1
    assert caught == 4, "the control must fail on every unit: the feed-forward carries the parity of the reads"


# ---------------------------------------------------------------- n-use codes

def test_n_use_formula_reduces_to_the_two_use_closed_form_and_beats_it_at_large_d():
    import hw_nuse as NU
    for m in range(1, 9):
        assert NU.f_closed(2, m) == pytest.approx(H.f_e_closed(m), abs=1e-15)
    assert NU.f_closed(3, 7) == pytest.approx(0.8382, abs=5e-5)
    assert NU.f_closed(3, 7) > NU.f_closed(2, 7) + 0.02
    assert NU.f_closed(3, 2) < NU.f_closed(2, 2)            # and loses at small d: the formula is not monotone in n


@pytest.mark.parametrize("n,m", [(3, 1), (4, 1), (3, 2)])
def test_n_use_circuits_reproduce_the_formula_and_a_wrong_frame_does_not(n, m):
    import hw_nuse as NU
    total = sum(W * NU.stratum_exact(n, m, k) for k, W in NU.strata(n, m)) + NU.all_erased_value(n, m)
    assert (3 * total - 1) / 2 == pytest.approx(NU.f_closed(n, m), abs=1e-12)
    # control: drop the other uses' frame bits from Bob's correction -> a different (lower) fidelity
    real = NU.score
    def no_frame(mm, u, decoder="cz2"):
        sc = real(mm, u, decoder)
        if u.e == u.n:
            return sc
        k = F.n_pair_bits(mm, u, decoder)
        return lambda bits: sc(tuple(bits[:1 + k]) + (0,) * (u.n - 1) + tuple(bits[k + u.n:]))
    NU.score, NU._EX = no_frame, {}
    try:
        bad = sum(W * NU.stratum_exact(n, m, k) for k, W in NU.strata(n, m)) + NU.all_erased_value(n, m)
    finally:
        NU.score, NU._EX = real, {}
    assert (3 * bad - 1) / 2 < NU.f_closed(n, m) - 0.05


# ---------------------------------------------------------------- the model-test run's guards and estimator

def _rec_and_blob():
    rp = sorted((ROOT / "evidence" / "superactivation" / "hw_dynamic_m5").glob("rehearsal_frozen_*.json"))
    if not rp:
        pytest.skip("no rehearsal freeze committed")
    import json
    return json.loads(rp[-1].read_text(encoding="utf-8")), rp[-1].with_suffix(".qpy").read_bytes()


def _covered_model_record(tmp_path, monkeypatch):
    """the rehearsal record with a model that covers every edge, stored AND written to a 'committed' file: the
    positive case every refusal below is a single change away from"""
    import json
    import hw_dynamic_run as R
    from oneq.hw import snapshot as S
    rec, blob = _rec_and_blob()
    vendor = S.DeviceSnapshot.from_json(rec["snapshot"])
    model = S.DeviceSnapshot(**{**vendor.__dict__, "cz_structure": {e: (1 / 15,) * 15 for e in vendor.cz}})
    fp = tmp_path / "repaired.json"
    fp.write_text(json.dumps(model.to_json()))
    monkeypatch.setattr(R, "tracked_in_repo", lambda path: pathlib.Path(path).resolve() == fp.resolve())
    good = dict(rec, model_snapshot=model.to_json(), model_snapshot_sha256=model.sha256(),
                model_snapshot_path=str(fp), cz_factor=R.REPAIRED_CZ_FACTOR, schedule_model="alap",
                idle_modelled=True)
    return R, good, blob


def test_a_fully_covered_committed_model_is_accepted(tmp_path, monkeypatch):
    R, good, blob = _covered_model_record(tmp_path, monkeypatch)
    assert R.submit_refusals(good, blob) == []


@pytest.mark.parametrize("change,needle", [
    (lambda g: dict(g, model_snapshot=None), "stores no repaired model"),
    (lambda g: dict(g, model_snapshot_sha256="0" * 64), "does not hash"),
    (lambda g: dict(g, model_snapshot_path=None), "does not name the committed file"),
    (lambda g: dict(g, model_snapshot_path="/nonexistent/model.json"), "not a committed"),
    (lambda g: dict(g, cz_factor=2.01), "cz_factor"),
    (lambda g: dict(g, schedule_model=None), "schedule"),
    (lambda g: dict(g, idle_modelled=False), "idle"),
])
def test_each_submit_precondition_refuses_on_its_own(tmp_path, monkeypatch, change, needle):
    R, good, blob = _covered_model_record(tmp_path, monkeypatch)
    why = R.submit_refusals(change(good), blob)
    assert why and needle in " ".join(why), why


def test_a_model_missing_any_edge_is_refused_even_when_self_consistent(tmp_path, monkeypatch):
    """the vendor snapshot stored as the 'model', hash recomputed, committed: no measured CZ channel -> refused"""
    import json
    from oneq.hw import snapshot as S
    R, good, blob = _covered_model_record(tmp_path, monkeypatch)
    vendor = S.DeviceSnapshot.from_json(good["snapshot"])
    fp = pathlib.Path(good["model_snapshot_path"])
    fp.write_text(json.dumps(vendor.to_json()))
    forged = dict(good, model_snapshot=vendor.to_json(), model_snapshot_sha256=vendor.sha256())
    why = R.submit_refusals(forged, blob)
    assert why and "no measured CZ channel" in " ".join(why)


def test_a_record_whose_model_differs_from_its_committed_file_is_refused(tmp_path, monkeypatch):
    import json
    R, good, blob = _covered_model_record(tmp_path, monkeypatch)
    fp = pathlib.Path(good["model_snapshot_path"])
    d = json.loads(fp.read_text())
    d["p01"] = [None if x is None else x * 0.5 for x in d["p01"]]
    fp.write_text(json.dumps(d))
    why = R.submit_refusals(good, blob)
    assert why and "is not the model this record stores" in " ".join(why)


def test_the_committed_rehearsal_record_cannot_be_submitted():
    import hw_dynamic_run as R
    rec, blob = _rec_and_blob()
    assert R.submit_refusals(rec, blob)
    assert R.submit_refusals(dict(rec, repaired_model=True, model_unmeasured_edges=[]), blob)


def test_a_live_freeze_without_a_repaired_model_and_schedule_is_refused_before_any_network(monkeypatch):
    import hw_dynamic_run as R
    monkeypatch.setattr(sys, "argv", ["hw_dynamic_run.py", "freeze", "--live", "ibm_kingston"])
    with pytest.raises(SystemExit, match="REFUSED"):
        R.main()


def test_fetch_takes_only_the_job_of_this_record(tmp_path):
    import json
    import hw_dynamic_run as R
    rec = {"qpy_sha256": "a" * 64}
    (tmp_path / f"{R.TAG}_old.job.json").write_text(json.dumps({"job_id": "old", "qpy_sha256": "b" * 64}))
    with pytest.raises(SystemExit, match="REFUSED"):
        R.job_for(rec, tmp_path)
    (tmp_path / f"{R.TAG}_mine.job.json").write_text(json.dumps({"job_id": "mine", "qpy_sha256": "a" * 64}))
    assert R.job_for(rec, tmp_path)["job_id"] == "mine"


def test_the_control_variate_returns_the_closed_form_on_ideal_means_and_moves_with_a_loss():
    import hw_dynamic_run as R
    rec, _ = _rec_and_blob()
    main = [u for u in rec["units"] if u["kind"] == "main"]
    ideal = {u["name"]: u["ideal"] for u in main}
    assert R.control_variate(ideal, main)["F_e"] == pytest.approx(H.f_e_closed(R.M), abs=1e-12)
    lossy = {n: v - 0.1 for n, v in ideal.items()}
    assert R.control_variate(lossy, main)["F_e"] == pytest.approx(H.f_e_closed(R.M) - 0.15, abs=1e-9)


def test_the_control_variate_sd_includes_the_both_erased_shot_noise():
    import hw_dynamic_run as R
    rec, _ = _rec_and_blob()
    main = [u for u in rec["units"] if u["kind"] == "main"]
    ideal = {u["name"]: u["ideal"] for u in main}
    zero = {u["name"]: 0.0 for u in main}
    shots = {u["name"]: 400 for u in main}
    noisy_ee = {n: (0.25 if "erased_erased" in n else 0.0) for n in ideal}
    base = R.control_variate(ideal, main, zero, shots)["sd_F_e"]
    with_ee = R.control_variate(ideal, main, noisy_ee, shots)["sd_F_e"]
    ee = [u for u in main if u["stratum"] == "erased_erased"]
    want = 1.5 * math.sqrt(base ** 2 / 2.25 + sum(u["weight"] ** 2 * 0.25 / 400 for u in ee))
    assert with_ee > base and with_ee == pytest.approx(want, rel=1e-12)


# ---------------------------------------------------------------- R153: the early/inline schedule must be IN the circuit

def test_early_and_inline_are_different_circuits_only_when_enforced_and_both_stay_exact():
    """R152's twins were one circuit twice (instruction order is not a dependency): the barrier makes them differ,
    and neither changes the ideal outcome"""
    from qiskit.converters import circuit_to_dag
    import hw_dynamic as D
    from oneq.hw import dynamic_exact as X
    units = _matched_dynamic_units(3, 3, seed=5)
    assert len(units) == 3                             # the loop below must not pass vacuously
    for u in units:
        e, i = (D.lean_dynamic(3, u, "cz2", early=x, one_register=True) for x in (True, False))
        assert circuit_to_dag(e) != circuit_to_dag(i)
        e0, i0 = (D.lean_dynamic(3, u, "cz2", early=x, one_register=True, enforce=False) for x in (True, False))
        assert circuit_to_dag(e0) == circuit_to_dag(i0), "the control: without the barrier they are identical"
        nb = F.n_pair_bits(3, u, "cz2") + 3
        sc = F.score(3, u, "cz2")
        for qc in (e, i):
            dist = X.distribution(qc)
            probs = dist.probabilities if hasattr(dist, "probabilities") else dist.probs
            val = sum(p * sc(tuple(int(b) for b in k)[:nb]) for k, p in probs.items())
            assert val == pytest.approx(F.lean_ideal(3, u, "cz2"), abs=1e-12)
