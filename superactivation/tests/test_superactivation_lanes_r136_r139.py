"""R136-R139 superactivation lanes: the scoring in each is tested on inputs whose answer is known, and each one's
committed evidence is rebuilt from the committed inputs.

  stored_maps_entrywise  (R136) block enumeration, the prediction's index arithmetic, the source-hash refusal
  erasure_sensitivity    (R136) tolerance(Fk): an analytic crossing, the FIRST of three, a refusal below 3/4
  ppt_relaxation_probe   (R137) the channel Choi matrices and their n-fold assembly (the SDP needs cvxpy)
  inspect_witness        (R139) encoder isometry reading, Kraus / support ranks, against its committed log
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
import pathlib
import re
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUP = ROOT / "experiments" / "superactivation"
EV = ROOT / "evidence" / "superactivation"
sys.path.insert(0, str(ROOT / "src"))
from oneq import superactivation as sa  # noqa: E402
from oneq import superactivation_certificate as sc  # noqa: E402


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SM = _load("stored_maps_entrywise", SUP / "stored_maps_entrywise.py")
ES = _load("erasure_sensitivity", SUP / "erasure_sensitivity.py")
PP = _load("ppt_relaxation_probe", SUP / "ppt_relaxation_probe.py")
IW = _load("inspect_witness", SUP / "compile" / "inspect_witness.py")


# ---------------------------------------------------------------- stored_maps_entrywise (R136)

@pytest.mark.parametrize("r", range(1, 12))
def test_irreps_enumerate_the_schur_weyl_decomposition_exactly(r):
    """sum over the listed j of dim V_J x multiplicity = 2^r, with J = r/2 - j: every irrep once, none missing"""
    js = SM.irreps(r)
    assert sorted(js) == list(range(r // 2 + 1)) and js[-1] == 0
    total = sum((r - 2 * j + 1) * (math.comb(r, j) - (math.comb(r, j - 1) if j else 0)) for j in js)
    assert total == 2 ** r


def test_dropping_the_symmetric_irrep_breaks_the_dimension_count():
    """negative control for the count above: it can fail"""
    r = 6
    js = [j for j in SM.irreps(r) if j != 0]
    assert sum((r - 2 * j + 1) * (math.comb(r, j) - math.comb(r, j - 1)) for j in js) != 2 ** r


def test_block_enumeration_reproduces_the_committed_block_counts():
    rec = json.loads((EV / "stored_maps_entrywise.json").read_text(encoding="utf-8"))
    n = rec["n"]
    allb = [(k, b) for k in range(n + 1) for b in SM.blocks(n, k)]
    assert len(allb) == rec["blocks_compared"] == 330
    assert sum(1 for _, (_, jnk) in allb if jnk == 0) == rec["nonzero_predicted_blocks"] == 90


def test_the_committed_entrywise_check_passed_and_could_have_failed():
    rec = json.loads((EV / "stored_maps_entrywise.json").read_text(encoding="utf-8"))
    assert rec["verdict"] == "M_k = N_k o E entry by entry"
    assert rec["best_convention"]["max_abs_residual"] < rec["tolerance"] == 1e-12
    # the three other basis conventions are the witness that the comparison discriminates
    assert len(rec["other_conventions_max_abs_residual"]) == 3
    assert min(rec["other_conventions_max_abs_residual"]) > 1e-3


def test_predicted_blocks_fill_every_diagonal_slot_once_and_nothing_else():
    n, k = 6, 4
    X = sa.random_encoder(n, np.random.default_rng(5))
    rho = sa.bob_states(X, n, k)
    blocks = SM.blocks(n, k)
    assert len(blocks) == len(SM.irreps(k)) * len(SM.irreps(n - k)) == 3 * 2
    for jk, jnk in blocks:
        dk, dnk = k - 2 * jk + 1, (n - k) - 2 * jnk + 1
        P = SM.predicted(X, n, k, jk, jnk, rho, False, False).reshape(2, dk, dnk, 2, dk, dnk)
        if jnk != 0:
            assert not np.any(P)                         # an encoder on Sym^n leaves the unflagged uses symmetric
            continue
        for t, u in itertools.product(range(dk), repeat=2):
            slot = P[:, t, :, :, u, :]
            if t == u:
                b = t + jk
                assert np.allclose(slot.reshape(2 * dnk, 2 * dnk), rho[b] / math.comb(k, b), atol=0, rtol=0)
            else:
                assert not np.any(slot)


def test_the_flag_order_convention_is_a_reversal_of_the_same_blocks():
    n, k, jk = 5, 4, 1
    X = sa.random_encoder(n, np.random.default_rng(6))
    rho = sa.bob_states(X, n, k)
    dk, dnk = k - 2 * jk + 1, n - k + 1
    A = SM.predicted(X, n, k, jk, 0, rho, False, False).reshape(2, dk, dnk, 2, dk, dnk)
    B = SM.predicted(X, n, k, jk, 0, rho, True, False).reshape(2, dk, dnk, 2, dk, dnk)
    assert np.array_equal(B, A[:, ::-1, :, :, ::-1, :])
    assert not np.array_equal(A, B)                      # the conventions are distinguishable on this input


def test_a_source_archive_with_the_wrong_hash_is_refused(tmp_path, monkeypatch):
    fake = tmp_path / "2604.27042v2.tar.gz"
    fake.write_bytes(b"not the authors' archive")
    monkeypatch.setattr(sys, "argv", ["stored_maps_entrywise.py", "--source", str(fake), "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit, match="REFUSING: source sha256"):
        SM.main()
    assert not (tmp_path / "o.json").exists()


# ---------------------------------------------------------------- erasure_sensitivity (R136)

def test_tolerance_finds_the_analytic_crossing():
    """F_k = a - b k gives F(q) = a - b n q exactly (binomial mean), so the crossing is (a - 3/4) / (b n)"""
    n, a, b = 4, 0.86, 0.05
    t = ES.tolerance(np.array([a - b * k for k in range(n + 1)]))
    assert t["F_at_half"] == pytest.approx(0.76, abs=1e-15)
    assert t["q_where_F_reaches_three_quarters"] == pytest.approx(0.55, abs=1e-10)
    assert t["dF_dq_at_half"] == pytest.approx(-b * n, abs=1e-6)


def test_tolerance_reports_the_FIRST_crossing_not_any_crossing():
    """F(q) = 3/4 - c (q - .52)(q - .55)(q - .58) crosses three times; only .52 makes 'F > 3/4 below q*' true"""
    n, c = 3, 100.0
    qs = np.array([0.1, 0.4, 0.7, 0.9])
    target = 0.75 - c * (qs - 0.52) * (qs - 0.55) * (qs - 0.58)
    Bern = np.array([[math.comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(n + 1)] for q in qs])
    Fk = np.linalg.solve(Bern, target)
    assert sa.fidelity_at_erasure(Fk, 0.55) == pytest.approx(0.75, abs=1e-12)    # the middle root is real
    assert ES.tolerance(Fk)["q_where_F_reaches_three_quarters"] == pytest.approx(0.52, abs=1e-9)


def test_tolerance_is_none_when_the_code_never_falls_to_three_quarters():
    t = ES.tolerance(np.full(6, 0.8))
    assert t["q_where_F_reaches_three_quarters"] is None and t["tolerance_above_half"] is None


def test_a_code_below_three_quarters_has_no_tolerance_and_is_refused():
    with pytest.raises(sa.SuperactivationRefusal, match="refusing"):
        ES.tolerance(np.full(6, 0.7))


def test_the_committed_n17_sensitivity_is_rebuilt_from_its_witness():
    rec = json.loads((EV / "erasure_sensitivity.json").read_text(encoding="utf-8"))["codes"]["17"]
    n, X, decs = sc.load_witness(EV / "witness_n17.npz")
    Fk = sa.branch_fidelities(X, decs, n)
    assert np.allclose(Fk, rec["branch_fidelities"], atol=1e-12, rtol=0)
    t = ES.tolerance(Fk)
    assert t["F_at_half"] == pytest.approx(rec["F_at_half"], abs=1e-12)
    assert t["q_where_F_reaches_three_quarters"] == pytest.approx(rec["q_where_F_reaches_three_quarters"], abs=1e-9)
    assert 0 < t["tolerance_above_half"] < 1e-3                     # the n = 17 code survives only ~4e-4 of drift


# ---------------------------------------------------------------- ppt_relaxation_probe (R137)

def _ptrans_np(J, dA, dB):
    return J.reshape(dA, dB, dA, dB).transpose(0, 3, 2, 1).reshape(dA * dB, dA * dB)


def _tr_out(J, dA, dB):
    return np.einsum("aibi->ab", J.reshape(dA, dB, dA, dB))


@pytest.mark.parametrize("make", ["choi_id", "choi_noisy", "choi_erased"])
def test_each_single_use_choi_is_a_channel(make):
    J = getattr(PP, make)()
    assert np.linalg.eigvalsh(J).min() > -1e-12                     # CP
    assert np.allclose(_tr_out(J, 2, 2), np.eye(2))                 # TP


@pytest.mark.parametrize("make", ["choi_noisy", "choi_erased"])
def test_the_entanglement_breaking_branches_are_PPT(make):
    assert np.linalg.eigvalsh(_ptrans_np(getattr(PP, make)(), 2, 2)).min() > -1e-12


def test_the_identity_branch_is_NOT_PPT():
    """negative control: the PPT test above must be able to fail"""
    assert np.linalg.eigvalsh(_ptrans_np(PP.choi_id(), 2, 2)).min() == pytest.approx(-1.0)


def test_kron_branch_is_the_product_choi_in_A_then_Q_order():
    J1, J2 = PP.choi_noisy(), PP.choi_erased()
    T = PP.kron_branch([J1, J2])
    ref = np.einsum("aibj,ckdl->acikbdjl", J1.reshape(2, 2, 2, 2), J2.reshape(2, 2, 2, 2)).reshape(16, 16)
    assert np.allclose(T, ref, atol=0, rtol=0)
    assert np.allclose(_tr_out(T, 4, 4), np.eye(4))
    assert not np.allclose(PP.kron_branch([J2, J1]), T)            # the order is visible


def test_the_committed_probe_records_the_R137_kill():
    rec = json.loads((EV / "ppt_relaxation_probe.json").read_text(encoding="utf-8"))
    assert rec["sanity"]["identity"] == pytest.approx(1.0, abs=1e-6)
    assert rec["sanity"]["entanglement_breaking"] == pytest.approx(0.5, abs=1e-6)
    rows = {r["n"]: r for r in rec["rows"]}
    assert sorted(rows) == [1, 2]
    assert rows[2]["F_PPT_flagged"] > 0.75                          # the relaxation already passes 3/4 at n = 2
    assert rows[2]["F_PPT_flagged"] > rows[2]["unassisted_lower"]   # so it cannot certify an onset at 17


# ---------------------------------------------------------------- inspect_witness (R139)

def _isometry_encoder(n, rng):
    m = n + 1
    Q, _ = np.linalg.qr(rng.normal(size=(m, 2)) + 1j * rng.normal(size=(m, 2)))
    v = np.concatenate([Q[:, 0], Q[:, 1]])                          # |V> = sum_r |r> (x) V|r>, R first
    return np.outer(v, v.conj())


def test_an_exact_isometry_reads_as_rank_one_with_identity_gram():
    n = 7
    X = _isometry_encoder(n, np.random.default_rng(2))
    sa.check_encoder(X, n)
    s = IW.encoder_structure(X, n)
    assert s["top_eigenvalues"][0] == pytest.approx(2.0, abs=1e-12)
    assert abs(s["top_eigenvalues"][1]) < 1e-12
    assert np.allclose(s["gram"], np.eye(2), atol=1e-12)


def test_a_rank_two_encoder_does_not_pass_the_isometry_reading():
    """negative control: the Gram check above must be able to fail"""
    n = 7
    X = sa.random_encoder(n, np.random.default_rng(3), rank=2)
    s = IW.encoder_structure(X, n)
    assert s["top_eigenvalues"][1] > 1e-2
    assert np.abs(s["gram"] - np.eye(2)).max() > 1e-2


def test_numerical_rank_counts_a_planted_rank():
    rng = np.random.default_rng(4)
    for r in (1, 3, 5):
        K = rng.normal(size=(8, r)) + 1j * rng.normal(size=(8, r))
        assert IW.numerical_rank(K @ K.conj().T, 1e-6) == r


def _logged_ranks(n):
    text = (SUP / "compile" / "inspect_witness.log").read_text(encoding="utf-8")
    sec = text.split(f"=== n = {n} ")[1].split("=== n = ")[0]
    dec = {int(k): json.loads(v) for k, _, v in re.findall(r"k=\s*(\d+) d=\s*(\d+) ranks: (\[[^\]]*\])", sec)}
    sup_text = sec.split("restricted support")[1]
    sup = {int(k): json.loads(v) for k, v in re.findall(r"k=\s*(\d+) (\[[^\]]*\])", sup_text)}
    return dec, sup


def test_the_committed_inspect_log_is_rebuilt_from_the_n17_witness():
    n, X, decs = sc.load_witness(EV / "witness_n17.npz")
    dec_log, sup_log = _logged_ranks(17)
    assert sorted(dec_log) == sorted(sup_log) == list(range(18))
    assert IW.decoder_ranks(decs, n) == dec_log
    assert IW.support_ranks(X, n) == sup_log
    s = IW.encoder_structure(X, n)
    assert s["top_eigenvalues"][:2] == pytest.approx([1.99991714, 8.285e-05], abs=1e-8)
