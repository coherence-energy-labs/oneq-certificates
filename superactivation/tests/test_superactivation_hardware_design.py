"""R143: the two-use superactivation protocol as hardware circuits (shield_family/hardware_design.py,
measured_decoder.py). Each claim the go/no-go rests on is checked here, and each check is shown able to fail:
  - P_d IS a family of randomized Clifford(+T) circuits: their mixture equals P_d's Choi state, and the explicit gate
    sequences equal the verified matrices;
  - Bob's measure-and-correct decoder (Pauli frame X^l Z^[product = -1]) recovers the logical qubit EXACTLY for every
    outcome, and dropping X^l (the first draft's bug) does not;
  - the whole circuit-level protocol reproduces the closed form F_e = (7 - 6p + 2p^2)/8 exactly;
  - the noise-budget model returns the ideal value at zero error and falls with the error rate.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "superactivation" / "shield_family"))
import hardware_design as hd  # noqa: E402
import measured_decoder as md  # noqa: E402


@pytest.mark.parametrize("m", [1, 2])
def test_p_d_is_a_mixture_of_cheap_unitaries(m):
    Us = hd.p_unitaries(m)
    assert len(Us) == 4 ** m + 2 ** m                     # d^2 sector-0 and d sector-1 unitaries
    assert sum(w for w, _, _ in Us) == pytest.approx(1.0, abs=1e-14)
    for _, _, U in Us:
        assert np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-12)
    assert hd.check_p_decomposition(m) < 1e-14


def test_a_wrong_sector_weight_breaks_the_decomposition():
    """negative control: swapping the sector weights (p <-> 1 - p) must not reproduce P_d"""
    import ppt_family as pf
    m, d = 1, 2
    p = 1 / (math.sqrt(d) + 1)
    J = 0
    for w, (kind, _), U in hd.p_unitaries(m):
        w2 = (p / d ** 2) if kind == "s0" else ((1 - p) / d)
        vec = np.zeros(4 * d * d, dtype=complex)
        for i in range(2 * d):
            vec[i * 2 * d:(i + 1) * 2 * d] = U[:, i]
        J = J + w2 * np.outer(vec, vec.conj())
    assert np.abs(J / (2 * d) - pf.choi_state(pf.sylvester(m), p)).max() > 1e-2


@pytest.mark.parametrize("m", [1, 2])
def test_the_explicit_gate_sequences_are_the_verified_unitaries(m):
    assert hd.check_apply_p(m) < 1e-12


@pytest.mark.parametrize("m", [1, 2])
def test_the_measure_and_correct_decoder_is_exact(m):
    rng = np.random.default_rng(m)
    psi = rng.normal(size=2) + 1j * rng.normal(size=2)
    psi /= np.linalg.norm(psi)
    worst, total = md.decode_all_outcomes(m, psi)
    assert total == pytest.approx(1.0, abs=1e-10)
    assert worst > 1 - 1e-10


def test_dropping_the_key_flip_breaks_the_decoder():
    """negative control: the first draft omitted X^l in the parity-1 sector; it must fail on some outcome"""
    psi = np.array([0.6, 0.8j])
    worst, _ = md.decode_all_outcomes(1, psi, with_x=False)
    assert worst < 0.9


def test_the_circuit_level_protocol_reproduces_the_closed_form():
    p = 1 / (math.sqrt(2) + 1)
    F, Fz = hd.simulate(1)
    assert F == pytest.approx((7 - 6 * p + 2 * p * p) / 8, abs=1e-12)
    assert Fz[(0, 0)] == pytest.approx(1.0, abs=1e-12)
    assert Fz[(0, 1)] == pytest.approx(1 - p, abs=1e-12) == Fz[(1, 0)]
    assert Fz[(1, 1)] == pytest.approx(0.5 - p * (1 - p), abs=1e-12)


def test_the_budget_model_is_ideal_at_zero_and_falls_with_noise():
    p = 1 / 9
    Fid = hd.ideal_branches(p)
    ncz = {(0, 0): 40.0, (0, 1): 25.0, (1, 1): 8.0}
    vals = [hd.budget(Fid, ncz, e) for e in (0.0, 0.001, 0.003, 0.01)]
    assert vals[0] == pytest.approx((7 - 6 * p + 2 * p * p) / 8, abs=1e-15)
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert vals[-1] > 0.25                                 # never below a fully depolarized logical qubit
