"""K-group codes (experiments/superactivation/general_codes/kgroup.py): one reduction for every code family.

K = 1 must BE oneq.superactivation, K = 2 must BE twogroup.py, and m = (1, ..., 1) -- every n-use code -- must BE the
dense model written from first principles (general_codes.py): <b|sigma_z|b> on the flagged qubits equals rho_b / 2,
and the flagged qubits carry no coherence. Each identity is also shown able to fail.
"""
import itertools
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GC = ROOT / "experiments" / "superactivation" / "general_codes"
sys.path.insert(0, str(GC))
sys.path.insert(0, str(ROOT / "src"))
import general_codes as gcm  # noqa: E402
import kgroup as kg  # noqa: E402
import twogroup as tg  # noqa: E402
from oneq import superactivation as sa  # noqa: E402


@pytest.mark.parametrize("a, b", [(2, 3), (3, 1), (1, 1)])
def test_two_groups_are_twogroup_py(a, b):
    X = tg.random_encoder(a, b, np.random.default_rng(a * 10 + b))
    br = list(tg.branches(a, b))
    assert len(br) == (a + 1) * (b + 1)
    for k1, k2, w in br:
        assert np.array_equal(kg.bob_states(X, (a, b), (k1, k2)), tg.bob_states(X, a, b, k1, k2)[0])
    assert [w for _, w in kg.branches((a, b))] == [w for _, _, w in br]


@pytest.mark.parametrize("n", [3, 5])
def test_one_group_is_the_symmetric_reduction(n):
    X = sa.random_encoder(n, np.random.default_rng(n))
    for k in range(n + 1):
        assert np.allclose(kg.bob_states(X, (n,), (k,)), sa.bob_states(X, n, k), atol=1e-15, rtol=0)


def test_the_group_order_is_visible():
    """negative control for the identities above: (3,2) is a different tensor order from twogroup's (2,3)"""
    X = tg.random_encoder(2, 3, np.random.default_rng(4))
    assert not np.allclose(kg.bob_states(X, (3, 2), (1, 1)), tg.bob_states(X, 2, 3, 1, 1)[0])


def _dense_blocks(X, n):
    """(rho_b / 2 predicted, dense diagonal block, max dense off-diagonal block) for every branch and observation"""
    ms = (1,) * n
    out, off = [], 0.0
    for z, sig in gcm.branch_states(X / 2, n).items():
        T = sig.reshape([2] + [2] * n + [2] + [2] * n)
        flagged = [q for q in range(n) if z[q]]
        dB = 2 ** (n - len(flagged))
        R = kg.bob_states(X, ms, z)
        for i, bs in enumerate(kg.observed(z)):
            bits = [bs[q] for q in flagged]
            for bits2 in itertools.product((0, 1), repeat=len(flagged)):
                idx = [slice(None)] * (2 * n + 2)
                for q, v, v2 in zip(flagged, bits, bits2, strict=True):
                    idx[1 + q] = v
                    idx[2 + n + q] = v2
                blk = T[tuple(idx)].reshape(2 * dB, 2 * dB)
                if list(bits2) == bits:
                    out.append((R[i] / 2, blk))
                else:
                    off = max(off, float(np.abs(blk).max()))
    return out, off


@pytest.mark.parametrize("n", [2, 3, 4])
def test_all_ones_is_the_dense_general_model(n):
    X = kg.random_encoder((1,) * n, np.random.default_rng(100 + n))
    pairs, off = _dense_blocks(X, n)
    assert len(pairs) == 3 ** n                       # sum_k C(n,k) 2^k branch-observation pairs
    assert max(float(np.abs(p - d).max()) for p, d in pairs) < 1e-15
    assert off == 0.0


def test_a_caller_at_another_flip_probability_does_not_change_the_dense_model():
    """onset_vs_p used to WRITE general_codes.P = p and never restore it: after any shield-family call at d != 2 in the
    same process, the dense model above silently ran at that d's p and the identity missed by 0.03 (the 2026-10-19
    wall, once the suite's worker order put a shield-family test first). The p is now passed, never written"""
    sys.path.insert(0, str(ROOT / "experiments" / "superactivation" / "shield_family"))
    import onset_vs_p as ovp
    p32 = 1 / (np.sqrt(32) + 1)
    X2 = ovp.random_encoder(2, np.random.default_rng(7))
    ovp.adjoint_states(ovp.decode(X2, 2, p32, iters=5)[0], 2, p32)
    assert gcm.P == 1 / (1 + np.sqrt(2))
    X = kg.random_encoder((1,) * 3, np.random.default_rng(103))
    pairs, _ = _dense_blocks(X, 3)
    assert max(float(np.abs(p - d).max()) for p, d in pairs) < 1e-15
    # the channel the caller asked for is the one it got: p32's Kraus operators, not the module default's
    assert np.allclose(gcm.noisy_kraus(p32)[2], np.sqrt(p32) * gcm.X @ np.diag([1.0, 0]))


def test_the_dense_comparison_can_fail():
    """negative control: the dense model on a DIFFERENT encoder must not match the reduction"""
    n = 3
    X = kg.random_encoder((1,) * n, np.random.default_rng(1))
    Y = kg.random_encoder((1,) * n, np.random.default_rng(2))
    pairs, _ = _dense_blocks(X, n)
    pairs_y, _ = _dense_blocks(Y, n)
    assert max(float(np.abs(p - d).max()) for (p, _), (_, d) in zip(pairs, pairs_y, strict=True)) > 1e-3


def test_power_decoders_reach_the_round2_value_through_the_general_reduction():
    """the Round-2 2-use encoder (H x H) sqrt(iSWAP) (1 x |0>) scores 0.6897047609 in the dense exact-decoder model"""
    s = 1 / np.sqrt(2)
    plus, minus = np.array([s, s]), np.array([s, -s])
    v0 = np.kron(plus, plus)
    v1 = (np.kron(minus, plus) + 1j * np.kron(plus, minus)) * s
    vec = np.concatenate([v0, v1])                    # sum_r |r> (x) E|r>, R first
    X = np.outer(vec, vec.conj())
    assert np.allclose(X.reshape(2, 4, 2, 4).trace(axis1=1, axis2=3), np.eye(2))     # an isometric encoder
    _, F = kg.decode(X, (1, 1), iters=4000)
    assert F == pytest.approx(0.6897047609, abs=2e-9)
