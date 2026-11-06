"""Paper 4's family P_d against its origin (final referee, MAJOR 1, 2026-11-03).

The state of K. Horodecki, Pankowski, M. Horodecki and P. Horodecki (IEEE Trans. Inf. Theory 54, 2621 (2008),
quant-ph/0506203, Eq. "standform"; written out in Badziag et al., arXiv:1309.7992, App. A) is a mixture of two private
bits with X-forms X1 = W_U / ||W_U||_1 and X2 = W_U^Gamma / ||W_U^Gamma||_1, W_U = sum_ij U_ij |ij><ji|, Gamma the
partial transpose on the second shield, weights fixed by the two norms. Held here, as the paper's "Origin" paragraph
states it: Eq. (8) with a complex Hadamard u is that state with X1 built from u^dagger / sqrt d and X2 from conj(u) /
sqrt d -- one U = conj(u) / sqrt d when u is symmetric (Hadamard, Fourier), two transposed twists when it is not; and
the comparison can fail (one common U does not reproduce a non-symmetric u's state).
"""
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "superactivation" / "shield_family"))
import ppt_family as pf  # noqa: E402


def _w(V):
    d = V.shape[0]
    W = np.zeros((d * d, d * d), complex)
    for i in range(d):
        for j in range(d):
            W[i * d + j, j * d + i] = V[i, j]
    return W


def _sqrt_psd(M):
    w, V = np.linalg.eigh((M + M.conj().T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.conj().T


def hphh_state(U1, U2):
    """their Eq. "standform" on (A, B, A', B'): key basis |00>, |01>, |10>, |11>, X1 from U1, X2 = W_{U2}^Gamma"""
    d = U1.shape[0]
    W1 = _w(U1)
    W2 = _w(U2).reshape(d, d, d, d).transpose(0, 3, 2, 1).reshape(d * d, d * d)
    n1, n2 = np.linalg.svd(W1, compute_uv=False).sum(), np.linalg.svd(W2, compute_uv=False).sum()
    X1, X2, p1, p2 = W1 / n1, W2 / n2, n1 / (n1 + n2), n2 / (n1 + n2)
    Z = np.zeros_like(X1)
    return 0.5 * np.block([[p1 * _sqrt_psd(X1 @ X1.conj().T), Z, Z, p1 * X1],
                           [Z, p2 * _sqrt_psd(X2 @ X2.conj().T), p2 * X2, Z],
                           [Z, p2 * X2.conj().T, p2 * _sqrt_psd(X2.conj().T @ X2), Z],
                           [p1 * X1.conj().T, Z, Z, p1 * _sqrt_psd(X1.conj().T @ X1)]]), p2


def ours(u):
    d = u.shape[0]
    rho = pf.choi_state(u, 1 / (math.sqrt(d) + 1))                      # (R', R'', B', B'')
    return rho.reshape(2, d, 2, d, 2, d, 2, d).transpose(0, 2, 1, 3, 4, 6, 5, 7).reshape(4 * d * d, 4 * d * d)


NONSYM = pf.fourier(3)[[1, 2, 0]] * np.array([1, np.exp(0.7j), np.exp(-1.9j)])   # a non-symmetric complex Hadamard


@pytest.mark.parametrize("u", [pf.hadamard2(), pf.fourier(3), pf.fourier(4), pf.sylvester(2)],
                         ids=["hadamard2", "fourier3", "fourier4", "sylvester4"])
def test_eq8_is_the_hphh_state_for_a_symmetric_twist(u):
    d = u.shape[0]
    assert np.allclose(u, u.T)
    H, p2 = hphh_state(u.conj() / math.sqrt(d), u.conj() / math.sqrt(d))
    assert abs(p2 - 1 / (math.sqrt(d) + 1)) < 1e-12                     # their weights are the PPT point
    assert np.abs(ours(u) - H).max() < 1e-8                               # (eigh square roots: ~1e-9 noise)


def test_a_non_symmetric_twist_carries_transposed_twists():
    d = 3
    assert np.abs(NONSYM @ NONSYM.conj().T - d * np.eye(d)).max() < 1e-12 and not np.allclose(NONSYM, NONSYM.T)
    H, _ = hphh_state(NONSYM.conj().T / math.sqrt(d), NONSYM.conj() / math.sqrt(d))
    assert np.abs(ours(NONSYM) - H).max() < 1e-8
    # without a local unitary no single U among u, conj u, u^T, u^dagger reproduces it
    for U in (NONSYM, NONSYM.conj(), NONSYM.T, NONSYM.conj().T):
        assert np.abs(ours(NONSYM) - hphh_state(U / math.sqrt(d), U / math.sqrt(d))[0]).max() > 1e-2


def _diag_equivalence(u):
    """the paper's local equivalence: if conj(u_ij)/conj(u_ji) = r_i/r_j, then diag(sqrt r) on R'' and its conjugate on
    B'' (key-independent, one unitary per party) maps Eq. (8) onto the HPHH state with U = conj(u)/sqrt d; returns
    (how far the ratios are from factorizing, the distance after the map, the distance with the conjugate map)"""
    d = u.shape[0]
    Q = u.conj() / u.conj().T
    r = Q[:, 0]
    fact = np.abs(Q - r[:, None] / r[None, :]).max()
    H, _ = hphh_state(u.conj() / math.sqrt(d), u.conj() / math.sqrt(d))
    rho = ours(u)
    out = []
    for s in (np.sqrt(r), np.sqrt(r).conj()):
        V = np.kron(np.kron(np.eye(4), np.diag(s)), np.diag(s.conj()))          # (R' B') (x) R'' (x) B''
        out.append(np.abs(V @ rho @ V.conj().T - H).max())
    return fact, out[0], out[1]


def test_that_non_symmetric_twist_is_locally_equivalent_to_the_hphh_state():
    """the final referee's witness (2026-11-03): this u is not a new state, only a local relabelling of HPHH's, so the
    paper claims nothing new for it beyond the "only if"; the conjugate map is the control that must fail"""
    fact, after, control = _diag_equivalence(NONSYM)
    assert fact < 1e-12                                   # the ratios factorize: r_i / r_j
    assert after < 1e-8                                   # equal to HPHH's state after the product of local unitaries
    assert control > 1e-2                                 # the conjugate local unitary does not do it
    # and the state is PPT at the PPT point (Theorem 3), as the equivalence with HPHH's PPT state requires
    rho = ours(NONSYM)
    pt = rho.reshape(2, 2, 3, 3, 2, 2, 3, 3).transpose(0, 5, 2, 7, 4, 1, 6, 3).reshape(36, 36)
    assert np.linalg.eigvalsh((pt + pt.conj().T) / 2).min() > -1e-12
