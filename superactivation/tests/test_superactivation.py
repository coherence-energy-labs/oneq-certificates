"""The reduced superactivation picture against a brute-force 2^n-dimensional construction.

`oneq.superactivation.bob_states` claims: for an encoder on Sym^n, after k flagged uses of N~ and
a Z measurement of the flagged qubits, the state on R (x) (unflagged qubits) summed over all
flag-qubit outcomes of Hamming weight b is rho_{k,b}, supported on R (x) Sym^{n-k}. Here that is
rebuilt WITHOUT the reduction: the encoder is lifted to an explicit operator on R (x) (C^2)^n,
the flagged channel is applied qubit by qubit as its Kraus operators, the k qubits are projected
on every computational string, and the remainder is compressed onto Dicke states only at the end.
The test can fail on any of: the Dicke amplitude ratio, the binomial weights, the bit-flip
transition, the support claim, or the reference ordering.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from oneq import superactivation as sa


def dicke(n: int, w: int) -> np.ndarray:
    v = np.zeros(2 ** n)
    for z in itertools.combinations(range(n), w):
        idx = sum(1 << (n - 1 - q) for q in z)        # qubit 0 is the most significant bit
        v[idx] = 1.0
    return v / math.sqrt(math.comb(n, w))


def lift(X: np.ndarray, n: int) -> np.ndarray:
    """Encoder Choi on R (x) Sym^n -> explicit operator on R (x) (C^2)^n."""
    V = np.stack([dicke(n, w) for w in range(n + 1)], axis=1)      # 2^n x (n+1)
    K = np.kron(np.eye(2), V)
    return K @ X @ K.T


def flagged_branch(rho_full: np.ndarray, n: int, k: int, p: float) -> np.ndarray:
    """Apply Z-measure-then-bitflip(p) to qubits 0..k-1 of an operator on R (x) (C^2)^n."""
    dim = 2 ** n
    out = rho_full
    I2 = np.eye(2)
    P0, P1 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
    Xg = np.array([[0.0, 1.0], [1.0, 0.0]])
    kraus1 = [math.sqrt(1 - p) * P0, math.sqrt(1 - p) * P1, math.sqrt(p) * Xg @ P0, math.sqrt(p) * Xg @ P1]
    for q in range(k):
        new = np.zeros_like(out)
        for K1 in kraus1:
            ops = [I2] * n
            ops[q] = K1
            Kq = ops[0]
            for o in ops[1:]:
                Kq = np.kron(Kq, o)
            Kfull = np.kron(np.eye(2), Kq)
            new += Kfull @ out @ Kfull.conj().T
        out = new
    assert out.shape == (2 * dim, 2 * dim)
    return out


def brute_states(X: np.ndarray, n: int, k: int, p: float) -> list[np.ndarray]:
    full = flagged_branch(lift(X, n), n, k, p)
    rest = n - k
    Vr = np.stack([dicke(rest, w) for w in range(rest + 1)], axis=1) if rest else np.ones((1, 1))
    out = [np.zeros((2 * (rest + 1), 2 * (rest + 1)), dtype=complex) for _ in range(k + 1)]
    for y in itertools.product((0, 1), repeat=k):
        b = sum(y)
        ystr = np.zeros(2 ** k)
        ystr[int("".join(map(str, y)), 2) if k else 0] = 1.0
        proj = np.kron(np.eye(2), np.kron(ystr[None, :], np.eye(2 ** rest)))     # <y| on flagged
        cond = proj @ full @ proj.T
        # support claim: nothing outside R (x) Sym^{rest}
        Pr = np.kron(np.eye(2), Vr)
        compressed = Pr.T @ cond @ Pr
        back = Pr @ compressed @ Pr.T
        assert np.max(np.abs(back - cond)) < 1e-11, "conditional state leaves R (x) Sym^{n-k}"
        out[b] += compressed
    return out


@pytest.mark.parametrize("n", [3, 5])
def test_reduced_states_equal_the_brute_force_construction(n):
    rng = np.random.default_rng(100 + n)
    X = sa.random_encoder(n, rng, rank=2)
    for k in range(n + 1):
        red = sa.bob_states(X, n, k)
        bru = brute_states(X, n, k, sa.P_FLIP)
        for b in range(k + 1):
            assert np.max(np.abs(red[b] - bru[b])) < 1e-12, (n, k, b)


def test_the_reduction_preserves_trace_branch_by_branch():
    n = 6
    X = sa.random_encoder(n, np.random.default_rng(7), rank=3)
    for k in range(n + 1):
        assert np.real(np.trace(sa.bob_states(X, n, k).sum(axis=0))) == pytest.approx(2.0)


def test_the_adjoint_is_the_adjoint():
    n, k = 7, 3
    rng = np.random.default_rng(3)
    X = sa.random_encoder(n, rng)
    d = n - k + 1
    Ds = np.stack([sa.herm(rng.normal(size=(2 * d, 2 * d)) + 1j * rng.normal(size=(2 * d, 2 * d)))
                   for _ in range(k + 1)])
    lhs = sum(np.trace(r @ D) for r, D in zip(sa.bob_states(X, n, k), Ds))
    rhs = np.trace(X @ sa.adjoint(Ds, n, k))
    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


def test_power_decoders_are_feasible_and_bounded_by_the_trivial_values():
    n = 8
    X = sa.random_encoder(n, np.random.default_rng(11))
    decs, F, per_k = sa.decode_power(X, n, iters=30)
    for k, D in decs.items():
        d = n - k + 1
        assert np.max(np.abs(sa.ptrace_ref(D, d) - np.eye(d))) < 1e-9          # unital
        assert np.linalg.eigvalsh(D).min() > -1e-9                               # CP
    # any code beats 1/4 (output the maximally mixed state) and no code beats 1
    assert 0.25 <= F <= 1.0
    # k = n: Bob holds only classical bits, so the fidelity is that of a separable state <= 1/2
    assert per_k[n] <= 0.5 + 1e-12


def test_the_refusals():
    with pytest.raises(sa.SuperactivationRefusal):
        sa.check_encoder(np.eye(4), 3)                              # wrong shape
    with pytest.raises(sa.SuperactivationRefusal):
        sa.check_encoder(np.eye(8), 3)                              # Tr_Sym = 4 I, not I
    with pytest.raises(sa.SuperactivationRefusal):
        sa.depolarizing_budget(0.74)
    assert sa.depolarizing_budget(0.750138) == pytest.approx(0.000138 / 0.500138)


def test_each_encoder_defect_is_refused_by_name():
    n = 3
    X = sa.random_encoder(n, np.random.default_rng(2))
    sa.check_encoder(X, n)                                          # the honest encoder passes
    for bad, reason in [(X * np.nan, "non-finite"),
                        (X + 1e-3 * np.triu(np.ones_like(X), 1), "not Hermitian"),
                        (X - 2.0 * np.eye(2 * (n + 1)), "not positive semidefinite")]:
        with pytest.raises(sa.SuperactivationRefusal, match=reason):
            sa.check_encoder(bad, n)
    with pytest.raises(sa.SuperactivationRefusal, match="not positive definite"):
        sa.normalize_encoder(np.zeros((2 * (n + 1), 2 * (n + 1))), n)


def test_the_channel_tables_are_stochastic_and_refuse_impossible_indices():
    for k in range(6):
        T = sa.flip_transition(k)
        assert np.allclose(T.sum(axis=0), 1.0) and T.min() >= 0       # each column is a distribution
    for k, p in [(-1, 0.4), (3, 1.5), (3, -0.1)]:
        with pytest.raises(sa.SuperactivationRefusal, match="binary symmetric"):
            sa.flip_transition(k, p)
    with pytest.raises(sa.SuperactivationRefusal, match="0 <= a <= k <= n"):
        sa.selection(3, 2, 3)                                       # more flagged ones than flags


def test_the_encoder_gradient_is_the_fidelity_functional():
    """F(X, D) = Tr(X G(D)) exactly: the see-saw's encoder half-step maximizes the right thing."""
    n = 7
    X = sa.random_encoder(n, np.random.default_rng(4))
    decs, F, _ = sa.decode_power(X, n, iters=20)
    G = sa.encoder_gradient(decs, n)
    assert float(np.real(np.trace(X @ G))) == pytest.approx(F, rel=1e-12, abs=1e-12)
    assert sa.fidelity(X, decs, n) == pytest.approx(F, rel=1e-12, abs=1e-12)


def test_resampling_is_the_identity_at_equal_length_and_valid_across_lengths():
    n = 6
    X = sa.random_encoder(n, np.random.default_rng(8))
    assert np.allclose(sa.resample_encoder(X, n, n), X, atol=1e-12)
    for n_to in (n + 1, n + 4, n - 2):
        sa.check_encoder(sa.resample_encoder(X, n, n_to), n_to)     # a valid encoder at the new n
