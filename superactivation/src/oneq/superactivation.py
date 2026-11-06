r"""Finite-blocklength superactivation of quantum capacity: the reduced exact picture.

THE OBJECT. Parentin, Bergh, Datta, Wilde (arXiv 2604.27042) show that one joint use of a PPT
channel and a 50% erasure channel -- each of zero quantum capacity, each limited to entanglement
fidelity 3/4 at EVERY blocklength -- simulates the flagged qubit channel N~: with probability 1/2
the qubit passes untouched (flag 0); otherwise (flag 1) it is Z-measured and bit-flipped with
probability p = sqrt(2) - 1. Superactivation at blocklength n means F_c(N~^{(x)n}, 2) > 3/4; they
exhibit n = 17 with F = 0.750131 (float, see-saw).

THE REPRESENTATION. Their symmetric see-saw works in the Schur-Weyl block basis of the
permutation-invariant algebra (669 operator blocks up to 180 x 180 at n = 17). The stored n = 17
encoder is supported ENTIRELY on the symmetric subspace Sym^n (every other block ~1e-17), and for
an encoder on Sym^n the problem collapses exactly:

  * branch k of the flags (weight C(n,k)/2^n; by symmetry the k flagged uses come first): the k
    flagged qubits are classical after N~, so Bob measures them in Z without loss, and the
    observed Hamming weight b is a sufficient statistic;
  * conditioned on the true weight a of those k qubits, the n - k unflagged qubits are in
    Sym^{n-k}, with D^n_w -> sqrt(C(n-k, w-a)/C(n, w)) D^{n-k}_{w-a} per string.

So with X the encoder Choi on R (x) Sym^n (unnormalized: X >= 0, Tr_Sym X = I_R), Bob's
unnormalized state for (k, b) on R (x) Sym^{n-k} is

    rho_{k,b}(X) = sum_a c_{k,b,a} (I_R (x) P_a) X (I_R (x) P_a)^T,
    c_{k,b,a} = Pr(b | a) C(k, a),   P_a[i, i+a] = sqrt(C(n-k, i) / C(n, i+a)),

and F = sum_k C(n,k)/2^n sum_b F_D(rho_{k,b}) / 4, F_D the maximal recovery value
max { Tr(rho D) : D >= 0, Tr_R D = I } (the adjoint-decoder Choi convention, d = 2). Every
decoder problem is at most 2(n+1) square instead of 180: the ansatz is "encoder on Sym^n", the
decoding given it is exactly optimal (Lemma S5 / Corollary of arXiv 2604.27042 for the flags,
the support argument above for the rest). `tests/test_superactivation.py` holds rho_{k,b}
against a brute-force 2^n-dimensional construction.

This module is numpy-only: the reduced maps, their adjoint, trace-normalisation, and a batched
power-iteration decoder (Reimpell) used as a FAST PROPOSER. Exact decoders come from an SDP solver
in the producer lane; correctness is owned by `superactivation_certificate`, which never trusts
either.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

#: Bit-flip probability of the flagged branch: Choi diag (1-p, p, p, 1-p), p = 1/(1+sqrt 2).
P_FLIP = math.sqrt(2.0) - 1.0
#: The uniform bound: neither component channel exceeds this at any blocklength.
THRESHOLD = 0.75


class SuperactivationRefusal(ValueError):
    """An input that does not define the object the formulas are about."""


def herm(Z: np.ndarray) -> np.ndarray:
    return (Z + np.conj(np.swapaxes(Z, -1, -2))) / 2


@lru_cache(maxsize=None)
def flip_transition(k: int, p: float = P_FLIP) -> np.ndarray:
    """T[b, a] = Pr(observed Hamming weight b | true weight a), k independent bit flips."""
    if k < 0 or not 0.0 <= p <= 1.0:
        raise SuperactivationRefusal(f"k={k}, p={p} is not a binary symmetric channel")
    T = np.zeros((k + 1, k + 1))
    for a in range(k + 1):
        for f1 in range(a + 1):
            for f0 in range(k - a + 1):
                T[a - f1 + f0, a] += (math.comb(a, f1) * math.comb(k - a, f0)
                                      * p ** (f1 + f0) * (1 - p) ** (k - f1 - f0))
    return T


@lru_cache(maxsize=None)
def selection(n: int, k: int, a: int) -> np.ndarray:
    """I_R (x) P_a, shape (2(n-k+1), 2(n+1)): Dicke D^n_{i+a} -> D^{n-k}_i with its amplitude."""
    if not 0 <= a <= k <= n:
        raise SuperactivationRefusal(f"need 0 <= a <= k <= n, got a={a}, k={k}, n={n}")
    d = n - k + 1
    P = np.zeros((d, n + 1))
    for i in range(d):
        P[i, i + a] = math.sqrt(math.comb(n - k, i) / math.comb(n, i + a))
    return np.kron(np.eye(2), P)


def branch_coeffs(n: int, k: int, p: float = P_FLIP) -> np.ndarray:
    """C[b, a] = Pr(b | a) C(k, a)."""
    return flip_transition(k, p) * np.array([math.comb(k, a) for a in range(k + 1)])[None, :]


def check_encoder(X: np.ndarray, n: int, tol: float = 1e-8) -> None:
    m = n + 1
    if X.shape != (2 * m, 2 * m):
        raise SuperactivationRefusal(f"encoder Choi must be {2 * m} square for n={n}, got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise SuperactivationRefusal("encoder Choi has non-finite entries")
    if np.max(np.abs(X - herm(X))) > tol:
        raise SuperactivationRefusal("encoder Choi is not Hermitian")
    if np.linalg.eigvalsh(herm(X)).min() < -tol:
        raise SuperactivationRefusal("encoder Choi is not positive semidefinite")
    if np.max(np.abs(ptrace_sym(X, m) - np.eye(2))) > tol:
        raise SuperactivationRefusal("encoder is not trace preserving: Tr_Sym X != I_R")


def bob_states(X: np.ndarray, n: int, k: int, p: float = P_FLIP) -> np.ndarray:
    """Stacked rho_{k,b}, b = 0..k, each on R (x) Sym^{n-k} (R index first)."""
    C = branch_coeffs(n, k, p)
    S = np.stack([selection(n, k, a) @ X @ selection(n, k, a).T for a in range(k + 1)])
    return np.einsum("ba,aij->bij", C, S)


def adjoint(Ds: np.ndarray, n: int, k: int, p: float = P_FLIP) -> np.ndarray:
    """sum_b L_{k,b}^dag(D_b) on R (x) Sym^n, so that sum_b Tr(rho_{k,b}(X) D_b) = Tr(X adjoint)."""
    C = branch_coeffs(n, k, p)
    out = np.zeros((2 * (n + 1), 2 * (n + 1)), dtype=complex)
    for a in range(k + 1):
        Pa = selection(n, k, a)
        out += Pa.T @ np.einsum("b,bij->ij", C[:, a], Ds) @ Pa
    return out


def ptrace_ref(Z: np.ndarray, d: int) -> np.ndarray:
    """Tr_R of an operator on R (x) C^d (R first); works on stacks."""
    return np.einsum("...rirj->...ij", Z.reshape(Z.shape[:-2] + (2, d, 2, d)))


def ptrace_sym(Z: np.ndarray, d: int) -> np.ndarray:
    """Tr over the C^d factor of an operator on R (x) C^d."""
    return np.einsum("...risi->...rs", Z.reshape(Z.shape[:-2] + (2, d, 2, d)))


def normalize_encoder(X: np.ndarray, n: int) -> np.ndarray:
    """Congruence by T^{-1/2} (x) I so Tr_Sym X = I_R exactly up to rounding."""
    m = n + 1
    w, v = np.linalg.eigh(herm(ptrace_sym(X, m)))
    if w.min() <= 0:
        raise SuperactivationRefusal("encoder trace operator is not positive definite")
    Ti = np.kron((v / np.sqrt(w)) @ v.conj().T, np.eye(m))
    return herm(Ti @ X @ Ti)


def random_encoder(n: int, rng: np.random.Generator, rank: int = 2) -> np.ndarray:
    m = n + 1
    K = rng.normal(size=(2 * m, rank)) + 1j * rng.normal(size=(2 * m, rank))
    return normalize_encoder(K @ K.conj().T, n)


def power_step(R: np.ndarray, D: np.ndarray, d: int) -> np.ndarray:
    """One batched Reimpell power step for stacked states R and adjoint-decoder Chois D.

    The normalizer S = Tr_R(R D R) is singular whenever rho has low rank (it does: an encoder of
    Kraus rank r gives rank <= r(k+1)); its inverse square root is taken on the support only
    (relative threshold) and the decoder is completed by I_R/2 on the kernel, so Tr_R D = I holds
    on all of Sym^{n-k}. An absolute floor here made the first version blow up to 1e270."""
    G = herm(R @ D @ R)
    S = ptrace_ref(G, d)
    w, v = np.linalg.eigh(herm(S))
    keep = w > 1e-13 * np.maximum(w.max(axis=-1, keepdims=True), 1e-300)
    inv = np.where(keep, 1.0 / np.sqrt(np.where(keep, w, 1.0)), 0.0)
    vh = np.conj(np.swapaxes(v, -1, -2))
    Sih = (v * inv[..., None, :]) @ vh
    Pk = (v * keep[..., None, :]) @ vh
    I2 = np.eye(2)
    Si = np.einsum("rs,bij->brisj", I2, Sih).reshape(-1, 2 * d, 2 * d)
    comp = np.einsum("rs,bij->brisj", I2, np.eye(d)[None] - Pk).reshape(-1, 2 * d, 2 * d) / 2
    return herm(Si @ G @ Si + comp)


def decode_power(X: np.ndarray, n: int, warm: dict | None = None, iters: int = 60,
                 p: float = P_FLIP):
    """Feasible decoders for every (k, b) by batched power iteration; returns (decoders, F, F_k)."""
    decs, F, per_k = {}, 0.0, []
    for k in range(n + 1):
        d = n - k + 1
        R = bob_states(X, n, k, p)
        D = warm[k] if warm is not None else np.broadcast_to(
            np.eye(2 * d) / 2, (k + 1, 2 * d, 2 * d)).copy()
        for _ in range(iters):
            D = power_step(R, D, d)
        decs[k] = D
        per_k.append(float(np.real(np.einsum("bij,bji->b", R, D)).sum() / 4))
        F += math.comb(n, k) / 2 ** n * per_k[-1]
    return decs, F, per_k


def fidelity(X: np.ndarray, decs: dict, n: int, p: float = P_FLIP) -> float:
    """Entanglement fidelity of the pair (X, decoders) -- float, for the record, not a proof."""
    F = 0.0
    for k in range(n + 1):
        R = bob_states(X, n, k, p)
        F += math.comb(n, k) / 2 ** n * float(np.real(np.einsum("bij,bji->", R, decs[k]))) / 4
    return F


def branch_fidelities(X: np.ndarray, decs: dict, n: int, p: float = P_FLIP) -> np.ndarray:
    """F_k for k = 0..n: the code's fidelity conditioned on exactly k flagged uses."""
    return np.array([float(np.real(np.einsum("bij,bji->", bob_states(X, n, k, p), np.asarray(decs[k])))) / 4
                     for k in range(n + 1)])


def fidelity_at_erasure(Fk: np.ndarray, q: float) -> float:
    """The FIXED code's fidelity when each use is flagged with probability q instead of 1/2.

    The flag of N~ is the erasure outcome of A, so a miscalibrated erasure probability reweights the
    branches binomially and moves nothing else. q < 1/2 voids A's zero-capacity premise (it is no
    longer two-extendible): such values are descriptive, not a superactivation statement.
    """
    if not 0.0 <= q <= 1.0:
        raise SuperactivationRefusal(f"q={q} is not a probability")
    n = len(Fk) - 1
    return float(sum(math.comb(n, k) * q ** k * (1 - q) ** (n - k) * Fk[k] for k in range(n + 1)))


def encoder_gradient(decs: dict, n: int, p: float = P_FLIP) -> np.ndarray:
    """G with F = Tr(X G): the linear functional the encoder half-step maximizes."""
    G = np.zeros((2 * (n + 1), 2 * (n + 1)), dtype=complex)
    for k in range(n + 1):
        G += math.comb(n, k) / 2 ** n * adjoint(decs[k], n, k, p) / 4
    return G


def resample_encoder(X: np.ndarray, n_from: int, n_to: int) -> np.ndarray:
    """Seed for another blocklength: Dicke profiles linearly resampled in w/n, renormalized.
    A heuristic starting point for the see-saw; nothing is claimed for its fidelity."""
    m0, m1 = n_from + 1, n_to + 1
    Wm = np.zeros((m1, m0))
    for i, x in enumerate(np.linspace(0, 1, m1)):
        j = min(int(x * (m0 - 1)), m0 - 2)
        t = x * (m0 - 1) - j
        Wm[i, j] += 1 - t
        Wm[i, j + 1] += t
    K = np.kron(np.eye(2), Wm)
    return normalize_encoder(K @ X @ K.T, n_to)


def depolarizing_budget(fidelity_value: float) -> float:
    """Largest final replacement probability delta keeping (1-delta) F + delta/4 above 3/4."""
    if fidelity_value <= THRESHOLD:
        raise SuperactivationRefusal(f"F = {fidelity_value} does not exceed {THRESHOLD}")
    return (fidelity_value - THRESHOLD) / (fidelity_value - 0.25)
