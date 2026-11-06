"""K-group codes for N~^n: encoder output in Sym^{m_1} (x) ... (x) Sym^{m_K}, n = m_1 + ... + m_K.

One exact reduction for every code family studied so far:
  K = 1             the symmetric codes (oneq.superactivation, R132)
  K = 2             the two-group codes (twogroup.py, R140)
  m = (1, ..., 1)   ALL n-use codes: Sym^1 = C^2, so the encoder is unrestricted (general_codes.py's model)
Per group g: flag count k_g, true weight a_g and observed weight b_g among its flagged uses. Bob's state per branch
(k_1..k_K) on R (x) (x)_g Sym^{m_g - k_g}, with weight prod_g C(m_g, k_g) / 2^n:
  rho_(b) = sum_(a) prod_g C_g[b_g, a_g] (1_R (x) (x)_g P^{m_g}_{k_g, a_g}) X (.)^T          (C_g = tg.coeffs(k_g))
The groups' tensor factors are in the order given, so m = (1, ..., 1) orders them as the dense model's qubits.

  python kgroup.py check                         against twogroup.py, oneq.superactivation and the dense model
  python kgroup.py search M1,M2,.. R S [seed]    random-restart see-saw: power decoders, exact encoder step (dual)
The search and the dense check need the `superactivation` extra (cvxpy); the reduction itself is numpy only.
"""
from __future__ import annotations

import functools
import itertools
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from oneq import superactivation as sa  # noqa: E402

import twogroup as tg  # noqa: E402


def dim_sym(ms) -> int:
    return math.prod(m + 1 for m in ms)


def branches(ms):
    """(k_1..k_K, weight) over every flag-count pattern"""
    n = sum(ms)
    for ks in itertools.product(*[range(m + 1) for m in ms]):
        yield ks, math.prod(math.comb(m, k) for m, k in zip(ms, ks, strict=True)) / 2 ** n


def observed(ks) -> list:
    """the weight tuples among the flagged uses: indexes both the true (a) and the observed (b) weights"""
    return list(itertools.product(*[range(k + 1) for k in ks]))


def selection(ms, ks, as_) -> np.ndarray:
    mats = [tg.sel(m, k, a) for m, k, a in zip(ms, ks, as_, strict=True)]
    return np.kron(np.eye(2), functools.reduce(np.kron, mats, np.eye(1)))


def _coeff(Cs, bs, as_) -> float:
    return math.prod(C[b, a] for C, b, a in zip(Cs, bs, as_, strict=True))


def bob_states(X, ms, ks) -> np.ndarray:
    """stack over observed weights (lexicographic, group 1 outermost) of rho on R (x) (x)_g Sym^{m_g - k_g}"""
    Cs = [tg.coeffs(k) for k in ks]
    obs = observed(ks)
    S = {}
    for as_ in obs:
        Op = selection(ms, ks, as_)
        S[as_] = Op @ X @ Op.T
    return np.stack([sum(_coeff(Cs, bs, as_) * S[as_] for as_ in obs) for bs in obs])


def adjoint(Ds, ms, ks) -> np.ndarray:
    """sum_b L_b^dag(D_b): Tr(X adjoint) = sum_b Tr(rho_b D_b)"""
    Cs = [tg.coeffs(k) for k in ks]
    obs = observed(ks)
    out = 0
    for as_ in obs:
        Op = selection(ms, ks, as_)
        Dsum = sum(_coeff(Cs, bs, as_) * Ds[i] for i, bs in enumerate(obs))
        out = out + Op.T @ Dsum @ Op
    return out


def decode(X, ms, warm=None, iters=60):
    decs, F = {}, 0.0
    for ks, w in branches(ms):
        R = bob_states(X, ms, ks)
        d = math.prod(m - k + 1 for m, k in zip(ms, ks, strict=True))
        D = warm[ks] if warm is not None else np.stack([np.eye(2 * d) / 2] * len(R))
        for _ in range(iters):
            D = sa.power_step(R, D, d)
        decs[ks] = D
        F += w * float(np.real(np.einsum("bij,bji->", R, D))) / 4
    return decs, F


def gradient(decs, ms) -> np.ndarray:
    return sa.herm(sum(w * adjoint(decs[ks], ms, ks) / 4 for ks, w in branches(ms)))


def random_encoder(ms, rng) -> np.ndarray:
    m = dim_sym(ms)
    K = rng.normal(size=(2 * m, 2)) + 1j * rng.normal(size=(2 * m, 2))
    X = K @ K.conj().T
    T = X.reshape(2, m, 2, m).trace(axis1=1, axis2=3)
    tw, tv = np.linalg.eigh(sa.herm(T))
    Ti = np.kron((tv / np.sqrt(tw)) @ tv.conj().T, np.eye(m))
    return sa.herm(Ti @ X @ Ti)


def search(ms, restarts, steps, seed=7, dec_iters=60):
    """random-restart see-saw with EXACT encoder steps (tg.encoder_exact, the 3-parameter dual)"""
    n, m = sum(ms), dim_sym(ms)
    rng = np.random.default_rng(seed)
    best = (0.0, None)
    tag = "_".join(map(str, ms))
    for r in range(restarts):
        X = random_encoder(ms, rng)
        decs, F = decode(X, ms, iters=200)
        z, hist = None, [F]
        for _ in range(steps):
            X, z, _ = tg.encoder_exact(gradient(decs, ms), m, z0=z)
            decs, F = decode(X, ms, warm=decs, iters=dec_iters)
            hist.append(F)
            if len(hist) > 21 and hist[-1] - hist[-11] < 1e-9:
                break
        decs, F = decode(X, ms, warm=decs, iters=1000)
        print(f"   m={ms} restart {r}: F = {F:.10f} after {len(hist) - 1} steps", flush=True)
        if F > best[0]:
            best = (F, X)
            np.save(HERE / f"kgroup_X_{tag}.npy", X)
    sym = tg.SYM.get(n, float("nan"))
    print(f"SEARCH m={ms} n={n}: best F = {best[0]:.10f} | symmetric {sym:.10f} | gain {best[0] - sym:+.6f} | "
          f"{'ABOVE 3/4' if best[0] > 0.75 else 'below 3/4'}", flush=True)
    return best


def check() -> int:
    rng = np.random.default_rng(1)
    worst = 0.0
    # (1) K = 2 is twogroup.py, branch by branch
    for a, b in ((2, 3), (3, 1)):
        X = tg.random_encoder(a, b, rng)
        for k1, k2, _ in tg.branches(a, b):
            worst = max(worst, float(np.abs(bob_states(X, (a, b), (k1, k2)) - tg.bob_states(X, a, b, k1, k2)[0]).max()))
    print(f"K = 2 vs twogroup.py: max |diff| {worst:.2e}")
    # (2) K = 1 is oneq.superactivation
    w1 = 0.0
    for n in (3, 5):
        X = sa.random_encoder(n, rng)
        for k in range(n + 1):
            w1 = max(w1, float(np.abs(bob_states(X, (n,), (k,)) - sa.bob_states(X, n, k)).max()))
    print(f"K = 1 vs oneq.superactivation: max |diff| {w1:.2e}")
    # (3) m = (1,...,1) against the dense model: <b|sigma_z|b> on the flagged qubits = rho_b / 2, and the flagged
    # qubits are classical (every block with b != b' vanishes)
    import general_codes as gc
    w3, off = 0.0, 0.0
    for n in (2, 3, 4):
        ms = (1,) * n
        X = random_encoder(ms, rng)
        dense = gc.branch_states(X / 2, n)
        for z, sig in dense.items():
            T = sig.reshape([2] + [2] * n + [2] + [2] * n)
            flagged = [q for q in range(n) if z[q]]
            R = bob_states(X, ms, z)
            for i, bs in enumerate(observed(z)):
                bits = [bs[q] for q in flagged]
                for bits2 in itertools.product((0, 1), repeat=len(flagged)):
                    idx = [slice(None)] * (2 * n + 2)
                    for q, v, v2 in zip(flagged, bits, bits2, strict=True):
                        idx[1 + q] = v
                        idx[2 + n + q] = v2
                    blk = T[tuple(idx)]
                    dB = 2 ** (n - len(flagged))
                    blk = blk.reshape(2 * dB, 2 * dB)
                    if list(bits2) == bits:
                        w3 = max(w3, float(np.abs(blk - R[i] / 2).max()))
                    else:
                        off = max(off, float(np.abs(blk).max()))
    print(f"m = (1,...,1) vs the dense model (n = 2, 3, 4): max |diff| {w3:.2e}; flagged coherences {off:.2e}")
    ok = max(worst, w1, w3, off) < 1e-13
    print("CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "check":
        sys.exit(check())
    if sys.argv[1] == "search":
        ms = tuple(int(x) for x in sys.argv[2].split(","))
        search(ms, int(sys.argv[3]), int(sys.argv[4]), seed=int(sys.argv[5]) if len(sys.argv) > 5 else 7)
        sys.exit(0)
    raise SystemExit(__doc__)
