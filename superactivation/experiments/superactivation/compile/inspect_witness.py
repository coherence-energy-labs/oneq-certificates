"""Inspect the certified witnesses: encoder rank / reality / Dicke profile, decoder Kraus ranks.

The decoder compilation (decoder.py, R139) cuts each decoder's Kraus rank to the subspace it acts on; these are the
numbers that cut was read from. Each measurement is a function, so it is testable on a synthetic code whose answer
is known (tests/test_superactivation_lanes_r136_r139.py).

Usage: python inspect_witness.py
"""
import pathlib
import sys

import numpy as np

ONEQ = pathlib.Path(__file__).resolve().parents[3]          # the ONE_Q repo root (experiments/superactivation/compile)
sys.path.insert(0, str(ONEQ / "src"))
from oneq.superactivation_certificate import load_witness  # noqa: E402
from oneq import superactivation as sa  # noqa: E402


def encoder_structure(X: np.ndarray, n: int) -> dict:
    """Top eigenvalues of the encoder Choi, and the rank-1 reading of it: V|r> in the Dicke basis.

    An exact isometry V: C^2 -> Sym^n has Choi rank 1 with eigenvalue 2, and the Gram matrix of V|0>, V|1> is the
    identity. A rank-2 Choi read through its top eigenvector gives a Gram matrix visibly short of I.
    """
    m = n + 1
    w, v = np.linalg.eigh(sa.herm(X))
    top = v[:, -1] * np.sqrt(w[-1])
    Vm = top.reshape(2, m)                                  # row r = V|r> in the Dicke basis
    return {"max_abs_imag": float(np.abs(X.imag).max()), "top_eigenvalues": w[::-1][:4].copy(),
            "gram": Vm.conj() @ Vm.T, "dicke_profiles": np.abs(Vm) ** 2}


def numerical_rank(D: np.ndarray, rel: float) -> int:
    ew = np.linalg.eigvalsh(sa.herm(np.asarray(D)))[::-1]
    return int(np.sum(ew > rel * max(ew[0], 1e-300)))


def decoder_ranks(decs: dict, n: int, rel: float = 1e-6) -> dict:
    """Kraus rank of every decoder Choi D_{k,b} (eigenvalues above rel x the largest)."""
    return {k: [numerical_rank(decs[k][b], rel) for b in range(k + 1)] for k in range(n + 1)}


def support_ranks(X: np.ndarray, n: int, rel: float = 1e-9) -> dict:
    """Rank of Tr_R rho_{k,b} on Sym^{n-k}: the subspace each branch's decoder must act on."""
    out = {}
    for k in range(n + 1):
        R = sa.bob_states(X, n, k)
        out[k] = [numerical_rank(sa.ptrace_ref(R[b], n - k + 1), rel) for b in range(k + 1)]
    return out


def main() -> int:
    for n in (17, 24):
        n_, X, decs = load_witness(ONEQ / "evidence" / "superactivation" / f"witness_n{n}.npz")
        if n_ != n:
            raise SystemExit(f"REFUSING: witness_n{n}.npz holds n = {n_}")
        s = encoder_structure(X, n)
        print(f"=== n = {n}  X shape {X.shape}  max|Im X| = {s['max_abs_imag']:.3e}")
        print("  top eigenvalues of X:", np.round(s["top_eigenvalues"], 8))
        print("  Gram of V|r> (should be I):", np.round(s["gram"], 6).tolist())
        for r in range(2):
            print(f"  |V|{r}>|^2 over weight w:", " ".join(f"{x:.3f}" for x in s["dicke_profiles"][r]))
        F_ref = sa.fidelity(X, {k: np.stack(decs[k]) for k in decs}, n)
        print(f"  float fidelity with witness decoders: {F_ref:.9f}")
        print("  decoder (k,b,d,rank@1e-6):")
        for k, ranks in decoder_ranks(decs, n).items():
            print("   k=%2d d=%2d ranks:" % (k, n - k + 1), ranks)
        print("  rank of rho_{k,b} restricted support on Sym^{n-k} (1e-9 rel):")
        for k, sup in support_ranks(X, n).items():
            print("   k=%2d" % k, sup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
