"""Is N~_p = 1/2 id (x) |0><0| + 1/2 (X^p o Delta) (x) |1><1| achievable from ONE use of P_d (x) A_{2d}, p = 1/(sqrt d+1)?

Checked here independently of the paper's general effective-channel theorem, with Prop. 1's encoder generalized to
shield dimension d (Bell pair Phi^d on the two shields, CNOT copy of the key into A):
    V|r> = (1/sqrt d) sum_s |r, s>_P-in (x) |r, s>_A-in.
  KEPT branch (A delivers):  Bob holds P's output and A's input intact. Perfect recovery of the logical qubit exists iff
      the Knill-Laflamme condition holds for the code V against P's Kraus operators K_a (x) 1_A:
      V^dag (K_b^dag K_a (x) 1) V = c_ab 1_2 for all a, b. Measured as the max deviation over the Gram matrix.
  ERASED branch (A lost):    Bob holds P(Tr_A V rho V^dag) only. Tr_A kills every coherence between the key values
      (the key's copy went into A), so the branch is a classical channel from the key bit to Bob's states
      Q_i = P(|i><i| (x) 1/d). Bob's best guess errs with the Helstrom probability 1/2 (1 - ||Q_0 - Q_1||_1 / 2); that
      is exactly X^p o Delta iff it equals p.
Both properties hold per use, so N~_p^n is achievable with per-use operations, and F_c((P_d (x) A)^n) >= F_c(N~_p^n).

  python effective_channel_check.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ppt_family as pf  # noqa: E402


def kraus_from_choi_state(rho: np.ndarray, d_in: int, d_out: int, tol=1e-13):
    J = rho * d_in                                     # unnormalized Choi, input first: J = sum |K>><<K|
    w, v = np.linalg.eigh((J + J.conj().T) / 2)
    keep = w > tol * w.max()
    return [np.sqrt(w[k]) * v[:, k].reshape(d_in, d_out).T for k in np.flatnonzero(keep)]


def encoder(d: int) -> np.ndarray:
    D = 2 * d
    V = np.zeros((D * D, 2), dtype=complex)
    for r in range(2):
        for s in range(d):
            i = r * d + s
            V[i * D + i, r] = 1 / math.sqrt(d)
    return V


def check(d: int, u: np.ndarray):
    D = 2 * d
    p = 1 / (math.sqrt(d) + 1)
    Ks = kraus_from_choi_state(pf.choi_state(u, p), D, D)
    V = encoder(d)
    # KEPT: W[:, (a, r)] = (K_a (x) 1_A) V|r>;  Gram G = W^dag W must be block-diagonal-proportional: G[(a,r),(b,s)] = c_ba delta_rs
    W = np.concatenate([np.kron(K, np.eye(D)) @ V for K in Ks], axis=1)
    G = (W.conj().T @ W).reshape(len(Ks), 2, len(Ks), 2)
    off = np.abs(G[:, 0, :, 1]).max()                  # <K_b V 1 | K_a V 0> must vanish
    diff = np.abs(G[:, 0, :, 0] - G[:, 1, :, 1]).max() # and the two diagonals must agree
    kl = max(off, diff)
    # ERASED: Q_i = P(|i><i| (x) 1/d) on P's output
    Q = []
    for i in range(2):
        rin = np.zeros((D, D), dtype=complex)
        for s in range(d):
            rin[i * d + s, i * d + s] = 1 / d
        Q.append(sum(K @ rin @ K.conj().T for K in Ks))
    helstrom = 0.5 * (1 - 0.5 * np.abs(np.linalg.eigvalsh(Q[0] - Q[1])).sum())
    return len(Ks), kl, helstrom, p


def main() -> int:
    ok = True
    cases = [(2, pf.hadamard2()), (3, pf.fourier(3)), (4, pf.sylvester(2)), (8, pf.sylvester(3)),
             (16, pf.sylvester(4)), (22, pf.fourier(22)), (32, pf.sylvester(5))]
    for d, u in cases:
        nk, kl, h, p = check(d, u)
        good = kl < 1e-12 and abs(h - p) < 1e-12
        ok &= good
        print(f"d = {d:2d}: {nk:4d} Kraus; kept-branch Knill-Laflamme deviation {kl:.1e}; "
              f"erased-branch Helstrom error {h:.12f} vs p = {p:.12f}  {'ok' if good else 'FAIL'}", flush=True)
    print("CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
