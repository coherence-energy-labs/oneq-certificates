"""Load the certified witnesses (ONE_Q main 8b8b6247) and derive the ISOMETRIC encoder.

Encoder Choi X on R (x) Sym^n (R first, Dicke basis D^n_w, w = 0..n), Tr_Sym X = I_R; for an
isometry V, X = |v><v| with v = sum_r |r> (x) V|r>. The witness X has Kraus rank 1 up to an SDP
residue (second eigenvalue 8.3e-5 at n=17, 4.0e-5 at n=24); we keep the top eigenvector and make it
exactly isometric by polar decomposition. Decoders: adjoint-decoder Chois D_{k,b} on R (x) Sym^{n-k},
Tr_R D = I; Kraus K_m[r, i] = conj(w_m[r, i]) for D = sum_m |w_m><w_m| (derived in RESULTS.md).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ONEQ = pathlib.Path(__file__).resolve().parents[3]          # the ONE_Q repo root (experiments/superactivation/compile)
sys.path.insert(0, str(ONEQ / "src"))
from oneq import superactivation as sa  # noqa: E402
from oneq.superactivation_certificate import load_witness  # noqa: E402


def isometric_encoder(X: np.ndarray, n: int):
    """(V, X_iso, info): V is (n+1) x 2 with V[:, r] = V|r> in the Dicke basis."""
    w, U = np.linalg.eigh(sa.herm(X))
    v = U[:, -1] * np.sqrt(w[-1])
    Vm = v.reshape(2, n + 1).T                      # columns V|0>, V|1>
    Uu, s, Wh = np.linalg.svd(Vm, full_matrices=False)
    Vp = Uu @ Wh                                    # polar factor: nearest isometry
    viso = np.concatenate([Vp[:, 0], Vp[:, 1]])
    Xiso = np.outer(viso, viso.conj())
    info = {"eig_top": float(w[-1]), "eig_second": float(w[-2]),
            "polar_singular_values": s.tolist(),
            "||X - X_iso||_max": float(np.abs(X - Xiso).max())}
    return Vp, Xiso, info


def load_code(n: int):
    n_, X, decs = load_witness(ONEQ / "evidence" / "superactivation" / f"witness_n{n}.npz")
    assert n_ == n
    return X, decs


def decoder_kraus(D: np.ndarray, d: int, rel=1e-13):
    """Kraus operators (2 x d) of the decoder whose adjoint-Choi is D (on R (x) C^d)."""
    w, U = np.linalg.eigh(sa.herm(D))
    Ks = []
    for lam, vec in zip(w[::-1], U[:, ::-1].T):
        if lam <= rel * max(w[-1], 1e-300):
            break
        Ks.append(np.conj(np.sqrt(lam) * vec.reshape(2, d)))
    return Ks


def adjoint_choi_from_kraus(Ks, d):
    D = np.zeros((2 * d, 2 * d), dtype=complex)
    for K in Ks:
        wv = np.conj(K).reshape(-1)
        D += np.outer(wv, wv.conj())
    return D


def fidelity(Xenc, decs, n):
    return sa.fidelity(Xenc, {k: np.stack(decs[k]) for k in decs}, n)
