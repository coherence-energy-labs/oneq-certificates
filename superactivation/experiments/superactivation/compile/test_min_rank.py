"""Check the fixed-fidelity Kraus-rank reduction on the certified n = 17 code, a few branches:
rank before/after, trace preservation on S, branch fidelity before/after.

Usage: python test_min_rank.py [n]
"""
import sys

import numpy as np

import codes
import decoder as Dc
from codes import sa

n = int(sys.argv[1]) if len(sys.argv) > 1 else 17
X, decs = codes.load_code(n)
V, Xiso, _ = codes.isometric_encoder(X, n)
for k in (1, 4, 6, 8, 9, 12):
    m = n - k
    R = sa.bob_states(Xiso, n, k)
    for b in (0, k // 2):
        Ks = codes.decoder_kraus(np.asarray(decs[k][b]), m + 1)
        Sb, _ = Dc.support_basis(sa.ptrace_ref(R[b], m + 1))
        s = Sb.shape[1]
        KS = Dc.restricted_kraus(Ks, Sb)
        I2S = np.kron(np.eye(2), Sb)
        rhoS = I2S.conj().T @ R[b] @ I2S
        K2 = Dc.min_rank_kraus(KS, rhoS)
        D1 = codes.adjoint_choi_from_kraus(KS, s)
        D2 = codes.adjoint_choi_from_kraus(K2, s)
        f1 = np.real(np.trace(rhoS @ D1)) / 4
        f2 = np.real(np.trace(rhoS @ D2)) / 4
        tp = np.abs(sum(K.conj().T @ K for K in K2) - np.eye(s)).max()
        print(f"k={k:2d} b={b:2d} s={s:2d}: rank {len(KS):2d} -> {len(K2):2d}  TP err {tp:.1e}  "
              f"branch F {f1:.15f} -> {f2:.15f} (d {f2 - f1:+.1e})")
