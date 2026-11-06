"""Re-certify the code THAT IS COMPILED: the isometric (top-eigenvector, polar) encoder X_iso and the
minimum-rank restricted decoders (completed on S-perp by 'prepare |0>'), with ONE_Q's exact-arithmetic
certificate (oneq.superactivation_certificate.certify: dyadic rounding, Sylvester positivity with
integer determinants, Gershgorin normalization, 256-bit ball arithmetic). The circuits implement
exactly these maps (verify_encoder.py, verify_decoder.py), so this certificate covers the compiled
protocol on N~ (given the two imported theorems of arXiv 2604.27042 that the module states).

Usage: python certify_compiled.py [n ...]  (default 17 24) -> certificate_compiled_n{n}.json
"""
import json
import sys
import time

import numpy as np

import codes
import decoder as Dc
from codes import sa
from oneq.superactivation_certificate import certify


def full_decoder_adjoint_choi(KS, Sb, m):
    """Adjoint-Choi on R (x) Sym^m of: K_S on the support (basis Sb), |0><f_j| on S-perp."""
    L_ = m + 1
    Ks = [K @ Sb.conj().T for K in KS]
    P = np.eye(L_) - Sb @ Sb.conj().T
    w, U = np.linalg.eigh((P + P.conj().T) / 2)
    for f in U[:, w > 0.5].T:
        K = np.zeros((2, L_), dtype=complex)
        K[0, :] = f.conj()
        Ks.append(K)
    tp = np.abs(sum(K.conj().T @ K for K in Ks) - np.eye(L_)).max()
    assert tp < 1e-10, tp
    return codes.adjoint_choi_from_kraus(Ks, L_)


NS = [int(a) for a in sys.argv[1:]] or [17, 24]
for n in NS:
    t0 = time.time()
    X, decs = codes.load_code(n)
    V, Xiso, _ = codes.isometric_encoder(X, n)
    newdecs = {}
    for k in range(n + 1):
        m = n - k
        R = sa.bob_states(Xiso, n, k)
        newdecs[k] = []
        for b in range(k + 1):
            Ks = codes.decoder_kraus(np.asarray(decs[k][b]), m + 1)
            Sb, _ = Dc.support_basis(sa.ptrace_ref(R[b], m + 1))
            KS0 = Dc.restricted_kraus(Ks, Sb)
            I2S = np.kron(np.eye(2), Sb)
            KS = Dc.min_rank_kraus(KS0, I2S.conj().T @ R[b] @ I2S)
            newdecs[k].append(full_decoder_adjoint_choi(KS, Sb, m))
    F_float = sa.fidelity(Xiso, {k: np.stack(v) for k, v in newdecs.items()}, n)
    cert = certify(n, Xiso, newdecs)
    cert["float_fidelity_compiled_code"] = F_float
    cert["what"] = "isometric encoder (polar of top eigenvector) + minimum-rank restricted decoders (|0> on S-perp)"
    cert["seconds_total"] = round(time.time() - t0, 1)
    json.dump(cert, open(f"certificate_compiled_n{n}.json", "w"), indent=1, default=str)
    print(f"n={n}: float F {F_float:.12f}  certified lower bound {cert['certified_lower_bound_float']:.12f}"
          f"  exceeds 3/4: {cert['exceeds_three_quarters']}  margin >= {cert['certified_margin_lower_bound']:.4e}"
          f"  ({cert['seconds_total']}s)", flush=True)
