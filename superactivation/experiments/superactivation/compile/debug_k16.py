"""Debug: where does the 1e-8 Choi error of the n=17, k=16 decoder come from? Compiles the decoder
unitary alone and compares the compiled operator with the requested one on the used columns.

Usage: python debug_k16.py [k]
"""
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

import codes
import decoder as Dc
import sclib as L
from codes import sa

n = 17
k = int(sys.argv[1]) if len(sys.argv) > 1 else 16
m = n - k
X, decs = codes.load_code(n)
V, Xiso, _ = codes.isometric_encoder(X, n)
R = sa.bob_states(Xiso, n, k)
for b in range(k + 1):
    Ks = codes.decoder_kraus(np.asarray(decs[k][b]), m + 1)
    Sb, _ = Dc.support_basis(sa.ptrace_ref(R[b], m + 1))
    KS0 = Dc.restricted_kraus(Ks, Sb)
    I2S = np.kron(np.eye(2), Sb)
    KS = Dc.min_rank_kraus(KS0, I2S.conj().T @ R[b] @ I2S)
    qb = Dc.nbits(m + 1)
    Vc = Dc.restricted_columns(KS, Sb, qb, 0)
    U = Dc.complete_unitary(Vc)
    qc = QuantumCircuit(qb + 1)
    qc.append(UnitaryGate(U), list(range(qb + 1)))
    for lvl in (1, 0):
        try:
            cc = L.compile_cz(qc, level=lvl)
        except Exception as e:
            print("level", lvl, "failed", e)
            continue
        Uc = Operator(cc).data
        cols = Vc.shape[1]
        ph = np.vdot(Uc[:, :cols].reshape(-1), U[:, :cols].reshape(-1)); ph /= abs(ph)
        print(f"b={b} level {lvl}: CZ {L.count_ops(cc).get('cz', 0)}  max|U_c - U| on used cols "
              f"{np.abs(Uc[:, :cols] * ph - U[:, :cols]).max():.2e}  full {np.abs(Uc * ph - U).max():.2e}")
    if b >= 3:
        break
