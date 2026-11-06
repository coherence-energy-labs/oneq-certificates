"""Debug: accuracy of qiskit's 2-qubit synthesis on the n=17, k=16, b=3 decoder unitary.

Usage: python debug_kak.py
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate, CZGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer

import codes
import decoder as Dc
from codes import sa

n, k, m, b = 17, 16, 1, 3
X, decs = codes.load_code(n)
V, Xiso, _ = codes.isometric_encoder(X, n)
R = sa.bob_states(Xiso, n, k)
Ks = codes.decoder_kraus(np.asarray(decs[k][b]), m + 1)
Sb, _ = Dc.support_basis(sa.ptrace_ref(R[b], m + 1))
KS0 = Dc.restricted_kraus(Ks, Sb)
I2S = np.kron(np.eye(2), Sb)
KS = Dc.min_rank_kraus(KS0, I2S.conj().T @ R[b] @ I2S)
Vc = Dc.restricted_columns(KS, Sb, 1, 0)
print("V isometric:", np.abs(Vc.conj().T @ Vc - np.eye(2)).max())
U = Dc.complete_unitary(Vc, 11)
print("U unitary:", np.abs(U.conj().T @ U - np.eye(4)).max(), " det", np.linalg.det(U))
dec = TwoQubitBasisDecomposer(CZGate())
for approx in (None, 1.0):
    qc = dec(U) if approx is None else dec(U, basis_fidelity=1.0, approximate=False)
    Uc = Operator(qc).data
    ph = np.vdot(Uc.reshape(-1), U.reshape(-1)); ph /= abs(ph)
    print(f"decomposer approx={approx}: CZ {qc.count_ops().get('cz', 0)}  err {np.abs(Uc * ph - U).max():.2e}")
qc = QuantumCircuit(2); qc.append(UnitaryGate(U), [0, 1])
for lvl in (0, 1, 2, 3):
    t = transpile(qc, basis_gates=["cz", "sx", "rz", "x"], optimization_level=lvl, approximation_degree=1.0)
    Uc = Operator(t).data
    ph = np.vdot(Uc.reshape(-1), U.reshape(-1)); ph /= abs(ph)
    print(f"transpile level {lvl}: CZ {t.count_ops().get('cz', 0)}  err {np.abs(Uc * ph - U).max():.2e}")
