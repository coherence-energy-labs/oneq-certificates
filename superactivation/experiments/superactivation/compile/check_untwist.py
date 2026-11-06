"""The paper's untwisting unitary V^dag: generic 4-qubit synthesis vs the structured form
(CX + one uniformly controlled single-qubit gate + CX). Operator equality and CZ counts.

Usage: python check_untwist.py
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

import channels as C
import sclib as L

for order in ("s2_first", "s1_first"):
    Vd = C.untwist_unitary(order)
    g = QuantumCircuit(4); g.append(UnitaryGate(Vd), [0, 1, 2, 3])   # qargs (k1, k2, s1, s2)
    s = QuantumCircuit(4); C.untwist_structured(s, 0, 2, 1, 3)        # (k1, s1, k2, s2) roles
    for nm, c in (("generic", g), ("structured", s)):
        cc = L.compile_cz(c, level=3)
        U = Operator(cc).data
        ph = np.vdot(U.reshape(-1), Vd.reshape(-1)); ph /= abs(ph)
        print(f"V^dag ({order}) {nm:10s}: CZ {L.count_ops(cc).get('cz', 0):3d}   ||U - V^dag||_max = {np.abs(U * ph - Vd).max():.2e}")
