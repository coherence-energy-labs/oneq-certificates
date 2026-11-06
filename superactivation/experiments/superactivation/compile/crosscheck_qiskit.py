"""Independent-simulator cross-check (common-mode guard): the branch simulator in sclib.py verified
every circuit; here qiskit's own quantum_info simulators re-check the key artifacts.

  1. compiled encoders (all-to-all and nearest-neighbour) at n = 17: qiskit Statevector vs target;
  2. the per-use block with the measurement-free D~ (resets only, no mid-circuit reads), averaged over
     the controller's 12 draws: qiskit DensityMatrix (handles reset) vs the Choi matrix of N~.

Usage: python crosscheck_qiskit.py
"""
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

import channels as C
import codes
import dicke as Dk
import sclib as L

n = 17
X, decs = codes.load_code(n)
V, Xiso, _ = codes.isometric_encoder(X, n)
for name, builder in (("all-to-all", lambda: Dk.encoder_circuit(V, "givens", "num4")[0]),
                      ("nearest-neighbour", lambda: Dk.encoder_circuit_lnn(V)[0])):
    q = L.compile_cz(builder())
    tg = [Dk.symmetric_vector(n, V[:, r]) for r in range(2)]
    sv = []
    for r in range(2):
        init = QuantumCircuit(n)
        if r:
            init.x(n - 1)
        sv.append(Statevector.from_instruction(init.compose(q)).data)
    ov = np.vdot(tg[0], sv[0]) + np.vdot(tg[1], sv[1])
    ph = ov / abs(ov)
    err = max(float(np.abs(sv[r] - ph * tg[r]).max()) for r in range(2))
    print(f"encoder ({name}) n={n}: qiskit Statevector error {err:.2e}")

# per-use block, measurement-free D~, reference qubit R = qubit 4
J = np.zeros((8, 8), dtype=complex)
for jP in range(6):
    for er in (0, 1):
        blk = C.block_randomized(jP, er, decoder="coherent")
        blk = L.compile_cz(blk, level=3)
        qc = QuantumCircuit(5)
        qc.h(4); qc.cx(4, 0)                       # |Phi> on (R, k1)
        qc.compose(blk.remove_final_measurements(inplace=False) if blk.num_clbits else blk,
                   qubits=[0, 1, 2, 3], inplace=True)
        rho = DensityMatrix.from_instruction(qc)
        red = partial_trace(rho, [1, 2, 3]).data   # keeps qubits (0 = k1, 4 = R), little-endian
        # little-endian over kept (k1, R): row index = k1 + 2 R = (R, k1) big-endian -> already R-first.
        # (N~'s Choi is symmetric under R <-> k1, so this ordering is NOT tested by the comparison;
        #  it is fixed here by construction, not by the check.)
        M = red * 2                                # Choi = d_in x state
        E = np.zeros((2, 2)); E[er, er] = 1
        J += C.P_UNITARY_WEIGHTS[jP] * 0.5 * np.kron(M, E)
print(f"per-use block (randomized P/A, measurement-free D~) vs N~: qiskit DensityMatrix Choi trace distance "
      f"{L.choi_dist(J, L.ntilde_choi()):.2e}")
