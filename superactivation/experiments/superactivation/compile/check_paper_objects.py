"""Check the transcription of the paper's channel definitions before compiling anything.

  1. the six Kraus operators of P reproduce the Choi state of eq. (choi-state-horo) -- the paper
     asserts this (l.392); we test it, including the M_1 sign the paper flags as a Smith-Yard typo;
  2. P's Choi is PPT, rank 6, trace preserving;
  3. Lemma choi-as-mixture: Phi^P = (1-p) gamma^0 + p gamma^1 with the stated U^{i,l}, sigma^l;
  4. P is a MIXED-UNITARY channel (six unitaries), which is what makes a 0-ancilla randomized
     implementation exact: P = sum_j w_j U_j . U_j^dag  (derived here, checked to machine precision).

Usage: python check_paper_objects.py
"""
import math

import numpy as np

import sclib as L

p = L.P_FLIP
Ks = L.horodecki_kraus(p)
J = L.choi_from_kraus(Ks)               # unnormalized, R=(key,shield) first
Jpaper = 4 * L.horodecki_choi_state(p)  # normalized state x d_in
print("P: ||J_kraus - 4 Phi^P||_max =", L.max_abs(J - Jpaper))
print("P: sum K^dag K = I ?", L.max_abs(sum(K.conj().T @ K for K in Ks) - np.eye(4)))
ev = np.linalg.eigvalsh(J)
print("P: Choi rank =", int(np.sum(ev > 1e-12)), " min eig", ev.min())
# partial transpose on B
JT = J.reshape(4, 4, 4, 4).transpose(0, 3, 2, 1).reshape(16, 16)
print("P: min eig of Choi^{T_B} =", np.linalg.eigvalsh(JT).min(), "(PPT iff >= 0)")
# wrong sign of M1 (Smith-Yard supplement) does NOT reproduce the Choi state:
c = math.sqrt(2 + L.SQ2) / 2
s = math.sqrt(2 - L.SQ2) / 2
Kbad = list(Ks)
Kbad[5] = math.sqrt(p) * L.kron(L.Y, np.diag([s, c]).astype(complex))
print("P: with the misprinted M1 sign, ||J - 4 Phi^P||_max =", L.max_abs(L.choi_from_kraus(Kbad) - Jpaper))

# Lemma choi-as-mixture (l.1128-1162): gamma^l = U^l (Phi^l_{R'B'} (x) sigma^l_{R''B''}) U^l^dag
def gamma(l, U0, U1, sig):
    out = np.zeros((16, 16), dtype=complex)
    for i in range(2):
        for j in range(2):
            Ui = U0 if i == 0 else U1
            Uj = U0 if j == 0 else U1
            key = np.zeros((4, 4), dtype=complex)  # |i><j|_{R'} (x) X^l|i><j|X^l _{B'}
            key[i * 2 + ((i + l) % 2), j * 2 + ((j + l) % 2)] = 0.5
            sh = Ui @ sig @ Uj.conj().T          # on (R'', B'')
            out += np.kron(key, sh)              # order (R', B', R'', B'')
    return out.reshape([2] * 8).transpose(0, 2, 1, 3, 4, 6, 5, 7).reshape(16, 16)

U10 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]], dtype=complex)
h = 1 / L.SQ2
U11 = np.array([[h, 0, 0, h], [0, 0, 1, 0], [0, 1, 0, 0], [h, 0, 0, -h]], dtype=complex)
sig0 = np.eye(4) / 4
sig1 = np.diag([0.5, 0, 0, 0.5]).astype(complex)
mix = (1 - p) * gamma(0, np.eye(4), U10, sig0) + p * gamma(1, np.eye(4), U11, sig1)
print("Lemma choi-as-mixture: ||(1-p)g0 + p g1 - Phi^P||_max =", L.max_abs(mix - L.horodecki_choi_state(p)))

# Mixed-unitary decomposition (derived in RESULTS.md):
#   {K1,K3} = Ad_CZ o (shield dephasing) = 1/2 Ad_CZ + 1/2 Ad_{CZ Z_s}
#   {K2,K4} = mixture of 1(x)X and Z(x)Y
#   {K5,K6} = Ad_{X(x)1} o {1(x)M0, i Z(x)M1};  1(x)M0 +- i Z(x)M1 are diagonal unitaries D+-
CZ = np.diag([1, 1, 1, -1]).astype(complex)
Zs = L.kron(L.I2, L.Z)
Dp = np.kron(np.eye(2), np.diag([c, s])) + 1j * np.kron(L.Z, np.diag([s, -c]))
Dm = np.kron(np.eye(2), np.diag([c, s])) - 1j * np.kron(L.Z, np.diag([s, -c]))
print("D+ unitary:", L.max_abs(Dp.conj().T @ Dp - np.eye(4)), " D- unitary:", L.max_abs(Dm.conj().T @ Dm - np.eye(4)))
print("D+ diag phases / pi:", np.round(np.angle(np.diag(Dp)) / math.pi, 6))
XI = L.kron(L.X, L.I2)
units = [(CZ, (1 - p) / 4), (CZ @ Zs, (1 - p) / 4), (L.kron(L.I2, L.X), (1 - p) / 4),
         (L.kron(L.Z, L.Y), (1 - p) / 4), (XI @ Dp, p / 2), (XI @ Dm, p / 2)]
Jmu = L.choi_from_kraus([math.sqrt(w) * U for U, w in units])
print("P mixed-unitary decomposition: ||J_mu - J||_max =", L.max_abs(Jmu - J))

# erasure and N~
JA = L.choi_from_kraus(L.erasure_kraus())
print("A: Choi shape", JA.shape, " trace", np.trace(JA).real, " TP:",
      L.max_abs(sum(K.conj().T @ K for K in L.erasure_kraus()) - np.eye(4)))
JN = L.ntilde_choi()
print("N~: Choi trace", np.trace(JN).real, " eigenvalues", np.round(np.linalg.eigvalsh(JN), 6))
