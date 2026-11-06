"""What does a good two-group code look like? Codeword amplitude matrices over (Dicke_A index i, Dicke_B index j),
their Schmidt spectra across the A:B cut, and how the two codewords relate (swap A<->B, phases)."""
import sys

import numpy as np

path, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
X = np.load(path)
w, v = np.linalg.eigh((X + X.conj().T) / 2)
print(f"rank check: top eigenvalues {np.round(w[::-1][:3], 6)}")
m = (a + 1) * (b + 1)
psi = v[:, -1] * np.sqrt(w[-1])                          # sum_r |r> V|r> (Tr = 2)
C = [psi[r * m:(r + 1) * m].reshape(a + 1, b + 1) for r in range(2)]
np.set_printoptions(precision=3, suppress=True, linewidth=160)
for r in range(2):
    s = np.linalg.svd(C[r], compute_uv=False)
    print(f"codeword {r}: norm {np.linalg.norm(C[r]):.4f}; Schmidt spectrum A:B {np.round(s / np.linalg.norm(s), 4)}")
    print("  |amplitude| over (i = Dicke_A weight, j = Dicke_B weight):")
    print(np.abs(C[r]))
    print("  phase/pi:")
    ph = np.angle(C[r]) / np.pi
    ph[np.abs(C[r]) < 1e-3] = np.nan
    print(ph)
if a == b:
    for r in range(2):
        print(f"codeword {r} vs its A<->B swap: |<c|swap c>| = {abs(np.vdot(C[r].ravel(), C[r].T.ravel())):.4f}")
    print(f"|<c0|swap c1>| = {abs(np.vdot(C[0].ravel(), C[1].T.ravel())):.4f}")
