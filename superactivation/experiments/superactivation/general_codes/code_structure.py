"""Where do the optimal NON-symmetric codes live? Spin-sector (S_n irrep) content of each codeword.

Loads general_best_rho_n{n}.npy (rank-1 reference-encoded state on R (x) A^n), extracts the two codewords
V|0>, V|1> (up to a basis change on R, so it reports the 2-dim code SPACE's weights), and decomposes the code space
by total spin J of the n qubits: J = n/2 is Sym^n; J = n/2 - 1 the (n-1,1) irreps; and so on.
"""
import sys

import numpy as np

X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)


def total_spin_sq(n):
    d = 2 ** n
    S2 = np.zeros((d, d), complex)
    for P in (X, Y, Z):
        Sa = np.zeros((d, d), complex)
        for q in range(n):
            op = np.array([[1.0]])
            for k in range(n):
                op = np.kron(op, P / 2 if k == q else np.eye(2))
            Sa += op
        S2 += Sa @ Sa
    return S2


def main(n):
    rho = np.load(f"general_best_rho_n{n}.npy")
    w, v = np.linalg.eigh((rho + rho.conj().T) / 2)
    psi = v[:, -1] * np.sqrt(w[-1])                     # (1/sqrt2) sum_r |r> V|r>, R first
    dA = 2 ** n
    V = np.stack([psi[:dA], psi[dA:]], axis=1) * np.sqrt(2)
    Q, _ = np.linalg.qr(V)                               # orthonormal basis of the code space
    ev, U = np.linalg.eigh(total_spin_sq(n))
    print(f"n={n}: top eigenvalue of rho {w[-1]:.6f} (2nd {w[-2]:.2e}); |V^dag V - 1| = "
          f"{np.abs(V.conj().T @ V - np.eye(2)).max():.2e}")
    for J in np.arange(n / 2, -0.1, -1):
        cols = np.abs(ev - J * (J + 1)) < 1e-6
        if not cols.any():
            continue
        P = U[:, cols] @ U[:, cols].conj().T
        weight = np.real(np.trace(Q.conj().T @ P @ Q)) / 2      # average weight of the code space in sector J
        print(f"   spin J = {J:4.1f} (dim {cols.sum():3d}): code-space weight {weight:.6f}")


if __name__ == "__main__":
    for n in map(int, sys.argv[1:]):
        main(n)
