"""Independent check (written from the supplement text, not from the compile agent's code): the Smith-Yard
private Horodecki channel P (supplemental.tex eq. ppt_channel, K1..K6) equals the claimed mixture of six
2-qubit unitaries. Factor order as printed: K = A (x) B, the first factor is the 'key' k, the second the 'shield' s."""
import numpy as np

p = 1 / (1 + np.sqrt(2))
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
P0, P1 = np.diag([1.0, 0]), np.diag([0, 1.0])
M0 = 0.5 * np.diag([np.sqrt(2 + np.sqrt(2)), np.sqrt(2 - np.sqrt(2))])
M1 = 0.5 * np.diag([np.sqrt(2 - np.sqrt(2)), -np.sqrt(2 + np.sqrt(2))])
K = [np.sqrt((1 - p) / 2) * np.kron(I2, P0), np.sqrt((1 - p) / 4) * np.kron(I2, X),
     np.sqrt((1 - p) / 2) * np.kron(Z, P1), np.sqrt((1 - p) / 4) * np.kron(Z, Y),
     np.sqrt(p) * np.kron(X, M0), np.sqrt(p) * np.kron(Y, M1)]


def choi(kraus):
    d = 4
    phi = np.zeros((d * d, 1))
    for i in range(d):
        phi[i * d + i] = 1
    PHI = phi @ phi.T
    return sum(np.kron(np.eye(d), k) @ PHI @ np.kron(np.eye(d), k).conj().T for k in kraus)


CZ = np.diag([1, 1, 1, -1]).astype(complex)
Dp = np.kron(I2, M0) + 1j * np.kron(Z, M1)
Dm = np.kron(I2, M0) - 1j * np.kron(Z, M1)
U = [(CZ, (1 - p) / 4), (CZ @ np.kron(I2, Z), (1 - p) / 4), (np.kron(I2, X), (1 - p) / 4),
     (np.kron(Z, Y), (1 - p) / 4), (np.kron(X, I2) @ Dp, p / 2), (np.kron(X, I2) @ Dm, p / 2)]
print("sum K^dag K = 1:", np.abs(sum(k.conj().T @ k for k in K) - np.eye(4)).max())
print("each U unitary:", max(np.abs(u.conj().T @ u - np.eye(4)).max() for u, _ in U))
print("weights sum:", sum(w for _, w in U))
d = choi(K) - choi([np.sqrt(w) * u for u, w in U])
print("|Choi(paper Kraus) - Choi(mixture)|_max =", np.abs(d).max())
J = choi(K)
JT = J.reshape(4, 4, 4, 4).transpose(0, 3, 2, 1).reshape(16, 16)        # partial transpose on the output
print("Choi rank:", np.linalg.matrix_rank(J, tol=1e-10), "| PPT min eig:", np.linalg.eigvalsh(JT).min())
print("unital (P(1) = 1):", np.abs(sum(k @ k.conj().T for k in K) - np.eye(4)).max())
