"""The private Horodecki PPT channels with shield dimension d (the construction of K. Horodecki, Pankowski, M. Horodecki
and P. Horodecki, quant-ph/0506203; paper 4, Sec. IV), held here as a claim to verify, not a premise.

Claimed: Smith-Yard's P (d = 2) is the smallest member of a family whose Choi state is
    Phi^P = p0 gamma^0 + p1 gamma^1,    p1 = 1/(sqrt d + 1),  p0 = sqrt d/(sqrt d + 1),
two SHIFTED private states with shield states sigma^0 = 1/d^2 and sigma^1 = (1/d) sum_a |aa><aa|, twisted by a unitary
built from a d x d matrix u with unimodular entries (a complex Hadamard matrix: |u_ab| = 1, u u^dag = d 1):
    gamma^l = 1/2 sum_{i,j} |i><j|_R' (x) |i+l><j+l|_B' (x) U^{i,l} sigma^l U^{j,l dag},     U^{0,l} = 1,
    U^{1,0} = D_u F    (D_u = sum u_ab |ab><ab|, F the swap: at d = 2 this is the paper's eq. sector0-params),
    U^{1,1} = u/sqrt d on span{|aa>} (identity elsewhere: sigma^1 lives there; at d = 2 the paper's sector1 matrix).
By the paper's Theorem effective_channel_general (compiled supplement, l. 1306) such a channel with a 50% erasure
channel of input dimension 2d yields the flagged channel N~_p = 1/2 id (x) |0><0| + 1/2 (X^p o Delta) (x) |1><1|, p = p1.

  python ppt_family.py      checks: d = 2 reproduces the paper's Choi spectrum; PPT at p1 for d = 2..8; NOT PPT
                            just below p1 (the boundary is real); Tr_B = 1/(2d) (a channel); returns the min eigenvalues
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "compile"))


def fourier(d: int) -> np.ndarray:
    w = np.exp(2j * np.pi / d)
    return np.array([[w ** (a * b) for b in range(d)] for a in range(d)])


def hadamard2() -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=complex)


def sylvester(m: int) -> np.ndarray:
    """real Hadamard matrix of order 2^m (entries +-1): the twisting is then exactly rational"""
    H = np.array([[1.0]])
    for _ in range(m):
        H = np.kron(H, [[1, 1], [1, -1]])
    return H.astype(complex)


def twisting(u: np.ndarray):
    d = u.shape[0]
    D = np.diag([u[a, b] for a in range(d) for b in range(d)])
    F = np.zeros((d * d, d * d))
    for a in range(d):
        for b in range(d):
            F[b * d + a, a * d + b] = 1
    U10 = D @ F
    U11 = np.eye(d * d, dtype=complex)
    diag_idx = [a * d + a for a in range(d)]
    for i, a in enumerate(diag_idx):
        for j, b in enumerate(diag_idx):
            U11[b, a] = u[i, j] / math.sqrt(d)            # |aa> -> (1/sqrt d) sum_b u_ab |bb>
    return U10, U11


def choi_state(u: np.ndarray, p1: float) -> np.ndarray:
    """normalized Choi state on R' R'' B' B'' (R first), from the shifted private states above"""
    d = u.shape[0]
    U10, U11 = twisting(u)
    sig0 = np.eye(d * d) / d ** 2
    sig1 = np.zeros((d * d, d * d))
    for a in range(d):
        sig1[a * d + a, a * d + a] = 1 / d
    Us = {0: [np.eye(d * d), U10], 1: [np.eye(d * d), U11]}
    sig = {0: sig0, 1: sig1}
    rho = 0
    for ell, w in ((0, 1 - p1), (1, p1)):
        g = 0
        for i in range(2):
            for j in range(2):
                key = np.zeros((4, 4))
                key[i * 2 + ((i + ell) % 2), j * 2 + ((j + ell) % 2)] = 0.5      # |i, i+l><j, j+l| on R'B'
                g = g + np.kron(key, Us[ell][i] @ sig[ell] @ Us[ell][j].conj().T)
        rho = rho + w * g
    # (R', B', R'', B'') -> (R', R'', B', B'')
    T = rho.reshape(2, 2, d, d, 2, 2, d, d).transpose(0, 2, 1, 3, 4, 6, 5, 7)
    return T.reshape(4 * d * d, 4 * d * d)


def partial_transpose_B(rho: np.ndarray, dR: int, dB: int) -> np.ndarray:
    return rho.reshape(dR, dB, dR, dB).transpose(0, 3, 2, 1).reshape(dR * dB, dR * dB)


def check(d: int, u: np.ndarray, p1: float):
    rho = choi_state(u, p1)
    dR = dB = 2 * d
    trB = rho.reshape(dR, dB, dR, dB).trace(axis1=1, axis2=3)
    is_channel = np.allclose(trB, np.eye(dR) / dR, atol=1e-13)
    psd = np.linalg.eigvalsh(rho).min()
    pt = np.linalg.eigvalsh(partial_transpose_B(rho, dR, dB)).min()
    return is_channel, psd, pt


def main() -> int:
    import sclib
    ok = True
    # (1) d = 2 is the paper's channel: same Choi spectrum as sclib's transcription of eq. choi-state-horo
    rho2 = choi_state(hadamard2(), sclib.P_FLIP)
    ref = sclib.horodecki_choi_state()
    same = np.allclose(np.sort(np.linalg.eigvalsh(rho2)), np.sort(np.linalg.eigvalsh(ref)), atol=1e-13)
    same_pt = np.allclose(np.sort(np.linalg.eigvalsh(partial_transpose_B(rho2, 4, 4))),
                          np.sort(np.linalg.eigvalsh(partial_transpose_B(ref, 4, 4))), atol=1e-13)
    print(f"d = 2 vs the paper's Choi state: spectrum {'same' if same else 'DIFFERENT'}, "
          f"partial-transpose spectrum {'same' if same_pt else 'DIFFERENT'}")
    ok &= same and same_pt
    # (2) PPT at p1 = 1/(sqrt d + 1); (3) NOT PPT just below it
    for d in range(2, 9):
        u = hadamard2() if d == 2 else fourier(d)
        p1 = 1 / (math.sqrt(d) + 1)
        ch, psd, pt = check(d, u, p1)
        _, _, pt_lo = check(d, u, p1 * 0.97)
        print(f"d = {d}: channel {ch}, min eig {psd:+.2e}, PPT min eig {pt:+.2e} at p1 = {p1:.6f}; "
              f"at 0.97 p1: {pt_lo:+.2e}")
        ok &= ch and psd > -1e-13 and pt > -1e-12 and pt_lo < -1e-6
    print("CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
