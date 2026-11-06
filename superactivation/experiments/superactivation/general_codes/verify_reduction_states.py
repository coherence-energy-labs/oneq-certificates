"""Independent STATE-level check of the two-group reduction at large group sizes (no fidelity, no SDP).

Builds the reference-encoded 2^n state of a random two-group code, applies N~ per flag pattern (k1 flagged uses at
the start of group A, k2 at the start of group B: by symmetry within each group any positions give the same state),
then for observed flagged weights (b1, b2) sums the projections of the flagged qubits onto every outcome string with
those weights, and compares Bob's unflagged state -- entry by entry, mapped into the Dicke basis of the unflagged
parts of each group -- with twogroup.bob_states. The reduction's state carries the multiplicity C(k1,b1) C(k2,b2)
of outcome strings and the binomial position factor is in the branch weight, so the comparison is exact.
"""
import itertools
import sys

import numpy as np

import twogroup as tg

P = tg.P_FLIP


def dicke(m, w):
    v = np.zeros(2 ** m)
    if m == 0:
        v[0] = 1.0
        return v
    for bits in itertools.product((0, 1), repeat=m):
        if sum(bits) == w:
            v[int("".join(map(str, bits)), 2)] = 1
    return v / np.linalg.norm(v)


def check(a, b, seed=4):
    n = a + b
    rng = np.random.default_rng(seed)
    X = tg.random_encoder(a, b, rng)
    m = (a + 1) * (b + 1)
    E = np.zeros((2 ** n, m))
    for i in range(a + 1):
        for j in range(b + 1):
            E[:, i * (b + 1) + j] = np.kron(dicke(a, i), dicke(b, j))
    big = np.kron(np.eye(2), E)
    rho = big @ X @ big.T                                     # R (x) A^n, trace 2 (X normalised Tr_A X = 1_R)
    worst = 0.0
    for k1 in range(a + 1):
        for k2 in range(b + 1):
            R_red, _ = tg.bob_states(X, a, b, k1, k2)
            flagged = list(range(k1)) + list(range(a, a + k2))
            unflag_A = list(range(k1, a))
            unflag_B = list(range(a + k2, n))
            # dense: apply X^p o Z-dephasing on flagged qubits, then project them onto outcome strings
            T = rho.reshape([2] + [2] * n + [2] + [2] * n)
            idx = 0
            for b1 in range(k1 + 1):
                for b2 in range(k2 + 1):
                    acc = 0
                    for sA in itertools.product((0, 1), repeat=k1):
                        if sum(sA) != b1:
                            continue
                        for sB in itertools.product((0, 1), repeat=k2):
                            if sum(sB) != b2:
                                continue
                            out = dict(zip(flagged, (*sA, *sB)))
                            # P(observed out | true t) for each flagged qubit, then dephased: sum over true values t
                            for tvals in itertools.product((0, 1), repeat=len(flagged)):
                                pr = 1.0
                                for q, t in zip(flagged, tvals):
                                    pr *= (1 - P) if out[q] == t else P
                                sl = [slice(None)] * (2 * n + 2)
                                for q, t in zip(flagged, tvals):
                                    sl[1 + q] = t
                                    sl[2 + n + q] = t
                                acc = acc + pr * T[tuple(sl)]
                    # acc: tensor over R, unflagged qubits (ket), R', unflagged (bra), in original order
                    du = len(unflag_A) + len(unflag_B)
                    M = np.asarray(acc).reshape(2 * 2 ** du, 2 * 2 ** du)
                    Eu = np.zeros((2 ** du, (a - k1 + 1) * (b - k2 + 1)))
                    for i in range(a - k1 + 1):
                        for j in range(b - k2 + 1):
                            Eu[:, i * (b - k2 + 1) + j] = np.kron(dicke(a - k1, i), dicke(b - k2, j))
                    Bu = np.kron(np.eye(2), Eu)
                    dense_in_dicke = Bu.T @ M @ Bu
                    err = np.abs(dense_in_dicke - R_red[idx]).max()
                    worst = max(worst, err)
                    idx += 1
    print(f"(a,b)=({a},{b}) n={n}: worst |dense branch state - reduction| over all (k1,k2,b1,b2) = {worst:.2e}")
    return worst


if __name__ == "__main__":
    for tok in sys.argv[1:]:
        a, b = map(int, tok.split(","))
        check(a, b)
