"""Falsification probe (R137): can the PPT-assisted relaxation ever prove the superactivation onset n* = 17?

  python ppt_relaxation_probe.py [NMAX=2]     # writes evidence/superactivation/ppt_relaxation_probe.json
The SDP needs the `superactivation` extra (cvxpy, clarabel), imported where it is used; the channel
Choi matrices and their n-fold assembly are plain numpy and are tested without it. Dense: n = 3 needs
~9 GB and is not attempted by default (it is not needed -- n = 2 decides).

F_PPT(N, M) (Leung-Matthews) = max Tr(J_N W) s.t. 0 <= W <= rho_A (x) 1_B, Tr rho_A = 1,
-rho_A (x) 1_B <= M W^{T_B} <= rho_A (x) 1_B. It upper-bounds the unassisted F_c(N, M). For a flagged
channel the flags are classical, so W is block diagonal in the flag string z WLOG (pinching the
flag commutes with every constraint), giving one block per z.

If F_PPT(N~^{(x)n}, 2) already exceeds 3/4 at small n, this relaxation cannot prove that no code
with n <= 16 uses beats 3/4: the route dies. Sanity cases first: identity -> 1, an
entanglement-breaking channel -> 1/2 (PPT assistance alone reaches 1/M), and the 50% erasure channel.
"""
import itertools
import sys

import numpy as np

P = 1 / (1 + np.sqrt(2))
I2 = np.eye(2)


def choi_id():
    v = np.zeros(4)
    v[0] = v[3] = 1
    return np.outer(v, v)                                       # sum |ii><jj|, A first


def choi_noisy():                                               # X^p o full dephasing (Z basis)
    J = np.zeros((4, 4))
    for i in range(2):
        for o in range(2):
            J[2 * i + o, 2 * i + o] = (1 - P) if o == i else P
    return J


def choi_erased():                                              # output a fixed state |0><0| (erasure flag carries the info)
    return np.kron(I2, np.diag([1.0, 0.0]))


def ptrans_B(W, dA, dB):
    import cvxpy as cp
    return cp.partial_transpose(W, dims=[dA, dB], axis=1)


def kron_branch(Js):
    """Choi of the n-fold product in A^n (x) Q^n order, from per-use Chois in (A_i, Q_i) order."""
    n = len(Js)
    T = Js[0]
    for J in Js[1:]:
        T = np.kron(T, J)                                       # order A1 Q1 A2 Q2 ...
    T = T.reshape([2] * (4 * n))
    rows = [2 * i for i in range(n)] + [2 * i + 1 for i in range(n)]
    cols = [2 * n + c for c in rows]
    T = T.transpose(rows + cols)
    return T.reshape(4 ** n, 4 ** n)


def f_ppt(branches, n, M=2):
    """branches: list of (prob, Choi on A (x) Q) for one use; classical flag = branch index."""
    import cvxpy as cp
    dA = 2 ** n
    rho = cp.Variable((dA, dA), hermitian=True)
    cons = [rho >> 0, cp.real(cp.trace(rho)) == 1]
    obj = 0
    RI = cp.kron(rho, np.eye(dA))
    for z in itertools.product(range(len(branches)), repeat=n):
        w = float(np.prod([branches[i][0] for i in z]))
        Jz = w * kron_branch([branches[i][1] for i in z])
        W = cp.Variable((4 ** n, 4 ** n), hermitian=True)
        cons += [W >> 0, RI - W >> 0, RI - M * ptrans_B(W, dA, dA) >> 0, RI + M * ptrans_B(W, dA, dA) >> 0]
        obj += cp.real(cp.trace(Jz @ W))
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    return prob.value


if __name__ == "__main__":
    import json
    import pathlib
    out = {"sdp": "F_PPT(N, 2), Leung-Matthews PPT-assisted code fidelity, dense, flag-block-diagonal",
           "sanity": {"identity": f_ppt([(1.0, choi_id())], 1), "entanglement_breaking": f_ppt([(1.0, choi_noisy())], 1)},
           "rows": []}
    print(f"sanity identity            F_PPT = {out['sanity']['identity']:.6f}   (expect 1)")
    print(f"sanity EB (dephase+flip)   F_PPT = {out['sanity']['entanglement_breaking']:.6f}   (expect 1/2)")
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    lower = {1: 0.6464466094, 2: 0.6870387847, 3: 0.7070980316, 4: 0.7165773615}   # permqit 00c18d3d64
    for n in range(1, nmax + 1):
        fe = f_ppt([(0.5, choi_id()), (0.5, choi_erased())], n)
        fn = f_ppt([(0.5, choi_id()), (0.5, choi_noisy())], n)
        out["rows"].append({"n": n, "F_PPT_erasure": fe, "F_PPT_flagged": fn, "unassisted_lower": lower[n]})
        print(f"n={n}: F_PPT(erasure^n)={fe:.6f}   F_PPT(N~^n)={fn:.6f}   unassisted (authors) >= {lower[n]:.6f}"
              f"   gap {fn - lower[n]:+.4f}", flush=True)
    root = pathlib.Path(__file__).resolve().parents[2]
    (root / "evidence" / "superactivation" / "ppt_relaxation_probe.json").write_bytes(
        (json.dumps(out, indent=1) + "\n").encode("utf-8"))
