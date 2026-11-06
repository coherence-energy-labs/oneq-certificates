"""R170: the bar -- rigorous upper bounds on F_c(A_D^{(x)m}, 2), explicit codes, and the gap between them.

  python bar_evidence.py [MMAX_CERT=6] [MMAX_NUM=8]    -> evidence/superactivation/r170_bar/bar.json

1. PAIR BOUND (DERIVED, every m and D): F_c(A_D^m, 2) <= 3/4 - 2^-(m+2). Complementary Bob sets pair up; each pair's
   two decoded qubits form a 1 -> 2 cloner output, so F_S + F_{S^c} <= lambda_max(Pi_01 + Pi_02) = 3/2; the pair
   (empty, all) gives at most 1/4 + 1. So no finite m reaches 3/4.
2. MONOGAMY BOUND R(m) (monogamy_bound.py), CERTIFIED here for m <= MMAX_CERT in exact arithmetic:
   every cut's direction q is rounded to a dyadic grid, its right-hand side mu is a dyadic number >= the float
   lambda_max + 1e-10, and mu*1 - H(q) is PROVED positive definite by exact integer leading minors (Sylvester,
   python-flint) on the full (r + 1)-qubit matrix (entries in Z/2^31). The LP dual is rounded to rationals and the
   bound is  sum_c y_c mu_c + sum_k max(r_k lo_k, r_k hi_k),  r = w - A^T y exactly, which holds for ANY y >= 0.
3. EXPLICIT CODES (lower bounds): m = 2, the 1 -> 2 cloner with its ancilla in share 2 (D = 4): 11/16;
   m = 3, the 1 -> 3 Werner cloner, one clone per share, its spin-1 ancilla shared by the ((2,3)) qutrit threshold
   scheme (D = 6): a single share holds one clone (2/3); any two shares recover the ancilla and hold the full
   complement of (R, Q_i), whose fidelity is the cloning-region partner (5 + 2 sqrt 6)/12 -- total (9 + sqrt 6)/16,
   EQUAL to R(3). The best see-saw codes (erasure_codes.py) are re-scored from committed isometries.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys
from fractions import Fraction

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EV = ROOT / "evidence" / "superactivation" / "r170_bar"
sys.path.insert(0, str(HERE))
import erasure_codes as ec  # noqa: E402
import monogamy_bound as mb  # noqa: E402

QBITS = 30          # cut directions on the 2^-30 grid
MUBITS = 31         # right-hand sides on the 2^-31 grid (H(q) entries are multiples of 2^-31)


# ------------------------------------------------------------------ exact verification of a cut

def phi_pair_sum_int(nq: int, groups) -> list:
    """integer matrices 2*sum_{i in group} Pi_{0 i} on nq qubits (qubit 0 = R), one per group; entries 0/1"""
    dim = 2 ** nq
    mats = []
    for grp in groups:
        M = np.zeros((dim, dim), dtype=np.int64)
        for i in grp:
            sa, sb = nq - 1, nq - 1 - i
            for x in range(dim):
                if ((x >> sa) & 1) != ((x >> sb) & 1):
                    continue
                M[x, x] += 1
                M[x, x ^ (1 << sa) ^ (1 << sb)] += 1
        mats.append(M)
    return mats


def leading_minors_positive(Z: np.ndarray) -> bool:
    """exact Sylvester criterion on an integer symmetric matrix (python-flint fmpz determinants)"""
    import flint
    n = Z.shape[0]
    rows = [[int(v) for v in r] for r in Z]
    for s in range(1, n + 1):
        if flint.fmpz_mat([r[:s] for r in rows[:s]]).det() <= 0:
            return False
    return True


def certify_cut(lam, q_float):
    """round q, bound lambda_max of H(q) by a dyadic mu, and PROVE mu*1 - H(q) > 0. Returns (q_hat, mu) as Fractions."""
    sizes, mults = mb.partition_groups(lam)
    qn = np.maximum(np.asarray(q_float, dtype=float), 0.0)
    if qn.max() <= 0:
        raise ValueError("zero direction")
    qn = qn / qn.max()
    Q = [int(round(x * 2 ** QBITS)) for x in qn]
    q_hat = [Fraction(v, 2 ** QBITS) for v in Q]
    h = mb.support(sizes, mults, [float(x) for x in q_hat])
    mu = Fraction(math.ceil((h + 1e-10) * 2 ** MUBITS), 2 ** MUBITS)
    r = len(lam)
    groups, start = [], 1
    for n in mults:
        groups.append(list(range(start, start + n)))
        start += n
    mats = phi_pair_sum_int(r + 1, groups)
    # 2^31 (mu - H) = mu*2^31 - sum_j Q_j (2 Pi-sum_j)   since H = sum_j (Q_j/2^30) (2 Pi-sum_j)/2
    T = -sum(Qj * Mj for Qj, Mj in zip(Q, mats))
    T = T + int(mu * 2 ** MUBITS) * np.eye(2 ** (r + 1), dtype=np.int64)
    if not leading_minors_positive(T):
        raise RuntimeError(f"cut {lam} q={qn} failed its exact positivity proof")
    return q_hat, mu


def certified_bound(m: int, log=None):
    """run the cutting-plane LP, then certify every cut exactly and bound the LP by a rational dual"""
    from scipy.optimize import linprog
    value, F, cuts = mb.relaxation(m, rounds=80)
    certified = []
    for lam, q, _ in cuts:
        q_hat, mu = certify_cut(lam, q)
        certified.append((lam, q_hat, mu))
    w = [Fraction(math.comb(m, k), 2 ** m) for k in range(m + 1)]
    rows, rhs = [], []
    for lam, q_hat, mu in certified:
        sizes, mults = mb.partition_groups(lam)
        row = [Fraction(0)] * (m + 1)
        for qj, a, n in zip(q_hat, sizes, mults):
            row[a] += qj * n
        rows.append(row)
        rhs.append(mu)
    for k in range(m):
        row = [Fraction(0)] * (m + 1)
        row[k], row[k + 1] = Fraction(1), Fraction(-1)
        rows.append(row)
        rhs.append(Fraction(0))
    A = np.array([[float(x) for x in r] for r in rows])
    b = np.array([float(x) for x in rhs])
    bounds = [(0.25, 0.25)] + [(0.25, 1.0)] * m
    res = linprog(-np.array([float(x) for x in w]), A_ub=A, b_ub=b, bounds=bounds, method="highs")
    y = np.maximum(-np.asarray(res.ineqlin.marginals), 0.0)
    yq = [Fraction(int(round(v * 2 ** 50)), 2 ** 50) for v in y]
    resid = [w[k] - sum(yq[c] * rows[c][k] for c in range(len(rows))) for k in range(m + 1)]
    lo = [Fraction(1, 4)] * (m + 1)
    hi = [Fraction(1, 4)] + [Fraction(1)] * m
    bound = sum(yc * bc for yc, bc in zip(yq, rhs)) + sum(r * (hi[k] if r > 0 else lo[k]) for k, r in enumerate(resid))
    if log:
        log(f"   m={m}: float LP {value:.10f}, certified {float(bound):.10f}, {len(certified)} cuts proved")
    return {"m": m, "float_lp": value, "certified_upper_bound": float(bound),
            "certified_upper_bound_exact": f"{bound.numerator}/{bound.denominator}",
            "profile": [float(x) for x in F], "cuts_proved": len(certified),
            "max_matrix_dim": 2 ** (1 + max(len(lam) for lam, _, _ in certified))}


# ------------------------------------------------------------------ explicit codes

def werner_singlet(m: int) -> np.ndarray:
    """the unique SU(2) singlet in R (spin 1/2) (x) Sym^m (spin m/2, Dicke basis) (x) A (spin (m-1)/2):
    the 1 -> m Werner cloner with R as its reference. Returns psi[r, dicke, a]."""
    dims = [2, m + 1, m]
    J2s = [1, m, m - 1]
    tot = None
    for a in range(3):
        Ja = None
        for j, j2 in enumerate(J2s):
            ops = [np.eye(d) for d in dims]
            ops[j] = mb.spin_matrices(j2)[a]
            t = mb._kron_all(ops)
            Ja = t if Ja is None else Ja + t
        sq = Ja @ Ja
        tot = sq if tot is None else tot + sq
    w, v = np.linalg.eigh((tot + tot.conj().T) / 2)
    if not (abs(w[0]) < 1e-10 and w[1] > 0.5):
        raise RuntimeError(f"singlet not unique: {w[:3]}")
    return v[:, 0].reshape(dims)


def dicke_embedding(m: int) -> np.ndarray:
    """columns |m/2, m/2 - w> = uniform superposition of weight-w strings, in the qubit basis (qubit 0 most significant)"""
    E = np.zeros((2 ** m, m + 1))
    for x in range(2 ** m):
        E[x, bin(x).count("1")] = 1.0
    return E / np.sqrt(E.sum(axis=0))


def threshold_23() -> np.ndarray:
    """the pure ((2,3)) qutrit threshold scheme |k> -> 3^-1/2 sum_j |j, j+k, j+2k> as a 27 x 3 isometry"""
    V = np.zeros((27, 3))
    for k in range(3):
        for j in range(3):
            V[9 * j + 3 * ((j + k) % 3) + (j + 2 * k) % 3, k] = 1 / np.sqrt(3)
    return V


def code_cloner_threshold_m3():
    """psi[r, s1, s2, s3, e] with share s_i = (clone qubit i, ancilla share i), D = 6, e = 1"""
    W = werner_singlet(3)                                 # (2, 4, 3)
    T = np.einsum("rda,qd->rqa", W, dicke_embedding(3))   # (2, 8, 3): clones as 3 qubits
    T = np.einsum("rqa,ka->rqk", T, threshold_23())       # (2, 8, 27): ancilla as 3 qutrit shares
    T = T.reshape(2, 2, 2, 2, 3, 3, 3)                    # r, q1, q2, q3, a1, a2, a3
    T = T.transpose(0, 1, 4, 2, 5, 3, 6).reshape(2, 6, 6, 6, 1)
    return T / np.linalg.norm(T)


def code_cloner_m2():
    """the 1 -> 2 cloner, clone 1 -> share 1 (padded to D = 4), clone 2 and the ancilla -> share 2"""
    W = werner_singlet(2)                                 # (2, 3, 2)
    T = np.einsum("rda,qd->rqa", W, dicke_embedding(2)).reshape(2, 2, 2, 2)   # r, q1, q2, a
    out = np.zeros((2, 4, 4, 1), dtype=complex)
    out[:, :2, :, 0] = T.reshape(2, 2, 4)
    return out / np.linalg.norm(out)


def score_psi(psi: np.ndarray, m: int, iters: int = 3000):
    brs = ec.make_branches(m)
    F = ec.decode_all(brs, psi, m, iters=iters, reduce_above=64)
    return F, ec.per_size(brs, m)


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(mmax_cert: int = 6, mmax_num: int = 8):
    import psutil
    if psutil.virtual_memory().available < 4e9:
        raise SystemExit("REFUSING: under 4 GB available")
    out = {"lane": "R170", "pair_bound": {m: ec.pair_bound(m) for m in range(1, 31)}, "certified": [], "numerical": []}
    for m in range(2, mmax_cert + 1):
        out["certified"].append(certified_bound(m, log=print))
    for m in range(mmax_cert + 1, mmax_num + 1):
        v, F, cuts = mb.relaxation(m, rounds=80)
        out["numerical"].append({"m": m, "float_lp": v, "profile": [float(x) for x in F], "cuts": len(cuts)})
        print(f"   m={m}: float LP {v:.10f} (not certified)")
    F2, p2 = score_psi(code_cloner_m2(), 2)
    F3, p3 = score_psi(code_cloner_threshold_m3(), 3)
    out["constructions"] = {
        "m2_cloner_D4": {"F": F2, "per_size": p2, "exact": "11/16"},
        "m3_werner_threshold_D6": {"F": F3, "per_size": p3, "exact": "(9 + sqrt 6)/16",
                                   "exact_value": (9 + math.sqrt(6)) / 16}}
    print(f"   constructions: m=2 {F2:.10f} (11/16 = {11/16:.10f}); m=3 {F3:.10f} ((9+sqrt6)/16 = {(9 + math.sqrt(6)) / 16:.10f})")
    codes = []
    for path, m, D, e in sorted((p, int(p.stem.split("_")[1][1:]), int(p.stem.split("_")[2][1:]), int(p.stem.split("_")[3][1:]))
                                for p in EV.glob("seesaw_m*_D*_e*.npy")):
        V = np.load(path)
        F, prof = score_psi(ec.psi_from_isometry(V, m, D, e), m)
        codes.append({"file": path.name, "sha256": sha(path), "m": m, "D": D, "e": e, "F": F, "per_size": prof,
                      "isometry_error": ec.isometry_error(V)})
        print(f"   see-saw code {path.name}: F = {F:.10f}")
    out["seesaw_codes"] = codes
    EV.mkdir(parents=True, exist_ok=True)
    (EV / "bar.json").write_bytes((json.dumps(out, indent=1) + "\n").encode("utf-8"))
    return out


if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:3]]
    main(*a)
