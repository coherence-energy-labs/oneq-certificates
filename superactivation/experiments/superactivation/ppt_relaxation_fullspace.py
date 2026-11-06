"""Proposition 11 of paper 4 on the FULL output space, exactly: the feasible point AND a dual certificate of the
PPT-assisted fidelity SDP, with no block decomposition assumed (referee finding W1, 2026-11-03).

The program is that of ppt_relaxation_probe.py (Leung-Matthews), F_PPT(N, M) = max Tr(J W) over (W, rho) with
Tr rho = 1, 0 <= W <= rho (x) 1_B and -rho (x) 1_B <= M W^{T_B} <= rho (x) 1_B, J = sum_ij |i><j| (x) N(|i><j|).
ppt_relaxation_probe.py and ppt_relaxation_exact.py both build ONE BLOCK PER FLAG PATTERN with a shared rho, which is the
decomposition the paper's proof derives, so neither could fail if that step were wrong; the probe also runs at one p.
Here every constraint is checked on the assembled operators of the whole space A^n (x) B^n:

  * the flagged channel N~_p: output (qubit, flag), 4 dims per use; the 50% erasure channel on a qubit: output C^2 (+)
    |e>, the erasure symbol a genuine ONE-dimensional sector (3 dims per use); n = 1, 2, 3 (full dimensions 8, 64, 512
    and 6, 36, 216); the value and the dual at each of six rational p in [0, 1] (the value is a polynomial of degree
    <= n in p, so n + 1 agreeing points already make it constant);
  * Smith and Yard's JOINT channel P_2 (x) A_4 at n = 1 (input 16, output 20, full dimension 320), exact in Q(sqrt 2):
    P_2 is the family's d = 2 member (Eq. (8), u = [[1, 1], [1, -1]], p1 = 1/(sqrt 2 + 1)), built from its definition
    and compared entrywise with ppt_family.choi_state; A_4 the 50% erasure channel on C^4.

PRIMAL: the paper's feasible point (rho = the logical qubit maximally mixed; the block of each pattern with a kept use
is (1/2) Phi on the first kept use (x) rho (x) 1 elsewhere on its sector; the all-erased block is (1/2) rho (x) 1), of
value 1 - 2^-(n+1).
DUAL: (Y, P, N, t) = (J - J_e, J_e^{T_B} / M, 0, 1 - 2^-n (1 - 1/M)), J_e the block of J on the all-erased sector. It
bounds every feasible (W, rho) by weak duality, Tr(J W) = Tr(Y W) + M Tr(P W^{T_B}) - M Tr(N W^{T_B})
<= Tr((Y + P + N)(rho (x) 1)) <= t, once Y, P, N >= 0, Y + M (P - N)^{T_B} >= J and Tr_B(Y + P + N) <= t 1 -- each
checked here as an exact PSD test on the full space. P >= 0 is exactly the statement that the all-erased block is PPT,
which is why the bound also holds for the joint channel (P_2 is PPT, Theorem 3). That J does not couple the all-erased
sector to any other (the flag is classical) is tested, not assumed.

Controls that must FAIL: the kept block without its 1/2; the all-erased block without its 1/2; each block scaled by
1/2 + 1/64; the dual with t - 1/64; the dual of a channel whose erased branch is the identity (not PPT); the joint dual
with P_2 at p1 = 1/2 (not PPT); and the PSD tester is calibrated on matrices of known sign, over Q and over Q(sqrt 2).
The probe's floating-point optima are compared with the exact values.

Provenance: the rational part is ported from an INDEPENDENT check of Proposition 11 written by an adversarial AI agent
from the paper's text alone (2026-11-03, it had not written the proposition), extended here with the dual at every p,
the joint channel in Q(sqrt 2), the two further controls and the probe comparison. Exact arithmetic throughout:
fractions.Fraction and QS2 (a + b sqrt 2); PSD by LDL^T with largest-diagonal pivoting.

  python ppt_relaxation_fullspace.py OUT.json        (seconds)
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import pathlib
import platform
import random
import sys
import time
from fractions import Fraction as Q

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBE = ROOT / "evidence" / "superactivation" / "ppt_relaxation_probe.json"
CODE_FILES = ("experiments/superactivation/ppt_relaxation_fullspace.py",
              "experiments/superactivation/ppt_relaxation_probe.py",
              "experiments/superactivation/shield_family/ppt_family.py")
HALF = Q(1, 2)
M = 2
PS = (Q(0), Q(1, 10), Q(1, 3), Q(1, 2), Q(3, 4), Q(1))


# ---------------------------------------------------------------- exact Q(sqrt 2)
class QS2:
    """a + b sqrt 2 with a, b rational: exact field arithmetic and an exact sign"""
    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a, self.b = Q(a), Q(b)

    @staticmethod
    def of(x) -> "QS2":
        return x if isinstance(x, QS2) else QS2(x, 0)

    def __add__(self, o):
        o = QS2.of(o)
        return QS2(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __neg__(self):
        return QS2(-self.a, -self.b)

    def __sub__(self, o):
        o = QS2.of(o)
        return QS2(self.a - o.a, self.b - o.b)

    def __rsub__(self, o):
        return QS2.of(o) - self

    def __mul__(self, o):
        o = QS2.of(o)
        return QS2(self.a * o.a + 2 * self.b * o.b, self.a * o.b + self.b * o.a)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = QS2.of(o)
        nrm = o.a * o.a - 2 * o.b * o.b                       # nonzero for o != 0: sqrt 2 is irrational
        return self * QS2(o.a / nrm, -o.b / nrm)

    def __rtruediv__(self, o):
        return QS2.of(o) / self

    def sign(self) -> int:
        a, b = self.a, self.b
        if b == 0:
            return (a > 0) - (a < 0)
        if a == 0:
            return (b > 0) - (b < 0)
        if a > 0 and b > 0:
            return 1
        if a < 0 and b < 0:
            return -1
        s = (a * a > 2 * b * b) - (a * a < 2 * b * b)             # |a| vs |b| sqrt 2
        return s if a > 0 else -s

    def __eq__(self, o):
        o = QS2.of(o)
        return self.a == o.a and self.b == o.b

    def __hash__(self):
        return hash((self.a, self.b))

    def __lt__(self, o):
        return (self - o).sign() < 0

    def __gt__(self, o):
        return (self - o).sign() > 0

    def __le__(self, o):
        return (self - o).sign() <= 0

    def __ge__(self, o):
        return (self - o).sign() >= 0

    def __float__(self):
        return float(self.a) + float(self.b) * math.sqrt(2)

    def __str__(self):
        return str(self.a) if self.b == 0 else f"{self.a} + {self.b} sqrt2"


def is_zero(x) -> bool:
    return x == 0


# ---------------------------------------------------------------- sparse operators on A^n (x) B^n
def full_index(avec, bvec, dA, dB):
    a = 0
    for x in avec:
        a = a * dA + x
    b = 0
    for x in bvec:
        b = b * dB + x
    return a * dB ** len(avec) + b


def tensor(per_use: list[dict], dA: int, dB: int) -> dict:
    """per_use[k]: {((a, b), (a2, b2)): value} on use k; the full operator on A^n (x) B^n, A = A_1..A_n first"""
    out = {}
    for combo in itertools.product(*[list(op.items()) for op in per_use]):
        v = 1
        av, bv, av2, bv2 = [], [], [], []
        for ((a, b), (a2, b2)), x in combo:
            v = v * x
            av.append(a)
            bv.append(b)
            av2.append(a2)
            bv2.append(b2)
        if not is_zero(v):
            key = (full_index(av, bv, dA, dB), full_index(av2, bv2, dA, dB))
            out[key] = out.get(key, 0) + v
    return {k: v for k, v in out.items() if not is_zero(v)}


def decode(idx, n, dB):
    return idx // dB ** n, idx % dB ** n


def ptrans_B(W: dict, n, dB) -> dict:
    out = {}
    for (r, c), v in W.items():
        a, b = decode(r, n, dB)
        a2, b2 = decode(c, n, dB)
        key = (a * dB ** n + b2, a2 * dB ** n + b)
        out[key] = out.get(key, 0) + v
    return {k: v for k, v in out.items() if not is_zero(v)}


def add(*terms) -> dict:
    out = {}
    for coef, Mx in terms:
        for k, v in Mx.items():
            out[k] = out.get(k, 0) + coef * v
    return {k: v for k, v in out.items() if not is_zero(v)}


def trace_prod(A: dict, B: dict):
    s = 0
    for (r, c), v in A.items():
        w = B.get((c, r))
        if w is not None:
            s = s + v * w
    return s


def is_psd(S: dict, dim: int):
    """exact PSD test of a real symmetric matrix (entries in Q or Q(sqrt 2)): LDL^T, largest diagonal pivot first"""
    for (r, c), v in S.items():
        if S.get((c, r), 0) != v:
            return False, f"not symmetric at {(r, c)}"
    rows: dict = {}
    for (r, c), v in S.items():
        rows.setdefault(r, {})[c] = v
    alive = set(range(dim))
    while alive:
        k = max(alive, key=lambda i: rows.get(i, {}).get(i, 0))
        dk = rows.get(k, {}).get(k, 0)
        if dk < 0:
            return False, "negative pivot"
        if is_zero(dk):                                          # the largest remaining diagonal is 0
            for i in alive:
                for j, v in rows.get(i, {}).items():
                    if j in alive and not is_zero(v):
                        return False, ("a negative diagonal entry" if i == j else
                                       "a zero diagonal entry with a nonzero off-diagonal one")
            return True, None
        rk = {j: v for j, v in rows.get(k, {}).items() if j in alive and j != k}
        alive.discard(k)
        for i, vik in rk.items():
            ri = rows.setdefault(i, {})
            f = vik / dk
            for j, vkj in rk.items():
                nv = ri.get(j, 0) - f * vkj
                if is_zero(nv):
                    ri.pop(j, None)
                else:
                    ri[j] = nv
        rows.pop(k, None)
        for i in rk:
            rows[i].pop(k, None)
    return True, None


def calibrate_psd() -> dict:
    """the tester on matrices of known sign: it must say yes and no correctly, over Q and over Q(sqrt 2)"""
    out = {"[[1,1],[1,1]] is psd": is_psd({(0, 0): Q(1), (0, 1): Q(1), (1, 0): Q(1), (1, 1): Q(1)}, 2)[0],
           "[[1,2],[2,1]] is not psd": not is_psd({(0, 0): Q(1), (0, 1): Q(2), (1, 0): Q(2), (1, 1): Q(1)}, 2)[0]}
    rnd = random.Random(11)
    V = [[Q(rnd.randint(-3, 3)) for _ in range(5)] for _ in range(12)]
    G = {(i, j): s for i in range(12) for j in range(12) if (s := sum(V[i][k] * V[j][k] for k in range(5))) != 0}
    out["rank-5 Gram 12x12 is psd"] = is_psd(G, 12)[0]
    A = [[V[i][k] for i in range(12)] for k in range(5)]          # a kernel vector of G, by exact elimination
    piv, r = [], 0
    for c in range(12):
        pr = next((i for i in range(r, 5) if A[i][c] != 0), None)
        if pr is None:
            continue
        A[r], A[pr] = A[pr], A[r]
        A[r] = [x / A[r][c] for x in A[r]]
        for i in range(5):
            if i != r and A[i][c] != 0:
                fac = A[i][c]
                A[i] = [x - fac * y for x, y in zip(A[i], A[r])]
        piv.append(c)
        r += 1
        if r == 5:
            break
    free = next(c for c in range(12) if c not in piv)
    v = [Q(0)] * 12
    v[free] = Q(1)
    for i, c in enumerate(piv):
        v[c] = -A[i][free]
    Gm = add((1, G), (-Q(1, 1000), {(i, j): v[i] * v[j] for i in range(12) for j in range(12) if v[i] * v[j] != 0}))
    out["that Gram minus 1/1000 v v^T, v in its kernel, is not psd"] = not is_psd(Gm, 12)[0]
    s2 = QS2(0, 1)
    out["[[1, sqrt2], [sqrt2, 2]] is psd (singular)"] = is_psd({(0, 0): QS2(1), (0, 1): s2, (1, 0): s2,
                                                               (1, 1): QS2(2)}, 2)[0]
    out["[[1, sqrt2], [sqrt2, 199/100]] is not psd"] = not is_psd({(0, 0): QS2(1), (0, 1): s2, (1, 0): s2,
                                                                  (1, 1): QS2(Q(199, 100))}, 2)[0]
    out["[[3 - 2 sqrt2]] is psd, [[2 sqrt2 - 3]] is not"] = (is_psd({(0, 0): QS2(3, -2)}, 1)[0]
                                                           and not is_psd({(0, 0): QS2(-3, 2)}, 1)[0])
    return out


# ---------------------------------------------------------------- the channels, one use each
def model_flagged() -> dict:
    """N~_p: input qubit; output (qubit q, flag f), b = 2 q + f; f = 0 kept (identity), f = 1 erased (X^p o dephasing)"""
    def J(p):
        e = {}
        for i in range(2):
            for j in range(2):
                e[((i, 2 * i), (j, 2 * j))] = HALF
            e[((i, 2 * i + 1), (i, 2 * i + 1))] = e.get(((i, 2 * i + 1), (i, 2 * i + 1)), 0) + HALF * (1 - p)
            k = 2 * (1 - i) + 1
            e[((i, k), (i, k))] = e.get(((i, k), (i, k)), 0) + HALF * p
        return {key: v for key, v in e.items() if v != 0}
    return {"name": "flagged channel N~_p", "dA": 2, "dB": 4, "J": J, "sector": {0: [0, 2], 1: [1, 3]},
            "enc": [0, 1], "kept_out": lambda x: [2 * x], "field": "Q"}


def model_erasure() -> dict:
    """the 50% erasure channel on a qubit: output C^2 (+) |e>, |e> = basis state 2, a one-dimensional sector"""
    def J(p):
        e = {}
        for i in range(2):
            for j in range(2):
                e[((i, i), (j, j))] = HALF
            e[((i, 2), (i, 2))] = HALF
        return e
    return {"name": "50% erasure channel (qubit)", "dA": 2, "dB": 3, "J": J, "sector": {0: [0, 1], 1: [2]},
            "enc": [0, 1], "kept_out": lambda x: [x], "field": "Q"}


def model_erased_branch_identity() -> dict:
    """CONTROL: a flagged channel whose erased branch is the identity, so the all-erased block is not PPT"""
    def J(p):
        return {((i, 2 * i + f), (j, 2 * j + f)): HALF for f in range(2) for i in range(2) for j in range(2)}
    return {"name": "control: erased branch = identity", "dA": 2, "dB": 4, "J": J, "sector": {0: [0, 2], 1: [1, 3]},
            "enc": [0, 1], "kept_out": lambda x: [2 * x], "field": "Q"}


def p2_choi(p1: QS2) -> dict:
    """J_P (Tr_B J_P = 1) of the family's d = 2 member, Eq. (8) with u = [[1, 1], [1, -1]], as {(r, c): QS2} on
    (R' R'')(B' B''), R' most significant; built from the definition, exactly"""
    u = [[1, 1], [1, -1]]
    inv_s2 = QS2(0, HALF)                                       # 1 / sqrt 2

    def mat(rows):
        return [[QS2.of(x) for x in row] for row in rows]

    def mul(A, B):
        return [[sum((A[i][k] * B[k][j] for k in range(4)), QS2()) for j in range(4)] for i in range(4)]

    def tr(A):
        return [[A[j][i] for j in range(4)] for i in range(4)]
    I4 = mat([[int(i == j) for j in range(4)] for i in range(4)])
    Du = mat([[u[i // 2][i % 2] if i == j else 0 for j in range(4)] for i in range(4)])
    F = mat([[1 if r == (c % 2) * 2 + c // 2 else 0 for c in range(4)] for r in range(4)])   # swap R'' <-> B''
    U10 = mul(Du, F)
    U11 = [row[:] for row in I4]
    for i, a in enumerate((0, 3)):                              # |aa> -> (1/sqrt d) sum_b u_ab |bb>
        for j, b in enumerate((0, 3)):
            U11[b][a] = u[i][j] * inv_s2
    sig = {0: mat([[Q(1, 4) if i == j else 0 for j in range(4)] for i in range(4)]),
           1: mat([[HALF if i == j and i in (0, 3) else 0 for j in range(4)] for i in range(4)])}
    Us = {0: [I4, U10], 1: [I4, U11]}
    rho = {}                                                     # on (R', B', R'', B''), index r' b' r'' b''
    for ell, w in ((0, 1 - p1), (1, p1)):
        for i in range(2):
            for j in range(2):
                tau = mul(mul(Us[ell][i], sig[ell]), tr(Us[ell][j]))      # real twists: dagger = transpose
                kr, kc = i * 2 + (i + ell) % 2, j * 2 + (j + ell) % 2      # |i, i+l><j, j+l| on R'B'
                for s in range(4):
                    for t in range(4):
                        v = w * HALF * tau[s][t]
                        if not is_zero(v):
                            key = (kr * 4 + s, kc * 4 + t)
                            rho[key] = rho.get(key, QS2()) + v

    def perm(x):                                                 # (r', b', r'', b'') -> (r', r'', b', b'')
        r1, b1, r2, b2 = (x >> 3) & 1, (x >> 2) & 1, (x >> 1) & 1, x & 1
        return ((r1 * 2 + r2) * 4) + (b1 * 2 + b2)
    return {(perm(r), perm(c)): 4 * v for (r, c), v in rho.items() if not is_zero(v)}


def model_joint(p1: QS2, name: str) -> dict:
    """one use of P_2 (x) A_4: input a = aP * 4 + aA (16), output b = bP * 5 + bA (20), bA = 4 the erasure symbol"""
    JP = p2_choi(p1)
    JA = {}
    for i in range(4):
        for j in range(4):
            JA[((i, i), (j, j))] = HALF
        JA[((i, 4), (i, 4))] = HALF
    J1 = {}
    for (rp, cp), vp in JP.items():
        aP, bP, aP2, bP2 = rp // 4, rp % 4, cp // 4, cp % 4
        for ((aA, bA), (aA2, bA2)), va in JA.items():
            J1[((aP * 4 + aA, bP * 5 + bA), (aP2 * 4 + aA2, bP2 * 5 + bA2))] = vp * va
    return {"name": name, "dA": 16, "dB": 20, "J": lambda p: J1, "JP": JP,
            "sector": {0: [bP * 5 + q for bP in range(4) for q in range(4)], 1: [bP * 5 + 4 for bP in range(4)]},
            "enc": [0, 1],                                       # P's input |0>, A_4's input |0>, |1> (a real isometry)
            "kept_out": lambda x: [bP * 5 + x for bP in range(4)],   # P's output traced: the identity on it
            "field": "Q(sqrt 2)"}


# ---------------------------------------------------------------- the paper's feasible point and the dual
def feasible_point(model: dict, n: int, kept_scale=HALF, erased_scale=HALF):
    dA, dB, enc = model["dA"], model["dB"], model["enc"]
    W = {}
    for z in itertools.product((0, 1), repeat=n):
        per_use = []
        for q in range(n):
            if 0 in z and q == z.index(0):                      # (1/2) Phi on the first kept use
                ko = {x: model["kept_out"](x) for x in (0, 1)}
                per_use.append({((enc[x], ko[x][e]), (enc[y], ko[y][e])): kept_scale * HALF
                                for e in range(len(ko[0])) for x in (0, 1) for y in (0, 1)})
            else:                                                # rho (x) 1 on the use's sector
                per_use.append({((a, b), (a, b)): HALF for a in enc for b in model["sector"][z[q]]})
        Wz = tensor(per_use, dA, dB)
        if 0 not in z:
            Wz = {k: erased_scale * v for k, v in Wz.items()}
        W = add((1, W), (1, Wz))
    rho_diag = {}
    for avec in itertools.product(enc, repeat=n):                # rho = (logical qubit maximally mixed)^(x) n
        a = 0
        for x in avec:
            a = a * dA + x
        rho_diag[a] = Q(1, 2 ** n)
    return W, rho_diag


def check_primal(model: dict, n: int, ps=PS, **kw) -> dict:
    dA, dB = model["dA"], model["dB"]
    dim = dA ** n * dB ** n
    W, rho_diag = feasible_point(model, n, **kw)
    RI = {(a * dB ** n + b, a * dB ** n + b): v for a, v in rho_diag.items() for b in range(dB ** n)}
    PT = ptrans_B(W, n, dB)
    conds = {}
    for name, S in (("W >= 0", W), ("rho (x) 1 - W >= 0", add((1, RI), (-1, W))),
                    ("rho (x) 1 - M W^TB >= 0", add((1, RI), (-M, PT))),
                    ("rho (x) 1 + M W^TB >= 0", add((1, RI), (M, PT)))):
        ok, why = is_psd(S, dim)
        conds[name] = True if ok else f"FAILS ({why})"
    vals = {str(pv): str(trace_prod(tensor([model["J"](pv)] * n, dA, dB), W)) for pv in ps}
    return {"feasible": all(v is True for v in conds.values()), "conditions": conds,
            "trace_rho": str(sum(rho_diag.values())), "value_at_p": vals,
            "value_constant_in_p": len(set(vals.values())) == 1, "full_dimension": dim}


def check_dual(model: dict, n: int, p=Q(1, 3), t_shift=Q(0)) -> dict:
    dA, dB = model["dA"], model["dB"]
    dim = dA ** n * dB ** n
    Jn = tensor([model["J"](p)] * n, dA, dB)
    erased_b = {full_index([0] * n, list(bv), dA, dB) for bv in itertools.product(model["sector"][1], repeat=n)}
    Je, Y, coupled = {}, {}, 0
    for (r, c), v in Jn.items():
        b, b2 = r % dB ** n, c % dB ** n
        if b in erased_b and b2 in erased_b:
            Je[(r, c)] = v
        elif b in erased_b or b2 in erased_b:
            coupled += 1                                         # J would couple the all-erased sector to another
        else:
            Y[(r, c)] = v
    P = {k: v / M for k, v in ptrans_B(Je, n, dB).items()}
    t = 1 - Q(1, 2 ** n) * (1 - Q(1, M)) + t_shift
    okY = is_psd(Y, dim)[0]
    okP = is_psd(P, dim)[0]
    slack = add((1, Y), (M, ptrans_B(P, n, dB)), (-1, Jn))      # Y + M (P - N)^{T_B} - J, N = 0
    okS = is_psd(slack, dim)[0]
    marg = {}
    for (r, c), v in add((1, Y), (1, P)).items():
        (a, b), (a2, b2) = decode(r, n, dB), decode(c, n, dB)
        if b == b2:
            marg[(a, a2)] = marg.get((a, a2), 0) + v
    mslack = add((1, {(a, a): t for a in range(dA ** n)}), (-1, marg))
    okM = is_psd(mslack, dA ** n)[0]
    return {"J_couples_all_erased_sector_to_another": coupled > 0, "Y >= 0": okY, "P >= 0": okP,
            "Y + M P^TB - J >= 0": okS, "Y + M P^TB == J": len(slack) == 0, "t 1 - Tr_B(Y + P) >= 0": okM,
            "t": str(t), "valid": coupled == 0 and okY and okP and okS and okM}


def git_blob(path: pathlib.Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def p2_crosscheck(JP: dict) -> dict:
    """the exact P_2 against ppt_family.choi_state (the repo's float construction), entrywise, and its channel and
    PPT properties exactly (the PPT property is Theorem 3 at d = 2)"""
    sys.path.insert(0, str(HERE / "shield_family"))
    import numpy as np
    import ppt_family
    ref = 4 * ppt_family.choi_state(ppt_family.hadamard2(), 1 / (1 + math.sqrt(2)))
    ours = np.zeros((16, 16))
    for (r, c), v in JP.items():
        ours[r, c] = float(v)
    trB = {}
    for (r, c), v in JP.items():
        if r % 4 == c % 4:
            trB[(r // 4, c // 4)] = trB.get((r // 4, c // 4), QS2()) + v
    unit = all(trB.get((i, j), QS2()) == (1 if i == j else 0) for i in range(4) for j in range(4))
    return {"max_abs_difference_vs_ppt_family_choi_state": float(np.abs(ours - ref).max()),
            "partial_trace_over_output_is_identity_exactly": unit, "J_P >= 0 exactly": is_psd(JP, 16)[0],
            "J_P^TB >= 0 exactly (Theorem 3, d = 2)": is_psd(ptrans_B(JP, 1, 4), 16)[0]}


def main(argv) -> int:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    t0 = time.time()
    rec = {"schema": "oneq.superactivation.ppt-relaxation-fullspace/1",
           "sdp": "F_PPT(N, 2) of ppt_relaxation_probe.py (Leung-Matthews), on the FULL output space, no blocks assumed",
           "supersedes": "ppt_relaxation_exact.json (one block per flag pattern, feasible point only)",
           "calibration": calibrate_psd(), "rows": [], "joint": None, "controls": []}
    print("calibration", rec["calibration"], flush=True)
    probe = {r["n"]: r for r in json.loads(PROBE.read_text(encoding="utf-8"))["rows"]}
    for n in (1, 2, 3):
        for mk, key in ((model_flagged, "F_PPT_flagged"), (model_erasure, "F_PPT_erasure")):
            m = mk()
            r = {"n": n, "channel": m["name"], "field": m["field"], "target": str(1 - Q(1, 2 ** (n + 1))),
                 "block_decomposition_assumed": False, "primal": check_primal(m, n),
                 "dual_at_p": {str(pv): check_dual(m, n, p=pv) for pv in PS}}
            if n in probe:
                exact = Q(r["target"])
                r["probe_measured"] = probe[n][key]
                r["probe_minus_exact"] = probe[n][key] - float(exact)
            rec["rows"].append(r)
            print(json.dumps({"n": n, "channel": m["name"], "feasible": r["primal"]["feasible"],
                              "values": sorted(set(r["primal"]["value_at_p"].values())),
                              "dual_valid_at_every_p": all(d["valid"] for d in r["dual_at_p"].values()),
                              "seconds": round(time.time() - t0, 1)}), flush=True)
    p1 = QS2(-1, 1)                                              # 1 / (sqrt 2 + 1) = sqrt 2 - 1
    jm = model_joint(p1, "Smith and Yard's joint channel P_2 (x) A_4")
    rec["joint"] = {"n": 1, "channel": jm["name"], "field": jm["field"], "target": "3/4",
                    "block_decomposition_assumed": False, "P_2": p2_crosscheck(jm["JP"]),
                    "primal": check_primal(jm, 1, ps=(Q(0),)), "dual": check_dual(jm, 1)}
    rec["joint"]["primal"]["value_at_p"] = {"(no free p)": rec["joint"]["primal"]["value_at_p"]["0"]}
    print("joint", json.dumps({k: rec["joint"][k] for k in ("P_2",)}), rec["joint"]["primal"]["feasible"],
          rec["joint"]["primal"]["value_at_p"], rec["joint"]["dual"]["valid"], round(time.time() - t0, 1), flush=True)
    fl, er = model_flagged(), model_erasure()
    tiny = Q(1, 64)
    controls = [
        ("kept block without its 1/2 (n = 2, flagged)", check_primal(fl, 2, kept_scale=Q(1))["feasible"]),
        ("kept block without its 1/2 (n = 2, erasure)", check_primal(er, 2, kept_scale=Q(1))["feasible"]),
        ("all-erased block without its 1/2 (n = 2, flagged)", check_primal(fl, 2, erased_scale=Q(1))["feasible"]),
        ("all-erased block without its 1/2 (n = 2, erasure)", check_primal(er, 2, erased_scale=Q(1))["feasible"]),
        ("kept block scaled 1/2 + 1/64 (n = 2, erasure)", check_primal(er, 2, kept_scale=HALF + tiny)["feasible"]),
        ("kept block scaled 1/2 + 1/64 (n = 3, flagged)", check_primal(fl, 3, kept_scale=HALF + tiny)["feasible"]),
        ("all-erased block scaled 1/2 + 1/64 (n = 3, flagged)",
         check_primal(fl, 3, erased_scale=HALF + tiny)["feasible"]),
        ("all-erased block scaled 1/2 + 1/64 (n = 2, erasure)",
         check_primal(er, 2, erased_scale=HALF + tiny)["feasible"]),
        ("dual with t - 1/64 (n = 2, flagged)", check_dual(fl, 2, t_shift=-tiny)["valid"]),
        ("dual with t - 1/64 (n = 3, erasure)", check_dual(er, 3, t_shift=-tiny)["valid"]),
        ("dual of a channel whose erased branch is the identity, not PPT (n = 2)",
         check_dual(model_erased_branch_identity(), 2)["valid"]),
        ("joint dual with t - 1/64 (n = 1)", check_dual(jm, 1, t_shift=-tiny)["valid"]),
        ("joint dual with P_2 at p1 = 1/2, not PPT (n = 1)",
         check_dual(model_joint(QS2(HALF), "control: P_2 at p1 = 1/2"), 1)["valid"]),
    ]
    for name, passed in controls:
        rec["controls"].append({"control": name, "passed_but_must_fail": bool(passed)})
        print("control", name, "->", "PASSED (BAD)" if passed else "fails (good)", flush=True)

    def row_ok(r):
        vals = set(r["primal"]["value_at_p"].values())
        return (r["primal"]["feasible"] and vals == {r["target"]} and len(r["dual_at_p"]) == len(PS)
                and all(d["valid"] and d["t"] == r["target"] for d in r["dual_at_p"].values())
                and abs(r.get("probe_minus_exact", 0.0)) < 1e-6)
    j = rec["joint"]
    joint_ok = (j["primal"]["feasible"] and set(j["primal"]["value_at_p"].values()) == {"3/4"} and j["dual"]["valid"]
                and j["dual"]["t"] == "3/4" and j["P_2"]["max_abs_difference_vs_ppt_family_choi_state"] < 1e-14
                and j["P_2"]["partial_trace_over_output_is_identity_exactly"] and j["P_2"]["J_P >= 0 exactly"]
                and j["P_2"]["J_P^TB >= 0 exactly (Theorem 3, d = 2)"])
    ok = (all(rec["calibration"].values()) and all(row_ok(r) for r in rec["rows"]) and joint_ok
          and not any(c["passed_but_must_fail"] for c in rec["controls"]))
    rec["verdict"] = "EXACT" if ok else "FAILED"
    rec["seconds"] = round(time.time() - t0, 1)
    rec["provenance"] = {"code_git_blobs": {f: git_blob(ROOT / f) for f in CODE_FILES},
                         "python": platform.python_version(),
                         "ported_from": "an independent adversarial check of Proposition 11 (s1_exact.py, 2026-11-03)"}
    pathlib.Path(argv[0]).write_bytes((json.dumps(rec, indent=1) + "\n").encode("utf-8"))
    print("VERDICT", rec["verdict"], "in", rec["seconds"], "s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
