"""R178 (b): the 50% erasure channel's coding fidelity F_c(A^m, 2), m = 1..4, and its supremum over m.

R142's open item (b): Result 1 of arXiv 2604.27042 gives sup_m F_c(A^m, 2) <= 3/4 for the antidegradable 50%
erasure channel A. If the supremum were BELOW 3/4, one use of P_d (x) A_{2d} (which reaches 1/2 + (1-p)/4 -> 3/4,
R141/R142) could beat the smaller bar. This lane decides what can be decided.

THEOREM (proved here; the one SDP in it is certified by an exact rational dual):
  (i)  Broadcast lemma. For ANY channel R -> B (x) C from a qubit, F_e(B) + F_e(C) <= 3/2, where F_e(X) is the
       entanglement fidelity of the marginal channel R -> X against |Phi_2>. The SDP max Tr(J W) s.t. J >= 0,
       Tr_{BC} J = 1 has the dual feasible point Y = (3/4) 1, because W = (Phi_RB (x) 1_C + Phi_RC (x) 1_B)/2 has
       spectrum {3/4 (x2), 1/4 (x2), 0 (x4)} EXACTLY (checked in rationals); the symmetric universal cloner attains it.
  (ii) For m uses of a 50% erasure channel with ANY input dimension D and ANY encoder (mixed allowed):
            F_c(A^m, 2) <= 3/4 - 2^-(m+2).
       Proof: F = 2^-m sum_z F_z over erasure patterns z (Bob knows z). Pair z with its complement: Bob's decoders on
       the kept set S and on its complement S' compose with the encoder to a broadcast channel R -> B (x) C, so
       F_S + F_S' <= 3/2 by (i). The pair (everything kept, nothing kept) does better: F_all <= 1 and F_none = 1/4
       exactly (Bob's output is independent of R), so it sums to <= 5/4. Hence
            F <= 2^-m [ (2^(m-1) - 1) 3/2 + 5/4 ] = 3/4 - 2^-(m+2).
       So EVERY finite m is STRICTLY below 3/4, by at least 2^-(m+2).
  (iii) The bound is attained at m = 1 (5/8, trivially) and at m = 2 for every D >= 4 (11/16, by the Buzek-Hillery
       cloner: clone B into use 1, clone C and the cloner's ancilla into use 2; checked in exact arithmetic here).
       A_{2d} has D = 2d >= 4 for every shield dimension d >= 2, so F_c(A_{2d}^2, 2) = 11/16 EXACTLY.

COMPUTED (see-saw, lower bounds only): m = 2 with D = 2, m = 3 with D = 2 and 4, m = 4 with D = 2: exact SDP decoders
(cvxpy/Clarabel) per erasure pattern, exact dual encoder steps (twogroup.encoder_exact), restarts.

  python r178_erasure_sup.py [--quick]          writes evidence/superactivation/r178_erasure_sup.json
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import pathlib
import sys
import time
from fractions import Fraction

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def _rel(p):
    """p relative to ROOT when it lies inside it, else p itself: relative_to raises on a path outside the
    root, and a scratch output path would crash the run after its work was done"""
    return p.relative_to(ROOT) if p.is_relative_to(ROOT) else p

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "general_codes"))
from oneq import superactivation as sa  # noqa: E402

OUT = ROOT / "evidence" / "superactivation" / "r178_erasure_sup.json"


# ----------------------------------------------------------------------------------------------- exact pieces (pure)
def pairing_bound(m: int) -> Fraction:
    """the theorem's upper bound on F_c(A^m, 2) for the 50% erasure channel, any input dimension"""
    if m < 1:
        raise ValueError("m >= 1")
    return Fraction(1, 2 ** m) * ((2 ** (m - 1) - 1) * Fraction(3, 2) + Fraction(5, 4))


def _phi_proj(order: str) -> list:
    """|Phi><Phi| on two qubits of R B C (order 'RB' or 'RC') tensored with 1 on the third, as an exact 8x8 matrix
    of Fractions; basis index r*4 + b*2 + c"""
    M = [[Fraction(0)] * 8 for _ in range(8)]
    for r, b, c, r2, b2, c2 in itertools.product((0, 1), repeat=6):
        if order == "RB":
            val = Fraction(1, 2) if (r == b and r2 == b2 and c == c2) else Fraction(0)
        else:
            val = Fraction(1, 2) if (r == c and r2 == c2 and b == b2) else Fraction(0)
        M[r * 4 + b * 2 + c][r2 * 4 + b2 * 2 + c2] += val
    return M


def broadcast_objective() -> list:
    """W such that F_e(B) + F_e(C) = Tr(J W) for a channel Choi J (unnormalised, Tr_BC J = 1_R)"""
    A, B = _phi_proj("RB"), _phi_proj("RC")
    return [[(A[i][j] + B[i][j]) / 2 for j in range(8)] for i in range(8)]


def _exact_psd(M: list) -> bool:
    """exact PSD test of a rational symmetric matrix: LDL^T with symmetric pivoting on Fractions"""
    A = [row[:] for row in M]
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i][j] != A[j][i]:
                return False
    idx = list(range(n))
    while idx:
        piv = max(idx, key=lambda k: A[k][k])
        d = A[piv][piv]
        if d < 0:
            return False
        if d == 0:
            if any(A[piv][k] != 0 for k in idx):
                return False
            idx.remove(piv)
            continue
        rest = [k for k in idx if k != piv]
        for a in rest:
            for b in rest:
                A[a][b] -= A[a][piv] * A[piv][b] / d
        idx = rest
    return True


def broadcast_certificate() -> dict:
    """exact dual certificate for lemma (i): y 1 - W >= 0 with y = 3/4 gives F_B + F_C <= Tr(Y) = 2y = 3/2;
    and the tightness check: (3/4 - eps) 1 - W is NOT PSD (so the dual value is exactly the max eigenvalue)"""
    W = broadcast_objective()
    y = Fraction(3, 4)
    ok = _exact_psd([[(y if i == j else 0) - W[i][j] for j in range(8)] for i in range(8)])
    y2 = Fraction(3, 4) - Fraction(1, 10 ** 12)
    tight = not _exact_psd([[(y2 if i == j else 0) - W[i][j] for j in range(8)] for i in range(8)])
    trW = sum(W[i][i] for i in range(8))
    return {"dual_y": str(y), "bound": str(2 * y), "dual_feasible_exact": ok, "dual_tight_exact": tight,
            "trace_W": str(trW)}


def cloner_isometry() -> np.ndarray:
    """Buzek-Hillery universal 1->2 qubit cloner as an isometry C^2 -> B (x) C (x) E (qubits, index b*4 + c*2 + e):
       |0> -> sqrt(2/3)|00>|0> + sqrt(1/6)(|01> + |10>)|1>,  |1> -> sqrt(2/3)|11>|1> + sqrt(1/6)(|01> + |10>)|0>"""
    V = np.zeros((8, 2))
    a, b = math.sqrt(2 / 3), math.sqrt(1 / 6)
    V[0b000, 0] = a
    V[0b011, 0] = b
    V[0b101, 0] = b
    V[0b111, 1] = a
    V[0b010, 1] = b
    V[0b100, 1] = b
    return V


def cloner_split_fidelities() -> dict:
    """entanglement fidelities of the cloner split used for m = 2: use 1 = B, use 2 = (C, E). Exact in closed form:
    F(B) = F(C) = 3/4, F(B,C,E) = 1, F(nothing) = 1/4; F(C,E) <= 3/2 - F(B) and >= F(C), so = 3/4.
    Returned in floats for the record; `cloner_exact_fb` gives F(B) as an exact rational."""
    V = cloner_isometry()
    psi = np.zeros((2, 8))
    for r in range(2):
        psi[r] = V[:, r] / math.sqrt(2)                   # |Phi> through V: sum_r |r>_R V|r> / sqrt 2
    rho = np.einsum("ri,sj->risj", psi, psi).reshape(16, 16)   # R (x) BCE
    T = rho.reshape(2, 2, 2, 2, 2, 2, 2, 2)                 # r b c e ; r' b' c' e'
    rho_RB = np.einsum("rbcesfce->rbsf", T).reshape(4, 4)
    phi = np.zeros(4)
    phi[0] = phi[3] = 1 / math.sqrt(2)
    return {"F_B": float(phi @ rho_RB @ phi)}


def cloner_exact_fb() -> Fraction:
    """F(B) = <Phi| rho_RB |Phi> with the cloner's amplitudes as exact rationals of their squares:
    the amplitudes are sqrt(2/3) and sqrt(1/6), and every product entering <Phi|rho|Phi> pairs them as
    a*a = 2/3, b*b = 1/6 or a*b = 1/3 (sqrt(2/3 * 1/6) = 1/3), so the value is rational. Enumerated symbolically."""
    # amplitude squares and cross products in exact form, keyed by (x, y) for x, y in {"a", "b"}
    prod = {("a", "a"): Fraction(2, 3), ("b", "b"): Fraction(1, 6), ("a", "b"): Fraction(1, 3),
            ("b", "a"): Fraction(1, 3)}
    amp = {0: {0b000: "a", 0b011: "b", 0b101: "b"}, 1: {0b111: "a", 0b010: "b", 0b100: "b"}}
    # <Phi_RB| rho_RB |Phi_RB> = (1/2)(1/2) sum_{r,r'} sum_{c,e} <r_B c e | V | r> <r' | V^dag | r'_B c e>
    tot = Fraction(0)
    for r in (0, 1):
        for r2 in (0, 1):
            for c in (0, 1):
                for e in (0, 1):
                    i1 = (r << 2) | (c << 1) | e
                    i2 = (r2 << 2) | (c << 1) | e
                    if i1 in amp[r] and i2 in amp[r2]:
                        tot += prod[(amp[r][i1], amp[r2][i2])]
    return tot / 4


# ------------------------------------------------------------------------------------------- dense numerical model
def _ptrace_erased(X: np.ndarray, m: int, D: int, z) -> np.ndarray:
    T = X.reshape([2] + [D] * m + [2] + [D] * m)
    letters = "abcdefghijklmnopqrstuvwxyz"
    left = ["R"] + [letters[u] for u in range(m)]
    right = ["Q"] + [letters[u].upper() if z[u] else letters[u] for u in range(m)]
    outl = ["R"] + [letters[u] for u in range(m) if z[u]] + ["Q"] + [letters[u].upper() for u in range(m) if z[u]]
    Y = np.einsum("".join(left) + "".join(right) + "->" + "".join(outl), T)
    dS = D ** sum(z)
    return Y.reshape(2 * dS, 2 * dS)


def states(X, m, D):
    return {z: _ptrace_erased(X, m, D, z) for z in itertools.product((0, 1), repeat=m)}


def _embed_adjoint(Dz: np.ndarray, m: int, D: int, z) -> np.ndarray:
    """the operator on R (x) (C^D)^m whose trace against X equals Tr(R_S Dz): Dz on R and the kept uses, 1 elsewhere"""
    k = sum(z)
    T = Dz.reshape([2] + [D] * k + [2] + [D] * k)
    letters = "abcdefghijklmnopqrstuvwxyz"
    kept = [u for u in range(m) if z[u]]
    inl = "R" + "".join(letters[u] for u in kept) + "Q" + "".join(letters[u].upper() for u in kept)
    eye = np.eye(D)
    ops, subs = [T], [inl]
    for u in range(m):
        if not z[u]:
            ops.append(eye)
            subs.append(letters[u] + letters[u].upper())
    outs = "R" + "".join(letters[u] for u in range(m)) + "Q" + "".join(letters[u].upper() for u in range(m))
    G = np.einsum(",".join(subs) + "->" + outs, *ops)
    return G.reshape(2 * D ** m, 2 * D ** m)


def decode(X, m, D, warm=None, iters=60):
    decs, F = {}, 0.0
    for z, Rz in states(X, m, D).items():
        d = D ** sum(z)
        Dz = warm[z] if warm is not None else np.eye(2 * d)[None] / 2
        for _ in range(iters):
            Dz = sa.power_step(Rz[None], Dz, d)
        decs[z] = Dz
        F += float(np.real(np.trace(Rz @ Dz[0]))) / 4 / 2 ** m
    return decs, F


def per_pattern(X, decs, m, D):
    return {z: float(np.real(np.trace(Rz @ decs[z][0]))) / 4 for z, Rz in states(X, m, D).items()}


def gradient(decs, m, D):
    return sa.herm(sum(_embed_adjoint(decs[z][0], m, D, z) for z in decs) / 4 / 2 ** m)


def random_encoder(m, D, rng, rank=2):
    M = D ** m
    K = rng.normal(size=(2 * M, rank)) + 1j * rng.normal(size=(2 * M, rank))
    X = K @ K.conj().T
    T = X.reshape(2, M, 2, M).trace(axis1=1, axis2=3)
    tw, tv = np.linalg.eigh(sa.herm(T))
    Ti = np.kron((tv / np.sqrt(tw)) @ tv.conj().T, np.eye(M))
    return sa.herm(Ti @ X @ Ti)


def cloner_encoder_m2_D4() -> np.ndarray:
    """X for the m = 2, D = 4 cloner code: use 1 = (B, |0>) padded to C^4, use 2 = (C, E)"""
    V = cloner_isometry()                                    # index b*4 + c*2 + e
    W = np.zeros((16, 2))
    for r in range(2):
        for b, c, e in itertools.product((0, 1), repeat=3):
            W[(b * 2 + 0) * 4 + (c * 2 + e), r] = V[b * 4 + c * 2 + e, r]
    X = np.zeros((32, 32), complex)
    for r in range(2):
        for s in range(2):
            X[r * 16:(r + 1) * 16, s * 16:(s + 1) * 16] = np.outer(W[:, r], W[:, s])
    return X


def exact_decoders(X, m, D):
    sys.path.insert(0, str(HERE))
    from seesaw_lane import exact_decoder
    F, per = 0.0, {}
    for z, Rz in states(X, m, D).items():
        d = D ** sum(z)
        Dz = exact_decoder(Rz, d)
        per[z] = float(np.real(np.trace(Rz @ Dz))) / 4
        F += per[z] / 2 ** m
    return F, per


def seesaw(m, D, restarts, steps, seed, log=print, budget_s=600.0):
    import twogroup as tg
    rng = np.random.default_rng(seed)
    M = D ** m
    best, hists = (0.0, None), []
    t0 = time.time()
    for r in range(restarts):
        X = random_encoder(m, D, rng, rank=int(rng.integers(1, 3)))
        decs, F = decode(X, m, D, iters=200)
        z0, hist = None, [F]
        for s in range(steps):
            X, z0, _ = tg.encoder_exact(gradient(decs, m, D), M, z0=z0)
            decs, F = decode(X, m, D, warm=decs, iters=60)
            hist.append(F)
            if s > 20 and hist[-1] - hist[-11] < 1e-10:
                break
            if time.time() - t0 > budget_s:
                break
        decs, F = decode(X, m, D, warm=decs, iters=600)
        hists.append({"restart": r, "steps": len(hist) - 1, "F_power": F, "trajectory_every10": hist[::10]})
        log(f"   m={m} D={D} restart {r}: F = {F:.10f} after {len(hist) - 1} steps")
        if F > best[0]:
            best = (F, X)
        if time.time() - t0 > budget_s:
            break
    return best, hists


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv) -> int:
    quick = "--quick" in argv
    t_start = time.time()
    rec = {"lane": "R178(b)", "question": "sup_m F_c(A^m, 2) for the 50% erasure channel A",
           "inputs": {str(_rel(p)): sha(p) for p in
                      (HERE / "r178_erasure_sup.py", HERE / "general_codes" / "twogroup.py",
                       HERE / "seesaw_lane.py", ROOT / "src" / "oneq" / "superactivation.py")}}
    # (i) the broadcast lemma, exact
    cert = broadcast_certificate()
    rec["broadcast_lemma"] = cert
    print("broadcast lemma:", cert)
    # numerical primal of the same SDP (cvxpy) -- must reach 3/2
    import cvxpy as cp
    W = np.array([[float(x) for x in row] for row in broadcast_objective()])
    J = cp.Variable((8, 8), hermitian=True)
    pr = cp.Problem(cp.Maximize(cp.real(cp.trace(W @ J))),
                    [J >> 0, cp.partial_trace(J, dims=[2, 4], axis=1) == np.eye(2)])
    pr.solve(solver=cp.CLARABEL)
    rec["broadcast_lemma"]["primal_cvxpy"] = float(pr.value)
    print(f"   cvxpy primal {pr.value:.12f}")
    # (ii) the bound per m
    rec["pairing_bound"] = {m: {"exact": str(pairing_bound(m)), "float": float(pairing_bound(m))} for m in range(1, 9)}
    # (iii) attainment at m = 2, D >= 4
    fb = cloner_exact_fb()
    Xc = cloner_encoder_m2_D4()
    Fc, perc = exact_decoders(Xc, 2, 4)
    rec["m2_D4_cloner"] = {"F_B_exact": str(fb), "F_exact_claim": str(Fraction(1, 4) * (1 + Fraction(1, 4) + 2 * fb)),
                           "F_numeric_exact_sdp_decoders": Fc,
                           "per_pattern": {"".join(map(str, z)): v for z, v in perc.items()},
                           "F_B_float": cloner_split_fidelities()["F_B"]}
    print(f"m=2 D=4 cloner code: F(B) = {fb}; F = {Fc:.12f} (bound {float(pairing_bound(2)):.12f})")
    # falsifier / negative control: a code that must NOT reach the bound (repetition |0>->|00>, |1>->|11>)
    Xr = np.zeros((8, 8), complex)
    v = np.zeros((4, 2))
    v[0, 0] = v[3, 1] = 1
    for r in range(2):
        for s in range(2):
            Xr[r * 4:(r + 1) * 4, s * 4:(s + 1) * 4] = np.outer(v[:, r], v[:, s])
    Fr, perr = exact_decoders(Xr, 2, 2)
    rec["negative_control_repetition_m2_D2"] = {"F": Fr, "per_pattern": {"".join(map(str, z)): x for z, x in perr.items()},
                                                "below_bound": Fr < float(pairing_bound(2)) - 1e-6}
    print(f"negative control (repetition code, m=2 D=2): F = {Fr:.10f}")
    # see-saw lower bounds
    runs = [(2, 2, 6, 300), (3, 2, 6, 300), (4, 2, 4, 200), (3, 4, 3, 150)]
    if quick:
        runs = [(2, 2, 2, 60), (3, 2, 2, 60)]
    rec["seesaw"] = {}
    for m, D, R_, S_ in runs:
        t0 = time.time()
        (Fb, Xb), hists = seesaw(m, D, R_, S_, seed=178 + 10 * m + D, budget_s=420.0)
        Fx, perx = exact_decoders(Xb, m, D)
        bnd = float(pairing_bound(m))
        rec["seesaw"][f"m{m}_D{D}"] = {"F_power": Fb, "F_exact_decoders": Fx, "bound": bnd, "gap": bnd - Fx,
                                       "per_pattern": {"".join(map(str, z)): x for z, x in perx.items()},
                                       "restarts": hists, "seconds": time.time() - t0}
        print(f"SEESAW m={m} D={D}: F = {Fx:.10f}  bound {bnd:.10f}  gap {bnd - Fx:.3e}", flush=True)
    rec["nonreproducible"] = {"wall_clock_s": time.time() - t_start,
                              "stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
