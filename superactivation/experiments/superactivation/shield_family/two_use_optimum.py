"""The optimal two-use code for N~_p^{(x)2}: the global search, a closed-form one-CNOT family, and explicit circuits.

N~_p = 1/2 id (x) |0><0| + 1/2 (X^p o Delta) (x) |1><1|, p = 1/(sqrt d + 1) for the shield family (R142). For an
isometric encoder V: C^2 -> C^2 (x) C^2 the four flag patterns decouple (weight 1/4 each):
  (0,0)  nothing lost: V^dag recovers exactly, F = 1;
  (1,0)  qubit 1 is a classical Z bit through BSC(p): Bob reads it and corrects qubit 2 with a 2 x 2 map;
  (0,1)  the mirror image;
  (1,1)  both are classical: measure-and-prepare, F = (1/4) sum_bc lambda_max(M_bc) EXACTLY,
         M_bc = sum T(b|a) T(c|e) V^dag |ae><ae| V.
For (1,0)/(0,1) this module uses the best UNITARY correction per observed bit (a 4 x 4 real quaternion eigenproblem;
a lower bound on the CPTP optimum) and brackets the CPTP optimum separately: decoder_bracket() runs the repo's power
iteration (onset_vs_p.decode) for a primal value and builds a dual witness Y (1 (x) Y >= branch state) for an upper
bound. At every optimum reported below the two agree to 1e-15, so the unitary corrections ARE optimal there.

FAMILY A (one CNOT; DERIVED here, checked numerically): |0_L> = |+>|b(+h)>, |1_L> = |->|b(-h)>,
b(phi) = (|0> + e^{i phi}|1>)/sqrt 2, h = Delta/2. With s = sin(Delta/2) and r = sqrt(1 - 4p(1-p)s^2):
  F_10 = (1-p)(1+s)/2,   F_01 = (1 + r)/2,   F_11 = (1 + (1-2p) r)/4    (unitary corrections for F_10, F_01;
  decoder_bracket shows they are CPTP-optimal at Delta* for d = 16, 20, 32, 64),
  F(s) = (1/4)[7/4 + (1-p)(1+s)/2 + (3-2p) r/4], concave in s, maximized at s* = 1/sqrt(4p[(1-p) + p(3-2p)^2]):
  F_A* = [9 - 2p + sqrt(sqrt d + (3-2p)^2)] / 16     when s* <= 1, i.e. d <= 21 + 12 sqrt 3 = 41.78
         (the condition is (t^2 - 6t - 3)(t^2 + 6t + 1) <= 0 with t = sqrt d);
  F_A* = (7 - 6p + 2p^2)/8 (the closed form, Delta = pi) for d >= 21 + 12 sqrt 3.
  F_A* > 3/4  <=>  24p^2 + p - 1 < 0  <=>  p < (sqrt 97 - 1)/48 = 0.18435  <=>  d > (49 - sqrt 97)/2 = 19.58.
  Every 1-CNOT code is the span of two orthogonal product states |a0 b0>, |a1 b1> with a0 _|_ a1 (or b0 _|_ b1),
  and the best of them (best_one_cnot) equals F_A* at every d tested: family A IS the 1-CNOT optimum (MEASURED).

  python two_use_optimum.py search D STARTS SEED     global search over isometries (+ the R142 codes as seeds)
  python two_use_optimum.py family-a D               F_A*, Delta*, the circuit, and its simulated protocol
  python two_use_optimum.py bracket CODE.npy D       CPTP decoder bracket and the repo's CPTP-encoder see-saw
  python two_use_optimum.py structure CODE.npy D     product states, 1-CNOT test, a 2-CNOT circuit fit
Certify any code with two_use_certificate.py (exact, PSD by construction).
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "general_codes"))
import onset_vs_p as ovp  # noqa: E402

EV = HERE.parents[2] / "evidence" / "superactivation" / "shield_family"
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1.0, -1.0]).astype(complex)
H = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)
CNOT = np.eye(4, dtype=complex)[[0, 1, 3, 2]]
PAULI = (I2, X, Y, Z)
D_STAR = 21 + 12 * math.sqrt(3)            # family A's optimum is the closed form iff d >= D_STAR
D_CROSS = (49 - math.sqrt(97)) / 2         # F_A* > 3/4 iff d > D_CROSS


def p_of(d: float) -> float:
    return 1 / (math.sqrt(d) + 1)


def closed_form(p: float) -> float:
    return (7 - 6 * p + 2 * p * p) / 8


def phase(phi: float) -> np.ndarray:
    return np.diag([1, np.exp(1j * phi)])


def bsc(obs: int, true: int, p: float) -> float:
    return 1 - p if obs == true else p


def _check_iso(V: np.ndarray) -> None:
    if V.shape != (4, 2) or not np.allclose(V.conj().T @ V, I2, atol=1e-10):
        raise ValueError("a two-use code is a 4 x 2 isometry (columns |0_L>, |1_L>)")


# ----------------------------------------------------------------------------------------------- branch optima
def _amps(M: np.ndarray) -> np.ndarray:
    """Tr(U M) = q . amps for U = q0 1 + i (q1 X + q2 Y + q3 Z), q a real unit 4-vector"""
    return np.array([np.trace(M)] + [1j * np.trace(P @ M) for P in PAULI[1:]])


def _su2(q: np.ndarray) -> np.ndarray:
    return q[0] * I2 + 1j * (q[1] * X + q[2] * Y + q[3] * Z)


def flagged_one(V: np.ndarray, p: float, flagged: int):
    """best unitary correction of the kept qubit per observed bit of the flagged one: (F, [U_0, U_1])"""
    S = V.reshape(2, 2, 2)                                   # (qubit 1, qubit 2, logical)
    total, Us = 0.0, []
    for b in (0, 1):
        K = np.zeros((4, 4))
        for a in (0, 1):
            v = _amps(S[a] if flagged == 0 else S[:, a])
            K += bsc(b, a, p) * np.real(np.outer(v, v.conj()))
        w, U = np.linalg.eigh((K + K.T) / 2)
        total += w[-1]
        Us.append(_su2(U[:, -1]))
    return total / 4, Us


def flagged_both(V: np.ndarray, p: float):
    """exact measure-and-prepare optimum: (F, prepared logical state per observed (b, c))"""
    total, prep = 0.0, {}
    for b in (0, 1):
        for c in (0, 1):
            M = sum(bsc(b, i >> 1, p) * bsc(c, i & 1, p) * np.outer(V[i].conj(), V[i]) for i in range(4))
            w, U = np.linalg.eigh((M + M.conj().T) / 2)
            total += w[-1]
            # given the outcome the reference holds ~ the ROW V[ae, :]; M's top eigenvector ~ its conjugate, which
            # is exactly the output that maximizes <Phi| ref (x) out |Phi>
            prep[(b, c)] = U[:, -1]
    return total / 4, prep


def branch_values(V: np.ndarray, p: float) -> dict:
    return {"00": 1.0, "10": flagged_one(V, p, 0)[0], "01": flagged_one(V, p, 1)[0], "11": flagged_both(V, p)[0]}


def fidelity(V: np.ndarray, p: float) -> float:
    b = branch_values(V, p)
    return (b["00"] + b["10"] + b["01"] + b["11"]) / 4


# ------------------------------------------------------------------------------------- the CPTP decoder bracket
def decoder_bracket(V: np.ndarray, p: float, iters: int = 20000) -> dict:
    """For this encoder: primal = the repo's power-iteration CPTP decoders, dual = a feasible Y per pattern.
    max {Tr(R D)/4 : D >= 0, Tr_R D = 1} <= Tr(Y')/4 for any 1_R (x) Y' >= R (weak duality; Y from the primal's
    KKT condition Y = Tr_R(R D), shifted by lambda_max(R - 1 (x) Y) to feasibility)."""
    _check_iso(V)
    w = np.concatenate([V[:, 0], V[:, 1]])
    Xc = np.outer(w, w.conj())
    decs, F = ovp.decode(Xc, 2, p, iters=iters)
    lo, hi = {}, {}
    for z, R in ovp.branch_states(Xc, 2, p).items():
        D = decs[z][0]
        lo[z] = float(np.real(np.trace(R @ D))) / 4
        Yz = np.trace((R @ D).reshape(2, 4, 2, 4), axis1=0, axis2=2)
        Yz = (Yz + Yz.conj().T) / 2
        t = np.linalg.eigvalsh(R - np.kron(I2, Yz))[-1]
        hi[z] = float(np.real(np.trace(Yz)) + 4 * t) / 4
    return {"F_primal": sum(lo.values()) / 4, "F_dual": sum(hi.values()) / 4, "primal": lo, "dual": hi}


# ------------------------------------------------------------------------------------------- family A, closed form
def family_a_code(delta: float) -> np.ndarray:
    h = delta / 2
    plus, minus = H[:, 0], H[:, 1]
    b = lambda phi: np.array([1, np.exp(1j * phi)]) / math.sqrt(2)  # noqa: E731
    return np.stack([np.kron(plus, b(h)), np.kron(minus, b(-h))], axis=1)


def family_a_value(p: float, delta: float) -> float:
    s = math.sin(delta / 2)
    r = math.sqrt(1 - 4 * p * (1 - p) * s * s)
    return (7 / 4 + (1 - p) * (1 + s) / 2 + (3 - 2 * p) * r / 4) / 4


def family_a_optimum(p: float):
    """(F_A*, Delta*) in closed form"""
    t = 1 / p - 1
    if t * t - 6 * t - 3 >= 0:
        return closed_form(p), math.pi
    s = 1 / math.sqrt(4 * p * ((1 - p) + p * (3 - 2 * p) ** 2))
    return (9 - 2 * p + math.sqrt(t + (3 - 2 * p) ** 2)) / 16, 2 * math.asin(s)


def family_a_circuit(p: float) -> dict:
    """Every gate of the optimal 1-CNOT protocol, angles in closed form (qubit 1 = the input/first use).
    Encoder on |psi>|0>:  (H (x) 1) . CNOT_{1->2} . (P(-h) (x) P(h) H),  P(phi) = diag(1, e^{i phi}).
    (0,0): the encoder's inverse (1 CNOT), or read the logical Paulis directly as single-qubit products:
           Z_L = X_1,  cos h X_L + sin h Y_L = Z_1 X_2,  sin h X_L - cos h Y_L = Y_1 X_2.
    (0,1): read c on qubit 2, apply D(-(c h + (-1)^c alpha/2)) H to qubit 1, D(x) = diag(e^{ix}, e^{-ix}),
           alpha = arg(1 - p + p e^{i Delta})  [the compromise between the two possible phase errors].
    (1,0): read b on qubit 1, apply Z^b U0 to qubit 2, U0 = [cos(Delta/4)(1 - iX) - i sin(Delta/4)(Y + Z)]/sqrt 2.
    (1,1): read (b, c), prepare (|0> + e^{i mu}|1>)/sqrt 2, mu = b pi + arg(T(c|0) + T(c|1) e^{i Delta})."""
    F, delta = family_a_optimum(p)
    h = delta / 2
    enc = np.kron(H, I2) @ CNOT @ np.kron(phase(-h), phase(h) @ H)
    alpha = np.angle(1 - p + p * np.exp(1j * delta))
    Dg = lambda x: np.diag([np.exp(1j * x), np.exp(-1j * x)])  # noqa: E731
    c4, s4 = math.cos(delta / 4), math.sin(delta / 4)
    U0 = (c4 * (I2 - 1j * X) - 1j * s4 * (Y + Z)) / math.sqrt(2)
    dec01 = [Dg(-(c * h + (-1) ** c * alpha / 2)) @ H for c in (0, 1)]
    dec10 = [U0, Z @ U0]
    prep = {}
    for b in (0, 1):
        for c in (0, 1):
            mu = b * math.pi + np.angle(bsc(c, 0, p) + bsc(c, 1, p) * np.exp(1j * delta))
            prep[(b, c)] = np.array([1, np.exp(1j * mu)]) / math.sqrt(2)
    return {"p": p, "F": F, "delta": delta, "encoder_unitary": enc, "encoder": enc[:, [0, 2]],
            "dec00": enc.conj().T, "dec01": dec01, "dec10": dec10, "dec11": prep, "cnots": 1}


def generic_protocol(V: np.ndarray, p: float) -> dict:
    """the optimal unitary/measure-prepare decoders of any code, with a unitary completion for (0,0)"""
    _check_iso(V)
    Q, _ = np.linalg.qr(np.concatenate([V, np.eye(4)], axis=1))
    comp = Q[:, 2:4]
    comp = comp - V @ (V.conj().T @ comp)
    comp, _ = np.linalg.qr(comp)
    enc = np.zeros((4, 4), complex)
    enc[:, [0, 2]] = V
    enc[:, [1, 3]] = comp
    return {"p": p, "encoder": V, "encoder_unitary": enc, "dec00": enc.conj().T,
            "dec10": flagged_one(V, p, 0)[1], "dec01": flagged_one(V, p, 1)[1], "dec11": flagged_both(V, p)[1]}


def simulate_protocol(proto: dict, p: float) -> dict:
    """Run the protocol gate by gate on the reference-encoded state: the explicit Kraus operators of N~_p on each
    flagged use, a Z measurement of each flagged qubit, the decoders as given. Returns the fidelity per pattern."""
    V = proto["encoder"]
    psi = np.concatenate([V[:, 0], V[:, 1]]) / math.sqrt(2)          # R (x) q1 (x) q2
    rho = np.outer(psi, psi.conj()).reshape([2] * 6)
    kraus = [math.sqrt(1 - p) * np.diag([1.0, 0]), math.sqrt(1 - p) * np.diag([0, 1.0]),
             math.sqrt(p) * np.array([[0, 0], [1.0, 0]]), math.sqrt(p) * np.array([[0, 1.0], [0, 0]])]
    phi = np.array([1, 0, 0, 1]) / math.sqrt(2)

    def on(r, U, q):                                                  # U on qubit q (1 or 2) of R q1 q2
        r = np.moveaxis(np.tensordot(U, r, axes=([1], [q])), 0, q)
        return np.moveaxis(np.tensordot(r, U.conj(), axes=([3 + q], [1])), -1, 3 + q)

    def channel(r, q):
        return sum(on(r, k, q) for k in kraus)

    def fid(r2):                                                      # r2 on R (x) out
        return float(np.real(phi.conj() @ r2.reshape(4, 4) @ phi))

    def proj(r, q, v):
        P = np.zeros((2, 2))
        P[v, v] = 1
        return on(r, P, q)

    out = {}
    r = rho.reshape(8, 8)
    U = proto["dec00"]
    r00 = (np.kron(I2, U) @ r @ np.kron(I2, U).conj().T).reshape([2] * 6)
    out["00"] = fid(np.einsum("abcdec->abde", r00))                   # trace qubit 2 (returned to |0>)
    r01, r10, r11 = channel(rho, 2), channel(rho, 1), channel(channel(rho, 1), 2)
    s = 0.0
    for c in (0, 1):
        t = on(proj(r01, 2, c), proto["dec01"][c], 1)
        s += fid(np.einsum("abcdec->abde", t))
    out["01"] = s
    s = 0.0
    for b in (0, 1):
        t = on(proj(r10, 1, b), proto["dec10"][b], 2)
        s += fid(np.einsum("abcdbf->acdf", t))
    out["10"] = s
    s = 0.0
    for b in (0, 1):
        for c in (0, 1):
            t = proj(proj(r11, 1, b), 2, c)
            rR = np.einsum("abcdbc->ad", t)
            xi = proto["dec11"][(b, c)]
            s += fid(np.kron(rR, np.outer(xi, xi.conj())))           # rows (r, out): R (x) prepared output
    out["11"] = s
    out["F"] = (out["00"] + out["01"] + out["10"] + out["11"]) / 4
    return out


# ------------------------------------------------------------------------------------------ gate-count structure
_GENERIC = np.array([[0.6 + 0.48j, -0.64], [0.64, 0.6 - 0.48j]])      # a fixed unitary: no codeword is special


def product_states(V: np.ndarray):
    """The product states in the code space: det(t C0 + C1) = 0 (C_i the codewords as 2 x 2 matrices) in a generic
    logical basis, so neither root sits at t = infinity. Returns 'all' for a product code (det vanishes identically),
    else [(a, b, rank-2 residual), ...] -- two entries generically, equal ones when the quadratic has a double root."""
    W = V @ _GENERIC
    C0, C1 = W[:, 0].reshape(2, 2), W[:, 1].reshape(2, 2)
    c2, c0 = np.linalg.det(C0), np.linalg.det(C1)
    c1 = np.linalg.det(C0 + C1) - c2 - c0
    if max(abs(c2), abs(c1), abs(c0)) < 1e-12:
        return "all"
    out = []
    for t in np.roots([c2, c1, c0]):
        U, sv, Wh = np.linalg.svd(t * C0 + C1)
        out.append((U[:, 0], Wh[0].conj(), sv[1] / sv[0]))
    return out


def min_cnots(V: np.ndarray, tol: float = 1e-6) -> int:
    """0: a product code; 1: the code space is spanned by two DISTINCT product states with orthogonal factors on one
    qubit -- exactly the codes (k1 (x) k2) CNOT (C^2 (x) |phi>), up to which wire carries the input; else 2, which
    suffices for every 1 -> 2 qubit isometry (a 2-qubit unitary is two CNOTs up to a diagonal gate, and a diagonal
    acts on |psi>|0> as a phase on the input)."""
    _check_iso(V)
    ps = product_states(V)
    if ps == "all":
        return 0
    if len(ps) == 2:
        (a0, b0, _), (a1, b1, _) = ps
        distinct = abs(np.vdot(np.kron(a0, b0), np.kron(a1, b1))) < 1 - tol
        if distinct and min(abs(np.vdot(a0, a1)), abs(np.vdot(b0, b1))) < tol:
            return 1
    return 2


def _u3(t):
    a, b, c = t
    return np.array([[math.cos(a / 2), -np.exp(1j * c) * math.sin(a / 2)],
                     [np.exp(1j * b) * math.sin(a / 2), np.exp(1j * (b + c)) * math.cos(a / 2)]])


def fit_cnot_circuit(V: np.ndarray, n_cnot: int, restarts: int = 20, seed: int = 0):
    """single-qubit layers around n_cnot CNOTs on |psi>|0>; residual 1 - |det(V^dag W)| (0 iff same code space)"""
    rng = np.random.default_rng(seed)

    def W(x):
        U = np.kron(_u3(x[0:3]), _u3(x[3:6]))
        for k in range(n_cnot):
            U = np.kron(_u3(x[6 + 6 * k:9 + 6 * k]), _u3(x[9 + 6 * k:12 + 6 * k])) @ CNOT @ U
        return U[:, [0, 2]]

    f = lambda x: 1 - abs(np.linalg.det(V.conj().T @ W(x)))  # noqa: E731
    best = (2.0, None)
    for _ in range(restarts):
        r = minimize(f, rng.uniform(0, 2 * math.pi, 6 + 6 * n_cnot), method="BFGS", options={"gtol": 1e-14})
        if r.fun < best[0]:
            best = (r.fun, r.x)
    return best[0], (W(best[1]) if best[1] is not None else None)


# ------------------------------------------------------------------------------------------------- global search
def _iso(x):
    U, _, Wh = np.linalg.svd((x[:8] + 1j * x[8:]).reshape(4, 2), full_matrices=False)
    return U @ Wh


def search(p: float, starts: int, seed: int, seeds=()):
    """BFGS then Nelder-Mead on the isometry manifold (polar map) from each seed code and each random start"""
    rng = np.random.default_rng(seed)
    inits = [np.concatenate([V.reshape(-1).real, V.reshape(-1).imag]) for V in seeds]
    inits += [rng.normal(size=16) for _ in range(starts)]
    f = lambda x: -fidelity(_iso(x), p)  # noqa: E731
    best, vals = (-1.0, None), []
    for x0 in inits:
        r = minimize(f, x0, method="BFGS", options={"gtol": 1e-11, "maxiter": 4000})
        r = minimize(f, r.x, method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-15, "maxiter": 20000,
                                                             "maxfev": 40000})
        vals.append(-r.fun)
        if -r.fun > best[0]:
            best = (-r.fun, _iso(r.x))
    return best[0], best[1], np.array(vals)


def best_one_cnot(p: float, starts: int = 30, seed: int = 3):
    """max over every 1-CNOT code: span{|a0 b0>, |a1 b1>}, a0 _|_ a1 (and the mirror image, b0 _|_ b1)"""
    ket = lambda th, ph: np.array([math.cos(th / 2), np.exp(1j * ph) * math.sin(th / 2)])  # noqa: E731

    def code(x, mirror):
        a0 = ket(x[0], x[1])
        a1 = np.array([-np.conj(a0[1]), np.conj(a0[0])])
        V = np.stack([np.kron(a0, ket(x[2], x[3])), np.kron(a1, ket(x[4], x[5]))], axis=1)
        return V[[0, 2, 1, 3]] if mirror else V
    rng = np.random.default_rng(seed)
    best = (-1.0, None)
    for _ in range(starts):
        x0 = rng.uniform(0, 2 * math.pi, 6)
        for mirror in (False, True):
            f = lambda x: -fidelity(code(x, mirror), p)  # noqa: E731
            r = minimize(f, x0, method="BFGS", options={"gtol": 1e-11})
            r = minimize(f, r.x, method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-15, "maxiter": 20000})
            if -r.fun > best[0]:
                best = (-r.fun, code(r.x, mirror))
    return best


def _r142_seeds():
    return [np.load(EV / f"two_use_code_d{k}.npy") for k in (19, 20, 21) if (EV / f"two_use_code_d{k}.npy").exists()]


# --------------------------------------------------------------------------------------------------------- CLI
def main(argv) -> int:
    np.set_printoptions(precision=8, suppress=True, linewidth=140)
    mode = argv[0]
    if mode == "search":
        d, starts, seed = int(argv[1]), int(argv[2]), int(argv[3])
        p = p_of(d)
        F, V, vals = search(p, starts, seed, _r142_seeds())
        FA, delta = family_a_optimum(p)
        out = pathlib.Path(argv[4]) if len(argv) > 4 else None
        print(f"d={d} p={p:.12f}: best {F:.12f}; closed form {closed_form(p):.12f}; family A {FA:.12f} "
              f"(Delta* {math.degrees(delta):.4f} deg); starts within 1e-8 of best {np.mean(abs(vals - F) < 1e-8):.2f} "
              f"of {len(vals)}; branches {branch_values(V, p)}; min CNOTs {min_cnots(V)}")
        if out:
            np.save(out, V)
        return 0
    if mode == "family-a":
        d = float(argv[1])
        p = p_of(d)
        c = family_a_circuit(p)
        sim = simulate_protocol(c, p)
        print(f"d={d:g} p={p:.12f}: F_A* = {c['F']:.15f} (closed form {closed_form(p):.15f}), Delta* = "
              f"{c['delta']:.12f} rad = {math.degrees(c['delta']):.6f} deg; gate-level simulation {sim}")
        print("encoder isometry (columns |0_L>, |1_L>):\n", c["encoder"])
        return 0
    if mode == "bracket":
        V, d = np.load(argv[1]), float(argv[2])
        br = decoder_bracket(V, p_of(d))
        print(f"unitary-decoder value {fidelity(V, p_of(d)):.15f}; CPTP primal {br['F_primal']:.15f} dual "
              f"{br['F_dual']:.15f}; per pattern {br['primal']} / {br['dual']}")
        return 0
    if mode == "structure":
        V, d = np.load(argv[1]), float(argv[2])
        for a, b, ratio in product_states(V):
            print("product state: qubit 1", a, "qubit 2", b, "(rank-2 residual", ratio, ")")
        print("min CNOTs:", min_cnots(V))
        for n in (1, 2):
            print(f"{n}-CNOT circuit fit residual: {fit_cnot_circuit(V, n)[0]:.3e}")
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
