"""R186: what the CHANNEL-FAITHFUL two-use superactivation protocol needs to exceed F_e = 3/4 -- the hardware, the
noise, the design -- and how far ibm_kingston is from it, component by component.

THE OBJECT. hw_faithful.circuit (R147): the code {|++>, |-->}, P_d's real randomized Clifford(+T) gates on BOTH uses,
the 50% erasure, Bob's lean decoder on channel outputs only. R170's adversary pass: only these circuits can carry a
two-use claim (hw_two_use, the LEAN design, runs P_d on at most one use). The erased register is never touched
again (erase="none"): the same outcome law as a physical reset (hw_faithful.reset_check), without the reset.

THE REPRESENTATION (why this lane can run m = 5..7 at all). The faithful circuit has n = 4 + 4m qubits (24 at m = 5,
28 at m = 6); oneq.hw.engine simulates a 2^n statevector and stops at 24, and samples every error a non-Clifford gate
blocks. This lane evaluates the SAME model -- the engine's own compiled op list (noise.compile_ops: every gate, every
Pauli error with its probability, the asymmetric readout) -- EXACTLY, in the Heisenberg picture:
  - the score's Walsh modes become product observables prod_j (beta_j + gamma_j Z_j) on the measured qubits, with the
    readout folded in exactly (beta = p10 - p01, gamma = 1 - p01 - p10);
  - each op maps a Pauli string by its Pauli transfer matrix (one image for a Clifford, a real combination for an rz of
    any angle), each Pauli error multiplies a string by its fidelity f(P) = 1 - 2 sum_{Q anticommuting} p_Q;
  - a qubit whose earliest op has been passed is CLOSED against |0>: strings with X or Y there vanish, Z and I give 1,
    and strings that now coincide are merged.
Nothing is sampled: the value is the model's exact expectation, every blocked error included. The contraction order
is a depth-first walk down the HUB's chain (key0, the qubit every shield talks to): before a CZ(key0, s_t) it finishes
pair t's later history, and right after it drains pair t's earlier history and closes the pair -- so at most one pair
is ever open beside the hub and the work is linear in m (a time-ordered sweep would hold up to 9^m strings).
Every coefficient carries a VECTOR over noise configurations (Config: per-class multipliers), so one sweep returns
the circuit's exact value at every scenario a budget, a lambda scan or a requirement surface needs -- paired by
construction.

ESTIMATOR over the design (hw_predict's control variate, faithful strata): F_avg = F_avg_ideal - sum_b W_b
mean_{u in b} [ideal_u - noisy_u], the ideal design value in closed form (7 - 6p + 2p^2)/8, every unit's ideal and
noisy values exact (the ideal is the all-zero configuration of the same sweep). The only randomness left is which
units are drawn; its standard error is reported, and every configuration is scored on the same units.

CALIBRATION (R183, job dbap0i1gv2cc73e76mb0, 2026-10-19, MEASURED): frame-tracking R147's feed-forward Pauli removed
the key's 2.57 us conditional wait and raised F_e by +0.0123 (paired sd 0.0042) where the model projected +0.030 to
+0.047: the model over-charges that wait's dephasing 2.4-3.8x. The static faithful design has no feed-forward wait, so
the lever itself does not apply; the 'cal:' configurations carry the over-charge as a PROJECTED variant (white T2 and
sigma rates x 1/3.8, 1/3.1, 1/2.4 everywhere), the transfer from one wait on one qubit being a HYPOTHESIS.

  python r186_faithful_requirements.py selfcheck   exactness vs the density-matrix oracle (144 cases), two planted
                                                   defects that must be caught, the vectorized idle, the m = 1 design
                                                   enumerated against the closed form, the full faithful m = 2 circuit
                                                   vs the engine's trajectories; exit 1 on any failure
  python r186_faithful_requirements.py run --device D --m 5,6 --decoder cz1,cz2,mix --units U --configs C
        D: kingston_hh | kingston_hh_accum | kingston_hh_active | kingston_hh_zz | kingston_a2a | surface |
           spec:<key of device_specs.json>;  C: standard | basic | surface
        -> evidence/superactivation/r186_faithful_requirements/runs/<D>_m<M>_<dec>_<C>.{npz,json}
  python r186_faithful_requirements.py report      -> evidence/superactivation/r186_faithful_requirements/
                                                      r186_summary.json (every number re-derived from the runs)
"""
from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass, field, replace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

OUTD = ROOT / "evidence" / "superactivation" / "r186_faithful_requirements"
BAR = 0.75

# ================================================================================================ Pauli algebra (pure)
# one qubit: code = 2 x + z -> 0 I, 1 Z, 2 X, 3 Y (the engine's _CODE1); k qubits: packed = sum_i code_i << 2i, with
# the op's qubits[0] in the low bits -- the same qubit order as a qiskit matrix on (qubits[0], qubits[1])
LETTER = {"I": 0, "Z": 1, "X": 2, "Y": 3}
_PM = {0: np.eye(2, dtype=complex), 1: np.diag([1.0, -1.0]).astype(complex),
       2: np.array([[0, 1], [1, 0]], dtype=complex), 3: np.array([[0, -1j], [1j, 0]], dtype=complex)}


def unpack(c: int, k: int) -> tuple:
    return tuple((c >> (2 * i)) & 3 for i in range(k))


def pauli_matrix(c: int, k: int) -> np.ndarray:
    """the packed k-qubit Pauli as a qiskit-ordered matrix (qubits[0] least significant)"""
    M = np.eye(1, dtype=complex)
    for ci in unpack(c, k):
        M = np.kron(_PM[ci], M)
    return M


def anticommutes(c1: int, c2: int, k: int) -> int:
    """1 when the packed Paulis anticommute (odd symplectic product), else 0"""
    s = 0
    for i in range(k):
        a, b = (c1 >> (2 * i)) & 3, (c2 >> (2 * i)) & 3
        s ^= ((a >> 1) & (b & 1)) ^ ((a & 1) & (b >> 1))
    return s


def heisenberg_ptm(U: np.ndarray) -> np.ndarray:
    """R[c_in, c_out] = Tr(P_out U^dag P_in U) / 2^k: the Heisenberg image of each Pauli as a real combination of
    Paulis. A Clifford has one entry +-1 per row; an rz of a generic angle splits X and Y."""
    dim = U.shape[0]
    k = int(round(math.log2(dim)))
    mats = [pauli_matrix(c, k) for c in range(4 ** k)]
    R = np.zeros((4 ** k, 4 ** k))
    Ud = U.conj().T
    for ci in range(4 ** k):
        H = Ud @ mats[ci] @ U
        for co in range(4 ** k):
            R[ci, co] = float(np.real(np.trace(mats[co] @ H))) / dim
    R[np.abs(R) < 1e-13] = 0.0
    return R


ANTI = {k: np.array([[anticommutes(c, q, k) for q in range(4 ** k)] for c in range(4 ** k)], dtype=float)
        for k in (1, 2)}


def pauli_fidelity(probs: np.ndarray, k: int) -> np.ndarray:
    """f[c] = 1 - 2 sum_Q p_Q [Q anticommutes with c] for a Pauli channel given as probs[Q] (packed codes; probs[0], the
    identity, is ignored); probs may carry a trailing configuration axis"""
    p = np.array(probs, dtype=float, copy=True)
    p[0] = 0.0
    return 1.0 - 2.0 * np.tensordot(ANTI[k], p, axes=([1], [0]))


def engine_probs(op) -> np.ndarray:
    """an engine Op's error as probabilities over packed codes (index 0 unused)"""
    k = len(op.qubits)
    out = np.zeros(4 ** k)
    if k == 1:
        for g, pp in zip("XYZ", op.pauli_probs):
            out[LETTER[g]] = pp
        return out
    from oneq.hw.noise import PAULIS2
    probs = op.pauli_probs if op.pauli_probs else (op.p_error / 15,) * 15
    for (a, b), pp in zip(PAULIS2, probs):
        out[LETTER[a] + 4 * LETTER[b]] = pp
    return out


def readout_affine(p01: float, p10: float) -> tuple:
    """(beta, gamma) with E[(-1)^observed | true b] = beta + gamma (-1)^b"""
    return p10 - p01, 1.0 - p01 - p10


# ================================================================================================ configurations
@dataclass(frozen=True)
class Config:
    """one noise scenario, as multipliers on the BASE model the circuit was compiled under.
    Probability multipliers (NoiseParams.scaled's convention, so lambda* is comparable with R144/R178): cz (every CZ
    Pauli), sx (every sx/x pulse and dressing op), idle (every idle and quasi-static Pauli), ro01 / ro10 (p01, p10),
    other (spectator, crosstalk). Physical RATE multipliers (a device with T1/t1, T2/t2, sigma*qs): t1, t2, qs."""
    cz: float = 1.0
    sx: float = 1.0
    idle: float = 1.0
    ro01: float = 1.0
    ro10: float = 1.0
    other: float = 1.0
    t1: float = 1.0
    t2: float = 1.0
    qs: float = 1.0
    label: str = ""

    def scaled(self, lam: float, label: str = "") -> "Config":
        return replace(self, cz=self.cz * lam, sx=self.sx * lam, idle=self.idle * lam, ro01=self.ro01 * lam,
                       ro10=self.ro10 * lam, other=self.other * lam, label=label or f"lambda={lam:g}")


IDEAL = Config(cz=0.0, sx=0.0, idle=0.0, ro01=0.0, ro10=0.0, other=0.0, label="ideal")
FULL = Config(label="full")


# ================================================================================================ the exact evaluator
class EvaluatorError(ValueError):
    """the circuit cannot be evaluated exactly by this representation"""


def _idle_duration(pp, t1: float, t2: float) -> float:
    """the idle's duration t, recovered from the engine's (px, py, pz) at idle_factor 1, through the BETTER-conditioned
    channel. The engine computes a = 1 - exp(-t/T1) and b = 1 - exp(-t/T2) by subtraction, so each carries a relative
    error ~ 1e-16 / (t/T); a trapped-ion device (T1 = 1e6 s, t = 1 ms: t/T1 = 1e-9) has px good to ~1e-7 only, and
    recovering t from it fails any tight check (found on Helios, 2026-10-19). pz = b/2 - a/4 with a = 4 px gives
    b = 2 (pz + px) exactly, and t/T2 >> t/T1 there. The check tolerance is the engine's own cancellation error, never
    looser than 1e-6 relative."""
    px, _py, pz = pp
    T2 = min(t2, 2 * t1)
    a = 4.0 * px
    b = 2.0 * (pz + px)
    if b <= 0 and a <= 0:
        return 0.0
    if b > 0 and (a <= 0 or T2 <= t1):
        t = -T2 * math.log1p(-b)
        x = t / T2
    else:
        t = -t1 * math.log1p(-a)
        x = t / t1
    from oneq.hw.noise import idle_pauli
    back = idle_pauli(t, t1, t2)
    rel = min(1e-6, 1e-9 + 4e-16 / max(x, 1e-300))
    if any(abs(bk - v) > 1e-15 + rel * abs(v) + 4e-16 * (t / t1) for v, bk in zip(pp, back)):
        raise EvaluatorError(f"idle {pp} is not an idle_pauli at idle_factor 1 (got {back})")
    return t


def idle_probs_under(t: float, t1: float, t2: float, cfg: Config) -> tuple:
    """the engine's idle Pauli (noise.idle_pauli: T2 clamped to 2 T1) for a device whose relaxation and dephasing
    RATES are the config's multiples (T1 / cfg.t1, T2 / cfg.t2), times its probability multiplier cfg.idle"""
    px, py, pz = idle_probs_vec(t, t1, t2, np.array([cfg.idle]), np.array([cfg.t1]), np.array([cfg.t2]))
    return float(px[0]), float(py[0]), float(pz[0])


def idle_probs_vec(t: float, t1: float, t2: float, idle: np.ndarray, r1: np.ndarray, r2: np.ndarray):
    """idle_probs_under over arrays of configurations. A zero rate is an infinite time; with both zero the idle is
    free. Equal to noise.idle_pauli(t, t1 / r1, t2 / r2) x idle wherever both rates are positive (tested)."""
    with np.errstate(divide="ignore"):
        g1 = np.where(r1 > 0, r1 / t1, 0.0)                     # 1 / T1'
        g2 = np.where(r2 > 0, r2 / t2, 0.0)                     # 1 / T2'
    g2 = np.maximum(g2, g1 / 2.0)                               # T2' <= 2 T1'
    a = -np.expm1(-t * g1)
    b = -np.expm1(-t * g2)
    px = a / 4.0
    pz = np.maximum(0.0, b / 2.0 - a / 4.0)
    return idle * px, idle * px, idle * pz


@dataclass
class _OpRec:
    qubits: tuple
    k: int
    out: np.ndarray | None = None        # gate: (4^k, w) output codes (padded with 0)
    coef: np.ndarray | None = None       # gate: (4^k, w) coefficients (0 where padded)
    clifford: bool = True
    F: np.ndarray | None = None          # error: (4^k, G) Pauli fidelities per configuration


@dataclass
class Compiled:
    """a circuit compiled once by the engine's model, with its exact per-configuration tables"""
    recs: list
    active: list
    meas: list
    beta: np.ndarray                     # (n_meas, G)
    gamma: np.ndarray
    n_ops: int
    counts: dict = field(default_factory=dict)


_PTM_CACHE: dict = {}


def _gate_tables(U: np.ndarray):
    key = (U.shape, np.round(U, 12).tobytes())
    hit = _PTM_CACHE.get(key)
    if hit is not None:
        return hit
    R = heisenberg_ptm(U)
    n = R.shape[0]
    w = int(max(1, (R != 0).sum(axis=1).max()))
    out = np.zeros((n, w), dtype=np.int64)
    coef = np.zeros((n, w))
    for c in range(n):
        nz = np.flatnonzero(R[c])
        out[c, :len(nz)] = nz
        coef[c, :len(nz)] = R[c, nz]
    clifford = bool(w == 1 and np.allclose(np.abs(coef[:, 0]), 1.0, atol=1e-12))
    if clifford:
        coef[:, 0] = np.sign(coef[:, 0])
    identity = clifford and np.array_equal(out[:, 0], np.arange(n)) and np.all(coef[:, 0] == 1.0)
    hit = (out, coef, clifford, identity)
    _PTM_CACHE[key] = hit
    return hit


def compile_circuit(qc, snap, params, configs: list) -> Compiled:
    """the engine's op list for qc (noise.compile_ops, the model the engine itself samples) turned into exact tables
    for every configuration"""
    from oneq.hw import noise as N
    if params.idle_factor != 1.0 or params.readout_factor != 1.0:
        raise EvaluatorError("compile under the base model (idle_factor = readout_factor = 1); scale with Config")
    ops, active, meas = N.compile_ops(qc, snap, params)
    G = len(configs)
    col = {f: np.array([getattr(c, f) for c in configs], dtype=float)
           for f in ("cz", "sx", "idle", "ro01", "ro10", "other", "t1", "t2", "qs")}
    recs = []
    counts: dict = {}
    for op in ops:
        k = len(op.qubits)
        rec = _OpRec(qubits=tuple(op.qubits), k=k)
        if op.matrix is not None:
            out, coef, cliff, ident = _gate_tables(op.matrix)
            if not ident:
                rec.out, rec.coef, rec.clifford = out, coef, cliff
        if op.p_error > 0:
            base = engine_probs(op)
            q = active[op.qubits[0]]
            if op.name == "cz":
                P = base[:, None] * col["cz"][None, :]
            elif op.name in ("sx", "x", "dressing"):
                P = base[:, None] * col["sx"][None, :]
            elif op.name == "idle":
                t = _idle_duration(op.pauli_probs, snap.t1[q], snap.t2[q])
                P = np.zeros((4, G))
                px, py, pz = idle_probs_vec(t, snap.t1[q], snap.t2[q], col["idle"], col["t1"], col["t2"])
                P[LETTER["X"]], P[LETTER["Y"]], P[LETTER["Z"]] = px, py, pz
                # where a configuration leaves the idles unscaled, use the engine's own numbers (bit for bit)
                same = (col["idle"] == 1.0) & (col["t1"] == 1.0) & (col["t2"] == 1.0)
                if same.any():
                    for g_letter, v in zip("XYZ", op.pauli_probs):
                        P[LETTER[g_letter], same] = v
            elif op.name == "quasistatic":
                pz0 = op.pauli_probs[2]
                P = np.zeros((4, G))
                # pz = (1 - exp(-sigma^2 T^2 / 2)) / 2, so sigma -> qs sigma raises (1 - 2 pz) to the power qs^2
                P[LETTER["Z"]] = col["idle"] * (1.0 - (1.0 - 2.0 * pz0) ** (col["qs"] ** 2)) / 2.0
            elif op.name in ("spectator", "crosstalk"):
                P = base[:, None] * col["other"][None, :]
            else:
                raise EvaluatorError(f"unknown error op {op.name!r}")
            rec.F = pauli_fidelity(P, k)
            counts[op.name] = counts.get(op.name, 0) + 1
        recs.append(rec)
    beta = np.zeros((len(meas), G))
    gamma = np.ones((len(meas), G))
    if params.readout:
        for j, (a, _) in enumerate(meas):
            q = active[a]
            p01 = np.minimum(0.5, col["ro01"] * snap.p01[q])
            p10 = np.minimum(0.5, col["ro10"] * snap.p10[q])
            beta[j], gamma[j] = readout_affine(p01, p10)
    return Compiled(recs, active, meas, beta, gamma, len(ops), counts)


def score_modes(score, n_bits: int) -> list:
    """the score's Walsh expansion on OBSERVED bits: [(subset of clbits, coefficient)], score(o) = sum_s c_s
    prod_{j in s} (-1)^{o_j}; exact (a power-of-two transform of the score vector)"""
    from oneq.hw.engine import score_vector, wht
    sh = wht(score_vector(score, n_bits)) / float(1 << n_bits)
    out = []
    for idx in np.flatnonzero(np.abs(sh) > 1e-15):
        subset = tuple(j for j in range(n_bits) if (int(idx) >> (n_bits - 1 - j)) & 1)
        out.append((subset, float(sh[idx])))
    return out


class _Terms:
    """a sum of Pauli strings over the active qubits (uint64 masks x, z), each with a pending-readout mask and a
    coefficient vector over the configurations"""

    def __init__(self, X, Z, PD, C):
        self.X, self.Z, self.PD, self.C = X, Z, PD, C

    def __len__(self):
        return len(self.X)

    def merge(self):
        if len(self.X) <= 1:
            return
        key = np.stack([self.X, self.Z, self.PD], axis=1)
        uq, inv = np.unique(key, axis=0, return_inverse=True)
        if len(uq) == len(self.X):
            return
        inv = inv.reshape(-1)
        order = np.argsort(inv, kind="stable")
        inv_s = inv[order]
        starts = np.r_[0, np.flatnonzero(np.diff(inv_s)) + 1]
        self.C = np.add.reduceat(self.C[order], starts, axis=0)
        self.X, self.Z, self.PD = uq[:, 0].copy(), uq[:, 1].copy(), uq[:, 2].copy()


def _bit(q: int) -> np.uint64:
    return np.uint64(1) << np.uint64(q)


def _local(T: _Terms, qs: tuple) -> np.ndarray:
    c = np.zeros(len(T), dtype=np.int64)
    for i, q in enumerate(qs):
        sh = np.uint64(q)
        x = ((T.X >> sh) & np.uint64(1)).astype(np.int64)
        z = ((T.Z >> sh) & np.uint64(1)).astype(np.int64)
        c |= (2 * x + z) << (2 * i)
    return c


def _set_local(X, Z, qs: tuple, codes: np.ndarray):
    for i, q in enumerate(qs):
        b = _bit(q)
        ci = (codes >> (2 * i)) & 3
        X = (X & ~b) | (((ci >> 1) & 1).astype(np.uint64) << np.uint64(q))
        Z = (Z & ~b) | ((ci & 1).astype(np.uint64) << np.uint64(q))
    return X, Z


def contraction_order(recs: list, n: int, hub: int | None, measured: set) -> list:
    """a valid backward order of the op list (every op after all LATER ops on its qubits): depth-first down the hub's
    chain, each blocker's own later history first, and after every step a drain of the ops that need nothing new
    (all their qubits already open), so a pair is closed as soon as its history is done"""
    ops_on: list = [[] for _ in range(n)]
    for i, r in enumerate(recs):
        for q in r.qubits:
            ops_on[q].append(i)
    nxt = [len(v) - 1 for v in ops_on]
    done = np.zeros(len(recs), dtype=bool)
    opened = set(q for q in range(n) if not ops_on[q])
    order: list = []

    def head(q):
        return ops_on[q][nxt[q]] if nxt[q] >= 0 else None

    def ready(i):
        return all(head(q) == i for q in recs[i].qubits)

    def take(i):
        done[i] = True
        order.append(i)
        for q in recs[i].qubits:
            nxt[q] -= 1
            opened.add(q)

    def drain():
        while True:
            best = None
            for q in list(opened):
                i = head(q)
                if i is None or done[i] or not ready(i):
                    continue
                if not all(x in opened for x in recs[i].qubits):
                    continue
                closes = any(nxt[x] == 0 for x in recs[i].qubits)
                key = (0 if closes else 1, -i)
                if best is None or key < best[0]:
                    best = (key, i)
            if best is None:
                return
            take(best[1])

    def ensure(i):
        stack = [i]
        while stack:
            top = stack[-1]
            if done[top]:
                stack.pop()
                continue
            blockers = sorted({head(q) for q in recs[top].qubits if head(q) != top}, reverse=True)
            if blockers:
                stack.extend(b for b in blockers if not done[b])
                continue
            take(top)
            stack.pop()
            drain()

    if hub is not None:
        for i in reversed(ops_on[hub]):
            ensure(i)
    for i in reversed(range(len(recs))):
        if not done[i]:
            ensure(i)
    if len(order) != len(recs):
        raise EvaluatorError("internal: the contraction order missed ops")
    return order


def evaluate(comp: Compiled, score, n_bits: int, hub: int | None = None, stats: dict | None = None) -> np.ndarray:
    """the exact expected score at every configuration the circuit was compiled for"""
    n = len(comp.active)
    if n > 63:
        raise EvaluatorError(f"{n} active qubits: the bit-mask representation stops at 63")
    G = comp.beta.shape[1]
    modes = score_modes(score, n_bits)
    clbit_qubit = {cb: a for a, cb in comp.meas}
    X = np.zeros(len(modes), dtype=np.uint64)
    Z = np.zeros(len(modes), dtype=np.uint64)
    PD = np.zeros(len(modes), dtype=np.uint64)
    C = np.zeros((len(modes), G))
    for t, (subset, c) in enumerate(modes):
        m = np.uint64(0)
        for j in subset:
            m |= _bit(clbit_qubit[j])
        PD[t] = m
        C[t, :] = c
    T = _Terms(X, Z, PD, C)
    T.merge()
    meas_row = {a: j for j, (a, _) in enumerate(comp.meas)}
    injected: set = set()
    peak = len(T)

    def inject(q):
        if q in injected:
            return
        injected.add(q)
        if q not in meas_row:
            return
        b = _bit(q)
        sel = (T.PD & b) != 0
        if not sel.any():
            return
        j = meas_row[q]
        keepX, keepZ, keepP, keepC = T.X[~sel], T.Z[~sel], T.PD[~sel], T.C[~sel]
        sx, sz, sp, sc = T.X[sel], T.Z[sel], T.PD[sel] & ~b, T.C[sel]
        T.X = np.concatenate([keepX, sx, sx])
        T.Z = np.concatenate([keepZ, sz, sz | b])
        T.PD = np.concatenate([keepP, sp, sp])
        T.C = np.concatenate([keepC, sc * comp.beta[j][None, :], sc * comp.gamma[j][None, :]])
        T.merge()

    def close(q):
        b = _bit(q)
        keep = (T.X & b) == 0
        T.X, T.Z, T.PD, T.C = T.X[keep], T.Z[keep] & ~b, T.PD[keep], T.C[keep]
        T.merge()

    ops_count = [0] * n
    for r in comp.recs:
        for q in r.qubits:
            ops_count[q] += 1
    remaining = list(ops_count)
    for q in range(n):                                  # measured qubits no op ever touches
        if remaining[q] == 0:
            inject(q)
            close(q)
    order = contraction_order(comp.recs, n, hub, set(meas_row))
    for i in order:
        r = comp.recs[i]
        for q in r.qubits:
            inject(q)
        if r.F is not None and len(T):
            codes = _local(T, r.qubits)
            if codes.any():
                T.C = T.C * r.F[codes]
        if r.out is not None and len(T):
            codes = _local(T, r.qubits)
            if r.clifford:
                newc = r.out[codes, 0]
                sgn = r.coef[codes, 0]
                T.X, T.Z = _set_local(T.X, T.Z, r.qubits, newc)
                if np.any(sgn < 0):
                    T.C = T.C * sgn[:, None]
            else:
                w = r.out.shape[1]
                rep_out = r.out[codes]                 # (T, w)
                rep_coef = r.coef[codes]
                nz = rep_coef != 0
                rows = np.repeat(np.arange(len(T)), nz.sum(axis=1))
                newc = rep_out[nz]
                cf = rep_coef[nz]
                X2, Z2 = _set_local(T.X[rows], T.Z[rows], r.qubits, newc)
                T.X, T.Z, T.PD = X2, Z2, T.PD[rows]
                T.C = T.C[rows] * cf[:, None]
                T.merge()
                del w
        peak = max(peak, len(T))
        for q in r.qubits:
            remaining[q] -= 1
            if remaining[q] == 0:
                close(q)
    if np.any(T.X) or np.any(T.Z) or np.any(T.PD):
        raise EvaluatorError("internal: open strings left after the sweep")
    if stats is not None:
        stats["peak_terms"] = max(stats.get("peak_terms", 0), peak)
        stats["ops"] = comp.n_ops
    return T.C.sum(axis=0) if len(T) else np.zeros(G)


def exact_value(qc, score, snap, params, configs: list, hub: int | None = None, stats: dict | None = None):
    comp = compile_circuit(qc, snap, params, configs)
    return evaluate(comp, score, qc.num_clbits, hub=hub, stats=stats)


def hub_of(qc, logical: int = 0) -> int | None:
    """the ACTIVE index of the physical qubit that logical `logical` was placed on (the initial layout)"""
    lay = getattr(qc, "layout", None)
    phys = lay.initial_index_layout()[logical] if lay is not None else logical
    used = sorted({qc.find_bit(x).index for inst in qc.data for x in inst.qubits
                   if inst.operation.name not in ("delay", "barrier")})
    return used.index(phys) if phys in used else None


# ================================================================================================ devices
# The validated device model v2 (R175; tools/model_regression.py): the vendor calibration REPAIRED by R152's in-circuit
# fit (cz_factor 1 on the repaired rates, the measured CZ channel shapes, the in-circuit readout), R155's white T2 and
# quasi-static sigma (V3c), the vendor's sx error, ALAP; held out on R152 it closes R146 to +0.003 and R147 to +0.001
# with nothing fitted. Its repaired numbers exist only on R152's qubits, so for a NEW layout this lane carries the same
# repair as a PRESET onto every qubit and edge (the treatment hw_diag_run.repaired_snapshot gives an unladdered edge):
#   CZ    vendor r x the median in-circuit factor over R152's 29 laddered edges, with their component-wise median shape
#   T2    R155's white T2 where mapped (142 of 156 qubits) and its quasi-static sigma; elsewhere the medians
#   p01/p10, sx: the vendor's (R152's in-circuit readout sat at 0.86-1.0x vendor on its qubits; the repaired sx is
#   below the gate's decoherence floor and the model regression uses the vendor's)
# Twirling: CLIENT mode (oneq.hw.twirl_mode: the compiled pulses, a Pauli-twirled CZ) -- R172/R175's best mode, whose
# prediction is exact in the engine's model; the server modes are priced beside it.
R152_DIR = ROOT / "evidence" / "superactivation" / "campaign_r152" / "20261017T003152Z"
R155_IDLE = ROOT / "evidence" / "superactivation" / "r155_diag" / "idle_map_db9dsqpgv2cc73e5c8fg.json"


def median_in_circuit_cz() -> dict:
    """R152's laddered edges (both blocks): the median in-circuit/vendor factor and the component-wise median shape"""
    import json
    res = json.loads((R152_DIR / "result_db9c50oc77vs73817rog.json").read_text(encoding="utf-8"))
    lads = list(res["r146"]["fit"]["ladders"].values()) + list(res["block"]["fit"]["ladders"].values())
    fac = sorted(float(x["factor"]) for x in lads)
    shape = np.median(np.array([x["shape"] for x in lads], dtype=float), axis=0)
    shape = shape / shape.sum()
    return dict(factor=float(np.median(fac)), factors=fac, shape=tuple(float(x) for x in shape), n_edges=len(lads))


def kingston_v2_snapshot():
    """(snapshot, provenance): ibm_kingston at R152's calibration (2026-10-16 20:12 EDT) with the v2 repair as a
    device-wide preset (see above)"""
    import json

    from oneq.hw import snapshot as S
    rec = json.loads((R152_DIR / "r146.json").read_text(encoding="utf-8"))
    d = json.loads(json.dumps(rec["snapshot"]))
    cz = median_in_circuit_cz()
    d["cz"] = {k: [v[0] * cz["factor"], v[1]] for k, v in d["cz"].items()}
    d["cz_structure"] = {k: list(cz["shape"]) for k in d["cz"]}
    idle = json.loads(R155_IDLE.read_text(encoding="utf-8"))["per_qubit"]
    ratio = [min(r["white_T2_s"], 2 * d["t1"][int(q)]) / d["t2"][int(q)] for q, r in idle.items()
             if r["white_T2_s"] and math.isfinite(r["white_T2_s"]) and d["t2"][int(q)]]
    sig_med = float(np.median([r["sigma_rad_per_s"] for r in idle.values()]))
    t2_ratio = float(np.median(ratio))
    sigma = {}
    mapped = 0
    for q in range(d["num_qubits"]):
        r = idle.get(str(q))
        t1 = d["t1"][q]
        if t1 is None or d["t2"][q] is None:
            continue
        if r and r["white_T2_s"] and math.isfinite(r["white_T2_s"]):
            d["t2"][q] = min(float(r["white_T2_s"]), 2 * t1)
            sigma[str(q)] = float(r["sigma_rad_per_s"])
            mapped += 1
        else:
            d["t2"][q] = min(d["t2"][q] * t2_ratio, 2 * t1)
            sigma[str(q)] = sig_med
    d["sigma"] = sigma
    d["name"] = "ibm_kingston_r152_v2preset"
    snap = S.DeviceSnapshot.from_json(d)
    prov = dict(base=f"R152 vendor snapshot {rec['snapshot']['calibrated_utc']} (r146.json)",
                cz_in_circuit_factor=cz["factor"], cz_factor_range=[cz["factors"][0], cz["factors"][-1]],
                cz_edges=cz["n_edges"], cz_shape="component-wise median of R152's measured channels",
                t2="R155 white T2 (mapped %d qubits; elsewhere vendor x %.3f)" % (mapped, t2_ratio),
                sigma_median_rad_s=sig_med, readout="vendor", sx="vendor", schedule="alap", twirl="client",
                evidence_class="MEASURED inputs (R152, R155), preset extrapolated to unmeasured qubits (HYPOTHESIS)")
    return snap, prov


def _median(xs) -> float:
    v = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.median(v))


def device_medians(snap, qubits=None) -> dict:
    """the snapshot's median per-operation numbers (over `qubits` and the edges among them when given)"""
    qs = range(snap.num_qubits) if qubits is None else qubits
    qset = set(qs)
    edges = [v for (a, b), v in snap.cz.items() if a in qset and b in qset]
    shape = None
    if snap.cz_structure:
        shape = tuple(np.median(np.array([s for (a, b), s in snap.cz_structure.items() if a in qset and b in qset]),
                                axis=0))
        shape = tuple(float(x) / sum(shape) for x in shape)
    return dict(r2q=_median(e for e, _ in edges), t2q=_median(t for _, t in edges),
                e1q=_median(snap.sx_error[q] for q in qs), t1q=_median(snap.sx_duration[q] for q in qs),
                p01=_median(snap.p01[q] for q in qs), p10=_median(snap.p10[q] for q in qs),
                T1=_median(snap.t1[q] for q in qs), T2=_median(snap.t2[q] for q in qs),
                tmeas=_median(snap.meas_duration[q] for q in qs),
                sigma=_median(snap.sigma[q] for q in qs if q in snap.sigma) if snap.sigma else 0.0,
                cz_shape=shape)


def uniform_snapshot(name: str, n: int, edges, num: dict):
    """a DeviceSnapshot with the same numbers on every qubit and edge. num: r2q (the vendor AVERAGE gate infidelity:
    the engine plays p = 1.25 r), t2q, e1q (average infidelity per sx pulse: p = 1.5 e), t1q, p01, p10, T1, T2,
    tmeas, optional sigma (rad/s) and cz_shape (15 fractions)"""
    from oneq.hw import snapshot as S
    cz = {(min(a, b), max(a, b)): (float(num["r2q"]), float(num["t2q"])) for a, b in edges}
    st = {k: tuple(num["cz_shape"]) for k in cz} if num.get("cz_shape") else {}
    sig = {q: float(num["sigma"]) for q in range(n)} if num.get("sigma") else {}
    T2 = min(float(num["T2"]), 2 * float(num["T1"]))
    return S.DeviceSnapshot(name=name, num_qubits=n, calibrated_utc=None, dt=2.2222222222222221e-10, cz=cz,
                            t1=(float(num["T1"]),) * n, t2=(T2,) * n, p01=(float(num["p01"]),) * n,
                            p10=(float(num["p10"]),) * n, sx_error=(float(num["e1q"]),) * n,
                            sx_duration=(float(num["t1q"]),) * n, meas_duration=(float(num["tmeas"]),) * n,
                            cz_structure=st, sigma=sig)


def generic_backend(n: int, coupling):
    from qiskit.providers.fake_provider import GenericBackendV2
    return GenericBackendV2(num_qubits=n, basis_gates=["cz", "sx", "x", "rz", "id"], coupling_map=coupling, seed=1)


def all_to_all_device(n: int, num: dict, name: str):
    from qiskit.transpiler import CouplingMap
    cm = CouplingMap.from_full(n)
    edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
    return generic_backend(n, cm), uniform_snapshot(name, n, edges, num)


def square_device(rows: int, cols: int, num: dict, name: str):
    from qiskit.transpiler import CouplingMap
    cm = CouplingMap.from_grid(rows, cols, bidirectional=True)
    edges = sorted({(min(a, b), max(a, b)) for a, b in cm.get_edges()})
    return generic_backend(rows * cols, cm), uniform_snapshot(name, rows * cols, edges, num)


def heavy_hex_device(num: dict, name: str):
    """ibm_kingston's heavy-hex graph (FakeKingston) with uniform numbers"""
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    be = FakeKingston()
    edges = sorted({tuple(sorted(e)) for e in be.coupling_map.get_edges()})
    return be, uniform_snapshot(name, be.num_qubits, edges, num)


# ---------------------------------------------------------------------------------------------- spec-driven devices
SPEC_FILE = OUTD / "device_specs.json"


def spec_numbers(spec: dict) -> dict:
    """a published spec row -> the engine's uniform numbers, every conversion stated:
      two_qubit_error, kind "average" (RB / vendor gate error r: the engine plays p = 1.25 r) or "process" (EPLG, a
        process infidelity per layered gate: r = 0.8 x EPLG, so p = EPLG);
      one_qubit_error: the average infidelity of ONE pulse (the ISA plays a generic 1q gate as two sx: an upper bound
        for a native-rotation device);
      p01 = P(read 1 | prep 0), p10 = P(read 0 | prep 1); a single readout number is used for both;
      memory_error_per_layer (trapped ions): pure dephasing of probability p_mem per idle qubit per depth-1 layer of
        duration t_2q -> T2 = -t_2q / ln(1 - 2 p_mem), T1 = 1e6 s (no relaxation); otherwise T1, T2 as given."""
    e2 = float(spec["two_qubit_error"])
    kind = spec.get("two_qubit_error_kind", "average")
    if kind not in ("average", "process"):
        raise ValueError(f"two_qubit_error_kind {kind!r}")
    r2q = e2 if kind == "average" else 0.8 * e2
    ro = spec.get("readout_error")
    p01 = float(spec.get("p01", ro))
    p10 = float(spec.get("p10", ro))
    t2q = float(spec["t_2q_s"])
    if spec.get("memory_error_per_layer") is not None:
        pm = float(spec["memory_error_per_layer"])
        T1 = 1e6
        T2 = -t2q / math.log1p(-2.0 * pm)
    else:
        T1 = float(spec["t1_s"])
        T2 = float(spec["t2_s"])
    return dict(r2q=r2q, t2q=t2q, e1q=float(spec["one_qubit_error"]), t1q=float(spec.get("t_1q_s", 0.0)),
                p01=p01, p10=p10, T1=T1, T2=T2, tmeas=float(spec.get("t_meas_s", t2q)))


def all_to_all_from_spec(spec: dict, n: int):
    """(backend, snapshot) for a spec row on an all-to-all device of n qubits (spec_numbers' conversions)"""
    return all_to_all_device(n, spec_numbers(spec), f"spec_{spec['key']}_a2a{n}")


def device_from_spec(spec: dict, n_logical: int):
    """the spec row on its own topology: all_to_all (n_logical qubits), heavy_hex (Kingston's graph) or square
    (spec['grid'] = [rows, cols])"""
    num = spec_numbers(spec)
    topo = spec["topology"]
    if topo == "all_to_all":
        return all_to_all_device(n_logical, num, f"spec_{spec['key']}_a2a{n_logical}")
    if topo == "heavy_hex":
        return heavy_hex_device(num, f"spec_{spec['key']}_hh")
    if topo == "square":
        r, c = spec["grid"]
        return square_device(int(r), int(c), num, f"spec_{spec['key']}_sq{r}x{c}")
    raise ValueError(f"unknown topology {topo!r}")


# ================================================================================================ the faithful design
def faithful_strata(m: int) -> list:
    """(key, weight in F_avg): key = (branch, sector of use 0, beta, sector of use 1). The ideal copy bit equals use
    0's sector, so beta matched/mismatched splits units whose scores differ by 2; use 1's sector IS the erased use's
    flip. kept_kept: use 1's sector is drawn inside the unit (it is traced: only its timing reaches Bob);
    erased_erased: beta is unused."""
    import hw_two_use as H
    p = H.p_flip(m)
    sec = (("s0", 1 - p), ("s1", p))
    out = []
    for s0, w0 in sec:
        for beta in (0, 1):
            out.append((("kept_kept", s0, beta, None), 0.25 * w0 * 0.5))
            for s1, w1 in sec:
                out.append((("kept_erased", s0, beta, s1), 0.5 * w0 * 0.5 * w1))
    for s0, w0 in sec:
        for s1, w1 in sec:
            out.append((("erased_erased", s0, 0, s1), 0.25 * w0 * w1))
    return out


def _label(m: int, sector: str, rng) -> tuple:
    if sector == "s1":
        return ("s1", tuple(int(x) for x in rng.integers(2, size=m)))
    return ("s0", tuple("IXYZ"[int(x)] for x in rng.integers(4, size=m)))


def sample_faithful(m: int, key: tuple, rng):
    import hw_faithful as HF
    import hw_two_use as H
    branch, s0, beta, s1 = key
    c0 = _label(m, s0, rng)
    c1 = H.hd.sample_choice(m, rng) if s1 is None else _label(m, s1, rng)
    return HF.FUnit(branch, (c0, c1), beta, H.INPUTS[int(rng.integers(6))])


def representatives(m: int, decoder: str) -> list:
    """faithful circuits that span the family (hw_predict's lesson: one random unit misleads a layout chooser)"""
    import hw_faithful as HF
    rng = np.random.default_rng(1)
    s1 = ("s1", (0,) * m)
    s0 = ("s0", ("I",) * m)
    units = [HF.FUnit("kept_kept", (s1, s1), 0, ("Z", 1)), HF.FUnit("kept_erased", (s1, ("s1", (1,) * m)), 1, ("Y", -1)),
             HF.FUnit("kept_kept", (s0, s0), 1, ("X", 1)),
             sample_faithful(m, ("kept_kept", "s0", 0, None), rng), sample_faithful(m, ("kept_erased", "s0", 1, "s1"), rng)]
    return [HF.circuit(m, u, decoder, erase="none") for u in units]


@dataclass
class Device:
    name: str
    backend: object
    snap: object
    params: object
    topology: str
    layout: list | None = None
    coupling_map: object = None
    provenance: dict = field(default_factory=dict)

    def transpile(self, qc, seed: int):
        from qiskit import transpile
        lay = self.layout if self.layout is not None else list(range(qc.num_qubits))
        return transpile(qc, self.backend, initial_layout=lay, optimization_level=3, seed_transpiler=seed,
                         coupling_map=self.coupling_map)


def choose_device_layout(dev: Device, m: int, decoder: str, policy=None) -> dict:
    """oneq.hw.layout over the faithful representatives, on the device's own model; returns the choice's record"""
    from oneq.hw import layout as LY
    reps = representatives(m, decoder)
    ch = LY.choose_layout(reps, dev.backend, dev.snap, dev.params, policy=policy or LY.QubitPolicy())
    dev.layout, dev.coupling_map = ch.initial_layout, ch.coupling_map
    return dict(initial_layout=ch.initial_layout, proxy=ch.best.proxy, mean_cz=ch.best.mean_cz,
                candidates=ch.n_candidates, physical_qubits=list(ch.best.physical_qubits))


def run_design(m: int, decoder: str, dev, configs: list, units: int, seed: int = 7,
               progress: bool = True):
    """every stratum's sampled units, each scored EXACTLY at every configuration. The unit stream depends only on the
    seed (and the strata), never on the device or decoder, so devices and decoders are scored on the same units.
    dev may be a LIST of devices sharing one graph and layout: each unit is transpiled once (by the first) and scored
    under every device's model -- exactly paired. Returns a run (or a list of runs, one per device)."""
    import time

    import hw_faithful as HF
    devs = dev if isinstance(dev, list) else [dev]
    unit_rng = np.random.default_rng(seed)
    G = len(configs)
    rows: list = [[] for _ in devs]
    cz_counts = []
    peak = [0] * len(devs)
    t0 = time.time()
    i = 0
    for key, W in faithful_strata(m):
        for _ in range(units):
            u = sample_faithful(m, key, unit_rng)
            tqc = devs[0].transpile(HF.circuit(m, u, decoder, erase="none"), seed=1000 + i)
            i += 1
            cz_counts.append(int(tqc.count_ops().get("cz", 0)))
            # the per-shot score's support, for the shot budget: kept units score 0 or 2, both-erased X inputs 0 or 1,
            # both-erased Z / Y inputs exactly 1/2
            kind = "kept" if u.branch != "erased_erased" else ("ee_x" if u.inp[0] == "X" else "ee_const")
            for j, d in enumerate(devs):
                st: dict = {}
                comp = compile_circuit(tqc, d.snap, d.params, configs)
                vals = evaluate(comp, HF.score(m, u, decoder), tqc.num_clbits, hub=hub_of(tqc, 0), stats=st)
                peak[j] = max(peak[j], st["peak_terms"])
                rows[j].append(dict(key=key, weight=W, name=u.name, values=vals, kind=kind))
        if progress:
            print(f"   [{devs[0].name}{'+' if len(devs) > 1 else ''} m={m} {decoder}] {'|'.join(map(str, key)):28s} "
                  f"done  ({time.time() - t0:.0f}s)", flush=True)
    out = [dict(m=m, decoder=decoder, device=d.name, configs=configs, rows=rows[j], cz=cz_counts, peak_terms=peak[j],
                seconds=time.time() - t0, units_per_stratum=units, seed=seed, G=G) for j, d in enumerate(devs)]
    return out if isinstance(dev, list) else out[0]


def combine_design(run: dict) -> dict:
    """F_e at every configuration with its unit-sampling se (control variate on the exact ideal per unit):
    F_avg(g) = F_avg_ideal - sum_b W_b mean_b [ideal_u - value_u(g)],  se^2 = sum_b W_b^2 var_b / n_b,
    F_e = (3 F_avg - 1)/2. Config 0 must be IDEAL."""
    import hw_two_use as H
    m = run["m"]
    if run["configs"][0] != IDEAL:
        raise ValueError("the first configuration must be IDEAL: the control variate needs each unit's ideal")
    strata: dict = {}
    for r in run["rows"]:
        strata.setdefault(r["key"], dict(W=r["weight"], vals=[]))["vals"].append(r["values"])
    total_w = sum(s["W"] for s in strata.values())
    if abs(total_w - 1.0) > 1e-12:
        raise ValueError(f"strata carry weight {total_w}, not 1")
    f_e_ideal = H.f_e_closed(m)
    f_avg_ideal = (2 * f_e_ideal + 1) / 3
    G = len(run["configs"])
    loss = np.zeros(G)
    var = np.zeros(G)
    ideal_check = 0.0
    for s in strata.values():
        V = np.array(s["vals"])                         # (n_b, G)
        L = V[:, :1] - V                                # ideal - value
        loss += s["W"] * L.mean(axis=0)
        n_b = len(V)
        var += (s["W"] ** 2) * (L.var(axis=0, ddof=1) / n_b if n_b > 1 else 0.0)
        ideal_check += s["W"] * V[:, 0].mean()
    f_avg = f_avg_ideal - loss
    return dict(F_e=(3 * f_avg - 1) / 2, se=1.5 * np.sqrt(var), F_e_ideal=f_e_ideal,
                sampled_ideal_F_e=(3 * ideal_check - 1) / 2, per_unit_loss_strata=len(strata))


# ================================================================================================ analyses (pure)
def crossing(xs, Fs, bar: float = BAR) -> dict:
    """the largest x in the scanned range with F(x) >= bar, interpolated linearly between the bracketing grid points
    of the FIRST downward crossing after the smallest x; F is expected to fall with x (checked: 'monotone')"""
    xs = np.asarray(xs, float)
    Fs = np.asarray(Fs, float)
    o = np.argsort(xs)
    xs, Fs = xs[o], Fs[o]
    mono = bool(np.all(np.diff(Fs) <= 1e-12))
    if Fs[0] < bar:
        return dict(x_star=None, kind="never", monotone=mono)
    below = np.flatnonzero(Fs < bar)
    if len(below) == 0:
        return dict(x_star=None, kind="always", x_max=float(xs[-1]), monotone=mono)
    j = int(below[0])
    x0, x1, f0, f1 = xs[j - 1], xs[j], Fs[j - 1], Fs[j]
    return dict(x_star=float(x0 + (x1 - x0) * (f0 - bar) / (f0 - f1)), kind="bracket", monotone=mono,
                bracket=[float(x0), float(x1)])


def shots_to_certify(F_e: float, var_per_shot: float, alpha: float = 0.01) -> dict:
    """shots for a one-sided certificate that F_e > 3/4 at confidence 1 - alpha, if the prediction is the truth:
    CLT: N = (z sd_Fe / margin)^2 with sd_Fe = 1.5 sqrt(var(score per shot)); HOEFFDING (score range [0, 2], no
    variance assumption): N = R^2 ln(1/alpha) / (2 t^2), t = (2/3) margin in F_avg. Margin <= 0: None."""
    from statistics import NormalDist
    margin = F_e - BAR
    if margin <= 0:
        return dict(margin=margin, clt=None, hoeffding=None)
    z = NormalDist().inv_cdf(1 - alpha)
    clt = (z * 1.5 * math.sqrt(var_per_shot) / margin) ** 2
    t = 2.0 * margin / 3.0
    hoeff = 4.0 * math.log(1 / alpha) / (2.0 * t * t)
    return dict(margin=margin, clt=float(math.ceil(clt)), hoeffding=float(math.ceil(hoeff)), alpha=alpha)


def additivity(F_full: float, F_ideal: float, off: dict, alone: dict) -> dict:
    """the loss budget's closure: total = F_ideal - F_full against the sum of 'switch one off' gains and of
    'only this one on' losses"""
    total = F_ideal - F_full
    s_off = sum(F_off - F_full for F_off in off.values())
    s_alone = sum(F_ideal - F_a for F_a in alone.values())
    return dict(total=total, sum_off=s_off, sum_alone=s_alone, off_error=s_off - total, alone_error=s_alone - total)


# ================================================================================================ the configurations
BUDGET = ("cz", "sx", "ro", "idle", "other")          # probability-convention classes (NoiseParams.scaled's)
SPLIT = ("ro01", "ro10", "t1", "t2", "qs")              # finer physical splits, switched off one at a time
REQ = ("cz", "sx", "ro", "t1", "t2", "qs")              # requirement components (physical convention)
LEVELS = (1.0, 0.5, 0.25, 0.1, 0.0)                     # the other components' level in a requirement scan
LAMS = tuple(round(x, 4) for x in np.arange(0.025, 1.0001, 0.025)) + (1.25, 1.5, 2.0)
XGRID = (0.0,) + tuple(float(x) for x in np.geomspace(0.003, 10.0, 36))
EPS = 0.05
CAL = (1 / 3.8, 1 / 3.1, 1 / 2.4)                      # R183: the model's wait dephasing over-charged 2.4-3.8x


def _set(cfg: Config, comp: str, v: float) -> Config:
    if comp == "ro":
        return replace(cfg, ro01=v, ro10=v)
    if comp == "idle":
        return replace(cfg, idle=v)
    return replace(cfg, **{comp: v})


def physical_level(L: float) -> Config:
    """every component at L x the base device, decoherence as RATES (T1/L, T2/L, sigma L): the physical lambda"""
    return Config(cz=L, sx=L, ro01=L, ro10=L, idle=1.0, t1=L, t2=L, qs=L, other=L, label=f"plam:{L:g}")


def standard_configs() -> list:
    cfgs = [IDEAL, FULL]
    for c in BUDGET:
        cfgs.append(replace(_set(FULL, c, 0.0), label=f"off:{c}"))
        alone = replace(IDEAL, label=f"alone:{c}")
        cfgs.append(_set(alone, c, 1.0))
        cfgs.append(replace(_set(FULL, c, 1.0 + EPS), label=f"lr+:{c}"))
        cfgs.append(replace(_set(FULL, c, 1.0 - EPS), label=f"lr-:{c}"))
    for c in SPLIT:
        cfgs.append(replace(_set(FULL, c, 0.0), label=f"off:{c}"))
    for k in CAL:
        cfgs.append(replace(FULL, t2=k, qs=k, label=f"cal:{k:.4f}"))
    cfgs += [FULL.scaled(lam, label=f"lam:{lam:g}") for lam in LAMS]
    cfgs += [physical_level(lam) for lam in LAMS]
    for c in REQ:
        for L in LEVELS:
            base = physical_level(L)
            for x in XGRID:
                cfgs.append(replace(_set(base, c, x), label=f"req:{c}:{L:g}:{x:.6g}"))
    labels = [c.label for c in cfgs]
    if len(set(labels)) != len(labels):
        raise ValueError("configuration labels must be unique")
    return cfgs


def index_of(configs: list) -> dict:
    return {c.label: i for i, c in enumerate(configs)}


def basic_configs() -> list:
    """the budget and both lambda scans only (every device of the spec table)"""
    return [c for c in standard_configs() if not c.label.startswith("req:")]


# the requirement SURFACE: an all-to-all device in the ion template (1q gates inside the 2q layer: t_1q = 0; memory as
# pure dephasing per idle qubit per depth-1 layer), scanned over (2Q error, readout error, memory error per layer)
SURF_BASE = dict(r2q=1e-3, e1q=3e-5, p01=1e-3, p10=1e-3, mem=1e-4, t2q=1e-3, t1q=0.0, tmeas=1e-3)
SURF_CZ = tuple(round(x, 4) for x in np.arange(0.0, 10.0001, 0.25))           # x 1e-3 (avg infidelity)
SURF_RO = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)                                 # x 1e-3 (p01 = p10)
SURF_MEM = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)                          # x 1e-4 per layer


def surface_configs() -> list:
    """IDEAL, FULL, then the (cz, ro, mem) grid; memory scales as a dephasing RATE (T2 / s), i.e. a per-layer error of
    (1 - (1 - 2 mem)^s) / 2"""
    cfgs = [IDEAL, FULL]
    for a in SURF_CZ:
        for b in SURF_RO:
            for c in SURF_MEM:
                cfgs.append(Config(cz=a, sx=1.0, ro01=b, ro10=b, idle=1.0, t1=0.0, t2=c, qs=0.0, other=0.0,
                                   label=f"surf:{a:g}:{b:g}:{c:g}"))
    return cfgs


def surface_device(n: int):
    num = dict(r2q=SURF_BASE["r2q"], t2q=SURF_BASE["t2q"], e1q=SURF_BASE["e1q"], t1q=SURF_BASE["t1q"],
               p01=SURF_BASE["p01"], p10=SURF_BASE["p10"], T1=1e6,
               T2=-SURF_BASE["t2q"] / math.log1p(-2.0 * SURF_BASE["mem"]), tmeas=SURF_BASE["tmeas"])
    return all_to_all_device(n, num, f"surface_template_a2a{n}")


def shot_sd(V: np.ndarray, keys: list, W: np.ndarray, kinds: list, col: int, shots: float) -> float:
    """the F_e standard deviation of a hardware estimate from `shots` shots allocated to the strata in proportion to
    their weight, each shot on a freshly drawn unit: sd(F_e) = 1.5 sqrt(sum_b W_b Var_b(S) / N), with Var_b the
    within-stratum per-shot variance from each unit's exact E[S] and the score's support (E[S^2] = 2 E[S] for a kept
    unit, E[S] for a both-erased X input, 1/4 for a both-erased Z/Y input)"""
    groups: dict = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    tot = 0.0
    for idx in groups.values():
        es = V[idx, col]
        es2 = np.array([2 * e if kinds[i] == "kept" else (e if kinds[i] == "ee_x" else 0.25)
                        for i, e in zip(idx, es)])
        tot += W[idx[0]] * (es2.mean() - es.mean() ** 2)
    return 1.5 * math.sqrt(max(tot, 0.0) / shots)


# ================================================================================================ reading a run
def _matrix(run: dict):
    """(values (U, G), keys, weights) of a run, units in order"""
    V = np.array([r["values"] for r in run["rows"]])
    keys = [tuple(r["key"]) for r in run["rows"]]
    W = np.array([r["weight"] for r in run["rows"]])
    return V, keys, W


def design_curve(V: np.ndarray, keys: list, W: np.ndarray, m: int, cols, rng=None) -> np.ndarray:
    """F_e at the configuration columns `cols` from the unit matrix; with rng, one stratified bootstrap draw"""
    import hw_two_use as H
    f_avg_ideal = (2 * H.f_e_closed(m) + 1) / 3
    groups: dict = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    Vc = V[:, [0] + list(cols)]                      # slice the needed columns ONCE (a full-row copy per draw cost
    loss = np.zeros(len(cols))                       # ~0.2 TB of traffic over a report)
    for k, idx in groups.items():
        idx = np.array(idx)
        if rng is not None:
            idx = rng.choice(idx, size=len(idx), replace=True)
        sub = Vc[idx]
        loss += W[idx[0]] * (sub[:, :1] - sub[:, 1:]).mean(axis=0)
    return (3 * (f_avg_ideal - loss) - 1) / 2


def analyse(run: dict, base_numbers: dict | None = None, boot: int = 200, seed: int = 11) -> dict:
    """everything the report needs from one run: F_e, the loss budget, lambda* (both conventions), the requirement
    surface (with absolute numbers when base_numbers is given), each with a stratified bootstrap over units"""
    configs = run["configs"]
    ix = index_of(configs)
    V, keys, W = _matrix(run)
    comb = combine_design(run)
    F, se = comb["F_e"], comb["se"]
    m = run["m"]
    rng = np.random.default_rng(seed)
    out: dict = dict(m=m, decoder=run["decoder"], device=run["device"], F_e_ideal=comb["F_e_ideal"],
                     sampled_ideal_F_e=comb["sampled_ideal_F_e"], F_e=float(F[ix["full"]]), se=float(se[ix["full"]]),
                     units=len(run["rows"]), mean_cz=float(np.mean(run["cz"])), peak_terms=run["peak_terms"])
    # ---- the loss budget
    fi, ff = float(F[ix["ideal"]]), float(F[ix["full"]])
    off = {c: float(F[ix[f"off:{c}"]]) for c in BUDGET}
    alone = {c: float(F[ix[f"alone:{c}"]]) for c in BUDGET}
    lr = {c: float((F[ix[f"lr+:{c}"]] - F[ix[f"lr-:{c}"]]) / (2 * EPS)) for c in BUDGET}
    out["budget"] = dict(
        total_loss=fi - ff,
        off_gain={c: off[c] - ff for c in BUDGET},
        off_gain_se={c: float(se_diff(V, keys, W, m, ix[f"off:{c}"], ix["full"])) for c in BUDGET},
        alone_loss={c: fi - alone[c] for c in BUDGET},
        linear_response={c: -lr[c] for c in BUDGET},
        split_off_gain={c: float(F[ix[f"off:{c}"]]) - ff for c in SPLIT},
        additivity=additivity(ff, fi, off, alone),
        linear_sum=-sum(lr.values()))
    out["calibrated_R183"] = {f"{k:.4f}": float(F[ix[f"cal:{k:.4f}"]]) for k in CAL if f"cal:{k:.4f}" in ix}
    # ---- lambda*: probability convention (R144/R178's) and physical
    for conv, pre in (("prob", "lam"), ("phys", "plam")):
        cols = [ix[f"{pre}:{lam:g}"] for lam in LAMS]
        Fs = F[cols]
        lam_cross = crossing(LAMS, Fs)
        draws = []
        for _ in range(boot):
            Fb = design_curve(V, keys, W, m, cols, rng)
            draws.append(crossing(LAMS, Fb)["x_star"])
        good = [d for d in draws if d is not None]
        out[f"lambda_star_{conv}"] = dict(**lam_cross, curve=dict(zip([float(x) for x in LAMS],
                                                                       [float(x) for x in Fs])),
                                          boot_ci=[float(np.percentile(good, 2.5)), float(np.percentile(good, 97.5))]
                                          if len(good) > 10 else None, boot_defined=len(good) / boot)
    # ---- the requirement surface (standard runs only: a basic run carries no 'req:' configurations)
    req: dict = {}
    for c in (REQ if f"req:{REQ[0]}:{LEVELS[0]:g}:{XGRID[0]:.6g}" in ix else ()):
        req[c] = {}
        for L in LEVELS:
            cols = [ix[f"req:{c}:{L:g}:{x:.6g}"] for x in XGRID]
            cr = crossing(XGRID, F[cols])
            draws = []
            for _ in range(boot):
                Fb = design_curve(V, keys, W, m, cols, rng)
                draws.append(crossing(XGRID, Fb)["x_star"])
            good = [d for d in draws if d is not None]
            row = dict(**cr, boot_ci=[float(np.percentile(good, 2.5)), float(np.percentile(good, 97.5))]
                       if len(good) > 10 else None, boot_defined=len(good) / boot)
            if base_numbers and cr.get("x_star") is not None:
                row["absolute"] = absolute_requirement(c, cr["x_star"], base_numbers)
            req[c][f"{L:g}"] = row
    out["requirements"] = req
    return out


def se_diff(V, keys, W, m, a: int, b: int) -> float:
    """the unit-sampling se of F_e(a) - F_e(b) (paired: the same units)"""
    groups: dict = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    var = 0.0
    for idx in groups.values():
        d = V[idx, a] - V[idx, b]
        if len(idx) > 1:
            var += W[idx[0]] ** 2 * d.var(ddof=1) / len(idx)
    return 1.5 * math.sqrt(var)


def absolute_requirement(comp: str, x: float, num: dict) -> dict:
    """a requirement multiplier in device units: probabilities multiply, decoherence TIMES divide"""
    if comp == "cz":
        return dict(cz_avg_infidelity=x * num["r2q"], cz_pauli_error=1.25 * x * num["r2q"])
    if comp == "sx":
        return dict(sx_avg_infidelity=x * num["e1q"])
    if comp == "ro":
        return dict(p01=x * num["p01"], p10=x * num["p10"], mean=x * (num["p01"] + num["p10"]) / 2)
    if comp == "t1":
        return dict(T1_s=(num["T1"] / x) if x > 0 else math.inf)
    if comp == "t2":
        return dict(T2_s=(num["T2"] / x) if x > 0 else math.inf)
    if comp == "qs":
        return dict(sigma_rad_s=x * num.get("sigma", 0.0))
    raise ValueError(comp)


# ================================================================================================ evidence I/O
def save_run(run: dict, path: pathlib.Path, extra: dict | None = None) -> None:
    """the unit matrix (npz) and its index (json): enough to re-derive every number without re-running"""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    V, keys, W = _matrix(run)
    np.savez_compressed(path.with_suffix(".npz"), values=V, weights=W)
    meta = {k: v for k, v in run.items() if k not in ("rows", "configs")}
    meta["config_labels"] = [c.label for c in run["configs"]]
    meta["configs"] = [c.__dict__ for c in run["configs"]]
    meta["unit_keys"] = [list(k) for k in keys]
    meta["unit_names"] = [r["name"] for r in run["rows"]]
    meta["unit_kinds"] = [r["kind"] for r in run["rows"]]
    if extra:
        meta.update(extra)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str) + "\n", encoding="utf-8")


def load_run(path: pathlib.Path) -> dict:
    import json
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    z = np.load(path.with_suffix(".npz"))
    configs = [Config(**c) for c in meta["configs"]]
    rows = [dict(key=tuple(None if x is None else x for x in k), weight=float(w), name=n, values=v, kind=kd)
            for k, w, n, v, kd in zip(meta["unit_keys"], z["weights"], meta["unit_names"], z["values"],
                                      meta["unit_kinds"])]
    run = {k: v for k, v in meta.items()
           if k not in ("configs", "unit_keys", "unit_names", "unit_kinds", "config_labels")}
    run.update(configs=configs, rows=rows)
    return run


# ================================================================================================ the device registry
def device_specs() -> dict:
    import json
    return {s["key"]: s for s in json.loads(SPEC_FILE.read_text(encoding="utf-8"))["devices"]}


def make_device(name: str, m: int, decoder: str, layout_from: Device | None = None) -> Device:
    """(layout_from: adopt that device's layout and coupling map instead of choosing one -- a group that shares the
    transpiled circuits)"""
    if layout_from is not None:
        import oneq.hw.layout as LY
        saved = LY.choose_layout
        LY.choose_layout = lambda *a, **k: _AdoptedChoice(layout_from)
        try:
            return _make_device(name, m, decoder)
        finally:
            LY.choose_layout = saved
    return _make_device(name, m, decoder)


@dataclass
class _AdoptedChoice:
    """a LayoutChoice stand-in carrying another device's layout (make_device(layout_from=...))"""
    src: Device

    @property
    def initial_layout(self):
        return list(self.src.layout)

    @property
    def coupling_map(self):
        return self.src.coupling_map

    @property
    def best(self):
        lay = self.src.provenance.get("layout", {})
        return type("B", (), dict(proxy=lay.get("proxy"), mean_cz=lay.get("mean_cz"),
                                  physical_qubits=tuple(lay.get("physical_qubits", ()))))()

    n_candidates = 0


def _make_device(name: str, m: int, decoder: str) -> Device:
    """kingston_hh: the v2 preset on Kingston's heavy-hex, layout by oneq.hw.layout over faithful representatives;
    kingston_a2a: all-to-all with the v2 preset's device-wide medians; spec:<key>: a row of device_specs.json on its
    own topology; surface: the ion-template all-to-all device the requirement surface is scanned on"""
    from oneq.hw import noise as N
    n = 4 + 4 * m
    if name in ("kingston_hh", "kingston_hh_accum", "kingston_hh_active", "kingston_hh_zz"):
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        snap, prov = kingston_v2_snapshot()
        params = N.NoiseParams(cz_factor=1.0)
        if name == "kingston_hh_accum":                 # what the runtime's default twirl PLAYS (model v2, R175)
            params = N.NoiseParams(cz_factor=1.0, twirl_template=True, twirl_template_dress=True)
        if name == "kingston_hh_active":
            params = N.NoiseParams(cz_factor=1.0, twirl_template=True)
        if name == "kingston_hh_zz":                    # static ZZ on every edge at R161's upper reading
            from oneq.hw import snapshot as S
            d = snap.to_json()
            d["zz"] = {k: 2 * math.pi * 3.0e3 for k in d["cz"]}
            snap = S.DeviceSnapshot.from_json(d)
            prov = dict(prov, zz="2 pi x 3 kHz on every edge (R161/R153 upper reading; HYPOTHESIS)")
        dev = Device(name, FakeKingston(), snap, params, "heavy_hex", provenance=prov)
        dev.provenance["layout"] = choose_device_layout(dev, m, decoder)
        return dev
    if name == "kingston_a2a":
        snap, prov = kingston_v2_snapshot()
        med = device_medians(snap)
        be, s2 = all_to_all_device(n, med, f"kingston_v2preset_median_a2a{n}")
        return Device(name, be, s2, N.NoiseParams(cz_factor=1.0), "all_to_all",
                      provenance=dict(prov, medians={k: v for k, v in med.items() if k != "cz_shape"}))
    if name == "surface":
        be, s2 = surface_device(n)
        return Device(name, be, s2, N.NoiseParams(cz_factor=1.0), "all_to_all", provenance=dict(SURF_BASE))
    if name.startswith("spec:"):
        spec = device_specs()[name.split(":", 1)[1]]
        be, s2 = device_from_spec(spec, n)
        dev = Device(name, be, s2, N.NoiseParams(cz_factor=1.0), spec["topology"],
                     provenance=dict(spec=spec, numbers=spec_numbers(spec)))
        if spec["topology"] != "all_to_all":
            from oneq.hw import layout as LY
            pol = LY.QubitPolicy(min_t2=min(20e-6, 0.5 * spec_numbers(spec)["T2"]))
            dev.provenance["layout"] = choose_device_layout(dev, m, decoder, policy=pol)
        return dev
    raise ValueError(f"unknown device {name!r}")


def base_numbers(dev: Device) -> dict:
    """the device's own median numbers: what a requirement multiplier multiplies"""
    if dev.layout is not None:
        return device_medians(dev.snap, dev.provenance.get("layout", {}).get("physical_qubits") or None)
    return device_medians(dev.snap)


def run_path(name: str, m: int, decoder: str, configs: str) -> pathlib.Path:
    return OUTD / "runs" / f"{name.replace(':', '_')}_m{m}_{decoder}_{configs}"


def cmd_run(argv) -> int:
    import argparse
    import time
    ap = argparse.ArgumentParser(prog="run")
    ap.add_argument("--device", required=True)
    ap.add_argument("--m", default="5,6")
    ap.add_argument("--decoder", default="cz1")
    ap.add_argument("--units", type=int, default=16)
    ap.add_argument("--configs", default="standard", choices=("standard", "basic", "surface"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    cfgs = {"standard": standard_configs, "basic": basic_configs, "surface": surface_configs}[a.configs]()
    names = a.device.split(",")
    for m in (int(x) for x in a.m.split(",")):
        for dec in a.decoder.split(","):
            t0 = time.time()
            devs = [make_device(names[0], m, dec)]
            devs += [make_device(nm, m, dec, layout_from=devs[0]) if devs[0].layout is not None
                     else make_device(nm, m, dec) for nm in names[1:]]
            runs = run_design(m, dec, devs, cfgs, a.units, seed=a.seed)
            ix = index_of(cfgs)
            for nm, dev, run in zip(names, devs, runs):
                path = run_path(nm, m, dec, a.configs)
                save_run(run, path, extra=dict(provenance=dev.provenance, base_numbers=base_numbers(dev),
                                               noise=dev.params.describe(), topology=dev.topology,
                                               group=names, sources_sha256=sources_sha(),
                                               stamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
                comb = combine_design(run)
                print(f"RUN {nm} m={m} {dec}: F_e {comb['F_e'][ix['full']]:.4f} +- {comb['se'][ix['full']]:.4f} "
                      f"(ideal {comb['F_e_ideal']:.4f}; sampled ideal {comb['sampled_ideal_F_e']:.4f}); mean CZ "
                      f"{np.mean(run['cz']):.1f}; peak terms {run['peak_terms']}; {time.time() - t0:.0f}s -> "
                      f"{path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)}", flush=True)
    return 0


def _hash_sources() -> dict:
    import hashlib
    out = {}
    for rel in ("experiments/superactivation/shield_family/r186_faithful_requirements.py",
                "experiments/superactivation/shield_family/hw_faithful.py",
                "experiments/superactivation/shield_family/hw_two_use.py",
                "src/oneq/hw/noise.py", "src/oneq/hw/engine.py"):
        out[rel] = hashlib.sha256((ROOT / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return out


# hashed when this process LOADED the code: an edit made while a run is going cannot stamp its record with bytes that
# did not produce it (a receipt hashing the working copy at save time verifies only the working copy)
_SOURCES_AT_IMPORT = _hash_sources()


def sources_sha() -> dict:
    return dict(_SOURCES_AT_IMPORT)


# ================================================================================================ the self-check
def _check_device(n: int, structure: bool, sigma: float, seed: int = 5):
    """a small all-to-all device with INFLATED, heterogeneous noise (every number differs per qubit/edge), so every
    term of the model moves the answer"""
    from oneq.hw import snapshot as S
    rng = np.random.default_rng(seed)
    be = generic_backend(n, __import__("qiskit.transpiler", fromlist=["CouplingMap"]).CouplingMap.from_full(n))
    cz = {(a, b): (4e-3 * (1 + 0.3 * rng.random()), 68e-9) for a in range(n) for b in range(a + 1, n)}
    st = {}
    if structure:
        for k in cz:
            f = rng.random(15) ** 3
            st[k] = tuple(float(x) for x in f / f.sum())
    snap = S.DeviceSnapshot(name=f"check_a2a_{n}", num_qubits=n, calibrated_utc=None, dt=be.dt, cz=cz,
                            t1=tuple(60e-6 * (1 + 0.2 * rng.random()) for _ in range(n)),
                            t2=tuple(30e-6 * (1 + 0.2 * rng.random()) for _ in range(n)),
                            p01=tuple(0.02 * (1 + rng.random()) for _ in range(n)),
                            p10=tuple(0.05 * (1 + rng.random()) for _ in range(n)),
                            sx_error=tuple(3e-3 * (1 + rng.random()) for _ in range(n)),
                            sx_duration=(32e-9,) * n, meas_duration=(1.2e-6,) * n, cz_structure=st,
                            sigma={q: sigma * (1 + rng.random()) for q in range(n)} if sigma else {})
    return be, snap


def _tr(qc, be, seed=3, level=3):
    from qiskit import transpile
    return transpile(qc, be, initial_layout=list(range(qc.num_qubits)), optimization_level=level, seed_transpiler=seed)


def cmd_selfcheck(argv) -> int:
    """every check here can fail, and one is a control that MUST fail on a planted defect"""
    import time

    import hw_faithful as HF
    import hw_two_use as H

    from oneq.hw import engine as E
    from oneq.hw import noise as N
    t0 = time.time()
    fails = []
    PL = [dict(), dict(cz=0.0), dict(sx=0.0, idle=0.0), dict(ro=0.0), dict(cz=3.0, sx=2.0, idle=2.5, ro=1.5),
          dict(cz=0.0, sx=0.0, idle=0.0, ro=0.0)]
    cfgs = [Config(cz=k.get("cz", 1.0), sx=k.get("sx", 1.0), idle=k.get("idle", 1.0), ro01=k.get("ro", 1.0),
                   ro10=k.get("ro", 1.0)) for k in PL]
    base = N.NoiseParams(cz_factor=1.0)
    rng = np.random.default_rng(1)
    # ---- 1. the exact evaluator equals the density-matrix oracle (engine.exact_expectation, no shared code but the
    #         op list), to rounding, on lean m = 1, 2 and faithful m = 1, with uniform and structured CZ channels and
    #         quasi-static dephasing; rate configurations (T1 / t1, T2 / t2) against a rebuilt snapshot
    worst, n_cases = 0.0, 0
    cases = []
    for structure, sigma in ((False, 0.0), (True, 2e5)):
        for m in (1, 2):
            be, snap = _check_device(4 + 2 * m, structure, sigma)
            for branch in ("kept_kept", "kept_erased", "erased_erased"):
                u = H.sample_unit(m, branch, rng)
                cases.append((be, snap, _tr(H.circuit(m, u), be), H.score(m, u)))
        be, snap = _check_device(8, structure, sigma)
        for branch in ("kept_kept", "kept_erased", "erased_erased"):
            for dec in ("cz1", "cz2"):
                u = HF.FUnit(branch, (H.hd.sample_choice(1, rng), H.hd.sample_choice(1, rng)), int(rng.integers(2)),
                             H.INPUTS[int(rng.integers(6))])
                cases.append((be, snap, _tr(HF.circuit(1, u, dec, erase="none"), be), HF.score(1, u, dec)))
    for be, snap, tqc, sc in cases:
        vals = exact_value(tqc, sc, snap, base, cfgs, hub=hub_of(tqc, 0))
        for k, v in zip(PL, vals):
            p = N.NoiseParams(cz_factor=k.get("cz", 1.0), sx_factor=k.get("sx", 1.0), idle_factor=k.get("idle", 1.0),
                              readout_factor=k.get("ro", 1.0))
            worst = max(worst, abs(E.exact_expectation(tqc, sc, snap, p) - v))
            n_cases += 1
    print(f"[1] evaluator vs density-matrix oracle: {n_cases} (circuit, config) cases, worst |diff| {worst:.2e}")
    if not worst < 1e-10:
        fails.append(f"oracle agreement {worst:.2e} >= 1e-10")
    # rate configurations against a snapshot with T1 / 2.5 and T2 / 1.7
    from oneq.hw import snapshot as S
    be, snap, tqc, sc = cases[3]
    v_rate = exact_value(tqc, sc, snap, base, [Config(t1=2.5, t2=1.7)])[0]
    d = snap.to_json()
    d["t1"] = [x / 2.5 for x in d["t1"]]
    d["t2"] = [x / 1.7 for x in d["t2"]]
    o_rate = E.exact_expectation(tqc, sc, S.DeviceSnapshot.from_json(d), base)
    print(f"[1b] rate configuration (T1/2.5, T2/1.7) vs oracle on a rebuilt snapshot: |diff| {abs(v_rate - o_rate):.2e}")
    if not abs(v_rate - o_rate) < 1e-10:
        fails.append(f"rate configuration {abs(v_rate - o_rate):.2e}")
    # ---- 2. CONTROLS: two planted defects must each be caught by the same comparison (a mutant that touches a
    #         table the observable never reaches is dark, so each defect is planted in EVERY gate of its kind)
    #         (a) the Y-image sign of every Clifford 1q gate flipped (a phase-convention error);
    #         (b) the CZ's kick-back dropped: X (x) I maps to X (x) I instead of X (x) Z
    caught = {"y_sign": 0, "cz_kickback": 0}
    for be, snap, tqc, sc in cases:
        good = E.exact_expectation(tqc, sc, snap, base)
        for defect in caught:
            comp = compile_circuit(tqc, snap, base, cfgs[:1])
            for r in comp.recs:
                if r.out is None or not r.clifford:
                    continue
                if defect == "y_sign" and r.k == 1:
                    r.coef = r.coef.copy()
                    r.coef[3, 0] *= -1.0
                if defect == "cz_kickback" and r.k == 2 and r.out[2, 0] != 2:
                    r.out = r.out.copy()
                    r.out[2, 0] = 2
            bad = evaluate(comp, sc, tqc.num_clbits, hub=hub_of(tqc, 0))[0]
            caught[defect] += abs(bad - good) > 1e-6
    print(f"[2] CONTROLS caught on {caught} of {len(cases)} circuits (each must be > 0)")
    if min(caught.values()) == 0:
        fails.append(f"a planted defect was never caught ({caught}): check [1] is dark for it")
    # ---- 3. the vectorized idle equals noise.idle_pauli, the clamp included
    worst_i = 0.0
    for t, t1, t2, r1, r2, f in ((1e-6, 100e-6, 50e-6, 1.0, 1.0, 1.0), (3e-6, 80e-6, 300e-6, 2.0, 0.5, 0.7),
                                 (5e-7, 1e-3, 1e-3, 0.0, 3.0, 1.0)):
        got = idle_probs_vec(t, t1, t2, np.array([f]), np.array([r1]), np.array([r2]))
        T1 = t1 / r1 if r1 > 0 else 1e30
        want = tuple(f * x for x in N.idle_pauli(t, T1, t2 / r2))
        worst_i = max(worst_i, max(abs(g[0] - w) for g, w in zip(got, want)))
    print(f"[3] vectorized idle vs noise.idle_pauli: worst {worst_i:.2e}")
    if not worst_i < 1e-15:
        fails.append(f"idle vectorization {worst_i:.2e}")
    # regression (Helios, 2026-10-19): an ion idle (T1 = 1e6 s, t = 1 ms, T2 = 1 s) must recover its duration
    rec_err = []
    for t, t1, t2 in ((1e-3, 1e6, 1.0), (0.07, 1e6, 291.0), (5e-7, 230e-6, 90e-6), (2e-6, 100e-6, 199e-6)):
        rec_err.append(abs(_idle_duration(N.idle_pauli(t, t1, t2), t1, t2) - t) / t)
    print(f"[3b] idle duration recovery (ion and transmon idles): worst relative error {max(rec_err):.2e}")
    if not max(rec_err) < 1e-8:
        fails.append(f"idle duration recovery {max(rec_err):.2e}")
    # ---- 4. the faithful DESIGN's ideal F_e at m = 1, every unit enumerated, equals (7 - 6p + 2p^2)/8
    be, snap = _check_device(8, False, 0.0)
    for dec in ("cz1", "cz2"):
        per = 0.0
        for w, u in HF.design_units(1):
            tqc = _tr(HF.circuit(1, u, dec, erase="none"), be, level=1)
            per += w * exact_value(tqc, HF.score(1, u, dec), snap, base, [IDEAL], hub=hub_of(tqc, 0))[0]
        fe = (3 * per - 1) / 2
        print(f"[4] faithful design m=1 {dec}: enumerated ideal F_e {fe:.12f}, closed form {H.f_e_closed(1):.12f}")
        if not abs(fe - H.f_e_closed(1)) < 1e-10:
            fails.append(f"design ideal m=1 {dec}: {fe} vs {H.f_e_closed(1)}")
    # ---- 5. the FULL faithful circuit at m = 2 (12 qubits) against the engine's own trajectories
    be, snap = _check_device(12, True, 2e5)
    zs = []
    for branch in ("kept_kept", "kept_erased", "kept_kept", "kept_erased"):
        u = sample_faithful(2, (branch, "s1" if len(zs) % 2 else "s0", int(rng.integers(2)), "s1"), rng)
        tqc = _tr(HF.circuit(2, u, "cz2", erase="none"), be)
        v = exact_value(tqc, HF.score(2, u, "cz2"), snap, base, [FULL], hub=hub_of(tqc, 0))[0]
        r = E.expectation(tqc, HF.score(2, u, "cz2"), snap, base, 400, np.random.default_rng(len(zs)))
        # an absolute floor: when every blocked error is harmless the engine's se is rounding (1e-19), and a 1e-16
        # rounding difference would read as z ~ 1000 (found by this check's first run)
        z = (v - r.mean) / math.hypot(r.se, 1e-10)
        zs.append(z)
    print(f"[5] full faithful m=2 vs engine trajectories: z = {', '.join(f'{z:+.2f}' for z in zs)}")
    if not all(abs(z) < 4.5 for z in zs):
        fails.append(f"engine trajectories disagree: z {zs}")
    # ---- 6. the strata carry the design's whole weight; the shot-variance supports are the score's
    for m in (5, 6, 7):
        tw = sum(w for _, w in faithful_strata(m))
        if abs(tw - 1.0) > 1e-12:
            fails.append(f"strata weight {tw} at m={m}")
    print("[6] strata weights sum to 1 at m = 5, 6, 7")
    print(f"SELFCHECK {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}  ({time.time() - t0:.0f}s)")
    return 0 if not fails else 1


# ================================================================================================ the report
def _binding(budget: dict) -> dict:
    gains = budget["off_gain"]
    top = max(gains, key=lambda c: gains[c])
    lr = budget["linear_response"]
    return dict(by_switch_off=top, gain=gains[top], by_linear_response=max(lr, key=lambda c: lr[c]))


def summarize_run(path: pathlib.Path, shots=(1e5, 1e6)) -> dict:
    run = load_run(path)
    a = analyse(run, base_numbers=run.get("base_numbers"))
    V, keys, W = _matrix(run)
    kinds = [r["kind"] for r in run["rows"]]
    ix = index_of(run["configs"])
    var1 = shot_sd(V, keys, W, kinds, ix["full"], 1.0) ** 2         # F_e variance per shot
    a["shot_variance_F_e"] = var1
    a["margin"] = a["F_e"] - BAR
    a["shots_3sigma"] = (math.ceil(9 * var1 / a["margin"] ** 2) if a["margin"] > 0 else None)
    a["shots_certify_alpha0.00135"] = shots_to_certify(a["F_e"], var1 / 2.25, alpha=0.00135)
    a["binding"] = _binding(a["budget"])
    a["topology"] = run.get("topology")
    a["provenance"] = run.get("provenance")
    a["base_numbers"] = run.get("base_numbers")
    return a


def surface_table(paths: list, shots_list=(1e5, 1e6)) -> dict:
    """for each (readout, memory) point: the largest 2Q error (vendor average infidelity) at which the design clears
    3/4 at 3 sigma of a hardware estimate with N shots (and with no shot noise), on the ion-template all-to-all device,
    with the better decoder AT THAT POINT (the experimenter chooses it per device); paths: one run per decoder, same m"""
    runs = [load_run(p) for p in paths]
    ms = {r["m"] for r in runs}
    if len(ms) != 1:
        raise ValueError("one m per surface")
    out: dict = dict(m=ms.pop(), decoders=[r["decoder"] for r in runs], base=SURF_BASE, axes=dict(
        cz_avg_infidelity=[x * SURF_BASE["r2q"] for x in SURF_CZ],
        readout=[x * SURF_BASE["p01"] for x in SURF_RO],
        memory_per_layer=[(1 - (1 - 2 * SURF_BASE["mem"]) ** x) / 2 for x in SURF_MEM]), table={}, decoder_used={})
    pre = []
    for run in runs:
        V, keys, W = _matrix(run)
        pre.append((run, combine_design(run), V, keys, W, [r["kind"] for r in run["rows"]], index_of(run["configs"])))
    for N in (None,) + tuple(shots_list):
        tab, used = {}, {}
        for b in SURF_RO:
            for c in SURF_MEM:
                key = f"ro={b * SURF_BASE['p01']:.4g}|mem={(1 - (1 - 2 * SURF_BASE['mem']) ** c) / 2:.4g}"
                best = None
                for run, comb, V, keys, W, kinds, ix in pre:
                    cols = [ix[f"surf:{a:g}:{b:g}:{c:g}"] for a in SURF_CZ]
                    Fs = comb["F_e"][cols]
                    if N is not None:
                        Fs = np.array([f - 3 * shot_sd(V, keys, W, kinds, col, N) for f, col in zip(Fs, cols)])
                    cr = crossing(SURF_CZ, Fs)
                    x = math.inf if cr["kind"] == "always" else (cr["x_star"] if cr["x_star"] is not None else -1.0)
                    if best is None or x > best[0]:
                        best = (x, run["decoder"])
                x, dec = best
                tab[key] = (None if x < 0 else (f">{SURF_CZ[-1] * SURF_BASE['r2q']:.4g}" if math.isinf(x)
                                                 else x * SURF_BASE["r2q"]))
                used[key] = dec
        lab = "no_shot_noise" if N is None else f"N={int(N)}"
        out["table"][lab] = tab
        out["decoder_used"][lab] = used
    out["prediction_se_at_base"] = {r[0]["decoder"]: float(r[1]["se"][r[6]["full"]]) for r in pre}
    return out


def cmd_report(argv) -> int:
    import json
    runs = sorted((OUTD / "runs").glob("*.json"))
    out: dict = dict(schema="oneq.r186.faithful_requirements/1", bar=BAR, runs={}, surface={},
                     sources_sha256=sources_sha())
    surf: dict = {}
    for p in runs:
        base = p.with_suffix("")
        if base.name.endswith("_surface"):
            surf.setdefault(base.name.split("_")[1], []).append(base)
            continue
        out["runs"][base.name] = summarize_run(base)
        a = out["runs"][base.name]
        print(f"{base.name:44s} F_e {a['F_e']:.4f} +- {a['se']:.4f}  margin {a['margin']:+.4f}  "
              f"lambda*_prob {a['lambda_star_prob'].get('x_star')}  binding {a['binding']['by_switch_off']} "
              f"(+{a['binding']['gain']:.4f})  3-sigma shots {a['shots_3sigma']}")
    for mlab, paths in sorted(surf.items()):
        out["surface"][mlab] = surface_table(paths)
        t = out["surface"][mlab]["table"]
        print(f"surface {mlab}: largest CZ error clearing 3/4 at 3 sigma, N = 1e6 shots (decoder per point):")
        for k, v in t["N=1000000"].items():
            print(f"   {k:28s} {v}")
    out["headline"] = headline(out["runs"])
    for line in out["headline"]["lines"]:
        print(line)
    dest = OUTD / "r186_summary.json"
    dest.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    print("wrote", dest.relative_to(ROOT).as_posix() if dest.is_relative_to(ROOT) else str(dest))
    return 0


def headline(runs: dict) -> dict:
    """per (device, m): the better decoder (higher F_e), its margin, 3-sigma shot budget and binding lever; and the
    component improvement factors on Kingston (1 / x*: how many times smaller each error, or longer each T, must be
    with the others at a stated level)"""
    import re
    best: dict = {}
    for name, a in runs.items():
        hit = re.match(r"^(.*)_m(\d+)_(cz1|cz2|mix)_(standard|basic)$", name)
        if not hit:
            continue
        dev = hit.group(1)                             # (a plain rsplit on '_m' split 'mix' runs off their device)
        key = (dev, a["m"])
        if key not in best or a["F_e"] > best[key]["F_e"]:
            best[key] = dict(run=name, **{k: a[k] for k in ("F_e", "se", "margin", "decoder", "mean_cz",
                                                             "shots_3sigma", "binding", "calibrated_R183")},
                             lambda_star_prob=a["lambda_star_prob"].get("x_star"),
                             lambda_star_phys=a["lambda_star_phys"].get("x_star"),
                             lambda_ci_prob=a["lambda_star_prob"].get("boot_ci"),
                             lambda_ci_phys=a["lambda_star_phys"].get("boot_ci"),
                             off_gain=a["budget"]["off_gain"], total_loss=a["budget"]["total_loss"])
    lines = [f"{'device':28s} m  dec   F_e      se      margin   lambda*(phys)    binding (gain)        3-sigma shots"]
    rows = []
    for (dev, m), b in sorted(best.items()):
        lam = b["lambda_star_phys"]                    # < 1: the device must be 1/lambda* x quieter; > 1: headroom
        inv = f"{lam:6.3f}" if lam else (" >2.000" if b["margin"] > 0 else "  none ")
        bind = f"{b['binding']['by_switch_off']} (+{b['binding']['gain']:.3f})"
        lines.append(f"{dev:28s} {m}  {b['decoder']:4s} {b['F_e']:.4f}  {b['se']:.4f}  {b['margin']:+.4f}  {inv:>14s}  "
                     f"{bind:22s} {b['shots_3sigma']}")
        rows.append(dict(device=dev, m=m, **b))
    factors: dict = {}
    for name, a in runs.items():
        if name.startswith("kingston_") and name.endswith("_standard") and "requirements" in a:
            fx = {}
            for c, levels in a["requirements"].items():
                fx[c] = {L: (None if r.get("x_star") in (None, 0) else 1.0 / r["x_star"]) if r["kind"] == "bracket"
                         else r["kind"] for L, r in levels.items()}
            factors[name] = fx
    return dict(best=rows, lines=lines, kingston_improvement_factors=factors)


def main(argv) -> int:
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    if cmd in ("selfcheck", "validate"):
        return cmd_selfcheck(rest)
    if cmd == "run":
        return cmd_run(rest)
    if cmd == "report":
        return cmd_report(rest)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
