"""Two-qubit resynthesis against what the circuit ASKS of each block, not against the block's full unitary (R177).

A compiler synthesizes every maximal two-qubit block (a run of gates on one pair, single-qubit gates between) as its
whole 4x4 unitary, and qiskit does that at the KAK floor: R152's frozen R146 and R147 circuits spend exactly their
floor, 5,688 and 3,979 CZ. But the circuit does not ask for the whole unitary everywhere:

  STATE   both inputs fresh |00>  -> only u|00> matters: a state preparation, 0 CZ if it is a product state, else 1
          (its Schmidt form: ry, one CX, local unitaries);
  ISO     one input fresh |0>     -> only u on that half matters: an isometry from one qubit to two, which needs at
          most 2 CZ. The completion u (|0><0| (x) I + |1><1| (x) G) is searched over G until its third Weyl
          coordinate vanishes;
  DIAG    both outputs measured in Z next -> u is needed only up to a diagonal: exp(i phi ZZ) u is scanned over phi
          (a local Z part never moves the Weyl class), at most 2 CZ for every u (Shende, Markov and Bullock).

"Fresh" is nothing but barriers or delays on the wire since the circuit's start or a reset. "Measured next" is a
measurement with only rz, barriers or delays between (rz is diagonal and folds into the dropped diagonal). Blocks
inside control-flow bodies are left alone: a body's wires start mid-circuit.

Every rewrite is CHECKED against its contract before it is used (state: the prepared vector; iso: the fresh half's
columns; diag: w u^dagger diagonal), to 1e-8, and a block whose rewrite misses is kept as it was and reported.
`equivalent` then checks the WHOLE circuit: the exact outcome distribution over every classical bit, original against
rewritten (oneq.hw.dynamic_exact on the circuits compacted to their active qubits).

A rewrite changes WHICH gates run, not only how many: under the engine and on hardware the fewer CZs are only worth
what they cost there, which the device model prices, not this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TOL = 1e-8
PRODUCT_TOL = 1e-10       # a state whose second Schmidt coefficient is below this is prepared as a product
QUARTER = math.pi / 4
ZZ_DIAG = np.array([1.0, -1.0, -1.0, 1.0])
PHI_SCAN = 721
ISO_RESTARTS = 12
POST_TRANSPARENT = {"barrier", "delay", "rz"}
PRE_TRANSPARENT = {"barrier", "delay"}
I2 = np.eye(2, dtype=complex)
P0 = np.diag([1.0, 0.0]).astype(complex)
P1 = np.diag([0.0, 1.0]).astype(complex)


class ContractRefused(ValueError):
    """a block or circuit this pass cannot read or rewrite faithfully"""


# ================================================================== the algebra
def weyl(u: np.ndarray) -> tuple:
    """the UNSPECIALIZED Weyl coordinates of u. qiskit's default (fidelity 1 - 1e-9) snaps coordinates within that
    fidelity to the nearest special class, which makes every objective below flat across a basin: the first version
    stopped 2.6e-5 from a controlled phase's root, where the exact CZ floor still read 2, not 0"""
    from qiskit.synthesis import TwoQubitWeylDecomposition
    d = TwoQubitWeylDecomposition(u, fidelity=None)
    return float(d.a), float(d.b), float(d.c)


def _decomposer():
    from qiskit.circuit.library import CZGate
    from qiskit.synthesis import TwoQubitBasisDecomposer
    return TwoQubitBasisDecomposer(CZGate(), euler_basis="ZSX")


def cz_floor(u: np.ndarray) -> int:
    return int(_decomposer().num_basis_gates(u))


def _equal_up_to_phase(a: np.ndarray, b: np.ndarray, tol: float = TOL) -> bool:
    k = np.vdot(a.reshape(-1), b.reshape(-1))
    if abs(k) < 1e-12:
        return False
    return float(np.max(np.abs(a * (k / abs(k)) - b))) < tol


def state_unitary(psi: np.ndarray) -> np.ndarray:
    """a two-qubit unitary w with w|00> = psi (little-endian: index = 2 q1 + q0) and at most one CZ: psi's Schmidt
    form sum_k s_k |a_k>_1 |b_k>_0 is ry(2 theta) on q0, a CX from q0 to q1, then A on q1 and B on q0"""
    psi = np.asarray(psi, dtype=complex).reshape(4)
    psi = psi / np.linalg.norm(psi)
    m = psi.reshape(2, 2)                                    # m[i1, i0]
    a, s, vh = np.linalg.svd(m)
    b = vh.T                                                 # columns b_k with b_k[i0] = vh[k, i0]
    if s[1] < PRODUCT_TOL:
        # a product state: local unitaries alone prepare it. (The CX below would be a no-op on |00> but still a CZ
        # in the unitary -- the first version kept it, and so found none of the 772 CZs that act on |0> (x) anything)
        return np.kron(a, b)
    theta = math.atan2(float(s[1]), float(s[0]))
    ry = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=complex)
    cx = np.eye(4, dtype=complex)[[0, 3, 2, 1]]              # control q0, target q1: |q1 q0>: 01 -> 11, 11 -> 01
    return np.kron(a, b) @ cx @ np.kron(I2, ry)


def isometry_completion(u: np.ndarray, fresh_qubit: int, rng: np.random.Generator) -> np.ndarray | None:
    """a unitary that equals u on the half where `fresh_qubit` is |0> and has a vanishing third Weyl coordinate (so 2
    CZ), or None if the search misses: u (P0 (x) I + P1 (x) G) over G in SU(2) (ZYZ angles), Nelder-Mead restarts"""
    from scipy.optimize import minimize

    def complete(x):
        g = _zyz(*x)
        r = (np.kron(I2, P0) + np.kron(g, P1)) if fresh_qubit == 0 else (np.kron(P0, I2) + np.kron(P1, g))
        return u @ r

    def cost(x):
        return abs(weyl(complete(x))[2])
    best = None
    for _ in range(ISO_RESTARTS):
        res = minimize(cost, rng.uniform(-math.pi, math.pi, 3), method="Nelder-Mead",
                       options=dict(xatol=1e-12, fatol=1e-14, maxiter=4000))
        if best is None or res.fun < best.fun:
            best = res
        if best.fun < 1e-10:
            break
    if best is None:                    # no restart ran (ISO_RESTARTS = 0): no completion was searched for
        return None
    w = complete(best.x)
    return w if cz_floor(w) <= 2 else None


def _zyz(a: float, b: float, c: float) -> np.ndarray:
    rz = lambda t: np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])  # noqa: E731
    ry = np.array([[math.cos(b / 2), -math.sin(b / 2)], [math.sin(b / 2), math.cos(b / 2)]], dtype=complex)
    return rz(a) @ ry @ rz(c)


def _with_zz(u: np.ndarray, phi: float) -> np.ndarray:
    return np.diag(np.exp(1j * phi * ZZ_DIAG)) @ u


def diagonal_completion(u: np.ndarray) -> np.ndarray:
    """exp(i phi ZZ) u at a phi of fewest CZ. A 2-CZ form needs the third Weyl coordinate EXACTLY zero, which no grid
    lands on: c(phi) is scanned over one period [0, pi) (exp(i pi ZZ) is a global phase), every sign change between
    neighbours is refined to its root by brentq, and a root is kept only if its CZ floor really drops (a sign change
    can also be the chamber's own jump, which gives no drop). The grid points themselves catch a 1- or 0-CZ form."""
    from scipy.optimize import brentq, minimize_scalar
    best_w, best = u, cz_floor(u)
    phis = np.linspace(0.0, math.pi, PHI_SCAN, endpoint=False)
    step = phis[1] - phis[0]
    coords = np.array([weyl(_with_zz(u, p)) for p in phis])
    cs = coords[:, 2]
    # a 0-CZ form (all three coordinates zero) or a 1-CZ form (b = c = 0 and a = pi/4) is a touch-zero of a
    # non-negative distance, not a sign change: refine the best grid point of each by a bounded minimization
    for target in (0, 1):
        if best <= target:
            continue

        def dist(x, target=target):
            a, b, c = weyl(_with_zz(u, x))
            return (a if target == 0 else a - QUARTER) ** 2 + b * b + c * c
        d_grid = np.array([(r[0] if target == 0 else r[0] - QUARTER) ** 2 + r[1] ** 2 + r[2] ** 2 for r in coords])
        i0 = int(np.argmin(d_grid))
        res = minimize_scalar(dist, bounds=(phis[i0] - step, phis[i0] + step), method="bounded",
                              options=dict(xatol=1e-13, maxiter=500))
        w = _with_zz(u, float(res.x))
        f = cz_floor(w)
        if f < best:
            best_w, best = w, f
    for i, p in enumerate(phis):
        if best == 0:
            break
        w = _with_zz(u, p)
        f = cz_floor(w)
        if f < best:
            best_w, best = w, f
        c0, c1 = cs[i], cs[(i + 1) % len(phis)]
        if best > 2 and c0 * c1 < 0:
            root = brentq(lambda x: weyl(_with_zz(u, x))[2], p, p + step, xtol=1e-15, rtol=1e-15, maxiter=200)
            w = _with_zz(u, root)
            f = cz_floor(w)
            if f < best:
                best_w, best = w, f
    return best_w


def satisfies(u: np.ndarray, w: np.ndarray, fresh: tuple, measured: tuple, rule: str) -> bool:
    """does w stand for u under the contract `rule`?"""
    if rule == "state":
        return _equal_up_to_phase(w[:, 0], u[:, 0])
    if rule == "iso":
        f = 0 if fresh[0] else 1
        cols = [i for i in range(4) if ((i >> f) & 1) == 0]
        return _equal_up_to_phase(w[:, cols], u[:, cols])
    if rule == "diag":
        # w u^dagger is unitary, so being diagonal is the whole condition (unit-modulus entries follow)
        d = w @ u.conj().T
        return float(np.max(np.abs(d - np.diag(np.diag(d))))) < TOL
    if rule == "full":
        return _equal_up_to_phase(w, u)
    raise ContractRefused(f"no contract rule {rule!r}")


# ================================================================== reading a circuit
def _wire_orders(dag) -> dict:
    orders: dict = {}
    for n in dag.topological_op_nodes():
        for q in n.qargs:
            orders.setdefault(q, []).append(n)
    return orders


def _neighbour(wire: list, node, forward: bool, skip: set):
    i = wire.index(node)
    step = 1 if forward else -1
    j = i + step
    while 0 <= j < len(wire):
        if wire[j].op.name not in skip:
            return wire[j]
        j += step
    return None


@dataclass
class Block:
    nodes: list
    qubits: tuple                 # the DAG's qubit objects; qubits[0] is position (little-endian qubit) 0
    unitary: np.ndarray
    cz: int
    fresh: tuple
    measured: tuple
    floor: int
    rule: str = "full"
    target: int = 0
    replacement: np.ndarray | None = None
    note: str = ""


def _block_unitary(nodes, qubits: tuple) -> np.ndarray:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(2)
    pos = {q: i for i, q in enumerate(qubits)}
    for n in nodes:
        if n.op.name in PRE_TRANSPARENT:
            continue
        if n.op.name in ("measure", "reset") or getattr(n.op, "blocks", None):
            raise ContractRefused(f"{n.op.name} inside a two-qubit block")
        qc.append(n.op, [pos[q] for q in n.qargs])
    return Operator(qc).data


def blocks(dag, unknown=()) -> list:
    """every top-level maximal two-qubit block of `dag` with its contract context. `unknown`: qubit indices whose
    starting state is NOT |0> in every unit of the design (the protocol's input wire): they are never "fresh" until a
    reset, so no rewrite can lean on the input a particular unit happens to feed the channel"""
    from qiskit.transpiler.passes import Collect2qBlocks
    p = Collect2qBlocks()
    p.run(dag)
    wires = _wire_orders(dag)
    out = []
    for nodes in p.property_set["block_list"]:
        qs = tuple(sorted({q for n in nodes for q in n.qargs}, key=lambda q: dag.find_bit(q).index))
        if len(qs) != 2 or not any(len(n.qargs) == 2 for n in nodes):
            continue
        fresh, measured = [], []
        for q in qs:
            on_q = [n for n in wires[q] if n in set(nodes)]
            before = _neighbour(wires[q], on_q[0], False, PRE_TRANSPARENT)
            at_start = before is None
            fresh.append((at_start and dag.find_bit(q).index not in unknown) or
                         (before is not None and before.op.name == "reset"))
            after = _neighbour(wires[q], on_q[-1], True, POST_TRANSPARENT)
            measured.append(after is not None and after.op.name == "measure")
        u = _block_unitary(nodes, qs)
        out.append(Block(nodes=list(nodes), qubits=qs, unitary=u, cz=sum(1 for n in nodes if n.op.name == "cz"),
                         fresh=tuple(fresh), measured=tuple(measured), floor=cz_floor(u)))
    return out


def plan(b: Block, rng: np.random.Generator) -> Block:
    """the cheapest contract-faithful replacement of one block (b.replacement None: keep it)"""
    u = b.unitary
    cands = []
    if all(b.fresh):
        cands.append(("state", state_unitary(u[:, 0])))
    elif any(b.fresh) and b.floor > 2:
        w = isometry_completion(u, 0 if b.fresh[0] else 1, rng)
        if w is not None:
            cands.append(("iso", w))
    if all(b.measured) and b.floor > 0:
        cands.append(("diag", diagonal_completion(u)))
    best = None
    for rule, w in cands:
        f = cz_floor(w)
        if f >= b.floor:
            continue
        if not satisfies(u, w, b.fresh, b.measured, rule):
            b.note += f"{rule} candidate missed its contract; "
            continue
        if best is None or f < best[1]:
            best = (rule, f, w)
    if best is not None:
        b.rule, b.target, b.replacement = best[0], best[1], best[2]
    else:
        b.target = b.floor
    return b


def _synth(w: np.ndarray):
    """w as a two-qubit circuit in the device basis (cz, rz, sx, x) at its CZ floor"""
    from qiskit import QuantumCircuit
    c = _decomposer()(w)
    out = QuantumCircuit(2)
    for inst in c.data:
        out.append(inst.operation, [c.find_bit(q).index for q in inst.qubits])
    return out


@dataclass
class Report:
    blocks: int = 0
    rewritten: int = 0
    cz_before: int = 0
    cz_after: int = 0
    by_rule: dict = field(default_factory=dict)
    misses: list = field(default_factory=list)
    skipped: int = 0

    def to_json(self) -> dict:
        return dict(blocks=self.blocks, rewritten=self.rewritten, cz_before=self.cz_before, cz_after=self.cz_after,
                    saved=self.cz_before - self.cz_after, by_rule=self.by_rule, misses=self.misses,
                    skipped=self.skipped)


def rewrite(qc, seed: int = 177, unknown=(), skip_pairs=()):
    """(qc rewritten, Report): every top-level two-qubit block replaced by its cheapest contract-faithful form.
    Pass the protocol's input wire(s) as `unknown` (see `input_wires`): a rewrite that leaned on one unit's input
    would specialize the channel to it -- exact for that unit, and no longer the same experiment (`faithful`).
    `skip_pairs`: qubit-index pairs whose blocks are left exactly as they are (a prediction stand-in's zero-duration
    VIRTUAL edges, which model a feed-forward rather than run a gate)."""
    from qiskit.circuit.library import UnitaryGate
    from qiskit.converters import circuit_to_dag, dag_to_circuit
    dag = circuit_to_dag(qc)
    rng = np.random.default_rng(seed)
    skip = {frozenset(map(int, p)) for p in skip_pairs}
    rep = Report(cz_before=sum(1 for i in qc.data if i.operation.name == "cz"))
    for b in blocks(dag, unknown=frozenset(unknown)):
        if frozenset(dag.find_bit(q).index for q in b.qubits) in skip:
            rep.skipped += 1
            continue
        rep.blocks += 1
        plan(b, rng)
        if b.note:
            rep.misses.append(b.note)
        if b.replacement is None:
            continue
        sub = _synth(b.replacement)
        if sum(1 for i in sub.data if i.operation.name == "cz") != b.target:
            rep.misses.append(f"synthesis gave {sum(1 for i in sub.data if i.operation.name == 'cz')} CZ for a "
                              f"target of {b.target}")
            continue
        node = dag.replace_block_with_op(b.nodes, UnitaryGate(b.replacement, check_input=False),
                                         {q: i for i, q in enumerate(b.qubits)}, cycle_check=True)
        dag.substitute_node_with_dag(node, circuit_to_dag(sub))
        rep.rewritten += 1
        rep.by_rule[b.rule] = rep.by_rule.get(b.rule, 0) + 1
    out = dag_to_circuit(dag)
    rep.cz_after = sum(1 for i in out.data if i.operation.name == "cz")
    return out, rep


# ================================================================== is it still the same experiment?
def input_wires(qc, logical=(0,)) -> tuple:
    """the physical qubits that START the given logical qubits (the circuit's initial layout); a circuit with no
    layout is its own logical circuit"""
    if qc.layout is None:
        return tuple(logical)
    lay = qc.layout.initial_index_layout(filter_ancillas=True)
    return tuple(int(lay[i]) for i in logical)


def _pauli_preps():
    """six preparations whose states (the Pauli eigenstates) span a qubit's operators: equal outcomes on all six mean
    equal outcomes on EVERY input, by linearity in the input density matrix"""
    from qiskit import QuantumCircuit
    out = []
    for seq in ((), ("x",), ("h",), ("x", "h"), ("h", "s"), ("x", "h", "s")):
        c = QuantumCircuit(1)
        for g in seq:
            getattr(c, g)(0)
        out.append(c)
    return out


def faithful(original, rewritten, qubit: int) -> float:
    """the largest outcome difference between original and rewritten over every input on `qubit`: each of the six
    Pauli preparations is put at the START of that wire in both circuits. 0 (to rounding) means the rewrite holds for
    any state the protocol might feed there -- it did not specialize the circuit to this unit's input"""
    worst = 0.0
    for prep in _pauli_preps():
        pair = []
        for c in (original, rewritten):
            new = c.copy_empty_like()
            for inst in prep.data:
                new.append(inst.operation, [new.qubits[qubit]])
            for inst in c.data:
                new.append(inst)
            pair.append(new)
        worst = max(worst, equivalent(*pair))
    return worst


# ================================================================== checking a whole circuit
def active(qc) -> list:
    return sorted({qc.find_bit(q).index for inst in qc.data for q in inst.qubits})


def compact(qc, used: list | None = None):
    """(qc on the qubits `used` -- by default its ACTIVE ones, those any instruction touches -- with clbits and
    registers unchanged, used)"""
    from qiskit import QuantumCircuit, QuantumRegister
    used = active(qc) if used is None else used
    new = QuantumCircuit(QuantumRegister(len(used), "q"), *qc.cregs)
    loose = [c for c in qc.clbits if not qc.find_bit(c).registers]
    if loose:
        new.add_bits(loose)
    index = {qc.qubits[i]: new.qubits[k] for k, i in enumerate(used)}
    for inst in qc.data:
        new.append(inst.operation, [index[q] for q in inst.qubits], list(inst.clbits))
    return new, used


def equivalent(a, b) -> float:
    """the largest difference between the exact outcome distributions of a and b over every classical bit"""
    from oneq.hw import dynamic_exact as DX
    if a.num_qubits != b.num_qubits or a.num_clbits != b.num_clbits:
        raise ContractRefused("the two circuits do not have the same qubits and clbits")
    # on the union of both circuits' active qubits: a rewrite can leave a wire idle that the original touched
    both = sorted(set(active(a)) | set(active(b)))
    pa, pb = DX.distribution(compact(a, both)[0]).probs, DX.distribution(compact(b, both)[0]).probs
    keys = set(pa) | set(pb)
    return max(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys) if keys else 0.0


__all__ = ["rewrite", "Report", "blocks", "plan", "equivalent", "compact", "state_unitary", "isometry_completion",
           "diagonal_completion", "satisfies", "cz_floor", "weyl", "ContractRefused"]
