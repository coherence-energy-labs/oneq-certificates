"""Feed-forward Pauli corrections tracked in SOFTWARE: a conditional Pauli leaves the device, so nothing waits for it.

R147's dynamic design corrects the key with `if_test(parity of the shield reads): z(key)`. On IBM hardware a
conditioned qubit waits for the read's broadcast: the conditional plays 2.704 us after the read STARTS (R166), and in
the circuits R152 ran the key had finished its preparation long before, so it idled 2.572 us at every conditional
(`dynamic_timing.conditional_waits`). The device model charges that wait and closes R147 (R171); the wait is the
design's largest loss.

A Pauli correction need not be applied by the device at all. If every operation after it, on the qubits its image
touches, is a Clifford gate (or a measurement), the Pauli propagates exactly to the measurements:
  - an X or Y component on a qubit at its Z measurement FLIPS that outcome -- and stays on the qubit afterwards
    (the post-measurement state is flipped too);
  - a Z component at a Z measurement does nothing and is dropped.
So `track(qc)` removes each conditional Pauli, propagates it, and returns the static circuit plus a FRAME: for each
classical bit, the parity of mid-circuit bits it must be XORed with. `apply(bits, frame)` corrects one outcome;
`corrected_counts` a histogram. The correction is exact and deterministic, not statistical: the corrected outcome
law equals the original circuit's EXACTLY (`check`, by oneq.hw.dynamic_exact, for every input on a wire with the six
Pauli preparations of contract_synth.faithful).

Refusals (FrameRefused): a conditional whose body is not a single-qubit Pauli; a condition that is not a clbit or
an XOR of clbits; a Pauli whose image reaches a non-Clifford gate (an rz off multiples of pi/2, say) on a qubit it
acts on; a conditional inside another control-flow body.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

PAULI_GATES = {"x": "X", "y": "Y", "z": "Z", "id": "I"}
CLIFFORD_1Q = {"h", "s", "sdg", "sx", "sxdg", "x", "y", "z", "id"}
CLIFFORD_2Q = {"cz", "cx", "swap", "ecr"}
TOL = 1e-9


class FrameRefused(ValueError):
    """a conditional this module cannot move into software exactly"""


@dataclass
class Frame:
    flips: dict = field(default_factory=dict)    # clbit index -> frozenset of clbit indices whose parity it XORs
    removed: int = 0

    def to_json(self) -> dict:
        return dict(removed=self.removed, flips={str(k): sorted(v) for k, v in self.flips.items() if v})


def _clifford_angle(theta: float) -> bool:
    k = theta / (math.pi / 2)
    return abs(k - round(k)) < TOL


def _cond_bits(cond, qc) -> frozenset:
    """the clbit indices whose XOR the condition is (a (clbit, 1) pair, or an expr of bit_xor over clbit vars)"""
    from qiskit.circuit import Clbit
    if isinstance(cond, tuple):
        c, v = cond
        if not isinstance(c, Clbit) or int(v) != 1:
            raise FrameRefused(f"condition {cond!r} is not 'this clbit is 1'")
        return frozenset({qc.find_bit(c).index})
    from qiskit.circuit.classical import expr
    if isinstance(cond, expr.Var):
        if not isinstance(cond.var, Clbit):
            raise FrameRefused("a condition on a classical variable is not a parity of measured bits")
        return frozenset({qc.find_bit(cond.var).index})
    if isinstance(cond, expr.Binary) and cond.op == expr.Binary.Op.BIT_XOR:
        return _cond_bits(cond.left, qc) ^ _cond_bits(cond.right, qc)
    if isinstance(cond, expr.Cast):
        return _cond_bits(cond.operand, qc)
    raise FrameRefused(f"condition {cond!r} is not an XOR of clbits")


_PAULI_MATS = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
               "Z": np.diag([1, -1])}


def _body_pauli(op, qc_outer, inst):
    """(qubit index in the outer circuit, Pauli letter) of a conditional whose true body is a single-qubit Pauli UP TO
    A PHASE -- read off the body's unitary, so a compiled z (an rz(pi)) or an x (sx sx) counts"""
    from qiskit.quantum_info import Operator
    blocks = op.blocks
    if len(blocks) > 1 and any(len(b.data) for b in blocks[1:]):
        raise FrameRefused("a conditional with an else branch")
    body = blocks[0]
    gates = [i for i in body.data if i.operation.name not in ("barrier", "delay")]
    touched = {body.find_bit(q).index for g in gates for q in g.qubits}
    if len(touched) != 1 or any(len(g.qubits) != 1 for g in gates):
        raise FrameRefused(f"a conditional whose body is not on one qubit: {[g.operation.name for g in gates]}")
    (inner_q,) = touched
    from qiskit import QuantumCircuit
    one = QuantumCircuit(1)
    for g in gates:
        one.append(g.operation, [0])
    u = Operator(one).data
    for letter, m in _PAULI_MATS.items():
        k = np.vdot(m.reshape(-1), u.reshape(-1)) / 2
        if abs(abs(k) - 1) < TOL and np.max(np.abs(u - k * m)) < TOL:
            outer_q = qc_outer.find_bit(inst.qubits[inner_q]).index
            return outer_q, letter
    raise FrameRefused(f"a conditional whose body is not a Pauli: {[g.operation.name for g in gates]}")


def _evolve(pauli, inst, qc):
    """the Pauli after one more instruction (Heisenberg-forward: P -> G P G^dagger), refusing a non-Clifford gate that
    acts where the Pauli does"""
    from qiskit.quantum_info import Clifford
    qs = [qc.find_bit(q).index for q in inst.qubits]
    acts = any(pauli.x[q] or pauli.z[q] for q in qs)
    name = inst.operation.name
    if not acts or name in ("barrier", "delay"):
        return pauli
    if name == "rz":
        if not _clifford_angle(float(inst.operation.params[0])):
            raise FrameRefused(f"the frame meets rz({float(inst.operation.params[0]):.4f}) on qubit {qs[0]}")
    elif name not in CLIFFORD_1Q | CLIFFORD_2Q:
        raise FrameRefused(f"the frame meets a non-Clifford {name} on qubits {qs}")
    return pauli.evolve(Clifford(inst.operation), qargs=qs, frame="s")


def track(qc):
    """(static circuit, Frame): every top-level conditional Pauli removed and moved into the frame"""
    from qiskit.quantum_info import Pauli
    n = qc.num_qubits
    frame = Frame()
    out = qc.copy_empty_like()
    pending = []                                   # (pauli on all qubits, frozenset of condition clbits)
    for inst in qc.data:
        op = inst.operation
        if getattr(op, "blocks", None):
            if op.name != "if_else":
                raise FrameRefused(f"{op.name}: only if_test conditionals can move to software")
            q, letter = _body_pauli(op, qc, inst)
            bits = _cond_bits(op.condition, qc)
            # A CHAINED conditional -- its control bit is itself frame-corrected -- would be applied to the RAW bit by
            # `apply`, which reads uncorrected outcomes: `if c0: x q2; measure q2 -> c1; if c1: z q0` diverges by 1.0
            # with an ancilla in |1> while check() (ancillas in |0>) saw 0 (R183 adversary, 2026-10-18). Refused.
            chained = sorted(b for b in bits if frame.flips.get(b))
            if chained:
                raise FrameRefused(f"a conditional controlled by frame-corrected bit(s) {chained}: chained "
                                   "conditionals are not supported")
            label = ["I"] * n
            label[n - 1 - q] = letter
            pending.append((Pauli("".join(label)), bits))
            frame.removed += 1
            continue
        if op.name == "measure":
            q = qc.find_bit(inst.qubits[0]).index
            c = qc.find_bit(inst.clbits[0]).index
            nxt = []
            for p, bits in pending:
                if p.x[q]:                         # an X or Y component flips the outcome, and stays on the qubit
                    frame.flips[c] = frame.flips.get(c, frozenset()) ^ bits
                if p.z[q]:                         # a Z component at a Z readout does nothing: dropped
                    p = p.copy()
                    p.z[q] = False
                nxt.append((p, bits))
            pending = nxt
            out.append(inst)
            continue
        if op.name == "reset":
            q = qc.find_bit(inst.qubits[0]).index
            nxt = []
            for p, bits in pending:
                p = p.copy()
                p.x[q] = False
                p.z[q] = False
                nxt.append((p, bits))
            pending = nxt
            out.append(inst)
            continue
        pending = [(_evolve(p, inst, qc), bits) for p, bits in pending]
        out.append(inst)
    # whatever is left sits on qubits never measured again: a local operation on a discarded subsystem leaves the
    # measured qubits' joint law unchanged, so it is dropped (check() confirms the whole law exactly)
    return out, frame


def apply(bits: tuple, frame: Frame) -> tuple:
    """one outcome (clbit values in clbit order) corrected by the frame"""
    b = list(bits)
    for c, src in frame.flips.items():
        if src:
            b[c] ^= sum(bits[s] for s in src) & 1
    return tuple(b)


def corrected_counts(counts: dict, frame: Frame, n_clbits: int) -> dict:
    """a qiskit-style histogram (clbit 0 rightmost) corrected by the frame"""
    out: dict = {}
    for k, v in counts.items():
        bits = tuple(int(ch) for ch in k.replace(" ", "")[::-1].ljust(n_clbits, "0")[:n_clbits])
        nb = apply(bits, frame)
        key = "".join(str(x) for x in nb[::-1])
        out[key] = out.get(key, 0) + v
    return out


def check(original, tracked, frame: Frame, wire: int | None = None) -> float:
    """the largest difference between the original's outcome law and the tracked circuit's frame-corrected one, over
    every clbit; with `wire`, over every input on it too (the six Pauli preparations of contract_synth.faithful)"""
    from oneq.hw import contract_synth as CS
    from oneq.hw import dynamic_exact as DX
    preps = CS._pauli_preps() if wire is not None else [None]
    worst = 0.0
    for prep in preps:
        pair = []
        for c in (original, tracked):
            if prep is None:
                pair.append(c)
                continue
            new = c.copy_empty_like()
            for inst in prep.data:
                new.append(inst.operation, [new.qubits[wire]])
            for inst in c.data:
                new.append(inst)
            pair.append(new)
        used = sorted(set(CS.active(pair[0])) | set(CS.active(pair[1])))
        pa = DX.distribution(CS.compact(pair[0], used)[0]).probs
        raw = DX.distribution(CS.compact(pair[1], used)[0]).probs
        pb: dict = {}
        for k, p in raw.items():
            kk = apply(k, frame)
            pb[kk] = pb.get(kk, 0.0) + p
        keys = set(pa) | set(pb)
        worst = max(worst, max(abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys))
    return worst


__all__ = ["track", "apply", "corrected_counts", "check", "Frame", "FrameRefused"]
