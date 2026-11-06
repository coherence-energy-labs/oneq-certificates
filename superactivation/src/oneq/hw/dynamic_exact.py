"""Exact outcome distributions of DYNAMIC circuits -- mid-circuit measurement, reset, and classically controlled blocks
(if_test / if_else) -- by branching the state at every measurement instead of sampling it.

Why it exists. The R147 dynamic circuits (real mid-circuit reads and feed-forward,
experiments/superactivation/shield_family/hw_dynamic.py) were checked only against qiskit-aer, which the suite's
interpreter does not have, so the four tests that check them SKIPPED -- four dark gates on exactly the circuits the
IBM model test would run (found 2026-10-16, when the integration suite reported 4 skips against README's "0 skips").
A sampled check at 400-4000 shots could only bound a mean; this one computes it: every outcome's probability is a
sum of |amplitude|^2 over the branches that produce it, exact to rounding.

The model. A branch is (weight, normalised statevector, classical bits). A measurement splits each branch in two by
the Born rule and writes the bit; a reset splits it the same way and returns the qubit to |0> without writing
anything (so the mixture over branches IS the reset channel); a gate acts on every branch; an if_else evaluates its
condition per branch and runs the chosen body on that branch alone. Branches of probability below 1e-24 are dropped
(a deterministic outcome's complement is ~1e-33 of rounding, a real outcome is far above it); the total dropped is
reported and the distribution is refused if it does not sum to 1 within 1e-12.

Supported: every unitary gate qiskit can turn into an Operator; measure; reset; barrier and delay (no-ops); if_else
whose condition is a (Clbit | ClassicalRegister, value) pair or a classical expression built from Var (a clbit or a
register), Value, Cast, bit_not / logic_not and bit_and / bit_or / bit_xor / logic_and / logic_or / equal /
not_equal / less / less_equal / greater / greater_equal. Anything else -- while / for / switch / box, a simulator's
save instruction, a standalone classical variable, an expression node outside that list -- is REFUSED, never
approximated. Independent check: tests/test_hw_dynamic_exact.py compares it with qiskit's own unitary Statevector on
the deferred-measurement equivalent of random dynamic circuits."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PRUNE = 1e-24
NORM_TOL = 1e-12
_NOOPS = {"barrier", "delay"}


class DynamicRefusal(ValueError):
    """a circuit element this simulator does not model exactly"""


@dataclass(frozen=True)
class Distribution:
    probs: dict          # tuple of clbit values, in the circuit's clbit order -> probability
    dropped: float       # total weight of branches pruned below PRUNE
    branches: int        # branches alive at the end (before merging equal classical outcomes)

    def expectation(self, score, nbits: int | None = None) -> float:
        """sum over outcomes of p(bits) score(bits[:nbits])"""
        return math.fsum(p * score(k if nbits is None else k[:nbits]) for k, p in self.probs.items())


def _qiskit():
    from qiskit.circuit import ClassicalRegister, Clbit
    from qiskit.circuit.classical import expr, types
    from qiskit.quantum_info import Operator
    return ClassicalRegister, Clbit, expr, types, Operator


# ---------------------------------------------------------------- classical conditions

def _bit_index(circ, cidx, bit) -> int:
    return cidx[circ.find_bit(bit).index]


def _register_value(circ, cidx, bits, reg) -> int:
    return sum(bits[_bit_index(circ, cidx, b)] << i for i, b in enumerate(reg))


def _width(t) -> int | None:
    """None for Bool, the width for Uint; refuses any other classical type"""
    _, _, _, types, _ = _qiskit()
    if isinstance(t, types.Bool):
        return None
    if isinstance(t, types.Uint):
        return int(t.width)
    raise DynamicRefusal(f"classical type {t} is not modelled")


def _eval(node, circ, cidx, bits) -> int:
    ClassicalRegister, Clbit, expr, _, _ = _qiskit()
    if isinstance(node, expr.Var):
        v = node.var
        if isinstance(v, Clbit):
            return bits[_bit_index(circ, cidx, v)]
        if isinstance(v, ClassicalRegister):
            return _register_value(circ, cidx, bits, v)
        raise DynamicRefusal("a standalone classical variable is not modelled")
    if isinstance(node, expr.Value):
        return int(node.value)
    if isinstance(node, expr.Cast):
        x = _eval(node.operand, circ, cidx, bits)
        w = _width(node.type)
        return int(x != 0) if w is None else x & ((1 << w) - 1)
    if isinstance(node, expr.Unary):
        x = _eval(node.operand, circ, cidx, bits)
        if node.op is expr.Unary.Op.LOGIC_NOT:
            return int(not x)
        if node.op is expr.Unary.Op.BIT_NOT:
            w = _width(node.type)
            return 1 - x if w is None else ~x & ((1 << w) - 1)
        raise DynamicRefusal(f"unary {node.op} is not modelled")
    if isinstance(node, expr.Binary):
        a = _eval(node.left, circ, cidx, bits)
        b = _eval(node.right, circ, cidx, bits)
        Op = expr.Binary.Op
        table = {Op.BIT_AND: lambda: a & b, Op.BIT_OR: lambda: a | b, Op.BIT_XOR: lambda: a ^ b,
                 Op.LOGIC_AND: lambda: int(bool(a) and bool(b)), Op.LOGIC_OR: lambda: int(bool(a) or bool(b)),
                 Op.EQUAL: lambda: int(a == b), Op.NOT_EQUAL: lambda: int(a != b), Op.LESS: lambda: int(a < b),
                 Op.LESS_EQUAL: lambda: int(a <= b), Op.GREATER: lambda: int(a > b),
                 Op.GREATER_EQUAL: lambda: int(a >= b)}
        if node.op not in table:
            raise DynamicRefusal(f"binary {node.op} is not modelled")
        return table[node.op]()
    raise DynamicRefusal(f"expression node {type(node).__name__} is not modelled")


def _condition(cond, circ, cidx, bits) -> bool:
    ClassicalRegister, Clbit, expr, _, _ = _qiskit()
    if isinstance(cond, tuple):
        target, value = cond
        if isinstance(target, Clbit):
            return bits[_bit_index(circ, cidx, target)] == int(value)
        if isinstance(target, ClassicalRegister):
            return _register_value(circ, cidx, bits, target) == int(value)
        raise DynamicRefusal(f"condition target {target!r} is not modelled")
    if isinstance(cond, expr.Expr):
        return bool(_eval(cond, circ, cidx, bits))
    raise DynamicRefusal(f"condition {cond!r} is not modelled")


# ---------------------------------------------------------------- the state

class _Run:
    def __init__(self, n: int):
        self.n = n
        idx = np.arange(1 << n)
        self.bit = [(idx >> q) & 1 for q in range(n)]          # qiskit order: bit q of the index is qubit q
        self.dropped = 0.0
        self.ops: dict = {}

    def gate(self, op, psi, qargs):
        from qiskit.quantum_info import Statevector
        key = id(op)
        if key not in self.ops:
            _, _, _, _, Operator = _qiskit()
            try:
                self.ops[key] = (Operator(op), op)          # keep op alive: its id stays unique for the run
            except Exception as exc:                        # qiskit raises several types for a non-unitary op
                raise DynamicRefusal(f"{op.name} is not a unitary qiskit can represent") from exc
        return Statevector(psi, dims=(2,) * self.n).evolve(self.ops[key][0], qargs).data

    def split(self, w, psi, q):
        """the Born-rule children (value, weight, projected-and-renormalised state) of measuring qubit q"""
        one = self.bit[q] == 1
        p1 = float(np.sum(np.abs(psi[one]) ** 2))
        out = []
        for v, pv in ((0, 1.0 - p1), (1, p1)):
            if pv < PRUNE:
                self.dropped += w * max(pv, 0.0)
                continue
            phi = np.where(one if v else ~one, psi, 0)
            out.append((v, w * pv, phi / math.sqrt(pv)))
        return out

    def flip(self, psi, q):
        return psi[np.arange(1 << self.n) ^ (1 << q)]

    def run(self, circ, branches, qidx, cidx):
        for inst in circ.data:
            op = inst.operation
            name = op.name
            qargs = [qidx[circ.find_bit(x).index] for x in inst.qubits]
            cargs = [cidx[circ.find_bit(x).index] for x in inst.clbits]
            if name in _NOOPS:
                continue
            if name == "measure":
                q, c = qargs[0], cargs[0]
                nxt = []
                for w, psi, bits in branches:
                    for v, wv, phi in self.split(w, psi, q):
                        nb = list(bits)
                        nb[c] = v
                        nxt.append((wv, phi, tuple(nb)))
                branches = nxt
            elif name == "reset":
                for q in qargs:
                    nxt = []
                    for w, psi, bits in branches:
                        for v, wv, phi in self.split(w, psi, q):
                            nxt.append((wv, self.flip(phi, q) if v else phi, bits))
                    branches = nxt
            elif name == "if_else":
                blocks = op.blocks
                taken, other = [], []
                for br in branches:
                    (taken if _condition(op.condition, circ, cidx, br[2]) else other).append(br)
                sub_q = qargs
                sub_c = cargs
                taken = self.run(blocks[0], taken, sub_q, sub_c) if taken else []
                if len(blocks) > 1 and blocks[1] is not None and other:
                    other = self.run(blocks[1], other, sub_q, sub_c)
                branches = taken + other
            elif name in ("while_loop", "for_loop", "switch_case", "box") or name.startswith("save_"):
                raise DynamicRefusal(f"{name} is not modelled")
            elif inst.clbits:
                raise DynamicRefusal(f"{name} touches classical bits and is not modelled")
            else:
                branches = [(w, self.gate(op, psi, qargs), bits) for w, psi, bits in branches]
        return branches


def distribution(qc) -> Distribution:
    """the exact distribution of qc's classical bits (tuple in qc.clbits order) over every branch"""
    if qc.num_qubits > 20:
        raise DynamicRefusal(f"{qc.num_qubits} qubits: a branching statevector beyond 20 qubits is not attempted")
    r = _Run(qc.num_qubits)
    psi0 = np.zeros(1 << qc.num_qubits, dtype=complex)
    psi0[0] = 1.0
    branches = r.run(qc, [(1.0, psi0, (0,) * qc.num_clbits)], list(range(qc.num_qubits)),
                     list(range(qc.num_clbits)))
    probs: dict = {}
    for w, _psi, bits in branches:
        probs[bits] = probs.get(bits, 0.0) + w
    total = math.fsum(probs.values()) + r.dropped
    if abs(total - 1.0) > NORM_TOL:
        raise AssertionError(f"branch weights sum to {total!r}: the simulation lost probability")
    return Distribution(probs=probs, dropped=r.dropped, branches=len(branches))


__all__ = ["distribution", "Distribution", "DynamicRefusal", "PRUNE"]
