"""R186: independent shot-free statevector-branch interpreter for Bob's quantum decoder.

Supported native Qiskit operations: unitary 1q/2q gates, 1q projective
computational-basis measurement, reset, and nested conditional single-qubit
or two-qubit gates. NO qubit noise, timing, provider data, or sampling.
Branches are projected WITHOUT normalization and squared norms summed.
Reset discards its hidden outcome into unrecorded branches.

Independence: imports neither the R186 circuit producer nor the historic
PPT/erasure algebra. Conditions on unmeasured classical bits REFUSE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import Statevector


class DynamicRefused(ValueError):
    pass


@dataclass(frozen=True)
class Branch:
    psi: np.ndarray
    bits: tuple
    measured: tuple


def _split(state: Branch, wire: int, classical=None, *, reset=False):
    """Branch by projectors |0><0| and |1><1|, leaving amplitudes unnormalized."""
    indices = np.arange(len(state.psi), dtype=np.int64)
    outputs = []
    for value in (0, 1):
        psi = state.psi.copy()
        psi[((indices >> wire) & 1) != value] = 0
        if np.vdot(psi, psi).real < 1e-17:
            continue
        bits = list(state.bits)
        measured = list(state.measured)
        if classical is not None:
            bits[classical] = value
            measured[classical] = True
        if reset and value:
            from qiskit.circuit.library import XGate

            psi = Statevector(psi).evolve(XGate(), qargs=[wire]).data
        outputs.append(Branch(np.asarray(psi), tuple(bits), tuple(measured)))
    return outputs


def _branch_apply(op, branch: Branch, qargs):
    if len(qargs) > 2:
        raise DynamicRefused("only one- and two-qubit gates supported")
    if not hasattr(op, "to_matrix") or getattr(op, "blocks", None):
        raise DynamicRefused("unknown operation or nested control flow")
    return Branch(
        Statevector(branch.psi).evolve(op, qargs=qargs).data,
        branch.bits,
        branch.measured,
    )


def _evolve_seq(seq, branches, qmap, cmap, *, max_branches=10000):
    for inst in seq.data:
        op = inst.operation
        name = op.name
        qargs = [qmap[seq.find_bit(q).index] for q in inst.qubits]
        cargs = [cmap[seq.find_bit(c).index] for c in inst.clbits]
        if name in ("barrier", "delay"):
            continue
        if name == "measure":
            if len(qargs) != 1 or len(cargs) != 1:
                raise DynamicRefused("measurement source is not a single qubit")
            branches = [b2 for b in branches for b2 in _split(b, qargs[0], cargs[0])]
        elif name == "reset":
            if len(qargs) != 1:
                raise DynamicRefused("only one-qubit reset supported")
            branches = [b2 for b in branches for b2 in _split(b, qargs[0], reset=True)]
        elif name == "if_else":
            cond = op.condition
            if not isinstance(cond, tuple) or len(cond) != 2 or cond[1] not in (0, 1):
                raise DynamicRefused("condition must be a measured bit equals 0 or 1")
            if len(op.blocks) != 1:
                raise DynamicRefused("conditional with else body unsupported")
            target = cmap[seq.find_bit(cond[0]).index]
            body_qmap = qargs
            body_cmap = cargs
            updates = []
            for b in branches:
                if not b.measured[target]:
                    raise DynamicRefused("conditioned on a bit never actually measured")
                if b.bits[target] != cond[1]:
                    updates.append(b)
                else:
                    updates.extend(_evolve_seq(op.blocks[0], [b], body_qmap, body_cmap))
            branches = updates
        elif cargs:
            raise DynamicRefused(
                "unexpected classical feedback outside known condition"
            )
        else:
            branches = [_branch_apply(op, b, qargs) for b in branches]
        if len(branches) > max_branches:
            raise DynamicRefused("unbounded branching: unexpected hidden measurement")
    return branches


def run(circuit, *, max_qubits=13):
    """Exact distribution on all classical bits after a dynamic quantum circuit."""
    n = circuit.num_qubits
    if n > max_qubits or circuit.num_clbits > 15:
        raise DynamicRefused("bounded exact experiment: refuse exponential overflow")
    first = np.zeros(2**n, dtype=complex)
    first[0] = 1
    init = Branch(first, (0,) * circuit.num_clbits, (False,) * circuit.num_clbits)
    branches = _evolve_seq(
        circuit,
        [init],
        tuple(range(n)),
        tuple(range(circuit.num_clbits)),
    )
    results = {}
    for b in branches:
        prob = float(np.vdot(b.psi, b.psi).real)
        results[b.bits] = results.get(b.bits, 0) + prob
    if abs(sum(results.values()) - 1) > 1e-9:
        raise DynamicRefused("non-unit trace: discarded or duplicated quantum branches")
    return results


def success_probability(outcomes, m):
    total = 0.0
    for bits, p in outcomes.items():
        if len(bits) not in (m + 4, 2 * m + 4):
            raise DynamicRefused("Bell measurement has incorrect register width")
        if bits[-2] == 0 and bits[-1] == 0:
            total += p
    return total
