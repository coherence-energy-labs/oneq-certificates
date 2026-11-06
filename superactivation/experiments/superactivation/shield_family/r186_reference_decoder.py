"""R186 | entangled-reference operational decoder for the real two-use P_d x erasure channel.

Unlike a six-cardinal-state *observable estimator*, this measures an actual
output reference entanglement witness after Bob's quantum-output decoder.
A successful Bell projection occurs AFTER conditional quantum corrections
on Bob's key, not by reconstructing a never-physically-created state in
postprocessing. This is still only a finite-blocklength test, NOT proof
of any channel's asymptotic quantum capacity or new IBM hardware evidence.

Key physical contract:
  * Alice's logical input qubit starts Bell-entangled to untouched R.
  * She encodes into |++>, |--> across two key wires.
  * Both P_d channel uses execute their independently sampled native
    random-unitary choices, **before** Bob learns any measurement bit.
  * A's erasure outputs (copy, shield ancilla) are physically RESET.
  * Bob observes only P/A output registers and the classical erasure flags.
  * On retained A0, Bob measures the copy l, selects shield-pair decoding
    BASIS beta=l, corrects his surviving quantum key *on hardware*, then
    a fixed Bell measurement of (R,key0) certifies entanglement fidelity.
  * Erased-erased is an explicit measure-and-reprepare quantum instrument,
    not an invented coherent successful branch.
  * NO P-d unitary selector label is available to Bob. Choices only
    determine which source channel randomizer ran on that shot.

This is a *circuit construction*, validated on m=1/m=2 against the exact
trace-over-environment fidelity of the incumbent coherent channel decoder.
The operation is materially different from a Pauli shadow's scalar estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import hashlib
import json

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


BRANCHES = ("kept_kept", "kept_erased", "erased_erased")


class ReferenceRefused(ValueError):
    """The builder cannot represent the exact requested quantum channel."""


@dataclass(frozen=True)
class ReferenceUnit:
    m: int
    branch: str
    choices: tuple

    def verify(self):
        if type(self.m) is not int or not 1 <= self.m <= 7:
            raise ReferenceRefused("m must be a small bounded positive shield size")
        if self.branch not in BRANCHES:
            raise ReferenceRefused("unknown channel erasure branch")
        if not isinstance(self.choices, tuple) or len(self.choices) != 2:
            raise ReferenceRefused("both physical PPT channel uses are required")
        for sector, labels in self.choices:
            if sector == "s0":
                if (
                    not isinstance(labels, tuple)
                    or len(labels) != self.m
                    or any(type(v) is not str or v not in "IXYZ" for v in labels)
                ):
                    raise ReferenceRefused("invalid s0 random unitary choice")
            elif sector == "s1":
                if (
                    not isinstance(labels, tuple)
                    or len(labels) != self.m
                    or any(type(v) is not int or v not in (0, 1) for v in labels)
                ):
                    raise ReferenceRefused("invalid s1 random unitary choice")
            else:
                raise ReferenceRefused("not a P_d random-unitary sector")
        return True


def reference_circuit(
    unit: ReferenceUnit, *, decoder: str = "cz2", erasure: str = "reset"
) -> QuantumCircuit:
    """A true quantum-state-output decoder with final R/output Bell verification.

    cz2: incumbent CNOT+CH plus adaptive Ry on Bob's pair ancilla.
    cz1: one-CZ adaptive spectral measurement of the same O_l, with
    two readouts for beta0 and one effective readout for beta1.
    This is an OPERATIONAL physical decoder, not a Pauli shadow.
    """
    unit.verify()
    if decoder not in ("cz2", "cz1"):
        raise ReferenceRefused("unknown adaptive decoder circuit")
    if erasure not in ("reset", "none"):
        raise ReferenceRefused("unknown erasure semantics")
    from hw_two_use import _apply_p

    m = unit.m
    n = 5 + 4 * m
    Q = QuantumRegister(n, "q")
    b1_index = m + 1 if decoder == "cz2" else 2 * m + 1
    C = ClassicalRegister(b1_index + 3, "c")
    qc = QuantumCircuit(Q, C, name=f"R186_{unit.branch}_{decoder}_m{m}")

    def key(u):
        return 1 + u

    def copy(u):
        return 3 + u

    def shield(u, t):
        return 5 + 2 * u * m + t

    def ancilla(u, t):
        return 5 + (2 * u + 1) * m + t

    # Untouched R=0, logical input K0=1, K1=2.
    qc.h(0)
    qc.cx(0, key(0))
    qc.cx(key(0), key(1))
    qc.h(key(0))
    qc.h(key(1))
    # The full channel is two independent P_d uses, and both A inputs.
    for u in range(2):
        for t in range(m):
            qc.h(ancilla(u, t))
            qc.cx(ancilla(u, t), shield(u, t))
        qc.cx(key(u), copy(u))
    for u in range(2):
        _apply_p(
            qc, unit.choices[u], key=key(u), shields=[shield(u, t) for t in range(m)]
        )
    erased = {
        "kept_kept": (False, False),
        "kept_erased": (False, True),
        "erased_erased": (True, True),
    }[unit.branch]
    if erasure == "reset":
        for u in range(2):
            if erased[u]:
                qc.reset([copy(u)] + [ancilla(u, t) for t in range(m)])

    if unit.branch == "erased_erased":
        # The best documented classical fallback: Bob measures both P keys
        # and prepares |+> or |-> conditioned on their parity. It creates no
        # entanglement with R, even if Bell success is nonzero.
        qc.measure(key(0), C[0])
        qc.measure(key(1), C[1])
        qc.reset(key(0))
        qc.h(key(0))
        with qc.if_test((C[0], 1)):
            qc.z(key(0))
        with qc.if_test((C[1], 1)):
            qc.z(key(0))
    else:
        # Physical Bob action. Measure A0 copy, adapt the actual pair
        # measurement unitary to l (the original R145 Monte-Carlo estimate
        # retained beta=l only by CLASSICAL IMPORTANCE SELECTION).
        qc.cx(key(0), copy(0))
        qc.measure(copy(0), C[0])
        for t in range(m):
            s, a = shield(0, t), ancilla(0, t)
            if decoder == "cz2":
                qc.cx(a, s)
                qc.ch(s, a)
                with qc.if_test((C[0], 1)):
                    qc.ry(-math.pi / 4, a)
                qc.measure(a, C[1 + t])
            else:
                with qc.if_test((C[0], 0)):
                    qc.ry(3 * math.pi / 2, a)
                    qc.rz(math.pi, s)
                    qc.ry(3 * math.pi / 2, s)
                    qc.cz(a, s)
                    qc.ry(7 * math.pi / 4, a)
                    qc.ry(3 * math.pi / 4, s)
                with qc.if_test((C[0], 1)):
                    qc.rz(math.pi, a)
                    qc.ry(3 * math.pi / 2, s)
                    qc.rz(math.pi, s)
                    qc.cz(a, s)
                    qc.ry(math.pi / 4, a)
                qc.measure(a, C[1 + t])
                qc.measure(s, C[1 + m + t])
        read_b1 = copy(1) if unit.branch == "kept_kept" else key(1)
        qc.measure(read_b1, C[b1_index])
        # Correct the quantum OUTCOME, not merely the score:
        # U = H X^{b1} X^{ell} Z^{par}.
        for t in range(m):
            with qc.if_test((C[1 + t], 1)):
                qc.z(key(0))
            if decoder == "cz1":
                with qc.if_test((C[0], 0)):
                    with qc.if_test((C[1 + m + t], 1)):
                        qc.z(key(0))
        with qc.if_test((C[0], 1)):
            qc.x(key(0))
        with qc.if_test((C[b1_index], 1)):
            qc.x(key(0))
        qc.h(key(0))

    # Independent Bell projection onto Phi+ in the reference/output pair,
    # F_e = Pr(00) after this circuit. Ancillas are never part of acceptance.
    qc.cx(0, key(0))
    qc.h(0)
    qc.measure(0, C[b1_index + 1])
    qc.measure(key(0), C[b1_index + 2])
    if sum(i.operation.name == "measure" for i in qc.data) < 4:
        raise ReferenceRefused("not an operational quantum-output probe")
    return qc


def entanglement_success(counts: dict[str, int], m: int) -> tuple[float, int]:
    """A Bell-00 win on both FINAL bits, with no selection on midcircuit outcomes."""
    if type(m) is not int or m < 1:
        raise ReferenceRefused("invalid shield size")
    total, wins = 0, 0
    for word, n in counts.items():
        if (
            type(word) is not str
            or any(c not in "01 " for c in word)
            or type(n) is not int
            or n < 0
        ):
            raise ReferenceRefused("invalid result histogram")
        bitstr = word.replace(" ", "")
        if len(bitstr) not in (m + 4, 2 * m + 4):
            raise ReferenceRefused("wrong classical readout width")
        total += n
        # Qiskit renders the highest clbit first: Bell bits are the LEFTMOST.
        if bitstr[:2] == "00":
            wins += n
    if total <= 0:
        raise ReferenceRefused("no shots; missing Bell witness")
    return wins / total, total


def source_channel_digest(unit: ReferenceUnit):
    """Digest *both* independently sampled P_d channel gate streams.

    No dynamic output corrector may depend on these labels.
    """
    unit.verify()
    from hw_two_use import _apply_p

    names = []
    for sector, labels in unit.choices:
        qc = QuantumCircuit(unit.m + 1)
        _apply_p(qc, (sector, labels), key=0, shields=[t + 1 for t in range(unit.m)])
        names.append(
            [
                [
                    inst.operation.name,
                    [qc.find_bit(q).index for q in inst.qubits],
                    [str(p) for p in inst.operation.params],
                ]
                for inst in qc.data
            ]
        )
    return hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()
