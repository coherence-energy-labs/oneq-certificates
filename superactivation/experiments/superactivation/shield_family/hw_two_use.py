"""The two-use superactivation protocol as HARDWARE circuits with a per-shot score -- the first client of oneq.hw.

Built on lean_protocol.py (every erasure branch needs at most one quantum decode, each compilation an exact
identity). Three changes make it a hardware experiment, each checked exactly below against the closed form
F_e = (7 - 6p + 2p^2)/8:
  1. NO REFERENCE QUBIT. The logical input is one of the 6 cardinal states, prepared on key0 before the encoder
     (|0_L> = |++>, |1_L> = |-->). The 6 states form a 2-design, so their mean success F_avg is the average fidelity
     and F_e = (3 F_avg - 1)/2: the bar F_e > 3/4 is F_avg > 5/6. This drops a qubit that would idle through the
     whole circuit, and one CNOT.
  2. BOB'S FRAME IS CLASSICAL. His correction U = H X^{b1} X^l Z^{par} is a Hadamard (fixed) times Paulis that
     depend on measured bits, so key0 is measured in the fixed basis H sigma H (X <-> Z, Y -> -Y) and the Paulis
     become sign flips of the outcome. No feed-forward.
  3. SELECTION REPLACES THE ONE FEED-FORWARD LEFT. Bob's shield-eigenbasis choice must equal the copy bit l; a
     uniformly random beta is used and only shots with l == beta count, weighted 2: E[2 1[l = beta] 1[correct]]
     is exactly the decoded fidelity, for any device, because beta is independent of everything else.
The per-shot score is therefore in [0, 2] (kept branches) or {0, 1/2, 1} (both erased: Bob's output is a classical
|+-> whose success on a Z or Y input is exactly 1/2, scored by its expectation).

Qubits: key0 = 0, key1 = 1, copy0 = 2, copy1 = 3, s[t] = 4 + t, a[t] = 4 + m + t.
Classical bits (kept branches): 0 = l (copy0), 1..m = a[t], m+1 = b1 (copy1 if kept, key1 if erased), m+2 = key0.
Classical bits (both erased): 0 = key0, 1 = key1.
Branches and their weights in F_avg: kept_kept 1/4, kept_erased 1/2 (one erased; the two orders are symmetric),
erased_erased 1/4.

  python hw_two_use.py check        exact ideal F_e from the hardware circuits and score, m = 1, 2
"""
from __future__ import annotations

import itertools
import math
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hardware_design as hd  # noqa: E402

BRANCH_WEIGHT = {"kept_kept": 0.25, "kept_erased": 0.5, "erased_erased": 0.25}
INPUTS = tuple((b, e) for b in "ZXY" for e in (+1, -1))       # (basis, eigenvalue): 6 cardinal states
BOB_BASIS = {"X": ("Z", +1), "Z": ("X", +1), "Y": ("Y", -1)}  # H sigma H = sign * sigma'


def p_flip(m: int) -> float:
    return 1 / (math.sqrt(2 ** m) + 1)


def f_e_closed(m: int) -> float:
    p = p_flip(m)
    return (7 - 6 * p + 2 * p * p) / 8


@dataclass(frozen=True)
class Unit:
    """one circuit of the design: everything that changes the gates"""
    branch: str
    choice: tuple | None            # P's unitary on the decoded use (hardware_design.sample_choice); None if both erased
    beta: int                       # Bob's shield-basis selector (kept branches)
    inp: tuple                      # (basis, eigenvalue) of the logical input
    flips: tuple                    # sector-1 key flips of the erased uses (use 0, use 1)

    @property
    def name(self) -> str:
        ch = "none" if self.choice is None else f"{self.choice[0]}_{''.join(map(str, self.choice[1]))}"
        return (f"{self.branch}__{ch}__b{self.beta}__{self.inp[0]}{'p' if self.inp[1] > 0 else 'm'}"
                f"__f{self.flips[0]}{self.flips[1]}")


def _prep(qc, q, inp):
    basis, e = inp
    if e < 0:
        qc.x(q)
    if basis in "XY":
        qc.h(q)
    if basis == "Y":
        qc.s(q)


def _measure_in(qc, q, basis):
    if basis == "X":
        qc.h(q)
    elif basis == "Y":
        qc.sdg(q)
        qc.h(q)


@dataclass(frozen=True)
class Compile:
    """exact compilation variants (each checked against the closed form in the tests):
      fanout  P's CZ(key, s_t) for odd t comes from copy0 instead of key0. copy0 holds key0's Z value from the first
              key-copy CNOT until P's final X on the key, so CZ(copy0, s_t) acts on the joint state exactly as
              CZ(key0, s_t) does -- and the star around the key, which heavy-hex can only route with SWAPs, halves.
      fanin   Bob's frame needs only the PARITY of the m shield-pair bits: a CNOT chain a[0] -> a[1] -> ... -> a[m-1]
              computes it, and only a[m-1] is read. 9 measured bits become 4 (m = 5)."""
    fanout: bool = False
    fanin: bool = False

    @property
    def name(self) -> str:
        return ("fanout" if self.fanout else "") + ("+" if self.fanout and self.fanin else "") + \
            ("fanin" if self.fanin else "") or "baseline"


BASELINE = Compile()


def circuit(m: int, u: Unit, cmp: Compile = BASELINE):
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    n = 4 + 2 * m
    s = [4 + t for t in range(m)]
    a = [4 + m + t for t in range(m)]
    q = QuantumRegister(n, "q")
    if u.branch == "erased_erased":
        c = ClassicalRegister(2, "c")
        qc = QuantumCircuit(q, c, name=u.name)
        _prep(qc, 0, u.inp)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)
        for k, f in ((0, u.flips[0]), (1, u.flips[1])):
            if f:
                qc.x(k)
        qc.measure([0, 1], [0, 1])
        return qc
    nb = 4 if cmp.fanin else m + 3
    c = ClassicalRegister(nb, "c")
    qc = QuantumCircuit(q, c, name=u.name)
    _prep(qc, 0, u.inp)
    qc.cx(0, 1)
    qc.h(0)
    qc.h(1)
    for t in range(m):
        qc.h(a[t])
        qc.cx(a[t], s[t])
    qc.cx(0, 2)
    sources = [2 if (cmp.fanout and t % 2) else 0 for t in range(m)]
    _apply_p(qc, u.choice, key=0, shields=s, sources=sources)
    if u.branch == "kept_kept":
        qc.cx(1, 3)
    elif u.flips[1]:
        qc.x(1)
    qc.cx(0, 2)
    for t in range(m):
        qc.cx(a[t], s[t])
        qc.ch(s[t], a[t])
        if u.beta:
            qc.ry(-math.pi / 4, a[t])
    _measure_in(qc, 0, BOB_BASIS[u.inp[0]][0])
    other = 3 if u.branch == "kept_kept" else 1
    if cmp.fanin:
        for t in range(m - 1):
            qc.cx(a[t], a[t + 1])                        # a[m-1] <- parity of all m pair bits
        qc.measure([2, a[m - 1], other, 0], [0, 1, 2, 3])
    else:
        qc.measure([2] + a + [other, 0], list(range(m + 3)))
    return qc


def _apply_p(qc, choice, key, shields, sources=None):
    """P_d's sampled unitary on (key, shields): hardware_design.p_unitaries, verified there against the Choi state.
    sources[t] is the qubit CZ(., s_t) is applied from: the key, or a Z-copy of it (Compile.fanout)."""
    sources = sources or [key] * len(shields)
    kind, lab = choice
    if kind == "s0":
        for t, g in enumerate(lab):
            if g != "I":
                getattr(qc, g.lower())(shields[t])
        if lab.count("Y") % 2:
            qc.z(key)
        for t, g in enumerate(lab):
            if g in "IZ":
                qc.cz(sources[t], shields[t])
    else:
        phase = 0.0
        for t, ct in enumerate(lab):
            (qc.s if ct == 0 else qc.sdg)(shields[t])
            phase += math.pi / 4 if ct == 0 else -math.pi / 4
            qc.cz(sources[t], shields[t])
        qc.p(phase, key)
        qc.x(key)


def score(m: int, u: Unit, cmp: Compile = BASELINE):
    """score(bits) in CLASSICAL-BIT order, the SAME function for the simulated prediction and the real counts"""
    basis, e = u.inp
    if u.branch == "erased_erased":
        def ee(bits):
            if basis != "X":
                return 0.5                                   # Bob's classical |+-> on a Z or Y input
            return 1.0 if (1 - 2 * (bits[0] ^ bits[1])) == e else 0.0
        return ee
    meas_basis, sign = BOB_BASIS[basis]
    e_prime = sign * e
    z_flips = meas_basis in "XY"                             # Z^{par} anticommutes with X and Y
    x_flips = meas_basis in "ZY"                             # X^{l + b1} anticommutes with Z and Y

    def kept(bits):
        if cmp.fanin:
            ell, par, b1, o = bits[0], bits[1], bits[2], bits[3]
        else:
            ell, par, b1, o = bits[0], sum(bits[1:1 + m]) % 2, bits[m + 1], bits[m + 2]
        if ell != u.beta:
            return 0.0
        want = e_prime * (-1) ** ((par if z_flips else 0) + ((ell ^ b1) if x_flips else 0))
        return 2.0 if (1 - 2 * o) == want else 0.0
    return kept


def p_labels(m: int):
    """P_d's unitary labels with their weights -- WITHOUT the matrices. hardware_design.p_unitaries builds every
    2^(m+1)-square matrix (17 GB at m = 7, killed by the memory cap) when only the labels are needed here."""
    d = 2 ** m
    p = p_flip(m)
    for g in itertools.product("IXYZ", repeat=m):
        yield (1 - p) / d ** 2, ("s0", g)
    for c in itertools.product((0, 1), repeat=m):
        yield p / d, ("s1", c)


def erased_erased_units(m: int):
    """the 24 both-erased units (6 inputs x 4 flip patterns) with their weights in F_avg"""
    p = p_flip(m)
    for inp in INPUTS:
        for f0, f1 in itertools.product((0, 1), repeat=2):
            pf = (p if f0 else 1 - p) * (p if f1 else 1 - p)
            yield BRANCH_WEIGHT["erased_erased"] * pf / 6, Unit("erased_erased", None, 0, inp, (f0, f1))


def design_units(m: int):
    """EVERY unit of the exact design with its weight in F_avg (sum of weights = 1)"""
    p = p_flip(m)
    for inp in INPUTS:
        for w, ch in p_labels(m):
            for beta in (0, 1):
                yield BRANCH_WEIGHT["kept_kept"] * w / 6 / 2, Unit("kept_kept", ch, beta, inp, (0, 0))
                for f1, pf in ((0, 1 - p), (1, p)):
                    yield BRANCH_WEIGHT["kept_erased"] * w * pf / 6 / 2, Unit("kept_erased", ch, beta, inp, (0, f1))
    yield from erased_erased_units(m)


def sample_unit(m: int, branch: str, rng: np.random.Generator) -> Unit:
    """one unit drawn from the design law WITHIN a branch (the branch itself is stratified by the caller)"""
    p = p_flip(m)
    inp = INPUTS[int(rng.integers(6))]
    if branch == "erased_erased":
        return Unit(branch, None, 0, inp, (int(rng.random() < p), int(rng.random() < p)))
    flips = (0, int(rng.random() < p)) if branch == "kept_erased" else (0, 0)
    return Unit(branch, hd.sample_choice(m, rng), int(rng.integers(2)), inp, flips)


def strata(m: int) -> list:
    """the kept branches split by (sector of P's unitary, beta, the erased use's flip), with exact weights in F_avg.
    Bob's copy bit l equals the sector in the ideal circuit, so beta either matches it (score 0 or 2) or never does
    (score 0): stratifying on it separates units whose losses differ by an order of magnitude."""
    p = p_flip(m)
    out = []
    for branch in ("kept_kept", "kept_erased"):
        for sector, ps in (("s0", 1 - p), ("s1", p)):
            for beta in (0, 1):
                for f1, pf in (((0, 1.0),) if branch == "kept_kept" else ((0, 1 - p), (1, p))):
                    out.append(((branch, sector, beta, f1), BRANCH_WEIGHT[branch] * ps * 0.5 * pf))
    return out


def sample_in_stratum(m: int, key: tuple, rng: np.random.Generator) -> Unit:
    branch, sector, beta, f1 = key
    lab = (tuple("IXYZ"[int(x)] for x in rng.integers(4, size=m)) if sector == "s0"
           else tuple(int(x) for x in rng.integers(2, size=m)))
    return Unit(branch, (sector, lab), beta, INPUTS[int(rng.integers(6))], (0, f1))


def ideal_expectation(m: int, u: Unit, cmp: Compile = BASELINE) -> float:
    """exact noiseless E[score] from the statevector of the logical circuit (independent of oneq.hw.engine)"""
    from qiskit.quantum_info import Statevector
    qc = circuit(m, u, cmp)
    nb = qc.num_clbits
    bare = qc.remove_final_measurements(inplace=False)
    probs = Statevector(bare).probabilities()
    measured = [qc.find_bit(inst.qubits[0]).index for inst in qc.data if inst.operation.name == "measure"]
    clbits = [qc.find_bit(inst.clbits[0]).index for inst in qc.data if inst.operation.name == "measure"]
    qubit_of = dict(zip(clbits, measured))
    sc = score(m, u, cmp)
    total = 0.0
    for idx, pr in enumerate(probs):
        if pr < 1e-15:
            continue
        bits = tuple((idx >> qubit_of[j]) & 1 for j in range(nb))
        total += pr * sc(bits)
    return total


def ideal_f_e(m: int, cmp: Compile = BASELINE) -> tuple[float, dict]:
    """F_e of the hardware design by exact enumeration of every unit (m <= 2: 72 P-unitaries x ...)"""
    per = dict.fromkeys(BRANCH_WEIGHT, 0.0)
    for w, u in design_units(m):
        per[u.branch] += w * ideal_expectation(m, u, cmp)
    f_avg = sum(per.values())
    return (3 * f_avg - 1) / 2, {b: (3 * v / BRANCH_WEIGHT[b] - 1) / 2 for b, v in per.items()}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        for mm in (1, 2):
            F, per = ideal_f_e(mm)
            print(f"m = {mm}: hardware design F_e = {F:.12f}   closed form {f_e_closed(mm):.12f}   "
                  f"{ {k: round(v, 10) for k, v in per.items()} }")
        sys.exit(0)
    raise SystemExit(__doc__)
