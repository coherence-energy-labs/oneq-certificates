"""CHANNEL-FAITHFUL hardware circuits for the two-use protocol, with a one-CZ pair decoder (R147).

Why. hw_two_use.py (the R145 run) compiles each erasure branch by identities that hold for the IDEAL channels: an
erased use becomes a classical flip of probability p, and P runs on at most one use. A hardware F_e above 3/4 from
those circuits would show the device ran a REDUCED protocol well; it could not show two physical zero-capacity
channels superactivating, because the channels are never physically present. Here they are:

  ALICE      the code |0_L> = |++>, |1_L> = |--> on key0, key1; per use, Bell pairs (a_t, s_t) and a copy
             CNOT key -> c. She holds nothing back: everything she prepares goes into P or A.
  P (x2)     P_d's real randomized Clifford(+T) gates on (key_u, s_u) for BOTH uses, with independent choices drawn
             from P_d's law, independent of the input.
  A (x2)     the 50% erasure: on an erased use the A register (c_u, a_u) is RESET (physical erasure; the flag is the
             branch label). Bob never touches an erased register.
  BOB        measures only channel outputs: A's outputs of kept uses, P's outputs of every use.

Bob's decoder is the lean one (hardware_design / lean_protocol), which IS a decoder of the faithful outputs: on a
kept use he decodes (key, s) with (c, a); for the second use he reads A's copy (kept) or P's key in Z (erased).

ONE-CZ PAIR DECODER (decoder="cz1"). Bob's per-pair measurement is FORCED: the outcome must separate
span{|00>, Psi+} from span{|11>, Psi-} (beta = 0) -- the observable (ZI + IZ + XX + YY)/2 of the R143 decoder
CNOT(a->s), CH(s->a), Z_a -- and (XX + ZI)/sqrt2 for beta = 1. Each is measurable with ONE CZ (searched over all
one-CZ circuits; exact pi/4 closed forms below; zero CZ is impossible), against two for CNOT + CH:
  beta = 0:  a: Ry(3pi/2);  s: Rz(pi) Ry(3pi/2);  CZ;  a: Ry(7pi/4);  s: Ry(3pi/4);  outcome = bit_a XOR bit_s
  beta = 1:  a: Rz(pi);  s: Ry(3pi/2) Rz(pi);  CZ;  a: Ry(pi/4);  outcome = bit_a  (s is not read)
The beta = 0 form reads BOTH qubits, so it trades one CZ for one more readout; the engine decides which is better.

Qubits: key0 = 0, key1 = 1, c0 = 2, c1 = 3, s_u[t] = 4 + 2um + t, a_u[t] = 4 + (2u+1)m + t  (n = 4 + 4m).
Classical bits, kept branches: 0 = l (c0), then the pair bits (cz2 / beta = 1: a_0[t]; cz1 with beta = 0: a_0[t], s_0[t]
interleaved), then b1 (c1 if use 1 kept, key1 if erased), then key0. Both erased: key0, key1.

  python hw_faithful.py check [M,..]   exact ideal F_e of faithful circuits (both decoders) = (7 - 6p + 2p^2)/8
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
import hw_two_use as H  # noqa: E402

BRANCH_WEIGHT = H.BRANCH_WEIGHT
#: R224 (2026-10-23, from R213's adversary finding): the FOLDED law above always erases use 1 when exactly one use is
#: erased. That is harmless for computing the protocol's F_e (the decoder is symmetric), but the hardware then
#: implements an erasure resource in which use 0 survives with weight 3/4. A strategy that ignores P and sends the
#: qubit through use 0 alone reaches F_e = 13/16 against it, so a hardware certificate needs BOTH orders, each at
#: 1/4: every use erased with probability 1/2, independently -- exactly A (x) A, whose null set is the antidegradable
#: channels.
BRANCH_WEIGHT_BOTH = {"kept_kept": 0.25, "kept_erased": 0.25, "erased_kept": 0.25, "erased_erased": 0.25}
ERASURE_ORDERS = {"folded": BRANCH_WEIGHT, "both": BRANCH_WEIGHT_BOTH}
INPUTS = H.INPUTS
BOB_BASIS = H.BOB_BASIS
p_flip = H.p_flip
f_e_closed = H.f_e_closed


@dataclass(frozen=True)
class FUnit:
    branch: str                     # kept_kept | kept_erased (use 1 erased) | erased_kept (use 0 erased) | erased_erased
    choices: tuple                  # P's sampled unitary on use 0 and on use 1 (hardware_design.sample_choice)
    beta: int
    inp: tuple

    @property
    def name(self) -> str:
        lab = "__".join(f"{c[0]}_{''.join(map(str, c[1]))}" for c in self.choices)
        return f"F_{self.branch}__{lab}__b{self.beta}__{self.inp[0]}{'p' if self.inp[1] > 0 else 'm'}"


class Q:
    def __init__(self, m):
        self.m = m
        self.n = 4 + 4 * m

    def s(self, u, t):
        return 4 + 2 * u * self.m + t

    def a(self, u, t):
        return 4 + (2 * u + 1) * self.m + t


def pair_decode(qc, a, s, beta, decoder):
    """Bob's measurement of one shield pair, up to the read-out; returns the qubits whose bits XOR to the eigenbit.
    decoder: 'cz2' (R143: CNOT + CH), 'cz1' (one CZ, both bases), 'mix' (one CZ for beta = 1 only)"""
    if decoder == "mix":
        decoder = "cz1" if beta else "cz2"
    if decoder == "cz2":
        qc.cx(a, s)
        qc.ch(s, a)
        if beta:
            qc.ry(-math.pi / 4, a)
        return [a]
    if beta == 0:
        qc.ry(3 * math.pi / 2, a)
        qc.rz(math.pi, s)
        qc.ry(3 * math.pi / 2, s)
        qc.cz(a, s)
        qc.ry(7 * math.pi / 4, a)
        qc.ry(3 * math.pi / 4, s)
        return [a, s]
    qc.rz(math.pi, a)
    qc.ry(3 * math.pi / 2, s)
    qc.rz(math.pi, s)
    qc.cz(a, s)
    qc.ry(math.pi / 4, a)
    return [a]


def mirror_unit(u: FUnit) -> FUnit:
    """the kept_erased unit whose qubit-mirror is the erased_kept unit u: P's choices trade uses with the registers"""
    if u.branch != "erased_kept":
        raise ValueError(f"only an erased_kept unit has a kept_erased mirror, not {u.branch}")
    return FUnit("kept_erased", (u.choices[1], u.choices[0]), u.beta, u.inp)


def mirror_qubits(m: int) -> list:
    """the use 0 <-> use 1 relabelling: keys 0 <-> 1, copies 2 <-> 3, every shield pair (s_0[t], a_0[t]) <-> (s_1[t],
    a_1[t]). The code |0_L> = |++>, |1_L> = |--> is symmetric under it (Alice's prep-then-CNOT gives a|00> + b|11>
    from either key), so a relabelled kept_erased circuit has the SAME outcome law as the erased_kept circuit that
    `circuit` builds directly (Alice shared, Bob mirrored): use 0's A register is reset, and Bob decodes use 1 and
    reads use 0's key in Z. Its classical bits keep their meaning, so the score is unchanged. Kept as an independent
    oracle for the direct construction (tests/test_erasure_orders.py)."""
    L = Q(m)
    perm = list(range(L.n))
    perm[0], perm[1], perm[2], perm[3] = 1, 0, 3, 2
    for t in range(m):
        for uu in range(2):
            perm[L.s(uu, t)], perm[L.a(uu, t)] = L.s(1 - uu, t), L.a(1 - uu, t)
    return perm


def circuit(m: int, u: FUnit, decoder: str = "cz1", erase: str = "reset"):
    """the faithful circuit. erase = 'reset' (hardware: physical erasure) or 'none' (simulation: an erased register is
    never touched again, which equals tracing it; identical outcome law)"""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    L = Q(m)
    erased = {"kept_kept": (0, 0), "kept_erased": (0, 1), "erased_kept": (1, 0), "erased_erased": (1, 1)}[u.branch]
    q = QuantumRegister(L.n, "q")
    # ---- Alice: input, code, per-use encoding
    pre = QuantumCircuit(q)
    H._prep(pre, 0, u.inp)
    pre.cx(0, 1)
    pre.h(0)
    pre.h(1)
    for uu in range(2):
        for t in range(m):
            pre.h(L.a(uu, t))
            pre.cx(L.a(uu, t), L.s(uu, t))
        pre.cx(uu, 2 + uu)
    # ---- the channels: P on both uses, then A
    for uu in range(2):
        H._apply_p(pre, u.choices[uu], key=uu, shields=[L.s(uu, t) for t in range(m)])
    for uu in range(2):
        if erased[uu] and erase == "reset":
            pre.reset([2 + uu] + [L.a(uu, t) for t in range(m)])
    # ---- Bob
    if u.branch == "erased_erased":
        c = ClassicalRegister(2, "c")
        qc = QuantumCircuit(q, c, name=u.name)
        qc.compose(pre, inplace=True)
        qc.measure([0, 1], [0, 1])
        return qc
    # Bob decodes the kept use (use 1 when use 0 is the erased one, R224) and reads the other: its A copy if kept,
    # its P key in Z if erased. Alice's part above is symmetric in the two uses, so this is the only mirrored step.
    dec = 1 if u.branch == "erased_kept" else 0
    oth = 1 - dec
    qc_body = pre.copy()
    qc_body.cx(dec, 2 + dec)
    pair_bits = []
    for t in range(m):
        pair_bits += pair_decode(qc_body, L.a(dec, t), L.s(dec, t), u.beta, decoder)
    H._measure_in(qc_body, dec, BOB_BASIS[u.inp[0]][0])
    b1 = 2 + oth if u.branch == "kept_kept" else oth
    order = [2 + dec] + pair_bits + [b1, dec]
    c = ClassicalRegister(len(order), "c")
    qc = QuantumCircuit(q, c, name=u.name)
    qc.compose(qc_body, inplace=True)
    qc.measure(order, list(range(len(order))))
    return qc


def lean_circuit(m: int, u, decoder: str = "cz1"):
    """hw_two_use.circuit (the R145 lean branch compilation, H.Unit) with Bob's pair decoder selectable. Used to score
    decoders on the engine; the faithful circuits relate to these exactly in the engine's model (see composition)."""
    if decoder == "cz2" or u.branch == "erased_erased":
        return H.circuit(m, u)
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    s = [4 + t for t in range(m)]
    a = [4 + m + t for t in range(m)]
    q = QuantumRegister(4 + 2 * m, "q")
    body = QuantumCircuit(q)
    H._prep(body, 0, u.inp)
    body.cx(0, 1)
    body.h(0)
    body.h(1)
    for t in range(m):
        body.h(a[t])
        body.cx(a[t], s[t])
    body.cx(0, 2)
    H._apply_p(body, u.choice, key=0, shields=s)
    if u.branch == "kept_kept":
        body.cx(1, 3)
    elif u.flips[1]:
        body.x(1)
    body.cx(0, 2)
    bits = []
    for t in range(m):
        bits += pair_decode(body, a[t], s[t], u.beta, decoder)
    H._measure_in(body, 0, BOB_BASIS[u.inp[0]][0])
    order = [2] + bits + [3 if u.branch == "kept_kept" else 1, 0]
    c = ClassicalRegister(len(order), "c")
    qc = QuantumCircuit(q, c, name=u.name)
    qc.compose(body, inplace=True)
    qc.measure(order, list(range(len(order))))
    return qc


def lean_ideal(m: int, u, decoder: str) -> float:
    from qiskit.quantum_info import Statevector
    qc = lean_circuit(m, u, decoder)
    nb = qc.num_clbits
    probs = Statevector(qc.remove_final_measurements(inplace=False)).probabilities()
    qubit_of = {qc.find_bit(i.clbits[0]).index: qc.find_bit(i.qubits[0]).index for i in qc.data
                if i.operation.name == "measure"}
    sc = score(m, u, decoder)
    return float(sum(probs[i] * sc(tuple((int(i) >> qubit_of[j]) & 1 for j in range(nb)))
                     for i in np.flatnonzero(probs > 1e-15)))


def n_pair_bits(m, u, decoder):
    return 2 * m if (decoder == "cz1" and u.beta == 0) else m


def score(m: int, u: FUnit, decoder: str = "cz1"):
    """score(bits) in classical-bit order; same law as hw_two_use.score, with the pair parity over all pair bits"""
    basis, e = u.inp
    if u.branch == "erased_erased":
        def ee(bits):
            if basis != "X":
                return 0.5
            return 1.0 if (1 - 2 * (bits[0] ^ bits[1])) == e else 0.0
        return ee
    k = n_pair_bits(m, u, decoder)
    meas_basis, sign = BOB_BASIS[basis]
    e_prime = sign * e
    z_flips = meas_basis in "XY"
    x_flips = meas_basis in "ZY"

    def kept(bits):
        ell, par, b1, o = bits[0], sum(bits[1:1 + k]) % 2, bits[k + 1], bits[k + 2]
        if ell != u.beta:
            return 0.0
        want = e_prime * (-1) ** ((par if z_flips else 0) + ((ell ^ b1) if x_flips else 0))
        return 2.0 if (1 - 2 * o) == want else 0.0
    return kept


def ideal_expectation(m: int, u: FUnit, decoder: str) -> float:
    from qiskit.quantum_info import Statevector
    qc = circuit(m, u, decoder, erase="none")
    nb = qc.num_clbits
    bare = qc.remove_final_measurements(inplace=False)
    probs = Statevector(bare).probabilities()
    meas = [(qc.find_bit(i.qubits[0]).index, qc.find_bit(i.clbits[0]).index) for i in qc.data
            if i.operation.name == "measure"]
    qubit_of = {cb: qb for qb, cb in meas}
    sc = score(m, u, decoder)
    tot = 0.0
    for idx in np.flatnonzero(probs > 1e-15):
        bits = tuple((int(idx) >> qubit_of[j]) & 1 for j in range(nb))
        tot += probs[idx] * sc(bits)
    return float(tot)


def design_units(m: int, erasure_orders: str = "folded"):
    """every faithful unit with its weight in F_avg: P's law on BOTH uses, beta uniform, 6 inputs; erasure_orders
    'both' (R224) splits the one-erased weight between the two orders"""
    weights = ERASURE_ORDERS[erasure_orders]
    labs = list(H.p_labels(m))
    for inp in INPUTS:
        for (w0, c0), (w1, c1) in itertools.product(labs, labs):
            for branch in ("kept_kept", "kept_erased", "erased_kept"):
                if branch not in weights:
                    continue
                for beta in (0, 1):
                    yield weights[branch] * w0 * w1 / 6 / 2, FUnit(branch, (c0, c1), beta, inp)
            yield weights["erased_erased"] * w0 * w1 / 6, FUnit("erased_erased", (c0, c1), 0, inp)


def ideal_f_e(m: int, decoder: str, erasure_orders: str = "folded"):
    weights = ERASURE_ORDERS[erasure_orders]
    per = dict.fromkeys(weights, 0.0)
    for w, u in design_units(m, erasure_orders):
        per[u.branch] += w * ideal_expectation(m, u, decoder)
    f_avg = sum(per.values())
    return (3 * f_avg - 1) / 2, {b: (3 * v / weights[b] - 1) / 2 for b, v in per.items()}


def erasure_marginals(erasure_orders: str) -> dict:
    """the implemented erasure resource: P(use u erased) and P(both erased). A (x) A needs 1/2, 1/2 and 1/4"""
    w = ERASURE_ORDERS[erasure_orders]
    erased = {"kept_kept": (0, 0), "kept_erased": (0, 1), "erased_kept": (1, 0), "erased_erased": (1, 1)}
    return dict(use0=sum(v for b, v in w.items() if erased[b][0]), use1=sum(v for b, v in w.items() if erased[b][1]),
                both=sum(v for b, v in w.items() if erased[b] == (1, 1)))


def p_free_use0_f_e(erasure_orders: str) -> float:
    """the adversary's P-free strategy (R213): ignore P, send the logical qubit through use 0's A register, output it
    unless use 0 is erased (then a maximally mixed guess, F_e 1/4). Its F_e must not exceed 3/4 for the bar to bind"""
    keep0 = 1 - erasure_marginals(erasure_orders)["use0"]
    return keep0 + (1 - keep0) / 4


def reset_check(m: int, n_units: int = 6, seed: int = 4) -> float:
    """max |E[score] with physical RESET (Aer density matrix) - E[score] with the register left untouched|"""
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    rng = np.random.default_rng(seed)
    sim = AerSimulator(method="density_matrix")
    worst = 0.0
    for branch in ("kept_erased", "erased_kept", "erased_erased"):
        for _ in range(n_units):
            u = FUnit(branch, (hd.sample_choice(m, rng), hd.sample_choice(m, rng)), int(rng.integers(2)),
                      INPUTS[int(rng.integers(6))])
            for decoder in ("cz1", "cz2"):
                qc = circuit(m, u, decoder, erase="reset")
                bare = qc.remove_final_measurements(inplace=False)
                # probabilities of the measured qubits in clbit order
                meas = [qc.find_bit(i.qubits[0]).index for i in qc.data if i.operation.name == "measure"]
                bare.save_probabilities_dict(qubits=meas, label="p")
                pr = sim.run(transpile(bare, sim, optimization_level=0)).result().data()["p"]
                sc = score(m, u, decoder)
                val = sum(v * sc(tuple((int(k, 16) if isinstance(k, str) else k) >> j & 1 for j in range(len(meas))))
                          for k, v in pr.items())
                worst = max(worst, abs(val - ideal_expectation(m, u, decoder)))
    return worst


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        ms = (1, 2) if len(sys.argv) < 3 else tuple(int(x) for x in sys.argv[2].split(","))
        for mm in ms:
            for dec in ("cz2", "cz1"):
                for orders in ("folded", "both"):
                    F, per = ideal_f_e(mm, dec, orders)
                    print(f"m = {mm}, decoder {dec}, {orders} erasure orders: FAITHFUL F_e = {F:.12f}   closed form "
                          f"{f_e_closed(mm):.12f}   { {k: round(v, 10) for k, v in per.items()} }", flush=True)
        print(f"physical reset vs untouched register (Aer density matrix, m = 1): max |diff| = {reset_check(1):.2e}")
        sys.exit(0)
    raise SystemExit(__doc__)
