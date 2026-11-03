r"""wu_core.py -- the finite data of WINDOW_PROOF.md, read off the lane's builder (READ-ONLY import).

Everything here is exact GF(2) bookkeeping on stim's instruction list (stim only PARSES the circuit; its error
analysis is used nowhere in this module):

  * segments / blocks: C_T = P . F . B^(T-1) . E . R  (prep, first body block, body block, tail, readout)
  * record labels (qubit, occurrence) and the closed-form detectors / observables
  * per-block backward pass: the block maps A (frame in -> frame out) and C (frame in -> block measurement flips),
    every noise component's direct effect, every gauge (reset / measurement basis Pauli) direct effect

Frame bits: x_q -> bit q, z_q -> bit NQ + q; block measurement i -> bit 2 NQ + i.
"""
from __future__ import annotations

import pathlib
import sys

LANE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(LANE))

import numpy as np  # noqa: E402
import stim  # noqa: E402

import bb_circuits as bc  # noqa: E402

ANNOT = ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS")
NOISE = ("X_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2")
PAULI2 = [(a, b) for a in "IXYZ" for b in "IXYZ" if (a, b) != ("I", "I")]

# period (TICK-segments per round) and tail length (segments after the last full block, before the readout)
SHAPE = {"ibm": (9, 0), "3c": (8, 5)}


def bits(v: int):
    while v:
        b = v & -v
        yield b.bit_length() - 1
        v ^= b


# ------------------------------------------------------------------------------------------- parsing
def flat(circuit: stim.Circuit) -> list:
    """[(name, targets tuple, args tuple)]; for DETECTOR / OBSERVABLE_INCLUDE the targets are rec offsets."""
    out = []
    for inst in circuit.flattened():
        tg = tuple(t.value for t in inst.targets_copy())
        out.append((inst.name, tg, tuple(inst.gate_args_copy())))
    return out


def segments(instrs: list) -> list:
    """Split after every TICK (the TICK stays with its segment); the trailing readout has no TICK."""
    segs, cur = [], []
    for ins in instrs:
        cur.append(ins)
        if ins[0] == "TICK":
            segs.append(cur)
            cur = []
    segs.append(cur)
    return segs


def strip(seg: list) -> list:
    return [ins for ins in seg if ins[0] not in ANNOT]


class Code:
    """Qubit layout and code data shared by both builders: data 0..2n2-1, X ancilla of check k = 2 n2 + k,
    Z ancilla of check k = 3 n2 + k."""

    def __init__(self, basis: str, l: int = 12):
        code = bc.BB(l)
        self.n2 = code.n2
        self.nd = 2 * self.n2
        self.nq = 4 * self.n2
        self.basis = basis
        self.bi = 1 if basis == "Z" else 0          # coordinate type of the memory-type detectors
        HX, HZ = code.check_matrices()
        self.H = {0: [frozenset(np.nonzero(r)[0].tolist()) for r in HX],
                  1: [frozenset(np.nonzero(r)[0].tolist()) for r in HZ]}
        LOG = code.logicals(HZ, HX) if basis == "Z" else code.logicals(HX, HZ)
        self.L = [frozenset(np.nonzero(v)[0].tolist()) for v in LOG]

    def anc(self, ti: int, k: int) -> int:
        return (2 if ti == 0 else 3) * self.n2 + k

    def check_of(self, q: int):
        """(type index, k) of an ancilla qubit, None for a data qubit."""
        if q < self.nd:
            return None
        return (0, q - 2 * self.n2) if q < 3 * self.n2 else (1, q - 3 * self.n2)


def build(kind: str, basis: str, T: int) -> stim.Circuit:
    return bc.build(kind, basis, T)


# ------------------------------------------------------------------------------------------- records
def records(instrs: list):
    """Label every measurement (qubit, occurrence index) in circuit order; detectors / observables as label sets.
    Returns (labels list, {coord tuple: frozenset of labels}, {obs index: frozenset of labels}, coord list in
    declaration order)."""
    labels, occ = [], {}
    dets, order, obs = {}, [], {}
    for name, tg, args in instrs:
        if name in ("M", "MX"):
            for q in tg:
                labels.append((q, occ.get(q, 0)))
                occ[q] = occ.get(q, 0) + 1
        elif name == "DETECTOR":
            co = tuple(int(round(v)) for v in args)
            if co in dets:
                raise ValueError(f"duplicate detector coordinate {co}")
            dets[co] = frozenset(labels[len(labels) + o] for o in tg)
            order.append(co)
        elif name == "OBSERVABLE_INCLUDE":
            k = int(args[0])
            obs[k] = obs.get(k, frozenset()) ^ frozenset(labels[len(labels) + o] for o in tg)
    return labels, dets, obs, order


def closed_form(code: Code, T: int):
    """WINDOW_PROOF.md section 1.4: the detectors and observables of C_T, as label sets keyed by coordinates."""
    dets = {}
    for s in range(T):
        for ti in (0, 1):
            for k in range(code.n2):
                a = code.anc(ti, k)
                if s == 0:
                    if ti == code.bi:
                        dets[(k, ti, 0)] = frozenset({(a, 0)})
                else:
                    dets[(k, ti, s)] = frozenset({(a, s), (a, s - 1)})
    for k in range(code.n2):
        dets[(k, code.bi, T)] = frozenset({(q, 0) for q in code.H[code.bi][k]} | {(code.anc(code.bi, k), T - 1)})
    obs = {j: frozenset((q, 0) for q in Lj) for j, Lj in enumerate(code.L)}
    return dets, obs


# ------------------------------------------------------------------------------------------- template
class Template:
    """P, F, B, E, R read off C_1 and C_2 (annotations stripped); C_T is P . F . B^(T-1) . E . R."""

    def __init__(self, kind: str, basis: str):
        self.kind, self.basis = kind, basis
        self.per, self.tail = SHAPE[kind]
        s1 = segments(flat(build(kind, basis, 1)))
        s2 = segments(flat(build(kind, basis, 2)))
        p, e = self.per, self.tail
        assert len(s1) == 1 + p + e + 1 and len(s2) == 1 + 2 * p + e + 1, "unexpected segment count"
        cat = lambda ss: [ins for seg in ss for ins in strip(seg)]  # noqa: E731
        self.P = cat(s1[:1])
        self.F = cat(s1[1:1 + p])
        self.B = cat(s2[1 + p:1 + 2 * p])
        self.E = cat(s1[1 + p:1 + p + e])
        self.R = cat(s1[1 + p + e:])
        # the same pieces read off C_2 must agree (F, E, R and P do not depend on T)
        assert cat(s2[:1]) == self.P and cat(s2[1:1 + p]) == self.F
        assert cat(s2[1 + 2 * p:1 + 2 * p + e]) == self.E and cat(s2[1 + 2 * p + e:]) == self.R

    def stream(self, T: int) -> list:
        return self.P + self.F + self.B * (T - 1) + self.E + self.R


def template_check(tpl: Template, code: Code, T: int, circuit: stim.Circuit | None = None) -> dict:
    """The builder's C_T against the template: instruction stream (annotations stripped) and closed-form
    detectors / observables (as label sets, keyed by coordinates)."""
    c = circuit if circuit is not None else build(tpl.kind, tpl.basis, T)
    ins = flat(c)
    stream_ok = [x for x in ins if x[0] not in ANNOT] == tpl.stream(T)
    _, dets, obs, _ = records(ins)
    cd, co = closed_form(code, T)
    return dict(T=T, stream=stream_ok, detectors=dets == cd, observables=obs == co,
                n_detectors=len(dets), agrees=bool(stream_ok and dets == cd and obs == co))


# ------------------------------------------------------------------------------------------- block maps
class Block:
    """Backward pass over one block: maps and direct effects.

    fx0[q] / fz0[q]: what an X_q / Z_q entering the block causes -- frame bits at the block's end (A) and block
    measurement bits (C). comps: [(instr index, {q: P}, mask)] for every noise component. gauges: [(instr index,
    q, basis Pauli, mask)] for every reset and measurement (the basis Pauli inserted right after it)."""

    def __init__(self, instrs: list, nq: int, name: str = ""):
        self.name, self.nq = name, nq
        self.meas = [(q, nm) for nm, tg, _ in instrs if nm in ("M", "MX") for q in tg]
        FR = 2 * nq
        self.FR = FR
        self.fmask = (1 << FR) - 1
        fx = [1 << q for q in range(nq)]
        fz = [1 << (nq + q) for q in range(nq)]
        mi = len(self.meas)
        comps, gauges = [], []

        def eff(q, p):
            return {"I": 0, "X": fx[q], "Z": fz[q], "Y": fx[q] ^ fz[q]}[p]

        for idx in range(len(instrs) - 1, -1, -1):
            nm, tg, args = instrs[idx]
            if nm == "TICK":
                continue
            if nm == "M":
                for q in reversed(tg):
                    mi -= 1
                    gauges.append((idx, q, "Z", fz[q]))
                    fx[q] ^= 1 << (FR + mi)
            elif nm == "MX":
                for q in reversed(tg):
                    mi -= 1
                    gauges.append((idx, q, "X", fx[q]))
                    fz[q] ^= 1 << (FR + mi)
            elif nm == "R":
                for q in tg:
                    gauges.append((idx, q, "Z", fz[q]))
                    fx[q] = fz[q] = 0
            elif nm == "RX":
                for q in tg:
                    gauges.append((idx, q, "X", fx[q]))
                    fx[q] = fz[q] = 0
            elif nm == "H":
                for q in tg:
                    fx[q], fz[q] = fz[q], fx[q]
            elif nm == "CX":
                pairs = list(zip(tg[0::2], tg[1::2]))
                for a, b in reversed(pairs):
                    fx[a] ^= fx[b]
                    fz[b] ^= fz[a]
            elif nm == "X_ERROR":
                comps += [(idx, {q: "X"}, fx[q]) for q in tg]
            elif nm == "Z_ERROR":
                comps += [(idx, {q: "Z"}, fz[q]) for q in tg]
            elif nm == "DEPOLARIZE1":
                for q in tg:
                    comps += [(idx, {q: p}, eff(q, p)) for p in "XYZ"]
            elif nm == "DEPOLARIZE2":
                for a, b in zip(tg[0::2], tg[1::2]):
                    comps += [(idx, {q: p for q, p in ((a, pa), (b, pb)) if p != "I"}, eff(a, pa) ^ eff(b, pb))
                              for pa, pb in PAULI2]
            else:
                raise ValueError(f"instruction {nm} is not modelled")
        assert mi == 0
        self.fx0, self.fz0 = fx, fz
        self.comps, self.gauges = comps, gauges
        # column images of the frame-in basis vectors
        self.col = [fx[q] for q in range(nq)] + [fz[q] for q in range(nq)]

    def apply(self, x: int) -> int:
        """Full effect (frame out | meas flips << FR) of a frame x entering the block."""
        out = 0
        for b in bits(x):
            out ^= self.col[b]
        return out

    def A(self, x: int) -> int:
        return self.apply(x) & self.fmask

    def C(self, x: int) -> int:
        return self.apply(x) >> self.FR
