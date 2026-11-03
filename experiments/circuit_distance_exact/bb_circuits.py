r"""R198: the two syndrome cycles whose circuit distance this lane settles, as stim memory experiments.

THE CODE. A bivariate bicycle code BB(l, m; A, B) on L and R data blocks of l*m qubits each, with
H_X = [A | B] and H_Z = [B^T | A^T]. The gross code [[144,12,12]] is l = 12, m = 6, A = x^3 + y + y^2,
B = y^3 + x + x^2 (Bravyi et al., Nature 627, 778 (2024)); l = 6 gives [[72,12,6]]. A monomial x^a y^b
acts on the index i * m + j as (i, j) -> (i + a, j + b).

*** THE IBM CYCLE (`ibm_cycle`). *** Seven CNOT layers, X checks on terms sX = [idle, 1, 4, 3, 5, 0, 2],
Z checks on sZ = [3, 5, 0, 1, 2, 4, idle]; term k < 3 of an X check couples to L via A_{k+1}, k >= 3 to R
via B_{k-2}; a Z check's term k < 3 couples to L via B_{k+1}^T, k >= 3 to R via A_{k-2}^T; the X ancilla
controls, the Z ancilla is the target. Verified term by term against IBM's own decoder_setup.py
(github.com/sbravyi/BivariateBicycleCodes) on 2026-10-20. The one difference: here every ancilla is reset
and measured in its own layer (IBM measures Z checks in layer 6 and prepares them in layer 7); no Z-ancilla
CNOT follows layer 5, so no fault propagates differently -- only idle timing moves.

*** THE 3-COLOUR CYCLE (`three_colour_cycle`). *** Strikis, Browne & Beverland, arXiv 2603.05481, Fig. 12:
a depth-8 left-right schedule, uniform on checks of one colour, conjectured to reach circuit distance 11.
Rebuilt from the paper's figures. The geometry is DERIVED, not read: an X check at cell (a, b) has index
x^b y^a, its up / down / left / right / long-upper-right / long-lower-left neighbours are L(y i), L(y^2 i),
R(x i), R(x^2 i), L(x^3 i), R(y^3 i), and a Z check j has R(y^-2 j), R(y^-1 j), L(x^-2 j), L(x^-1 j),
L(y^-3 j), R(x^-3 j). Of the four cell orientations consistent with the Tanner graph, this is the one
whose long edges point where Fig. 10 draws them. Colours (red, green, blue = 0, 1, 2) are (a + b) mod 3
for both types. Two independent confirmations: stim accepts every detector as deterministic, and the
schedule's hook residuals reproduce all six residual patterns of Fig. 11 (`residual_sets`, tested).

*** THE MORPHING CIRCUITS (`morphing_cycle`). *** Shaw & Terhal, arXiv 2407.16336 (PRL 134, 090602): no
ancillas -- the 2lm qubits of a MID-cycle BB code alternate between two contraction circuits F_0, F_1 (three CNOT
layers each, driven by a homomorphism f: G -> Z_2), each ending in an l*m-qubit END-cycle code with weight-9
checks on degree-5 connectivity. "morph72": mid-cycle [[144,12,12]] (the gross polynomials) -> [[72,12,6]];
"morph144": mid-cycle [[288,12,18]] (A = x^3 + y^7 + y^2) -> [[144,12,12]]. Their circuit-distance numbers are
BP-OSD upper bounds (<= 6, <= 12). The schedule is a port of the authors' builder and was compared with it gate
by gate for both codes and all three homomorphisms (`morph_vs_authors.py`: identical gates, measurements and
contraction rounds, 2026-10-20); the detectors are derived independently (`morphing_cycle`'s docstring).

*** NOISE. *** Standard circuit noise, every location can fail: DEPOLARIZE2 after every CNOT, DEPOLARIZE1
on idle qubits, flips after every preparation and before every measurement. Distance counts DEM fault
mechanisms, so p only labels the model. The 3-colour schedule leaves no ancilla idle (prep, six
consecutive CNOTs, measure, re-prep), so idle noise there is on data qubits only.
"""
from __future__ import annotations

import numpy as np
import stim

GROSS_A = (("x", 3), ("y", 1), ("y", 2))
GROSS_B = (("y", 3), ("x", 1), ("x", 2))
SX = ("idle", 1, 4, 3, 5, 0, 2)
SZ = (3, 5, 0, 1, 2, 4, "idle")

# Fig. 12 of 2603.05481: times within an 8-step cycle; values > 8 belong to the next cycle.
SCHEDULE_3C = {
    ("Z", 0): dict(prep=1, up=4, down=2, left=7, right=6, ur=5, ll=3, meas=8),
    ("Z", 1): dict(prep=1, up=4, down=2, left=5, right=7, ur=6, ll=3, meas=8),
    ("Z", 2): dict(prep=2, up=4, down=5, left=6, right=8, ur=7, ll=3, meas=9),
    ("X", 0): dict(prep=5, up=10, down=9, left=7, right=8, ur=11, ll=6, meas=12),
    ("X", 1): dict(prep=5, up=10, down=9, left=6, right=7, ur=11, ll=8, meas=12),
    ("X", 2): dict(prep=6, up=10, down=12, left=8, right=9, ur=11, ll=7, meas=13),
}
DIRECTIONS = ("up", "down", "left", "right", "ur", "ll")


class BB:
    """BB(l, m) with the gross-code polynomials; permutations, check matrices, logical operators."""

    def __init__(self, l: int = 12, m: int = 6, a_terms=GROSS_A, b_terms=GROSS_B):
        self.l, self.m, self.n2 = l, m, l * m
        self.AP = [self._mono(t) for t in a_terms]
        self.BP = [self._mono(t) for t in b_terms]

    def idx(self, i: int, j: int) -> int:
        return (i % self.l) * self.m + (j % self.m)

    def _mono(self, term) -> np.ndarray:
        """p with (monomial matrix)[r, p[r]] = 1."""
        var, pw = term
        p = np.zeros(self.n2, dtype=int)
        for i in range(self.l):
            for j in range(self.m):
                p[self.idx(i, j)] = self.idx(i + pw, j) if var == "x" else self.idx(i, j + pw)
        return p

    @staticmethod
    def inv(p: np.ndarray) -> np.ndarray:
        q = np.empty_like(p)
        q[p] = np.arange(len(p))
        return q

    def check_matrices(self):
        A = np.zeros((self.n2, self.n2), dtype=np.uint8)
        B = np.zeros((self.n2, self.n2), dtype=np.uint8)
        for p in self.AP:
            A[np.arange(self.n2), p] ^= 1
        for p in self.BP:
            B[np.arange(self.n2), p] ^= 1
        HX, HZ = np.hstack([A, B]), np.hstack([B.T, A.T])
        assert not ((HX.astype(int) @ HZ.T.astype(int)) % 2).any()
        return HX, HZ

    def logicals(self, H_same, H_other) -> np.ndarray:
        """12 logical operators of H_same's type: ker(H_other) modulo rowspace(H_same)."""
        m = H_other.copy() % 2
        rows, cols = m.shape
        pivots, r = [], 0
        for c in range(cols):
            piv = next((i for i in range(r, rows) if m[i, c]), None)
            if piv is None:
                continue
            m[[r, piv]] = m[[piv, r]]
            for i in range(rows):
                if i != r and m[i, c]:
                    m[i] ^= m[r]
            pivots.append(c)
            r += 1
        free = [c for c in range(cols) if c not in pivots]
        basis = []
        for f in free:
            v = np.zeros(cols, dtype=np.uint8)
            v[f] = 1
            for k, pc in enumerate(pivots):
                if m[k, f]:
                    v[pc] = 1
            basis.append(v)
        base = gf2_rank(H_same)
        cur, out = H_same.copy(), []
        for v in basis:
            trial = np.vstack([cur, v])
            if gf2_rank(trial) > gf2_rank(cur):
                cur = trial
                out.append(v)
            if len(out) == 12:
                break
        assert gf2_rank(cur) == base + 12
        return np.array(out, dtype=np.uint8)

    # neighbours, as data-qubit indices (L block 0..n2-1, R block n2..2*n2-1)
    def ibm_x_target(self, check: int, term: int) -> int:
        return int(self.AP[term][check]) if term < 3 else self.n2 + int(self.BP[term - 3][check])

    def ibm_z_target(self, check: int, term: int) -> int:
        return int(self.inv(self.BP[term])[check]) if term < 3 else self.n2 + int(self.inv(self.AP[term - 3])[check])

    def geo_neighbours(self, typ: str, k: int) -> dict:
        n2, AP, BP, inv = self.n2, self.AP, self.BP, self.inv
        if typ == "X":
            return dict(up=int(AP[1][k]), down=int(AP[2][k]), left=n2 + int(BP[1][k]), right=n2 + int(BP[2][k]),
                        ur=int(AP[0][k]), ll=n2 + int(BP[0][k]))
        return dict(up=n2 + int(inv(AP[2])[k]), down=n2 + int(inv(AP[1])[k]), left=int(inv(BP[2])[k]),
                    right=int(inv(BP[1])[k]), ur=int(inv(BP[0])[k]), ll=n2 + int(inv(AP[0])[k]))

    def colour(self, typ: str, k: int) -> int:
        al, be = divmod(k, self.m)
        return (al + be) % 3 if typ == "X" else (al + be + 2) % 3


def gf2_rank(mat: np.ndarray) -> int:
    m = mat.copy() % 2
    r = 0
    rows, cols = m.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if m[i, c]), None)
        if piv is None:
            continue
        m[[r, piv]] = m[[piv, r]]
        for i in range(rows):
            if i != r and m[i, c]:
                m[i] ^= m[r]
        r += 1
        if r == rows:
            break
    return r


def _final_readout(c: stim.Circuit, code: BB, basis: str, rounds: int, nmeas: int, last: dict, p: float) -> None:
    n2 = code.n2
    data = list(range(2 * n2))
    HX, HZ = code.check_matrices()
    if basis == "X":
        c.append("H", data)
    c.append("X_ERROR", data, p)
    c.append("M", data)
    base = nmeas
    nmeas += 2 * n2
    H = HX if basis == "X" else HZ
    for k in range(n2):
        recs = [stim.target_rec(base + q - nmeas) for q in np.nonzero(H[k])[0]]
        recs.append(stim.target_rec(last[(basis, k)] - nmeas))
        c.append("DETECTOR", recs, [k, 0 if basis == "X" else 1, rounds])
    LOG = code.logicals(HZ, HX) if basis == "Z" else code.logicals(HX, HZ)
    for i, v in enumerate(LOG):
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(base + q - nmeas) for q in np.nonzero(v)[0]], i)


def ibm_cycle(basis: str, rounds: int, l: int = 12, p: float = 1e-3) -> stim.Circuit:
    """IBM's depth-8 cycle, `rounds` rounds of memory in `basis`. Detector coords (check, type, round)."""
    code = BB(l)
    n2 = code.n2
    data = list(range(2 * n2))
    xanc = [2 * n2 + k for k in range(n2)]
    zanc = [3 * n2 + k for k in range(n2)]
    c = stim.Circuit()
    c.append("R", data + xanc + zanc)
    if basis == "X":
        c.append("H", data)
    c.append("TICK")
    nmeas, last = 0, {}
    for r in range(rounds):
        c.append("R", xanc + zanc)
        c.append("X_ERROR", xanc + zanc, p)
        c.append("H", xanc)
        c.append("DEPOLARIZE1", xanc, p)
        c.append("TICK")
        for t in range(7):
            pairs, busy = [], set()
            if SX[t] != "idle":
                for k in range(n2):
                    q = code.ibm_x_target(k, SX[t])
                    pairs += [xanc[k], q]
                    busy |= {xanc[k], q}
            if SZ[t] != "idle":
                for k in range(n2):
                    q = code.ibm_z_target(k, SZ[t])
                    pairs += [q, zanc[k]]
                    busy |= {zanc[k], q}
            assert len(busy) == len(pairs), "a qubit is used twice in one layer"
            c.append("CX", pairs)
            c.append("DEPOLARIZE2", pairs, p)
            idle = [q for q in data + xanc + zanc if q not in busy]
            if idle:
                c.append("DEPOLARIZE1", idle, p)
            c.append("TICK")
        c.append("H", xanc)
        c.append("DEPOLARIZE1", xanc, p)
        c.append("X_ERROR", xanc + zanc, p)
        c.append("M", xanc + zanc)
        base = nmeas
        nmeas += 2 * n2
        cur = {("X", k): base + k for k in range(n2)}
        cur.update({("Z", k): base + n2 + k for k in range(n2)})
        for k in range(n2):
            for typ in ("X", "Z"):
                now = cur[(typ, k)] - nmeas
                if r == 0:
                    if typ == basis:
                        c.append("DETECTOR", [stim.target_rec(now)], [k, 0 if typ == "X" else 1, r])
                else:
                    c.append("DETECTOR", [stim.target_rec(now), stim.target_rec(last[(typ, k)] - nmeas)],
                             [k, 0 if typ == "X" else 1, r])
        last = cur
        c.append("TICK")
    _final_readout(c, code, basis, rounds, nmeas, last, p)
    return c


def three_colour_cycle(basis: str, rounds: int, p: float = 1e-3) -> stim.Circuit:
    """The 3-colour tiled cycle of 2603.05481 Fig. 12 on the gross code (l = 12 only)."""
    code = BB(12)
    n2 = code.n2
    data = list(range(2 * n2))
    anc = {("X", k): 2 * n2 + k for k in range(n2)}
    anc.update({("Z", k): 3 * n2 + k for k in range(n2)})
    events: dict[int, list] = {}
    for r in range(rounds):
        for typ in ("X", "Z"):
            for k in range(n2):
                s = SCHEDULE_3C[(typ, code.colour(typ, k))]
                base, a = 8 * r, anc[(typ, k)]
                events.setdefault(base + s["prep"], []).append(("prep", typ, a))
                for d, q in code.geo_neighbours(typ, k).items():
                    events.setdefault(base + s[d], []).append(("cx", typ, a, q))
                events.setdefault(base + s["meas"], []).append(("meas", typ, k, a, r))
    c = stim.Circuit()
    c.append("R", data)
    if basis == "X":
        c.append("H", data)
    c.append("X_ERROR" if basis == "Z" else "Z_ERROR", data, p)
    c.append("TICK")
    nmeas, rec = 0, {}
    for t in sorted(events):
        ev, busy = events[t], set()
        preps = [e for e in ev if e[0] == "prep"]
        cxs = [e for e in ev if e[0] == "cx"]
        meas = [e for e in ev if e[0] == "meas"]
        for e in preps:
            busy.add(e[2])
        zprep = [e[2] for e in preps if e[1] == "Z"]
        xprep = [e[2] for e in preps if e[1] == "X"]
        if zprep:
            c.append("R", zprep)
            c.append("X_ERROR", zprep, p)
        if xprep:
            c.append("RX", xprep)
            c.append("Z_ERROR", xprep, p)
        pairs = []
        for _, typ, a, q in cxs:
            pairs += [a, q] if typ == "X" else [q, a]
            assert a not in busy and q not in busy, f"qubit double-booked at t={t}"
            busy |= {a, q}
        if pairs:
            c.append("CX", pairs)
            c.append("DEPOLARIZE2", pairs, p)
        zm = [e for e in meas if e[1] == "Z"]
        xm = [e for e in meas if e[1] == "X"]
        for e in meas:
            assert e[3] not in busy, f"measured ancilla busy at t={t}"
            busy.add(e[3])
        if zm:
            c.append("X_ERROR", [e[3] for e in zm], p)
            c.append("M", [e[3] for e in zm])
            for e in zm:
                rec[("Z", e[2], e[4])] = nmeas
                nmeas += 1
        if xm:
            c.append("Z_ERROR", [e[3] for e in xm], p)
            c.append("MX", [e[3] for e in xm])
            for e in xm:
                rec[("X", e[2], e[4])] = nmeas
                nmeas += 1
        idle = [q for q in data if q not in busy]
        if idle:
            c.append("DEPOLARIZE1", idle, p)
        for e in meas:
            typ, k, r = e[1], e[2], e[4]
            here = stim.target_rec(rec[(typ, k, r)] - nmeas)
            ti = 0 if typ == "X" else 1
            if r == 0:
                if typ == basis:
                    c.append("DETECTOR", [here], [k, ti, r])
            else:
                c.append("DETECTOR", [here, stim.target_rec(rec[(typ, k, r - 1)] - nmeas)], [k, ti, r])
        c.append("TICK")
    last = {(basis, k): rec[(basis, k, rounds - 1)] for k in range(n2)}
    _final_readout(c, code, basis, rounds, nmeas, last, p)
    return c


# Shaw & Terhal, arXiv 2407.16336 (PRL 134, 090602), Table 1: the MID-cycle BB code of each morphing circuit; the
# end-cycle code lives on the l*m left qubits. Their BP-OSD circuit-distance upper bounds: <= 6 and <= 12.
MORPH_CODES = {
    "morph72": (12, 6, GROSS_A, GROSS_B),                                 # [[144,12,12]] -> end-cycle [[72,12,6]]
    "morph144": (12, 12, (("x", 3), ("y", 7), ("y", 2)), GROSS_B),        # [[288,12,18]] -> end-cycle [[144,12,12]]
}
CIRCUITS = ("ibm", "3c", *MORPH_CODES)
MORPH_HOMS = {"x": lambda a, b: a % 2, "y": lambda a, b: b % 2, "xy": lambda a, b: (a + b) % 2}


def morphing_schedule(l: int, m: int, a_terms, b_terms, hom: str = "x"):
    """The standard morphing circuit (2407.16336 Table IV; the authors' `get_standard_morphing_specification`,
    github.com/Mac-Shaw/morphing_qec_circuits): for each contraction round r in {0, 1} three CNOT layers of
    (control, target) pairs, and the right qubits measured in X and in Z after it. Qubit L(g) = g, R(g) = l*m + g.

    The homomorphism f: G -> Z_2 must give exactly one element of A, and one of B, a different value; that element
    is SWAPPED to the front (their convention: a swap, not a move, so the other two keep or exchange places)."""
    f = MORPH_HOMS[hom]
    el = lambda t: (t[1], 0) if t[0] == "x" else (0, t[1])  # noqa: E731
    lists = []
    for terms in (a_terms, b_terms):
        c = [el(t) for t in terms]
        vals = [f(*e) for e in c]
        odd = [i for i, v in enumerate(vals) if vals.count(v) == 1]
        if len(c) != 3 or len(odd) != 1:
            raise ValueError("the homomorphism does not single out one element of A and of B")
        c[0], c[odd[0]] = c[odd[0]], c[0]
        lists.append(c)
    A, B = lists
    n2 = l * m
    R = lambda i, j: n2 + (i % l) * m + (j % m)  # noqa: E731
    gates = [[[] for _ in range(3)] for _ in range(2)]
    meas = {"X": [[], []], "Z": [[], []]}
    for i in range(l):
        for j in range(m):
            g, rg = i * m + j, f(i, j)
            rh = (rg + 1) % 2
            right_g = [R(i - A[0][0] + b[0], j - A[0][1] + b[1]) for b in B]
            right_h = [R(i - a[0] + B[0][0], j - a[1] + B[0][1]) for a in A]
            for k in (1, 2):
                gates[rg][k - 1].append((g, right_g[k]))
                gates[rh][k - 1].append((right_h[k], g))
            gates[rg][2].append((right_g[0], g))
            gates[rh][2].append((g, right_h[0]))
            meas["X"][rg].append(right_g[0])
            meas["Z"][rh].append(right_h[0])
    for r in range(2):
        for layer in gates[r]:
            used = [q for pair in layer for q in pair]
            assert len(used) == len(set(used)), "a qubit is used twice in one layer"
        assert sorted(meas["X"][r] + meas["Z"][r]) == list(range(n2, 2 * n2)), "every right qubit is measured"
    return gates, meas


def _contract(support, pauli: str, layers) -> set:
    """A Pauli-`pauli` product carried forward through CNOT layers: X spreads control -> target, Z target -> control."""
    op = set(support)
    for layer in layers:
        for c, t in layer:
            if pauli == "X" and c in op:
                op ^= {t}
            elif pauli == "Z" and t in op:
                op ^= {c}
    return op


def morphing_cycle(kind: str, basis: str, rounds: int, hom: str = "x", p: float = 1e-3, tags: list | None = None) -> stim.Circuit:
    """`rounds` morphing QEC rounds of memory in `basis` (round i: reset the right qubits round i-1 measured, in the
    basis they were measured in; expand by round (i-1)'s three layers in reverse; contract by round i's three;
    measure round i's right qubits). Data = the l*m left qubits, prepared at the start and read out at the end.

    DETECTORS ARE DERIVED HERE, not copied: a mid-cycle stabilizer S carried through round r's contraction is
    S_meas (on round-r measured qubits of its own type; touching the other type raises) times S_rest (on left
    qubits). With v_i the value of S at the mid-cycle point of round i, v_{i+1} = v_i + m_i(S_meas) (the measured
    qubits are reset in their own basis before the next expansion). S CONTRACTS in round r when S_rest is empty;
    there v_i = m_i(S_meas) is a detector whenever v_i is known (v_0 = 0 for the memory basis and for the other
    type's stabilizers that contract in round 1 -- the expansion makes them single-qubit operators on freshly
    reset qubits; unknown for the rest until their first contraction). The readout closes the memory type:
    v_T + data(S_rest). Logicals follow the same recursion and close on the data. Detector coords (site g,
    type 0 = X / 1 = Z, round). The one family of deterministic parities NOT emitted: products, over the
    code's stabilizer redundancies, of the other type's round-0 outcomes (the meta-checks that IBM-style memory
    circuits also omit); `tools_cde.detector_gap` measures it."""
    l, m, a_terms, b_terms = MORPH_CODES[kind]
    n2 = l * m
    gates, meas = morphing_schedule(l, m, a_terms, b_terms, hom)
    code = BB(l, m, a_terms, b_terms)
    HX, HZ = code.check_matrices()
    stabs = {"X": [set(np.nonzero(HX[g])[0].tolist()) for g in range(n2)],
             "Z": [set(np.nonzero(HZ[g])[0].tolist()) for g in range(n2)]}
    LOG = code.logicals(HZ, HX) if basis == "Z" else code.logicals(HX, HZ)
    left = list(range(n2))
    split = {}  # (pauli, operator id, r) -> (measured records' qubits, rest)

    def carry(key, support, pauli, r):
        if key not in split:
            op = _contract(support, pauli, gates[r])
            other = "Z" if pauli == "X" else "X"
            assert not op & set(meas[other][r]), "a stabilizer or logical anticommutes with a measurement"
            same = op & set(meas[pauli][r])
            rest = op - same
            assert rest <= set(left)
            split[key] = (same, rest)
        return split[key]

    for pauli in ("X", "Z"):
        for g in range(n2):
            assert sum(not carry((pauli, g, r), stabs[pauli][g], pauli, r)[1] for r in (0, 1)) == 1, \
                "every stabilizer contracts in exactly one of the two rounds"
    c = stim.Circuit()
    tagged = []

    def noise(name, targets, phase, rnd, layer=None):
        tagged.append((len(c), name, phase, rnd, layer))
        c.append(name, targets, p)

    c.append("R" if basis == "Z" else "RX", left)
    noise("X_ERROR" if basis == "Z" else "Z_ERROR", left, "prep", 0)
    c.append("TICK")
    nmeas = 0
    # v_0: S at round 0's mid-cycle point, pulled back through round 1's contraction, is S_meas (on qubits round 0
    # resets in S's own basis: +1) times S_rest (on the data, prepared in the memory basis). So v_0 = 0 for the
    # memory type, and for the other type exactly when S_rest is empty (S contracts in round 1); else random.
    val = {(pauli, g): (set() if pauli == basis or not carry((pauli, g, 1), stabs[pauli][g], pauli, 1)[1] else None)
           for pauli in ("X", "Z") for g in range(n2)}
    lval = [set() for _ in range(len(LOG))]
    for i in range(rounds):
        prev, cur = (i - 1) % 2, i % 2
        c.append("RX", meas["X"][prev])
        noise("Z_ERROR", meas["X"][prev], "reset", i)
        c.append("R", meas["Z"][prev])
        noise("X_ERROR", meas["Z"][prev], "reset", i)
        noise("DEPOLARIZE1", left, "reset", i)
        c.append("TICK")
        for j, layer in enumerate(list(reversed(gates[prev])) + gates[cur]):
            pairs = [q for pair in layer for q in pair]
            c.append("CX", pairs)
            phase = ("exp", j) if j < 3 else ("con", j - 3)
            noise("DEPOLARIZE2", pairs, phase[0], i, phase[1])
            idle = sorted(set(range(2 * n2)) - set(pairs))
            if idle:
                noise("DEPOLARIZE1", idle, phase[0], i, phase[1])
            c.append("TICK")
        rec = {}
        noise("Z_ERROR", meas["X"][cur], "meas", i)
        c.append("MX", meas["X"][cur])
        for q in meas["X"][cur]:
            rec[q] = nmeas
            nmeas += 1
        noise("X_ERROR", meas["Z"][cur], "meas", i)
        c.append("M", meas["Z"][cur])
        for q in meas["Z"][cur]:
            rec[q] = nmeas
            nmeas += 1
        noise("DEPOLARIZE1", left, "meas", i)
        for pauli in ("X", "Z"):
            for g in range(n2):
                same, rest = carry((pauli, g, cur), stabs[pauli][g], pauli, cur)
                v = val[(pauli, g)]
                now = None if v is None else v ^ {rec[q] for q in same}
                if not rest:
                    if now is not None:
                        c.append("DETECTOR", [stim.target_rec(x - nmeas) for x in sorted(now)],
                                 [g, 0 if pauli == "X" else 1, i])
                    now = set()
                val[(pauli, g)] = now
        for k, v in enumerate(LOG):
            same, _ = carry(("L", k, cur), set(np.nonzero(v)[0].tolist()), basis, cur)
            lval[k] ^= {rec[q] for q in same}
        c.append("TICK")
    last = (rounds - 1) % 2
    noise("X_ERROR" if basis == "Z" else "Z_ERROR", left, "readout", rounds - 1)
    c.append("M" if basis == "Z" else "MX", left)
    data = {q: nmeas + q for q in left}
    nmeas += n2
    for g in range(n2):
        _, rest = carry((basis, g, last), stabs[basis][g], basis, last)
        if rest:
            recs = val[(basis, g)] ^ {data[q] for q in rest}
            c.append("DETECTOR", [stim.target_rec(x - nmeas) for x in sorted(recs)],
                     [g, 0 if basis == "X" else 1, rounds])
    for k, v in enumerate(LOG):
        _, rest = carry(("L", k, last), set(np.nonzero(v)[0].tolist()), basis, last)
        recs = lval[k] ^ {data[q] for q in rest}
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(x - nmeas) for x in sorted(recs)], k)
    if tags is not None:
        flat = list(c.flattened())
        assert all(flat[t[0]].name == t[1] for t in tagged), "stim fused instructions; the tags are off"
        tags.extend(tagged)
    return c


def residual_sets(typ: str, colour: int) -> list[frozenset]:
    """The three multi-qubit hook residuals of a check, as direction names: after the 2nd, 3rd and 4th CNOT
    the ancilla fault leaves the first two / first three / last two qubits (mod the check)."""
    s = SCHEDULE_3C[(typ, colour)]
    order = sorted(DIRECTIONS, key=lambda d: s[d])
    return [frozenset(order[:2]), frozenset(order[:3]), frozenset(order[4:])]


def translations(kind: str, l: int = 12, m: int = 6) -> list[tuple[int, int]]:
    """The symmetry group the anchoring uses: all l*m translations (IBM cycle), the 24 that preserve every
    check's colour (3-colour cycle: a + b = 0 mod 3), or the kernel of the morphing homomorphism (the
    translations that keep every stabilizer's contraction round; "ex": a even)."""
    if kind == "full":
        return [(a, b) for a in range(l) for b in range(m)]
    if kind == "c3":
        return [(a, b) for a in range(l) for b in range(m) if (a + b) % 3 == 0]
    if kind == "ex":
        return [(a, b) for a in range(l) for b in range(m) if a % 2 == 0]
    raise ValueError(kind)


def dims(circuit: str, l: int = 12) -> tuple[int, int]:
    """(l, m) of the group the circuit's detector sites live in."""
    return MORPH_CODES[circuit][:2] if circuit in MORPH_CODES else (l, 6)


def time_drift(circuit: str) -> tuple[int, int]:
    """The spatial translation that maps round r's schedule onto round r + 1's: none for a cycle that repeats
    itself, x for a morphing circuit (f(x) = 1 swaps which stabilizers contract; homomorphism "x")."""
    return (1, 0) if circuit in MORPH_CODES else (0, 0)


def group_kind(circuit: str) -> str:
    return "c3" if circuit == "3c" else "ex" if circuit in MORPH_CODES else "full"


def build(circuit: str, basis: str, rounds: int, l: int = 12) -> stim.Circuit:
    if circuit == "ibm":
        return ibm_cycle(basis, rounds, l)
    if circuit == "3c":
        if l != 12:
            raise ValueError("the 3-colour cycle is defined for the gross code only")
        return three_colour_cycle(basis, rounds)
    if circuit in MORPH_CODES:
        return morphing_cycle(circuit, basis, rounds)
    raise ValueError(circuit)
