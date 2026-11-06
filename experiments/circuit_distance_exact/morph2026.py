r"""morph2026 (R198 part 17): Abelian-2BGA morphing circuits of Shaw & Terhal, arXiv 2604.09797
(Sec. 3.3, App. C), as stim memory experiments in EXACTLY the conventions of the lane's
`bb_circuits.morphing_cycle` -- so the lane's exporter, engine, checker, window lemma and fold lemma apply unchanged.

    python morph2026.py validate --code c42 --basis Z --tmax 3      # every validation below, JSON lines on stdout

WHAT IS NEW HERE. 2604.09797 searched two-round homomorphism-based morphing circuits over weight-5 and weight-6
Abelian 2BGA mid-cycle codes and reports END-cycle code parameters only (Figs. 11, 25, 26): "we did not check the
circuit-level distance of the bare-ancilla or the morphing circuits" (App. C.4). Its highlighted examples are a
[[42,6,6]] end-cycle code with a degree-4 circuit and a [[30,8,5]] end-cycle code with a degree-5 circuit; the
paper gives neither their groups nor their polynomials, so `morph2026_codes.py` re-finds them. The circuit-level
distance of these circuits is bounded above by min(d_mid, d_end) and is otherwise UNKNOWN.

THE CODE (`Code2BGA`). G = Z_l x Z_m, A and B lists of exponent pairs (a, b) = x^a y^b (a cyclic group is m = 1).
Index and matrix conventions are bb_circuits.BB's: element (i, j) has index i*m + j; X-check g = X on L(g + a),
a in A, and R(g + b), b in B; Z-check g = Z on L(g - b), R(g - a) -- the paper's Eq. (7au) with G written
additively. H_X = [A | B], H_Z = [B^T | A^T].

THE SCHEDULE (`morphing_schedule`). The authors' `get_standard_morphing_specification` (github.com/Mac-Shaw/
morphing_qec_circuits, commit 1ee705f), which already handles any |A|, |B| >= 2 on Z_l x Z_m; bb_circuits ports
it for |A| = |B| = 3 only. Homomorphism f(i, j) = (cx*i + cy*j) mod 2. In each of A and B exactly one element must
take the odd-one-out value of f (for a two-element list: the element with f = 0 -- the authors' tie rule); it is
SWAPPED to the front. Depth max(|A|, |B|) per contraction round; each left qubit meets |A| + |B| - 1 right qubits,
so the connectivity is degree w - 1. Compared gate by gate with the authors' builder by `morph2026_vs_authors.py`.

THE CIRCUIT (`morphing_cycle`). bb_circuits.morphing_cycle with the code and schedule as parameters -- the same
detector derivation by stabilizer flow, the same noise locations and tags, coords (site, type, round). Regression:
on the lane's two morph codes this builder emits the lane's circuit byte for byte (`same_as_lane`).

SYMMETRY. The translations in ker f (|G|/2 of them) map the 1-round and T-round circuits to themselves; round r+1
is round r translated by any element with f = 1 (the drift). `export_circuit` writes the engine's model format with
the lane's own `export_model.orbits` / `observables_affine` checks (asserting invariance, free action, affine
observables).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import stim  # noqa: E402

import bb_circuits as bc  # noqa: E402
import dem_independent as di  # noqa: E402
import export_model as em  # noqa: E402
import morph2026_gf2 as gf2  # noqa: E402
import serialization as ser  # noqa: E402
import tools_cde as tc  # noqa: E402


class Code2BGA:
    def __init__(self, l: int, m: int, A, B):
        self.l, self.m, self.n2 = l, m, l * m
        self.A = [tuple(int(v) for v in e) for e in A]
        self.B = [tuple(int(v) for v in e) for e in B]
        if len(set(self.shift_perm(e).tobytes() for e in self.A)) != len(self.A) or \
                len(set(self.shift_perm(e).tobytes() for e in self.B)) != len(self.B):
            raise ValueError("A or B lists the same group element twice")

    def idx(self, i: int, j: int) -> int:
        return (i % self.l) * self.m + (j % self.m)

    def shift_perm(self, e) -> np.ndarray:
        """p[g] = index of g + e."""
        p = np.zeros(self.n2, dtype=int)
        for i in range(self.l):
            for j in range(self.m):
                p[self.idx(i, j)] = self.idx(i + e[0], j + e[1])
        return p

    def check_matrices(self):
        n2 = self.n2
        Am = np.zeros((n2, n2), dtype=np.uint8)
        Bm = np.zeros((n2, n2), dtype=np.uint8)
        for e in self.A:
            Am[np.arange(n2), self.shift_perm(e)] ^= 1
        for e in self.B:
            Bm[np.arange(n2), self.shift_perm(e)] ^= 1
        HX, HZ = np.hstack([Am, Bm]), np.hstack([Bm.T, Am.T])
        assert not ((HX.astype(int) @ HZ.T.astype(int)) % 2).any()
        return HX, HZ

    def k(self) -> int:
        HX, HZ = self.check_matrices()
        return 2 * self.n2 - bc.gf2_rank(HX) - bc.gf2_rank(HZ)

    def logicals(self, H_same, H_other) -> np.ndarray:
        """bb_circuits.BB.logicals with k computed instead of fixed at 12 (same algorithm, so on the lane's codes the
        same 12 vectors in the same order)."""
        k = self.k()
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
            for kk, pc in enumerate(pivots):
                if m[kk, f]:
                    v[pc] = 1
            basis.append(v)
        base = bc.gf2_rank(H_same)
        cur, out = H_same.copy(), []
        for v in basis:
            trial = np.vstack([cur, v])
            if bc.gf2_rank(trial) > bc.gf2_rank(cur):
                cur = trial
                out.append(v)
            if len(out) == k:
                break
        assert len(out) == k and bc.gf2_rank(cur) == base + k
        return np.array(out, dtype=np.uint8)

    def rows(self):
        HX, HZ = self.check_matrices()
        to_int = lambda r: sum(1 << q for q in np.nonzero(r)[0].tolist())  # noqa: E731
        return [to_int(r) for r in HX], [to_int(r) for r in HZ]

    def distance(self, half: int = 3):
        """(d_X, d_Z) exactly when <= 2*half, else None in that slot."""
        hx, hz = self.rows()
        n = 2 * self.n2
        return gf2.min_logical(hx, hz, n, half), gf2.min_logical(hz, hx, n, half)


# ---------------------------------------------------------------------------------------------- the schedule

def hom_fn(hom):
    cx, cy = hom
    return lambda i, j: (cx * i + cy * j) % 2


def morphing_schedule(code: Code2BGA, hom=(1, 0)):
    """The authors' standard morphing circuit for any |A|, |B| >= 2 (see the module docstring). Returns
    (gates[round][layer] = [(control, target)], meas[pauli][round] = [qubits], A, B after the swap)."""
    l, m, n2 = code.l, code.m, code.n2
    if (hom[0] % 2 and l % 2) or (hom[1] % 2 and m % 2) or not (hom[0] % 2 or hom[1] % 2):
        raise ValueError("f is not a surjective homomorphism Z_l x Z_m -> Z_2")
    f = hom_fn(hom)
    lists = []
    for terms in (code.A, code.B):
        c = list(terms)
        vals = [f(*e) for e in c]
        if len(c) >= 2 and vals.count(0) == 1 and vals.count(1) == len(c) - 1:
            special = 0
        elif len(c) >= 2 and vals.count(1) == 1 and vals.count(0) == len(c) - 1:
            special = 1
        else:
            raise ValueError("the homomorphism does not single out one element of A and of B")
        s = vals.index(special)
        c[0], c[s] = c[s], c[0]
        lists.append(c)
    A, B = lists
    L = max(len(A), len(B))
    R = lambda i, j: n2 + (i % l) * m + (j % m)  # noqa: E731
    gates = [[[] for _ in range(L)] for _ in range(2)]
    meas = {"X": [[], []], "Z": [[], []]}
    for i in range(l):
        for j in range(m):
            g, rg = i * m + j, f(i, j)
            rh = (rg + 1) % 2
            right_g = [R(i - A[0][0] + b[0], j - A[0][1] + b[1]) for b in B]
            right_h = [R(i - a[0] + B[0][0], j - a[1] + B[0][1]) for a in A]
            for k in range(1, L):  # bb_circuits' loop order, so the lane's codes give the lane's exact layers
                if k < len(B):
                    gates[rg][k - 1].append((g, right_g[k]))
                if k < len(A):
                    gates[rh][k - 1].append((right_h[k], g))
            gates[rg][L - 1].append((right_g[0], g))
            gates[rh][L - 1].append((g, right_h[0]))
            meas["X"][rg].append(right_g[0])
            meas["Z"][rh].append(right_h[0])
    for r in range(2):
        for layer in gates[r]:
            used = [q for pair in layer for q in pair]
            assert layer and len(used) == len(set(used)), "a qubit is used twice in one layer (or a layer is empty)"
        assert sorted(meas["X"][r] + meas["Z"][r]) == list(range(n2, 2 * n2)), "every right qubit is measured"
    return gates, meas, A, B


def connectivity_degree(gates, n: int) -> int:
    nb = [set() for _ in range(n)]
    for rnd in gates:
        for layer in rnd:
            for c, t in layer:
                nb[c].add(t)
                nb[t].add(c)
    return max(len(s) for s in nb)


def end_cycle_code(code: Code2BGA, gates, meas, r: int):
    """(H_X rows, H_Z rows) of the end-cycle code of round r on its data qubits (bit t = the t-th unmeasured qubit;
    the left qubits for the standard schedule): the rests of the stabilizers that EXPAND in round r
    (bb_circuits._contract + the measured part removed)."""
    HX, HZ = code.check_matrices()
    data = sorted(set(range(2 * code.n2)) - set(meas["X"][r]) - set(meas["Z"][r]))
    pos = {q: t for t, q in enumerate(data)}
    out = {}
    for pauli, H in (("X", HX), ("Z", HZ)):
        rows = []
        for g in range(code.n2):
            op = bc._contract(set(np.nonzero(H[g])[0].tolist()), pauli, gates[r])
            other = "Z" if pauli == "X" else "X"
            assert not op & set(meas[other][r])
            rest = op - set(meas[pauli][r])
            if rest:
                assert rest <= set(pos)
                rows.append(sum(1 << pos[q] for q in rest))
        out[pauli] = rows
    return out["X"], out["Z"]


# ---------------------------------------------------------------------------------------------- the circuit

def morphing_cycle(code: Code2BGA, basis: str, rounds: int, hom=(1, 0), p: float = 1e-3,
                   tags: list | None = None) -> stim.Circuit:
    """The standard schedule's memory circuit (= bb_circuits.morphing_cycle on the lane's codes, byte for byte)."""
    gates, meas, _, _ = morphing_schedule(code, hom)
    return morphing_cycle_from(code, gates, meas, basis, rounds, p, tags)


def data_qubits(code: Code2BGA, meas, r: int) -> list[int]:
    """The end-cycle data qubits of round r: every qubit round r does not measure (the left qubits for the standard
    schedule; half left, half right for the ZX-dual-symmetric tree circuits of Def. 10)."""
    return sorted(set(range(2 * code.n2)) - set(meas["X"][r]) - set(meas["Z"][r]))


def morphing_cycle_from(code: Code2BGA, gates, meas, basis: str, rounds: int, p: float = 1e-3,
                        tags: list | None = None) -> stim.Circuit:
    """bb_circuits.morphing_cycle with the code and the two contraction rounds (gates, meas) as parameters. Its
    docstring states the detector derivation; the only generalisations are the number of layers and the data
    qubits of round r, data_qubits(r), in place of the left block (identical for the standard schedule): the
    memory prepares round 1's data qubits (round 0's expansion starts from end-cycle code 1), the reset step idles
    data_qubits(prev), the measurement step data_qubits(cur), and the readout measures data_qubits(last)."""
    n2 = code.n2
    L = len(gates[0])
    HX, HZ = code.check_matrices()
    stabs = {"X": [set(np.nonzero(HX[g])[0].tolist()) for g in range(n2)],
             "Z": [set(np.nonzero(HZ[g])[0].tolist()) for g in range(n2)]}
    LOG = code.logicals(HZ, HX) if basis == "Z" else code.logicals(HX, HZ)
    data_r = {r: data_qubits(code, meas, r) for r in (0, 1)}
    assert len(data_r[0]) == len(data_r[1]) == n2, "each round measures half of the qubits"
    split = {}

    def carry(key, support, pauli, r):
        if key not in split:
            op = bc._contract(support, pauli, gates[r])
            other = "Z" if pauli == "X" else "X"
            assert not op & set(meas[other][r]), "a stabilizer or logical anticommutes with a measurement"
            same = op & set(meas[pauli][r])
            rest = op - same
            assert rest <= set(data_r[r])
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

    c.append("R" if basis == "Z" else "RX", data_r[1])
    noise("X_ERROR" if basis == "Z" else "Z_ERROR", data_r[1], "prep", 0)
    c.append("TICK")
    nmeas = 0
    val = {(pauli, g): (set() if pauli == basis or not carry((pauli, g, 1), stabs[pauli][g], pauli, 1)[1] else None)
           for pauli in ("X", "Z") for g in range(n2)}
    lval = [set() for _ in range(len(LOG))]
    for i in range(rounds):
        prev, cur = (i - 1) % 2, i % 2
        c.append("RX", meas["X"][prev])
        noise("Z_ERROR", meas["X"][prev], "reset", i)
        c.append("R", meas["Z"][prev])
        noise("X_ERROR", meas["Z"][prev], "reset", i)
        noise("DEPOLARIZE1", data_r[prev], "reset", i)
        c.append("TICK")
        for j, layer in enumerate(list(reversed(gates[prev])) + gates[cur]):
            pairs = [q for pair in layer for q in pair]
            c.append("CX", pairs)
            phase = ("exp", j) if j < L else ("con", j - L)
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
        noise("DEPOLARIZE1", data_r[cur], "meas", i)
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
    noise("X_ERROR" if basis == "Z" else "Z_ERROR", data_r[last], "readout", rounds - 1)
    c.append("M" if basis == "Z" else "MX", data_r[last])
    data = {q: nmeas + t for t, q in enumerate(data_r[last])}
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


# ---------------------------------------------------------------------------------------------- the targets

# morph2026_targets.json: name -> {"kind": "std", "l", "m", "A", "B", "hom"}  (standard schedule, morph2026_codes.py)
#                   or {"kind": "tree", "N", "A", "B", "sched", "case"} (Def. 10 tree circuit, morph2026_trees.py)
TARGETS: dict[str, dict] = {}


def bc_el(t):
    return (t[1], 0) if t[0] == "x" else (0, t[1])


def lane_code(kind: str) -> Code2BGA:
    l, m, a, b = bc.MORPH_CODES[kind]
    return Code2BGA(l, m, [bc_el(t) for t in a], [bc_el(t) for t in b])


def tree_schedule(N: int, A, B, sched, case: str):
    """The two contraction rounds of a Def. 10 tree circuit on Z_N (morph2026_trees.Circuit2 builds F_0 and checks it
    parallelisable, proper, distinct roots); round 1 = round 0 translated by 1. Layers are sorted gate lists."""
    import morph2026_trees as TS
    c2 = TS.Circuit2(N, A, B, [[tuple(p) for p in layer] for layer in sched], case)
    built = c2.build()
    if built is None or not c2.proper(built[0]):
        raise ValueError("the tree circuit is not parallelisable / proper / root-distinct")
    layers, xroots, zroots = built
    sh = lambda q: (q + 1) % N if q < N else N + (q - N + 1) % N  # noqa: E731
    g0 = [sorted(layer) for layer in layers]
    g1 = [sorted((sh(c), sh(t)) for c, t in layer) for layer in layers]
    meas = {"X": [sorted(xroots), sorted(sh(q) for q in xroots)], "Z": [sorted(zroots), sorted(sh(q) for q in zroots)]}
    for r, gr in enumerate((g0, g1)):
        for layer in gr:
            used = [q for pair in layer for q in pair]
            assert layer and len(used) == len(set(used))
    return [g0, g1], meas


def spec(name: str) -> dict:
    """{"code", "gates", "meas", "hom"} of a target."""
    t = TARGETS[name]
    if t["kind"] == "std":
        code = Code2BGA(t["l"], t["m"], t["A"], t["B"])
        gates, meas, _, _ = morphing_schedule(code, tuple(t["hom"]))
        return dict(name=name, code=code, gates=gates, meas=meas, hom=tuple(t["hom"]))
    code = Code2BGA(t["N"], 1, [(a, 0) for a in t["A"]], [(b, 0) for b in t["B"]])
    gates, meas = tree_schedule(t["N"], t["A"], t["B"], t["sched"], t["case"])
    return dict(name=name, code=code, gates=gates, meas=meas, hom=(1, 0))


def build(sp: dict, basis: str, rounds: int, tags: list | None = None) -> stim.Circuit:
    return morphing_cycle_from(sp["code"], sp["gates"], sp["meas"], basis, rounds, tags=tags)


def target(name: str):
    sp = spec(name)
    return sp["code"], sp["hom"]


def kernel_shifts(code: Code2BGA, hom) -> list[tuple[int, int]]:
    f = hom_fn(hom)
    return [(a, b) for a in range(code.l) for b in range(code.m) if f(a, b) == 0]


def drift(hom) -> tuple[int, int]:
    return (1, 0) if hom[0] % 2 else (0, 1)


# ---------------------------------------------------------------------------------------------- export / checks

def export_circuit(circuit, basis: str, dims, shifts, out: pathlib.Path, *, project: bool = True,
                   time_order: bool = False) -> dict:
    """export_model.export with the circuit and the symmetry passed in (the lane's own orbit / affine checks)."""
    dem, mech, coords = em.mechanisms(circuit)
    n_full = len(mech)
    if project:
        mech, coords = em.project_same_type(mech, coords, basis)
    if time_order:
        mech = [m for _, m in sorted(enumerate(mech), key=lambda t: (min(int(coords[d][2]) for d in t[1][0]), t[0]))]
    l, m = dims
    orb, reps = em.orbits(mech, coords, shifts, l, m)
    assert em.observables_affine(mech, coords, shifts, l, m), "observables are not affine under a declared translation"
    # ASCII, CRLF on every platform: the archived models' bytes, which run records bind by sha256 (serialization.py)
    with ser.open_model(out) as f:
        f.write(f"{len(coords)} {len(mech)} {dem.num_observables} {len(reps)} {l} {m}\n")
        for (dets, obs), o in zip(mech, orb):
            mask = sum(1 << x for x in obs)
            f.write(f"{len(dets)} {' '.join(map(str, dets))} {mask} {o}\n")
        f.write(" ".join(map(str, reps)) + "\n")
        for d in range(len(coords)):
            k, typ, r = coords[d]
            f.write(f"{int(k)} {int(typ)} {int(r)}\n")
    return dict(projected=project, full_mechanisms=n_full, mechanisms=len(mech), detectors=len(coords),
                orbits=len(reps), group=len(shifts), time_order=time_order)


def lift(circuit, basis: str, projected_ids: list[int]) -> list[int] | None:
    """tools_cde.lift with the circuit passed in (same meet-in-the-middle choice of one full preimage per projected
    mechanism; re-verified on the full DEM)."""
    import itertools
    _, mech, coords = em.mechanisms(circuit)
    keep_type = 1 if basis == "Z" else 0
    keep = sorted(d for d, v in coords.items() if int(v[1]) == keep_type)
    new_id = {d: i for i, d in enumerate(keep)}
    groups, order = {}, []
    for j, (dets, obs) in enumerate(mech):
        pd = tuple(sorted(new_id[d] for d in dets if d in new_id))
        if not pd and not obs:
            continue
        key = (pd, obs)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(j)
    cand = [groups[order[r]] for r in projected_ids]
    syn = {j: frozenset(mech[j][0]) for g in cand for j in g}
    half = len(cand) // 2

    def sums(gs):
        for choice in itertools.product(*gs):
            s = frozenset()
            for j in choice:
                s = s ^ syn[j]
            yield s, choice
    table: dict = {}
    for s, choice in sums(cand[:half]):
        table.setdefault(s, []).append(choice)
    for s, right in sums(cand[half:]):
        for left in table.get(s, ()):
            choice = left + right
            if len(set(choice)) == len(choice) and tc._undetectable_logical(mech, choice):
                return list(choice)
    return None


def _cx_layer(layer, x: int, z: int):
    for c, t in layer:
        if x >> c & 1:
            x ^= 1 << t
        if z >> t & 1:
            z ^= 1 << c
    return x, z


def fault_table(sp: dict, basis: str, T: int):
    """fold_lemma.fault_table for this builder (L layers instead of 3; logic unchanged)."""
    gates = sp["gates"]
    tags: list = []
    circuit = build(sp, basis, T, tags=tags)
    comps, coords, nd, _ = di.pauli_components(circuit)
    keep_type = 1 if basis == "Z" else 0
    pos = {d: i for i, d in enumerate(d for d in range(nd) if int(coords[d][1]) == keep_type)}
    tag_of = {t[0]: t for t in tags}
    rows = []
    for idx, pauli, mask in comps:
        _, _, phase, rnd, layer = tag_of[idx]
        prev, cur = (rnd - 1) % 2, rnd % 2
        expansion = list(reversed(gates[prev]))
        if phase in ("prep", "reset"):
            seq = expansion
        elif phase == "exp":
            seq = expansion[layer + 1:]
        elif phase == "con":
            seq = list(reversed(gates[cur][:layer + 1]))
        else:
            seq = list(reversed(gates[cur]))
        x = sum(1 << q for q, pp in pauli.items() if pp in "XY")
        z = sum(1 << q for q, pp in pauli.items() if pp in "ZY")
        for layer_pairs in seq:
            x, z = _cx_layer(layer_pairs, x, z)
        det = sum(1 << pos[d] for d in pos if mask >> d & 1)
        rows.append((det, mask >> nd, x if basis == "Z" else z, rnd, phase))
    return rows, len(pos)


def fold_check(sp: dict, basis: str, T: int) -> dict:
    """fold_lemma.fold_check for this builder: d_proj(1) <= d_proj(T) when it holds."""
    one, nd1 = fault_table(sp, basis, 1)
    table, conflicts = {}, 0
    for det, obs, key, _, _ in one:
        if key in table and table[key] != (det, obs):
            conflicts += 1
        table.setdefault(key, (det, obs))
    rows, _ = fault_table(sp, basis, T)
    pairs, missing = [], 0
    for det, obs, key, _, _ in rows:
        if not det and not obs:
            continue
        hit = table.get(key)
        if hit is None:
            missing += 1
            continue
        pairs.append((det, hit[0] | (hit[1] ^ obs) << nd1))
    consistent = tc._affine_consistent(pairs)
    return dict(T=T, components=len(rows), mapped=len(pairs), missing=missing, conflicts=conflicts,
                consistent=consistent, holds=missing == 0 and conflicts == 0 and consistent)


def expected_gap(code: Code2BGA, basis: str) -> int:
    """The redundancy meta-checks of the other type: |G| - rank(H_other)."""
    HX, HZ = code.check_matrices()
    return code.n2 - bc.gf2_rank(HX if basis == "Z" else HZ)


def validate(sp: dict, basis: str, tmax: int, work: pathlib.Path, emit=print) -> dict:
    code, gates, hom = sp["code"], sp["gates"], sp["hom"]
    shifts = kernel_shifts(code, hom)
    dims = (code.l, code.m)
    ok = True
    for T in range(1, tmax + 1):
        c = build(sp, basis, T)
        dem = c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)  # raises on a
        gap = tc.deterministic_gap(c)                                                           # non-deterministic
        cmp = di.compare(c)                                                                     # detector
        info = export_circuit(c, basis, dims, shifts, work / f"m_{basis}{T}.txt", time_order=True)
        row = dict(T=T, basis=basis, detectors=c.num_detectors, dem_errors=dem.num_errors,
                   gap=len(gap["gap"]), gap_expected=expected_gap(code, basis),
                   gap_ok=len(gap["gap"]) == expected_gap(code, basis), dem_independent_equal=cmp["equal"],
                   **{"export_" + k: v for k, v in info.items()})
        if T >= 2:
            row["fold"] = fold_check(sp, basis, T)
            row["window"] = tc.window_preconditions(work / f"m_{basis}{T - 1}.txt", work / f"m_{basis}{T}.txt",
                                                    drift(hom))
        ok &= row["gap_ok"] and row["dem_independent_equal"] and (T < 2 or row["fold"]["holds"]) and \
            (T < 2 or row["window"]["holds"])
        emit(json.dumps(row))
    return dict(all_ok=bool(ok), degree=connectivity_degree(gates, 2 * code.n2), layers=len(gates[0]))


def same_as_lane(kind: str, basis: str, T: int) -> bool:
    """Regression: on the lane's morph codes this builder reproduces bb_circuits.morphing_cycle exactly."""
    hom = (1, 0)
    return str(morphing_cycle(lane_code(kind), basis, T, hom)) == str(bc.morphing_cycle(kind, basis, T, "x"))


def load_targets(path: pathlib.Path = HERE / "morph2026_targets.json") -> None:
    if path.exists():
        TARGETS.update(json.loads(path.read_text()))


load_targets()


def _sha(p: pathlib.Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def full_undetectable_logical(circuit, ids) -> bool:
    """ids (full-DEM mechanism indices) form an undetectable logical error of the circuit's full DEM."""
    _, mech, _ = em.mechanisms(circuit)
    return bool(ids) and tc._undetectable_logical(mech, list(ids))


def model(sp: dict, basis: str, T: int, work: pathlib.Path, time_order: bool):
    """(circuit, exported projected model path, export info) -- the ex group: the kernel of f = (1, 0)."""
    if tuple(sp["hom"]) != (1, 0):
        raise ValueError("the checker's 'ex' group is the kernel of f(i, j) = i mod 2 only")
    c = build(sp, basis, T)
    path = work / f"{sp['name']}_{basis}{T}{'_t' if time_order else ''}.txt"
    info = export_circuit(c, basis, (sp["code"].l, sp["code"].m), kernel_shifts(sp["code"], sp["hom"]), path,
                          time_order=time_order)
    return c, path, info


def search(sp: dict, basis: str, T: int, wmax: int, exe, work, threads: int, maxround: int = -1) -> dict:
    """cdist on the projected T-round model, NONE <= w per weight; a FOUND witness is lifted to the full DEM and
    re-verified there, and every one of its single deletions must fail (it is minimal, so it is the distance)."""
    import run_lane
    c, path, info = model(sp, basis, T, work, maxround >= 0)
    t0 = time.time()
    r = tc.run(exe["cdist"], path, 1, wmax, 20, *([maxround] if maxround >= 0 else []), threads=threads)
    got = run_lane.parse_search(r.stdout)
    rec = dict(code=sp["name"], basis=basis, rounds=T, maxround=maxround, wmax=wmax, **info,
               model_sha256=_sha(path), **got, seconds=round(time.time() - t0, 2))
    if got["witness"]:
        full = lift(c, basis, got["witness"])
        rec.update(full=full, verified=bool(full) and full_undetectable_logical(c, full),
                   every_deletion_fails=bool(full) and not any(
                       full_undetectable_logical(c, full[:i] + full[i + 1:]) for i in range(len(full))))
    return rec


def certify(sp: dict, basis: str, T: int, w: int, exe, work, threads: int, maxround: int = -1) -> dict:
    """cdist_prove writes a certificate of "no solution of weight <= w" (scoped to rounds <= maxround when >= 0);
    the independent ccheck re-derives the symmetry (group ex) and every node."""
    _, path, info = model(sp, basis, T, work, maxround >= 0)
    proof = work / f"proof_{sp['name']}_{basis}{T}_w{w}.bin"
    shift = [maxround] if maxround >= 0 else []
    t0 = time.time()
    p = tc.run(exe["cdist_prove"], path, w, proof, *shift, threads=threads)
    rec = dict(code=sp["name"], basis=basis, rounds=T, maxround=maxround, w=w, **info, model_sha256=_sha(path),
               prover=p.stdout.strip(), prover_exit=p.returncode, prover_s=round(time.time() - t0, 2))
    if p.returncode == 0:
        t1 = time.time()
        k = tc.run(exe["ccheck"], path, proof, "ex", threads=threads)
        rec.update(checker=k.stdout.strip(), checker_exit=k.returncode, checker_s=round(time.time() - t1, 2),
                   proof_sha256=_sha(proof), proof_bytes=proof.stat().st_size)
    return rec


def end_cycle_distance(sp: dict) -> int:
    """min over both rounds and both types of the end-cycle code's distance (the circuit can do no better)."""
    code, gates, meas = sp["code"], sp["gates"], sp["meas"]
    out = []
    for r in (0, 1):
        hx, hz = end_cycle_code(code, gates, meas, r)
        out += [gf2.min_logical(hx, hz, code.n2, 4), gf2.min_logical(hz, hx, code.n2, 4)]
    return min(d for d in out if d)


def campaign(names, threads: int, work: pathlib.Path, emit=print) -> None:
    """Every record the morph2026 claims rest on, per target and basis: the T = 1 search up to the end-cycle
    distance (its FOUND weight d, with a lifted, verified, minimal witness, is the circuit distance at T = 1);
    searches at T = 2..w+2 (w = d - 1, so a witness of the same weight in each); unscoped certificates of
    "no solution of weight <= w" at T = 1, 2; time-shift certificates (anchors in rounds <= 1) at T = 1..w+2; and
    the space-time window preconditions on every pair (T, T + 1), T = 1..w+2 -- the morph72 route to every T."""
    exe = tc.build_c(work)
    for name in names:
        sp = spec(name)
        d_end = end_cycle_distance(sp)
        emit(dict(kind="info", code=name, d_end=d_end, degree=connectivity_degree(sp["gates"], 2 * sp["code"].n2),
                  layers=len(sp["gates"][0]), n_end=sp["code"].n2, k=sp["code"].k()))
        for basis in "ZX":
            s1 = search(sp, basis, 1, d_end, exe, work, threads)
            emit(dict(kind="search", **s1))
            if not (s1["found_weight"] and s1.get("verified") and s1.get("every_deletion_fails")):
                continue
            w = s1["found_weight"] - 1
            for T in range(2, w + 3):
                emit(dict(kind="search", **search(sp, basis, T, w + 1, exe, work, threads)))
            for T in (1, 2):
                emit(dict(kind="cert", **certify(sp, basis, T, w, exe, work, threads)))
            for T in range(1, w + 3):
                emit(dict(kind="shift_cert", **certify(sp, basis, T, w, exe, work, threads, maxround=1)))
            for T in range(1, w + 3):
                _, m0, _ = model(sp, basis, T, work, False)
                _, m1, _ = model(sp, basis, T + 1, work, False)
                emit(dict(kind="window", code=name, basis=basis, pair=[T, T + 1],
                          **tc.window_preconditions(m0, m1, drift(sp["hom"]))))


def main():
    import tempfile
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("validate", "regress", "info", "search", "certify", "window", "fold",
                                    "campaign"))
    ap.add_argument("--codes", nargs="*", help="campaign: the targets (default: every one)")
    ap.add_argument("--code")
    ap.add_argument("--basis", default="Z")
    ap.add_argument("--tmax", type=int, default=3)
    ap.add_argument("--tmin", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--w", type=int, default=5)
    ap.add_argument("--maxround", type=int, default=-1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--work")
    a = ap.parse_args()
    work = pathlib.Path(a.work) if a.work else pathlib.Path(tempfile.mkdtemp(prefix="morph2026_"))
    work.mkdir(parents=True, exist_ok=True)
    if a.cmd == "campaign":
        campaign(a.codes or sorted(TARGETS), a.threads, work, emit=lambda r: print(json.dumps(r), flush=True))
        return
    if a.cmd in ("search", "certify"):
        sp = spec(a.code)
        exe = tc.build_c(work)
        # a time-shift run (--maxround >= 0) sweeps T = tmin..tmax; otherwise the single --rounds model
        for T in range(a.tmin, a.tmax + 1) if a.maxround >= 0 else [a.rounds]:
            f = search if a.cmd == "search" else certify
            print(json.dumps(f(sp, a.basis, T, a.w, exe, work, a.threads, a.maxround)), flush=True)
        return
    if a.cmd == "window":
        sp = spec(a.code)
        for T in range(a.tmin, a.tmax + 1):
            _, m0, _ = model(sp, a.basis, T, work, False)
            _, m1, _ = model(sp, a.basis, T + 1, work, False)
            print(json.dumps(dict(code=a.code, basis=a.basis, pair=[T, T + 1],
                                  **tc.window_preconditions(m0, m1, drift(sp["hom"])))), flush=True)
        return
    if a.cmd == "fold":
        sp = spec(a.code)
        for T in range(max(2, a.tmin), a.tmax + 1):
            print(json.dumps(dict(code=a.code, basis=a.basis, **fold_check(sp, a.basis, T))), flush=True)
        return
    if a.cmd == "regress":
        for kind in bc.MORPH_CODES:
            for basis in "ZX":
                for T in (1, 2):
                    print(json.dumps(dict(kind=kind, basis=basis, T=T, same_as_lane=same_as_lane(kind, basis, T))))
        return
    sp = spec(a.code)
    code, gates, meas = sp["code"], sp["gates"], sp["meas"]
    if a.cmd == "info":
        d_end = []
        for r in (0, 1):
            hx, hz = end_cycle_code(code, gates, meas, r)
            d_end.append([gf2.min_logical(hx, hz, code.n2, 4), gf2.min_logical(hz, hx, code.n2, 4)])
        print(json.dumps(dict(code=a.code, n_mid=2 * code.n2, k=code.k(), d_mid=code.distance(3), n_end=code.n2,
                              d_end_rounds=d_end, degree=connectivity_degree(gates, 2 * code.n2),
                              layers=len(gates[0]))))
        return
    print(json.dumps(dict(code=a.code, **validate(sp, a.basis, a.tmax, work))))


if __name__ == "__main__":
    main()
