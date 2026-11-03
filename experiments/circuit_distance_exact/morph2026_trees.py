r"""General two-round purely contracting homomorphism-based morphing circuits (2604.09797, Sec. 3.2-3.3, Def. 10,
App. C.3) on a cyclic group Z_N, by brute force over contraction trees -- the search the paper ran, which is wider
than the standard schedule `morph2026.morphing_schedule` ports.

    python morph2026_trees.py --N 30 --w 6 --k 8 --d 5 [--dmid 5]

THE CLASS (Def. 10 for J = 2, H = Z_2, phi(g) = g mod 2). F_0 contains one rooted, layered contraction tree T for
s(X, 0), translated to every s(X, g) with g even (Def. 10a: F_0 is invariant under the even translations, and
F_1 = F_0 translated by 1). The Z-trees are ZX duals (Def. 10b): SWAP_ZX L(x) <-> R(-x) with Hadamards maps
s(X, g) to s(Z, -g) and reverses every CNOT; the dual of F_0's X-trees lies either in F_0 ("same": Z-stabilizers
h even contract in round 0) or in F_1 ("cross": round 0 holds the duals of F_1's X-trees, h odd).
A TREE is a contraction schedule on the w support qubits: in each layer a set of disjoint (parent -> child) pairs
of still-present qubits, the child leaving the support (an X-stabilizer's CNOT is parent -> child, the dual
Z-tree's is child -> parent); after D layers one qubit (the root, measured) remains. Every depth-D schedule is
enumerated (5,760 for w = 6, D = 3).
CONDITIONS (Sec. 3.2), on the merged layers of F_0: PARALLELISABLE -- in each layer two gates sharing a qubit are
identical; PROPER -- a gate in layer l that is not one of stabilizer s's own layer-l gates must not change s's
support (for an X-type support S: its control is not in S; Z-type: its target is not in S), checked for one X-
and one Z-representative (the gate set is invariant under even translations); all roots distinct. Then the
end-cycle code of round 0 (the expanding stabilizers carried through F_0, measured qubits removed) and the
connectivity degree of F_0 + F_1. JSON lines: every circuit whose end-cycle code is [[N, k, >= d]], then a summary.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import morph2026_gf2 as gf2  # noqa: E402
import morph2026_codes as S  # noqa: E402


def schedules(w: int, D: int):
    """Every depth-D contraction schedule on vertices 0..w-1: list of layers, each a tuple of (parent, child)."""
    out = []

    def rec(present: tuple, layers: list):
        if len(layers) == D:
            if len(present) == 1:
                out.append(tuple(layers))
            return
        rest_layers = D - len(layers)
        # removing at most floor(|S|/2) per layer: need |present| <= 2**rest_layers
        if len(present) > 2 ** rest_layers:
            return
        pres = list(present)
        # all ordered matchings (nonempty) on pres
        def matchings(avail, acc):
            if not avail:
                yield list(acc)
                return
            first, others = avail[0], avail[1:]
            yield from matchings(others, acc)  # first unmatched
            for i, v in enumerate(others):
                rem = others[:i] + others[i + 1:]
                yield from matchings(rem, acc + [(first, v)])
                yield from matchings(rem, acc + [(v, first)])
        for mt in matchings(pres, []):
            if not mt:
                continue
            removed = {c for _, c in mt}
            rec(tuple(v for v in present if v not in removed), layers + [tuple(mt)])

    rec(tuple(range(w)), [])
    return out


class Circuit2:
    """F_0 of a tree-based circuit on Z_N: merged layers of (control, target) and the measured qubits."""

    def __init__(self, N: int, A, B, sched, case: str):
        self.N = N
        self.verts = [("L", a) for a in A] + [("R", b) for b in B]
        self.sched, self.case = sched, case
        self.root = next(iter(set(range(len(self.verts))) - {c for layer in sched for _, c in layer}))

    def q(self, v: int, g: int) -> int:
        side, off = self.verts[v]
        return (off + g) % self.N if side == "L" else self.N + (off + g) % self.N

    def sigma(self, q: int) -> int:
        N = self.N
        return N + (-q) % N if q < N else (-(q - N)) % N

    def build(self):
        """(layers of gates or None if not parallelisable, x_roots, z_roots, own gates for the representatives)."""
        N = self.N
        xs = range(0, N, 2)
        zs = range(0, N, 2) if self.case == "same" else range(1, N, 2)
        layers = []
        for layer in self.sched:
            used: dict[int, tuple] = {}
            gates = set()
            for g in xs:
                for p, c in layer:
                    gates.add((self.q(p, g), self.q(c, g)))
            for g in zs:
                for p, c in layer:
                    gates.add((self.sigma(self.q(c, g)), self.sigma(self.q(p, g))))
            for gt in gates:
                for qq in gt:
                    if used.get(qq, gt) != gt:
                        return None
                    used[qq] = gt
            layers.append(gates)
        xroots = [self.q(self.root, g) for g in xs]
        zroots = [self.sigma(self.q(self.root, g)) for g in zs]
        if len(set(xroots) | set(zroots)) != len(xroots) + len(zroots):
            return None
        return layers, xroots, zroots

    def proper(self, layers) -> bool:
        zrep = 0 if self.case == "same" else 1
        for kind, g in (("X", 0), ("Z", zrep)):
            sup = {self.q(v, g) if kind == "X" else self.sigma(self.q(v, g)) for v in range(len(self.verts))}
            for li, layer in enumerate(self.sched):
                own = set()
                for p, c in layer:
                    if kind == "X":
                        own.add((self.q(p, g), self.q(c, g)))
                    else:
                        own.add((self.sigma(self.q(c, g)), self.sigma(self.q(p, g))))
                for gt in layers[li]:
                    if gt in own:
                        continue
                    if (kind == "X" and gt[0] in sup) or (kind == "Z" and gt[1] in sup):
                        return False
                for gt in own:
                    sup.discard(gt[1] if kind == "X" else gt[0])
            assert len(sup) == 1
        return True


def stabilizers(N: int, A, B):
    X = [frozenset([(a + g) % N for a in A] + [N + (b + g) % N for b in B]) for g in range(N)]
    Z = [frozenset([(g - b) % N for b in B] + [N + (g - a) % N for a in A]) for g in range(N)]
    return X, Z


def carry(sup, pauli, layers):
    op = set(sup)
    for layer in layers:
        for c, t in layer:
            if pauli == "X" and c in op:
                op ^= {t}
            elif pauli == "Z" and t in op:
                op ^= {c}
    return op


def end_cycle(N, A, B, layers, xroots, zroots):
    """End-cycle check rows on the unmeasured qubits (re-indexed 0..N-1), or None if inconsistent."""
    X, Z = stabilizers(N, A, B)
    meas = {"X": set(xroots), "Z": set(zroots)}
    allm = meas["X"] | meas["Z"]
    data = [q for q in range(2 * N) if q not in allm]
    pos = {q: i for i, q in enumerate(data)}
    rows = {"X": [], "Z": []}
    contracted = {"X": 0, "Z": 0}
    for pauli, stabs in (("X", X), ("Z", Z)):
        other = "Z" if pauli == "X" else "X"
        for s in stabs:
            op = carry(s, pauli, layers)
            if op & meas[other]:
                return None
            rest = op - meas[pauli]
            if not rest:
                contracted[pauli] += 1
                continue
            rows[pauli].append(sum(1 << pos[q] for q in rest))
    if contracted["X"] != N // 2 or contracted["Z"] != N // 2:
        return None
    return rows["X"], rows["Z"], data


def degree(N, layers):
    nb = [set() for _ in range(2 * N)]
    sh = lambda q: (q + 1) % N if q < N else N + (q - N + 1) % N  # noqa: E731
    for layer in layers:
        for c, t in layer:
            for a, b in ((c, t), (sh(c), sh(t))):
                nb[a].add(b)
                nb[b].add(a)
    return max(len(s) for s in nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--dmid", type=int, default=None, help="skip mid-cycle codes of smaller distance (default d)")
    ap.add_argument("--D", type=int, default=None)
    ap.add_argument("--code", type=int, nargs="*", help="restrict to one class: A... -1 B...")
    a = ap.parse_args()
    N, w, kt, dt = a.N, a.w, a.k, a.d
    D = a.D or math.ceil(math.log2(w))
    dmid = a.dmid if a.dmid is not None else dt
    t0 = time.time()
    scheds = schedules(w, D)
    classes = []
    if a.code:
        i = a.code.index(-1)
        classes = [(tuple(a.code[:i]), tuple(a.code[i + 1:]))]
    else:
        seen = set()
        for nb in range(2, w // 2 + 1):
            na = w - nb
            for Arest in itertools.combinations(range(1, N), na - 1):
                for Brest in itertools.combinations(range(1, N), nb - 1):
                    key = S.canonical(N, (0,) + Arest, (0,) + Brest)
                    if key in seen:
                        continue
                    seen.add(key)
                    if S.k_of(N, *key) == kt:
                        classes.append(key)
    half = (max(dt, dmid) + 1) // 2
    n_valid = 0
    hits = []
    for A, B in classes:
        import morph2026 as M
        code = M.Code2BGA(N, 1, [(x, 0) for x in A], [(x, 0) for x in B])
        dm = code.distance(half)
        if any(x is not None and x < dmid for x in dm):
            continue
        for case in ("same", "cross"):
            for sched in scheds:
                c2 = Circuit2(N, A, B, sched, case)
                built = c2.build()
                if built is None:
                    continue
                layers, xr, zr = built
                if not c2.proper(layers):
                    continue
                ec = end_cycle(N, A, B, layers, xr, zr)
                if ec is None:
                    continue
                n_valid += 1
                hx, hz, data = ec
                kend = N - gf2.rank(hx) - gf2.rank(hz)
                if kend != kt:
                    continue
                dx, dz = gf2.min_logical(hx, hz, N, half), gf2.min_logical(hz, hx, N, half)
                if any(x is not None and x < dt for x in (dx, dz)):
                    continue
                row = dict(N=N, A=list(A), B=list(B), d_mid=dm, case=case, sched=[list(map(list, l)) for l in sched],
                           root=c2.root, verts=c2.verts, d_end=[dx, dz], degree=degree(N, layers),
                           roots_all_right=all(q >= N for q in xr + zr))
                hits.append(row)
                print(json.dumps(row), flush=True)
    print(json.dumps(dict(summary=True, N=N, w=w, k=kt, d=dt, D=D, schedules=len(scheds), classes=len(classes),
                          valid_circuits=n_valid, hits=len(hits),
                          hits_degree=sorted({h["degree"] for h in hits}), seconds=round(time.time() - t0, 1))))


if __name__ == "__main__":
    main()
