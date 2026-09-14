r"""Sound duals for QEC decoding: a cut packing, not node potentials.

WHY THE OBVIOUS DUAL WAS WRONG. QEC decoding is a minimum-weight T-JOIN: find
the cheapest edge set whose odd-degree vertices are exactly the fired detectors.
The natural-looking dual -- node potentials with y_u + y_v <= w_e per edge -- is
the dual of a DIFFERENT primal. It is necessary but not sufficient, because no
constraint binds a pair of detectors joined by a PATH rather than an edge. Run
against PyMatching at d=7 it produced a "bound" of 75.008 against a primal of
73.971: above the optimum, which weak duality forbids. The checker caught it.

THE CORRECT DUAL is Edmonds' cut/moat packing:

    max  sum_S z_S
    s.t. sum_{S : e in delta(S)} z_S  <=  w_e     for every edge e
         z_S >= 0,  S ranges over T-ODD vertex sets

Any feasible z is a lower bound on the T-join weight -- that is weak duality and
it needs no assumption about how z was produced. Crucially the constraints are
EDGE-LOCAL, so a checker verifies them in O(nnz * |support(z)|) with no LP and
no matching algorithm: exactly the property the certifying-algorithms discipline
requires.

HOW THIS PRODUCES z. Balls (moats) grown around each T-vertex:

    B_i(r) = { v : dist_i(v) < r }

with candidate radii taken from the distinct shortest-path distances. A ball is
an admissible dual variable only when |B_i(r) ∩ T| is ODD. An edge lies in
delta(B_i(r)) exactly when one endpoint is inside and the other is not, which is
read straight off the Dijkstra distances. Maximising sum z over that restricted
family is a small LP -- and RESTRICTING the family costs nothing in soundness,
because any feasible point of a restricted dual is still feasible for the full
one. It can only cost tightness, and tightness is reported as a gap rather than
assumed.

BOUNDARY. The decoding graph has a virtual boundary vertex. When the syndrome
has odd parity the boundary joins T, which is what makes |T| even and the
T-join well posed.

TWO IMPLEMENTATIONS, ONE RULE. `moat_dual` is the reference: a direct
transcription of the definitions above, slow and obvious. `MoatEngine` is the
same mathematics restructured so the work happens inside numpy and scipy
instead of inside Python loops -- profiling put 28 ms/shot in Dijkstra, 53 ms
in building a dense constraint matrix and 46 ms in the solver at d=7, and only
the last of those is real work. Both are kept because a fast implementation
nobody can read is not a specification, and a specification nobody can run at
scale is not an experiment. `test_engine_agrees_with_the_reference_producer`
holds them together; without it this file would be exactly the twin-copy trap
where a rule is fixed in one place and rots in the other.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

BOUNDARY = -1


@dataclass(frozen=True)
class Moat:
    """One dual variable: the ball of radius `radius` around `center`."""
    center: int
    radius: float
    members: frozenset
    z: float = 0.0


def dijkstra(adj: dict[int, list[tuple[int, float]]], src: int) -> dict[int, float]:
    dist = {src: 0.0}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, float("inf")) - 1e-15:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def build_adj(edges) -> dict[int, list[tuple[int, float]]]:
    adj: dict[int, list[tuple[int, float]]] = {}
    for u, v, w in edges:
        u, v, w = int(u), int(v), float(w)
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))
    return adj


def moat_dual(edges, syndrome, *, max_radii_per_center: int = 24):
    """A FEASIBLE cut packing and its objective value.

    Returns (z, objective) where z maps frozenset -> value. Soundness does not
    depend on the LP finding the optimum: any feasible z is a valid lower
    bound, so a weak solve yields a weak certificate, never a wrong one.
    """
    from scipy.optimize import linprog

    T = set(int(s) for s in syndrome)
    if len(T) % 2 == 1:
        T = T | {BOUNDARY}            # boundary absorbs odd parity
    if not T:
        return {}, 0.0

    adj = build_adj(edges)
    E = [(int(u), int(v), float(w)) for u, v, w in edges]

    # candidate moats: balls around each T-vertex at the distinct radii where
    # the ball's membership actually changes
    cand: list[tuple[frozenset, int]] = []
    seen: set[frozenset] = set()
    for t in sorted(T):
        dist = dijkstra(adj, t)
        radii = sorted({round(d, 12) for d in dist.values() if d > 0})
        for r in radii[:max_radii_per_center]:
            members = frozenset(v for v, d in dist.items() if d < r - 1e-12)
            if not members or members in seen:
                continue
            # ADMISSIBLE ONLY IF T-ODD -- an even ball is not a dual variable
            if len(members & T) % 2 == 1:
                seen.add(members)
                cand.append((members, t))
    if not cand:
        return {}, 0.0

    # maximise sum z subject to edge-local packing constraints
    n = len(cand)
    A, b = [], []
    for (u, v, w) in E:
        row = [0.0] * n
        hit = False
        for j, (S, _c) in enumerate(cand):
            if (u in S) ^ (v in S):        # e crosses delta(S)
                row[j] = 1.0
                hit = True
        if hit:
            A.append(row)
            b.append(w)
    if not A:
        return {}, 0.0
    res = linprog([-1.0] * n, A_ub=A, b_ub=b,
                  bounds=[(0.0, None)] * n, method="highs")
    if not res.success:
        return {}, 0.0
    z = {cand[j][0]: float(res.x[j]) for j in range(n) if res.x[j] > 1e-12}
    return z, float(sum(z.values()))


class MoatEngine:
    """The same cut packing, restructured so the loops run in C.

    Per-graph structures are built ONCE and reused across shots, which is the
    whole point: the decoding graph is fixed for a run of a million shots and
    only the syndrome changes. Nothing here alters what is computed -- the
    candidate family, the T-ODD admissibility rule and the edge-local packing
    constraints are identical to `moat_dual` above, and a conformance test
    pins the two together.
    """

    def __init__(self, edges, *, max_radii_per_center: int = 24):
        import numpy as np
        from scipy.sparse import csr_matrix

        E = [(int(u), int(v), float(w)) for u, v, w in edges]
        nodes = sorted({u for u, _v, _w in E} | {v for _u, v, _w in E})
        self.index = {v: i for i, v in enumerate(nodes)}
        self.nodes = np.asarray(nodes)
        n = len(nodes)
        eu = np.fromiter((self.index[u] for u, _v, _w in E), np.int32, len(E))
        ev = np.fromiter((self.index[v] for _u, v, _w in E), np.int32, len(E))
        ew = np.fromiter((w for _u, _v, w in E), np.float64, len(E))
        # PARALLEL EDGES COLLAPSE TO THE CHEAPEST. Two edges on the same pair
        # give two constraints, and the tighter one implies the looser, so
        # keeping both only costs solver time. The checker does the same.
        key = np.minimum(eu, ev).astype(np.int64) * n + np.maximum(eu, ev)
        order = np.lexsort((ew, key))
        keep = np.ones(len(E), bool)
        keep[1:] = key[order][1:] != key[order][:-1]
        sel = order[keep]
        self.eu, self.ev, self.ew = eu[sel], ev[sel], ew[sel]
        self.n = n
        self.max_radii = max_radii_per_center
        # undirected weighted graph for multi-source shortest paths
        r = np.concatenate([self.eu, self.ev])
        c = np.concatenate([self.ev, self.eu])
        d = np.concatenate([self.ew, self.ew])
        self.csr = csr_matrix((d, (r, c)), shape=(n, n))

    def dual(self, syndrome, *, max_radii: int | None = None, extra_sets=(),
             radii_by_node=None):
        """(z, objective) -- a feasible packing and its value.

        ``max_radii`` caps how many nested balls each T-vertex contributes and
        is the single knob that trades tightness for solver time: at d=7 the
        family shrinks 3x and the LP gets 3.6x cheaper while the certification
        rate falls only a few points, because a tight packing uses about |T|
        moats however many candidates it is offered. ``extra_sets`` injects
        additional vertex sets -- moat-growth components, typically -- which
        are filtered by the SAME T-odd rule as the balls and so cannot smuggle
        in an inadmissible dual variable no matter what the caller passes.
        """
        import numpy as np
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse.csgraph import dijkstra as sp_dijkstra

        cap = self.max_radii if max_radii is None else max_radii
        T = {int(s) for s in syndrome}
        if len(T) % 2 == 1:
            T = T | {BOUNDARY}
        T = {t for t in T if t in self.index}
        if not T:
            return {}, 0.0
        t_idx = np.array(sorted(self.index[t] for t in T), dtype=np.int32)

        blocks = []
        if cap > 0 or radii_by_node:
            # ONE scipy call replaces |T| Python Dijkstras.
            dist = sp_dijkstra(self.csr, directed=False, indices=t_idx)
            # ENUMERATE AGAINST t_idx, NOT AGAINST len(blocks). A row with no
            # finite positive distance is skipped, so the two indices drift
            # apart and every primal radius after the first skip would be
            # attached to the WRONG centre -- producing candidates that are
            # merely wrong rather than helpful, and silently.
            for ti, row in zip(t_idx, dist):
                finite = row[np.isfinite(row) & (row > 0)]
                if finite.size == 0:
                    continue
                radii = np.unique(np.round(finite, 12))[:cap]
                # PRIMAL-DERIVED RADII ARE NOT SUBJECT TO THE CAP. The cap
                # bounds a guess; these are not guesses -- each is the distance
                # at which this ball cuts the correction exactly once, the only
                # place complementary slackness allows a tight constraint.
                hint = (radii_by_node or {}).get(int(self.nodes[ti]))
                if hint:
                    radii = np.unique(np.round(
                        np.concatenate([radii, np.asarray(hint, float)]), 12))
                blocks.append(row[None, :] < radii[:, None] - 1e-12)
        for S in extra_sets:
            row = np.zeros(self.n, bool)
            for v in S:
                i = self.index.get(int(v))
                if i is not None:
                    row[i] = True
            if row.any():
                blocks.append(row[None, :])
        if not blocks:
            return {}, 0.0
        memb = np.concatenate(blocks, axis=0)

        # ADMISSIBLE ONLY IF T-ODD -- an even ball is not a dual variable.
        # Same rule as the reference; here it is a parity of a row sum.
        odd = (memb[:, t_idx].sum(axis=1) % 2) == 1
        nonempty = memb.any(axis=1)
        memb = memb[odd & nonempty]
        if memb.size == 0:
            return {}, 0.0
        # dedupe identical balls (different centres reach the same set)
        packed = np.packbits(memb, axis=1)
        _u, first = np.unique(packed, axis=0, return_index=True)
        memb = memb[np.sort(first)]

        # e crosses delta(S) exactly when one endpoint is in S and one is not
        cross = memb[:, self.eu] ^ memb[:, self.ev]      # (ncand, nedge)
        hit = cross.any(axis=0)
        if not hit.any():
            return {}, 0.0
        C = csr_matrix(cross[:, hit].astype(np.float64))  # (ncand, nhit)
        rhs = self.ew[hit]

        # THE LP IS BLOCK DIAGONAL AND SOLVING IT WHOLE IS WASTE.
        #
        # Two moats interact only if some edge crosses BOTH -- otherwise their
        # constraints share no row and their z's cannot trade against each
        # other. On a 25-round space-time graph a shot lights up dozens of
        # detectors that are nowhere near one another, so one LP over every
        # moat at once pays superlinear solver cost to couple variables that
        # are independent by construction. Splitting into connected components
        # of the shared-edge graph changes NOTHING about what is computed --
        # the union of per-block optima IS the global optimum for a separable
        # program -- and turns one large solve into many small ones.
        n_blocks, label = connected_components(C @ C.T, directed=False)
        z: dict = {}
        for blk in range(n_blocks):
            rows = np.flatnonzero(label == blk)
            sub = C[rows]
            cols = np.flatnonzero(np.asarray(sub.sum(axis=0)).ravel() > 0)
            if cols.size == 0:
                continue
            res = linprog(-np.ones(rows.size), A_ub=sub[:, cols].T,
                          b_ub=rhs[cols], bounds=(0.0, None), method="highs")
            if not res.success:
                # FAIL CLOSED PER BLOCK. A block that will not solve costs its
                # own contribution to the bound and nothing else; returning a
                # partial packing is still FEASIBLE, so the shot degrades to a
                # gap rather than to a wrong certificate.
                continue
            for j, val in zip(rows, res.x):
                if val > 1e-12:
                    z[frozenset(self.nodes[memb[j]].tolist())] = float(val)
        return z, float(sum(z.values()))
