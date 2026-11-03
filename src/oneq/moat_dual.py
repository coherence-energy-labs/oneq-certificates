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


def _candidate_components(crossings):
    """Connected candidate blocks, for a matrix with no unused columns.

    Walk the bipartite incidence graph instead of projecting C @ C.T.
    A shared edge can turn k candidates into a k-by-k dense clique;
    incidence stores only two entries per actual crossing. Connectivity
    is identical: each candidate/edge/candidate path corresponds to a
    nonzero of C @ C.T. Candidate vertices come first, retaining the
    first-candidate label order. Each column has an incident candidate,
    so there are no column-only components.
    """
    from scipy.sparse import bmat
    from scipy.sparse.csgraph import connected_components

    incidence = bmat([[None, crossings], [crossings.T, None]], format="csr")
    count, labels = connected_components(incidence, directed=False)
    return count, labels[:crossings.shape[0]]


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
        self._csr = None

    @property
    def csr(self):
        """Build the native graph only when shortest paths are needed.

        Empty syndromes need neither shortest paths nor a solver. Importing
        scipy during graph setup made those shots pay for both anyway.
        """
        if self._csr is None:
            import numpy as np
            from scipy.sparse import csr_matrix

            r = np.concatenate([self.eu, self.ev])
            c = np.concatenate([self.ev, self.eu])
            d = np.concatenate([self.ew, self.ew])
            self._csr = csr_matrix((d, (r, c)), shape=(self.n, self.n))
        return self._csr

    def shot(self, syndrome):
        """A call-local workspace; no syndrome data survives on this engine."""
        return _MoatShot(self, syndrome)

    def distance_fields(self, sources):
        """Native shortest-path fields on this engine's owned graph.

        These are proposer inputs, never verification evidence. Missing source
        vertices retain the reference Dijkstra's isolated-source behaviour.
        No field is retained across shots or exposed as shared cached state.
        """
        import numpy as np

        sources = sorted({int(t) for t in sources})
        known = [t for t in sources if t in self.index]
        out = {t: {t: 0.0} for t in sources if t not in self.index}
        if known:
            from scipy.sparse.csgraph import dijkstra as sp_dijkstra

            fields = sp_dijkstra(self.csr, directed=False,
                                 indices=[self.index[t] for t in known])
            for t, row in zip(known, fields):
                out[t] = {int(self.nodes[i]): float(row[i])
                          for i in np.flatnonzero(np.isfinite(row))}
        return out

    def dual(self, syndrome, *, max_radii: int | None = None, extra_sets=(),
             radii_by_node=None, _shot=None):
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
            if _shot is None:
                from scipy.sparse.csgraph import dijkstra as sp_dijkstra
                dist = sp_dijkstra(self.csr, directed=False, indices=t_idx)
            else:
                if _shot.engine is not self or tuple(t_idx) != _shot.indices:
                    raise ValueError("shortest-path workspace belongs to another graph or syndrome")
                dist = _shot.distances()
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
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix

        # Convert only the nonzero entries to floats. Densifying first costs
        # eight bytes for EVERY candidate/edge pair, including every zero.
        C = csr_matrix(cross[:, hit], dtype=np.float64)  # (ncand, nhit)
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
        n_blocks, label = _candidate_components(C)
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


class _MoatShot:
    """Reuse one shortest-path matrix across a shot's solves and translation.

    Radius caps, extra candidates and LP results remain independent per solve.
    Only distances on the fixed graph are shared. The workspace is owned by
    the caller, so interleaved shots cannot evict or mutate one another's data.
    """

    def __init__(self, engine, syndrome):
        self.engine = engine
        self.syndrome = tuple(syndrome)
        terminals = {int(t) for t in self.syndrome}
        if len(terminals) % 2:
            terminals.add(BOUNDARY)
        self.indices = tuple(sorted(engine.index[t] for t in terminals if t in engine.index))
        self._distances = None

    def distances(self):
        if self._distances is None:
            import numpy as np
            from scipy.sparse.csgraph import dijkstra as sp_dijkstra

            self._distances = sp_dijkstra(self.engine.csr, directed=False,
                indices=np.asarray(self.indices, dtype=np.int32))
            self._distances.flags.writeable = False
        return self._distances

    def dual(self, **kwargs):
        return self.engine.dual(self.syndrome, _shot=self, **kwargs)

    def distance_fields(self, sources):
        """Blossom translation reads the same fields without another walk."""
        import numpy as np

        sources = sorted({int(t) for t in sources})
        positions = {int(self.engine.nodes[i]): j for j, i in enumerate(self.indices)}
        out = self.engine.distance_fields([t for t in sources if t not in positions])
        for t in sources:
            if t in positions:
                row = self.distances()[positions[t]]
                out[t] = {int(self.engine.nodes[i]): float(row[i])
                          for i in np.flatnonzero(np.isfinite(row))}
        return out
