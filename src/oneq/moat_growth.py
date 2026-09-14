r"""Primal-dual moat GROWTH: a feasible cut packing without solving an LP.

WHY THIS EXISTS. `moat_dual` builds a candidate family of balls and maximises
sum z over it with scipy's LP. That is fine for hundreds of shots and hopeless
for a million: one LP per shot, with a constraint row per edge, is the entire
cost of the experiment. Milestone 0.2c asks for >= 10^6 shots, so the dual
producer has to stop being an LP.

THE ALGORITHM is Edmonds' moat growth, the primal-dual method that the blossom
algorithm itself is built on. Every T-vertex starts as its own moat and all
T-ODD moats grow at the same rate. An edge's packing constraint

    sum_{S : e in delta(S)} z_S  <=  w_e

is exactly "the moats growing toward each other from u and v have not yet met",
so the constraint that binds first is the edge that goes tight first. When two
moats meet they merge, and the merged moat keeps growing only if its
intersection with T is still ODD -- an even moat is not a dual variable at all
and must stop. Since |T| is even, everything eventually merges into an even
component and the process halts.

WHAT IS GUARANTEED. Feasibility, by construction: growth stops at the exact
moment a constraint becomes tight, so no constraint is ever violated. That is
the only property soundness needs -- weak duality makes ANY feasible packing a
lower bound, however it was produced. Optimality of the packing is NOT claimed
and is not needed; a loose packing yields a loose bound, which the checker
reports as a gap rather than as a pass.

WHAT IS NOT GUARANTEED, AND WHY IT DOES NOT MATTER. Floating point can leave a
constraint tight by a few ulps in the wrong direction. This module does not
ask to be trusted about that: the INDEPENDENT checker re-verifies every
constraint from the edge list, and a packing that drifts infeasible is refused,
not silently accepted. A producer bug degrades the certification rate. It
cannot manufacture a false certificate.

COST. Union-find with small-to-large member merging, over a frontier that only
ever covers the moats themselves rather than the whole decoding graph. The
graph beyond the moats is never touched, which is what makes a million shots
affordable: at realistic error rates a shot's moats cover a few hundred
vertices of a graph with tens of thousands.
"""

from __future__ import annotations

BOUNDARY = -1


class _DSU:
    """Union-find carrying, per component, the T-parity and the accumulated
    moat radius shared by its members.

    y[v] -- how far v's moat has grown in total -- is stored split as
    ``base[v] + A[root(v)]`` so that growing a whole component is one addition
    instead of one per member. On merge only the SMALLER side is rebased into
    the larger side's frame, which is the standard small-to-large argument and
    keeps the total rebasing work near-linear.
    """

    __slots__ = ("parent", "members", "base", "A", "tcount")

    def __init__(self):
        self.parent: dict[int, int] = {}
        self.members: dict[int, list[int]] = {}
        self.base: dict[int, float] = {}
        self.A: dict[int, float] = {}
        self.tcount: dict[int, int] = {}

    def add(self, v: int, in_T: bool) -> None:
        if v in self.parent:
            return
        self.parent[v] = v
        self.members[v] = [v]
        self.base[v] = 0.0
        self.A[v] = 0.0
        self.tcount[v] = 1 if in_T else 0

    def find(self, v: int) -> int:
        p = self.parent
        r = v
        while p[r] != r:
            r = p[r]
        while p[v] != r:          # path compression
            p[v], v = r, p[v]
        return r

    def y(self, v: int) -> float:
        return self.base[v] + self.A[self.find(v)]

    def active(self, r: int) -> bool:
        return self.tcount[r] % 2 == 1

    def grow(self, r: int, delta: float) -> None:
        self.A[r] += delta

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if len(self.members[ra]) > len(self.members[rb]):
            ra, rb = rb, ra                      # ra is now the smaller side
        shift = self.A[ra] - self.A[rb]
        for v in self.members[ra]:
            self.base[v] += shift                # preserve y[v] across the move
            self.parent[v] = rb
        self.members[rb].extend(self.members[ra])
        self.tcount[rb] += self.tcount[ra]
        del self.members[ra]
        self.parent[ra] = rb
        return rb


def build_adj(edges) -> dict[int, list[tuple[int, float]]]:
    adj: dict[int, list[tuple[int, float]]] = {}
    for u, v, w in edges:
        u, v, w = int(u), int(v), float(w)
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))
    return adj


def moat_growth(edges, syndrome, *, adj=None, max_events: int = 100_000):
    """Grow moats until every component is T-even. Returns (z, objective).

    ``z`` maps a frozenset of vertices (a T-ODD moat) to its dual value; the
    objective is their sum, a certified lower bound on the minimum-weight
    T-join. Pass ``adj`` to reuse the adjacency across shots on one graph --
    rebuilding it per shot dominates the runtime otherwise.
    """
    T = {int(s) for s in syndrome}
    if len(T) % 2 == 1:
        T = T | {BOUNDARY}          # the boundary absorbs odd parity
    if not T:
        return {}, 0.0
    if adj is None:
        adj = build_adj(edges)

    dsu = _DSU()
    for t in T:
        dsu.add(t, True)

    # The frontier: every edge with at least one endpoint inside a moat. Only
    # these can ever go tight, so the graph outside the moats is never scanned.
    frontier: dict[tuple[int, int, float], None] = {}

    def touch(v: int) -> None:
        for (nb, w) in adj.get(v, ()):
            key = (v, nb, w) if v <= nb else (nb, v, w)
            frontier[key] = None
            dsu.add(nb, nb in T)

    for t in T:
        touch(t)

    z: dict[frozenset, float] = {}
    active_roots = {dsu.find(t) for t in T if dsu.active(dsu.find(t))}

    events = 0
    while events < max_events:
        events += 1
        # Which packing constraint binds first?
        #
        # ONE VARIABLE, ONE FACT. This was `best_t: float | None` and
        # `best_e = None` assigned together and read under a check on
        # `best_e` alone -- so `best_t > 0.0` compared a `float | None`
        # against a float. It happens to be safe (the two are set in the
        # same statement and nowhere else), but nothing in the code says
        # so: a checker narrows one and not the other, and neither can a
        # reader. The same shape `coset_enclosure` was rebuilt to remove.
        best: tuple[float, tuple[int, int, float]] | None = None
        dead = []
        for key in frontier:
            u, v, w = key
            ru, rv = dsu.find(u), dsu.find(v)
            if ru == rv:
                dead.append(key)              # swallowed by a moat; never tight
                continue
            rate = (ru in active_roots) + (rv in active_roots)
            if rate == 0:
                continue                      # dormant, but may wake on a merge
            slack = w - dsu.y(u) - dsu.y(v)
            if slack < 0.0:
                slack = 0.0                   # numerical floor: never negative
            tt = slack / rate
            if best is None or tt < best[0]:
                best = (tt, key)
        for k in dead:
            del frontier[k]
        if best is None:
            break                             # every moat is T-even: done
        best_t, best_e = best

        if best_t > 0.0:
            for r in active_roots:
                S = frozenset(dsu.members[r])
                z[S] = z.get(S, 0.0) + best_t
                dsu.grow(r, best_t)

        u, v, _w = best_e
        ra, rb = dsu.find(u), dsu.find(v)
        merged = dsu.union(u, v)
        touch(u)
        touch(v)                              # one side may be newly enclosed
        active_roots.discard(ra)              # both old identities are gone
        active_roots.discard(rb)
        if dsu.active(merged):                # odd + even stays odd; odd + odd dies
            active_roots.add(merged)
        del frontier[best_e]

    z = {S: val for S, val in z.items() if val > 1e-12}
    return z, float(sum(z.values()))


def primal_guided_sets(correction, syndrome):
    r"""The sets that CAN carry dual weight, read off the ANSWER.

    WHY THE BALL FAMILY RUNS OUT. Candidates drawn as balls around single
    T-vertices at the first few shortest-path radii are chosen from the GRAPH's
    metric, which knows nothing about the instance. As codes and round counts
    grow the correction's paths get longer, the moats a tight packing needs sit
    deeper than any fixed radius cap, and the certification rate falls exactly
    where it matters -- 96% at d=3 over 13 rounds down to 49% at d=7 over 30.

    COMPLEMENTARY SLACKNESS SAYS WHICH SETS MATTER, and it names them exactly.
    In an optimal primal-dual pair every S carrying z_S > 0 satisfies

        |delta(S) inter J| = 1

    -- it cuts the correction in exactly one place. And a minimum-weight T-join
    is a FOREST (a cycle could be deleted to reduce weight), so every edge of J
    splits its tree into two parts. Both parts are T-ODD, and provably: the only
    J-edge crossing either part is the one that was removed, and a vertex set's
    intersection with T has the same parity as the number of J-edges leaving it.

    So the candidates are |J| sets rather than |T| x radii guesses, they are
    laminar by construction, and they are the only ones the theory permits to be
    tight. Generating candidates from the DECODER'S ANSWER is sound for exactly
    the reason the whole method is sound: the checker re-derives feasibility from
    the edge list and never takes the producer's word for anything. A wrong
    answer produces useless candidates, not a false certificate.
    """
    T = {int(s) for s in syndrome}
    if len(T) % 2 == 1:
        T = T | {BOUNDARY}

    adjJ: dict[int, list[tuple[int, int]]] = {}
    for e in correction:
        u, v = int(e[0]), int(e[1])
        key = (v, u) if u == BOUNDARY else ((u, v) if (v == BOUNDARY or u <= v) else (v, u))
        adjJ.setdefault(u, []).append((v, key))
        adjJ.setdefault(v, []).append((u, key))

    out: list[frozenset] = []
    seen: set[frozenset] = set()
    for e in correction:
        u, v = int(e[0]), int(e[1])
        cut = (v, u) if u == BOUNDARY else ((u, v) if (v == BOUNDARY or u <= v) else (v, u))
        # the component of u in J MINUS this edge
        comp = {u}
        stack = [u]
        removed = False
        while stack:
            x = stack.pop()
            for (y, key) in adjJ.get(x, ()):
                if key == cut and not removed:
                    removed = True          # drop ONE copy of the cut edge
                    continue
                if y not in comp:
                    comp.add(y)
                    stack.append(y)
        if v in comp:
            # the endpoints are still connected, so this edge lies on a cycle
            # and cutting it splits nothing. A minimum T-join has no cycles, so
            # this means the correction is not minimal -- skip rather than emit
            # a set that is not T-odd.
            continue
        S = frozenset(comp)
        if S and S not in seen and len(S & T) % 2 == 1:
            seen.add(S)
            out.append(S)
    return out


def primal_radii(correction, W, syndrome):
    r"""The radii at which a ball around a T-vertex cuts the correction ONCE.

    WHY THE OBVIOUS PRIMAL-GUIDED CONSTRUCTION FAILED. Complementary slackness
    says every S carrying z_S > 0 cuts J exactly once, so the tree-split sets
    of the correction look like the right candidates. They are not: they are
    MINIMAL. A set containing only correction vertices has a wide, cheap
    boundary in the full graph, dozens of non-J edges cross it, and their
    constraints crush z_S to nearly nothing. Offered nothing else, the LP
    certified 0 of 150 shots -- strictly worse than the metric balls it was
    meant to replace.

    The theory names WHICH CUT, not which set. The set still has to be grown
    outward until its boundary is expensive, and a ball does that correctly.
    So the primal's contribution is the RADIUS: walk J outward from t, and each
    accumulated path distance is a radius at which the ball around t contains
    everything up to one correction edge and excludes the far side of it.

    That is the piece a fixed radius cap cannot supply. Balls at the first six
    distinct shortest-path distances are chosen from the graph's metric and
    know nothing about the instance; as round counts grow the correction's
    paths get longer and the radii that matter sit far deeper than any cap. The
    prefix distances along J are exactly those radii, there are |J| of them,
    and they cost nothing to compute.

    Sound for the usual reason: the checker re-derives feasibility from the
    edge list. A wrong correction yields useless radii, never a false proof.
    """
    T = {int(s) for s in syndrome}
    if len(T) % 2 == 1:
        T = T | {BOUNDARY}

    adjJ: dict[int, list[tuple[int, float]]] = {}
    for e in correction:
        u, v = int(e[0]), int(e[1])
        k = (v, u) if u == BOUNDARY else ((u, v) if (v == BOUNDARY or u <= v) else (v, u))
        w = float(W[k])
        adjJ.setdefault(u, []).append((v, w))
        adjJ.setdefault(v, []).append((u, w))

    out: dict[int, list[float]] = {}
    for t in T:
        if t not in adjJ:
            continue
        radii: list[float] = []
        seen = {t}
        stack = [(t, 0.0)]
        while stack:
            x, acc = stack.pop()
            for (y, w) in adjJ.get(x, ()):
                if y in seen:
                    continue
                seen.add(y)
                # ball {v : dist(v,t) < acc+w} holds x (at acc) and excludes y
                radii.append(acc + w)
                stack.append((y, acc + w))
        if radii:
            out[t] = radii
    return out


def tight_components(edges, z, syndrome, tol=1e-9):
    r"""The T-odd groups a packing could still grow -- read off its own slack.

    THE DIAGNOSIS THIS ANSWERS. Balls are centred on single detectors. When a
    shot lights ninety of them, any ball of useful radius swallows several, only
    the T-ODD ones are admissible, and what survives cannot express a moat
    around a GROUP. Measured at d=7 over 25 rounds, raising the radius cap from
    8 to 128 -- sixteen times deeper -- closed the same 3 of 19 hard shots and
    left the same 0.42% residual gap. Depth is not the missing ingredient;
    groups are.

    WHERE THE GROUPS COME FROM. A set S can only carry more dual weight if
    every edge on its boundary still has slack: one saturated edge in delta(S)
    pins z_S where it is. So the sets worth adding are unions of components of
    the TIGHT subgraph -- the edges the current packing has already saturated.
    Those components are exactly the moats the packing has grown into, and the
    T-odd ones are admissible dual variables.

    This is moat growth again, but merged along the edges the LP SAYS are tight
    rather than along whichever edge a greedy sweep reached first. The greedy
    version was far looser than the LP (31/120 against 113/120 at d=7) because
    it merges the wrong pairs; here the LP chooses.
    """
    T = {int(s) for s in syndrome}
    if len(T) % 2 == 1:
        T = T | {BOUNDARY}

    # load per edge: the duals whose boundary it crosses. Same walk the checker
    # uses -- each moat's own boundary, not every edge times every dual.
    incident: dict[int, set] = {}
    W: dict[tuple[int, int], float] = {}
    for u, v, w in edges:
        u, v = int(u), int(v)
        k = (v, u) if u == BOUNDARY else ((u, v) if (v == BOUNDARY or u <= v) else (v, u))
        if k not in W or w < W[k]:
            W[k] = float(w)
        incident.setdefault(k[0], set()).add(k)
        incident.setdefault(k[1], set()).add(k)

    load: dict[tuple[int, int], float] = {}
    for S, val in z.items():
        if val == 0.0:
            continue
        for node in S:
            for k in incident.get(node, ()):
                other = k[1] if k[0] == node else k[0]
                if other not in S:
                    load[k] = load.get(k, 0.0) + val

    # union-find over the tight edges
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, w in W.items():
        if w - load.get(k, 0.0) <= tol:          # saturated
            a, b = find(k[0]), find(k[1])
            if a != b:
                parent[a] = b

    comps: dict[int, set] = {}
    for k in W:
        for node in k:
            comps.setdefault(find(node), set()).add(node)

    out = []
    for members in comps.values():
        if len(members) < 2:
            continue                              # a singleton is already a ball
        if len(members & T) % 2 == 1:             # ADMISSIBLE ONLY IF T-ODD
            out.append(frozenset(members))
    return out


def cs_family(edges, correction, syndrome, *, adj=None,
              radii_per_edge: int = 10, max_pop: int = 900):
    r"""Sets that cut the correction EXACTLY ONCE, grown from its tree-splits.

    WHY THIS FAMILY, not merely a richer one. At an optimal primal-dual pair,
    complementary slackness forces every set carrying dual weight to satisfy
    |delta(S) inter J| = 1 -- and the T-join polytope is integral
    (Edmonds-Johnson), so whenever the correction is optimal an optimal dual
    SUPPORTED ENTIRELY ON CUT-ONCE SETS exists. Any such set obeys

        sum_S z_S = sum_S z_S |delta(S) inter J| = sum_{e in J} load(e)

    so making every J-edge tight forces the objective to meet the primal.

    THE FIRST VERSION CLOSED 0 OF 17 HARD SHOTS, and the flaw was measurable,
    not conceptual. J is a FOREST OF MANY SEPARATE PATHS -- one per defect
    pair, ~40 trees on a d=7 25-round shot -- and v1 blocked every J-vertex
    outside the split side A_e. But a set that swallows another tree WHOLE
    still cuts J exactly once, and in dense regions the optimal moats do
    exactly that: they wrap clusters of neighbouring defect pairs. Forbidding
    whole-tree absorption forbade the very sets the optimal dual is made of,
    and the residual gap sat at 20% of the primal.

    So growth now absorbs foreign trees ATOMICALLY: when the outward Dijkstra
    reaches any vertex of another tree, every vertex of that tree enters at
    the same radius (the cheapest entry found), and growth continues from all
    of them. All-in-or-all-out is what preserves the cut-once property; only
    the far side of the edge's own tree and the boundary stay blocked.
    """
    if adj is None:
        adj = build_adj(edges)
    J = [(int(e[0]), int(e[1])) for e in correction]
    if not J:
        return []

    jadj: dict[int, list[tuple[int, int]]] = {}
    jverts: set[int] = set()
    for i, (u, v) in enumerate(J):
        jverts |= {u, v}
        jadj.setdefault(u, []).append((v, i))
        jadj.setdefault(v, []).append((u, i))

    # tree id per J-vertex, and each tree's full vertex set
    tree_of: dict[int, int] = {}
    trees: list[set[int]] = []
    for start in jverts:
        if start in tree_of:
            continue
        tid = len(trees)
        comp = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            tree_of[x] = tid
            for (y, _i) in jadj.get(x, ()):
                if y not in comp:
                    comp.add(y)
                    stack.append(y)
        trees.append(comp)

    import heapq
    out = []
    seen = set()

    def side_of(start, skip_edge):
        comp = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for (y, ei) in jadj.get(x, ()):
                if ei == skip_edge or y in comp:
                    continue
                comp.add(y)
                stack.append(y)
        return comp

    for i, (u, v) in enumerate(J):
        side_u = side_of(u, i)
        if v in side_u:
            continue                     # e lies on a cycle: J is not minimal
        A = side_of(v, i) if BOUNDARY in side_u else side_u
        if BOUNDARY in A:
            continue
        home = tree_of[u]
        # blocked forever: the far side of e's OWN tree, the boundary -- and
        # EVERY VERTEX of a foreign tree that contains the boundary. Such a
        # tree cannot be absorbed whole (the boundary must never enter S) and
        # partial absorption slices it, so its members entering `reached`
        # individually was exactly the cut-twice case the property test caught
        # on first run, before any mutant was tried.
        blocked = (trees[home] - A) | {BOUNDARY}
        for tr in trees:
            if BOUNDARY in tr and tr is not trees[home]:
                blocked |= tr

        S0 = frozenset(A)
        if S0 not in seen:
            seen.add(S0)
            out.append(S0)

        dist = {a: 0.0 for a in A}
        pq = [(0.0, a) for a in A]
        heapq.heapify(pq)
        pops = 0
        reached = {}
        while pq and pops < max_pop:
            d, x = heapq.heappop(pq)
            if d > dist.get(x, float("inf")):
                continue
            pops += 1
            # ATOMIC TREE ABSORPTION: a foreign J-vertex drags its whole tree
            # in at this radius, and growth continues from every member.
            if x in tree_of and tree_of[x] != home and x not in A:
                for z in trees[tree_of[x]]:
                    if d < dist.get(z, float("inf")) - 1e-15:
                        dist[z] = d
                        reached[z] = d
                        heapq.heappush(pq, (d, z))
            for (y, w) in adj.get(x, ()):
                if y in blocked:
                    continue
                nd = d + w
                if nd < dist.get(y, float("inf")) - 1e-15:
                    dist[y] = nd
                    if y not in A:
                        reached[y] = nd
                    heapq.heappush(pq, (nd, y))
        if not reached:
            continue
        # radii only where the set changes; entry values are shared tree-wide,
        # so the inclusive comparison below always takes a tree whole
        radii = sorted({round(d, 12) for d in reached.values()})[:radii_per_edge]
        for r in radii:
            S = frozenset(A | {y for y, d in reached.items() if d <= r + 1e-12})
            if S not in seen:
                seen.add(S)
                out.append(S)
    return out


def matching_dual_candidates(edges, syndrome, *, adj=None,
                             max_rounds: int = 12, levels_per_set: int = 6):
    r"""Blossom-SHAPED candidates from a cutting-plane matching dual.

    THE IDEA, stated once. The optimal cut packing is laminar, cut-once, and
    corresponds exactly to the optimal DUAL of minimum-weight perfect matching
    on the metric closure over T: node duals y_t are moat depths around single
    terminals, blossom duals z_B are moats around GROUPS. Nobody exports those
    duals -- PyMatching does not, networkx does not -- and implementing
    Galil's algorithm with dual tracking is a project. But the closure has
    only |T| ~ 100-170 nodes, so the matching LP is TINY, and its blossom
    facets can be added lazily: solve the degree-constrained relaxation, and
    while the solution is fractional, add odd-set cuts x(delta(B)) >= 1 for
    the odd components of the fractional support (basic solutions of the
    relaxation are half-integral disjoint odd cycles, so the violated sets sit
    in plain sight). scipy returns the duals of the final LP for free.

    THE DUALS ARE NOT TRUSTED -- THEY ARE TRANSLATED INTO CANDIDATES. y_t
    names the radius at which terminal t's moat should stop; each blossom B
    names a group whose union-moat should grow, expanded from the y-inflated
    balls of its members. Both become vertex sets handed to the SAME max-form
    LP as every other family, filtered by the same T-odd rule, re-weighted
    under the graph's own edge constraints, and adjudicated by the same
    independent checker. A wrong dual yields useless candidates, never a
    false certificate -- so none of the boundary-copy sign subtleties that
    make the pure translation delicate can cost soundness, only rate.

    IS A PRODUCER THAT SOLVES THE PROBLEM CIRCULAR? No, and the distinction is
    the whole method: what ships is the CERTIFICATE, checked from the edge
    list by arithmetic that never asks how the packing was found. The producer
    may consult any oracle it likes; the checker cannot be talked into
    anything by it.
    """
    import numpy as np
    from scipy.optimize import linprog
    if adj is None:
        adj = build_adj(edges)
    T = sorted({int(s) for s in syndrome})
    if not T:
        return []

    # shortest-path fields from every terminal (reused for the translation)
    import heapq

    def dij(src):
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

    dists = {t: dij(t) for t in T}
    BIG = 1e9

    # closure nodes: ('t', t) and one boundary copy ('b', t) per terminal
    nodes = [("t", t) for t in T] + [("b", t) for t in T]
    nidx = {n: i for i, n in enumerate(nodes)}
    evars = []          # (i, j, cost)
    for a in range(len(T)):
        ta = T[a]
        for b in range(a + 1, len(T)):
            evars.append((nidx[("t", ta)], nidx[("t", T[b])],
                          dists[ta].get(T[b], BIG)))
        evars.append((nidx[("t", ta)], nidx[("b", ta)],
                      dists[ta].get(BOUNDARY, BIG)))
        for b in range(a):
            evars.append((nidx[("b", ta)], nidx[("b", T[b])], 0.0))
    ne, nv = len(evars), len(nodes)
    cost = np.array([c for (_i, _j, c) in evars])
    # SPARSE, because dense killed it. On a 170-detector hardware shot the
    # closure has ~29k edge variables, and a dense 340 x 29k float64 A_eq is
    # 78 MB before the solver copies it -- the allocation that took the
    # hardware spot-check down. The matrix has exactly two nonzeros per
    # column; store exactly those.
    from scipy.sparse import csr_matrix
    rows = [x for (i, j, _c) in evars for x in (i, j)]
    cols = [k for k in range(ne) for _ in (0, 1)]
    A_eq = csr_matrix((np.ones(2 * ne), (rows, cols)), shape=(nv, ne))

    cuts: list[set] = []           # each: a set of closure-node indices, |odd|
    y = None
    zB = []
    jittered = False
    for _round in range(max_rounds):
        ub_r, ub_c, b_ub = [], [], []
        for ci, B in enumerate(cuts):
            for k, (i, j, _c) in enumerate(evars):
                if (i in B) != (j in B):
                    ub_r.append(ci)
                    ub_c.append(k)         # -x(delta(B)) <= -1  ==  >= 1
            b_ub.append(-1.0)
        A_ub = (csr_matrix((-np.ones(len(ub_r)), (ub_r, ub_c)),
                           shape=(len(cuts), ne)) if cuts else None)
        res = linprog(cost, A_eq=A_eq, b_eq=np.ones(nv),
                      A_ub=A_ub, b_ub=np.array(b_ub) if cuts else None,
                      bounds=(0.0, 1.0), method="highs")
        if not res.success:
            return []
        x = res.x
        y = np.array(res.eqlin.marginals)
        zB = (list(np.abs(res.ineqlin.marginals))
              if A_ub is not None else [])
        frac = [(k, evars[k]) for k in range(ne) if 1e-6 < x[k] < 1 - 1e-6]
        if not frac:
            break                          # integral: duals are exact
        # odd components of the fractional support are the violated blossoms
        comp_adj: dict[int, set] = {}
        for _k, (i, j, _c) in frac:
            comp_adj.setdefault(i, set()).add(j)
            comp_adj.setdefault(j, set()).add(i)
        seen: set[int] = set()
        added = False
        for start in comp_adj:
            if start in seen:
                continue
            comp = {start}
            stack = [start]
            while stack:
                u = stack.pop()
                for v in comp_adj.get(u, ()):
                    if v not in comp:
                        comp.add(v)
                        stack.append(v)
            seen |= comp
            if len(comp) % 2 == 1 and comp not in cuts:
                cuts.append(comp)
                added = True
        if not added:
            # SEPARATION STALL: the solution is fractional but every odd
            # component of its support is already a cut -- a degenerate
            # optimal face where this separation heuristic goes blind (found
            # live: one hardware shot sat here with a 0.88 closure gap while
            # eleven siblings converged). General odd-cut separation is
            # Padberg-Rao; the cheap cure that fits this loop is a one-time
            # deterministic cost jitter to tilt the LP off the degenerate
            # face so the next basic solution exposes new odd components.
            # Costs move by <=1e-7 total, orders below any real gap; the
            # duals are TRANSLATED into candidates and re-optimized under
            # true weights, so jitter can cost only rate, never soundness.
            if not jittered:
                jittered = True
                rng = np.random.default_rng(20260801)
                cost = cost + 1e-9 * rng.random(ne)
                continue
            break                          # jitter spent: use the LP as-is

    if y is None:
        return []

    out: list[frozenset] = []
    seen_sets: set[frozenset] = set()
    allv = frozenset(adj.keys())

    def emit(S):
        S = frozenset(S)
        if not S:
            return
        # A SET CONTAINING THE BOUNDARY DEFINES THE SAME CUT AS ITS
        # COMPLEMENT, WHICH DOES NOT. The first version refused
        # boundary-containing sets outright, and that refusal was exactly the
        # lossiness in the translation: the closure LP's boundary copies carry
        # NEGATIVE duals (zero-weight b-b edges force them down), which lets a
        # terminal's dual depth y_t exceed its distance to the boundary -- a
        # moat that grows THROUGH the boundary. Legal in the T-join picture,
        # inexpressible as a boundary-free ball. Its complement expresses it:
        # delta(S) = delta(V minus S), and the T-odd parity is preserved
        # because |T| is even. So the moat is emitted as the rest of the graph.
        if BOUNDARY in S:
            S = allv - S
            if not S or BOUNDARY in S:
                return
        if S not in seen_sets:
            seen_sets.add(S)
            out.append(S)

    # terminal moats: balls around t at the breakpoints BELOW its dual depth,
    # ending exactly at y_t -- the radius the optimal schedule stops at.
    # y_t MAY EXCEED d(t, boundary): the boundary copy's negative dual pays
    # for the overshoot, and the ball then contains the boundary. emit()
    # handles that by complement, so the depth is NOT clamped here.
    for a, t in enumerate(T):
        yt = float(y[nidx[("t", t)]])
        if yt <= 1e-12:
            continue
        vals = sorted({round(d, 12) for d in dists[t].values()
                       if 0 < d <= yt + 1e-9})
        picks = vals[-levels_per_set:] if len(vals) > levels_per_set else vals
        for r in picks:
            emit({v for v, d in dists[t].items() if d < r - 1e-12} | {t})
        emit({v for v, d in dists[t].items() if d <= yt + 1e-9} | {t})

    # blossom moats. Three defects in the first version, found when a probe
    # showed the closure LP converging EXACTLY to the primal on 10 of 12
    # stubborn hardware shots while the translated bound still gapped -- the
    # entire loss was in this emission, and each defect is a piece of the
    # primal-dual correspondence the first version dropped:
    #
    #   NESTING. For blossoms B' inside B, the moat of B grows from members
    #   inflated by y_t PLUS the inner blossoms' z -- the primal-dual schema
    #   grows duals simultaneously, so the outer moat starts where the inner
    #   growth stopped. Growing from y_t alone put outer rings at radii the
    #   optimal schedule had already passed, crossing the wrong edges.
    #   Cuts are processed smallest-first so inner inflations accumulate.
    #
    #   BREAKPOINTS. Ring radii must sit at the values where the crossing
    #   set actually changes -- (d(t,v) - infl_t) breakpoints -- exactly as
    #   the ball emission already did with its distance values. A linspace
    #   grid misses them, and the value in any annulus between two emitted
    #   rings whose crossing set changes inside is unchargeable.
    #
    #   BOUNDARY BLOSSOMS. A blossom containing boundary copies was dropped
    #   outright (its z thrown away) -- the same boundary lossiness already
    #   fixed once for balls. Its terminal-side region is meaningful; if the
    #   grown region swallows BOUNDARY, emit() already complements it.
    infl = {t: float(y[nidx[("t", t)]]) for t in T}
    full_regions: list[frozenset] = []
    for B, z in sorted(zip(cuts, zB), key=lambda bz: len(bz[0])):
        Bt = [nodes[i][1] for i in B if nodes[i][0] == "t"]
        if not Bt or z <= 1e-12:
            continue
        rel: dict[int, float] = {}
        for t in Bt:
            it = infl[t]
            for v, d in dists[t].items():
                r = d - it
                if r < rel.get(v, float("inf")):
                    rel[v] = r
        brk = sorted({round(r, 12) for r in rel.values() if 0 < r <= z + 1e-9})
        if len(brk) > levels_per_set:
            # BOTH ends of the growth schedule, not the top alone: the small
            # rings hug the cluster and are the ones most likely to cut J
            # exactly once; the top ring at z is where the dual's width ends.
            # Picking only the largest rings -- the first fix's choice --
            # made the Z-basis config WORSE at base caps, because big rings
            # swallow neighboring terminals and displace useful candidates.
            h = levels_per_set // 2
            picks = brk[:h] + brk[-(levels_per_set - h):]
        else:
            picks = brk
        base = set(Bt) | {v for v, r in rel.items() if r <= 1e-12}
        for s in picks:
            emit(base | {v for v, r in rel.items() if r <= s + 1e-9})
        full = frozenset(base | {v for v, r in rel.items() if r <= z + 1e-9})
        emit(full)
        full_regions.append(full)
        for t in Bt:
            infl[t] += float(z)

    # UNCROSSING, approximated. The cutting-plane loop accumulates odd sets
    # across rounds with no laminarity guarantee, while the nested-inflation
    # schedule above is only exact for laminar supports -- the measured
    # residue after the nesting fix (integral closure, translated bound still
    # short) is consistent with exactly this. The classical repair is to
    # uncross the dual; the candidate-generation shortcut is to emit the
    # unions and intersections of overlapping blossom regions and let the
    # packing LP choose -- an uncrossed laminar family lives inside that
    # closure, and a useless set costs rate, never soundness.
    for i in range(len(full_regions)):
        for j in range(i + 1, len(full_regions)):
            A, Br = full_regions[i], full_regions[j]
            inter = A & Br
            if len(inter) < 2 or not (A - Br) or not (Br - A):
                continue                # disjoint or nested: nothing to uncross
            emit(inter)
            # a union that balloons past its operands is a moat covering half
            # the graph -- it crosses everything, cuts nothing once, and its
            # dense crossing row is pure allocation cost (measured: the
            # unbounded version died on a 13.5 MiB crossing matrix)
            U = A | Br
            if len(U) <= 1.5 * max(len(A), len(Br)):
                emit(U)
    return out
