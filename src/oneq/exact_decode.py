r"""Exact minimum-weight T-join decode: the REPAIR half of certify-or-repair.

WHY THIS EXISTS. Measured on d5 hardware shots: some corrections a float
decoder returns are provably NOT minimum-weight -- alternative optimal
pairings tie in float and differ by an ulp in exact arithmetic, and the
decoder sometimes returns the exactly-heavier one. No dual can attain a
primal that is not the optimum (weak duality), so those shots are
UNCERTIFIABLE as shipped -- correctly. The strongest true claim is better
than a refusal: decode EXACTLY, emit the exactly-minimum correction, and
certify THAT. The certificate need not explain the original decoder; it
needs to prove the answer that ships is optimal.

METHOD. Derived-graph reduction (Edmonds-Johnson): terminals = fired
detectors; one boundary copy per terminal; pairwise weights = exact
shortest-path distances (Dijkstra in Fractions -- float appears nowhere);
terminal-terminal and terminal-own-copy edges carry real distances,
copy-copy edges are free. A minimum-weight perfect matching there equals
the minimum T-join. networkx's blossom implementation is pure Python and
runs on Fraction weights directly, so SELECTION is exact, not merely
scoring -- measured to pick the ulp-lighter pairing where floats tie.
Matched pairs are then expanded through recorded Dijkstra predecessors
into an actual edge set, XOR-combined (a path used twice cancels).

REQUIRES NONNEGATIVE WEIGHTS (the reduction's precondition) -- asserted,
not assumed, because PyMatching has a negative-weight path and a silent
precondition is how reductions rot.

WHAT THIS MODULE MUST NEVER DO: certify anything. It is a producer. Its
output goes through the same independent checker as every other
correction, and the caller must verify syndrome reproduction and weight
before trusting either. A defect here costs rate or a refused repair --
never a false certificate.
"""

from __future__ import annotations

import heapq
from fractions import Fraction

BOUNDARY = -1


def _edge_key(u, v):
    # The boundary is always SECOND, whichever side it arrives on. A path
    # that runs THROUGH the boundary walks the predecessor chain with
    # BOUNDARY as `u`, and canonicalising only `v` made (B, x) and (x, B)
    # distinct keys -- caught by the ulp-tie test, whose optimal route is
    # exactly boundary-through.
    if u == BOUNDARY:
        u, v = v, u
    # BOUNDARY IS ALWAYS SECOND. Written boundary-first,
    # u = -1 makes `u <= v` already true, so the swap never
    # fired: (-1, 3) and (3, -1) both survived as distinct
    # keys, the same edge was counted twice, and a valid
    # document was refused with GRAPH MISMATCH -- the exact
    # outcome graph_fingerprint exists to avoid.
    if u == BOUNDARY:
        return (v, u)
    if v == BOUNDARY:
        return (u, v)
    return (u, v) if u <= v else (v, u)


def _dijkstra(adj, src):
    """Exact single-source shortest paths. Returns (dist, predecessor)."""
    dist = {src: Fraction(0)}
    pred = {}
    heap = [(Fraction(0), src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, d):
            continue
        for v, w in adj[u]:
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, pred


def _walk(pred, src, dst):
    """The edge set of the recorded shortest path src -> dst."""
    out = []
    cur = dst
    while cur != src:
        p = pred[cur]
        out.append(_edge_key(p, cur))
        cur = p
    return out


def exact_min_tjoin(edges, syndrome):
    """The exactly-minimum correction for `syndrome`, or (None, None, why).

    Returns (correction, weight): `correction` a tuple of canonical edge
    keys whose odd-degree vertex set equals the syndrome (boundary
    absorbing parity), `weight` its EXACT total as a Fraction. The caller
    re-verifies both properties independently -- this function's word is
    never the acceptance evidence.
    """
    W = {}
    adj = {}
    for u, v, w in edges:
        wf = Fraction(w)
        if wf < 0:
            return None, None, "negative edge weight: the Edmonds-Johnson " \
                               "reduction requires nonnegative weights"
        k = _edge_key(int(u), int(v))
        if k not in W or wf < W[k]:
            W[k] = wf
    for (u, v), w in W.items():
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))

    T = sorted({int(s) for s in syndrome})
    if not T:
        return (), Fraction(0), "empty syndrome: empty correction"
    if any(t not in adj for t in T):
        return None, None, "syndrome names a vertex absent from the graph"

    sp = {t: _dijkstra(adj, t) for t in T}

    import networkx as nx
    G = nx.Graph()
    for i, u in enumerate(T):
        du, _ = sp[u]
        for v in T[i + 1:]:
            if v in du:
                G.add_edge(("t", u), ("t", v), weight=-du[v])
        if BOUNDARY in du:
            G.add_edge(("t", u), ("b", u), weight=-du[BOUNDARY])
    for i, u in enumerate(T):
        for v in T[i + 1:]:
            G.add_edge(("b", u), ("b", v), weight=Fraction(0))

    M = nx.max_weight_matching(G, maxcardinality=True)
    if len(M) * 2 != G.number_of_nodes():
        return None, None, "no perfect matching on the derived graph " \
                           "(disconnected syndrome component?)"

    # XOR the expanded paths: an edge used an even number of times cancels,
    # which is what makes the union of shortest paths a valid T-join.
    count = {}
    for a, b in M:
        if a[0] == "b" and b[0] == "b":
            continue
        if a[0] == "b":
            a, b = b, a
        u = a[1]
        _, pu = sp[u]
        target = b[1] if b[0] == "t" else BOUNDARY
        for k in _walk(pu, u, target):
            count[k] = count.get(k, 0) + 1
    corr = tuple(sorted(k for k, c in count.items() if c % 2 == 1))
    weight = sum((W[k] for k in corr), Fraction(0))
    return corr, weight, "ok"


def reproduces_syndrome(correction, syndrome):
    """Does the edge set's odd-degree vertex set equal the syndrome?

    The boundary absorbs parity and is excluded. This is the caller-side
    verification exact_min_tjoin's output must pass before anything
    downstream sees it -- stated here so no caller has to reinvent it,
    and tested independently of the producer above.
    """
    deg = {}
    for (u, v) in correction:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    odd = {v for v, d in deg.items() if d % 2 == 1 and v != BOUNDARY}
    return odd == {int(s) for s in syndrome}
