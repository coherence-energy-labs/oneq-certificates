r"""The exact tier: T-join duals generated NATIVELY, no closure, no translation.

Every earlier rung produces candidate moats by translating duals from a
DIFFERENT space -- ball radii, growth schedules, the metric-closure matching
LP -- and the campaign's measured lesson is that translation is where value
dies: the closure converged EXACTLY on 10 of 12 stubborn hardware shots while
the translated bound still gapped, and the uncrossing approximation recovered
most but not all of it (7 of 500 spot shots left).

This module removes the translation instead of sharpening it. The T-join LP

    min w.x   s.t.   x(delta(S)) >= 1  for every T-odd S,   x >= 0

has, by Edmonds-Johnson, an optimum EQUAL to the minimum T-join weight -- and
its dual is precisely the cut packing the independent checker verifies:
y_S >= 0 per generated set, edge load sum y_S <= w_e, value = sum y_S. Solve
the LP lazily: separate violated T-odd cuts with Padberg-Rao (the minimum
T-odd cut lives among Gomory-Hu tree cuts), add, re-solve. At convergence the
row duals ARE an exact optimal packing over the generated sets, in the
original graph, with nothing lost in any translation -- the bound equals the
true optimum whenever the decoder's answer is optimal.

THE COST, stated so nobody wires it into the hot path: each separation round
builds a Gomory-Hu tree (|V|-1 max-flows). This is the FINAL tier, paid only
by shots that refused the base ladder AND the escalation tier -- a handful
per 50,000 on the hardest hardware configuration.

Soundness is unchanged by construction: the emitted sets and duals go through
the same independent checker as every other rung; this module can be wrong
only about RATE, never about a certificate.
"""

from __future__ import annotations

from .matching_cert import BOUNDARY


def _tjoin_edge_index(edges):
    """Canonical edge keys, cheapest weight each, and the index maps.

    Returns (keys, kidx, wvec, allv). Parallel edges collapse to the
    cheapest, exactly as the checker does, so the packing is priced
    against the same graph the certificate will be checked against.
    """
    import numpy as np

    ekeys: dict[tuple, float] = {}
    for u, v, w in edges:
        u, v = int(u), int(v)
        k = (v, u) if u == BOUNDARY else ((u, v) if (v == BOUNDARY or u <= v) else (v, u))
        w = float(w)
        if k not in ekeys or w < ekeys[k]:
            ekeys[k] = w
    keys = sorted(ekeys, key=str)
    kidx = {k: i for i, k in enumerate(keys)}
    wvec = np.array([ekeys[k] for k in keys])
    allv = frozenset(v for k in keys for v in k)
    return keys, kidx, wvec, allv


def _solve_packing(pool, crossing, wvec, ne):
    """LP-pack the current cut pool. Returns (x, bound, duals) or None.

    Each pooled set S contributes the constraint sum_{delta(S)} x >= 1,
    written as a <= row with negated coefficients.
    """
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix

    rows_r, rows_c = [], []
    for ci, S in enumerate(pool):
        for k in crossing(S):
            rows_r.append(ci)
            rows_c.append(k)
    A_ub = csr_matrix((-np.ones(len(rows_r)), (rows_r, rows_c)),
                      shape=(len(pool), ne))
    res = linprog(wvec, A_ub=A_ub, b_ub=-np.ones(len(pool)),
                  bounds=(0.0, None), method="highs")
    if not res.success:
        return None
    return (np.asarray(res.x), float(res.fun),
            np.abs(res.ineqlin.marginals))


def _min_odd_cut(nx, G, Tp):
    """SEPARATION (Padberg-Rao): the minimum Tp-odd cut under capacities x
    is realized by a Gomory-Hu tree edge; scan tree edges whose removal
    leaves an odd share of Tp. Returns (value, side) or (None, None).

    Capacities get a floor upstream so the tree is well-defined on
    zero-usage edges.
    """
    gh = nx.gomory_hu_tree(G, capacity="capacity")
    best_val, best_side = None, None
    for (a, b, data) in gh.edges(data=True):
        H = gh.copy()
        H.remove_edge(a, b)
        side = nx.node_connected_component(H, a)
        if len(set(side) & Tp) % 2 == 1:
            val = data["weight"]
            if best_val is None or val < best_val:
                best_val, best_side = val, frozenset(side)
    return best_val, best_side


# WHY THE EXIT REASON IS RECORDED, and why it is a module slot rather than a
# third return value: the producer equivalence gate unpacks a 2-TUPLE against
# a FROZEN snapshot, so widening the signature would break the very gate that
# guards this function. The caller reads this immediately after the call, in
# its own process.
#
# THE DISTINCTION THIS EXISTS TO MAKE. By Edmonds-Johnson the CONVERGED bound
# equals the minimum T-join weight, so a converged exact tier certifies
# wherever the decoder's answer is optimal. But the loop has two other exits:
# separation returning nothing new (a degenerate face where the separator goes
# blind) and the round cap. A bound from either is SHORT, and short means
# refuse -- so a stalled tier is currently INDISTINGUISHABLE in the evidence
# from a genuinely suboptimal correction. Those are opposite findings sharing
# one bucket: "we proved no better answer exists" versus "we ran out of
# separation". 8-32% of d5 shots refuse at this tier and nothing recorded
# which. That is the same defect shape as ONEQ_ESCALATE defaulting to 0 -- a
# silent weakening no gate can see.
LAST_EXIT: dict = {"reason": "not_run", "rounds": 0}


def exact_tjoin_sets(edges, syndrome, *, adj=None, max_rounds: int = 200,
                     seed_sets=None, tol: float = 1e-9):
    """Generate T-odd cut sets whose LP packing attains the exact optimum.

    Returns (sets, bound): `sets` is a list of frozensets of vertices (never
    containing BOUNDARY -- complemented when needed), `bound` the converged
    LP value (== min T-join weight at convergence). `seed_sets` warm-starts
    the cut pool (e.g. the escalated ladder's family) so most rounds are
    already separated before the first Gomory-Hu tree is built.

    The three moving parts -- edge indexing, LP packing, Padberg-Rao
    separation -- are separate functions, so the round loop below reads as
    the algorithm it is: pack, separate, admit, repeat until nothing is
    violated.
    """
    import networkx as nx
    import numpy as np

    T = {int(s) for s in syndrome}
    if not T:
        return [], 0.0

    keys, kidx, wvec, allv = _tjoin_edge_index(edges)
    ne = len(keys)

    # parity: a T-join may absorb parity at the boundary, so the boundary
    # node joins T exactly when |T| is odd
    Tp = set(T) | ({BOUNDARY} if len(T) % 2 else set())

    def crossing(S: frozenset):
        return [kidx[k] for k in keys if (k[0] in S) != (k[1] in S)]

    def canon(S) -> frozenset | None:
        S = frozenset(S)
        if BOUNDARY in S:
            S = allv - S
        if not S or BOUNDARY in S:
            return None
        return S

    # a boundary-free S is usable iff it holds an odd share of Tp; BOUNDARY
    # always counts as outside S (canon() complements it away), and because
    # |Tp| is even by construction, complementing preserves the parity
    def admissible(S: frozenset) -> bool:
        return len(S & Tp) % 2 == 1

    pool: list[frozenset] = []
    seen: set[frozenset] = set()
    for S in (seed_sets or ()):
        S = canon(S)
        if S is not None and S not in seen and admissible(S):
            seen.add(S)
            pool.append(S)

    G = nx.Graph()
    for (u, v) in keys:
        G.add_edge(u, v)

    bound = 0.0
    x = np.zeros(ne)
    duals = np.zeros(0)
    for _round in range(max_rounds):
        if pool:
            got = _solve_packing(pool, crossing, wvec, ne)
            if got is None:
                return [], 0.0
            x, bound, duals = got
        else:
            x = np.zeros(ne)
            bound = 0.0
            duals = np.zeros(0)

        for (u, v) in keys:
            G[u][v]["capacity"] = float(x[kidx[(u, v)]]) + 1e-12
        best_val, best_side = _min_odd_cut(nx, G, Tp)
        if best_val is None or best_val >= 1.0 - 1e-7:
            LAST_EXIT.update(reason="converged", rounds=_round + 1)
            break                       # no violated T-odd cut: CONVERGED
        S = canon(best_side)
        if S is None or S in seen or not admissible(S):
            LAST_EXIT.update(reason="separation_stalled", rounds=_round + 1)
            break                       # separation returned nothing new
        seen.add(S)
        pool.append(S)
    else:
        LAST_EXIT.update(reason="max_rounds_exhausted", rounds=max_rounds)

    # the loop can exit right after appending a set the LP never saw; the
    # duals must correspond to the FINAL pool or the zip below silently
    # drops the last set's contribution
    if pool and len(duals) != len(pool):
        got = _solve_packing(pool, crossing, wvec, ne)
        if got is not None:
            _x, bound, duals = got
    out = [S for S, y in zip(pool, duals) if y > tol] if len(pool) else []
    return (out or pool), bound
